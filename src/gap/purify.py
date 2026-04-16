import torch


def purify_ddpm(x_adv, diffusion, t_max_ratio=0.15, device="cpu"):
    """
    Purifies adversarial examples using a DDPM forward-reverse process (DiffPure style).
    
    Args:
        x_adv: Adversarial image tensor (B, C, H, W)
        diffusion: The trained GaussianDiffusion model
        t_max_ratio: Fraction of total timesteps to diffuse. 
                     (e.g., 0.15 means add noise up to 15% of max timesteps).
        device: torch device
        
    Returns:
        A numpy array containing the trajectory of images. The final cleaned 
        image is at index [-1].
    """
    diffusion = diffusion.to(device)
    diffusion.eval()
    
    # Calculate the integer timestep to diffuse up to
    num_timesteps = diffusion.num_timesteps
    t_max = int(t_max_ratio * num_timesteps)
    
    if t_max <= 0:
        return x_adv.clone().cpu().unsqueeze(0).numpy()

    x_adv = x_adv.to(device)
    batch_size = x_adv.shape[0]
    
    # ==========================================
    # 1. FORWARD PROCESS (Add noise up to t_max)
    # ==========================================
    t_tensor = torch.full((batch_size,), t_max, device=device, dtype=torch.long)
    noise = torch.randn_like(x_adv)
    
    # Fetch cumulative alphas from the diffusion model
    sqrt_ac = diffusion.sqrt_alphas_cumprod[t_tensor][:, None, None, None]
    sqrt_om = diffusion.sqrt_one_minus_alphas_cumprod[t_tensor][:, None, None, None]
    
    # Create the diffused image
    x_t = sqrt_ac * x_adv + sqrt_om * noise
    
    # ==========================================
    # 2. REVERSE PROCESS (Denoise t_max -> 0)
    # ==========================================
    # Reconstruct alphas and betas on the correct device for the loop
    betas = torch.linspace(1e-4, 0.02, num_timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    x = x_t
    trajectory = [x.clone().cpu()]
    
    with torch.no_grad():
        for i in reversed(range(t_max)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # Predict the noise using the UNet
            pred_noise = diffusion.model(x, t)
            
            alpha_t = alphas[i]
            alpha_t_cumprod = alphas_cumprod[i]
            beta_t = betas[i]
            
            # DDPM mean calculation
            sqrt_inv_alpha_t = 1.0 / torch.sqrt(alpha_t)
            c = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_t_cumprod)
            mean = sqrt_inv_alpha_t * (x - c * pred_noise)
            
            # Add variance (Langevin noise) if not at the final step
            if i > 0:
                z = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = mean + sigma_t * z
            else:
                x = mean
                
            trajectory.append(x.clone().cpu())
            
    return torch.stack(trajectory).squeeze().numpy()


def purify(x_adv, model, sigmas, device, steps_per_sigma=20, lr=0.01, simple=True):
    x_adv = x_adv.to(device)
    sigmas, _ = torch.sort(sigmas.to(device).view(-1), descending=True)
    model.eval()
    return (
        purify_simple(x_adv, model, sigmas[-1:], steps_per_sigma, lr)
        if simple
        else purify_annealed(x_adv, model, sigmas, steps_per_sigma, lr)
    )


def purify_simple(x_adv, model, sigma, num_steps, lr=0.01):
    """
    Fixed-Sigma Langevin Dynamics (Simple Purification).

    This method attempts to project the point onto the manifold using the score
    at a single, fixed noise level (usually the smallest sigma).

    Math: x_{t+1} = x_t + lr * delta log p_sigma(x_t) + sqrt(2 * lr) * epsilon
    where sigma is constant.
    """
    sigma = sigma.reshape(-1)

    x = x_adv.clone().detach()
    trajectory = [x.clone().cpu()]

    with torch.no_grad():
        step_size = lr * (sigma[0] ** 2)
        sigma_batch = sigma.repeat(x.shape[0])

        for _ in range(num_steps):
            score = model(x, sigma_batch)

            # Stochastic Gradient Ascent (Langevin Dynamics)
            noise = torch.randn_like(x)
            x = x + step_size * score + torch.sqrt(2 * step_size) * noise * 0.1
            trajectory.append(x.clone().cpu())

    return torch.stack(trajectory).squeeze().numpy()


def purify_annealed(x_adv, model, sigmas, steps_per_sigma=20, lr=0.01):
    """
    Annealed Langevin Dynamics: The standard way to move points from high-noise regions back to the data manifold.
    """
    x = x_adv.clone().detach()
    trajectory = [x.clone().cpu()]

    sigma_max = sigmas[0]

    with torch.no_grad():
        for sigma in sigmas:  # Start high, go low
            # Step size adjusted by sigma as per Yang Song's Score-SDE papers
            step_size = lr * (sigma / sigma_max) ** 2
            sigma_batch = sigma.view(1).repeat(x.shape[0], 1)

            for _ in range(steps_per_sigma):
                score = model(x, sigma_batch)

                # Langevin update
                noise = torch.randn_like(x)
                x = x + step_size * score + torch.sqrt(2 * step_size) * noise * 0.01
                trajectory.append(x.clone().cpu())

    return torch.stack(trajectory).squeeze().numpy()

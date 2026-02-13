import torch


def purify(x_adv, model, sigmas, device, steps_per_sigma=20, lr=0.01, simple=True):
        x_adv = x_adv.to(device)
        sigmas, _ = torch.sort(sigmas.to(device).view(-1), descending=True)
        model.eval()
        return purify_simple(x_adv, model, sigmas[-1:], steps_per_sigma, lr) if simple else purify_annealed(x_adv, model, sigmas, steps_per_sigma, lr)

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
        step_size = lr * (sigma[0]**2)
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
        for sigma in sigmas: # Start high, go low
            # Step size adjusted by sigma as per Yang Song's Score-SDE papers
            step_size = lr * (sigma / sigma_max)**2
            sigma_batch = sigma.view(1).repeat(x.shape[0], 1)

            for _ in range(steps_per_sigma):
                score = model(x, sigma_batch)

                # Langevin update
                noise = torch.randn_like(x)
                x = x + step_size * score + torch.sqrt(2 * step_size) * noise * 0.01 
                trajectory.append(x.clone().cpu())

    return torch.stack(trajectory).squeeze().numpy()

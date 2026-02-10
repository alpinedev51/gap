import torch


def purify(x_adv, model, sigmas, steps_per_sigma=20, lr=0.01, simple=True):
        return purify_simple(x_adv, model, sigmas[0], steps_per_sigma, lr) if simple else purify_annealed(x_adv, model, sigmas, steps_per_sigma, lr)

def purify_simple(x_adv, model, sigma, num_steps, lr=0.01):
    """
    Fixed-Sigma Langevin Dynamics (Simple Purification).

    This method attempts to project the point onto the manifold using the score 
    at a single, fixed noise level (usually the smallest sigma).

    Math: x_{t+1} = x_t + lr * delta log p_sigma(x_t) + sqrt(2 * lr) * epsilon
    where sigma is constant.
    """
    x = x_adv.clone().detach().requires_grad_(False)
    trajectory = [x.clone()]

    # Use a fixed step size
    step_size = lr * (sigma / sigma) * 0.01

    for _ in range(num_steps):
        score = model(x, sigma.view(1))

        # Stochastic Gradient Ascent (Langevin Dynamics)
        noise = torch.randn_like(x)
        x = x + step_size * score + torch.sqrt(2 * step_size) * noise * 0.01
        trajectory.append(x.clone())

    return torch.stack(trajectory).squeeze().detach().numpy()

def purify_annealed(x_adv, model, sigmas, steps_per_sigma=20, lr=0.01):
    """
    Annealed Langevin Dynamics: The standard way to move points from high-noise regions back to the data manifold.
    """
    x = x_adv.clone().detach().requires_grad_(False)
    trajectory = [x.clone()]

    for sigma in reversed(sigmas): # Start high, go low
        # Step size adjusted by sigma as per Yang Song's Score-SDE papers
        step_size = lr * (sigma / sigmas[0])**2

        for _ in range(steps_per_sigma):
            score = model(x, sigma.view(1))

            # Langevin update
            noise = torch.randn_like(x)
            x = x + step_size * score + torch.sqrt(2 * step_size) * noise * 0.01 
            trajectory.append(x.clone())

    return torch.stack(trajectory).squeeze().detach().numpy()

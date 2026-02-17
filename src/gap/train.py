import numpy as np
import torch

from gap.ema import EMA
from gap.utils import logger


def train_score_model(
    model, dataloader, optimizer, device, epochs=100, ema_decay=0.999, log_interval=10
):
    """
    Main training loop.
    """
    model = model.to(device)
    model.train()
    ema = EMA(model, decay=ema_decay)

    logger.info(f"Training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch_idx, (data, _) in enumerate(dataloader):
            x_clean = data.to(device)

            # Sample log-sigma uniformly to cover all scales (0.01 to 1.0)
            log_sigma = torch.rand(x_clean.shape[0], 1, device=device) * np.log(
                1.0 / 0.01
            ) + np.log(0.01)
            sigma = torch.exp(log_sigma)

            # Add Noise
            noise = torch.randn_like(x_clean)
            x_noisy = x_clean + sigma * noise

            # Predict Score
            score_pred = model(x_noisy, sigma)

            # Calculate Loss (Denoising Score Matching)
            # Target is -noise / sigma (The direction pointing to clean data)
            target_term = score_pred * sigma + noise
            loss = torch.mean(torch.sum(target_term**2, dim=-1))

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema.update(model)

            epoch_loss += loss.item()

        if (epoch + 1) % log_interval == 0:
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}/{epochs} | Average Loss: {avg_loss:.4f}")

    logger.info("Training complete.")
    return model, ema

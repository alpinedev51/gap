from typing import Tuple
import numpy as np

import matplotlib.pyplot as plt
import torch


class SpectralAnalyzer:
    """
    Computes 2D and 1D radially averaged Power Spectral Density (PSD)
    for image tensors. Includes Hanning windowing to prevent edge artifacts
    and handles batch processing natively.
    """

    def __init__(self, spatial_shape: Tuple[int, int] = (32, 32), device: str = "cpu"):
        self.H, self.W = spatial_shape
        self.device = torch.device(device)

        # Precompute 2D Hanning window and scaling factor
        window_h = torch.hann_window(self.H, periodic=False, device=self.device)
        window_w = torch.hann_window(self.W, periodic=False, device=self.device)
        self.window_2d = window_h.unsqueeze(1) * window_w.unsqueeze(0)
        self.scaling_factor = torch.mean(self.window_2d**2)

        # Precompute radial bins for 1D azimuthal averaging
        y, x = torch.meshgrid(
            torch.arange(self.H, device=self.device) - self.H // 2,
            torch.arange(self.W, device=self.device) - self.W // 2,
            indexing="ij",
        )
        r = torch.sqrt(x**2 + y**2)
        self.r_int = torch.round(r).to(torch.int64).flatten()

        # Precompute pixel counts per bin to avoid recomputing and division by zero
        self.max_radius = self.r_int.max().item() + 1
        self.bin_counts = torch.bincount(self.r_int, minlength=self.max_radius).clamp(
            min=1
        )

    def _format_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """Ensures tensor is shape (B, C, H, W) and on the correct device."""
        images = images.to(self.device)
        if images.dim() == 2:  # (H, W) -> (1, 1, H, W)
            images = images.unsqueeze(0).unsqueeze(0)
        elif images.dim() == 3:  # (C, H, W) -> (1, C, H, W)
            images = images.unsqueeze(0)
        return images

    def compute_2d_psd(self, images: torch.Tensor) -> torch.Tensor:
        """
        Computes the corrected 2D PSD of a batch of images.
        Returns tensor of shape (B, H, W).
        """
        images = self._format_tensor(images)

        # Zero-mean the image to prevent a massive DC spike
        image_mean = images.mean(dim=(-2, -1), keepdim=True)
        zero_mean_img = images - image_mean

        # Apply Hanning Window (broadcasts over B and C)
        windowed_img = zero_mean_img * self.window_2d

        # Compute 2D FFT and shift DC to center
        fft_complex = torch.fft.fft2(windowed_img, dim=(-2, -1))
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))

        # Calculate Power and correct for window's energy reduction
        power = torch.abs(fft_shifted) ** 2
        corrected_power = power / self.scaling_factor

        # Average across color channels to get a single spatial spectrum per image
        return corrected_power.mean(dim=1)

    def compute_1d_psd(self, psd_2d: torch.Tensor) -> torch.Tensor:
        """
        Computes the 1D radially averaged PSD from a 2D PSD batch.
        Returns tensor of shape (B, R).
        """
        B = psd_2d.shape[0]
        radial_psd = torch.zeros((B, self.max_radius), device=self.device)

        for b in range(B):
            psd_flat = psd_2d[b].flatten()
            bin_sums = torch.bincount(
                self.r_int, weights=psd_flat, minlength=self.max_radius
            )
            radial_psd[b] = bin_sums / self.bin_counts

        return radial_psd

    def get_log_psd(
        self, psd_tensor: torch.Tensor, epsilon: float = 1e-8
    ) -> torch.Tensor:
        """Converts the PSD to a logarithmic scale (dB) for visualization."""
        return 10 * torch.log10(psd_tensor + epsilon)

    def analyze(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method to compute both 2D and 1D PSDs simultaneously."""
        psd_2d = self.compute_2d_psd(images)
        psd_1d = self.compute_1d_psd(psd_2d)
        return psd_2d, psd_1d

    # --- VISUALIZATION METHODS ---

    def plot_2d(self, images: torch.Tensor, title: str = "Mean 2D Log PSD"):
        """Computes and plots the mean 2D PSD of the provided images."""
        psd_2d = self.compute_2d_psd(images)
        mean_psd_2d = psd_2d.mean(dim=0)  # Average over batch for visualization
        log_psd = self.get_log_psd(mean_psd_2d)

        plt.figure(figsize=(6, 6))
        plt.imshow(log_psd.cpu().numpy(), cmap="inferno")
        plt.colorbar(label="Power (dB)")
        plt.title(title)
        plt.xlabel("Horizontal Frequency (u)")
        plt.ylabel("Vertical Frequency (v)")
        plt.show()

    def plot_2d_difference(
        self,
        images_base: torch.Tensor,
        images_adv: torch.Tensor,
        title: str = "2D PSD Difference (Adv - Base)",
    ):
        """Plots the difference in PSD between adversarial and clean images."""
        psd_base = self.compute_2d_psd(images_base).mean(dim=0)
        psd_adv = self.compute_2d_psd(images_adv).mean(dim=0)

        log_base = self.get_log_psd(psd_base)
        log_adv = self.get_log_psd(psd_adv)

        diff = (log_adv - log_base).cpu().numpy()

        # Center the colormap at 0 using the absolute max for symmetric scaling
        max_abs = max(abs(diff.min()), abs(diff.max()))

        plt.figure(figsize=(6, 6))
        # 'coolwarm' maps negative (less power) to blue, 0 to white, positive (more power) to red
        plt.imshow(diff, cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
        plt.colorbar(label="Power Difference (dB)")
        plt.title(title)
        plt.xlabel("Horizontal Frequency (u)")
        plt.ylabel("Vertical Frequency (v)")
        plt.show()

    def plot_1d(
        self,
        images_dict: dict[str, torch.Tensor],
        title: str = "1D Azimuthally Averaged PSD",
        zoom_high_freq=False,
    ):
        """
        Computes and plots 1D PSDs. Accepts a dictionary of labels and image tensors
        so you can easily compare multiple datasets (e.g., Clean vs. PGD vs. FGSM).
        """
        plt.figure(figsize=(12, 5))
        styles = [
            {"color": "blue", "ls": "-", "lw": 2},
            {"color": "red", "ls": "--", "lw": 2},
            {"color": "orange", "ls": "-.", "lw": 2},
            {"color": "green", "ls": ":", "lw": 2},
        ]

        nyquist = min(self.H, self.W) // 2

        for i, (label, images) in enumerate(images_dict.items()):
            _, psd_1d = self.analyze(images)
            mean_psd_1d = psd_1d.mean(dim=0).cpu().numpy()  # Average over batch

            style = styles[i % len(styles)]

            frequencies = torch.arange(nyquist + 1)
            y_data = mean_psd_1d[: nyquist + 1]

            plt.plot(frequencies, y_data, label=label, **style)

        plt.yscale("log")
        plt.title(title + (" (High Freq Zoom)" if zoom_high_freq else ""), fontsize=14)
        plt.xlabel("Radial Spatial Frequency (r)", fontsize=12)
        plt.ylabel("Power (Log Scale)", fontsize=12)
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(fontsize=12)

        if zoom_high_freq:
            plt.xlim(nyquist // 2, nyquist)
            plt.ylim(bottom=0, top=100)
        else:
            plt.xlim(0, nyquist)

        plt.tight_layout()
        plt.show()

    def plot_1d_ratio(
        self,
        images_base: torch.Tensor,
        images_adv: torch.Tensor,
        label: str = "PGD / Clean",
        zoom_high_freq: bool = False,
    ):
        """
        Plots the ratio of adversarial power to clean power.
        """
        plt.figure(figsize=(10, 6))

        _, psd_1d_base = self.analyze(images_base)
        _, psd_1d_adv = self.analyze(images_adv)

        mean_base = psd_1d_base.mean(dim=0).cpu().numpy()
        mean_adv = psd_1d_adv.mean(dim=0).cpu().numpy()

        nyquist = min(self.H, self.W) // 2
        mean_base = mean_base[: nyquist + 1]
        mean_adv = mean_adv[: nyquist + 1]

        ratio = mean_adv / (mean_base + 1e-12)
        frequencies = np.arange(len(ratio))

        plt.plot(frequencies, ratio, color="crimson", lw=2.5, label=label)

        plt.axhline(
            y=1.0, color="black", linestyle="--", alpha=0.6, label="Baseline (Clean)"
        )
        plt.fill_between(
            frequencies, 1.0, ratio, where=(ratio > 1.0), color="red", alpha=0.1
        )

        plt.title(
            f"Spectral Power Ratio: {label}"
            + (" (High-Freq Zoom)" if zoom_high_freq else ""),
            fontsize=14,
        )
        plt.xlabel("Spatial Frequency (Radial Distance)", fontsize=12)
        plt.ylabel("Ratio (Adv Power / Clean Power)", fontsize=12)
        plt.grid(True, which="both", alpha=0.2)
        plt.legend(fontsize=11)

        # Apply the requested zoom
        if zoom_high_freq:
            plt.xlim(nyquist // 2, nyquist)
        else:
            plt.xlim(0, nyquist)

        y_max = np.percentile(ratio, 99) * 1.2
        plt.ylim(0, max(2.0, y_max))

        plt.tight_layout()
        plt.show()

    def print_spectral_stats(self, images_dict: dict[str, torch.Tensor]):
        """Calculates and prints average power in low vs high frequency domains."""
        print(
            f"{'Dataset':<10} | {'Low-Freq Power (dB)':<20} | {'High-Freq Power (dB)':<20}"
        )
        print("-" * 55)

        nyquist = min(self.H, self.W) // 2
        mid_freq = nyquist // 2

        for label, images in images_dict.items():
            _, psd_1d = self.analyze(images)
            mean_psd_1d = psd_1d.mean(dim=0)
            log_profile = self.get_log_psd(mean_psd_1d).cpu().numpy()

            # Low frequencies: r < mid_freq
            low_freq_mean = log_profile[:mid_freq].mean()
            # High frequencies: r >= mid_freq up to nyquist
            high_freq_mean = log_profile[mid_freq : nyquist + 1].mean()

            print(f"{label:<10} | {low_freq_mean:<20.4f} | {high_freq_mean:<20.4f}")

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
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

    def plot_2d(self, images: torch.Tensor, title: str = "Mean 2D Log PSD", dataset_name="cifar10"):
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
        plt.savefig(f"../figs/{dataset_name}/psd_2d/{title}.png")
        plt.show()


    def plot_2d_difference(
        self,
        images_1: torch.Tensor,
        images_2: torch.Tensor,
        images_3: Optional[torch.Tensor],
        title_21: str = "Mean 2D PSD Difference (Image2 - Image1)",
        title_31: str = "Mean 2D PSD Difference (Image3 - Image1)",
        title_32: str = "Mean 2D PSD Difference (Image3 - Image2)",
        three_images: bool = True,
        dataset_name="cifar10",
        save=True
    ):
        """Plots the difference in PSD between two images."""
        psd_1 = self.compute_2d_psd(images_1).mean(dim=0)
        psd_2 = self.compute_2d_psd(images_2).mean(dim=0)

        log_1 = self.get_log_psd(psd_1)
        log_2 = self.get_log_psd(psd_2)

        diff_21 = (log_2 - log_1).cpu().numpy()

        # Center the colormap at 0 using the absolute max for symmetric scaling
        max_abs_21 = max(abs(diff_21.min()), abs(diff_21.max()))

        plt.figure(figsize=(6, 6))
        # 'coolwarm' maps negative (less power) to blue, 0 to white, positive (more power) to red
        plt.imshow(diff_21, cmap="coolwarm", vmin=-max_abs_21, vmax=max_abs_21)
        plt.colorbar(label="Power Difference (dB)")
        plt.title(title_21)
        plt.xlabel("Horizontal Frequency (u)")
        plt.ylabel("Vertical Frequency (v)")
        plt.savefig(f"../figs/{dataset_name}/psd_2d/{title_21}.png")
        plt.show()

        if three_images:
            psd_3 = self.compute_2d_psd(images_3).mean(dim=0)
            log_3 = self.get_log_psd(psd_3)
            diff_31 = (log_3 - log_1).cpu().numpy()
            diff_32 = (log_3 - log_2).cpu().numpy()
            max_abs_31 = max(abs(diff_31.min()), abs(diff_31.max()))
            max_abs_32 = max(abs(diff_32.min()), abs(diff_32.max()))

            plt.figure(figsize=(6, 6))
            # 'coolwarm' maps negative (less power) to blue, 0 to white, positive (more power) to red
            plt.imshow(diff_31, cmap="coolwarm", vmin=-max_abs_31, vmax=max_abs_31)
            plt.colorbar(label="Power Difference (dB)")
            plt.title(title_31)
            plt.xlabel("Horizontal Frequency (u)")
            plt.ylabel("Vertical Frequency (v)")
            plt.savefig(f"../figs/{dataset_name}/psd_1d/{title_31}.png")
            plt.show()

            plt.figure(figsize=(6, 6))
            # 'coolwarm' maps negative (less power) to blue, 0 to white, positive (more power) to red
            plt.imshow(diff_32, cmap="coolwarm", vmin=-max_abs_32, vmax=max_abs_32)
            plt.colorbar(label="Power Difference (dB)")
            plt.title(title_32)
            plt.xlabel("Horizontal Frequency (u)")
            plt.ylabel("Vertical Frequency (v)")
            plt.savefig(f"../figs/{dataset_name}/psd_1d/{title_32}.png")
            plt.show()


    def plot_1d(
        self,
        images_dict: dict[str, torch.Tensor],
        title: str = "Mean 1D Azimuthally Averaged PSD",
        zoom_high_freq=False,
        dataset_name="cifar10",
        save=True
    ):
        """
        Computes and plots 1D PSDs. Accepts a dictionary of labels and image tensors
        so you can easily compare multiple datasets (e.g., Original vs. PGD vs. FGSM).
        """
        plt.figure(figsize=(12, 5))
        styles = [
            {"color": "blue", "ls": "-", "lw": 2},
            {"color": "red", "ls": "--", "lw": 2},
            {"color": "orange", "ls": "-.", "lw": 2},
            {"color": "green", "ls": ":", "lw": 2},
            {"color": "purple", "ls": "dashdot", "lw": 2},
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
        title += "(High Freq Zoom)" if zoom_high_freq else ""
        plt.title(title, fontsize=14)
        plt.xlabel("Radial Spatial Frequency (r)", fontsize=12)
        plt.ylabel("Power (Log Scale)", fontsize=12)
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend(fontsize=12)

        if zoom_high_freq:
            plt.xlim(nyquist // 2, nyquist)
            top = 100
            if self.H == 128:
                top = 200
            plt.ylim(bottom=0, top=top)
        else:
            plt.xlim(0, nyquist)

        plt.tight_layout()
        plt.savefig(f"../figs/{dataset_name}/psd_1d/{title}.png")
        plt.show()

    def plot_1d_ratio(
        self,
        images_1: torch.Tensor,
        images_2: torch.Tensor,
        images_3: torch.Tensor,
        label_21: str = "PGD / Original",
        label_31: str = "Purified / Original",
        zoom_high_freq: bool = False,
        three_images: bool = True,
        dataset_name="cifar10",
        save=True
    ):
        """
        Plots the ratio of adversarial power to original power.
        """
        plt.figure(figsize=(10, 6))

        _, psd_1d_1 = self.analyze(images_1)
        _, psd_1d_2 = self.analyze(images_2)

        mean_1 = psd_1d_1.mean(dim=0).cpu().numpy()
        mean_2 = psd_1d_2.mean(dim=0).cpu().numpy()

        nyquist = min(self.H, self.W) // 2
        mean_1 = mean_1[: nyquist + 1]
        mean_2 = mean_2[: nyquist + 1]

        ratio_21 = mean_2 / (mean_1 + 1e-12)
        frequencies_21 = np.arange(len(ratio_21))

        plt.plot(frequencies_21, ratio_21, color="purple", lw=2.5, label=label_21)

        plt.axhline(
            y=1.0, color="black", linestyle="--", alpha=0.6, label="Baseline (Original)"
        )
        plt.fill_between(
            frequencies_21, 1.0, ratio_21, where=(ratio_21 > 1.0), color="blue", alpha=0.1
        )

        title = f"Mean Spectral Power Ratio: {label_21}"
        title += " (High-Freq Zoom)" if zoom_high_freq else ""
        plt.title(title, fontsize=14)
        plt.xlabel("Spatial Frequency (Radial Distance)", fontsize=12)
        plt.ylabel("Mean Ratio (Averaged over Dataset)", fontsize=12)
        plt.grid(True, which="both", alpha=0.2)
        plt.legend(fontsize=11)

        # Apply the requested zoom
        if zoom_high_freq:
            plt.xlim(nyquist // 2, nyquist)
        else:
            plt.xlim(0, nyquist)

        y_max_21 = np.percentile(ratio_21, 99) * 1.2
        plt.ylim(0, max(2.0, y_max_21))

        plt.tight_layout()
        plt.savefig(f"../figs/{dataset_name}/psd_1d/{title.replace(" / ", "_over_")}.png")
        plt.show()

        if three_images:
            plt.figure(figsize=(10, 6))

            _, psd_1d_1 = self.analyze(images_1)
            _, psd_1d_3 = self.analyze(images_3)

            mean_1 = psd_1d_1.mean(dim=0).cpu().numpy()
            mean_3 = psd_1d_3.mean(dim=0).cpu().numpy()

            nyquist = min(self.H, self.W) // 2
            mean_1 = mean_1[: nyquist + 1]
            mean_3 = mean_3[: nyquist + 1]

            ratio_31 = mean_3 / (mean_1 + 1e-12)
            frequencies_31 = np.arange(len(ratio_31))

            plt.plot(frequencies_31, ratio_31, color="blue", lw=2.5, label=label_31)

            plt.axhline(
                y=1.0, color="black", linestyle="--", alpha=0.6, label="Baseline (Original)"
            )
            plt.fill_between(
                frequencies_31, 1.0, ratio_31, where=(ratio_31 > 1.0), color="red", alpha=0.1
            )

            title = f"Mean Spectral Power Ratio: {label_31}"
            title += " (High-Freq Zoom)" if zoom_high_freq else ""
            plt.title(title,fontsize=14)
            plt.xlabel("Spatial Frequency (Radial Distance)", fontsize=12)
            plt.ylabel("Mean Ratio (Adv Power / Original Power)", fontsize=12)
            plt.grid(True, which="both", alpha=0.2)
            plt.legend(fontsize=11)

            # Apply the requested zoom
            if zoom_high_freq:
                plt.xlim(nyquist // 2, nyquist)
            else:
                plt.xlim(0, nyquist)

            y_max_31 = np.percentile(ratio_31, 99) * 1.2
            plt.ylim(0, max(2.0, y_max_31))

            plt.tight_layout()
            plt.savefig(f"../figs/{dataset_name}/psd_1d/{title.replace(" / ", "_over_")}.png")
            plt.show()


    def print_spectral_stats(self, images_dict: dict[str, torch.Tensor]):
        """Calculates and prints average power in low vs high frequency domains."""
        print(
            f"{'Dataset':<10} | {'Mean Low-Freq Power (dB)':<20} | {'Mean High-Freq Power (dB)':<20}"
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

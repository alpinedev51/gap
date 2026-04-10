from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.fft

from gap.utils import get_default_device


class FFT:
    """
    Wrapper class for Fast Fourier Transform operations.
    Handles 2D Hanning window generation, scaling, and the DFT.
    """

    def __init__(
        self,
        spatial_shape: tuple[int, int],
        device: torch.device = get_default_device(),
    ):
        """
        Args:
            spatial_shape: (Height, Width) of the images to be processed.
            device: The PyTorch device to store the window and perform calculations.
        """
        self.H, self.W = spatial_shape
        self.device = device

        # 1. Generate and store the 2D Hanning window
        self.window_2d = self._generate_hanning_window()

        # 2. Compute and store the scaling factor to correct for power loss
        # S = mean(w^2)
        self.scaling_factor = torch.mean(self.window_2d**2)

    def _generate_hanning_window(self) -> torch.Tensor:
        """Generates a 2D Hanning window using outer product of two 1D windows."""
        # Create 1D Hanning windows for height and width
        window_h = torch.hann_window(self.H, periodic=False, device=self.device)
        window_w = torch.hann_window(self.W, periodic=False, device=self.device)

        # 2D window is the outer product of the 1D windows
        window_2d = window_h.unsqueeze(1) * window_w.unsqueeze(0)
        return window_2d

    def apply_window(self, image: torch.Tensor) -> torch.Tensor:
        """
        Multiplies the image by the 2D Hanning window.
        Assumes image shape is (..., H, W).
        """
        # The window will broadcast automatically over batch/channel dimensions
        return image * self.window_2d

    def compute_fft(self, windowed_image: torch.Tensor) -> torch.Tensor:
        """
        Computes the 2D FFT and shifts the zero-frequency (DC) component to the center.
        """
        # Compute 2D FFT over the last two dimensions (Height, Width)
        fft_complex = torch.fft.fft2(windowed_image, dim=(-2, -1))

        # Shift DC component to center
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))
        return fft_shifted


class PSD:
    """
    Wrapper class for an image to compute its Power Spectral Density.
    """

    def __init__(self, image: torch.Tensor):
        """
        Args:
            image: A PyTorch tensor of shape (C, H, W) or (H, W).
                   Pixel values should ideally be normalized (e.g., [0, 1] or [-1, 1]).
        """
        self.image = image

        # Standardize shape to (C, H, W)
        if self.image.dim() == 2:
            self.image = self.image.unsqueeze(0)

        _, self.H, self.W = self.image.shape
        self.device = self.image.device

    def compute(self, fft_operator: Optional[FFT] = None) -> torch.Tensor:
        """
        Computes the corrected Power Spectral Density of the image.

        Returns:
            A 2D tensor of shape (H, W) representing the PSD.
        """
        # Instantiate an FFT operator if one isn't provided
        if fft_operator is None:
            fft_operator = FFT(spatial_shape=(self.H, self.W), device=self.device)

        # 1. Zero-mean the image to prevent a massive DC spike from washing out the plot
        image_mean = self.image.mean(dim=(-2, -1), keepdim=True)
        zero_mean_img = self.image - image_mean

        # 2. Apply Hanning Window
        windowed_img = fft_operator.apply_window(zero_mean_img)

        # 3. Compute FFT
        fft_shifted = fft_operator.compute_fft(windowed_img)

        # 4. Calculate Power (Squared Magnitude)
        # power = real^2 + imag^2
        power = torch.abs(fft_shifted) ** 2

        # 5. Apply Scaling Factor to correct for the window's energy reduction
        corrected_power = power / fft_operator.scaling_factor

        # 6. Average across color channels to get a single 2D spatial frequency map
        final_psd_2d = corrected_power.mean(dim=0)

        return final_psd_2d

    def get_log_psd(
        self, psd_tensor: torch.Tensor, epsilon: float = 1e-8
    ) -> torch.Tensor:
        """
        Converts the PSD to a logarithmic scale for visualization.
        Adds epsilon to prevent log(0).
        """
        return 10 * torch.log10(psd_tensor + epsilon)

    def plot(self, title: str = "Log Power Spectral Density"):
        """Utility method to quickly visualize the PSD."""
        psd = self.compute()
        log_psd = self.get_log_psd(psd)

        # Move to CPU and numpy for plotting
        log_psd_np = log_psd.cpu().numpy()

        plt.figure(figsize=(6, 6))
        plt.imshow(log_psd_np, cmap="inferno")
        plt.colorbar(label="Power (dB)")
        plt.title(title)
        plt.xlabel("Horizontal Frequency (u)")
        plt.ylabel("Vertical Frequency (v)")
        plt.show()

    def azimuthal_average(self, psd_2d: torch.Tensor) -> torch.Tensor:
        """
        Computes the 1D radially (azimuthally) averaged PSD.

        Args:
            psd_2d: The 2D power spectral density tensor of shape (H, W).

        Returns:
            A 1D tensor representing the average power at each radial frequency.
        """
        h, w = psd_2d.shape

        # 1. Create a grid of spatial coordinates centered at (0, 0)
        y, x = torch.meshgrid(
            torch.arange(h, device=self.device) - h // 2,
            torch.arange(w, device=self.device) - w // 2,
            indexing="ij",
        )

        # 2. Calculate the radius (distance from center) for every pixel
        # r = sqrt(u^2 + v^2)
        r = torch.sqrt(x**2 + y**2)

        # 3. Round radii to nearest integer to create discrete frequency bins
        r_int = torch.round(r).to(torch.int64).flatten()
        psd_flat = psd_2d.flatten()

        # 4. Sum the power in each radial bin using PyTorch's bincount
        power_sum = torch.bincount(r_int, weights=psd_flat)

        # 5. Count the number of pixels in each radial bin
        pixel_count = torch.bincount(r_int).clamp(min=1)  # clamp prevents div by zero

        # 6. Compute the average power per bin
        radial_profile = power_sum / pixel_count

        return radial_profile

    def plot_1d(self, title: str = "1D Azimuthally Averaged PSD"):
        """Utility method to quickly visualize the 1D radial profile."""
        psd_2d = self.compute()
        profile_1d = self.azimuthal_average(psd_2d)

        # Apply log scale for visualization
        log_profile_1d = self.get_log_psd(profile_1d)

        # Move to CPU/numpy for plotting
        frequencies = torch.arange(len(log_profile_1d)).cpu().numpy()
        power = log_profile_1d.cpu().numpy()

        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(frequencies, power, color="b", label="Original Image")

        plt.title(title)
        plt.xlabel("Radial Spatial Frequency (r)")
        plt.ylabel("Log Power (dB)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # For small images like CIFAR10 (32x32), the max radius is ~22.
        # We limit the x-axis to half the image width (Nyquist limit for straight lines)
        # to avoid the sparse corners of the rectangular grid.
        plt.xlim(0, min(self.H, self.W) // 2)

        plt.show()

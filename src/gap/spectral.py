import torch


class SpectralAnalyzer:
    """
    Computes 2D and 1D radially averaged Power Spectral Density (PSD)
    for image tensors to analyze high-frequency energy.
    """

    @staticmethod
    def compute_2d_psd(images: torch.Tensor) -> torch.Tensor:
        """
        Computes the 2D PSD of a batch of images.

        Args:
            images (torch.Tensor): Tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: 2D PSD of shape (B, H, W) averaged across channels.
        """
        # Compute 2D Fast Fourier Transform
        # Using fft.fft2 computes it over the last two dimensions (H, W)
        fft_coeffs = torch.fft.fft2(images)

        # Shift the zero-frequency component to the center of the spectrum
        fft_shifted = torch.fft.fftshift(fft_coeffs, dim=(-2, -1))

        # Calculate Power Spectral Density: magnitude squared
        psd_2d = torch.abs(fft_shifted) ** 2

        # Average across the channel dimension to get a single spatial spectrum per image
        psd_2d_mean = psd_2d.mean(dim=1)

        return psd_2d_mean

    @staticmethod
    def compute_1d_radial_psd(psd_2d: torch.Tensor) -> torch.Tensor:
        """
        Computes the 1D radially averaged PSD from a 2D PSD.

        Args:
            psd_2d (torch.Tensor): 2D PSD tensor of shape (B, H, W).

        Returns:
            torch.Tensor: 1D radially averaged PSD of shape (B, R) where R is max radius.
        """
        B, H, W = psd_2d.shape
        center_y, center_x = H // 2, W // 2

        # Create a grid of coordinates
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")

        # Calculate the distance of each pixel from the center (0 frequency)
        radii = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        radii = radii.round().long().to(psd_2d.device)

        max_radius = radii.max().item() + 1
        radial_psd = torch.zeros((B, max_radius), device=psd_2d.device)

        # Flatten tensors for efficient binning
        radii_flat = radii.view(-1)

        for b in range(B):
            psd_flat = psd_2d[b].view(-1)
            # Sum PSD values within each radial bin
            bin_sums = torch.bincount(
                radii_flat, weights=psd_flat, minlength=max_radius
            )
            # Count number of pixels in each radial bin
            bin_counts = torch.bincount(radii_flat, minlength=max_radius)

            # Avoid division by zero
            bin_counts[bin_counts == 0] = 1
            radial_psd[b] = bin_sums / bin_counts

        return radial_psd

    @classmethod
    def analyze_batch(cls, images: torch.Tensor):
        """Convenience method to return both 2D and 1D PSDs."""
        psd_2d = cls.compute_2d_psd(images)
        psd_1d = cls.compute_1d_radial_psd(psd_2d)
        return psd_2d, psd_1d

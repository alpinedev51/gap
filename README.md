<h1 align="center">SpectRA-GAP: Spectral Recovery Analysis of Generative Models in Adversarial Purification</h1>


Code repository for research project titled:

> SpectRA-GAP: Spectral Recovery Analysis of Generative Models in Adversarial Purification
>
> [Austin Barton](github.com/alpinedev51), Joshua Culwell
>
> Abstract (WIP): This project is an empirical comparative analysis of generative algorithms for adversarial purification. Specifically, we compare single-shot reconstruction via Denoising Autoencoders (DAEs) and iterative score-based projection with diffusion models (DMs) for purification of adversarially perturbed image data. We evaluate these methods against Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks using downstream classification evaluation metrics on purified samples and conduct spectral analysis via Power Spectral Density (PSD) to quantify and visualize how each method manages high-frequency adversarial energy (which adversarially crafted inputs using methods such as PGD or FGSM consist of). Understanding this spectral recovery is critical because traditional compression algorithms act as destructive low-pass filters that indiscriminately remove both adversarial noise and legitimate high-frequency predictive details. This generates data points inconsistent with true samples and can result in degradation of downstream performance. By focusing on the frequency domain, we investigate our overarching hypothesis: Traditional non-generative methods and DAEs achieve robustness by truncating or smoothing high-frequency energy, but at the cost of predictive power. Whereas DMs achieve superior performance by actively regenerating those semantically consistent high-frequency features.


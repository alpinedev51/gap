<h1 align="center">SpectRA-GAP: Spectral Recovery Analysis of Generative Models in Adversarial Purification</h1>


Code repository for research project titled:

> SpectRA-GAP: Spectral Recovery Analysis of Generative Models in Adversarial Purification
>
> [Austin Barton](github.com/alpinedev51), Joshua Culwell
>
> Abstract: We present an empirical comparative analysis of generative algorithms for adversarial purification. Specifically, we compare Denoising Autoencoders (DAEs) and diffusion models (DMs) for purification of adversarially perturbed image data. We evaluate these methods against Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks using downstream classification evaluation metrics on purified samples and conduct spectral analysis via Power Spectral Density (PSD) to quantify and visualize how each method manages high-frequency adversarial energy. Understanding this spectral recovery is critical because traditional compression algorithms act as destructive low-pass filters that indiscriminately remove both adversarial noise and legitimate high-frequency predictive details. This generates data points inconsistent with true samples and can result in degradation of downstream performance. We demonstrate that while DAEs achieve robustness through destructive low-pass filtering at the cost of predictive power, DMs maintain performance by regenerating semantically consistent high-frequency features.

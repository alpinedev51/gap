import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class AdversarialAttacker:
    """
    Generates adversarial examples using FGSM or PGD.
    Assumes inputs are normalized to [-1, 1].
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def get_correct_subset_loader(self, dataloader: DataLoader) -> DataLoader:
        """
        Filters the input DataLoader and returns a new DataLoader containing
        only the samples the model correctly classifies.
        """
        correct_images = []
        correct_labels = []

        print("Filtering for correctly classified samples...")
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                mask = preds.eq(labels)

                if mask.any():
                    correct_images.append(images[mask].cpu())
                    correct_labels.append(labels[mask].cpu())

        if not correct_images:
            raise ValueError(
                "Model classified 0 images correctly in the provided loader."
            )

        # Create new TensorDataset from correctly classified samples
        filtered_dataset = TensorDataset(
            torch.cat(correct_images), torch.cat(correct_labels)
        )

        return DataLoader(
            filtered_dataset, batch_size=dataloader.batch_size, shuffle=False
        )

    def fgsm_attack(
        self, images: torch.Tensor, labels: torch.Tensor, epsilon: float = 0.03
    ) -> torch.Tensor:
        """
        Generates adversarial examples using the Fast Gradient Sign Method (FGSM).
        """
        images = images.clone().detach().to(self.device).requires_grad_(True)
        labels = labels.to(self.device)

        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)

        self.model.zero_grad()
        loss.backward()

        # Collect the element-wise sign of the data gradient
        sign_data_grad = images.grad.data.sign()

        # Create the perturbed image and clamp to [-1, 1] range
        perturbed_images = images + epsilon * sign_data_grad
        perturbed_images = torch.clamp(perturbed_images, -1.0, 1.0)

        return perturbed_images

    def pgd_attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        epsilon: float = 0.03,
        alpha: float = 0.008,
        iters: int = 10,
    ) -> torch.Tensor:
        """
        Generates adversarial examples using Projected Gradient Descent (PGD).
        """
        images = images.clone().detach().to(self.device)
        labels = labels.to(self.device)

        # Start with a random perturbation within the epsilon ball
        perturbed_images = images.clone().detach()
        perturbed_images = perturbed_images + torch.empty_like(
            perturbed_images
        ).uniform_(-epsilon, epsilon)
        perturbed_images = torch.clamp(perturbed_images, -1.0, 1.0).requires_grad_(True)

        for _ in range(iters):
            outputs = self.model(perturbed_images)
            loss = F.cross_entropy(outputs, labels)

            self.model.zero_grad()
            loss.backward()

            # Update perturbed images using the gradient
            adv_images = perturbed_images + alpha * perturbed_images.grad.sign()

            # Project the perturbation back to the epsilon ball
            eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)

            # Apply perturbation and clamp to valid image range [-1, 1]
            perturbed_images = torch.clamp(images + eta, min=-1.0, max=1.0).detach_()
            perturbed_images.requires_grad = True

        return perturbed_images

    def generate_adversarial_dataset(
        self, dataloader: DataLoader, attack_type: str = "pgd", **kwargs
    ) -> DataLoader:
        """
        Takes a clean DataLoader and returns a new DataLoader containing adversarial examples.
        """
        adv_images_list = []
        labels_list = []

        print(f"Generating {attack_type.upper()} adversarial examples...")
        for images, labels in dataloader:
            if attack_type.lower() == "fgsm":
                adv_imgs = self.fgsm_attack(images, labels, **kwargs)
            elif attack_type.lower() == "pgd":
                adv_imgs = self.pgd_attack(images, labels, **kwargs)
            else:
                raise ValueError("attack_type must be 'fgsm' or 'pgd'")

            adv_images_list.append(adv_imgs.cpu().detach())
            labels_list.append(labels.cpu().detach())

        # Concatenate everything into a single dataset
        all_adv_images = torch.cat(adv_images_list)
        all_labels = torch.cat(labels_list)

        adv_dataset = TensorDataset(all_adv_images, all_labels)

        # Return a new dataloader matching the original batch size
        return DataLoader(adv_dataset, batch_size=dataloader.batch_size, shuffle=False)

    def save_adversarial_loader(self, loader, path):
        # Save the underlying TensorDataset
        torch.save(loader.dataset, path)

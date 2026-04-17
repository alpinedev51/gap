import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class AdversarialAttacker:
    """
    Generates adversarial examples using FGSM or PGD.
    Handles dynamic channel-wise normalization bounds (e.g., ImageNet).
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu", mean=None, std=None):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Default to standard [-1, 1] equivalent if no stats are provided
        if mean is None:
            mean = [0.5, 0.5, 0.5] 
        if std is None:
            std = [0.5, 0.5, 0.5]

        # Convert to tensors shaped for broadcasting (1, C, 1, 1) over image batches
        self.mean = torch.tensor(mean, device=device).view(1, len(mean), 1, 1)
        self.std = torch.tensor(std, device=device).view(1, len(std), 1, 1)

        # Calculate absolute min and max bounds for valid image pixels in normalized space
        self.min_val = (0.0 - self.mean) / self.std
        self.max_val = (1.0 - self.mean) / self.std

    def get_correct_subset_loader(self, dataloader: DataLoader) -> DataLoader:
        correct_images = []
        correct_labels = []
        total_samples = 0
        total_correct = 0

        self.model.eval()
        print("Filtering for correctly classified samples...")
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                mask = preds.eq(labels)
                total_samples += labels.size(0)
                total_correct += mask.sum().item()

                if mask.any():
                    correct_images.append(images[mask].cpu())
                    correct_labels.append(labels[mask].cpu())

        if total_samples == 0:
            print("The DataLoader is empty.")

        accuracy = total_correct / total_samples
        print(f"Filtering complete. Accuracy {accuracy * 100:.2f}.")
        if not correct_images:
            raise ValueError("Model classified 0 images correctly in the provided loader.")

        filtered_dataset = TensorDataset(torch.cat(correct_images), torch.cat(correct_labels))
        return DataLoader(filtered_dataset, batch_size=dataloader.batch_size, shuffle=False)

    def fgsm_attack(self, images: torch.Tensor, labels: torch.Tensor, epsilon: float = 0.03) -> torch.Tensor:
        images = images.clone().detach().to(self.device).requires_grad_(True)
        labels = labels.to(self.device)

        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)

        self.model.zero_grad()
        loss.backward()

        # Scale epsilon by std to match normalized feature space magnitudes
        eps_norm = epsilon / self.std
        sign_data_grad = images.grad.data.sign()

        # Apply perturbation
        perturbed_images = images + eps_norm * sign_data_grad

        # Clamp using torch.max and torch.min for channel-wise tensor bounds
        perturbed_images = torch.max(torch.min(perturbed_images, self.max_val), self.min_val)

        return perturbed_images

    def pgd_attack(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        epsilon: float = 0.03,
        alpha: float = 0.008,
        iters: int = 10,
    ) -> torch.Tensor:
        images = images.clone().detach().to(self.device)
        labels = labels.to(self.device)

        eps_norm = epsilon / self.std
        alpha_norm = alpha / self.std

        # Start with a random uniform perturbation within the epsilon ball
        perturbed_images = images.clone().detach()
        noise = torch.empty_like(perturbed_images).uniform_(-1.0, 1.0) * eps_norm
        perturbed_images = perturbed_images + noise

        # Clamp initial perturbation to valid image bounds
        perturbed_images = torch.max(torch.min(perturbed_images, self.max_val), self.min_val).requires_grad_(True)

        for _ in range(iters):
            outputs = self.model(perturbed_images)
            loss = F.cross_entropy(outputs, labels)

            self.model.zero_grad()
            loss.backward()

            # Update perturbed images
            adv_images = perturbed_images + alpha_norm * perturbed_images.grad.sign()

            # Project the perturbation back to the normalized epsilon ball
            eta = torch.max(torch.min(adv_images - images, eps_norm), -eps_norm)

            # Apply clipped perturbation and clamp to valid image range
            perturbed_images = torch.max(torch.min(images + eta, self.max_val), self.min_val).detach_()
            perturbed_images.requires_grad = True

        return perturbed_images

    def generate_adversarial_dataset(self, dataloader: DataLoader, attack_type: str = "pgd", **kwargs) -> DataLoader:
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

        adv_dataset = TensorDataset(torch.cat(adv_images_list), torch.cat(labels_list))
        return DataLoader(adv_dataset, batch_size=dataloader.batch_size, shuffle=False)


class AdversarialAttackerOld:
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

        total_samples = 0
        total_correct = 0


        self.model.eval()

        print("Filtering for correctly classified samples...")
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                mask = preds.eq(labels)

                total_samples += labels.size(0)
                total_correct += mask.sum().item()

                if mask.any():
                    correct_images.append(images[mask].cpu())
                    correct_labels.append(labels[mask].cpu())

        if total_samples == 0:
            print("The DataLoader is empty.")

        accuracy = total_correct / total_samples
        print(f"Filtering complete. Accuracy {accuracy * 100:.2f}.")
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
        Takes an original DataLoader and returns a new DataLoader containing adversarial examples.
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

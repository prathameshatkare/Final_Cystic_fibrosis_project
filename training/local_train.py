import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FocalLoss(nn.Module):
    """Focal loss for class imbalance mitigation in CF diagnosis.

    gamma controls how much to down-weight easy examples.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        pt = probs[range(len(targets)), targets]
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        return loss.mean()


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_focal: bool = False,
) -> float:
    """Generic local training loop used by both centralized and FL settings."""

    model.train()
    criterion = FocalLoss() if use_focal else nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    if total_samples == 0:
        return 0.0
    return total_loss / total_samples

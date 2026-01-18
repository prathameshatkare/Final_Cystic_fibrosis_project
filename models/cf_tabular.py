import torch
import torch.nn as nn


class CFTabularNet(nn.Module):
    """Simple MLP for tabular CF diagnosis from CSV features.

    This is used when only structured clinical features are available, while
    the CFVision ViT remains available for image-based experiments.
    """

    def __init__(self, input_dim: int, hidden_dims: tuple[int, int] = (128, 64), num_classes: int = 2) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

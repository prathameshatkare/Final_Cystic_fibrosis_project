from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import torch
from torch.utils.data import DataLoader

from models.cf_tabular import CFTabularNet
from training.local_train import train_one_epoch
from training.eval import evaluate_cfvision


class CFVisionFLClient(fl.client.NumPyClient):
    """Flower client for CF diagnosis federated learning on edge devices.

    Uses a tabular MLP model (CFTabularNet) when only CSV features are
    available, while keeping the rest of the FL pipeline unchanged.
    """

    def __init__(
        self,
        cid: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        input_dim: int,
        use_focal: bool = True,
    ) -> None:
        self.cid = cid
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.use_focal = use_focal

        self.model = CFTabularNet(input_dim=input_dim).to(self.device)

    # Helper functions -----------------------------------------------------

    def get_parameters(self, config: Dict) -> List:
        state_dict = self.model.state_dict()
        return [v.cpu().numpy() for _, v in state_dict.items()]

    def set_parameters(self, parameters: List) -> None:
        state_dict = OrderedDict()
        for (k, _), v in zip(self.model.state_dict().items(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    # Flower NumPyClient interface ----------------------------------------

    def fit(self, parameters: List, config: Dict) -> Tuple[List, int, Dict]:
        self.set_parameters(parameters)

        lr = float(config.get("lr", 1e-4))
        weight_decay = float(config.get("weight_decay", 1e-5))
        local_epochs = int(config.get("local_epochs", 1))

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        for _ in range(local_epochs):
            train_one_epoch(
                model=self.model,
                data_loader=self.train_loader,
                optimizer=optimizer,
                device=self.device,
                use_focal=self.use_focal,
            )

        updated_params = [v.cpu().numpy() for _, v in self.model.state_dict().items()]
        num_examples = len(self.train_loader.dataset)
        return updated_params, num_examples, {}

    def evaluate(self, parameters: List, config: Dict) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        metrics = evaluate_cfvision(self.model.to(self.device), self.test_loader, self.device)
        num_examples = len(self.test_loader.dataset)
        return float(metrics["loss"]), num_examples, metrics

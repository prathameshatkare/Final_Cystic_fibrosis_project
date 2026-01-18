from typing import Dict, List, Optional, Tuple

import flwr as fl


class FedProxStrategy(fl.server.strategy.FedAvg):
    """FedAvg-based strategy placeholder for FedProx-style training.

    In this minimal version, the proximal term is implemented on the client
    side, so server aggregation remains a weighted average.
    """

    def __init__(
        self,
        fraction_fit: float = 0.5,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs,
        )

    # The default aggregate_fit from FedAvg is sufficient for now.

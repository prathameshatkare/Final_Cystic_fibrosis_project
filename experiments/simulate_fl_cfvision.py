import argparse
from typing import List
import os
import sys

# Ensure project root is on sys.path when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import flwr as fl
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split

from federated.client import CFVisionFLClient
from federated.server_strategy import FedProxStrategy


def load_cf_tabular(path: str = "data/synthetic_cystic_fibrosis_dataset.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["cf_diagnosis"])

    y = df["cf_diagnosis"].astype(int).values
    feature_df = df.drop(columns=["cf_diagnosis", "age_at_diagnosis", "diagnostic_confidence"], errors="ignore")
    feature_df = pd.get_dummies(feature_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df.values.astype(float))
    return X_scaled, y


def create_client_loaders(X, y, num_clients: int, batch_size: int):
    """Create non-IID client loaders with imbalanced CF prevalence.

    Most clients receive very few CF-positive samples (simulating general
    centers), while the last two clients receive a higher proportion of
    CF-positive cases (simulating specialized CF centers).
    """

    n_samples = X.shape[0]
    indices_pos = np.where(y == 1)[0]
    indices_neg = np.where(y == 0)[0]
    np.random.shuffle(indices_pos)
    np.random.shuffle(indices_neg)

    client_loaders: List[tuple[DataLoader, DataLoader]] = []

    # Determine approximate number of samples per client
    base_n = n_samples // num_clients
    remainder = n_samples % num_clients

    pos_ptr = 0
    neg_ptr = 0

    for client_idx in range(num_clients):
        # Distribute leftover samples to the first `remainder` clients
        n_i = base_n + (1 if client_idx < remainder else 0)

        # Set target CF-positive fraction: low for most, higher for last two
        if client_idx < max(0, num_clients - 2):
            target_pos_frac = 0.05  # ~5% positives
        else:
            target_pos_frac = 0.30  # ~30% positives

        target_pos = int(n_i * target_pos_frac)

        # Allocate positives
        available_pos = len(indices_pos) - pos_ptr
        n_pos = min(target_pos, max(0, available_pos))

        # Allocate negatives for the rest
        remaining_needed = n_i - n_pos
        available_neg = len(indices_neg) - neg_ptr
        n_neg = min(remaining_needed, max(0, available_neg))

        # If we still don't have enough samples, backfill from the other class
        if n_pos + n_neg < n_i:
            shortfall = n_i - (n_pos + n_neg)
            # Prefer negatives for backfill if available, else positives
            extra_neg = min(shortfall, max(0, len(indices_neg) - (neg_ptr + n_neg)))
            n_neg += extra_neg
            shortfall -= extra_neg
            extra_pos = min(shortfall, max(0, len(indices_pos) - (pos_ptr + n_pos)))
            n_pos += extra_pos

        pos_idx = indices_pos[pos_ptr : pos_ptr + n_pos]
        neg_idx = indices_neg[neg_ptr : neg_ptr + n_neg]
        pos_ptr += n_pos
        neg_ptr += n_neg

        client_indices = np.concatenate([pos_idx, neg_idx])
        np.random.shuffle(client_indices)

        X_client = torch.tensor(X[client_indices], dtype=torch.float32)
        y_client = torch.tensor(y[client_indices], dtype=torch.long)

        dataset = TensorDataset(X_client, y_client)
        n_val = max(1, int(0.2 * len(dataset)))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        client_loaders.append((train_loader, val_loader))

    return client_loaders


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate CF tabular FL with Flower")
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_cf_tabular()
    input_dim = X.shape[1]

    client_loaders = create_client_loaders(X, y, args.num_clients, args.batch_size)

    def client_fn(cid: str):
        idx = int(cid)
        train_loader, val_loader = client_loaders[idx]
        return CFVisionFLClient(
            cid=cid,
            train_loader=train_loader,
            test_loader=val_loader,
            device=device,
            input_dim=input_dim,
        )

    strategy = FedProxStrategy()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()

import argparse
import os
import sys

# Ensure project root is on sys.path when running as a script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.cf_tabular import CFTabularNet
from training.eval import evaluate_cfvision
from training.local_train import train_one_epoch


def load_cf_tabular(path: str = "data/synthetic_cystic_fibrosis_dataset.csv"):
    df = pd.read_csv(path)
    # Drop rows without a diagnosis label
    df = df.dropna(subset=["cf_diagnosis"])

    y = df["cf_diagnosis"].astype(int).values

    # Drop target and optional non-predictive column(s)
    feature_df = df.drop(columns=["cf_diagnosis", "age_at_diagnosis", "diagnostic_confidence"], errors="ignore")

    # One-hot encode categorical features
    feature_df = pd.get_dummies(feature_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df.values.astype(float))

    return X_scaled, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Centralized CF tabular baseline training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_cf_tabular()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = CFTabularNet(input_dim=input_dim).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"Starting training on {device}...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, use_focal=True)
        metrics = evaluate_cfvision(model, test_loader, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"auc={metrics['auc']:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"sens={metrics['sensitivity']:.4f} | "
            f"spec={metrics['specificity']:.4f}"
        )

    # Save the trained model weights for the UI/Inference
    save_path = os.path.join(PROJECT_ROOT, "models", "cf_tabular_central.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    torch.save(model.state_dict(), "models/cf_tabular_central.pt")


if __name__ == "__main__":
    main()

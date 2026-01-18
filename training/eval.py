import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix


def evaluate_cfvision(model: nn.Module, data_loader, device: torch.device) -> dict:
    """Evaluate a CFVision-style model with clinical metrics.

    Returns accuracy, AUC, F1, sensitivity (recall for CF), specificity,
    balanced accuracy, and average loss.
    """

    model.eval()
    all_probs = []
    all_labels = []
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    ce_loss = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = ce_loss(logits, labels)
            total_loss += loss.item() * labels.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    if total_samples == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "auc": 0.0,
            "f1": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "balanced_accuracy": 0.0,
        }

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    accuracy = total_correct / total_samples

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    binary_preds = (all_probs >= 0.5).astype(int)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)

    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds).ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        precision = tp / (tp + fp + 1e-8)
    except ValueError:
        sensitivity = 0.0
        specificity = 0.0
        precision = 0.0

    balanced_acc = 0.5 * (sensitivity + specificity)

    return {
        "loss": total_loss / total_samples,
        "accuracy": accuracy,
        "auc": auc,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "balanced_accuracy": balanced_acc,
    }

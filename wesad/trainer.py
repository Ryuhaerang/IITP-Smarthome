"""Training helpers wrapping the PyTorch training loop for WESAD models."""

import copy
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wesad.data import DataPreparationResult


@dataclass
class DataLoaders:
    """Grouped dataloaders for train/val/test splits."""
    train: DataLoader
    val: DataLoader
    test: DataLoader


def _build_dataloader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Construct a simple TensorDataset-backed DataLoader from numpy arrays."""
    dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


def create_dataloaders(
    data: DataPreparationResult, batch_size: int, num_workers: int
) -> DataLoaders:
    """Create all dataloaders needed for training/evaluation."""
    train_loader = _build_dataloader(
        data.train_features, data.train_labels, batch_size, True, num_workers
    )
    val_loader = _build_dataloader(
        data.val_features, data.val_labels, batch_size, False, num_workers
    )
    test_loader = _build_dataloader(
        data.test_features, data.test_labels, batch_size, False, num_workers
    )
    return DataLoaders(train=train_loader, val=val_loader, test=test_loader)


class Trainer:
    """Thin wrapper for the training loop, tracking best validation accuracy."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        label_order: List[str],
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.label_order = label_order

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        best_state = None
        best_val_acc = -np.inf

        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            val_metrics = self.evaluate(val_loader, include_report=False)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            print(
                f"Epoch {epoch:03d} | "
                f"train loss {train_loss:.4f} | "
                f"val loss {val_metrics['loss']:.4f} | "
                f"val acc {val_metrics['accuracy']:.4f}"
            )

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_state = copy.deepcopy(self.model.state_dict())

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "history": history,
            "best_state": best_state,
            "best_val_accuracy": best_val_acc,
        }

    def _train_one_epoch(self, loader: DataLoader) -> float:
        """Run a single training epoch and return the average loss."""
        self.model.train()
        running_loss = 0.0
        batches = 0
        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            batches += 1
        return running_loss / max(1, batches)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        include_report: bool = True,
    ) -> Dict[str, float]:
        """Evaluate the model, optionally returning a full classification report."""
        self.model.eval()
        all_preds: List[int] = []
        all_targets: List[int] = []
        running_loss = 0.0
        batches = 0

        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            running_loss += loss.item()
            batches += 1

        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        metrics: Dict[str, float] = {
            "loss": running_loss / max(1, batches),
            "accuracy": float(accuracy),
            "f1": float(f1),
        }

        if include_report:
            report = classification_report(
                all_targets,
                all_preds,
                labels=list(range(len(self.label_order))),
                target_names=self.label_order,
                zero_division=0,
                output_dict=True,
            )
            metrics["classification_report"] = report

        return metrics

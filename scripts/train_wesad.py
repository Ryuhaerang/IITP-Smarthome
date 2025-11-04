#!/usr/bin/env python3
"""Train, evaluate, and optionally quantize the WESAD classifier."""

import os
from dataclasses import asdict
from typing import Optional

import torch
from torch import nn

from wesad.config import ExperimentConfig, parse_config
from wesad.data import DataPreparationResult, prepare_wesad_data
from wesad.model import build_model
from wesad.quantization import run_quantization
from wesad.trainer import Trainer, create_dataloaders
from wesad.utils import ensure_output_dir, persist_metrics, set_seed


def resolve_device(preference: Optional[str]) -> torch.device:
    """Select training device, preferring CUDA when available."""
    if preference is None or preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def create_optimizer(
    model: nn.Module, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    """Configure optimizer; easy swap point if we try different optimizers later."""
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def save_model_checkpoint(
    path: str,
    model: nn.Module,
    data: DataPreparationResult,
) -> None:
    """Persist model weights alongside preprocessing statistics needed for inference."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_id": data.label_to_id,
            "feature_columns": data.feature_columns,
            "scaler_mean": data.scaler.mean_.tolist(),
            "scaler_scale": data.scaler.scale_.tolist(),
            "imputer_statistics": data.imputer.statistics_.tolist(),
        },
        path,
    )


def main() -> None:
    config: ExperimentConfig = parse_config()
    set_seed(config.seed)

    prepared_data = prepare_wesad_data(config)
    device = resolve_device(config.device)
    output_dir = config.training.output_dir
    ensure_output_dir(output_dir)

    model = build_model(
        input_dim=prepared_data.train_features.shape[1],
        model_config=config.model,
        num_classes=len(config.label_order),
    ).to(device)

    optimizer = create_optimizer(
        model,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    dataloaders = create_dataloaders(
        prepared_data, batch_size=config.training.batch_size, num_workers=config.training.num_workers
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        label_order=config.label_order,
    )
    fit_result = trainer.fit(dataloaders.train, dataloaders.val, config.training.epochs)

    test_metrics = trainer.evaluate(dataloaders.test, include_report=True)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")

    quantization_results = []
    if config.quantization.enable_int8 or config.quantization.enable_int4:
        quantization_results = run_quantization(
            model=model,
            dataloaders=dataloaders, # just test on test dataset
            label_order=config.label_order,
            quant_config=config.quantization,
            output_dir=config.training.output_dir,
        )
        for result in quantization_results:
            print(f"{result.method} accuracy: {result.accuracy:.4f}")

    results = {
        "config": asdict(config),
        "splits": prepared_data.subject_splits,
        "dropped_columns": {
            "all_nan": prepared_data.dropped_all_nan,
        },
        "num_features": prepared_data.train_features.shape[1],
        "history": fit_result["history"],
        "validation": {
            "best_accuracy": fit_result["best_val_accuracy"],
        },
        "test": test_metrics,
    }
    if quantization_results:
        results["quantization"] = [res.to_dict() for res in quantization_results]

    metrics_path = os.path.join(output_dir, config.training.metrics_filename)
    persist_metrics(metrics_path, results)
    print(f"Saved metrics to {metrics_path}")

    if config.training.save_model:
        model_path = os.path.join(output_dir, "model.pt")
        save_model_checkpoint(model_path, model, prepared_data)
        print(f"Saved model checkpoint to {model_path}")

if __name__ == "__main__":
    main()

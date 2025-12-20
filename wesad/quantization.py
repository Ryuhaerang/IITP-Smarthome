"""Utilities for post-training quantization and evaluation of WESAD models."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, f1_score

from wesad.config import QuantizationConfig
from wesad.trainer import DataLoaders


@dataclass
class QuantizationResult:
    """Quantization summary including evaluation metrics and optional saved path."""

    method: str
    loss: float
    accuracy: float
    f1: float
    classification_report: Dict[str, Dict[str, float]]
    model_path: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """Serialize the result for logging or JSON persistence."""
        return asdict(self)


def _evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label_order: List[str],
) -> Dict[str, object]:
    """Run evaluation and return loss/accuracy/report without using Trainer."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    running_loss = 0.0
    batches = 0
    all_targets: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item()
            batches += 1

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_targets,
        all_preds,
        labels=list(range(len(label_order))),
        target_names=label_order,
        zero_division=0,
        output_dict=True,
    )
    return {
        "loss": running_loss / max(1, batches),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "classification_report": report,
    }


def _quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    """Apply dynamic INT8 quantization to Linear layers."""
    quant_ready = copy.deepcopy(model).to("cpu")
    quant_ready.eval()
    return torch.ao.quantization.quantize_dynamic(
        quant_ready,
        {nn.Linear},
        dtype=torch.qint8,
    )


def _quantize_uniform(model: nn.Module, num_bits: int) -> nn.Module:
    """Perform simple symmetric fake-quantization on weights to the specified bit-width."""
    quant_model = copy.deepcopy(model).to("cpu")
    quant_model.eval()
    max_q = 2 ** (num_bits - 1) - 1

    for module in quant_model.modules():
        if hasattr(module, "weight") and isinstance(getattr(module, "weight"), nn.Parameter):
            weight = module.weight.data
            max_abs = weight.abs().max()
            if max_abs == 0:
                continue
            scale = max_abs / max_q
            if scale == 0:
                continue
            quantized = torch.clamp((weight / scale).round(), -max_q, max_q)
            module.weight.data = quantized * scale
    return quant_model


def run_quantization(
    model: nn.Module,
    dataloaders: DataLoaders,
    label_order: List[str],
    quant_config: QuantizationConfig,
    output_dir: Optional[str],
) -> List[QuantizationResult]:
    """
    Quantize the trained model according to config and evaluate on the test loader.

    Returns a list of results for each enabled quantization method.
    """
    results: List[QuantizationResult] = []
    test_loader = dataloaders.test
    cpu_device = torch.device("cpu")

    if quant_config.enable_int8:
        int8_model = _quantize_dynamic_int8(model)
        metrics = _evaluate_model(int8_model, test_loader, cpu_device, label_order)
        model_path = None
        if quant_config.save_int8_model and output_dir:
            model_path = os.path.join(output_dir, quant_config.int8_model_filename)
            torch.save(int8_model.state_dict(), model_path)
        results.append(
            QuantizationResult(
                method="int8_dynamic",
                loss=metrics["loss"],
                accuracy=metrics["accuracy"],
                f1=metrics["f1"],
                classification_report=metrics["classification_report"],
                model_path=model_path,
            )
        )

    if quant_config.enable_int4:
        int4_model = _quantize_uniform(model, num_bits=4)
        metrics = _evaluate_model(int4_model, test_loader, cpu_device, label_order)
        model_path = None
        if quant_config.save_int4_model and output_dir:
            model_path = os.path.join(output_dir, quant_config.int4_model_filename)
            torch.save(int4_model.state_dict(), model_path)
        results.append(
            QuantizationResult(
                method="int4_uniform",
                loss=metrics["loss"],
                accuracy=metrics["accuracy"],
                f1=metrics["f1"],
                classification_report=metrics["classification_report"],
                model_path=model_path,
            )
        )

    return results

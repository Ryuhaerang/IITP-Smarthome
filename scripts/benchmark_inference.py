#!/usr/bin/env python3
"""Run one-sample CPU inference and latency checks for the WESAD model.

Default usage after training:
    python scripts/benchmark_inference.py

The benchmark measures model-only latency:
    processed feature vector -> model prediction

It does not include raw biosignal preprocessing latency.
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_CHECKPOINT = "runs/wesad/model.pt"
DEFAULT_DATASET = "data/processed/wesad/hf_dataset"
DEFAULT_LABELS = ["baseline", "stress", "amusement"]


class CheckpointMLP(nn.Module):
    """MLP reconstructed directly from saved Linear-layer tensors."""

    def __init__(self, linear_weights: List[torch.Tensor], linear_biases: List[torch.Tensor]):
        super().__init__()
        if len(linear_weights) != len(linear_biases):
            raise ValueError("Linear weight/bias count mismatch.")

        self.layers = nn.ModuleList()
        for weight, bias in zip(linear_weights, linear_biases):
            layer = nn.Linear(weight.shape[1], weight.shape[0])
            layer.weight.data.copy_(weight.float())
            layer.bias.data.copy_(bias.float())
            self.layers.append(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = F.relu(x)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark FP32 and INT8 CPU inference on one processed WESAD sample."
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="FP32 checkpoint created by scripts/train_wesad.py.",
    )
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET,
        help="Optional processed HF dataset path. Falls back to a zero vector if missing.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Processed dataset sample index to use when dataset-path exists.",
    )
    parser.add_argument("--runs", type=int, default=1000, help="Timed inference runs.")
    parser.add_argument("--warmup", type=int, default=100, help="Untimed warmup runs.")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPU thread count for stable latency measurement.",
    )
    return parser.parse_args()


def load_checkpoint(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Train and save a model first, for example:\n"
            "  python -m scripts.train_wesad"
        )

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint dict, got {type(checkpoint)!r}")
    return checkpoint


def get_model_state_dict(checkpoint: Dict[str, object]) -> Dict[str, torch.Tensor]:
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Could not find a model state dict in the checkpoint.")
    return state_dict


def linear_layer_tensors(
    state_dict: Dict[str, torch.Tensor]
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    weights: List[Tuple[int, torch.Tensor]] = []
    biases: Dict[int, torch.Tensor] = {}

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        parts = key.split(".")
        if len(parts) < 3 or parts[0] != "net":
            continue
        try:
            layer_idx = int(parts[1])
        except ValueError:
            continue

        if parts[-1] == "weight" and value.ndim == 2:
            weights.append((layer_idx, value.detach().cpu()))
        elif parts[-1] == "bias" and value.ndim == 1:
            biases[layer_idx] = value.detach().cpu()

    weights.sort(key=lambda item: item[0])
    linear_weights = [weight for _, weight in weights]
    linear_biases = [biases[idx] for idx, _ in weights if idx in biases]

    if not linear_weights or len(linear_weights) != len(linear_biases):
        raise ValueError("Failed to reconstruct Linear layers from checkpoint.")
    return linear_weights, linear_biases


def build_model_from_checkpoint(checkpoint: Dict[str, object]) -> nn.Module:
    state_dict = get_model_state_dict(checkpoint)
    weights, biases = linear_layer_tensors(state_dict)
    model = CheckpointMLP(weights, biases)
    model.eval()
    return model


def model_size_kb(model: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return len(buffer.getvalue()) / 1024.0


def checkpoint_labels(checkpoint: Dict[str, object]) -> Dict[int, str]:
    label_to_id = checkpoint.get("label_to_id")
    if isinstance(label_to_id, dict):
        return {int(idx): str(label) for label, idx in label_to_id.items()}
    return {idx: label for idx, label in enumerate(DEFAULT_LABELS)}


def apply_checkpoint_preprocessing(
    raw_features: np.ndarray, checkpoint: Dict[str, object]
) -> np.ndarray:
    imputer_stats = checkpoint.get("imputer_statistics")
    scaler_mean = checkpoint.get("scaler_mean")
    scaler_scale = checkpoint.get("scaler_scale")
    if imputer_stats is None or scaler_mean is None or scaler_scale is None:
        return raw_features.astype(np.float32)

    x = raw_features.astype(np.float32, copy=True)
    imputer_stats = np.asarray(imputer_stats, dtype=np.float32)
    scaler_mean = np.asarray(scaler_mean, dtype=np.float32)
    scaler_scale = np.asarray(scaler_scale, dtype=np.float32)
    scaler_scale = np.where(scaler_scale == 0, 1.0, scaler_scale)

    nan_mask = np.isnan(x)
    if nan_mask.any():
        x[nan_mask] = imputer_stats[nan_mask]
    return ((x - scaler_mean) / scaler_scale).astype(np.float32)


def try_load_dataset_sample(
    dataset_path: str, sample_index: int, checkpoint: Dict[str, object]
) -> Optional[np.ndarray]:
    feature_columns = checkpoint.get("feature_columns")
    if not feature_columns or not os.path.exists(dataset_path):
        return None

    try:
        from datasets import load_from_disk

        dataset = load_from_disk(dataset_path)
        if len(dataset) == 0:
            return None
        row = dataset[min(sample_index, len(dataset) - 1)]
        features = row["features"]
        raw_sample = np.array(
            [features.get(column, np.nan) for column in feature_columns],
            dtype=np.float32,
        )
        return apply_checkpoint_preprocessing(raw_sample, checkpoint)
    except Exception as exc:
        print(f"Could not load dataset sample; using zeros instead. Reason: {exc}")
        return None


def make_sample(
    checkpoint: Dict[str, object], dataset_path: str, sample_index: int, input_dim: int
) -> Tuple[np.ndarray, str]:
    sample = try_load_dataset_sample(dataset_path, sample_index, checkpoint)
    if sample is not None:
        return sample, f"{dataset_path}[{sample_index}]"
    return np.zeros(input_dim, dtype=np.float32), "zero vector"


def quantize_int8(model: nn.Module) -> nn.Module:
    quantized = torch.ao.quantization.quantize_dynamic(
        model.cpu().eval(),
        {nn.Linear},
        dtype=torch.qint8,
    )
    quantized.eval()
    return quantized


def benchmark(
    model: nn.Module, sample: np.ndarray, runs: int, warmup: int
) -> Dict[str, float]:
    x = torch.from_numpy(sample).unsqueeze(0).cpu()

    with torch.inference_mode():
        for _ in range(warmup):
            model(x)

        latencies_ms = []
        logits = None
        for _ in range(runs):
            start = time.perf_counter()
            logits = model(x)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "prediction_id": int(logits.argmax(dim=1).item()),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def print_result(
    name: str,
    model: nn.Module,
    result: Dict[str, float],
    id_to_label: Dict[int, str],
) -> None:
    prediction = id_to_label.get(int(result["prediction_id"]), str(result["prediction_id"]))
    print(f"\n{name}")
    print(f"  serialized_model_size_kb: {model_size_kb(model):.2f}")
    print(f"  prediction: {prediction}")
    print(f"  mean_ms: {result['mean_ms']:.4f}")
    print(f"  median_ms: {result['median_ms']:.4f}")
    print(f"  p95_ms: {result['p95_ms']:.4f}")
    print(f"  min_ms: {result['min_ms']:.4f}")
    print(f"  max_ms: {result['max_ms']:.4f}")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)

    checkpoint = load_checkpoint(args.checkpoint)
    fp32_model = build_model_from_checkpoint(checkpoint).cpu().eval()
    int8_model = quantize_int8(fp32_model)

    input_dim = fp32_model.layers[0].in_features
    sample, sample_source = make_sample(
        checkpoint=checkpoint,
        dataset_path=args.dataset_path,
        sample_index=args.sample_index,
        input_dim=input_dim,
    )
    id_to_label = checkpoint_labels(checkpoint)

    print("CPU single-sample inference benchmark")
    print(f"checkpoint: {args.checkpoint}")
    print(f"checkpoint_file_size_kb: {os.path.getsize(args.checkpoint) / 1024.0:.2f}")
    print(f"input_dim: {input_dim}")
    print(f"sample_source: {sample_source}")
    print(f"runs: {args.runs}")
    print(f"warmup: {args.warmup}")
    print(f"threads: {args.threads}")
    print("latency_scope: processed feature vector -> model prediction")

    fp32_result = benchmark(fp32_model, sample, args.runs, args.warmup)
    int8_result = benchmark(int8_model, sample, args.runs, args.warmup)

    print_result("FP32", fp32_model, fp32_result, id_to_label)
    print_result("INT8 dynamic", int8_model, int8_result, id_to_label)


if __name__ == "__main__":
    main()

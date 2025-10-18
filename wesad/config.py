"""Configuration parsing utilities supporting YAML + CLI overrides."""

import argparse
import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import yaml


def _default_hidden_sizes() -> List[int]:
    return [256, 128]


def _default_label_order() -> List[str]:
    return ["baseline", "stress", "amusement"]


@dataclass
class DataConfig:
    dataset_path: str = "data/processed/wesad/hf_dataset"
    train_frac: float = 0.7
    val_frac: float = 0.15
    seed: int = 42


@dataclass
class ModelConfig:
    hidden_sizes: List[int] = field(default_factory=_default_hidden_sizes)
    dropout: float = 0.2


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 64
    num_workers: int = 0
    output_dir: Optional[str] = None
    save_model: bool = False
    metrics_filename: str = "metrics.json"


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "auto"
    label_order: List[str] = field(default_factory=_default_label_order)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _deep_update_dict(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update nested dictionaries without mutating the original."""
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_update_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_override(override: str) -> Dict[str, Any]:
    """Parse a dot-notation `key=value` string into a nested dict structure."""
    if "=" not in override:
        raise ValueError(f"Override must be in key=value format: {override}")
    key, value = override.split("=", 1)
    parsed_value = yaml.safe_load(value)
    keys = key.split(".")
    root: Dict[str, Any] = {}
    current = root
    for part in keys[:-1]:
        current.setdefault(part, {})
        if not isinstance(current[part], dict):
            raise ValueError(f"Conflict while parsing override for key: {override}")
        current = current[part]
    current[keys[-1]] = parsed_value
    return root


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for configuration file + override flags."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a simple DNN on WESAD features."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/wesad/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        help="Optional key=value overrides (dot notation supported). Example: training.batch_size=128",
    )
    return parser


def load_config_from_yaml(path: str) -> Dict[str, Any]:
    """Load configuration from disk, returning an empty dict if the file is blank."""
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def apply_overrides(
    config_dict: Dict[str, Any], overrides: Optional[List[str]]
) -> Dict[str, Any]:
    """Apply CLI overrides on top of the base configuration dictionary."""
    if not overrides:
        return config_dict
    updated = copy.deepcopy(config_dict)
    for raw in overrides:
        override_dict = _parse_override(raw)
        updated = _deep_update_dict(updated, override_dict)
    return updated


def dict_to_dataclass(config_dict: Dict[str, Any]) -> ExperimentConfig:
    """Convert a plain dictionary into the strongly-typed ExperimentConfig."""
    default_config = ExperimentConfig()
    merged = _deep_update_dict(asdict(default_config), config_dict)
    return ExperimentConfig(
        seed=merged.get("seed", default_config.seed),
        device=merged.get("device", default_config.device),
        label_order=merged.get("label_order", default_config.label_order),
        data=DataConfig(**merged.get("data", {})),
        model=ModelConfig(**merged.get("model", {})),
        optimizer=OptimizerConfig(**merged.get("optimizer", {})),
        training=TrainingConfig(**merged.get("training", {})),
    )


def parse_config(argv: Optional[List[str]] = None) -> ExperimentConfig:
    """High-level helper combining YAML load + overrides into ExperimentConfig."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    base_config = load_config_from_yaml(args.config)
    final_dict = apply_overrides(base_config, args.override)
    return dict_to_dataclass(final_dict)

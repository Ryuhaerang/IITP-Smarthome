import json
import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def persist_metrics(path: str, metrics: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

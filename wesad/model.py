"""Model architectures for WESAD experiments."""

from typing import Sequence

from torch import nn

from wesad.config import ModelConfig


class SimpleFeedForward(nn.Module):
    """Baseline MLP that operates on flattened engineered feature vectors."""

    def __init__(
        self, input_dim: int, hidden_sizes: Sequence[int], num_classes: int, dropout: float
    ) -> None:
        super().__init__()
        layers = []
        in_features = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Run a batch of feature vectors through the MLP."""
        return self.net(x)


def build_model(input_dim: int, model_config: ModelConfig, num_classes: int) -> nn.Module:
    """Factory that instantiates the baseline feed-forward network."""
    return SimpleFeedForward(
        input_dim=input_dim,
        hidden_sizes=model_config.hidden_sizes,
        num_classes=num_classes,
        dropout=model_config.dropout,
    )

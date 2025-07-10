import torch.nn as nn
import torch


class FCNNBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        normalizaton: bool = True,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.activation = activation
        if normalizaton:
            self.model = nn.Sequential(
                nn.LayerNorm(in_dim), self.activation, nn.Linear(in_dim, out_dim)
            )
        else:
            self.model = nn.Sequential(self.activation, nn.Linear(in_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

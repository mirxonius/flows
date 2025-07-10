from abc import abstractmethod
import torch.nn as nn
import torch

from src.model.blocks import FCNNBlock
from src.utils import temporal_encoding


class SimpleVelocityModel(nn.Module):
    def __init__(self, out_dims: int, hidden_dims: list[int] = [64, 64, 64, 64]):
        super().__init__()
        self.embedding_dim = hidden_dims[0]
        self.hidden_dims = hidden_dims
        self.t_projection = FCNNBlock(
            in_dim=self.embedding_dim,
            out_dim=self.embedding_dim,
            activation=nn.Identity(),
            normalizaton=False,
        )

        self.x_projection = FCNNBlock(
            in_dim=out_dims,
            out_dim=self.embedding_dim,
            activation=nn.Identity(),
            normalizaton=False,
        )

        self.processor = nn.Sequential(
            *[
                FCNNBlock(in_dim=hidden_dims[i], out_dim=hidden_dims[i + 1])
                for i in range(len(hidden_dims) - 1)
            ]
        )
        self.out_projection = FCNNBlock(
            in_dim=hidden_dims[-1], out_dim=out_dims, activation=nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        embd_t = temporal_encoding(t, channels_t=self.embedding_dim)
        embd_t = self.t_projection(embd_t)
        embd = self.x_projection(x)
        embd = embd + embd_t
        embd = self.processor(embd)
        vel = self.out_projection(embd)
        return vel

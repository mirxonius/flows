import math
import torch.nn as nn
import torch
from flows.FlowMatching.model.layers import Mlp


class TemporalEmbedding(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, model_dim: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = Mlp(frequency_embedding_size, model_dim, model_dim)
        self.frequency_embedding_size: int = frequency_embedding_size
        self.model_dim = model_dim

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.

        :param t: a 1-D torch.Tensor of N indices, one per batch element. These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) torch.Tensor of positional embeddings.
        """
        half: int = dim // 2
        freqs: torch.Tensor = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args: torch.Tensor = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size)).view(
            -1, self.model_dim
        )

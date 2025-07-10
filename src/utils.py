import numpy as np
import torch
from ot import emd, unif


def temporal_encoding(
    t: torch.Tensor, channels_t: int, inv_freq: float = 10000.0
) -> torch.Tensor:

    freq = 2 * torch.arange(channels_t, device=t.device) * torch.pi
    embd = freq * t / inv_freq
    embd = torch.cat(
        [embd[:, slice(0, channels_t, 2)].sin(), embd[:, slice(1, channels_t, 2)]],
        dim=-1,
    )
    return embd


def optimal_transport_sampling(
    x0: torch.Tensor, x1: torch.Tensor
) -> tuple[torch.Tensor]:
    with torch.no_grad():
        batch = x0.shape[0]
        M = torch.cdist(x0.view(batch, -1), x1.view(batch, -1)).cpu().numpy()
        x0_bins = unif(batch)
        x1_bins = unif(batch)
        transport_matrix = emd(x0_bins, x1_bins, M)
        source, destinaton = np.where(transport_matrix > 0)
    return source, destinaton

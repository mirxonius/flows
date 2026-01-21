import torch
import numpy as np


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # todo: Add shape annotations
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: int, cls_token: bool = False, extra_tokens: int = 0
) -> np.ndarray:
    """
    :param embed_dim: Dimension of the positional embedding
    :param grid_size: Height and width of the grid
    :param cls_token: Whether to prepend extra tokens (e.g., [CLS])
    :param extra_tokens: Number of extra tokens to prepend
    :return: Positional embedding of shape [(extra_tokens + grid_size^2), embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    # shape: (2, grid_size, grid_size)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)  # shape: (2, grid_size, grid_size)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        extra = np.zeros([extra_tokens, embed_dim], dtype=np.float32)
        pos_embed = np.concatenate([extra, pos_embed], axis=0)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """
    :param embed_dim: Total embedding dimension (must be even)
    :param grid: Array of shape (2, 1, grid_size, grid_size)
    :return: Positional embedding of shape [grid_size^2, embed_dim]
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even"
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    :param embed_dim: Output embedding dimension for each position (must be even)
    :param pos: 1D array of positions to be encoded, shape (M,)
    :return: Positional encoding of shape (M, embed_dim)
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even"

    omega: np.ndarray = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)  # shape: (D/2,)

    pos = pos.reshape(-1)  # shape: (M,)
    out = np.einsum("m,d->md", pos, omega)  # shape: (M, D/2)

    return np.concatenate([np.sin(out), np.cos(out)], axis=1)  # shape: (M, D)

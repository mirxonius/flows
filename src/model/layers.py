from typing import Optional, Tuple, Union
import torch.nn as nn
import torch
import math
from einops import rearrange
from einops.layers.torch import Rearrange

from src.model.utils import modulate


class Mlp(nn.Sequential):
    """Multi-layer perceptron with SiLU activation."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
    ) -> None:
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        super().__init__(
            nn.Linear(in_features, hidden_features, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features, bias=True),
        )


class PatchEmbed(nn.Module):
    """
    Embeds images into patches using a convolutional layer.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        patch_size: Union[int, Tuple[int, int]],
        in_channels: int,
        model_dim: int,
    ) -> None:
        """
        Args:
            input_size: (height, width) of the input image.
            patch_size: Size of each patch (Ph, Pw) or single int for square patches.
            in_channels: Number of input channels (e.g., 3 for RGB).
            model_dim: Dimension of the patch embedding (model_dim).
        """
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.patch_size: Tuple[int, int] = patch_size
        self.input_size: Tuple[int, int] = input_size

        self.proj: nn.Conv2d = nn.Conv2d(
            in_channels,
            model_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.num_patches: int = (input_size[0] // patch_size[0]) * (
            input_size[1] // patch_size[1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Patch embeddings of shape (B, N_patches, model_dim)
        """
        x = self.proj(x)  # (B, model_dim, H', W')
        x = rearrange(x, "b c h w -> b (h w) c")  # (B, N_patches, model_dim)
        return x


class Unpatchify(Rearrange):
    """Converts patch tokens back to image format."""

    def __init__(
        self,
        horizontal_patches: int,
        vertical_patches: int,
        patch_size: int,
        out_channels: int,
    ):
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        super(Unpatchify, self).__init__(
            "n (h w) (p1 p2 c) -> n c (h p1) (w p2)",
            h=vertical_patches,
            w=horizontal_patches,
            p1=patch_size[0],
            p2=patch_size[1],
            c=out_channels,
        )


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        self.attn_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=model_dim, num_heads=num_heads, bias=True, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(model_dim * mlp_ratio)
        self.mlp = Mlp(in_features=model_dim, hidden_features=mlp_hidden_dim)
        self.adaptive_norm_params = nn.Sequential(
            nn.SiLU(), nn.Linear(model_dim, 6 * model_dim, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = (
            self.adaptive_norm_params(c).chunk(6, dim=1)
        )
        q = k = v = modulate(self.attn_norm(x), shift_msa, scale_msa)
        a, _ = self.attn(q, k, v)
        x = x + gate_msa.unsqueeze(1) * a
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.mlp_norm(x), shift_mlp, scale_mlp)
        )
        return x


class DiTFinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self, model_dim: int, patch_size: int | tuple[int], out_channels: int
    ) -> None:
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.proj = nn.Linear(
            model_dim, patch_size[0] * patch_size[1] * out_channels, bias=True
        )
        self.adaptive_norm_params = nn.Sequential(
            nn.SiLU(), nn.Linear(model_dim, 2 * model_dim, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaptive_norm_params(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.proj(x)
        return x

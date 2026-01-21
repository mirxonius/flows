from typing import Optional
import torch.nn as nn
import torch

from src.model.blocks import FCNNBlock
from src.utils import temporal_encoding
from src.model.layers import PatchEmbed, DiTBlock, Unpatchify, DiTFinalLayer
from src.model.utils import get_2d_sincos_pos_embed
from src.model.TemporalEmbedding import TemporalEmbedding
from src.model.LabelEmbedder import LabelEmbedder


class SimpleVelocityModel(nn.Module):
    """
    Simple MLP-based velocity model for flow matching.
    Works with flat/1D data representations.
    """

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


class ViTVelocityModel(nn.Module):
    """
    Vision Transformer based velocity model for flow matching with images.
    Implements DiT (Diffusion Transformer) architecture with conditional generation.
    """

    def __init__(
        self,
        input_size: int | tuple[int],
        patch_size: int | tuple[int],
        in_channels: int,
        model_dim: int,
        num_heads: int,
        depth: int = 6,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.2,
        num_classes: Optional[int] = 10,
        num_freqs: Optional[int] = 256,
    ):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes

        self.x_embedder = PatchEmbed(
            in_channels=self.in_channels,
            model_dim=model_dim,
            input_size=input_size,
            patch_size=patch_size,
        )
        self.time_embedder = TemporalEmbedding(
            model_dim=model_dim, frequency_embedding_size=num_freqs
        )

        self.conditional = num_classes is not None and class_dropout_prob > 0

        if self.conditional:
            self.label_embedder = LabelEmbedder(
                num_classes=num_classes, model_dim=model_dim, dropout_prob=class_dropout_prob  # type: ignore
            )
        self.blocks = nn.ModuleList(
            [DiTBlock(model_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )

        num_patches = self.x_embedder.num_patches
        self.unpatchify = Unpatchify(
            horizontal_patches=input_size[0] // patch_size[0],
            vertical_patches=input_size[1] // patch_size[1],
            patch_size=patch_size,
            out_channels=self.out_channels,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, model_dim), requires_grad=False
        )
        self._initialize_positional_encoding()

        self.final_layer = DiTFinalLayer(
            model_dim=model_dim,
            out_channels=self.out_channels,
            patch_size=self.patch_size,
        )

    def _initialize_positional_encoding(self) -> None:
        pos_embd_params = get_2d_sincos_pos_embed(
            embed_dim=self.pos_embed.shape[-1],
            grid_size=int(self.x_embedder.num_patches**0.5),
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embd_params).float().unsqueeze(0)
        )

    def forward(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        apply_class_dropout: bool = False,
    ) -> torch.Tensor:
        x = self.x_embedder(x) + self.pos_embed
        t = self.time_embedder(t)
        if self.conditional:
            y = self.label_embedder(y, apply_class_dropout)  # (B, D)
        else:
            y = torch.zeros_like(t)
        c = t + y  # (B, D)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

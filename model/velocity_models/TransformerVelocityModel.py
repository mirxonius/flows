from typing import Optional
import torch.nn as nn
import torch
from flows.FlowMatching.model.layers import (
    PatchEmbed,
    DiTBlock,
    Unpatchify,
    DiTFinalLayer,
)
from flows.FlowMatching.model.utils import get_2d_sincos_pos_embed
from flows.FlowMatching.model.TemporalEmbedding import TemporalEmbedding
from flows.FlowMatching.model.LabelEmbedder import LabelEmbedder


class ViTVelocityModel(nn.Module):
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


if __name__ == "__main__":
    model = ViTVelocityModel(
        input_size=32, patch_size=2, in_channels=4, model_dim=128, num_heads=8
    )
    print(model)
    x = torch.randn(3, 4, 32, 32)
    y = torch.randint(low=0, high=10, size=(3,))
    t = (
        0.5
        * torch.ones(
            3,
        ).float()
    )
    print(model(t, x, y).shape)

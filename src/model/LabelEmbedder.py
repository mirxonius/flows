import torch
import torch.nn as nn


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes: int, model_dim: int, dropout_prob: float) -> None:
        super().__init__()
        use_cfg_embedding: bool = dropout_prob > 0
        self.embedding = nn.Embedding(num_classes + int(use_cfg_embedding), model_dim)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Drops labels to enable classifier-free guidance.

        Args:
            labels: torch.Tensor of shape (B,) containing class indices.

        Returns:
            torch.Tensor of shape (B,) with some labels possibly replaced with cfg token.
        """
        batch_size, *_ = labels.shape
        drop_ids = torch.rand(batch_size, device=labels.device) < self.dropout_prob
        return torch.where(
            drop_ids, torch.full_like(labels, fill_value=self.num_classes), labels
        )

    def forward(self, labels: torch.Tensor, should_drop: bool) -> torch.Tensor:
        """
        Args:
            labels: torch.Tensor of shape (B,) containing class indices.
            should_drop: Whether to apply label dropout (usually True during training).

        Returns:
            torch.Tensor of shape (B, model_dim) containing label embeddings.
        """
        use_dropout: bool = self.dropout_prob > 0
        if use_dropout and should_drop:
            labels = self.token_drop(labels)
        return self.embedding(labels)

from pathlib import Path
import torch
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from src.model.DiTFlowMatchingModel import DiTFlowMatchingModel


def visualize_model(
    model: DiTFlowMatchingModel,
    fig: plt.Figure,
    axes: NDArray,
    epoch: int,
    save_path: str | Path,
    guidance_scale: float = 3,
):
    """
    Visualize model samples and save to disk.

    Args:
        model: The DiT flow matching model to visualize
        fig: Matplotlib figure
        axes: Array of axes for plotting
        epoch: Current epoch number
        save_path: Path to save the visualization
        guidance_scale: Guidance scale for sampling
    """
    labels = (torch.linspace(0, 15, 16) % 10).long().to(model.device)
    samples = (
        model.sample(guidance=labels, guidance_scale=guidance_scale)
        .cpu()
        .detach()
        .numpy()
    )
    axes = axes.flatten()

    for i in range(16):
        axes[i].set_title(f"{i%10}")
        axes[i].imshow(samples[i].squeeze(0))
        axes[i].axis("off")  # Optional: turn off axis for cleaner look

    fig.suptitle(f"Epoch {epoch}", fontsize=16)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path / f"epoch_{epoch}_guidance_{guidance_scale}_scale.png")

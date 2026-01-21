from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.distributions import Uniform
import matplotlib.pyplot as plt

from src.model.FlowMatchingModel import FlowMatchingModel
from src.model.DiTFlowMatchingModel import DiTFlowMatchingModel
from src.utils import optimal_transport_sampling, model_size_b
from src.callbacks import visualize_model


def train_flow_model(
    model: FlowMatchingModel,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    num_epochs: int = 300,
    lr: float = 1e-3,
    device: str = "cpu",
    optimal_transport: bool = True,
) -> FlowMatchingModel:

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    t_dist = Uniform(low=torch.zeros(1).float(), high=torch.ones(1).float())
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        for x1 in dataloader:
            optimizer.zero_grad()
            x1 = x1.to(device)
            x0 = model.base_dist.rsample(sample_shape=(x1.shape[0],))

            if optimal_transport:
                src, dst = optimal_transport_sampling(x0, x1)
                x0 = x0[src, ...]
                x1 = x1[dst, ...]

            x0, x1 = x0.to(device), x1.to(device)
            target_velocity = x1 - x0
            t = t_dist.rsample(sample_shape=(x1.shape[0],)).to(device).view(-1, 1)
            xt = t * x1 + (1 - t) * x0
            pred_velocity = model(xt, t)
            loss: torch.Tensor = loss_fn(pred_velocity, target_velocity)
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str(f"Epoch:{epoch:4d}\tLoss = {loss.item():10.6f}")
    return model


def train_dit_flow_model(
    model: DiTFlowMatchingModel,
    dataloader: DataLoader,
    loss_fn: nn.Module = nn.MSELoss(),
    num_epochs: int = 300,
    lr: float = 1e-3,
    device: str = "cpu",
    optimal_transport: bool = True,
    running_in_notebook: bool = False,
    weight_decay: float = 0.1,
    plot_every: int = 10,
    save_path: str = "figs",
) -> DiTFlowMatchingModel:
    """
    Train a DiT (Diffusion Transformer) flow matching model with conditional generation.

    Args:
        model: DiTFlowMatchingModel to train
        dataloader: DataLoader providing (image, label) pairs
        loss_fn: Loss function (default: MSELoss)
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        optimal_transport: Whether to use optimal transport matching
        running_in_notebook: Whether running in Jupyter notebook (for live visualization)
        weight_decay: Weight decay for AdamW optimizer
        plot_every: Plot samples every N epochs (when not in notebook)
        save_path: Path to save visualization plots

    Returns:
        Trained model
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for epoch in range(1, num_epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for i, (x1, label) in enumerate(pbar):
            batch_size = x1.shape[0]

            optimizer.zero_grad()
            label = label.to(device)
            x1 = x1.to(device)
            x0 = model.sample_noise(batch_size=batch_size)

            if optimal_transport:
                src, dst = optimal_transport_sampling(x0, x1)
                x0 = x0[src, ...]
                x1 = x1[dst, ...]

            x0, x1 = x0.to(device), x1.to(device)

            target_velocity = x1 - x0
            t = torch.rand(size=(batch_size,), device=device)

            # Reshape t to match data dimensions
            t_view = t.view(-1, *([1] * len(model.data_shape)))
            xt = torch.lerp(x0, x1, t_view)
            pred_velocity = model(t, xt, label, apply_class_dropout=True)

            loss: torch.Tensor = loss_fn(pred_velocity, target_velocity)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix_str(
                f"Loss: {loss.item():.4f} | Avg: {epoch_loss/num_batches:.4f}"
            )

        scheduler.step()

        # Visualization
        if epoch % plot_every == 0:
            model.eval()
            with torch.no_grad():
                for guidance_scale in [1, 3, 5]:
                    visualize_model(
                        model,
                        fig,
                        axes,
                        epoch=epoch,
                        save_path=save_path,
                        guidance_scale=guidance_scale,
                    )
            model.train()

    return model

from IPython.display import display, clear_output
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import matplotlib.pyplot as plt

from flows.FlowMatching.model.FlowMatchingModel import FlowMatchingModel

from flows.FlowMatching.utils import optimal_transport_sampling, model_size_b
from flows.FlowMatching.callbacks import visaulize_model


def train_flow_model(
    model: FlowMatchingModel,
    dataloader: DataLoader,
    loss_fn: nn.Module = nn.MSELoss(),
    num_epochs: int = 300,
    lr: float = 1e-3,
    device: str = "cpu",
    optimal_transport: bool = True,
    running_in_notebook: bool = True,
    weight_decay: float = 0.1,
    plot_every: int = 10,
) -> FlowMatchingModel:

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for epoch in range(1, num_epochs + 1):
        pbar = tqdm(dataloader)
        model.train()
        for i, (x1, label) in enumerate(pbar):
            batch_size = x1.shape[0]

            optimizer.zero_grad()
            label = label.to(device)
            x1 = x1.to(device)
            x0 = model.sample_noise(batch_size=batch_size).to(device)
            if optimal_transport:
                src, dst = optimal_transport_sampling(x0, x1)
                x0 = x0[src, ...]
                x1 = x1[dst, ...]

            x0, x1 = x0.to(device), x1.to(device)

            target_velocity = x1 - x0
            t = torch.rand(size=(batch_size,), device=device)

            # NOTE: `torch.lerp`: xt = t * x1 + (1 - t) * x0
            xt = torch.lerp(x0, x1, t.view(-1, 1, 1, 1))
            pred_velocity = model(t, xt, label, apply_class_dropout=True)

            loss: torch.Tensor = loss_fn(pred_velocity, target_velocity)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
            optimizer.step()
            pbar.set_postfix_str(
                f"Epoch: {epoch} | Loss: {loss.item():.4f} | Batch: {i+1}/{len(dataloader)}"
            )
        scheduler.step()

        model.eval()  # Set model to evaluation mode for sampling
        if running_in_notebook:
            # 1. Clear the output of the cell, wait for new output to be ready.
            clear_output(wait=True)

            # 2. Call the visualization function to draw on our existing figure.
            visaulize_model(model, fig, axes, epoch=epoch)
            fig.suptitle(
                f"Epoch: {epoch + 1}", fontsize=16
            )  # Add a title to track progress

            # 3. Display the updated figure.
            display(fig)
        else:
            if epoch % plot_every == 0:
                for guidance_scale in [1, 3, 5]:
                    visaulize_model(
                        model,
                        fig,
                        axes,
                        epoch=epoch,
                        save_path="figs",
                        guidance_scale=guidance_scale,
                    )

    return model


if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    from torchvision.datasets import MNIST
    from torchvision import transforms
    from flows.FlowMatching.model.FlowMatchingModel import FlowMatchingModel
    from flows.FlowMatching.model.velocity_models.TransformerVelocityModel import (
        ViTVelocityModel,
    )

    class MNISTDataset(Dataset):
        def __init__(self):
            super().__init__()
            self._data = MNIST(
                root=".",
                download=True,
                train=True,
                transform=transforms.Compose(
                    [
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,)),
                    ]
                ),
            )
            self.num_classes = 10

        def __getitem__(self, index):
            image, label = self._data[index]

            label = torch.tensor(label).long()
            return image, label

        def __len__(self):
            return len(self._data)

    mnist = MNISTDataset()
    dataloader = DataLoader(dataset=mnist, batch_size=256, shuffle=True, num_workers=6)

    velocity_model = ViTVelocityModel(
        model_dim=64, input_size=32, patch_size=4, in_channels=1, num_heads=4, depth=6
    )

    flow_model = FlowMatchingModel(
        velocity_model=velocity_model, data_shape=(1, 32, 32), dt=5e-3
    )
    print(
        f"Num params: {sum([param.numel() for param in flow_model.parameters() if param.requires_grad])/1e6} Million."
    )
    print(f"Model size: {model_size_b(flow_model)/1024 ** 2} MiB")
    model = train_flow_model(
        model=flow_model,
        dataloader=dataloader,
        lr=1e-3,
        num_epochs=200,
        device="cuda:3",
        optimal_transport=False,
        weight_decay=0.05,
        plot_every=20,
        running_in_notebook=False,
    )

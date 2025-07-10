from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.distributions import Uniform

from src.model.FlowMatchingModel import FlowMatchingModel
from src.utils import optimal_transport_sampling


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

import torch.nn as nn
import torch
from torch.distributions import Normal
from flows.FlowMatching.model.velocity_models import SimpleVelocityModel


class FlowMatchingModel(nn.Module):
    def __init__(self, data_dims: int, hidden_dim: int, num_layers: int, dt=1e-2):
        super().__init__()

        self.data_dims = data_dims
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.register_buffer("loc", torch.zeros(data_dims))
        self.register_buffer("scale", torch.ones(data_dims))
        self.velocity_model: nn.Module = SimpleVelocityModel(
            out_dims=data_dims, hidden_dims=[hidden_dim] * num_layers
        )

    @property
    def base_dist(self):
        return Normal(loc=self.loc, scale=self.scale)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u = self.velocity_model(xt, t)
        return u

    def integrate(
        self,
        x0: torch.Tensor,
        dt: float | None = None,
        return_intermediate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if dt is None:
            dt = torch.tensor(self.dt).float().to(x0.device)
        else:
            dt = torch.tensor(dt).float().to(x0.device)
        time_grid = torch.arange(0, 1, dt).to(x0.device)[:, None]
        dt = torch.tensor(dt).float().to(x0.device)

        if return_intermediate:
            intermediates = []
            for t in time_grid:
                x0 = x0 + self(x0, t[:, None]) * dt
                intermediates.append(x0)
            return intermediates
        else:
            for t in time_grid:
                x0 = x0 + self(x0, t[:, None]) * dt
            return x0

    def sample(
        self,
        num_samples: int,
        dt: float | None = None,
        enable_grad: bool = False,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        x0 = self.base_dist.rsample(sample_shape=(num_samples,)).to(self.device)
        with torch.set_grad_enabled(enable_grad):
            x = self.integrate(x0, return_intermediate=return_intermediates, dt=dt)
        return x

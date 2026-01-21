import torch.nn as nn
import torch


class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        velocity_model: nn.Module,
        data_shape: tuple[int],
        dt: int = 1e-2,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.data_shape = data_shape
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        guidance: torch.Tensor,
        apply_class_dropout: bool = False,
    ) -> torch.Tensor:

        velocity = self.velocity_model(
            t, xt, guidance, apply_class_dropout=apply_class_dropout
        )

        return velocity

    def drift_coefficient(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        guidance: torch.Tensor,
        guidance_scale: float,
    ):
        unguided_velocity: torch.Tensor = self.velocity_model(
            t, xt, guidance, apply_class_dropout=True
        )
        guided_velocity: torch.Tensor = self.velocity_model(
            t, xt, guidance, apply_class_dropout=False
        )
        return unguided_velocity.mul(1 - guidance_scale) + guided_velocity.mul(
            guidance_scale
        )

    def fast_euler(
        self,
        x0: torch.Tensor,
        guidance: torch.Tensor,
        t_span: torch.Tensor,
        dt: torch.Tensor,
        guidance_scale: float = None,
    ) -> torch.Tensor:
        """
        Simplest form of numerical integraion. Use when speed is prefered to precision.
        """
        for tt in t_span:
            tt = tt.repeat(x0.shape[0])
            x0 += self.drift_coefficient(
                tt, x0, guidance, guidance_scale=guidance_scale
            ).mul_(dt)
        return x0

    def fast_rk4(
        self,
        x0: torch.Tensor,
        guidance: torch.Tensor,
        t_span: torch.Tensor,
        dt: torch.Tensor,
        guidance_scale: float = None,
    ) -> torch.Tensor:
        """
        Simple form of Runge-Kutta 4th order integraion.
        Middle ground between speed and precision.
        """
        for tt in t_span:
            tt = tt.repeat(x0.shape[0])
            F1 = self.drift_coefficient(
                tt, x0, guidance, guidance_scale=guidance_scale
            ).mul_(dt)
            F2 = self.drift_coefficient(
                tt.add(0.5 * dt),
                x0 + F1.mul(0.5),
                guidance,
                guidance_scale=guidance_scale,
            ).mul_(dt)
            F3 = self.drift_coefficient(
                tt.add(0.5 * dt),
                x0 + F2.mul(0.5),
                guidance,
                guidance_scale=guidance_scale,
            ).mul_(dt)
            F4 = self.drift_coefficient(
                tt.add(dt), x0 + F3, guidance, guidance_scale=guidance_scale
            ).mul_(dt)
            x0.add_((F1 + F2.mul_(2) + F3.mul_(2) + F4).mul_(1 / 6))
        return x0

    def integrate(
        self,
        x0: torch.Tensor,
        guidance: torch.Tensor,
        integration_method: str = "fast_euler",
        guidance_scale: float = None,
        dt: float = None,  # dopri5
    ) -> torch.Tensor:
        if dt is None:
            dt = self.dt
        t_span = torch.arange(0, 1, dt).to(x0.device)
        if integration_method == "fast_euler":
            return self.fast_euler(
                x0, guidance, t_span, dt=dt, guidance_scale=guidance_scale
            )
        elif integration_method == "fast_rk4":
            return self.fast_rk4(
                x0, guidance, t_span, dt=dt, guidance_scale=guidance_scale
            )
        else:
            raise ValueError(f"Integration method: {integration_method} not supported!")

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        noise = torch.randn((batch_size, *self.data_shape), device=self.device)
        return noise

    @torch.no_grad
    def sample(
        self,
        num_samples: int = None,
        guidance: torch.Tensor = None,
        integration_method: str = "fast_euler",
        guidance_scale: float = None,
    ) -> torch.Tensor:
        if num_samples is not None and guidance is not None:
            raise ValueError(
                "Cannot specify both 'num_samples' and 'guidance'. Please provide only one."
            )

        if num_samples is None and guidance is None:
            raise ValueError("Must specify either 'num_samples' or 'guidance'.")
        if num_samples is None:
            num_samples = guidance.shape[0]

        x = self.sample_noise(num_samples)
        x = self.integrate(
            x,
            guidance,
            guidance_scale=guidance_scale,
            integration_method=integration_method,
        )
        return x

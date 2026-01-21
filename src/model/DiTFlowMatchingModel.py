import torch.nn as nn
import torch


class DiTFlowMatchingModel(nn.Module):
    """
    Flow Matching Model with DiT (Diffusion Transformer) support.

    This model supports conditional generation with classifier-free guidance
    and multiple integration methods (Euler, RK4).

    Args:
        velocity_model: A velocity model that accepts (t, x, guidance, apply_class_dropout)
        data_shape: Shape of the data (e.g., (3, 32, 32) for 32x32 RGB images)
        dt: Time step for integration (default: 1e-2)
    """

    def __init__(
        self,
        velocity_model: nn.Module,
        data_shape: tuple[int],
        dt: float = 1e-2,
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
        """
        Forward pass through the velocity model.

        Args:
            t: Time tensor of shape (batch_size,)
            xt: Current state of shape (batch_size, *data_shape)
            guidance: Guidance labels of shape (batch_size,)
            apply_class_dropout: Whether to apply classifier-free guidance dropout

        Returns:
            Velocity prediction of shape (batch_size, *data_shape)
        """
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
        """
        Compute drift coefficient with classifier-free guidance.

        Args:
            t: Time tensor
            xt: Current state
            guidance: Guidance labels
            guidance_scale: Guidance strength (higher = more guided)

        Returns:
            Guided velocity
        """
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
        Euler integration method.

        Simplest form of numerical integration. Use when speed is preferred to precision.

        Args:
            x0: Initial state
            guidance: Guidance labels
            t_span: Time points to integrate over
            dt: Time step
            guidance_scale: Guidance strength

        Returns:
            Final integrated state
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
        Runge-Kutta 4th order integration method.

        Middle ground between speed and precision.

        Args:
            x0: Initial state
            guidance: Guidance labels
            t_span: Time points to integrate over
            dt: Time step
            guidance_scale: Guidance strength

        Returns:
            Final integrated state
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
        dt: float = None,
    ) -> torch.Tensor:
        """
        Integrate from noise to data.

        Args:
            x0: Initial noise
            guidance: Guidance labels
            integration_method: "fast_euler" or "fast_rk4"
            guidance_scale: Guidance strength
            dt: Time step (uses self.dt if None)

        Returns:
            Generated samples
        """
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
        """
        Sample noise from standard Gaussian.

        Args:
            batch_size: Number of samples

        Returns:
            Noise tensor of shape (batch_size, *data_shape)
        """
        noise = torch.randn((batch_size, *self.data_shape), device=self.device)
        return noise

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = None,
        guidance: torch.Tensor = None,
        integration_method: str = "fast_euler",
        guidance_scale: float = None,
    ) -> torch.Tensor:
        """
        Sample from the model.

        Args:
            num_samples: Number of samples (if guidance not provided)
            guidance: Guidance labels (if num_samples not provided)
            integration_method: "fast_euler" or "fast_rk4"
            guidance_scale: Guidance strength

        Returns:
            Generated samples
        """
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

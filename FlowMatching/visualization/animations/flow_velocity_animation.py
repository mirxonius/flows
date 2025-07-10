import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from flows.FlowMatching.model.FlowMatchingModel import FlowMatchingModel


@torch.no_grad()
def animate_flow_velocity_field(
    flow_model: FlowMatchingModel,
    x_range=(-3, 3),
    y_range=(-3, 3),
    grid_density=20,
    arrow_scale=1.0,
    frames=110,
    interval=110,
    figsize=(10, 8),
    dt: float = 1e-2,
    arrow_length: float = 0.1,
    title: str = "Flow Velocity Field",
):
    """
    Animate a 2D vector field.

    Parameters:
    -----------
    flow_model : FlowMatchingModel
    x_range : tuple
        Range for x-axis (min, max)
    y_range : tuple
        Range for y-axis (min, max)
    grid_density : int
        Number of grid points along each axis
    arrow_scale : float
        Scale factor for arrow size
    frames : int
        Number of animation frames
    interval : int
        Delay between frames in milliseconds
    figsize : tuple
        Figure size (width, height)

    Returns:
    --------
    matplotlib.animation.FuncAnimation
    """

    flow_model.eval()
    # Create coordinate grids
    x = np.linspace(x_range[0], x_range[1], grid_density)
    y = np.linspace(y_range[0], y_range[1], grid_density)
    X, Y = np.meshgrid(x, y)
    grid = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    grid = torch.tensor(grid).float()
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Initialize quiver plot
    field = flow_model(grid, torch.tensor(1.0).view(1, 1))
    field = field.view(grid_density, grid_density, 2).numpy()
    U, V = field[..., 0], field[..., 1]

    magnitude = np.sqrt(U**2 + V**2)
    nonzero_mask = magnitude > 1e-10
    U_norm = np.zeros_like(U)
    V_norm = np.zeros_like(V)
    U_norm[nonzero_mask] = U[nonzero_mask] / magnitude[nonzero_mask] * arrow_length
    V_norm[nonzero_mask] = V[nonzero_mask] / magnitude[nonzero_mask] * arrow_length

    quiver = ax.quiver(
        X,
        Y,
        U_norm,
        V_norm,
        magnitude,
        angles="xy",
        scale_units="xy",
        scale=1 / arrow_scale,
        alpha=0.8,
        color="blue",
    )
    cbar = plt.colorbar(quiver, ax=ax, label="Magnitude")

    @torch.no_grad()
    def update(frame):
        """Update function for animation"""
        t = torch.tensor(frame * dt).float().view(1, 1)
        field: torch.Tensor = flow_model(grid, t)
        field = field.view(grid_density, grid_density, 2).numpy()
        U, V = field[..., 0], field[..., 1]  # Time step
        magnitude = np.sqrt(U**2 + V**2)
        nonzero_mask = magnitude > 1e-10
        U_norm = np.zeros_like(U)
        V_norm = np.zeros_like(V)
        U_norm[nonzero_mask] = U[nonzero_mask] / magnitude[nonzero_mask] * arrow_length
        V_norm[nonzero_mask] = V[nonzero_mask] / magnitude[nonzero_mask] * arrow_length

        quiver.set_UVC(U_norm, V_norm, magnitude)
        ax.set_title(f"2D Vector Field Animation (t = {t.item():.2f})")
        return (quiver,)

    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=frames, interval=interval, blit=False, repeat=True
    )

    plt.tight_layout()
    return anim

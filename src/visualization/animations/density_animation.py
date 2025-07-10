import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_density_evolution_animation(
    X, save_path=None, fps=10, zoom_factor=2, title: str = "Density Evolution"
):
    """
    Enhanced version with smoother interpolation and better controls.

    Args:
        X: Array of data samples for each time step
        save_path: Optional path to save the animation
        fps: Frames per second
        zoom_factor: Upsampling factor for smoother visualization
    """

    fig, ax = plt.subplots(figsize=(12, 10))

    # Pre-compute all histograms for consistent color scaling
    print("Pre-computing histograms...")
    all_histograms = []
    max_density = 0

    for tidx in range(len(X)):
        H, xbins, ybins = np.histogram2d(
            x=X[tidx, :, 0],
            y=X[tidx, :, 1],
            range=([-3, 3], [-3, 3]),
            bins=150,
            density=True,
        )

        # Upsample
        H_img = zoom(H, zoom=(zoom_factor, zoom_factor), order=1)
        x_edges = zoom(xbins, zoom=zoom_factor, order=1)
        y_edges = zoom(ybins, zoom=zoom_factor, order=1)

        all_histograms.append((H_img, x_edges, y_edges))
        max_density = max(max_density, H_img.max())

    print(f"Max density: {max_density:.4f}")

    # Initialize plot
    im = ax.pcolormesh(
        all_histograms[0][1][1:-1],
        all_histograms[0][2][1:-1],
        all_histograms[0][0].T,
        shading="auto",
        cmap="viridis",
        vmin=0,
        vmax=max_density,
    )

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Density", fontsize=12)

    # Add time step text
    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    def animate(tidx):
        H_img, x_edges, y_edges = all_histograms[tidx]

        # Update the plot data
        im.set_array(H_img.T.ravel())

        # Update time step text
        time_text.set_text(f"Time Step: {tidx}")

        return [im, time_text]

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(X), interval=1000 / fps, blit=True, repeat=True
    )

    # Save animation if path provided
    if save_path:
        print(f"Saving animation to {save_path}...")
        if save_path.endswith(".gif"):
            anim.save(save_path, writer="pillow", fps=fps)
        elif save_path.endswith(".mp4"):
            anim.save(save_path, writer="ffmpeg", fps=fps, bitrate=1800)
        print("Animation saved!")

    plt.tight_layout()
    return anim

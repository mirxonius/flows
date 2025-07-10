import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_generation_trajectory_animation(
    X,
    save_path=None,
    fps=10,
    title: str = "Data Trajectory",
    trace_paths=False,
    trace_alpha=0.3,
):
    """
    Enhanced version with smoother interpolation, better controls, and optional path tracing.

    Args:
        X: Array of data samples for each time step (shape: [n_timesteps, n_points, 2])
        save_path: Optional path to save the animation
        fps: Frames per second
        zoom_factor: Upsampling factor for smoother visualization
        title: Title for the animation
        trace_paths: If True, shows the path each point has traveled
        trace_alpha: Transparency of the traced paths (0-1)
    """

    fig, ax = plt.subplots(figsize=(12, 10))

    # Initialize scatter plot
    scatter = ax.scatter(X[0, :, 0], X[0, :, 1], c="tab:blue", s=50, alpha=0.7)

    # Initialize path lines if tracing is enabled
    path_lines = []
    if trace_paths:
        n_points = X[0].shape[0]
        for i in range(n_points):
            (line,) = ax.plot([], [], "-", alpha=trace_alpha, linewidth=1, color="gray")
            path_lines.append(line)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

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
        # Update scatter plot positions
        current_positions = X[tidx]
        scatter.set_offsets(current_positions)

        # Update path traces if enabled
        if trace_paths:
            for i, line in enumerate(path_lines):
                # Get trajectory for point i up to current time
                trajectory = X[: tidx + 1, i, :]
                line.set_data(trajectory[:, 0], trajectory[:, 1])

        # Update time step text
        time_text.set_text(f"Time Step: {tidx}")

        # Return all artists that need to be redrawn
        artists = [scatter, time_text]
        if trace_paths:
            artists.extend(path_lines)

        return artists

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

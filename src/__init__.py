"""Flow Matching: Continuous Normalizing Flows with Optimal Transport."""

from src.model import (
    FlowMatchingModel,
    DiTFlowMatchingModel,
    SimpleVelocityModel,
    ViTVelocityModel,
)
from src.train_flow import train_flow_model, train_dit_flow_model
from src.utils import temporal_encoding, optimal_transport_sampling, model_size_b
from src.callbacks import visualize_model

__version__ = "0.2.0"

__all__ = [
    "FlowMatchingModel",
    "DiTFlowMatchingModel",
    "SimpleVelocityModel",
    "ViTVelocityModel",
    "train_flow_model",
    "train_dit_flow_model",
    "temporal_encoding",
    "optimal_transport_sampling",
    "model_size_b",
    "visualize_model",
]

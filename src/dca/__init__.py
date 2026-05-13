"""Dual-channel attribution package for dynamic graph explanations."""

from .utils import build_financial_sandbox_graph, linear_interpolation_path
from .custom_message_passing import DualChannelMessagePassing
from .custom_autograd import AttributionSumGate

__all__ = [
    "build_financial_sandbox_graph",
    "linear_interpolation_path",
    "DualChannelMessagePassing",
    "AttributionSumGate",
]

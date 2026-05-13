"""Utility functions for dynamic graph attribution experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import torch


@dataclass(frozen=True)
class FinancialSandboxGraph:
    """Container for a 5-node toy interbank exposure network."""

    edge_index: torch.Tensor
    x0: torch.Tensor
    x1: torch.Tensor
    edge_attr0: torch.Tensor
    edge_attr1: torch.Tensor


def build_financial_sandbox_graph() -> FinancialSandboxGraph:
    """Build a tiny N=5 undirected financial contagion-style graph."""

    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 0, 4],
            [1, 0, 2, 1, 3, 2, 4, 3, 4, 0],
        ],
        dtype=torch.long,
    )

    # Node feature: institution liquidity buffer ratio in [0, 1].
    x0 = torch.tensor([[0.62], [0.55], [0.48], [0.65], [0.52]], dtype=torch.float64)
    x1 = torch.tensor([[0.58], [0.49], [0.51], [0.60], [0.56]], dtype=torch.float64)

    # edge attribute: bilateral exposure intensity
    edge_attr0 = torch.tensor(
        [0.20, 0.20, 0.15, 0.15, 0.30, 0.30, 0.18, 0.18, 0.22, 0.22],
        dtype=torch.float64,
    )
    edge_attr1 = torch.tensor(
        [0.24, 0.24, 0.19, 0.19, 0.26, 0.26, 0.23, 0.23, 0.25, 0.25],
        dtype=torch.float64,
    )

    return FinancialSandboxGraph(edge_index=edge_index, x0=x0, x1=x1, edge_attr0=edge_attr0, edge_attr1=edge_attr1)


def linear_interpolation_path(
    x0: torch.Tensor,
    x1: torch.Tensor,
    edge_attr0: torch.Tensor,
    edge_attr1: torch.Tensor,
    steps: int = 50,
) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    """Yield linear path points from baseline graph G0 to target graph G1.

    Returns:
        Tuples of (alpha, x_alpha, edge_alpha), where alpha is the interpolation
        coefficient, x_alpha is interpolated node features, and edge_alpha is
        interpolated edge attributes.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")

    dx = x1 - x0
    de = edge_attr1 - edge_attr0
    alphas = torch.linspace(0.0, 1.0, steps + 1, dtype=x0.dtype)

    for alpha in alphas:
        x_alpha = x0 + alpha * dx
        edge_alpha = edge_attr0 + alpha * de
        yield alpha, x_alpha, edge_alpha

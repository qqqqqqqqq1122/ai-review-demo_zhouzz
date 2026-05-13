"""Run 5-node sandbox verification for dual-channel attribution."""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dca.custom_autograd import AttributionSumGate
from dca.custom_message_passing import DualChannelMessagePassing
from dca.utils import build_financial_sandbox_graph, linear_interpolation_path


class TinyDynamicGNN(nn.Module):
    """Minimal linear dynamic GNN for exact IG completeness in sandbox."""

    def __init__(self):
        super().__init__()
        self.mp = DualChannelMessagePassing()
        self.readout = nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.readout.weight.fill_(1.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        out = self.mp(x, edge_index, edge_attr)
        return self.readout(out).squeeze(-1)


def integrated_gradients_joint(
    model: nn.Module,
    edge_index: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    e0: torch.Tensor,
    e1: torch.Tensor,
    target_node: int,
    steps: int = 50,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute joint integrated gradients for feature and structure channels.

    Args:
        model: Dynamic GNN model.
        edge_index: Graph connectivity tensor.
        x0: Baseline node features.
        x1: Target node features.
        e0: Baseline edge attributes.
        e1: Target edge attributes.
        target_node: Node index whose prediction is explained.
        steps: Number of path discretization steps.

    Returns:
        (attr_x, attr_e): feature-channel and edge-channel attribution tensors.
    """
    dx = x1 - x0
    de = e1 - e0

    grad_x_acc = torch.zeros_like(x0)
    grad_e_acc = torch.zeros_like(e0)

    for _, x_alpha, e_alpha in linear_interpolation_path(x0, x1, e0, e1, steps=steps):
        x_alpha = x_alpha.clone().detach().requires_grad_(True)
        e_alpha = e_alpha.clone().detach().requires_grad_(True)
        y = model(x_alpha, edge_index, e_alpha)[target_node]
        gx, ge = torch.autograd.grad(y, (x_alpha, e_alpha), create_graph=False)
        grad_x_acc += gx
        grad_e_acc += ge

    # We sample both endpoints alpha=0 and alpha=1, so the mean uses (steps + 1) points.
    avg_grad_x = grad_x_acc / (steps + 1)
    avg_grad_e = grad_e_acc / (steps + 1)

    attr_x = dx * avg_grad_x
    attr_e = de * avg_grad_e
    return attr_x, attr_e


def main():
    torch.set_default_dtype(torch.float64)
    sandbox = build_financial_sandbox_graph()
    model = TinyDynamicGNN()

    target_node = 2
    steps = 50

    with torch.no_grad():
        y0 = model(sandbox.x0, sandbox.edge_index, sandbox.edge_attr0)[target_node]
        y1 = model(sandbox.x1, sandbox.edge_index, sandbox.edge_attr1)[target_node]
        output_diff = y1 - y0

    attr_x, attr_e = integrated_gradients_joint(
        model=model,
        edge_index=sandbox.edge_index,
        x0=sandbox.x0,
        x1=sandbox.x1,
        e0=sandbox.edge_attr0,
        e1=sandbox.edge_attr1,
        target_node=target_node,
        steps=steps,
    )

    attribute_sum = attr_x.sum() + attr_e.sum()
    error = torch.abs(attribute_sum - output_diff).item()

    # Execute custom backward assertion.
    attr_x_check = attr_x.detach().requires_grad_(True)
    attr_e_check = attr_e.detach().requires_grad_(True)
    gated = AttributionSumGate.apply(attr_x_check, attr_e_check, output_diff.detach(), 1e-6)
    gated.backward()

    print(f"Target node: {target_node}")
    print(f"model_output_diff: {output_diff.item():.12f}")
    print(f"attribute_sum: {attribute_sum.item():.12f}")
    print(f"abs_error: {error:.12e}")

    assert error <= 1e-6, f"Summation-to-delta failed: error={error:.3e}"
    print("PASS: Summation-to-delta holds within 1e-6.")


if __name__ == "__main__":
    main()

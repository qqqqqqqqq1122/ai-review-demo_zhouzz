"""Custom autograd utilities for attribution consistency checks."""

from __future__ import annotations

import torch


class AttributionSumGate(torch.autograd.Function):
    """Gate that enforces summation-to-delta in backward with explicit assertions."""

    @staticmethod
    def forward(ctx, feature_attr: torch.Tensor, edge_attr: torch.Tensor, output_diff: torch.Tensor, atol: float):
        total_attr = feature_attr.sum() + edge_attr.sum()
        ctx.save_for_backward(total_attr.detach(), output_diff.detach())
        ctx.atol = float(atol)
        return total_attr

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        total_attr, output_diff = ctx.saved_tensors
        err = torch.abs(total_attr - output_diff).item()
        assert err <= ctx.atol, (
            f"Summation-to-delta violated in backward: |sum(attr)-delta|={err:.3e} > atol={ctx.atol:.1e}"
        )

        grad_feature = grad_output
        grad_edge = grad_output
        return grad_feature, grad_edge, None, None

"""Custom message passing implementation for dual-channel attribution."""

from __future__ import annotations

import torch
from torch import nn

try:
    from torch_geometric.nn import MessagePassing

    _HAS_PYG = True
except ImportError:  # pragma: no cover
    MessagePassing = nn.Module
    _HAS_PYG = False


class DualChannelMessagePassing(MessagePassing):
    """Simple weighted aggregation layer with edge-channel support."""

    def __init__(self):
        kwargs = {"aggr": "add"} if _HAS_PYG else {}
        super().__init__(**kwargs)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if x.requires_grad:
            x.retain_grad()
        if edge_attr.requires_grad:
            edge_attr.retain_grad()

        if _HAS_PYG:
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)

        src, dst = edge_index
        msg = x[src] * edge_attr.view(-1, 1)
        out = torch.zeros_like(x)
        out.index_add_(0, dst, msg)
        return out

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return x_j * edge_attr.view(-1, 1)

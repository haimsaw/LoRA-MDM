from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from lora_pytorch.modules.base import BaseLoRAModule


class LinearLoRAModule(BaseLoRAModule[nn.Linear]):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        # TODO: Add support for bias
        # bias: bool = False,
        alpha: float = 1.0,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.in_proj = nn.Parameter(
            torch.empty((in_features, rank), device=device, dtype=dtype),
            requires_grad=True,
        )
        self.out_proj = nn.Parameter(
            torch.empty((rank, out_features), device=device, dtype=dtype),
            requires_grad=True,
        )
        self.dropout = nn.Dropout(dropout)

        # NOTE: The original LoRA paper recommends multiplying the output of 'in_proj'
        # by (alpha / rank).  This adds more computation to the forward pass, and it's
        # mathematically equivalent to scaling 'in_proj' by (alpha / rank) ahead of
        # time.  I have chosen the second option for simplicity.
        nn.init.kaiming_uniform_(self.in_proj, alpha / rank)
        nn.init.zeros_(self.out_proj)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_features={self.in_features}, "
            f"out_features={self.out_features}, rank={self.rank})"
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.in_proj
        x = self.dropout(x)
        return x @ self.out_proj

    @torch.no_grad()
    def merge(self, module: nn.Linear, inplace: bool = False) -> nn.Linear:
        # einstein notation:
        # - i: input features
        # - o: output features
        # - r: rank
        lora_weight = torch.einsum("i r, r o -> o i", self.in_proj, self.out_proj)

        if inplace:
            module.weight.data += lora_weight
            return module

        out = nn.Linear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        out.weight.data = module.weight.data + lora_weight
        out.bias = module.bias
        return out

    @property
    def weight(self) -> Tensor:
        return torch.einsum("i r, r o -> o i", self.in_proj, self.out_proj)

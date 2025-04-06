from __future__ import annotations

from copy import deepcopy
from typing import NamedTuple, Optional

import torch
from torch import Tensor, nn

from lora_pytorch.modules.base import BaseLoRAModule


class OutProj(NamedTuple):
    weight: Tensor
    bias: Optional[Tensor]


class MultiheadAttentionLoRAModule(BaseLoRAModule[nn.MultiheadAttention]):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        no_lora_q: bool = False
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank
        self.alpha = alpha
        self.kdim = kdim or embed_dim
        self.vdim = vdim or embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        
        self.no_lora_q=no_lora_q
        if no_lora_q:
            self.q_zero = torch.nn.parameter.Parameter(torch.zeros((embed_dim, embed_dim)), requires_grad=False).to(device=device, dtype=dtype)
        else:
            self.q_in = nn.Linear(embed_dim, rank, bias=False).to(device=device, dtype=dtype)
            self.q_out = nn.Linear(rank, embed_dim, bias=False).to(device=device, dtype=dtype)
            
        self.k_in = nn.Linear(embed_dim, rank, bias=False).to(device=device, dtype=dtype)
        self.k_out = nn.Linear(rank, self.kdim, bias=False).to(device=device, dtype=dtype)
        self.v_in = nn.Linear(embed_dim, rank, bias=False).to(device=device, dtype=dtype)
        self.v_out = nn.Linear(rank, self.vdim, bias=False).to(device=device, dtype=dtype)
        # output projection
        self.o_in = nn.Linear(embed_dim, rank, bias=False).to(device=device, dtype=dtype)
        self.o_out = nn.Linear(rank, embed_dim, bias=bias).to(device=device, dtype=dtype)
        self._reset_parameters()

    @property
    def q_proj_weight(self) -> Tensor:
        # einstein notation
        # - o: output channels
        # - i: input channels
        # - r: rank
        if self.no_lora_q:
            return self.q_zero
        else:
            return torch.einsum("o r, r i -> i o", self.q_out.weight, self.q_in.weight) 


    @property
    def k_proj_weight(self) -> Tensor:
        return torch.einsum("o r, r i -> i o", self.k_out.weight, self.k_in.weight)

    @property
    def v_proj_weight(self) -> Tensor:
        return torch.einsum("o r, r i -> i o", self.v_out.weight, self.v_in.weight)

    @property
    def in_proj_weight(self) -> Optional[Tensor]:
        if (self.kdim == self.embed_dim) and (self.vdim == self.embed_dim):
            return torch.cat(
                (self.q_proj_weight, self.k_proj_weight, self.v_proj_weight), dim=0
            )
        else:
            return None

    @property
    def out_proj(self) -> OutProj:
        weight = torch.einsum("o r, r i -> o i", self.o_out.weight, self.o_in.weight)
        return OutProj(weight=weight, bias=self.o_out.bias)

    def _reset_parameters(self):
        # NOTE: The original LoRA paper recommends multiplying the output of 'in_proj'
        # by (alpha / rank).  This adds more computation to the forward pass, and it's
        # mathematically equivalent to scaling 'in_proj' by (alpha / rank) ahead of
        # time.  I have chosen the second option for simplicity.
        if not self.no_lora_q:
            nn.init.kaiming_uniform_(self.q_in.weight, self.alpha / self.rank)
            nn.init.zeros_(self.q_out.weight)
        nn.init.kaiming_uniform_(self.k_in.weight, self.alpha / self.rank)
        nn.init.zeros_(self.k_out.weight)
        nn.init.kaiming_uniform_(self.v_in.weight, self.alpha / self.rank)
        nn.init.zeros_(self.v_out.weight)
        nn.init.kaiming_uniform_(self.o_in.weight, self.alpha / self.rank)
        nn.init.zeros_(self.o_out.weight)
        if self.o_out.bias is not None:
            nn.init.zeros_(self.o_out.bias)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def merge(
        self, module: nn.MultiheadAttention, inplace: bool = False
    ) -> nn.MultiheadAttention:
        out = module if inplace else deepcopy(module)

        if out._qkv_same_embed_dim:
            in_proj_weight = self.in_proj_weight
            assert in_proj_weight is not None
            out.in_proj_weight.data += in_proj_weight.data
        else:
            out.q_proj_weight.data += self.q_proj_weight.data
            out.k_proj_weight.data += self.k_proj_weight.data
            out.v_proj_weight.data += self.v_proj_weight.data

        out.out_proj.weight.data += self.out_proj.weight.data
        if self.out_proj.bias is not None:
            if out.out_proj.bias is not None:
                out.out_proj.bias.data += self.out_proj.bias.data
            else:
                out.out_proj.bias = self.out_proj.bias.data

        return out

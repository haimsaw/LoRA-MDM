from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lora_pytorch.modules.base import BaseLoRAModule


class EmbeddingLoRAModule(BaseLoRAModule[nn.Embedding]):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        rank: int,
        sparse: bool = False,
        alpha: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.sparse = sparse

        self.in_proj = nn.Parameter(
            torch.empty((num_embeddings, rank), device=device, dtype=dtype),
            requires_grad=True,
        )
        self.out_proj = nn.Parameter(
            torch.empty((rank, embedding_dim), device=device, dtype=dtype),
            requires_grad=True,
        )

        # NOTE: The original LoRA paper recommends multiplying the output of 'in_proj'
        # by (alpha / rank).  This adds more computation to the forward pass, and it's
        # mathematically equivalent to scaling 'in_proj' by (alpha / rank) ahead of
        # time.  I have chosen the second option for simplicity.
        nn.init.kaiming_uniform_(self.in_proj, alpha / rank)
        nn.init.zeros_(self.out_proj)

    @property
    def weight(self) -> Tensor:
        return torch.einsum("i r, r o -> i o", self.in_proj, self.out_proj)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, rank={self.rank})"
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.embedding(x, self.weight, sparse=self.sparse)

    @torch.no_grad()
    def merge(self, module: nn.Embedding, inplace: bool = False) -> nn.Embedding:
        # einstein notation:
        # - i: input features
        # - o: output features
        # - r: rank
        lora_weight = torch.einsum("i r, r o -> i o", self.in_proj, self.out_proj)

        if inplace:
            module.weight.data += lora_weight
            return module

        return nn.Embedding(
            num_embeddings=module.num_embeddings,
            embedding_dim=module.embedding_dim,
            sparse=module.sparse,
            _weight=module.weight.data + lora_weight,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )

# This code is based on https://pypi.org/project/lora-pytorch/

from __future__ import annotations

from copy import deepcopy
from itertools import chain

from typing_extensions import Literal

from typing import (
    Generic,
    Iterable,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

import torch
from torch import Tensor, nn

from lora_pytorch.modules.attention import MultiheadAttentionLoRAModule
from lora_pytorch.modules.base import BaseLoRAModule
from lora_pytorch.modules.conv import (
    Conv1dLoRAModule,
    Conv2dLoRAModule,
    Conv3dLoRAModule,
    ConvType,
)
from lora_pytorch.modules.embedding import EmbeddingLoRAModule
from lora_pytorch.modules.linear import LinearLoRAModule

ModuleType = TypeVar("ModuleType", bound=nn.Module)


class LoRA(nn.Module, Generic[ModuleType]):
    def __init__(
        self,
        module: ModuleType,
        lora_module: Optional[BaseLoRAModule],
        enabled: bool = True,
    ):
        super().__init__()
        self.module = module.eval()
        self.lora_module = lora_module
        self.enabled = enabled and lora_module is not None

        if not enabled:
            self.disable_lora()
        else:
            self.enable_lora()

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        enable_grad = (self.lora_module is None) and torch.is_grad_enabled()
        with torch.set_grad_enabled(enable_grad):
            y = self.module(x, *args, **kwargs)
        if self.enabled and self.lora_module is not None:
            y = y + self.lora_module(x)

        return y

    def parameters(self) -> Iterable[nn.Parameter]:  # type: ignore[override]
        def _get_lora_parameters(module: nn.Module):
            parameters = chain(
                *[_get_lora_parameters(child) for child in module.children()]
            )
            if isinstance(module, LoRA) and module.lora_module is not None:
                parameters = chain(parameters, module.lora_module.parameters())

            return parameters

        return _get_lora_parameters(self)

    def enable_lora(self) -> None:
        return enable_lora(self)  # type: ignore

    def disable_lora(self) -> None:
        return disable_lora(self)  # type: ignore

    def remove_lora(self, inplace: bool = False) -> ModuleType:
        """Remove all LoRA modules from the model."""
        return remove_lora(self, inplace=inplace)  # type: ignore

    def merge_lora(self: LoRA[ModuleType], inplace: bool = False) -> ModuleType:
        return merge_lora(self, inplace=inplace)  # type: ignore

    @classmethod
    def _from_linear(cls, module: nn.Linear, rank: int) -> LoRA[nn.Linear]:
        out_size, in_size = module.weight.shape
        device = module.weight.device
        dtype = module.weight.dtype
        lora_module = LinearLoRAModule(
            in_size, out_size, rank=rank, device=device, dtype=dtype
        )
        return LoRA(module, lora_module)

    @classmethod
    def _from_conv(cls, module: ConvType, rank: int) -> LoRA[ConvType]:
        device = module.weight.device
        dtype = module.weight.dtype

        lora_module_cls: Union[
            Type[Conv1dLoRAModule], Type[Conv2dLoRAModule], Type[Conv3dLoRAModule]
        ]
        if isinstance(module, nn.Conv1d):
            lora_module_cls = Conv1dLoRAModule
        elif isinstance(module, nn.Conv2d):
            lora_module_cls = Conv2dLoRAModule
        elif isinstance(module, nn.Conv3d):
            lora_module_cls = Conv3dLoRAModule
        else:
            raise ValueError(f"Unsupported conv module type: {type(module)}")

        lora_module = lora_module_cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            rank=rank,
            kernel_size=module.kernel_size,  # type: ignore
            stride=module.stride,  # type: ignore
            padding=module.padding,  # type: ignore
            dilation=module.dilation,  # type: ignore
            groups=module.groups,
            device=device,
            dtype=dtype,
        )

        return LoRA(module, lora_module)

    @classmethod
    def _from_embedding(cls, module: nn.Embedding, rank: int) -> LoRA[nn.Embedding]:
        num_embeddings, embedding_dim = module.weight.shape
        device = module.weight.device
        dtype = module.weight.dtype
        lora_module = EmbeddingLoRAModule(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            rank=rank,
            device=device,
            dtype=dtype,
        )
        return LoRA(module, lora_module)

    @classmethod
    def _from_multihead_attention(
        cls, module: nn.MultiheadAttention, rank: int, no_lora_q: bool
    ) -> MultiheadAttentionLoRA:
        device = module.out_proj.weight.device
        dtype = module.out_proj.weight.dtype
        lora_module = MultiheadAttentionLoRAModule(
            embed_dim=module.embed_dim,
            num_heads=module.num_heads,
            rank=rank,
            bias=False,  # TODO: support bias
            kdim=module.kdim,
            vdim=module.vdim,
            device=device,
            dtype=dtype,
            no_lora_q=no_lora_q
        )
        return MultiheadAttentionLoRA(module, lora_module)

    @overload
    @classmethod
    def from_module(
        cls,
        module: ModuleType,
        rank: int,
        enabled: bool = True,
        is_root: Literal[True] = True,
        no_lora_q:bool=False,
        lora_ff: bool=False
    ) -> LoRA[ModuleType]:
        ...

    @overload
    @classmethod
    def from_module(
        cls,
        module: ModuleType,
        rank: int,
        enabled: bool = True,
        is_root: Literal[False] = False,
        no_lora_q:bool=False,
        lora_ff: bool=False
    ) -> Union[LoRA[ModuleType], ModuleType]:
        ...

    @classmethod
    def from_module(
        cls, module: ModuleType, rank: int, enabled: bool = True, is_root: bool = True, no_lora_q:bool=False, lora_ff: bool=False
    ):
        if isinstance(module, nn.Linear) and lora_ff:
            return LoRA._from_linear(module, rank)  # type: ignore
        # elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        #     return LoRA._from_conv(module, rank)  # type: ignore
        # elif isinstance(module, nn.Embedding):
        #     return LoRA._from_embedding(module, rank)
        if isinstance(module, nn.MultiheadAttention):
            return LoRA._from_multihead_attention(module, rank, no_lora_q)  # type: ignore

        for name, child in module.named_children():
            child = cast(ModuleType, child)
            module._modules[name] = cls.from_module(
                child, rank, enabled=enabled, is_root=False, no_lora_q=no_lora_q, lora_ff=lora_ff
            )

        if is_root:
            return LoRA(module, None, enabled=enabled)
        else:
            return module

    @property
    def weight(self) -> Tensor:
        if not hasattr(self.module, "weight"):
            raise AttributeError("Module has no attribute 'weight'")

        if self.enabled and self.lora_module is not None:
            assert hasattr(self.lora_module, "weight")
            return self.module.weight + self.lora_module.weight
        else:
            return self.module.weight

    @property
    def bias(self) -> Optional[Tensor]:
        if not hasattr(self.module, "bias"):
            return None

        if (
            self.enabled
            and self.lora_module is not None
            and hasattr(self.lora_module, "bias")
        ):
            return self.module.bias + self.lora_module.bias
        else:
            return self.module.bias


class MultiheadAttentionLoRA(LoRA[nn.MultiheadAttention]):
    """
    NOTE: MultiheadAttention doesn't quite fit the "sidecar" pattern, like essentially
    every other module does.  Unlike other modules, which typically have a single
    'weight' parameter that is modified by LoRA, MultiheadAttention has multiple
    parameters that are modified by LoRA, and those parameters interact in non-trivial
    ways (via attention) within the module itself.

    For that reason, we emulate all of the necessary properties of MultiheadAttention,
    and reuse the 'forward' method from MultiheadAttention.  This allows us to
    dynamically compute the LoRA-adjusted parameters without rewriting *all* of the
    logic from 'MultiheadAttention.forward'.
    """

    def __init__(
        self,
        module: nn.MultiheadAttention,
        lora_module: Optional[MultiheadAttentionLoRAModule],
        enabled: bool = True,
    ):
        super().__init__(module, lora_module, enabled=enabled)
        self.module = cast(nn.MultiheadAttention, self.module)
        self.lora_module = cast(
            Optional[MultiheadAttentionLoRAModule], self.lora_module
        )

    def forward(  # type: ignore
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if (not self.enabled) or self.lora_module is None:
            return self.module.forward(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        return nn.MultiheadAttention.forward(
            cast(nn.MultiheadAttention, self),
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            # average_attn_weights=average_attn_weights,
            # is_causal=is_causal,
        )

    def merge_masks(
        self,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        query: Tensor,
    ) -> Tuple[Optional[Tensor], Optional[int]]:
        return nn.MultiheadAttention.merge_masks(
            cast(nn.MultiheadAttention, self),
            attn_mask,
            key_padding_mask,
            query,
        )

    @property
    def embed_dim(self) -> int:
        return self.module.embed_dim

    @property
    def num_heads(self) -> int:
        return self.module.num_heads

    @property
    def dropout(self) -> float:
        return self.module.dropout

    @property
    def add_zero_attn(self) -> bool:
        return self.module.add_zero_attn

    @property
    def batch_first(self) -> bool:
        return self.module.batch_first

    @property
    def _qkv_same_embed_dim(self) -> bool:
        return self.module._qkv_same_embed_dim

    @property
    def bias_k(self) -> Optional[Tensor]:
        if self.module.bias_k is None:
            return None
        return self.module.bias_k.data.detach()

    @property
    def bias_v(self) -> Optional[Tensor]:
        if self.module.bias_v is None:
            return None
        return self.module.bias_v.data.detach()

    @property
    def in_proj_weight(self) -> Tensor:
        in_proj_weight = self.module.in_proj_weight
        if in_proj_weight is None:
            return None

        weight = in_proj_weight.data.detach()
        if (
            self.enabled
            and self.lora_module is not None
            and self.lora_module.in_proj_weight is not None
        ):
            return weight + self.lora_module.in_proj_weight
        else:
            return weight

    @property
    def in_proj_bias(self) -> Tensor:
        bias = self.module.in_proj_bias
        if bias is None:
            return None
        else:
            return bias.data.detach()
        # TODO: Add support for 'in_proj_bias' in MultiheadAttentionLoRAModule

    @property
    def q_proj_weight(self) -> Optional[Tensor]:
        weight = self.module.q_proj_weight.data.detach()
        if self.enabled and weight is not None and self.lora_module is not None:
            return weight + self.lora_module.q_proj_weight
        else:
            return weight

    @property
    def k_proj_weight(self) -> Optional[Tensor]:
        weight = self.module.k_proj_weight.data.detach()
        if self.enabled and (weight is not None) and (self.lora_module is not None):
            return weight + self.lora_module.k_proj_weight
        else:
            return weight

    @property
    def v_proj_weight(self) -> Optional[Tensor]:
        weight = self.module.v_proj_weight.data.detach()
        if self.enabled and (weight is not None) and (self.lora_module is not None):
            return weight + self.lora_module.v_proj_weight
        else:
            return weight

    @property
    def out_proj(self) -> OutProj:
        weight = self.module.out_proj.weight.data.detach()
        bias = self.module.out_proj.bias
        if self.enabled and self.lora_module is not None:
            lora_out_proj = cast(OutProj, self.lora_module.out_proj)
            weight = weight + lora_out_proj.weight
            if (bias is not None) and (lora_out_proj.bias is not None):
                # Mypy complains about a type mismatch here (Tensor vs. Parameter)
                # but Parameter is just a subclass of Tensor, so this is fine.
                bias = bias + lora_out_proj.bias  # type: ignore

        return OutProj(weight, bias)


class OutProj(NamedTuple):
    weight: Tensor
    bias: Optional[Tensor]


def enable_lora(module: Union[ModuleType, LoRA[ModuleType]]) -> None:
    for child in module.children():
        enable_lora(child)
    if isinstance(module, LoRA):
        module.enabled = True


def disable_lora(module: Union[ModuleType, LoRA[ModuleType]]) -> None:
    for child in module.children():
        disable_lora(child)
    if isinstance(module, LoRA):
        module.enabled = False


def merge_lora(
    module: Union[ModuleType, LoRA[ModuleType]], inplace: bool = False
) -> ModuleType:
    out = module if inplace else deepcopy(module)
    for name, child in out.named_children():
        out._modules[name] = merge_lora(child, inplace=inplace)

    if isinstance(out, LoRA):
        if out.lora_module is None:
            return out.module
        else:
            return out.lora_module.merge(out.module, inplace=inplace)
    else:
        return out


def remove_lora(
    module: Union[ModuleType, LoRA[ModuleType]], inplace: bool = False
) -> ModuleType:
    """Remove all LoRA modules from the model."""
    out = module if inplace else deepcopy(module)

    for name, child in out.named_children():
        out._modules[name] = remove_lora(child, inplace=inplace)

    if isinstance(out, LoRA):
        return out.module
    else:
        return out

from __future__ import annotations

from copy import deepcopy
from math import sqrt
from typing import Optional, Tuple, TypeVar, Union

import torch
from torch import Tensor, nn

from lora_pytorch.modules.base import BaseLoRAModule

Conv = Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]
ConvType = TypeVar("ConvType", nn.Conv1d, nn.Conv2d, nn.Conv3d)


class _ConvLoRA(BaseLoRAModule[ConvType]):
    def __init__(
        self,
        in_conv: ConvType,
        out_conv: ConvType,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_conv: ConvType = in_conv
        self.out_conv: ConvType = out_conv
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_conv(x)
        x = self.dropout(x)
        return self.out_conv(x)

    @torch.no_grad()
    def merge(self, module: ConvType, inplace: bool = False) -> ConvType:  # type: ignore
        # einstein notation:
        # - i: in channels
        # - o: out channels
        # - r: rank
        weight = torch.einsum(
            "o r ..., r i ... -> o i ...",
            self.out_conv.weight.data,
            self.in_conv.weight.data,
        )
        # TODO: Detailed error message
        assert module.weight.data.shape == weight.shape

        if inplace:
            module.weight.data += weight
            return module

        out_module = deepcopy(module)
        out_module.weight.data += weight
        if out_module.bias is None:
            out_module.bias = module.bias
        elif self.out_conv.bias is not None:
            out_module.bias.data += out_module.bias.data

        return out_module


class Conv1dLoRAModule(_ConvLoRA[nn.Conv1d]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = False,
        alpha: float = 1.0,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.rank = rank
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        # NOTE: Only include bias in 'out_conv', since it's normally applied
        # after the convolution.  When merging LoRA weights, we can't have a
        # bias term before the out convolution.
        in_conv = nn.Conv1d(
            in_channels,
            rank,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            device=device,
            dtype=dtype,
        )
        out_conv = nn.Conv1d(
            rank, out_channels, kernel_size=1, bias=bias, device=device, dtype=dtype
        )
        super().__init__(in_conv=in_conv, out_conv=out_conv, dropout=dropout)

        # NOTE: The original LoRA paper recommends multiplying the output of 'in_proj'
        # by (alpha / rank).  This adds more computation to the forward pass, and it's
        # mathematically equivalent to scaling 'in_proj' by (alpha / rank) ahead of
        # time.  I have chosen the second option for simplicity.
        #
        # Normally, the weights of 'in_proj' are initialized with 'kaiming_uniform_',
        # and 'a=sqrt(5)'.  Since we're scaling the weights by (alpha / rank), we
        # should scale 'a' by the same amount.
        nn.init.kaiming_uniform_(self.in_conv.weight, a=(sqrt(5) * alpha / rank))
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, "
            f"rank={self.rank}), kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, bias={self.bias})"
        )


class Conv2dLoRAModule(_ConvLoRA[nn.Conv2d]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = False,
        alpha: float = 1.0,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # NOTE: Only include bias in 'out_conv', since it's normally applied
        # after the convolution.  When merging LoRA weights, we can't have a
        # bias term before the out convolution.
        in_conv = nn.Conv2d(
            in_channels,
            rank,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            device=device,
            dtype=dtype,
        )
        out_conv = nn.Conv2d(
            rank, out_channels, kernel_size=1, bias=bias, device=device, dtype=dtype
        )
        super().__init__(in_conv=in_conv, out_conv=out_conv, dropout=dropout)

        # NOTE: The original LoRA paper recommends multiplying the output of 'in_proj'
        # by (alpha / rank).  This adds more computation to the forward pass, and it's
        # mathematically equivalent to scaling 'in_proj' by (alpha / rank) ahead of
        # time.  I have chosen the second option for simplicity.
        #
        # Normally, the weights of 'in_proj' are initialized with 'kaiming_uniform_',
        # and 'a=sqrt(5)'.  Since we're scaling the weights by (alpha / rank), we
        # should scale 'a' by the same amount.
        nn.init.kaiming_uniform_(self.in_conv.weight, a=(sqrt(5) * alpha / rank))
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)


class Conv3dLoRAModule(_ConvLoRA[nn.Conv3d]):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = False,
        alpha: float = 1.0,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # NOTE: Only include bias in 'out_conv', since it's normally applied
        # after the convolution.  When merging LoRA weights, we can't have a
        # bias term before the out convolution.
        in_conv = nn.Conv3d(
            in_channels,
            rank,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
            device=device,
            dtype=dtype,
        )
        out_conv = nn.Conv3d(
            rank, out_channels, kernel_size=1, bias=bias, device=device, dtype=dtype
        )
        super().__init__(in_conv=in_conv, out_conv=out_conv, dropout=dropout)

        # NOTE: The original LoRA paper recommends multiplying the output of 'in_proj'
        # by (alpha / rank).  This adds more computation to the forward pass, and it's
        # mathematically equivalent to scaling 'in_proj' by (alpha / rank) ahead of
        # time.  I have chosen the second option for simplicity.
        #
        # Normally, the weights of 'in_proj' are initialized with 'kaiming_uniform_',
        # and 'a=sqrt(5)'.  Since we're scaling the weights by (alpha / rank), we
        # should scale 'a' by the same amount.
        nn.init.kaiming_uniform_(self.in_conv.weight, a=(sqrt(5) * alpha / rank))
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

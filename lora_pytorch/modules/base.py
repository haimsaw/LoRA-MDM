from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar

from torch import Tensor, nn

ModuleType = TypeVar("ModuleType", bound=nn.Module)


class BaseLoRAModule(nn.Module, Generic[ModuleType]):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...

    @abstractmethod
    def merge(self, module: ModuleType, inplace: bool = False) -> ModuleType:
        ...

from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal, overload

from PIL.Image import Image


class Direction(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"


class Device(ABC):
    @overload
    def capture_screenshot(self, format: Literal["PIL"]) -> Image: ...

    @overload
    def capture_screenshot(self, format: Literal["B64"]) -> str: ...

    @abstractmethod
    def capture_screenshot(self, format: Literal["PIL", "B64"]) -> Image | str: ...

    @property
    @abstractmethod
    def size(self) -> tuple[int, int]: ...

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        return False

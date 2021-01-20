from typing import Any, Dict, Iterator

import numpy as np
from typing_extensions import Protocol


class FrameBuffer(Protocol):
    _header: Any
    array: np.ndarray

    def get_header(self) -> Dict: ...

    def demosaic(self) -> np.ndarray: ...


class AbstractRawSource:
    @staticmethod
    def get_implementation():
        try:
            from .camera_hq import CameraRawSource
            return CameraRawSource
        except (ImportError, OSError):
            pass
        try:
            from .camera_dummy import DummyRawSource
            return DummyRawSource
        except (ImportError, OSError):
            pass
        return None

    def __init__(self) -> None:
        super().__init__()

    def raw_captures(self) -> Iterator[FrameBuffer]:
        yield from []

    def get_ccm(self, temperature: float) -> np.ndarray:
        return np.eye(3)

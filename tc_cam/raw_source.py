import json
from pathlib import Path
from typing import Any, Dict, Iterator, Union, TextIO

import numpy as np
from scipy.interpolate import interp1d
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


class CalibrationData:

    def __init__(self, file: Union[Path, TextIO]) -> None:
        super().__init__()
        if isinstance(file, Path):
            file = file.open("rt")
        self.data = json.load(file)
        self.ccm_interpolation = None
        self.setup_ccm_interpolation()

    def get_ccms(self):
        try:
            jccms = self.data["rpi.ccm"]["ccms"]
            jccms = sorted(jccms, key=lambda item: item["ct"])
            ccm_dict = {item["ct"]: np.array(item["ccm"]) for item in jccms}
            if len(ccm_dict) == 1:
                t1 = list(ccm_dict.keys())[0]
                ccm_dict[t1 * 2] = ccm_dict[t1].copy()
            return ccm_dict
        except KeyError:
            return None

    def setup_ccm_interpolation(self):
        table = self.get_ccms()
        if len(table):
            xp = np.array(list(table.keys()))
            yp = np.array(list(table.values()))
            self.ccm_interpolation = interp1d(xp, yp.T, assume_sorted=True, fill_value="extrapolate", copy=False)

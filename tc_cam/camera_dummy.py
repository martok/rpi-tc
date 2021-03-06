from pathlib import Path
from typing import Dict, Iterator

import numpy as np

from tc_cam.raw_source import FrameBuffer, AbstractRawSource, CalibrationData
from tc_cam.bayer import BayerConvert


class TCDummyBayerArray(FrameBuffer, BayerConvert):

    def __init__(self, file, header: Dict) -> None:
        self.array = np.load(file)
        self.header = header.copy()
        self.header["width"] = self.array.shape[1]
        self.header["height"] = self.array.shape[0]
        self.header["padding_width"] = 0
        self.header["padding_height"] = 0

    def get_header(self) -> Dict:
        return self.header

    def demosaic(self) -> np.ndarray:
        return self.demosaic_array(self.array, self.header["bayer_order"], self.header["transform"])


class DummyCamera:
    pass


class DummyRawSource(AbstractRawSource):

    def __init__(self) -> None:
        super().__init__()
        print("Opening File Source")
        h = {
            "transform": 3,
            "bayer_order": 2,
            "bayer_format": 4
        }
        self.camera = DummyCamera()
        self.buffer = TCDummyBayerArray("data/test/bal.raw.npy", h)
        self.config = CalibrationData(Path(".") / "data" / "imx477.json")

    def raw_captures(self) -> Iterator[FrameBuffer]:
        for _ in range(10000):
            yield self.buffer

    def get_ccm(self, temperature: float) -> np.ndarray:
        if self.config.ccm_interpolation is not None:
            v = self.config.ccm_interpolation(temperature)
            return v.reshape((3,3))
        return None
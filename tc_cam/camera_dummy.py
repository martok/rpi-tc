from typing import Dict

import cv2
import numpy as np

from tc_cam import FrameBuffer
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

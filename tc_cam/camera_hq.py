from typing import Dict

from picamerax import PiCamera
from picamerax.array import PiBayerArray

from tc_cam import FrameBuffer
from tc_cam.bayer import BayerConvert


class TCCamera(PiCamera):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution = self.MAX_RESOLUTION
        self.meter_mode = 'backlit'
        self.exposure_mode = 'auto'
        self.awb_mode = 'greyworld'
        # self.sharpness = -100


class TCBayerArray(FrameBuffer, PiBayerArray, BayerConvert):

    def __init__(self, camera):
        super().__init__()
        # always need output_dims=2 for demosaic
        PiBayerArray.__init__(self, camera, output_dims=2)

    def reset(self):
        self.seek(0)
        self.truncate()

    def demosaic(self):
        return self.demosaic_array(self.array, self._header.bayer_order, self._header.transform)

    def get_header(self) -> Dict:
        s = self._header
        return {field_name: getattr(s, field_name) for field_name, field_type in s._fields_}


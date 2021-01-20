import time
from typing import Dict, Iterator

from picamerax import PiCamera
from picamerax.array import PiBayerArray

from tc_cam.raw_source import FrameBuffer, AbstractRawSource
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


class CameraRawSource(AbstractRawSource):

    def __init__(self) -> None:
        super().__init__()

        print("Opening Camera")
        self.camera = TCCamera()
        self.camera.resolution = (160, 120)
        self.buffer = TCBayerArray(self.camera)
        time.sleep(1)

    def raw_captures(self) -> Iterator[FrameBuffer]:
        for _ in self.camera.capture_continuous(self.buffer, format="jpeg", bayer=True, burst=True):
            self.buffer.reset()
            yield self.buffer

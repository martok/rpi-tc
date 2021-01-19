import time
from typing import Iterator, Optional

from tc_cam import FrameBuffer

try:
    from tc_cam.camera_hq import TCCamera, TCBayerArray

    CAMERA_MODE = "real"
except (ImportError, OSError):
    CAMERA_MODE = "file"


class AbstractCaptureCamera:
    def __init__(self) -> None:
        super().__init__()
        self.camera: Optional[TCCamera] = None

    def raw_captures(self) -> Iterator[FrameBuffer]:
        yield from []


if CAMERA_MODE == "real":
    class CaptureCamera(AbstractCaptureCamera):

        def __init__(self) -> None:
            super().__init__()

            print("Opening Camera")
            self.camera = TCCamera()
            self.camera.resolution = tuple(x // 8 for x in self.camera.resolution)
            self.buffer = TCBayerArray(self.camera)
            time.sleep(1)

        def raw_captures(self) -> Iterator[FrameBuffer]:
            for _ in self.camera.capture_continuous(self.buffer, format="jpeg", bayer=True, burst=True):
                self.buffer.reset()
                yield self.buffer

else:
    from tc_cam.camera_dummy import TCDummyBayerArray


    class DummyCamera:
        pass


    class CaptureCamera(AbstractCaptureCamera):

        def __init__(self) -> None:
            super().__init__()
            print("Opening File Source")
            h = {
                "transform": 3,
                "bayer_order": 2,
                "bayer_format": 4
            }
            self.camera = DummyCamera()
            # self.buffer = TCDummyBayerArray("test\\RedR.raw.npy", h)
            self.buffer = TCDummyBayerArray("test\\bayered.raw.npy", h)

        def raw_captures(self) -> Iterator[FrameBuffer]:
            for _ in range(10000):
                yield self.buffer

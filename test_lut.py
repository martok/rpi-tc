import cv2
import numpy as np

from tc_cam import Stopwatch
from tc_cam.cvext import CVWindow
from tc_cam.lut import LUT3D

img = cv2.imread("test/16bit.tiff", cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
win = CVWindow("image")
reps = 8
img = np.tile(img, (reps, reps, 1))
win.display_image(img)
# cv2.waitKey(0)

def make_lut(bgr: np.ndarray) -> np.ndarray:
    ccm = np.array([7930, -3604, -224, -698, 6082, -1282, 612, -3186, 6676, 0, 0, 0])[:-3].reshape(
        (3, 3)) / 4096.0
    bgr = LUT3D.xfer_whitebalance(bgr, 0.5, 0.5)
    bgr = LUT3D.xfer_gamma(bgr, 2.2)
    bgr = LUT3D.xfer_contrast(bgr, 0.9, 0.5)
    bgr = LUT3D.xfer_ccm(bgr, ccm)
    return bgr

w = Stopwatch()
with w:
    lut = LUT3D()
    lut.setup(32, (0, 2 ** 16 - 1), np.uint8, make_lut)
print(f"created lut: {w.duration * 1000:.3f} ms")

with w:
    mapped = lut.apply(img)
print(f"applied lut on {img.shape}: {w.duration * 1000:.3f} ms")

win.display_image(mapped)
cv2.waitKey(0)

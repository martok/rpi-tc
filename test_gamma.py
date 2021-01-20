import cv2
import numpy as np

from tc_cam import Stopwatch
from tc_cam.cvext import CVWindow
from tc_cam.process import gamma_convert, ExposureLut
from tc_cam.analyze import histogram_calc, histogram_draw

img = cv2.imread("test/16bit.tiff", cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)

w = Stopwatch()
# for reps in [10, 8, 6, 4, 2, 1]:
for reps in [8,2, 1]:
    repeated = np.tile(img, (reps, reps, 1))
    with w:
        lut = ExposureLut(20000, 2 ** 16 - 1, np.uint8, gamma=1.6)
        convD = lut.apply(repeated)
        # convD = gamma_convert(repeated, 20000, 2 ** 16 - 1, 1.6, dtype=np.uint8, use_lut=False)
    tD = w.duration
    with w:
        convL = gamma_convert(repeated, 20000, 2 ** 16 - 1, 1.6, dtype=np.uint8, use_lut=True)
    tL = w.duration
    print(reps, repeated.shape[0], tD*1000, tL*1000)

win = CVWindow("image")
win.display_image(img)

cv2.imshow("a", np.hstack((convD, convL)))
histogram_draw(convD, histogram_calc(convD))
cv2.imshow("hist", convD)
cv2.waitKey()

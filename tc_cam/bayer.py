import cv2

import numpy as np

PiBayerArray_BAYER_OFFSETS = {
    0: ((0, 0), (1, 0), (0, 1), (1, 1)),
    1: ((1, 0), (0, 0), (1, 1), (0, 1)),
    2: ((1, 1), (0, 1), (1, 0), (0, 0)),
    3: ((0, 1), (1, 1), (0, 0), (1, 0)),
}

# https://github.com/6by9/userland/blob/rawcam/host_applications/linux/apps/raspicam/raspiraw.c#L64
# https://www.mathworks.com/help/images/ref/demosaic.html#bu45ckm-3
# https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_bayer
BAYER2CV = {
    0: cv2.COLOR_BayerRG2BGR,  # BGGR
    1: cv2.COLOR_BayerGR2BGR,  # GBRG
    2: cv2.COLOR_BayerGB2BGR,  # GRBG
    3: cv2.COLOR_BayerBG2BGR,  # RGGB
}

BAYER_FORMAT = {
    0: 'VC_IMAGE_BAYER_RAW6',
    1: 'VC_IMAGE_BAYER_RAW7',
    2: 'VC_IMAGE_BAYER_RAW8',
    3: 'VC_IMAGE_BAYER_RAW10',
    4: 'VC_IMAGE_BAYER_RAW12',
    5: 'VC_IMAGE_BAYER_RAW14',
    6: 'VC_IMAGE_BAYER_RAW16',
}


class BayerConvert:
    @staticmethod
    def demosaic_array(array: np.ndarray, bayer_order: int, transform: int):
        hflip = transform & (1 << 0)
        vflip = transform & (1 << 1)
        transpose = transform & (1 << 2)
        assert transpose == 0, "Transpose data unimplemented"

        # from sensor manual:
        # htrans | vtrans | readout order | pixel image is | bayer_order
        #     0         0            RGGB   upright        |           3
        #     1         0            GRBG   hflipped       |           2
        #     0         1            GBRG   vflipped       |           1
        #     1         1            BGGR   180°           |           0

        # graphic is wrong, pixel image is always additionally hflipped
        # or: header(GRBG)^header(3) = GBRG, but needs BGGR (additional hflip) to decode correctly

        return cv2.demosaicing(array, BAYER2CV[bayer_order ^ (transform & 0b11) ^ 1], None, 3)


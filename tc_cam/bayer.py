from typing import Tuple

import cv2
from numpy.lib.stride_tricks import as_strided

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

class Demosaic:

    def __init__(self) -> None:
        super().__init__()
        self.initialized = False

    def init(self, bayer_order, shape):
        # Construct representation of the bayer pattern
        bayer = np.zeros(shape, dtype=np.uint8)
        (
            (ry, rx), (gy, gx), (Gy, Gx), (by, bx)
        ) = PiBayerArray_BAYER_OFFSETS[bayer_order]
        bayer[ry::2, rx::2, 0] = 1  # Red
        bayer[gy::2, gx::2, 1] = 1  # Green
        bayer[Gy::2, Gx::2, 1] = 1  # Green
        bayer[by::2, bx::2, 2] = 1  # Blue

        # Allocate output array with same shape as data and set up some
        # constants to represent the weighted average window
        window = (3, 3)
        borders = (window[0] - 1, window[1] - 1)
        border = (borders[0] // 2, borders[1] // 2)
        bayer = np.pad(bayer, [
            (border[0], border[0]),
            (border[1], border[1]),
            (0, 0),
        ], 'constant')

        self.window = window
        self.border = border
        self.borders = borders
        self.bayer = bayer
        self.bsums = self.sum_over_planes(shape, bayer, bayer.dtype)
        self.initialized = True

    def sum_over_planes(self, shape: Tuple, padded_data: np.ndarray, dtype):
        sums = np.ones(shape, dtype=dtype)
        viewshape = (padded_data.shape[0] - self.borders[0], padded_data.shape[1] - self.borders[1]) + self.window
        for plane in range(3):
            b = padded_data[..., plane]
            bview = as_strided(b, shape=viewshape, strides=b.strides * 2)
            bsum = np.einsum('ijkl->ij', bview)
            sums[..., plane] = bsum
        return sums

    def demosaic(self, array_3d: np.ndarray) -> np.ndarray:
        # Pad out the data and the bayer pattern
        rgb = np.pad(array_3d, [
            (self.border[0], self.border[0]),
            (self.border[1], self.border[1]),
            (0, 0),
        ], 'constant')
        # For each plane in the RGB data, construct a view over the plane
        # of 3x3 matrices. Then do the same for the bayer array and use
        # Einstein summation to get the weighted average
        rgbsums = self.sum_over_planes(array_3d.shape, rgb, array_3d.dtype)
        rgbsums = rgbsums // self.bsums
        return rgbsums


def demosaic(array_3d, bayer_order):
    # Construct representation of the bayer pattern
    bayer = np.zeros(array_3d.shape, dtype=np.uint8)
    (
        (ry, rx), (gy, gx), (Gy, Gx), (by, bx)
    ) = PiBayerArray_BAYER_OFFSETS[bayer_order]
    # IMX447:  2: ((1, 1), (0, 1), (1, 0), (0, 0)),
    bayer[ry::2, rx::2, 0] = 1  # Red
    bayer[gy::2, gx::2, 1] = 1  # Green
    bayer[Gy::2, Gx::2, 1] = 1  # Green
    bayer[by::2, bx::2, 2] = 1  # Blue
    # Allocate output array with same shape as data and set up some
    # constants to represent the weighted average window
    window = (3, 3)
    borders = (window[0] - 1, window[1] - 1)
    border = (borders[0] // 2, borders[1] // 2)
    # Pad out the data and the bayer pattern (np.pad is faster but
    # unavailable on the version of numpy shipped with Raspbian at the
    # time of writing)
    rgb = np.zeros((
        array_3d.shape[0] + borders[0],
        array_3d.shape[1] + borders[1],
        array_3d.shape[2]), dtype=array_3d.dtype)
    rgb[
    border[0]:rgb.shape[0] - border[0],
    border[1]:rgb.shape[1] - border[1],
    :] = array_3d
    bayer_pad = np.zeros((
        array_3d.shape[0] + borders[0],
        array_3d.shape[1] + borders[1],
        array_3d.shape[2]), dtype=bayer.dtype)
    bayer_pad[
    border[0]:bayer_pad.shape[0] - border[0],
    border[1]:bayer_pad.shape[1] - border[1],
    :] = bayer
    bayer = bayer_pad
    # For each plane in the RGB data, construct a view over the plane
    # of 3x3 matrices. Then do the same for the bayer array and use
    # Einstein summation to get the weighted average
    _demo = np.empty(array_3d.shape, dtype=array_3d.dtype)
    for plane in range(3):
        p = rgb[..., plane]
        b = bayer[..., plane]
        pview = as_strided(p, shape=(
                                        p.shape[0] - borders[0],
                                        p.shape[1] - borders[1]) + window, strides=p.strides * 2)
        bview = as_strided(b, shape=(
                                        b.shape[0] - borders[0],
                                        b.shape[1] - borders[1]) + window, strides=b.strides * 2)
        psum = np.einsum('ijkl->ij', pview)
        bsum = np.einsum('ijkl->ij', bview)
        _demo[..., plane] = psum // bsum
    return _demo


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


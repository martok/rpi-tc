import cv2

import numpy as np

from tc_cam.lut import LUT3D


def calc_black_level(image: np.ndarray, left=128, top=128, right=128, bottom=128):
    a = np.concatenate((image[:left, :, :].flat, image[-right:, :, :].flat,
                        image[:, :top, :].flat, image[:, -bottom:, :].flat))
    return np.mean(a) + np.std(a) * 2


def extract_region(image: np.ndarray, region) -> np.ndarray:
    h, w, d = image.shape
    t = int(h * region[0])
    l = int(w * region[1])
    b = int(h * region[2])
    r = int(w * region[3])
    roi = image[t:b, l:r, :]
    return roi


class ExposureLut(LUT3D):

    def __init__(self, blacklevel: int, whitelevel: int, dst_dtype, *,
                 gamma: float = 1.0, gain_b: float = 1.0, gain_r: float = 1.0,
                 use_matrix: bool = False,
                 brightess: float = 0.5, contrast: float = 1.0):
        super().__init__()
        ccm = np.array([7930, -3604, -224, -698, 6082, -1282, 612, -3186, 6676, 0, 0, 0])[:-3].reshape(
            (3, 3)) / 4096.0

        def make_lut(bgr: np.ndarray) -> np.ndarray:
            # follow the order of https://www.strollswithmydog.com/open-raspberry-pi-high-quality-camera-raw/
            # WB, project to sRGB, apply gamma, apply contrast
            bgr = LUT3D.xfer_whitebalance(bgr, gain_b, gain_r)
            if use_matrix:
                bgr = LUT3D.xfer_ccm(bgr, ccm.T)
            bgr = LUT3D.xfer_gamma(bgr, 1 / gamma)
            bgr = LUT3D.xfer_contrast(bgr, contrast, brightess)
            return bgr

        self.setup(32, (blacklevel, whitelevel), dst_dtype, make_lut)


def histogram_calc(image: np.ndarray, bins=256, value_range=None, mask=None):
    if value_range is None:
        value_range = [np.iinfo(image.dtype).min, np.iinfo(image.dtype).max + 1]
    bh = cv2.calcHist([image], [0], mask, [bins], value_range, accumulate=False).flatten()
    gh = cv2.calcHist([image], [1], mask, [bins], value_range, accumulate=False).flatten()
    rh = cv2.calcHist([image], [2], mask, [bins], value_range, accumulate=False).flatten()
    return np.array((bh, gh, rh))


def histogram_draw(image: np.ndarray, histogram):
    cm = np.iinfo(image.dtype).max
    COLORS = ((cm, 0, 0), (0, cm, 0), (0, 0, cm))
    hist_h, hist_w, _ = image.shape
    hist_size = histogram.shape[1]
    hist_h -= 5
    bin_w = int(round(hist_w / hist_size))
    cv2.normalize(histogram, histogram, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    for i in range(1, hist_size):
        for c in range(3):
            ch = histogram[c, :]
            cv2.line(image, (bin_w * (i - 1), hist_h - int(round(ch[i - 1]))),
                     (bin_w * (i), hist_h - int(round(ch[i]))),
                     COLORS[c], thickness=2)
    cv2.rectangle(image, (0, 0), (bin_w * hist_size, hist_h), color=(cm, cm, cm), thickness=1)

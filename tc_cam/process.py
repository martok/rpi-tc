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


def gamma_convert(image: np.ndarray, blacklevel: int, whitelevel: int, gamma: float, dtype=None, *,
                  use_lut=False) -> np.ndarray:
    if dtype is None:
        dtype = image.dtype
    invGamma = 1.0 / gamma
    input_dynrange = whitelevel - blacklevel
    output_range = np.iinfo(dtype).max
    if use_lut:
        # vectorized lookup table construction
        input_range = np.arange(0, whitelevel + 1)
        lut = output_range * (np.clip((input_range - blacklevel) / input_dynrange, 0, None) ** invGamma)
        lut = lut.astype(dtype)
        return np.take(lut, image, mode="wrap").astype(dtype)
    else:
        dynamic = np.clip(image - float(blacklevel), 0, None) / input_dynrange
        rc = dynamic ** invGamma
        return (rc * output_range).astype(np.uint8)


class IndependentExposureLut:

    def __init__(self, blacklevel: int, whitelevel: int, dst_dtype, *,
                 gamma: float = 1.0, gain_b: float = 1.0, gain_r: float = 1.0,
                 brightess: float = 0.5, contrast: float = 1.0):
        invGamma = 1.0 / gamma
        input_dynrange = whitelevel - blacklevel
        output_range = np.iinfo(dst_dtype).max
        # vectorized lookup table construction
        # 1. start with all valid input values
        input_range = np.arange(0, whitelevel + 1)
        # 2. convert to float and apply BR white balance
        lutbgr = np.array([
            np.clip((input_range - blacklevel) / input_dynrange, 0, None) * gain_b,
            np.clip((input_range - blacklevel) / input_dynrange, 0, None),
            np.clip((input_range - blacklevel) / input_dynrange, 0, None) * gain_r,
        ])
        # 4. apply gamma
        lutbgr = lutbgr ** invGamma
        # 5. apply contrast + brightness
        lutbgr = (lutbgr - 0.5) * contrast + brightess
        # 5. convert to target uint
        self.lut_wb = (np.clip(lutbgr, 0, 1) * output_range).astype(dst_dtype)
        self.ccm = np.array([7930, -3604, -224, -698, 6082, -1282, 612, -3186, 6676, 0, 0, 0])[:-3].reshape(
            (3, 3)) / 4096.0

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.merge(tuple(np.take(self.lut_wb[c], image[..., c], mode="clip") for c in range(3)))

    def apply_ccm(self, image: np.ndarray, ccm: np.ndarray):
        img2 = image.reshape((-1, 3))
        output = np.matmul(img2, ccm.T)
        return output.reshape(image.shape).astype(image.dtype)


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

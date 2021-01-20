from typing import Callable, Tuple, Optional

import numpy as np
import pyximport

pyximport.install(setup_args={"script_args": ["--verbose"]},
                  build_in_temp=False)

from .lut_native import lut_apply

ColorTransferFn = Callable[[np.ndarray], np.ndarray]


class LUT3D:

    def __init__(self) -> None:
        self.grid: int = 0
        self.binsize: float = 0
        self.cube: Optional[np.ndarray] = None

        self.setup(2, (0, 2 ** 8 - 1), np.uint8, lambda bgr: bgr)

    def setup(self, grid: int, irange: Tuple[int, int], otype, transfer: ColorTransferFn):
        blackpoint, whitepoint = irange
        idynrange = whitepoint - blackpoint
        odynrange = np.iinfo(otype).max

        def vtrans(arr):
            return np.array(
                [transfer(row) for row in arr])  # vtrans = np.vectorize(transfer, signature="(m)->(m)") is slower

        # a grid=5 over max = 10 has values with binsize 10/4 = 2.5:
        # 0. ,  2.5,  5. ,  7.5, 10.
        self.grid = grid
        self.binsize = whitepoint / (grid - 1)

        # sample transfer function
        # create indices for N x N x N x 3 Cube, but in sequential shape
        indexes = np.array(list(np.ndindex((grid, grid, grid))))

        # locate input for each voxel
        c_itype = indexes * self.binsize
        # rescale to actual dynamic range for mapping function
        c_dyn = np.clip((c_itype - blackpoint) / idynrange, 0, None)
        # apply map
        c_out = transfer(c_dyn)
        # cast [0..1] range to int output and assign cube shape
        lut = np.clip(c_out * odynrange, 0, odynrange).round().astype(otype).reshape((grid, grid, grid, 3))

        # pad right for trilinear
        lut = np.pad(lut, [(0, 1), (0, 1), (0, 1), (0, 0)], mode="edge")
        self.cube = lut

    @staticmethod
    def xfer_identity(bgr: np.ndarray) -> np.ndarray:
        bgr = np.atleast_2d(bgr)
        return bgr

    @staticmethod
    def xfer_whitebalance(bgr: np.ndarray, gain_b: float, gain_r: float) -> np.ndarray:
        bgr = np.atleast_2d(bgr)
        return bgr * [gain_b, 1.0, gain_r]

    @staticmethod
    def xfer_gamma(bgr: np.ndarray, gamma: float) -> np.ndarray:
        bgr = np.atleast_2d(bgr)
        return bgr ** gamma

    @staticmethod
    def xfer_contrast(bgr: np.ndarray, contrast: float, brightess: float) -> np.ndarray:
        bgr = np.atleast_2d(bgr)
        return (bgr - 0.5) * contrast + brightess

    @staticmethod
    def xfer_ccm(bgr: np.ndarray, ccm: np.ndarray) -> np.ndarray:
        bgr = np.atleast_2d(bgr)
        return np.matmul(bgr, ccm)

    def apply(self, img: np.ndarray):
        flat = img.flatten()
        out = np.empty(flat.shape, dtype=self.cube.dtype)
        lut_apply(flat, self.cube, self.binsize, out)
        return out.reshape(img.shape)

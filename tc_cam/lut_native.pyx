# cython: language_level=3, boundscheck=False, wraparound=False
import cython

cimport numpy as np

ctypedef fused T_INPUT:
    np.uint16_t
    np.uint8_t

ctypedef np.uint8_t T_OUTPUT

cdef int index_4d(int dim, int r, int g, int b, int c):
    return c + 3 * (b + dim * (g + dim * r))

@cython.cdivision(True)
cpdef void lut_apply(T_INPUT[::1] pixels,
              T_OUTPUT[:, :, :, ::1] lut, float binsize,
              T_OUTPUT[::1] output) nogil:
    cdef:
        int pxcount = pixels.shape[0] // 3
        int px, plane
        T_INPUT r, g, b
        float rf, gf, bf
        int r_id, g_id, b_id
        float r_d, g_d, b_d
        T_OUTPUT[:] lutid000, lutid100, lutid010, lutid110, lutid001, lutid101, lutid011, lutid111
        float w000, w100, w010, w110, w001, w101, w011, w111
        double tmp

    for px in range(pxcount):
        # get pixels in lut-float
        rf = pixels[3 * px]     / binsize
        gf = pixels[3 * px + 1] / binsize
        bf = pixels[3 * px + 2] / binsize

        # integer bin
        r_id = int(rf)
        g_id = int(gf)
        b_id = int(bf)

        # fraction to neighbor
        r_d = rf - r_id
        g_d = gf - g_id
        b_d = bf - b_id

        # trilinear interpolation
        for plane in range(3):
            tmp = 0
            tmp += (1 - r_d) * (1 - g_d) * (1 - b_d) * lut[r_id, g_id, b_id,             plane]
            tmp += r_d * (1 - g_d) * (1 - b_d)       * lut[r_id + 1, g_id, b_id,         plane]
            tmp += (1 - r_d) * g_d * (1 - b_d)       * lut[r_id, g_id + 1, b_id,         plane]
            tmp += r_d * g_d * (1 - b_d)             * lut[r_id + 1, g_id + 1, b_id,     plane]
            tmp += (1 - r_d) * (1 - g_d) * b_d       * lut[r_id, g_id, b_id + 1,         plane]
            tmp += r_d * (1 - g_d) * b_d             * lut[r_id + 1, g_id, b_id + 1,     plane]
            tmp += (1 - r_d) * g_d * b_d             * lut[r_id, g_id + 1, b_id + 1,     plane]
            tmp += r_d * g_d * b_d                   * lut[r_id + 1, g_id + 1, b_id + 1, plane]
            output[3 * px + plane] = int(tmp)

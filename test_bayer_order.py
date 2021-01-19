import numpy as np
import cv2

# see https://github.com/6by9/userland/blob/rawcam/host_applications/linux/apps/raspicam/raspiraw.c#L64
bayer_order_enum = (
    # 	//Carefully ordered so that an hflip is ^1,
    #   //and a vflip is ^2.
    'BAYER_ORDER_BGGR',
    'BAYER_ORDER_GBRG',
    'BAYER_ORDER_GRBG',
    'BAYER_ORDER_RGGB')
BAYER_OFFSETS_0 = {
    0: ((0, 0), (1, 0), (0, 1), (1, 1)),
    1: ((1, 0), (0, 0), (1, 1), (0, 1)),
    2: ((1, 1), (0, 1), (1, 0), (0, 0)),
    3: ((0, 1), (1, 1), (0, 0), (1, 0)),
}
# see https://www.mathworks.com/help/images/ref/demosaic.html#bu45ckm-3
BAYER_OFFSETS = {
    0: ((1, 1), (0, 1), (1, 0), (0, 0)),
    1: ((1, 0), (0, 0), (1, 1), (0, 1)),
    2: ((0, 1), (1, 1), (0, 0), (1, 0)),
    3: ((0, 0), (1, 0), (0, 1), (1, 1)),
}
BAYER2CV = {
    0: cv2.COLOR_BayerRG2BGR,
    1: cv2.COLOR_BayerGR2BGR,
    2: cv2.COLOR_BayerGB2BGR,
    3: cv2.COLOR_BayerBG2BGR,
}
for bayer_order, bayer_name in enumerate(bayer_order_enum):
    print(bayer_name, ":")
    print("  hflip = ", bayer_order_enum[bayer_order ^ 1])
    print("  vflip = ", bayer_order_enum[bayer_order ^ 2])
    (
        (ry, rx), (gy, gx), (Gy, Gx), (by, bx)
    ) = BAYER_OFFSETS[bayer_order]
    ff = np.empty((4, 4), dtype=str)
    ff[ry::2, rx::2] = 'r'
    ff[gy::2, gx::2] = 'g'
    ff[Gy::2, Gx::2] = 'G'
    ff[by::2, bx::2] = 'b'
    print(ff)
    bayer = np.zeros((16, 16, 3), dtype=np.uint8)
    (
        (ry, rx), (gy, gx), (Gy, Gx), (by, bx)
    ) = BAYER_OFFSETS[bayer_order]
    bayer[ry::2, rx::2, 0] = 255  # Red
    bayer[gy::2, gx::2, 1] = 255  # Green
    bayer[Gy::2, Gx::2, 1] = 255  # Green
    bayer[by::2, bx::2, 2] = 255  # Blue

    if 1:
        upscaled = cv2.resize(bayer, (bayer.shape[0] * 16, bayer.shape[1] * 16), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(bayer_name[-4:], cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR))
    if 0:
        bgr = cv2.demosaicing(bayer, BAYER2CV[bayer_order])
        cv2.imshow(bayer_name[-4:, bgr])
cv2.waitKey()

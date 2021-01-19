from typing import Optional

import cv2
import numpy as np


def display_shadow_text(img, x, y, text, font=cv2.FONT_HERSHEY_PLAIN, font_scale=1.25, thickness=1, line_spacing=1.5):
    """
    Displays with a grey shadow at point x,y
    """
    text_color = (255, 255, 255)  # color as (B,G,R)
    text_shadow = (0, 0, 0)

    (w, h), _ = cv2.getTextSize(
        text="jM^",
        fontFace=font,
        fontScale=font_scale,
        thickness=thickness,
    )
    org = np.array([x, y], dtype=np.float)

    for line in text.splitlines():
        cv2.putText(img, line, tuple((org + [1, 1]).astype(int)), font, font_scale, text_shadow, thickness=thickness,
                    lineType=cv2.LINE_AA)
        cv2.putText(img, line, tuple(org.astype(int)), font, font_scale, text_color, thickness=thickness,
                    lineType=cv2.LINE_AA)
        org += [0, h * line_spacing]
    return img


def rect_empty(rect):
    return rect[0] == rect[2] and rect[1] == rect[3]


def rect_to_region(left, top, width, height, w=1, h=1):
    return [top / h, left / w, (top + height) / h, (left + width) / w]


def region_reparent(region, reference_region):
    # if region is defined relative to reference_frame, where is it globally?
    if reference_region is None:
        return region
    outer_h = reference_region[2] - reference_region[0]
    outer_w = reference_region[3] - reference_region[1]

    r2 = [
        reference_region[0] + outer_h * region[0],
        reference_region[1] + outer_w * region[1],
        reference_region[0] + outer_h * region[2],
        reference_region[1] + outer_w * region[3],
        ]
    return r2


class CVWindow:
    def __init__(self, caption: str) -> None:
        super().__init__()
        self.name = caption
        self.showing: Optional[np.ndarray] = None
        self.key_buffer = []

    def destroy_window(self):
        cv2.destroyWindow(self.name)

    def display_image(self, img: np.ndarray, reduction=2, overlay_fn=None, is_rgb=False):
        """
        Resize image and display using imshow. Mainly for debugging
        Resizing the image allows us to see the full frame on the monitor
        as cv2.imshow only allows zooming in.
        The reduction factor can be specified, but defaults to half size
        Text can also be displayed - in white at top of frame
        """
        reduction = np.clip(reduction, 1 / 8, 8)
        newx, newy = int(img.shape[1] / reduction), int(img.shape[0] / reduction)  # new size (w,h)
        newimg = cv2.resize(img, (newx, newy), interpolation=cv2.INTER_NEAREST)
        if is_rgb:
            newimg = cv2.cvtColor(newimg, cv2.COLOR_RGB2BGR)
        if callable(overlay_fn):
            overlay_fn(newimg)
        self.show(newimg)

    def show(self, img: np.ndarray):
        cv2.imshow(self.name, img)
        self.showing = img

    VK_RightArrow = 65363
    VK_LeftArrow = 65361
    VK_UpArrow = 65362
    VK_DownArrow = 65364
    VK_Escape = 27
    VK_Enter = 10
    VK_Home = 65360
    VK_End = 65367
    VK_PgUp = 65365
    VK_PgDn = 65366
    VK_Tab = 9

    def wait_key(self, timeout=None):
        key = 0xffff & cv2.waitKey(timeout)
        if key == 0xffff:
            return None
        return key

    def key_loop(self, time=10):
        key = self.wait_key(time)
        if key is not None:
            self.key_buffer.append(key)

    def get_key(self):
        self.key_loop()
        if self.key_buffer:
            return self.key_buffer.pop(0)
        return None

    def select_roi(self, from_center=True):
        assert self.showing is not None
        rect = cv2.selectROI(self.name, self.showing, showCrosshair=True, fromCenter=from_center)
        if rect_empty(rect):
            return None
        scaled = rect_to_region(*rect, self.showing.shape[1], self.showing.shape[0])
        return scaled

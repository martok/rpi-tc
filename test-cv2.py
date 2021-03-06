#!/usr/bin/env -S python3 -u
import time
from typing import Optional

import cv2
import numpy as np

from tc_cam import Stopwatch
from tc_cam.process import ExposureLut
from tc_cam.analyze import calc_black_level, histogram_calc, histogram_draw
from tc_cam.cvext import CVWindow, region_reparent, extract_region, display_shadow_text
from tc_cam.raw_source import AbstractRawSource


class CaptureApp:

    def __init__(self) -> None:
        self.draw_overlay = True
        self.draw_hist = False
        self.gamma = 2.2
        self.gain_r = 2.1 # 2.425  # 2.191
        self.gain_b = 2.1  # 2.325  # 2.753
        self.ccm_matrix = True
        self.temperature = 4500
        self.saturation = 0.85
        self.brightness = 0.450
        self.contrast = 1.750
        self.shutter = 50
        self.crop_region = None
        self.downsample = 16

        self.capture: Optional[AbstractRawSource] = None
        self.window = CVWindow("Telecine")

    def start(self):
        cls = AbstractRawSource.get_implementation()
        self.capture = cls()

    def camera_set(self):
        # setup for next capture
        if hasattr(self.capture, "camera"):
            self.capture.camera.shutter_speed = int(self.shutter * 1000)

    def main_loop(self):
        def process_keys():
            nonlocal rawimage, raw_demo, blacklevel, lut, output
            while True:
                key = self.window.get_key()
                if key is None:
                    break
                cey = chr(key)
                incdec = lambda d: d if cey < "_" else -d
                if cey == "o":
                    self.draw_overlay = not self.draw_overlay
                elif cey == "h":
                    self.draw_hist = not self.draw_hist
                elif cey == "+":
                    self.downsample = max(1, self.downsample // 2)
                elif cey == "-":
                    self.downsample = min(32, self.downsample * 2)
                elif cey in "gG":
                    self.gamma += incdec(0.05)
                    lut = None
                elif cey in "rR":
                    self.gain_r += incdec(0.05)
                    lut = None
                elif cey in "bB":
                    self.gain_b += incdec(0.05)
                    lut = None
                elif cey == "m":
                    self.ccm_matrix = not self.ccm_matrix
                    lut = None
                elif cey in "iI":
                    self.temperature = min(10000, max(2000, self.temperature + incdec(250)))
                    lut = None
                elif cey in "sS":
                    self.saturation = min(1.0, max(0.0, self.saturation + incdec(0.05)))
                    lut = None
                elif cey in "lL":
                    self.brightness += incdec(0.05)
                    lut = None
                elif cey in "kK":
                    self.contrast += incdec(0.05)
                    lut = None
                elif cey in "tT":
                    self.shutter *= 2 ** incdec(0.5)
                elif cey == "w":
                    print("Setting White Balance")
                    area = self.window.select_roi()
                    if area is not None:
                        region = region_reparent(area, self.crop_region)
                        region[2] = max(region[2], region[0] + 1)
                        region[3] = max(region[3], region[1] + 1)
                        roi = extract_region(raw_demo, region)
                        avgcol = np.mean(roi, axis=(0, 1)) - blacklevel
                        print("average color = ", avgcol)
                        g = avgcol[1]
                        b = avgcol[0].mean() / g
                        r = avgcol[2].mean() / g
                        self.gain_b = 1 / b
                        self.gain_r = 1 / r
                        lut = None
                        print(f"ratios: {r:.3f}r/g {b:.3f}b/g -> gain_r={self.gain_r:.3f} gain_b={self.gain_b:.3f}")
                elif cey == "c":
                    print("Setting Crop Region")
                    area = self.window.select_roi(from_center=False)
                    if area is not None:
                        self.crop_region = region_reparent(area, self.crop_region)
                elif cey == "C":
                    print("Cleared Crop Region")
                    self.crop_region = None
                elif cey == "P":
                    f = f"{int(time.time())}-bayer.raw"
                    print("Raw snapshot: ", f)
                    np.save(f, rawimage)
                elif cey == "p":
                    f = f"{int(time.time())}-dev.png"
                    print("Exposed snapshot: ", f)
                    cv2.imwrite(f, output)
                elif key == CVWindow.VK_Escape:
                    return False
            return True

        lut = None
        self.camera_set()
        frametime = 0.05
        time = Stopwatch()
        time.start()
        for buffer in self.capture.raw_captures():
            rawimage = buffer.array
            print("Bayer Frame: %s = %.1f MB (%s)" % (str(rawimage.shape), rawimage.nbytes / 1e6, str(rawimage.dtype)))
            print(str(buffer.get_header()))
            self.window.key_loop()

            # process image
            raw_demo = buffer.demosaic()
            if self.downsample > 1:
                raw_demo = cv2.resize(raw_demo, (raw_demo.shape[1] // self.downsample, raw_demo.shape[0] // self.downsample), interpolation=cv2.INTER_AREA)
            bl_y = int(raw_demo.shape[0]*0.01 + 1)
            bl_x = int(raw_demo.shape[1]*0.01 + 1)
            blacklevel = calc_black_level(raw_demo, bl_x, bl_y, bl_x, bl_y)
            print(f"Black Level = {blacklevel:.1f}")
            self.window.key_loop()

            if self.crop_region:
                raw_demo = extract_region(raw_demo, self.crop_region)

            if not lut:
                lut = ExposureLut(blacklevel, 2 ** 12 - 1, np.uint8,
                                  gamma=self.gamma, gain_b=self.gain_b, gain_r=self.gain_r,
                                  ccm=self.capture.get_ccm(self.temperature) if self.ccm_matrix else None, saturation=self.saturation,
                                  brightess=self.brightness, contrast=self.contrast)
            white_balanced = lut.apply(raw_demo)

            self.window.key_loop()

            output = white_balanced

            def do_overlay(over_img):
                if self.draw_hist:
                    m = np.where(raw_demo.max(axis=-1) > blacklevel * 1.01, 255, 0).astype(np.uint8)
                    hist = histogram_calc(output, mask=m)
                    histogram_draw(over_img, hist)

                text = "\n".join([
                        f"ds = {self.downsample} ft={1000*frametime:.1f}ms",
                        f"gamma = {self.gamma:.2f}",
                        f"gain_r = {self.gain_r:.3f} gain_b = {self.gain_b:.3f} ccm = {self.ccm_matrix} temp = {self.temperature}K sat = {self.saturation:.2f}",
                        f"shutter = {self.shutter:.0f}ms (1/{1000 / self.shutter:.0f})",
                        f"bright = {self.brightness:.3f} contrast = {self.contrast:.3f}",
                    ])
                display_shadow_text(over_img, 20, 25, text)

            scale = max(output.shape[0] / 800, output.shape[1] / 1600)
            self.window.display_image(output, reduction=scale, overlay_fn=do_overlay if self.draw_overlay else None)
            print("displayed")

            self.window.key_loop()
            if not process_keys():
                break

            frametime = frametime * 0.5 + time.stop() * 0.5
            time.start()

            self.camera_set()

            print("next -------")
        self.window.destroy_window()




capture = CaptureApp()
capture.start()
capture.main_loop()

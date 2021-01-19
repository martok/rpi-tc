# RPi Telecine Project

This project hosts code and data for capturing high-quality data using Raspberry Pi cameras, mostly
using the HQ camera (Sony IMX477 sensor).

The goal / use case ist mostly digitizing 8mm Normal8 and Super8 film with some additional hardware by using
the large dynamic range of the IMX477 and its RAW output.


## Resources

* [RPi Telecine Project](https://github.com/Alexamder/rpitelecine)
* [PicameraX Hardware details](https://picamerax.readthedocs.io/en/latest/fov.html)
* [6by9's raspiraw](https://github.com/6by9/userland/tree/rawcam/host_applications/linux/apps/raspicam)
* [Opening Raspberry Pi High Quality Camera Raw Files](https://www.strollswithmydog.com/open-raspberry-pi-high-quality-camera-raw/)
* [libcamera source tree](https://git.linuxtv.org/libcamera.git/)
* [libcamera calibration data](https://git.libcamera.org/libcamera/libcamera.git/tree/src/ipa/raspberrypi/data)

## Tools

### test-cv2.py

Live view of the attached camera or dummy feed. Control via hotkeys:

Key   | Meaning
----  | ----
`ESC` | Quit
`o` | Toggle overlay
`h` | Toggle histogram
`-`,`+` | Decrease/Increase resolution
`g`,`G` | Decrease/Increase gamma
`b`,`B` | Decrease/Increase blue gain
`r`,`R` | Decrease/Increase red gain
`m`     | Toggle CCM (Camera RGB -> sRGB)
`l`,`L` | Decrease/Increase brightness
`k`,`K` | Decrease/Increase contrast
`t`,`T` | Decrease/Increase shutter time
`w`     | Interactive region white balance
`c`     | Interactive crop region
`C`     | Reset crop region
`P`     | Save de-mosaiced raw image data
`p`     | Save processed final image
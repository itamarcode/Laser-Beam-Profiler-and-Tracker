"""Geometric pixel-to-mm estimate for the Microsoft LifeCam HD-3000.

This is NOT an optical calibration. The official 68.5-degree diagonal FOV
applies to the native 1280x720 (16:9) video mode assumed here. At other
capture resolutions, crops or digital zoom, a measured calibration is needed.
A perpendicular planar wall, no lens distortion and square pixels are assumed.
"""

from math import hypot, isfinite, radians, tan

NATIVE_WIDTH = 1280
NATIVE_HEIGHT = 720
DIAGONAL_FOV_DEGREES = 68.5


def geometric_mm_per_pixel(distance_cm, image_width=1280, image_height=720):
    """Return approximate mm/px for native HD-3000 video and an orthogonal wall.

    Raises ValueError when the input is invalid, or if the capture mode does not
    match the video mode for which the field-of-view assumption was made.
    """
    d = float(distance_cm)
    if not isfinite(d) or d <= 0:
        raise ValueError("Camera-wall distance must be a finite, positive number")
    if (int(image_width), int(image_height)) != (NATIVE_WIDTH, NATIVE_HEIGHT):
        raise ValueError("Geometric FOV conversion supported for 1280x720 only")

    diagonal_pixels = hypot(image_width, image_height)
    wall_diagonal_mm = 2.0 * (d * 10.0) * tan(radians(DIAGONAL_FOV_DEGREES / 2.0))
    return wall_diagonal_mm / diagonal_pixels

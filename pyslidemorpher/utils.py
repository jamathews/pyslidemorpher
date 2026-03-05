"""
Utility functions for PySlide Morpher.
Contains image processing utilities, argument parsing helpers, and easing functions.
"""

import argparse
import logging
from pathlib import Path
import cv2
import numpy as np


def parse_size(s: str):
    """Parse size string in format 'WIDTHxHEIGHT' into tuple of integers."""
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("Size must look like WIDTHxHEIGHT, e.g. 1920x1080")


def list_images(folder: Path):
    """List all supported image files in the given folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
    if not files:
        logging.error(f"No images found in: {folder}")
        raise SystemExit(1)
    logging.debug(f"Found {len(files)} valid images in folder: {folder}")
    return files


def fit_letterbox(img, out_wh):
    """Resize image to fit within output dimensions while maintaining aspect ratio."""
    out_w, out_h = out_wh
    h, w = img.shape[:2]
    scale = min(out_w / w, out_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - new_w) // 2
    y0 = (out_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    logging.debug(f"Image resized to fit letterbox of {out_wh}, new size: {new_w}x{new_h}")
    return canvas


def downsample_for_particles(img, pixel_size):
    """Downsample image for particle-based transitions."""
    H, W = img.shape[:2]
    lw, lh = W // pixel_size, H // pixel_size
    low = cv2.resize(img, (lw, lh), interpolation=cv2.INTER_AREA)
    logging.debug(f"Downsampled image for particles with pixel size {pixel_size}, new dimensions: {lw}x{lh}")
    return low, (lh, lw)


def easing_fn(name):
    """Return an easing function based on the given name."""
    def linear(t):
        return t

    def smoothstep(t):
        return t * t * (3 - 2 * t)

    def cubic(t):
        return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2

    name = (name or "smoothstep").lower()
    if name == "linear": return linear
    if name in ("smoothstep", "smooth"): return smoothstep
    if name in ("cubic", "ease", "easeinoutcubic"): return cubic
    return smoothstep
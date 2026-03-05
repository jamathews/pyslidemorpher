#!/usr/bin/env python3
"""
Pixel-Morph Slideshow (Refactored)
----------------------------------
This is the main entry point for the PySlide Morpher application.
The functionality has been refactored into a modular package structure
while maintaining the same command-line interface.

Dependencies:
    pip install imageio[ffmpeg] opencv-python

For PyTorch GPU acceleration (optional):
    pip install torch
"""

from pyslidemorpher import main

if __name__ == "__main__":
    main()
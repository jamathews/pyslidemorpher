#!/usr/bin/env python3
"""
Test script to verify that the realtime window position logging works correctly.
This script will start a realtime slideshow and demonstrate the critical logging
when the window is moved.
"""

import logging
import sys
import os
from pathlib import Path
import argparse

# Add the pyslidemorpher module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from pyslidemorpher.realtime import play_realtime
from pyslidemorpher.utils import list_images, fit_letterbox
import cv2

def setup_logging():
    """Setup logging to show critical messages clearly."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_test_args():
    """Create test arguments for the realtime slideshow."""
    class TestArgs:
        def __init__(self):
            self.size = (800, 600)
            self.fps = 30
            self.seconds_per_transition = 2.0
            self.hold = 1.0
            self.pixel_size = 8
            self.transition = "swirl"
            self.easing = "smoothstep"
            self.audio = None
            self.audio_threshold = 0.1
            self.reactive = False
            self.seed = None
            self.web_gui = False

    return TestArgs()

def main():
    """Main test function."""
    print("=" * 60)
    print("WINDOW POSITION LOGGING TEST")
    print("=" * 60)
    print()
    print("This test will start a realtime slideshow using demo images.")
    print("Move the window around to see critical log messages with coordinates.")
    print("Press 'q' to quit the slideshow.")
    print()
    print("Expected behavior:")
    print("- When you move the window, you should see critical log messages")
    print("- Messages will show: 'Realtime window moved to coordinates: x=X, y=Y'")
    print()

    # Setup logging
    setup_logging()

    # Check for demo images
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print("Demo images directory not found. Creating test with minimal setup...")
        # Try to find any images in the current directory
        image_files = list(Path(".").glob("*.jpg")) + list(Path(".").glob("*.png"))
        if len(image_files) < 2:
            print("ERROR: Need at least 2 image files to test transitions.")
            print("Please ensure you have demo images or other image files available.")
            return 1
        files = image_files[:3]  # Use first 3 images found
    else:
        # Load demo images using list_images function
        try:
            files = list_images(demo_dir)
            files = files[:3]  # Use first 3 demo images
        except SystemExit:
            print("ERROR: No valid images found in demo_images directory.")
            return 1

    # Load and process images like in CLI
    args = create_test_args()
    W, H = args.size
    imgs = []
    for idx, p in enumerate(files):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Could not read {p}, skipping.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = fit_letterbox(img, (W, H))
        imgs.append(img)

    if len(imgs) < 2:
        print("ERROR: Need at least 2 readable images.")
        return 1

    imgs.append(imgs[0])  # Append the first image to the end to create a loop

    print(f"Loaded {len(imgs)} images for slideshow")
    print()
    print("Starting realtime slideshow...")
    print("MOVE THE WINDOW to see position logging in action!")
    print()

    try:
        # Start the realtime slideshow
        play_realtime(imgs, args)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during test: {e}")
        return 1

    print("\nTest completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

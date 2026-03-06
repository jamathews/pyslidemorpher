#!/usr/bin/env python3
"""
Test script to verify that the realtime window position loading works correctly.
This script will test that when the realtime window is created, it reads the saved
position from ~/.pyslidemorpher/window_position.json and opens at those coordinates.
"""

import json
import logging
import sys
import os
import time
from pathlib import Path
import argparse

# Add the pyslidemorpher module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from pyslidemorpher.realtime import play_realtime, load_window_position, save_window_position
from pyslidemorpher.utils import list_images, fit_letterbox
import cv2

def setup_logging():
    """Setup logging to show info messages clearly."""
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

def create_test_position():
    """Create a test position in the JSON file."""
    # Create a specific test position
    test_x, test_y = 100, 150
    save_window_position(test_x, test_y)
    print(f"Created test position: x={test_x}, y={test_y}")
    return test_x, test_y

def verify_position_loading():
    """Verify that the position loading function works."""
    position = load_window_position()
    if position is not None:
        x, y = position
        print(f"✓ Position loaded successfully: x={x}, y={y}")
        return True
    else:
        print("✗ Failed to load position")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("WINDOW POSITION LOADING TEST")
    print("=" * 60)
    print()
    print("This test will verify that the realtime window opens at the saved position.")
    print("The test will:")
    print("1. Create a test position in the JSON file")
    print("2. Verify the position can be loaded")
    print("3. Start the realtime slideshow")
    print("4. Check that the window opens at the saved position")
    print()

    # Setup logging
    setup_logging()

    # Step 1: Create test position
    print("Step 1: Creating test position...")
    test_x, test_y = create_test_position()
    print()

    # Step 2: Verify position loading
    print("Step 2: Verifying position loading...")
    if not verify_position_loading():
        print("ERROR: Position loading failed")
        return 1
    print()

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
    print("Step 3: Starting realtime slideshow...")
    print(f"Expected behavior: Window should open at position x={test_x}, y={test_y}")
    print("Look for the INFO message: 'Restored window position to: x=100, y=150'")
    print("Press 'q' to quit the slideshow.")
    print()

    try:
        # Start the realtime slideshow
        play_realtime(imgs, args)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during test: {e}")
        return 1

    print("\nTest completed!")
    print("If you saw the message 'Restored window position to: x=100, y=150',")
    print("then the window position loading is working correctly!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Test script to verify that window position is saved to JSON file.
This script will start a realtime slideshow and demonstrate that the window
position is saved to ~/.pyslidemorpher/window_position.json when moved.
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

def check_json_file():
    """Check if the JSON file exists and display its contents."""
    json_file = Path.home() / ".pyslidemorpher" / "window_position.json"
    
    if json_file.exists():
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            print(f"✓ JSON file exists: {json_file}")
            print(f"  Position: x={data.get('x')}, y={data.get('y')}")
            print(f"  Timestamp: {data.get('timestamp')}")
            if 'timestamp' in data:
                readable_time = time.ctime(data['timestamp'])
                print(f"  Time: {readable_time}")
            return True
        except Exception as e:
            print(f"✗ Error reading JSON file: {e}")
            return False
    else:
        print(f"✗ JSON file does not exist: {json_file}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("WINDOW POSITION JSON SAVING TEST")
    print("=" * 60)
    print()
    print("This test will start a realtime slideshow using demo images.")
    print("Move the window around to see position saved to JSON file.")
    print("Press 'q' to quit the slideshow.")
    print()
    print("Expected behavior:")
    print("- When you move the window, position is saved to ~/.pyslidemorpher/window_position.json")
    print("- JSON file contains x, y coordinates and timestamp")
    print()

    # Setup logging
    setup_logging()

    # Check initial state
    print("Checking initial state...")
    json_file = Path.home() / ".pyslidemorpher" / "window_position.json"
    if json_file.exists():
        print(f"JSON file already exists: {json_file}")
        check_json_file()
    else:
        print("JSON file does not exist yet (this is expected)")
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
    print("Starting realtime slideshow...")
    print("MOVE THE WINDOW to see position saving in action!")
    print()

    try:
        # Start the realtime slideshow
        play_realtime(imgs, args)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during test: {e}")
        return 1

    print("\nChecking final state...")
    success = check_json_file()
    
    if success:
        print("\n✓ Test completed successfully!")
        print("Window position JSON saving is working correctly.")
    else:
        print("\n✗ Test failed!")
        print("JSON file was not created or could not be read.")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
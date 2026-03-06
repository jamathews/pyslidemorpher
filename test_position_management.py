#!/usr/bin/env python3
"""
Test script for window position management functionality.
"""

import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

# Add the pyslidemorpher module to the path
sys.path.insert(0, str(Path(__file__).parent))

from pyslidemorpher.realtime import play_realtime

def create_test_images():
    """Create some test images for the slideshow."""
    images = []

    # Create 3 simple test images with different colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue

    for i, color in enumerate(colors):
        # Create a 400x300 image
        img_array = np.full((300, 400, 3), color, dtype=np.uint8)

        # Add some text to identify the image
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img_pil)

        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None

        text = f"Test Image {i+1}"
        if font:
            draw.text((50, 150), text, fill=(255, 255, 255), font=font)
        else:
            draw.text((50, 150), text, fill=(255, 255, 255))

        images.append(np.array(img_pil))

    return images

def test_position_management():
    """Test the position management functionality."""
    print("Testing window position management...")
    print("This test will:")
    print("1. Start a realtime slideshow")
    print("2. You can move the window around")
    print("3. Press 'f' to toggle fullscreen")
    print("4. Press 'q' to quit")
    print("5. Restart the test to see if position is remembered")
    print()
    print("Controls:")
    print("- 'q': quit")
    print("- 'p': pause/resume")
    print("- 'r': restart")
    print("- 'f': toggle fullscreen")
    print()

    # Create test images
    images = create_test_images()

    # Create a mock args object
    class MockArgs:
        def __init__(self):
            self.size = (400, 300)
            self.fps = 30
            self.seconds_per_transition = 2.0
            self.hold = 1.0
            self.pixel_size = 4
            self.transition = "swirl"
            self.easing = "ease_in_out"
            self.audio = None
            self.audio_threshold = 0.1
            self.reactive = False
            self.seed = None
            self.web_gui = False

    args = MockArgs()

    try:
        # Run the slideshow
        play_realtime(images, args)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_position_management()

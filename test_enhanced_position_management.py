#!/usr/bin/env python3
"""
Test script for enhanced window position management functionality.
Tests the new position saving system with multi-monitor support.
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

    # Create 3 simple test images with different colors and position info
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    names = ["Red - Position Test", "Green - Position Test", "Blue - Position Test"]

    for i, (color, name) in enumerate(zip(colors, names)):
        # Create a 600x400 image
        img_array = np.full((400, 600, 3), color, dtype=np.uint8)

        # Add some text to identify the image and provide instructions
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img_pil)

        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            font = None

        # Main title
        if font:
            draw.text((50, 50), name, fill=(255, 255, 255), font=font)
            draw.text((50, 100), "Press 's' to set window position", fill=(255, 255, 255), font=font)
            draw.text((50, 130), "Use arrow keys or WASD to move", fill=(255, 255, 255), font=font)
            draw.text((50, 160), "Press '1', '2', '3' for presets", fill=(255, 255, 255), font=font)
            draw.text((50, 190), "Press 'S' (capital) to save", fill=(255, 255, 255), font=font)
            draw.text((50, 220), "Press 'q' to quit position mode", fill=(255, 255, 255), font=font)
            draw.text((50, 280), "Multi-monitor support included!", fill=(255, 255, 255), font=font)
            draw.text((50, 320), "Position will be remembered", fill=(255, 255, 255), font=font)
            draw.text((50, 350), "on next startup", fill=(255, 255, 255), font=font)
        else:
            draw.text((50, 100), name, fill=(255, 255, 255))
            draw.text((50, 150), "Press 's' to set position", fill=(255, 255, 255))
            draw.text((50, 200), "Multi-monitor support!", fill=(255, 255, 255))

        images.append(np.array(img_pil))

    return images

def test_enhanced_position_management():
    """Test the enhanced position management functionality."""
    print("Enhanced Window Position Management Test")
    print("=" * 50)
    print("This test will:")
    print("1. Start a realtime slideshow with enhanced position management")
    print("2. You can test the new position saving system")
    print("3. Press 's' to enter position adjustment mode")
    print("4. Use the controls shown in the window to position it")
    print("5. Test multi-monitor positioning")
    print("6. Save your preferred position")
    print("7. Restart the test to verify position is remembered")
    print()
    print("Enhanced Controls:")
    print("- 's': Enter position adjustment mode")
    print("- Arrow keys or WASD: Move window in small steps")
    print("- Capital WASD: Move window in large steps")
    print("- '1': Preset - Main monitor (100, 100)")
    print("- '2': Preset - Left external monitor (-1820, 100)")
    print("- '3': Preset - Right external monitor (1920, 100)")
    print("- 'S' (capital): Save current position")
    print("- 'q': Quit position mode without saving")
    print("- 'f': Toggle fullscreen")
    print("- 'p': Pause/resume")
    print("- 'r': Restart")
    print("- 'q': Quit slideshow")
    print()

    # Create test images
    images = create_test_images()

    # Create a mock args object
    class MockArgs:
        def __init__(self):
            self.size = (600, 400)
            self.fps = 30
            self.seconds_per_transition = 3.0
            self.hold = 1.5
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

def check_saved_position():
    """Check what position is currently saved."""
    print("Checking saved position...")
    try:
        from pyslidemorpher.realtime import load_window_position
        position_data = load_window_position()
        print(f"Saved position data: {position_data}")

        if position_data.get('remember_position', False):
            x = position_data.get('x', 'Not set')
            y = position_data.get('y', 'Not set')
            print(f"Window will start at position: ({x}, {y})")

            # Provide interpretation for common positions
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if x == 100 and y == 100:
                    print("  → Main monitor, top-left area")
                elif x < 0:
                    print("  → Left external monitor")
                elif x > 1500:
                    print("  → Right external monitor")
                else:
                    print("  → Custom position")
            else:
                print("  → Position coordinates not properly saved")
        else:
            print("No position saved - will use default (100, 100)")
    except Exception as e:
        print(f"Error checking position: {e}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_saved_position()
    else:
        test_enhanced_position_management()

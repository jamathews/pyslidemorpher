#!/usr/bin/env python3
"""
Test script to verify that the position memory fix is working correctly.
This tests the complete position memory system.
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
    """Create test images that show position memory instructions."""
    images = []

    # Create 3 test images with instructions
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]  # Light Red, Green, Blue
    instructions = [
        ["Position Memory Test - Image 1/3", 
         "This tests the FIXED position memory system",
         "1. Move this window to your preferred position",
         "2. Press 's' to enter position mode",
         "3. Use arrow keys/WASD to fine-tune position",
         "4. Press 'S' (capital) to save the position",
         "5. Press 'q' to quit the slideshow",
         "6. Run this test again to verify position is remembered"],
        
        ["Position Memory Test - Image 2/3",
         "The position tracking system now works by:",
         "- Tracking all programmatic window movements",
         "- Saving the tracked position on exit",
         "- Loading and applying saved position on startup",
         "- Providing manual positioning tools ('s' key)",
         "",
         "This should fix the issue where the window",
         "always appeared on the laptop screen!"],
        
        ["Position Memory Test - Image 3/3",
         "Test Instructions:",
         "1. Position the window on your external monitor",
         "2. Use 's' key to save the position",
         "3. Quit and restart this test",
         "4. The window should appear on the external monitor",
         "",
         "If it works, the position memory fix is successful!",
         "Press 'q' to quit and test the fix."]
    ]

    for i, (color, instruction_set) in enumerate(zip(colors, instructions)):
        # Create a 700x500 image
        img_array = np.full((500, 700, 3), color, dtype=np.uint8)

        # Add instructions
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.load_default()
        except:
            font = None

        # Draw instructions
        y_offset = 50
        for line in instruction_set:
            if font:
                draw.text((50, y_offset), line, fill=(255, 255, 255), font=font)
            else:
                draw.text((50, y_offset), line, fill=(255, 255, 255))
            y_offset += 30

        images.append(np.array(img_pil))

    return images

def test_position_memory_fix():
    """Test the complete position memory fix."""
    print("Position Memory Fix Test")
    print("=" * 50)
    print("This test verifies that the position memory system is working correctly.")
    print()
    print("The fix includes:")
    print("- Position tracking for all programmatic window movements")
    print("- Saving tracked position on exit")
    print("- Loading and applying saved position on startup")
    print("- Manual positioning tools via 's' key")
    print()
    print("Instructions:")
    print("1. The slideshow will start")
    print("2. Move the window to your preferred position (e.g., external monitor)")
    print("3. Press 's' to enter position adjustment mode")
    print("4. Use the controls to fine-tune the position")
    print("5. Press 'S' (capital) to save the position")
    print("6. Press 'q' to quit")
    print("7. Run this test again - the window should appear in the saved position")
    print()
    input("Press Enter to start the test...")

    # Create test images
    images = create_test_images()

    # Create args for the slideshow
    class MockArgs:
        def __init__(self):
            self.size = (700, 500)
            self.fps = 30
            self.seconds_per_transition = 4.0
            self.hold = 2.0
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
        print("\nSlideshow ended.")
        print("Now run this test again to verify that the position is remembered!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

def check_current_position():
    """Check what position is currently saved."""
    print("Checking current saved position...")
    try:
        from pyslidemorpher.realtime import load_window_position
        position_data = load_window_position()
        print(f"Current position data: {position_data}")
        
        if position_data.get('remember_position', False):
            x = position_data.get('x', 'Not set')
            y = position_data.get('y', 'Not set')
            print(f"Saved position: ({x}, {y})")
            
            # Provide interpretation
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if x < 0:
                    print("  → Left external monitor")
                elif x > 1500:
                    print("  → Right external monitor")
                elif x == 100 and y == 100:
                    print("  → Default position (main monitor)")
                else:
                    print("  → Custom position")
        else:
            print("No position saved - will use default")
            
    except Exception as e:
        print(f"Error checking position: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_current_position()
    else:
        test_position_memory_fix()
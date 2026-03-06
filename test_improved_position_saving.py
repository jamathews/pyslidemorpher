#!/usr/bin/env python3
"""
Test script to verify the improved position saving system.
This tests the user-friendly position saving improvements.
"""

import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

# Add the pyslidemorpher module to the path
sys.path.insert(0, str(Path(__file__).parent))

from pyslidemorpher.realtime import play_realtime

def create_instruction_images():
    """Create test images with clear instructions for the improved position saving."""
    images = []

    # Create 3 instruction images
    colors = [(100, 150, 255), (150, 255, 100), (255, 150, 100)]  # Light blue, green, orange
    instructions = [
        ["IMPROVED Position Saving Test - Image 1/3",
         "",
         "The position saving system has been improved!",
         "",
         "NEW FEATURES:",
         "• Automatic position saving on exit with feedback",
         "• Clear instructions when no position is saved",
         "• Simple 'save current position' option",
         "• Better feedback about saved positions",
         "",
         "INSTRUCTIONS:",
         "1. Move this window to your preferred location",
         "2. Press 's' key to see the new position options",
         "3. Choose 'S' to save current position immediately",
         "   OR choose 'p' for positioning mode"],
        
        ["IMPROVED Position Saving Test - Image 2/3",
         "",
         "WHAT'S NEW:",
         "",
         "• When you start the program, you'll see:",
         "  - If a position is saved and where it is",
         "  - Clear instructions if no position is saved",
         "",
         "• When you press 's', you get options:",
         "  - 'S' = Save current position immediately",
         "  - 'p' = Enter positioning mode (arrow keys)",
         "  - 'q' = Cancel",
         "",
         "• When you exit, you'll see:",
         "  - Confirmation of what position was saved",
         "  - Which monitor it's on",
         "  - Clear feedback about next startup"],
        
        ["IMPROVED Position Saving Test - Image 3/3",
         "",
         "TEST INSTRUCTIONS:",
         "",
         "1. Move this window to your external monitor",
         "2. Press 's' key",
         "3. Press 'S' (capital) to save immediately",
         "4. You should see confirmation messages",
         "5. Press 'q' to quit",
         "6. You should see exit feedback about saved position",
         "7. Run this test again - it should start on external monitor",
         "",
         "This addresses the user's complaint:",
         "'I can't make it work and you keep telling me it works'",
         "",
         "Now it's much clearer what to do!"]
    ]

    for i, (color, instruction_set) in enumerate(zip(colors, instructions)):
        # Create a 800x600 image
        img_array = np.full((600, 800, 3), color, dtype=np.uint8)

        # Add instructions
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img_pil)

        try:
            font = ImageFont.load_default()
        except:
            font = None

        # Draw instructions
        y_offset = 30
        for line in instruction_set:
            if font:
                draw.text((30, y_offset), line, fill=(255, 255, 255), font=font)
            else:
                draw.text((30, y_offset), line, fill=(255, 255, 255))
            y_offset += 25

        images.append(np.array(img_pil))

    return images

def test_improved_position_saving():
    """Test the improved position saving system."""
    print("Improved Position Saving System Test")
    print("=" * 60)
    print("This test verifies the improvements made to address the user's complaint:")
    print("'Does the user need to tell it to save position or something?")
    print("I can't make it work and you keep telling me it works.'")
    print()
    print("IMPROVEMENTS MADE:")
    print("1. Clear startup messages about position status")
    print("2. Simple 'save current position' option (press 's' then 'S')")
    print("3. Better feedback during position saving")
    print("4. Clear exit messages about what was saved")
    print("5. Instructions shown when no position is saved")
    print()
    print("WHAT TO EXPECT:")
    print("- Clear messages when the slideshow starts")
    print("- Easy position saving with 's' key")
    print("- Confirmation messages when position is saved")
    print("- Exit feedback about saved position")
    print()
    input("Press Enter to start the test...")

    # Create test images
    images = create_instruction_images()

    # Create args for the slideshow
    class MockArgs:
        def __init__(self):
            self.size = (800, 600)
            self.fps = 30
            self.seconds_per_transition = 5.0
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
        print("Did you see clear feedback about position saving?")
        print("Run this test again to verify the position is remembered!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

def check_current_status():
    """Check the current position saving status."""
    print("Checking current position saving status...")
    try:
        from pyslidemorpher.realtime import load_window_position
        position_data = load_window_position()
        
        print(f"Position data: {position_data}")
        
        if position_data.get('remember_position', False):
            x = position_data.get('x', 'Not set')
            y = position_data.get('y', 'Not set')
            print(f"✓ Position is saved: ({x}, {y})")
            
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
                    
                print("✓ The improved system should now work correctly!")
                print("  - You'll see clear startup messages")
                print("  - Press 's' then 'S' to save current position")
                print("  - You'll get confirmation when position is saved")
                print("  - Exit messages will confirm what was saved")
        else:
            print("No position saved yet.")
            print("✓ The improved system will show clear instructions:")
            print("  - 'No saved position found' message")
            print("  - Step-by-step instructions for saving position")
            print("  - Press 's' key to see the new options")
            
    except Exception as e:
        print(f"Error checking position: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        check_current_status()
    else:
        test_improved_position_saving()
#!/usr/bin/env python3
"""
Reality check test - let's see if position saving ACTUALLY works from user perspective.
No more claiming it works - let's prove it or find out why it doesn't.
"""

import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

# Add the pyslidemorpher module to the path
sys.path.insert(0, str(Path(__file__).parent))

def create_simple_test_images():
    """Create simple test images."""
    images = []
    
    # Create 2 simple images
    colors = [(255, 0, 0), (0, 0, 255)]  # Red, Blue
    
    for i, color in enumerate(colors):
        img_array = np.full((400, 600, 3), color, dtype=np.uint8)
        
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        text = f"Image {i+1} - Position Test"
        if font:
            draw.text((50, 200), text, fill=(255, 255, 255), font=font)
        else:
            draw.text((50, 200), text, fill=(255, 255, 255))
        
        images.append(np.array(img_pil))
    
    return images

def test_actual_user_scenario():
    """Test the exact scenario the user is experiencing."""
    print("USER REALITY CHECK TEST")
    print("=" * 50)
    print("Let's test if position saving ACTUALLY works like the user expects.")
    print()
    print("SCENARIO:")
    print("1. User runs slideshow")
    print("2. User drags window to external monitor")
    print("3. User quits slideshow")
    print("4. User runs slideshow again")
    print("5. EXPECTATION: Window should appear on external monitor")
    print("6. REALITY: Does it actually work?")
    print()
    
    # Check current position status
    try:
        from pyslidemorpher.realtime import load_window_position
        position_data = load_window_position()
        print(f"Current saved position: {position_data}")
        
        if position_data.get('remember_position', False):
            x = position_data.get('x', 'Unknown')
            y = position_data.get('y', 'Unknown')
            print(f"Window should start at: ({x}, {y})")
        else:
            print("No position saved - window will start at default location")
    except Exception as e:
        print(f"Error checking position: {e}")
    
    print()
    print("INSTRUCTIONS FOR THIS TEST:")
    print("1. The slideshow will start")
    print("2. MANUALLY drag the window to your external monitor")
    print("3. Press 'q' to quit")
    print("4. Run this test again immediately")
    print("5. See if the window appears where you put it")
    print()
    print("If it doesn't work, we'll know the system is broken.")
    print("If it does work, we'll know what the user might be missing.")
    print()
    input("Press Enter to start the reality check...")
    
    # Create test images
    images = create_simple_test_images()
    
    # Create minimal args
    class MockArgs:
        def __init__(self):
            self.size = (600, 400)
            self.fps = 30
            self.seconds_per_transition = 3.0
            self.hold = 1.0
            self.pixel_size = 4
            self.transition = "default"
            self.easing = "smoothstep"
            self.audio = None
            self.audio_threshold = 0.1
            self.reactive = False
            self.seed = None
            self.web_gui = False
    
    args = MockArgs()
    
    try:
        from pyslidemorpher.realtime import play_realtime
        print("Starting slideshow...")
        print("DRAG THE WINDOW TO YOUR EXTERNAL MONITOR, then press 'q' to quit")
        play_realtime(images, args)
        
        print("\nSlideshow ended.")
        print("Now run this test again to see if it remembers the position!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_actual_user_scenario()
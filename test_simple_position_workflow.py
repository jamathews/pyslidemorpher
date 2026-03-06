#!/usr/bin/env python3
"""
Simple test that focuses on the user's actual workflow:
1. Start slideshow
2. User manually drags window
3. User quits
4. User restarts
5. Window should appear where they put it

This test focuses on the practical solution rather than perfect detection.
"""

import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

# Add the pyslidemorpher module to the path
sys.path.insert(0, str(Path(__file__).parent))

def create_simple_images():
    """Create 2 simple test images."""
    images = []
    
    colors = [(255, 100, 100), (100, 100, 255)]  # Light red, light blue
    texts = ["DRAG ME TO EXTERNAL MONITOR", "THEN PRESS 'q' TO QUIT"]
    
    for i, (color, text) in enumerate(zip(colors, texts)):
        img_array = np.full((300, 500, 3), color, dtype=np.uint8)
        
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        if font:
            draw.text((50, 100), text, fill=(255, 255, 255), font=font)
            draw.text((50, 150), f"Image {i+1}/2", fill=(255, 255, 255), font=font)
            draw.text((50, 200), "Position will be saved on exit", fill=(255, 255, 255), font=font)
        else:
            draw.text((50, 150), text, fill=(255, 255, 255))
        
        images.append(np.array(img_pil))
    
    return images

def test_simple_workflow():
    """Test the simple user workflow."""
    print("SIMPLE POSITION WORKFLOW TEST")
    print("=" * 50)
    
    # Check current position
    try:
        from pyslidemorpher.realtime import load_window_position
        position_data = load_window_position()
        
        if position_data.get('remember_position', False):
            x = position_data.get('x', 'Unknown')
            y = position_data.get('y', 'Unknown')
            print(f"Current saved position: ({x}, {y})")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if x < 0:
                    print("→ Should start on LEFT external monitor")
                elif x > 1500:
                    print("→ Should start on RIGHT external monitor")
                else:
                    print("→ Should start on MAIN monitor")
        else:
            print("No position saved - will start at default location")
    except Exception as e:
        print(f"Error checking position: {e}")
    
    print()
    print("WORKFLOW TEST:")
    print("1. Slideshow will start")
    print("2. MANUALLY DRAG the window to your external monitor")
    print("3. Press 'q' to quit")
    print("4. Run this test again")
    print("5. See if it starts where you put it")
    print()
    print("The system will try to detect manual movements,")
    print("but will definitely save position on exit.")
    print()
    input("Press Enter to start...")
    
    # Create test images
    images = create_simple_images()
    
    # Create minimal args
    class MockArgs:
        def __init__(self):
            self.size = (500, 300)
            self.fps = 30
            self.seconds_per_transition = 2.0
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
        print("DRAG THE WINDOW TO YOUR EXTERNAL MONITOR!")
        print("Then press 'q' to quit and test position saving.")
        play_realtime(images, args)
        
        print("\nSlideshow ended.")
        print("Position should have been saved automatically.")
        print("Run this test again to see if it remembers!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_workflow()
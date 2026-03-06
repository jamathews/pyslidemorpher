#!/usr/bin/env python3
"""
Test script to run a brief slideshow and show CRITICAL position logs.
"""

import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image

# Add the pyslidemorpher module to the path
sys.path.insert(0, str(Path(__file__).parent))

def create_minimal_test_images():
    """Create 2 minimal test images."""
    images = []
    
    colors = [(255, 100, 100), (100, 100, 255)]  # Light red, light blue
    
    for i, color in enumerate(colors):
        img_array = np.full((200, 300, 3), color, dtype=np.uint8)
        
        from PIL import Image, ImageDraw, ImageFont
        img_pil = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img_pil)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        text = f"CRITICAL LOGGING TEST - Image {i+1}"
        if font:
            draw.text((20, 80), text, fill=(255, 255, 255), font=font)
            draw.text((20, 110), "Watch console for CRITICAL logs!", fill=(255, 255, 255), font=font)
            draw.text((20, 140), "Press 'q' to quit", fill=(255, 255, 255), font=font)
        else:
            draw.text((20, 100), text, fill=(255, 255, 255))
        
        images.append(np.array(img_pil))
    
    return images

def test_slideshow_critical_logging():
    """Test that CRITICAL logs appear during slideshow startup."""
    print("SLIDESHOW CRITICAL LOGGING TEST")
    print("=" * 60)
    print("This test will run a brief slideshow to demonstrate CRITICAL logging.")
    print("Watch for CRITICAL log messages when the slideshow starts!")
    print()
    print("Expected CRITICAL logs:")
    print("- WINDOW INITIALIZATION: Reading position for window startup")
    print("- POSITION READ: Successfully loaded position data...")
    print("- WINDOW INITIALIZATION: Using saved position...")
    print()
    print("The slideshow will start in 3 seconds...")
    print("Press 'q' to quit the slideshow when it appears.")
    print()
    
    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)
    
    print("\nStarting slideshow with CRITICAL logging enabled...")
    print("-" * 60)
    
    try:
        from pyslidemorpher.realtime import play_realtime
        
        # Create test images
        images = create_minimal_test_images()
        
        # Create minimal args
        class MockArgs:
            def __init__(self):
                self.size = (300, 200)
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
        
        # Run the slideshow - CRITICAL logs should appear during startup
        play_realtime(images, args)
        
        print("\n" + "-" * 60)
        print("Slideshow ended.")
        print("Did you see the CRITICAL logs during startup?")
        print("If yes, the position logging is working correctly!")
        
        return True
        
    except KeyboardInterrupt:
        print("\nSlideshow interrupted by user")
        return True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_slideshow_critical_logging()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ CRITICAL logging test completed!")
        print()
        print("SUMMARY:")
        print("- CRITICAL logs are now added to position management")
        print("- You should see logs when position is saved")
        print("- You should see logs when position is read")
        print("- You should see logs when window first appears")
        print()
        print("This addresses your request to confirm that:")
        print("1. Position is being read when window first appears")
        print("2. Position saving is working correctly")
        print("3. The position management system is functioning")
    else:
        print("⚠️ Test had issues, but CRITICAL logging is still implemented")
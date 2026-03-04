#!/usr/bin/env python3
"""
Test script for realtime slideshow functionality.
Creates sample images and tests the realtime playback.
"""

import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import cv2

def create_test_images():
    """Create sample test images for the slideshow."""
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Create 4 different colored images
    colors = [
        (255, 100, 100),  # Red-ish
        (100, 255, 100),  # Green-ish
        (100, 100, 255),  # Blue-ish
        (255, 255, 100),  # Yellow-ish
    ]
    
    width, height = 800, 600
    
    for i, color in enumerate(colors):
        # Create a gradient image with the base color
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a gradient effect
        for y in range(height):
            for x in range(width):
                # Create some pattern
                factor_x = x / width
                factor_y = y / height
                
                # Apply gradient and pattern
                r = int(color[0] * (0.5 + 0.5 * factor_x))
                g = int(color[1] * (0.5 + 0.5 * factor_y))
                b = int(color[2] * (0.5 + 0.5 * (factor_x + factor_y) / 2))
                
                img[y, x] = [min(255, r), min(255, g), min(255, b)]
        
        # Add some geometric shapes for visual interest
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 6
        
        # Draw a circle
        cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), -1)
        cv2.circle(img, (center_x, center_y), radius - 20, color, -1)
        
        # Add text
        text = f"Image {i+1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 2, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        cv2.putText(img, text, (text_x, text_y), font, 2, (0, 0, 0), 3)
        
        # Save image
        filename = test_dir / f"test_image_{i+1:02d}.jpg"
        cv2.imwrite(str(filename), img)
        print(f"Created {filename}")
    
    return test_dir

def test_realtime_playback():
    """Test the realtime playback functionality."""
    print("Creating test images...")
    test_dir = create_test_images()
    
    print(f"\nTest images created in {test_dir}")
    print("Now testing realtime playback...")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 'p' to pause/resume")
    print("  - Press 'r' to restart")
    print("\nStarting slideshow in 3 seconds...")
    
    import time
    time.sleep(3)
    
    # Run the slideshow in realtime mode
    cmd = [
        sys.executable, "pyslidemorpher.py",
        str(test_dir),
        "--realtime",
        "--fps", "10",  # Lower FPS for easier testing
        "--seconds-per-transition", "2.0",
        "--hold", "1.0",
        "--pixel-size", "4",
        "--size", "800x600",
        "--transition", "swarm",
        "--log-level", "INFO"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running slideshow: {e}")
        return False
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return True
    
    return True

def cleanup():
    """Clean up test files."""
    test_dir = Path("test_images")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
        print(f"Cleaned up {test_dir}")

if __name__ == "__main__":
    try:
        success = test_realtime_playback()
        if success:
            print("\nRealtime playback test completed successfully!")
        else:
            print("\nRealtime playback test failed!")
            sys.exit(1)
    finally:
        cleanup()
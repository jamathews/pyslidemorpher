#!/usr/bin/env python3
"""
Demonstration script for the new realtime slideshow functionality.
This script shows how to use the --realtime flag and provides usage examples.
"""

import os
import sys
from pathlib import Path
import subprocess

def create_demo_images():
    """Create colorful demo images for the slideshow."""
    import numpy as np
    import cv2
    
    demo_dir = Path("demo_images")
    demo_dir.mkdir(exist_ok=True)
    
    # Create 5 different themed images
    themes = [
        ("Sunset", (255, 150, 50), (255, 100, 0)),
        ("Ocean", (50, 150, 255), (0, 100, 200)),
        ("Forest", (50, 255, 100), (0, 150, 50)),
        ("Purple", (200, 50, 255), (150, 0, 200)),
        ("Golden", (255, 215, 0), (255, 165, 0)),
    ]
    
    width, height = 1024, 768
    
    for i, (name, color1, color2) in enumerate(themes):
        # Create gradient background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            # Create vertical gradient
            factor = y / height
            r = int(color1[0] * (1 - factor) + color2[0] * factor)
            g = int(color1[1] * (1 - factor) + color2[1] * factor)
            b = int(color1[2] * (1 - factor) + color2[2] * factor)
            img[y, :] = [b, g, r]  # BGR for OpenCV
        
        # Add decorative elements
        center_x, center_y = width // 2, height // 2
        
        # Add some circles
        for j in range(3):
            radius = 50 + j * 30
            alpha = 0.3 - j * 0.1
            overlay = img.copy()
            cv2.circle(overlay, (center_x, center_y), radius, (255, 255, 255), -1)
            img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
        
        # Add title text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{name} Scene"
        text_size = cv2.getTextSize(text, font, 3, 4)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height - 100
        
        # Add text shadow
        cv2.putText(img, text, (text_x + 3, text_y + 3), font, 3, (0, 0, 0), 4)
        cv2.putText(img, text, (text_x, text_y), font, 3, (255, 255, 255), 4)
        
        # Add image number
        number_text = f"{i + 1}"
        cv2.putText(img, number_text, (50, 100), font, 4, (255, 255, 255), 6)
        
        # Save image
        filename = demo_dir / f"demo_{i+1:02d}_{name.lower()}.jpg"
        cv2.imwrite(str(filename), img)
        print(f"Created {filename}")
    
    return demo_dir

def show_usage_examples():
    """Show various usage examples for the realtime functionality."""
    print("=" * 60)
    print("REALTIME SLIDESHOW USAGE EXAMPLES")
    print("=" * 60)
    print()
    
    print("1. Basic realtime slideshow:")
    print("   python pyslidemorpher.py demo_images --realtime")
    print()
    
    print("2. High-performance realtime slideshow:")
    print("   python pyslidemorpher.py demo_images --realtime --fps 60 --pixel-size 2")
    print()
    
    print("3. Slow, detailed transitions:")
    print("   python pyslidemorpher.py demo_images --realtime --fps 30 --seconds-per-transition 5.0 --pixel-size 8")
    print()
    
    print("4. Different transition effects:")
    print("   python pyslidemorpher.py demo_images --realtime --transition swarm")
    print("   python pyslidemorpher.py demo_images --realtime --transition tornado")
    print("   python pyslidemorpher.py demo_images --realtime --transition drip")
    print("   python pyslidemorpher.py demo_images --realtime --transition sorted")
    print()
    
    print("5. Custom resolution and timing:")
    print("   python pyslidemorpher.py demo_images --realtime --size 1920x1080 --hold 3.0")
    print()
    
    print("CONTROLS DURING PLAYBACK:")
    print("  - Press 'q' to quit")
    print("  - Press 'p' to pause/resume")
    print("  - Press 'r' to restart slideshow")
    print()
    
    print("PERFORMANCE TIPS:")
    print("  - Lower --pixel-size for faster processing (but less detail)")
    print("  - Higher --fps for smoother playback (requires more CPU)")
    print("  - The slideshow uses threading and buffering for optimal performance")
    print("  - GPU acceleration is automatically enabled if available")
    print()

def run_demo():
    """Run a demonstration of the realtime slideshow."""
    print("Creating demo images...")
    demo_dir = create_demo_images()
    
    print(f"\nDemo images created in {demo_dir}/")
    
    show_usage_examples()
    
    response = input("Would you like to run a demo slideshow now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\nStarting demo slideshow...")
        print("This will show a swarm transition effect at 30 FPS.")
        print("Remember: Press 'q' to quit, 'p' to pause/resume, 'r' to restart")
        print("\nStarting in 3 seconds...")
        
        import time
        time.sleep(3)
        
        # Run the demo
        cmd = [
            sys.executable, "pyslidemorpher.py",
            str(demo_dir),
            "--realtime",
            "--fps", "30",
            "--seconds-per-transition", "3.0",
            "--hold", "1.5",
            "--pixel-size", "4",
            "--size", "1024x768",
            "--transition", "swarm",
            "--log-level", "INFO"
        ]
        
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
    
    # Ask about cleanup
    cleanup_response = input("\nWould you like to keep the demo images for testing? (y/n): ").lower().strip()
    
    if cleanup_response not in ['y', 'yes']:
        import shutil
        shutil.rmtree(demo_dir)
        print(f"Cleaned up {demo_dir}/")
    else:
        print(f"Demo images kept in {demo_dir}/")
        print("You can run more tests using the examples shown above.")

if __name__ == "__main__":
    print("PySlidemorpher Realtime Demo")
    print("=" * 40)
    print()
    
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\nDemo cancelled by user")
    except Exception as e:
        print(f"Error running demo: {e}")
        sys.exit(1)
    
    print("\nDemo completed!")
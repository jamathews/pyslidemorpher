#!/usr/bin/env python3
"""
Test script to reproduce the jarring transition issue.
This will create a short video showing the abrupt change from solid to pixelated.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add the current directory to path so we can import pyslidemorpher
sys.path.insert(0, '.')
from pyslidemorpher import make_transition_frames, fit_letterbox

def create_test_images():
    """Create two simple test images with different colors."""
    # Create a red image
    img_a = np.full((480, 640, 3), [0, 0, 255], dtype=np.uint8)  # Red
    
    # Create a blue image  
    img_b = np.full((480, 640, 3), [255, 0, 0], dtype=np.uint8)  # Blue
    
    return img_a, img_b

def test_jarring_transition():
    """Test the current transition to show the jarring effect."""
    print("Creating test images...")
    img_a, img_b = create_test_images()
    
    # Parameters for transition
    pixel_size = 8
    fps = 30
    seconds = 1.0  # Short transition to see the jarring effect clearly
    hold = 0.1     # Short hold time
    ease_name = "smoothstep"
    seed = 42
    
    print("Generating transition frames...")
    frames = list(make_transition_frames(
        img_a, img_b,
        pixel_size=pixel_size,
        fps=fps,
        seconds=seconds,
        hold=hold,
        ease_name=ease_name,
        seed=seed
    ))
    
    print(f"Generated {len(frames)} frames")
    
    # Save the first few frames to see the jarring transition
    output_dir = Path("test_jarring_frames")
    output_dir.mkdir(exist_ok=True)
    
    for i, frame in enumerate(frames[:10]):  # Save first 10 frames
        filename = output_dir / f"frame_{i:03d}.png"
        cv2.imwrite(str(filename), frame)
        print(f"Saved {filename}")
    
    # Create a video to see the jarring effect in motion
    print("Creating test video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_jarring_transition.mp4', fourcc, fps, (640, 480))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print("Created test_jarring_transition.mp4")
    print("You can see the jarring transition from solid red to pixelated blocks")

if __name__ == "__main__":
    test_jarring_transition()
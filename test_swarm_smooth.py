#!/usr/bin/env python3
"""
Test script to verify the swarm transition with smooth blending.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add the current directory to path so we can import pyslidemorpher
sys.path.insert(0, '.')
from pyslidemorpher import make_swarm_transition_frames

def create_test_images():
    """Create two simple test images with different colors."""
    # Create a red image
    img_a = np.full((480, 640, 3), [0, 0, 255], dtype=np.uint8)  # Red
    
    # Create a blue image  
    img_b = np.full((480, 640, 3), [255, 0, 0], dtype=np.uint8)  # Blue
    
    return img_a, img_b

def test_swarm_smooth_transition():
    """Test the swarm transition with smooth blending."""
    print("Creating test images...")
    img_a, img_b = create_test_images()
    
    # Parameters for transition
    pixel_size = 8
    fps = 30
    seconds = 1.5  # Slightly longer to see the swarm effect
    hold = 0.1     # Short hold time
    ease_name = "smoothstep"
    seed = 42
    
    print("Generating swarm transition frames...")
    frames = list(make_swarm_transition_frames(
        img_a, img_b,
        pixel_size=pixel_size,
        fps=fps,
        seconds=seconds,
        hold=hold,
        ease_name=ease_name,
        seed=seed
    ))
    
    print(f"Generated {len(frames)} frames")
    
    # Save the first few frames to see the smooth transition
    output_dir = Path("test_swarm_smooth_frames")
    output_dir.mkdir(exist_ok=True)
    
    for i, frame in enumerate(frames[:15]):  # Save first 15 frames to see the progression
        filename = output_dir / f"swarm_smooth_frame_{i:03d}.png"
        cv2.imwrite(str(filename), frame)
        print(f"Saved {filename}")
    
    # Create a video to see the smooth swarm effect in motion
    print("Creating swarm smooth transition video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_swarm_smooth_transition.mp4', fourcc, fps, (640, 480))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print("Created test_swarm_smooth_transition.mp4")
    print("The swarm transition should now be smooth from solid red to pixelated swarming and back to solid blue")

if __name__ == "__main__":
    test_swarm_smooth_transition()
#!/usr/bin/env python3
"""
Test script to compare the old jarring transition with the new smooth transition.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# Add the current directory to path so we can import pyslidemorpher
sys.path.insert(0, '.')

def create_test_images():
    """Create two simple test images with different colors."""
    # Create a red image
    img_a = np.full((480, 640, 3), [0, 0, 255], dtype=np.uint8)  # Red
    
    # Create a blue image  
    img_b = np.full((480, 640, 3), [255, 0, 0], dtype=np.uint8)  # Blue
    
    return img_a, img_b

def test_smooth_transition():
    """Test the new smooth transition."""
    print("Creating test images...")
    img_a, img_b = create_test_images()
    
    # Import after creating images to ensure the module loads correctly
    from pyslidemorpher import make_transition_frames
    
    # Parameters for transition
    pixel_size = 8
    fps = 30
    seconds = 1.0  # Short transition to see the effect clearly
    hold = 0.1     # Short hold time
    ease_name = "smoothstep"
    seed = 42
    
    print("Generating smooth transition frames...")
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
    
    # Save the first few frames to see the smooth transition
    output_dir = Path("test_smooth_frames")
    output_dir.mkdir(exist_ok=True)
    
    for i, frame in enumerate(frames[:15]):  # Save first 15 frames to see the progression
        filename = output_dir / f"smooth_frame_{i:03d}.png"
        cv2.imwrite(str(filename), frame)
        print(f"Saved {filename}")
    
    # Create a video to see the smooth effect in motion
    print("Creating smooth transition video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_smooth_transition.mp4', fourcc, fps, (640, 480))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print("Created test_smooth_transition.mp4")
    print("The transition should now be smooth from solid red to pixelated and back to solid blue")

def analyze_frames():
    """Analyze the generated frames to verify smooth blending."""
    print("\nAnalyzing frame transitions...")
    
    # Check if frames exist
    smooth_dir = Path("test_smooth_frames")
    if not smooth_dir.exists():
        print("No smooth frames found. Run the test first.")
        return
    
    # Load first few frames and analyze them
    frame_files = sorted(smooth_dir.glob("smooth_frame_*.png"))[:10]
    
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file))
        if frame is not None:
            # Calculate the variance to detect pixelation level
            # Higher variance indicates more pixelation/detail
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variance = np.var(gray)
            print(f"Frame {i:03d}: variance = {variance:.2f} (higher = more pixelated)")

if __name__ == "__main__":
    test_smooth_transition()
    analyze_frames()
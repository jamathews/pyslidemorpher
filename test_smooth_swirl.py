#!/usr/bin/env python3

import cv2
import numpy as np
from pyslidemorpher import make_swirl_transition_frames

def create_test_images():
    """Create two visually distinct test images for the swirl transition."""
    # Create a red image with a white center circle
    img_a = np.zeros((480, 640, 3), dtype=np.uint8)
    img_a[:, :] = [0, 0, 255]  # Red background
    # Add a white circle in the center
    center = (320, 240)
    cv2.circle(img_a, center, 50, (255, 255, 255), -1)
    
    # Create a blue image with a yellow center circle
    img_b = np.zeros((480, 640, 3), dtype=np.uint8)
    img_b[:, :] = [255, 0, 0]  # Blue background
    # Add a yellow circle in the center
    cv2.circle(img_b, center, 50, (0, 255, 255), -1)
    
    return img_a, img_b

def test_smooth_swirl_transition():
    """Test the smooth three-phase swirl transition function."""
    print("Creating test images...")
    img_a, img_b = create_test_images()
    
    print("Generating smooth swirl transition frames...")
    frames = list(make_swirl_transition_frames(
        img_a, img_b,
        pixel_size=6,  # Smaller pixels for more detail
        fps=30,
        seconds=3.0,   # Longer duration to see the three phases
        hold=0.5,
        ease_name="smoothstep",
        seed=42
    ))
    
    print(f"Generated {len(frames)} frames")
    
    # Save key frames to verify the three-phase transition
    # Phase 1: frames 0-33% (swirling current image)
    # Phase 2: frames 33-67% (transitioning between swirled images)
    # Phase 3: frames 67-100% (unswirling next image)
    
    total_transition_frames = len(frames) - int(0.5 * 30 * 2)  # Subtract hold frames
    phase1_end = int(total_transition_frames * 0.33)
    phase2_end = int(total_transition_frames * 0.67)
    
    key_frame_indices = [
        15,  # Hold frame (should be normal img_a)
        15 + phase1_end // 2,  # Middle of phase 1 (swirling img_a)
        15 + phase1_end,  # End of phase 1 (fully swirled img_a)
        15 + (phase1_end + phase2_end) // 2,  # Middle of phase 2 (transition between swirled images)
        15 + phase2_end,  # End of phase 2 (fully swirled img_b)
        15 + (phase2_end + total_transition_frames) // 2,  # Middle of phase 3 (unswirling img_b)
        len(frames) - 16,  # End of phase 3 (normal img_b)
        len(frames) - 1   # Final hold frame (should be normal img_b)
    ]
    
    phase_names = [
        "hold_start", "phase1_mid", "phase1_end", "phase2_mid", 
        "phase2_end", "phase3_mid", "phase3_end", "hold_end"
    ]
    
    for i, (frame_idx, phase_name) in enumerate(zip(key_frame_indices, phase_names)):
        if frame_idx >= len(frames):
            frame_idx = len(frames) - 1
        frame = frames[frame_idx]
        filename = f"smooth_swirl_frame_{i:02d}_{phase_name}_idx{frame_idx}.png"
        cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Saved {filename}")
    
    # Create a video to see the full smooth transition
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('smooth_swirl_test.mp4', fourcc, 30.0, (640, 480))
    
    for i, frame in enumerate(frames):
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if i % 10 == 0:
            print(f"Writing frame {i+1}/{len(frames)}")
    
    out.release()
    print("Test video saved as 'smooth_swirl_test.mp4'")
    print("Smooth swirl transition test completed successfully!")

if __name__ == "__main__":
    test_smooth_swirl_transition()
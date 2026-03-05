#!/usr/bin/env python3
"""
Integration test to verify that the random transition functionality works correctly
in the context of the main slideshow generation process.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2

# Add the current directory to the path so we can import pyslidemorpher
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyslidemorpher

def create_test_images():
    """Create simple test images for integration testing."""
    test_dir = Path("integration_test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Create 4 simple colored images
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow
    
    for i, color in enumerate(colors):
        img = np.full((100, 100, 3), color, dtype=np.uint8)
        filename = test_dir / f"test_{i+1}.jpg"
        cv2.imwrite(str(filename), img)
    
    return test_dir

def test_random_transition_integration():
    """Test that random transitions work correctly in the slideshow generation process."""
    print("Testing random transition integration...")
    
    # Reset the function's state
    if hasattr(pyslidemorpher.get_random_transition_function, '_last_selected'):
        delattr(pyslidemorpher.get_random_transition_function, '_last_selected')
    
    # Create test images
    test_dir = create_test_images()
    
    try:
        # Load and process images like the main function does
        files = pyslidemorpher.list_images(test_dir)
        W, H = 200, 200
        
        imgs = []
        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = pyslidemorpher.fit_letterbox(img, (W, H))
            imgs.append(img)
        
        if len(imgs) < 2:
            print("❌ Not enough test images")
            return False
        
        # Test multiple transitions with random selection
        selected_functions = []
        consecutive_duplicates = 0
        
        for i in range(min(10, len(imgs) - 1)):  # Test up to 10 transitions
            # Simulate the random transition selection like in the main function
            transition_fn = pyslidemorpher.get_random_transition_function()
            selected_functions.append(transition_fn.__name__)
            
            # Check for consecutive duplicates
            if i > 0 and selected_functions[i] == selected_functions[i-1]:
                consecutive_duplicates += 1
                print(f"ERROR: Consecutive duplicate at transition {i}: {transition_fn.__name__}")
            
            # Generate a few frames to ensure the transition function works
            a, b = imgs[i % len(imgs)], imgs[(i + 1) % len(imgs)]
            frame_gen = transition_fn(
                a, b,
                pixel_size=4,
                fps=10,
                seconds=0.5,  # Short duration for testing
                hold=0.0,
                ease_name="linear",
                seed=123 + i
            )
            
            # Generate just the first frame to verify it works
            try:
                frame = next(frame_gen)
                print(f"Transition {i+1}: {transition_fn.__name__} - Frame shape: {frame.shape}")
            except Exception as e:
                print(f"ERROR: Failed to generate frame for {transition_fn.__name__}: {e}")
                return False
        
        print(f"\nIntegration Test Results:")
        print(f"Total transitions tested: {len(selected_functions)}")
        print(f"Consecutive duplicates found: {consecutive_duplicates}")
        print(f"Selected functions: {selected_functions}")
        
        if consecutive_duplicates == 0:
            print("✅ SUCCESS: No consecutive duplicates in integration test!")
            return True
        else:
            print("❌ FAILURE: Consecutive duplicates found in integration test!")
            return False
            
    finally:
        # Clean up
        if test_dir.exists():
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    print("Running integration test for random transition functionality...\n")
    
    success = test_random_transition_integration()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 INTEGRATION TEST PASSED!")
        print("The random transition functionality works correctly in the slideshow context.")
    else:
        print("💥 INTEGRATION TEST FAILED!")
        sys.exit(1)
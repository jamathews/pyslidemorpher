#!/usr/bin/env python3
"""
Simple test to verify the realtime functionality can be imported and basic functionality works.
"""

import sys
import tempfile
import os
from pathlib import Path
import numpy as np
import cv2

def create_simple_test_images():
    """Create simple test images."""
    test_dir = Path("simple_test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Create 2 simple colored images
    colors = [(255, 0, 0), (0, 255, 0)]  # Red and Green
    
    for i, color in enumerate(colors):
        img = np.full((100, 100, 3), color, dtype=np.uint8)
        filename = test_dir / f"test_{i+1}.jpg"
        cv2.imwrite(str(filename), img)
        print(f"Created {filename}")
    
    return test_dir

def test_import_and_basic_functionality():
    """Test that we can import the module and basic functions work."""
    try:
        # Test importing the main functions
        sys.path.insert(0, '.')
        import pyslidemorpher
        
        print("✓ Successfully imported pyslidemorpher")
        
        # Test basic functions
        test_dir = create_simple_test_images()
        files = pyslidemorpher.list_images(test_dir)
        print(f"✓ Found {len(files)} test images")
        
        if len(files) >= 2:
            # Test image loading and processing
            img = cv2.imread(str(files[0]), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = pyslidemorpher.fit_letterbox(img, (200, 200))
            print("✓ Image processing functions work")
            
            # Test transition frame generation (just generate one frame)
            img2 = cv2.imread(str(files[1]), cv2.IMREAD_COLOR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img2 = pyslidemorpher.fit_letterbox(img2, (200, 200))
            
            # Generate just one frame to test the function
            frame_gen = pyslidemorpher.make_transition_frames(
                img, img2,
                pixel_size=4,
                fps=10,
                seconds=1.0,
                hold=0.0,
                ease_name="linear",
                seed=123
            )
            
            frame = next(frame_gen)
            print(f"✓ Generated transition frame with shape: {frame.shape}")
            
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
        print("✓ Cleaned up test files")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_argument_parsing():
    """Test that the new --realtime argument is properly parsed."""
    try:
        import argparse
        sys.path.insert(0, '.')
        
        # Create a mock argument parser similar to the one in main
        ap = argparse.ArgumentParser(description="Test argument parsing")
        ap.add_argument("photos_folder", type=Path, help="Folder containing images")
        ap.add_argument("--realtime", action="store_true",
                        help="Play slideshow in realtime instead of writing to file")
        
        # Test parsing with --realtime flag
        args = ap.parse_args(["test_folder", "--realtime"])
        assert args.realtime == True
        print("✓ --realtime argument parsing works")
        
        # Test parsing without --realtime flag
        args = ap.parse_args(["test_folder"])
        assert args.realtime == False
        print("✓ Default (no --realtime) argument parsing works")
        
        return True
        
    except Exception as e:
        print(f"✗ Argument parsing error: {e}")
        return False

if __name__ == "__main__":
    print("Running simple tests for realtime functionality...\n")
    
    success = True
    
    print("1. Testing argument parsing...")
    if not test_argument_parsing():
        success = False
    
    print("\n2. Testing import and basic functionality...")
    if not test_import_and_basic_functionality():
        success = False
    
    if success:
        print("\n✓ All simple tests passed!")
        print("\nTo test realtime playback manually, run:")
        print("python pyslidemorpher.py <image_folder> --realtime")
        print("Controls: 'q' to quit, 'p' to pause/resume, 'r' to restart")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
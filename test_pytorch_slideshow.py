#!/usr/bin/env python3
"""
Test script to verify PyTorch acceleration works with actual slideshow generation.
"""

import sys
import subprocess
import tempfile
import os
from pathlib import Path
import numpy as np
import cv2

def create_test_images():
    """Create simple test images for slideshow testing."""
    test_dir = Path("pytorch_slideshow_test")
    test_dir.mkdir(exist_ok=True)
    
    # Create 3 simple colored images
    colors = [
        (255, 100, 100),  # Red-ish
        (100, 255, 100),  # Green-ish
        (100, 100, 255),  # Blue-ish
    ]
    
    width, height = 400, 300
    
    for i, color in enumerate(colors):
        img = np.full((height, width, 3), color, dtype=np.uint8)
        
        # Add a simple pattern to make transitions more visible
        center_x, center_y = width // 2, height // 2
        cv2.circle(img, (center_x, center_y), 50, (255, 255, 255), -1)
        cv2.putText(img, f"Image {i+1}", (center_x-40, center_y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        filename = test_dir / f"test_{i+1}.jpg"
        cv2.imwrite(str(filename), img)
        print(f"Created {filename}")
    
    return test_dir

def test_slideshow_with_pytorch():
    """Test slideshow generation with PyTorch acceleration."""
    print("Testing slideshow generation with PyTorch acceleration...")
    
    test_dir = create_test_images()
    
    try:
        # Test with PyTorch enabled
        cmd = [
            sys.executable, "pyslidemorpher.py",
            str(test_dir),
            "--use-pytorch",
            "--fps", "10",
            "--seconds-per-transition", "1.0",
            "--hold", "0.5",
            "--pixel-size", "4",
            "--size", "400x300",
            "--transition", "swarm",
            "--out", "pytorch_test_output.mp4",
            "--log-level", "INFO"
        ]
        
        print("Running slideshow with PyTorch acceleration...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Slideshow with PyTorch acceleration completed successfully")
            
            # Check if output file was created
            if os.path.exists("pytorch_test_output.mp4"):
                file_size = os.path.getsize("pytorch_test_output.mp4")
                print(f"✓ Output video created: pytorch_test_output.mp4 ({file_size} bytes)")
                
                # Clean up output file
                os.remove("pytorch_test_output.mp4")
                return True
            else:
                print("✗ Output video file was not created")
                return False
        else:
            print(f"✗ Slideshow generation failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Slideshow generation timed out")
        return False
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        return False
    finally:
        # Clean up test images
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"Cleaned up {test_dir}")

def test_slideshow_comparison():
    """Compare slideshow generation with and without PyTorch."""
    print("\nTesting slideshow generation comparison...")
    
    test_dir = create_test_images()
    
    try:
        import time
        
        # Test without PyTorch
        cmd_numpy = [
            sys.executable, "pyslidemorpher.py",
            str(test_dir),
            "--fps", "10",
            "--seconds-per-transition", "1.0",
            "--hold", "0.5",
            "--pixel-size", "4",
            "--size", "400x300",
            "--transition", "default",
            "--out", "numpy_test_output.mp4",
            "--log-level", "ERROR"  # Reduce log output
        ]
        
        print("Running slideshow without PyTorch...")
        start_time = time.time()
        result_numpy = subprocess.run(cmd_numpy, capture_output=True, text=True, timeout=30)
        numpy_time = time.time() - start_time
        
        # Test with PyTorch
        cmd_pytorch = [
            sys.executable, "pyslidemorpher.py",
            str(test_dir),
            "--use-pytorch",
            "--fps", "10",
            "--seconds-per-transition", "1.0",
            "--hold", "0.5",
            "--pixel-size", "4",
            "--size", "400x300",
            "--transition", "default",
            "--out", "pytorch_test_output.mp4",
            "--log-level", "ERROR"  # Reduce log output
        ]
        
        print("Running slideshow with PyTorch...")
        start_time = time.time()
        result_pytorch = subprocess.run(cmd_pytorch, capture_output=True, text=True, timeout=30)
        pytorch_time = time.time() - start_time
        
        # Compare results
        if result_numpy.returncode == 0 and result_pytorch.returncode == 0:
            print(f"✓ Both implementations completed successfully")
            print(f"  NumPy time:   {numpy_time:.2f} seconds")
            print(f"  PyTorch time: {pytorch_time:.2f} seconds")
            
            if pytorch_time < numpy_time:
                speedup = numpy_time / pytorch_time
                print(f"  🚀 PyTorch is {speedup:.2f}x faster!")
            elif pytorch_time > numpy_time:
                slowdown = pytorch_time / numpy_time
                print(f"  ⚠️ PyTorch is {slowdown:.2f}x slower (expected for small datasets)")
            else:
                print(f"  📊 Both implementations performed similarly")
            
            # Clean up output files
            for filename in ["numpy_test_output.mp4", "pytorch_test_output.mp4"]:
                if os.path.exists(filename):
                    os.remove(filename)
            
            return True
        else:
            print("✗ One or both implementations failed")
            return False
            
    except Exception as e:
        print(f"✗ Comparison test failed: {e}")
        return False
    finally:
        # Clean up test images
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)

def main():
    print("PyTorch Slideshow Integration Test")
    print("=" * 50)
    
    success = True
    
    # Test slideshow with PyTorch
    if not test_slideshow_with_pytorch():
        success = False
    
    # Test comparison between NumPy and PyTorch
    if not test_slideshow_comparison():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All PyTorch slideshow tests passed!")
        print("\nPyTorch acceleration is working correctly with the slideshow.")
        print("You can now use PyTorch acceleration with confidence:")
        print("  # For video generation:")
        print("  python pyslidemorpher.py <folder> --use-pytorch")
        print("  # For realtime playback:")
        print("  python pyslidemorpher.py <folder> --use-pytorch --realtime")
    else:
        print("✗ Some PyTorch slideshow tests failed!")
        print("There may be issues with PyTorch integration.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
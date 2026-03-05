#!/usr/bin/env python3
"""Test script to verify that critical logs appear for all transition types."""

import sys
import os
import tempfile
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

def create_test_images():
    """Create temporary test images."""
    temp_dir = tempfile.mkdtemp()
    images = []

    for i in range(3):
        # Create a simple colored image
        img_array = np.full((100, 100, 3), i * 80, dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = Path(temp_dir) / f"test_image_{i}.jpg"
        img.save(img_path)
        images.append(img_path)

    return images, temp_dir

def test_transition_logging(transition_type):
    """Test that critical logs appear for a specific transition type."""
    print(f"Testing critical logging for transition type: {transition_type}")

    # Create test images
    test_images, temp_dir = create_test_images()

    try:
        output_path = Path(temp_dir) / f"test_output_{transition_type}.mp4"

        # Build command to run the CLI
        cmd = [
            sys.executable, "-m", "pyslidemorpher",
            "--transition", transition_type,
            "--size", "100x100",
            "--pixel-size", "4",
            "--fps", "30",
            "--seconds-per-transition", "0.1",
            "--hold", "0.1",
            "--log-level", "CRITICAL",
            "--out", str(output_path),
            str(temp_dir)  # Pass the folder containing the images
        ]

        print(f"Running command: {' '.join(cmd)}")

        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())

        print("STDERR:")
        print(result.stderr)

        # Check if the "Using transition" log message appears in the output
        if "Using transition:" in result.stderr or "Using transition:" in result.stdout:
            print(f"✅ SUCCESS: 'Using transition' log message found for {transition_type}!")
            return True
        else:
            print(f"❌ ISSUE: 'Using transition' log message not found for {transition_type}")
            return False

    except Exception as e:
        print(f"❌ Error during test for {transition_type}: {e}")
        return False
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    """Test critical logging for various transition types."""
    transition_types = ["default", "swarm", "tornado", "swirl", "drip", "sorted", "random"]
    
    results = {}
    for transition_type in transition_types:
        print(f"\n{'='*50}")
        results[transition_type] = test_transition_logging(transition_type)
    
    print(f"\n{'='*50}")
    print("SUMMARY:")
    for transition_type, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{transition_type}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Critical logging is working for all transition types.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    return all_passed

if __name__ == "__main__":
    main()
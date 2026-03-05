#!/usr/bin/env python3
"""Test script to verify that the 'Randomly selected transition' log message is now visible."""

import sys
import os
import logging
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

def test_random_transition_logging():
    """Test that the 'Randomly selected transition' log message appears."""
    print("Testing random transition logging...")

    # Create test images
    test_images, temp_dir = create_test_images()

    try:
        output_path = Path(temp_dir) / "test_output.mp4"

        # Build command to run the CLI
        cmd = [
            sys.executable, "-m", "pyslidemorpher",
            "--transition", "random",
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
        print("Looking for 'Randomly selected transition' log messages...")

        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())

        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)

        # Check if the log message appears in the output
        if "Randomly selected transition" in result.stderr or "Randomly selected transition" in result.stdout:
            print("✅ SUCCESS: 'Randomly selected transition' log message found!")
            return True
        else:
            print("❌ ISSUE: 'Randomly selected transition' log message not found in output")
            return False

    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_random_transition_logging()

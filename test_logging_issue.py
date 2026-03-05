#!/usr/bin/env python3
"""
Test script to reproduce the logging issue.
"""

import sys
import subprocess
from pathlib import Path

def test_logging():
    """Test if logging is working."""
    print("Testing logging functionality...")

    # Check if demo images exist
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print("Error: demo_images directory not found")
        return False

    # Test command with explicit log level
    cmd = [
        sys.executable, "-m", "pyslidemorpher",
        str(demo_dir),
        "--realtime",
        "--size", "400x300",
        "--fps", "10",
        "--seconds-per-transition", "1.0",
        "--pixel-size", "8",
        "--log-level", "INFO"
    ]

    print("Running command:", " ".join(cmd))
    print("Expected: Should see logging messages like 'Starting Pixel-Morph Slideshow in realtime mode.'")
    print("Press Ctrl+C to stop after a few seconds...")

    try:
        result = subprocess.run(cmd, timeout=5, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        return True
    except subprocess.TimeoutExpired as e:
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_logging()

#!/usr/bin/env python3
"""
Simple test script to check if logging is working without realtime mode.
"""

import sys
import subprocess
from pathlib import Path

def test_logging_simple():
    """Test logging with video generation mode (should exit quickly)."""
    print("Testing logging functionality (video generation mode)...")
    
    # Check if demo images exist
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print("Error: demo_images directory not found")
        return False
    
    # Test command for video generation (should show logging and exit)
    cmd = [
        sys.executable, "-m", "pyslidemorpher",
        str(demo_dir),
        "--out", "test_logging_output.mp4",
        "--size", "200x150",  # Small size for quick processing
        "--fps", "5",         # Low fps for quick processing
        "--seconds-per-transition", "0.5",  # Short transitions
        "--pixel-size", "16", # Large pixels for quick processing
        "--log-level", "INFO"
    ]
    
    print("Running command:", " ".join(cmd))
    print("This should generate a video and show logging messages...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print("Return code:", result.returncode)
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        
        # Clean up the test file if it was created
        test_file = Path("test_logging_output.mp4")
        if test_file.exists():
            test_file.unlink()
            print("Cleaned up test output file")
        
        return result.returncode == 0
    except subprocess.TimeoutExpired as e:
        print("Command timed out")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_logging_simple()
    if success:
        print("\n✓ Logging test completed successfully")
    else:
        print("\n✗ Logging test failed")
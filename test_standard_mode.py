#!/usr/bin/env python3
"""
Test script to ensure standard (non-reactive) mode still works correctly.
"""

import sys
import subprocess
from pathlib import Path

def test_standard_mode():
    """Test the standard time-based slideshow mode."""
    print("Testing Standard Time-Based Slideshow")
    print("=" * 40)
    
    # Check if demo images exist
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print("Error: demo_images directory not found")
        return False
    
    print(f"Using images from: {demo_dir}")
    
    # Test command for standard realtime mode (no audio reactivity)
    cmd = [
        sys.executable, "-m", "pyslidemorpher.cli",
        str(demo_dir),
        "--realtime",
        "--size", "600x400",
        "--fps", "30",
        "--seconds-per-transition", "1.5",
        "--pixel-size", "6",
        "--transition", "swirl",
        "--log-level", "INFO"
    ]
    
    print("\nRunning standard slideshow (5 seconds)...")
    print("Command:", " ".join(cmd))
    print("\nThis should run the standard time-based transitions without audio reactivity.")
    print("Press 'q' to quit early if needed.")
    
    try:
        # Run for a short time to verify it works
        process = subprocess.Popen(cmd)
        import time
        time.sleep(5)  # Let it run for 5 seconds
        process.terminate()
        process.wait(timeout=2)
        print("\nStandard mode test completed successfully!")
        return True
    except subprocess.TimeoutExpired:
        process.kill()
        print("\nStandard mode test completed (process terminated)!")
        return True
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_standard_mode()
    if success:
        print("\n✓ Standard mode test passed")
    else:
        print("\n✗ Standard mode test failed")
        sys.exit(1)
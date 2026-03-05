#!/usr/bin/env python3
"""
Manual test script for the stop command fix.
This script starts a slideshow and provides instructions for manual testing.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run a manual test of the stop command."""
    print("PySlidemorpher Stop Command Manual Test")
    print("=" * 40)

    # Check if demo images exist
    demo_dir = Path("demo_images")
    if not demo_dir.exists() or len(list(demo_dir.glob("*.jpg"))) == 0:
        print("Error: No demo images found. Please run demo_realtime.py first.")
        return False

    print(f"Found demo images in {demo_dir}/")

    # Start slideshow with web GUI
    cmd = [
        sys.executable, "pyslidemorpher.py",
        str(demo_dir),
        "--realtime",
        "--web-gui",
        "--fps", "30",
        "--seconds-per-transition", "3.0",
        "--hold", "1.0",
        "--pixel-size", "4",
        "--size", "600x400",
        "--log-level", "INFO"
    ]

    print("\nThis test will start a slideshow with the web GUI enabled.")
    print("Manual testing instructions:")
    print("1. The slideshow will start in a few seconds")
    print("2. Open http://localhost:5001 in your browser")
    print("3. Click the '⏹️ Stop' button")
    print("4. Verify that the slideshow terminates immediately")
    print("5. Press Ctrl+C here if the slideshow doesn't stop")
    print()

    response = input("Ready to start the test? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Test cancelled.")
        return True

    print(f"\nRunning command: {' '.join(cmd)}")
    print("\nStarting slideshow...")
    print("=" * 50)
    print("MANUAL TEST INSTRUCTIONS:")
    print("1. Wait for 'Web GUI available at http://localhost:5001' message")
    print("2. Open http://localhost:5001 in your browser")
    print("3. Click the '⏹️ Stop' button")
    print("4. The slideshow should terminate immediately")
    print("5. If it doesn't stop, press Ctrl+C to terminate")
    print("=" * 50)

    try:
        # Start the slideshow process and wait for it to complete
        result = subprocess.run(cmd)

        if result.returncode == 0:
            print("\n✓ Slideshow terminated normally")
            print("If you clicked the stop button and the slideshow stopped, the fix is working!")
            return True
        else:
            print(f"\n⚠ Slideshow terminated with code: {result.returncode}")
            print("This might be normal if you pressed Ctrl+C or closed the window.")
            return True

    except KeyboardInterrupt:
        print("\n⚠ Test interrupted by user (Ctrl+C)")
        print("If the stop button didn't work, there may still be an issue.")
        return True
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

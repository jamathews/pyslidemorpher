#!/usr/bin/env python3
"""
Test script to verify that the stop command fix works correctly.
This script tests that the stop button properly terminates the slideshow.
"""

import sys
import time
import subprocess
import threading
from pathlib import Path

def test_stop_command_functionality():
    """Test that the stop command properly terminates the slideshow."""
    print("Testing stop command functionality...")

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
        "--seconds-per-transition", "2.0",
        "--hold", "1.0",
        "--pixel-size", "4",
        "--size", "400x300",
        "--log-level", "INFO"
    ]

    print("Starting slideshow with web GUI...")
    print("This test will:")
    print("1. Start the slideshow")
    print("2. Wait 5 seconds for it to initialize")
    print("3. Send a stop command via the web API")
    print("4. Verify the slideshow terminates")

    try:
        # Start the slideshow process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Wait for the slideshow to start
        print("Waiting for slideshow to initialize...")
        time.sleep(5)

        # Check if process is still running
        if process.poll() is not None:
            print("✗ Slideshow terminated unexpectedly during startup")
            return False

        print("✓ Slideshow started successfully")

        # Send stop command via web API
        print("Sending stop command via web API...")
        try:
            import requests
            response = requests.post('http://localhost:5001/api/command',
                                   json={'command': 'stop'},
                                   timeout=5)
            if response.status_code == 200:
                print("✓ Stop command sent successfully")
            else:
                print(f"✗ Failed to send stop command: {response.status_code}")
                process.terminate()
                return False
        except ImportError:
            print("⚠ requests module not available, using alternative method")
            # Alternative: send stop command directly to controller
            try:
                from pyslidemorpher.web_gui import get_controller
                controller = get_controller()
                controller.send_command('stop')
                print("✓ Stop command sent via controller")
            except Exception as e:
                print(f"✗ Failed to send stop command: {e}")
                process.terminate()
                return False
        except Exception as e:
            print(f"✗ Failed to send stop command: {e}")
            process.terminate()
            return False

        # Wait for process to terminate
        print("Waiting for slideshow to stop...")
        try:
            process.wait(timeout=10)
            print("✓ Slideshow terminated successfully")
            return True
        except subprocess.TimeoutExpired:
            print("✗ Slideshow did not terminate within 10 seconds")
            process.terminate()
            process.wait(timeout=5)
            return False

    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        if 'process' in locals():
            process.terminate()
        return False

def test_controller_stop_command():
    """Test the controller stop command directly."""
    print("\nTesting controller stop command directly...")

    try:
        from pyslidemorpher.web_gui import RealtimeController

        controller = RealtimeController()

        # Send stop command
        controller.send_command('stop')

        # Check that the command was queued
        if not controller.command_queue.empty():
            command = controller.command_queue.get()
            if command == 'stop':
                print("✓ Stop command successfully queued in controller")
                return True
            else:
                print(f"✗ Wrong command in queue: {command}")
                return False
        else:
            print("✗ Command queue is empty")
            return False

    except Exception as e:
        print(f"✗ Controller test failed: {e}")
        return False

def main():
    """Run the stop command fix test."""
    print("PySlidemorpher Stop Command Fix Test")
    print("=" * 40)

    # Test controller functionality first
    if not test_controller_stop_command():
        print("\n❌ Controller test failed")
        return False

    # Test full functionality
    print("\nTesting full stop command functionality...")
    print("This will start a slideshow and test the stop command.")

    response = input("Do you want to run the full test? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Full test skipped.")
        return True

    if not test_stop_command_functionality():
        print("\n❌ Stop command functionality test failed")
        return False

    print("\n🎉 All tests passed! The stop command fix is working correctly.")
    print("\nThe stop button should now properly terminate the slideshow when clicked.")
    return True

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

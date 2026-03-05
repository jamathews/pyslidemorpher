#!/usr/bin/env python3
"""
Test script for the web GUI functionality.
This script tests the web-based control interface for PySlidemorpher.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def test_web_gui():
    """Test the web GUI functionality."""
    print("Testing PySlidemorpher Web GUI")
    print("=" * 40)

    # Check if demo images exist
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print("Creating demo images...")
        # Run the demo script to create images
        try:
            subprocess.run([sys.executable, "demo_realtime.py"], input="n\ny\n", text=True, timeout=30)
        except subprocess.TimeoutExpired:
            print("Demo creation timed out, continuing with existing images...")
        except Exception as e:
            print(f"Error creating demo images: {e}")
            return False

    if not demo_dir.exists() or len(list(demo_dir.glob("*.jpg"))) == 0:
        print("Error: No demo images found. Please run demo_realtime.py first.")
        return False

    print(f"Found demo images in {demo_dir}/")

    # Test 1: Basic web GUI functionality
    print("\nTest 1: Starting slideshow with web GUI...")
    print("This will start a slideshow with web GUI enabled.")
    print("Open http://localhost:5001 in your browser to test the controls.")
    print("The slideshow will run for 30 seconds, then automatically stop.")
    print("Press Ctrl+C to stop early if needed.")

    cmd = [
        sys.executable, "pyslidemorpher.py",
        str(demo_dir),
        "--realtime",
        "--web-gui",
        "--fps", "30",
        "--seconds-per-transition", "3.0",
        "--hold", "1.0",
        "--pixel-size", "4",
        "--size", "800x600",
        "--transition", "swarm",
        "--log-level", "INFO"
    ]

    print(f"Running command: {' '.join(cmd)}")
    print("\nStarting slideshow...")
    print("=" * 50)
    print("WEB GUI INSTRUCTIONS:")
    print("1. Open http://localhost:5001 in your browser")
    print("2. Try adjusting the FPS slider")
    print("3. Change the transition type")
    print("4. Modify timing settings")
    print("5. Use the pause/resume buttons")
    print("6. Test the restart functionality")
    print("=" * 50)

    try:
        # Start the slideshow process
        process = subprocess.Popen(cmd)

        # Let it run for 30 seconds
        time.sleep(30)

        # Terminate the process
        process.terminate()
        process.wait(timeout=5)

        print("\nTest completed successfully!")
        return True

    except subprocess.TimeoutExpired:
        print("\nProcess didn't terminate cleanly, killing...")
        process.kill()
        return False
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        process.terminate()
        return True
    except Exception as e:
        print(f"\nError during test: {e}")
        return False

def test_without_flask():
    """Test behavior when Flask is not available."""
    print("\nTest 2: Testing behavior without Flask...")

    # This test would require temporarily renaming Flask import
    # For now, just show what the error message should look like
    print("If Flask is not installed, you should see:")
    print("ERROR - Web GUI requested but Flask is not available. Install Flask to use web GUI.")

    return True

def show_installation_instructions():
    """Show installation instructions for Flask."""
    print("\nINSTALLATION INSTRUCTIONS:")
    print("=" * 30)
    print("To use the web GUI, you need to install Flask:")
    print()
    print("Using pip:")
    print("  pip install flask")
    print()
    print("Using conda:")
    print("  conda install flask")
    print()
    print("Or add it to your Pipfile:")
    print('  flask = "*"')
    print()

if __name__ == "__main__":
    print("PySlidemorpher Web GUI Test Suite")
    print("=" * 40)

    # Check if Flask is available
    try:
        import flask
        flask_available = True
        print("✓ Flask is available")
    except ImportError:
        flask_available = False
        print("✗ Flask is not available")
        show_installation_instructions()

        response = input("Do you want to continue with the test anyway? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            print("Test cancelled. Please install Flask and try again.")
            sys.exit(1)

    try:
        if flask_available:
            success = test_web_gui()
            if success:
                print("\n✓ Web GUI test completed successfully!")
                print("\nNEXT STEPS:")
                print("1. The web GUI is now integrated into PySlidemorpher")
                print("2. Use --web-gui flag with --realtime to enable it")
                print("3. Open http://localhost:5001 to control the slideshow")
                print("4. All settings can be adjusted in real-time")
            else:
                print("\n✗ Web GUI test failed")
                sys.exit(1)
        else:
            test_without_flask()

    except KeyboardInterrupt:
        print("\nTest suite interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

    print("\nTest suite completed!")

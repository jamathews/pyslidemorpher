#!/usr/bin/env python3
"""
Test script for the enhanced audio-reactive slideshow functionality.
This script tests the improved audio reactivity features.
"""

import sys
import subprocess
import time
from pathlib import Path

def test_enhanced_reactive_mode():
    """Test the enhanced reactive mode with audio."""
    print("Testing Enhanced Audio-Reactive Slideshow")
    print("=" * 50)
    
    # Check if demo images exist
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print("Error: demo_images directory not found")
        return False
    
    # Check if audio file exists
    audio_file = Path("audio/Fragment 0013.wav")
    if not audio_file.exists():
        print("Error: audio file not found")
        return False
    
    print(f"Using images from: {demo_dir}")
    print(f"Using audio file: {audio_file}")
    
    # Test command with enhanced reactive mode
    cmd = [
        sys.executable, "-m", "pyslidemorpher.cli",
        str(demo_dir),
        "--realtime",
        "--reactive",
        "--audio", str(audio_file),
        "--audio-threshold", "0.05",  # Lower threshold for more sensitivity
        "--transition", "random",     # Use random transitions to test audio-based selection
        "--size", "800x600",
        "--fps", "30",
        "--seconds-per-transition", "2.0",
        "--pixel-size", "8",
        "--log-level", "INFO"
    ]
    
    print("\nRunning enhanced reactive slideshow...")
    print("Command:", " ".join(cmd))
    print("\nFeatures being tested:")
    print("- Enhanced audio analysis (intensity, peak, spectral centroid, beat strength)")
    print("- Adaptive thresholds based on audio history")
    print("- Multiple trigger conditions (intensity, beat, peak)")
    print("- Audio-responsive transition speed")
    print("- Audio-responsive pixel size")
    print("- Audio-based transition type selection")
    print("- Brightness modulation during non-transition periods")
    print("- Dynamic minimum intervals based on beat strength")
    print("\nPress 'q' to quit the slideshow when you're done testing.")
    print("Watch for:")
    print("- Transitions should feel more connected to the audio")
    print("- Different audio characteristics should trigger different transition types")
    print("- Transition speed should vary with audio intensity")
    print("- Images should have subtle brightness changes with audio")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\nTest completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nTest failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return True

if __name__ == "__main__":
    success = test_enhanced_reactive_mode()
    if success:
        print("\n✓ Enhanced audio-reactive mode test completed")
    else:
        print("\n✗ Enhanced audio-reactive mode test failed")
        sys.exit(1)
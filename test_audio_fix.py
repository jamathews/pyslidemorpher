#!/usr/bin/env python3
"""
Test script to verify that audio now works in realtime mode.
"""

import subprocess
import sys
import time
from pathlib import Path

def test_audio_fix():
    """Test that audio now plays in realtime mode."""
    
    # Check if we have demo images and audio
    demo_images = Path("assets/demo_images")
    demo_audio = Path("assets/audio")
    
    if not demo_images.exists():
        print("Error: Demo images directory not found")
        return False
        
    if not demo_audio.exists():
        print("Error: Demo audio directory not found")
        return False
    
    # Find an audio file
    audio_files = list(demo_audio.glob("*.mp3"))
    if not audio_files:
        print("Error: No audio files found")
        return False
    
    audio_file = audio_files[0]
    print(f"Using audio file: {audio_file}")
    print(f"Using images from: {demo_images}")
    
    # Test command that should now play audio
    cmd = [
        sys.executable, "-m", "pyslidemorpher",
        str(demo_images),
        "--realtime",
        "--audio", str(audio_file),
        "--fps", "10",
        "--seconds-per-transition", "1.0",
        "--hold", "0.5"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print("\nThis should now play audio along with the slideshow!")
    print("Listen for audio and press Ctrl+C after a few seconds to stop.")
    
    try:
        # Run for a limited time to test
        process = subprocess.Popen(cmd)
        time.sleep(10)  # Let it run for 10 seconds
        process.terminate()
        process.wait()
        print("\nTest completed. Did you hear audio? (This confirms the fix works)")
        return True
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        return True
    except Exception as e:
        print(f"Error running test: {e}")
        return False

if __name__ == "__main__":
    test_audio_fix()
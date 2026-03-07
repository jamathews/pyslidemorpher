#!/usr/bin/env python3
"""
Test script to reproduce the audio issue in realtime mode.
This script demonstrates that --audio is ignored in realtime mode.
"""

import subprocess
import sys
from pathlib import Path

def test_audio_issue():
    """Test that audio doesn't play in realtime mode."""
    
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
    
    # Test command that should play audio but doesn't
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
    print("\nThis should play audio but won't because audio is not implemented in realtime mode.")
    print("Press Ctrl+C to stop the slideshow when you confirm no audio is playing.")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nTest completed. Audio was not heard as expected (bug confirmed).")
        return True
    
    return True

if __name__ == "__main__":
    test_audio_issue()
#!/usr/bin/env python3
"""
Test script for enhanced audio reactivity.
Run this to test the improved settings with the specified assets.
"""

import subprocess
import sys
import time

def main():
    print("Testing Enhanced Audio Reactivity Settings")
    print("=" * 50)
    
    # Enhanced command
    cmd = [
        sys.executable, "-m", "pyslidemorpher",
        "/Users/jmathews/src/github.com/jamathews/pyslidemorpher/assets/images",
        "--realtime",
        "--reactive",
        "--audio", "/Users/jmathews/src/github.com/jamathews/pyslidemorpher/assets/audio/Fragment 0007.mp3",
        "--audio-threshold", "0.04",
        "--web-gui",
        "--fps", "30",
        "--seconds-per-transition", "1.2",
        "--hold", "0.3",
        "--pixel-size", "3",
        "--transition", "random",
        "--log-level", "INFO"
    ]
    
    print("Running enhanced reactive slidemorpher...")
    print("Command:", " ".join(cmd))
    print()
    print("Instructions:")
    print("1. Open http://localhost:5001 in your browser")
    print("2. Adjust the following settings in the web GUI:")
    print("   - Beat Sensitivity: 0.45")
    print("   - Peak Sensitivity: 0.15") 
    print("   - Intensity Sensitivity: 0.08")
    print("   - Speed Modulation Range: 2.5")
    print("   - Pixel Size Modulation Range: 0.7")
    print("   - Brightness Modulation Range: 0.15")
    print("   - Enable Show Audio Debug")
    print("3. Observe the improved reactivity!")
    print("4. Press Ctrl+C to stop")
    print()
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nTest completed!")

if __name__ == "__main__":
    main()

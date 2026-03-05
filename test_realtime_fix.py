#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path
import time

def test_realtime_fix():
    """Test that realtime mode now produces logs at WARNING level."""
    
    # Check if demo images exist
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print("Error: demo_images directory not found")
        return False
    
    images = list(demo_dir.glob("*.jpg"))
    if len(images) < 2:
        print("Error: Need at least 2 demo images")
        return False
    
    print("Testing that realtime mode fix produces logs at WARNING level...")
    print("-" * 60)
    
    # Test WARNING level - should now show logs
    cmd = [
        sys.executable, "-m", "pyslidemorpher",
        str(demo_dir),
        "--realtime",
        "--log-level", "WARNING",
        "--fps", "10",
        "--seconds-per-transition", "2",
        "--size", "640x480"
    ]
    
    try:
        # Start the process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a short time to capture initial logs
        time.sleep(2)
        
        # Terminate the process
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        
        print("STDOUT:")
        print(stdout if stdout else "(empty)")
        print("\nSTDERR:")
        print(stderr if stderr else "(empty)")
        
        # Check for expected WARNING level logs
        output = stdout + stderr
        expected_warnings = [
            "Starting realtime slideshow",
            "Starting realtime playback"
        ]
        
        found_warnings = []
        for warning in expected_warnings:
            if warning in output:
                found_warnings.append(warning)
        
        print(f"\nExpected warnings: {len(expected_warnings)}")
        print(f"Found warnings: {len(found_warnings)}")
        
        if len(found_warnings) == len(expected_warnings):
            print("✅ SUCCESS: All expected WARNING level logs found!")
            print("✅ ISSUE FIXED: Realtime mode now has logs even at WARNING level")
            return True
        else:
            print("❌ FAILURE: Some expected WARNING level logs missing")
            missing = [w for w in expected_warnings if w not in output]
            print(f"Missing: {missing}")
            return False
            
    except subprocess.TimeoutExpired:
        print("Process communication timed out")
        return False
    except Exception as e:
        print(f"Error running test: {e}")
        return False

if __name__ == "__main__":
    success = test_realtime_fix()
    if success:
        print("\n🎉 REALTIME LOGGING ISSUE RESOLVED!")
        print("Users will now see important status messages even with WARNING log level.")
    else:
        print("\n❌ Test failed - issue may not be fully resolved")
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test just the position detection functions to see if they work.
"""

import sys
from pathlib import Path

# Add the pyslidemorpher module to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_position_detection():
    """Test the position detection functions directly."""
    print("Testing position detection functions...")
    
    try:
        from pyslidemorpher.realtime import get_actual_window_position
        
        print("Testing get_actual_window_position...")
        x, y = get_actual_window_position("test_window")
        print(f"Result: x={x}, y={y}")
        
        if x is not None and y is not None:
            print("✓ Position detection returned coordinates")
        else:
            print("✗ Position detection returned None")
            
    except Exception as e:
        print(f"✗ Error testing position detection: {e}")
        import traceback
        traceback.print_exc()

def test_applescript_directly():
    """Test AppleScript directly to see if it's working."""
    print("\nTesting AppleScript directly...")
    
    try:
        import subprocess
        
        # Simple AppleScript test
        script = '''
        tell application "System Events"
            return "AppleScript is working"
        end tell
        '''
        
        result = subprocess.run(['osascript', '-e', script], 
                              capture_output=True, text=True, timeout=5)
        
        print(f"AppleScript test result: {result.returncode}")
        print(f"Output: {result.stdout.strip()}")
        print(f"Error: {result.stderr.strip()}")
        
        if result.returncode == 0:
            print("✓ Basic AppleScript works")
        else:
            print("✗ Basic AppleScript failed")
            
    except Exception as e:
        print(f"✗ AppleScript test failed: {e}")

if __name__ == "__main__":
    test_applescript_directly()
    test_position_detection()
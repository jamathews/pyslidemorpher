#!/usr/bin/env python3
"""
Debug test to see why position detection is failing.
"""

import sys
import logging
from pathlib import Path

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

# Add the pyslidemorpher module to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_applescript_detection():
    """Test AppleScript detection directly."""
    print("Testing AppleScript position detection...")
    
    try:
        from pyslidemorpher.realtime import get_window_position_macos
        
        # Test with the actual window name used by the slideshow
        window_name = "PySlidemorpher - Realtime Slideshow"
        print(f"Testing detection for window: '{window_name}'")
        
        x, y = get_window_position_macos(window_name)
        print(f"AppleScript detection result: x={x}, y={y}")
        
        if x is not None and y is not None:
            print("✓ AppleScript detection worked!")
            return True
        else:
            print("✗ AppleScript detection failed")
            return False
            
    except Exception as e:
        print(f"✗ Error testing AppleScript: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generic_detection():
    """Test the generic get_actual_window_position function."""
    print("\nTesting generic position detection...")
    
    try:
        from pyslidemorpher.realtime import get_actual_window_position
        
        window_name = "PySlidemorpher - Realtime Slideshow"
        print(f"Testing detection for window: '{window_name}'")
        
        x, y = get_actual_window_position(window_name)
        print(f"Generic detection result: x={x}, y={y}")
        
        if x is not None and y is not None:
            print("✓ Generic detection worked!")
            return True
        else:
            print("✗ Generic detection failed")
            return False
            
    except Exception as e:
        print(f"✗ Error testing generic detection: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_applescript_manually():
    """Test AppleScript manually to see what's happening."""
    print("\nTesting AppleScript manually...")
    
    try:
        import subprocess
        
        # Test basic AppleScript functionality
        script1 = '''
        tell application "System Events"
            return count of (every process whose name contains "python")
        end tell
        '''
        
        print("Testing basic AppleScript...")
        result = subprocess.run(['osascript', '-e', script1], 
                              capture_output=True, text=True, timeout=5)
        print(f"Python processes found: {result.stdout.strip()}")
        
        # Test window detection
        script2 = '''
        tell application "System Events"
            set windowList to {}
            repeat with proc in (every process whose name contains "python")
                try
                    repeat with win in every window of proc
                        set end of windowList to (name of win)
                    end repeat
                end try
            end repeat
            return windowList
        end tell
        '''
        
        print("Testing window detection...")
        result = subprocess.run(['osascript', '-e', script2], 
                              capture_output=True, text=True, timeout=5)
        print(f"Windows found: {result.stdout.strip()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Manual AppleScript test failed: {e}")
        return False

if __name__ == "__main__":
    print("POSITION DETECTION DEBUG TEST")
    print("=" * 50)
    
    success1 = test_applescript_detection()
    success2 = test_generic_detection()
    success3 = test_applescript_manually()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"AppleScript detection: {'✓' if success1 else '✗'}")
    print(f"Generic detection: {'✓' if success2 else '✗'}")
    print(f"Manual AppleScript: {'✓' if success3 else '✗'}")
    
    if not any([success1, success2]):
        print("\nThe position detection is not working.")
        print("This explains why the same position is always saved.")
        print("The system falls back to tracked position instead of actual position.")
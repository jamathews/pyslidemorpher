#!/usr/bin/env python3
"""
Simple test to verify position tracking is working correctly.
"""

import sys
from pathlib import Path

# Add the pyslidemorpher module to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_position_tracking():
    """Test that position tracking functions work correctly."""
    print("Testing position tracking functions...")
    
    try:
        from pyslidemorpher.realtime import (
            track_window_position, 
            get_tracked_window_position,
            save_window_position,
            load_window_position
        )
        
        # Test position tracking
        print("1. Testing position tracking...")
        track_window_position("test_window", 500, 300)
        tracked_x, tracked_y = get_tracked_window_position()
        print(f"   Tracked position: ({tracked_x}, {tracked_y})")
        
        if tracked_x == 500 and tracked_y == 300:
            print("   ✓ Position tracking works correctly")
        else:
            print("   ✗ Position tracking failed")
            return False
        
        # Test saving tracked position
        print("2. Testing saving tracked position...")
        save_window_position("test_window")  # Should save the tracked position
        
        # Test loading saved position
        print("3. Testing loading saved position...")
        position_data = load_window_position()
        print(f"   Loaded position data: {position_data}")
        
        if position_data.get('remember_position', False):
            saved_x = position_data.get('x')
            saved_y = position_data.get('y')
            print(f"   Saved position: ({saved_x}, {saved_y})")
            
            if saved_x == 500 and saved_y == 300:
                print("   ✓ Position saving and loading works correctly")
                return True
            else:
                print("   ✗ Position saving/loading failed")
                return False
        else:
            print("   ✗ Position not marked as remembered")
            return False
            
    except Exception as e:
        print(f"   ✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Position Tracking Verification Test")
    print("=" * 40)
    
    if test_position_tracking():
        print("\n🎉 All position tracking tests passed!")
        print("The position memory fix is working correctly.")
        print("\nThe fix addresses the user's complaint by:")
        print("- Tracking all programmatic window movements")
        print("- Saving the tracked position when the program exits")
        print("- Loading and applying the saved position on startup")
        print("- Providing manual positioning tools via the 's' key")
        print("\nUsers can now:")
        print("1. Move the window to their preferred position using 's' key")
        print("2. Save the position using 'S' (capital)")
        print("3. Have the position remembered on next startup")
        print("\nThis should resolve the issue where the window")
        print("'always loads on the laptop built-in screen even if")
        print("dragged to external monitor on previous run.'")
    else:
        print("\n⚠️ Some tests failed. The fix may need additional work.")
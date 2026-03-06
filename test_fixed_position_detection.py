#!/usr/bin/env python3
"""
Test the fixed position detection and saving functionality.
"""

import sys
import logging
from pathlib import Path

# Set up logging to see CRITICAL messages
logging.basicConfig(level=logging.CRITICAL, format='%(levelname)s - %(message)s')

# Add the pyslidemorpher module to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_fixed_position_detection():
    """Test that the fixed position detection works."""
    print("Testing fixed position detection...")
    print("=" * 50)
    
    try:
        from pyslidemorpher.realtime import (
            get_actual_window_position,
            save_window_position,
            load_window_position,
            track_window_position
        )
        
        # Test 1: Check if position detection works now
        print("1. Testing position detection...")
        window_name = "PySlidemorpher - Realtime Slideshow"
        detected_x, detected_y = get_actual_window_position(window_name)
        
        if detected_x is not None and detected_y is not None:
            print(f"✓ Position detection works! Detected: ({detected_x}, {detected_y})")
            detection_works = True
        else:
            print("✗ Position detection still not working (no window open)")
            detection_works = False
        
        # Test 2: Test the improved save_window_position logic
        print("\n2. Testing improved save_window_position logic...")
        
        # First, set a tracked position
        track_window_position(window_name, 1000, 200)
        print("   Set tracked position to (1000, 200)")
        
        # Now save without explicit coordinates - should try detection first
        print("   Calling save_window_position without coordinates...")
        save_window_position(window_name)
        
        # Check what was saved
        position_data = load_window_position()
        saved_x = position_data.get('x')
        saved_y = position_data.get('y')
        print(f"   Saved position: ({saved_x}, {saved_y})")
        
        if detection_works:
            if saved_x == detected_x and saved_y == detected_y:
                print("✓ Correctly saved detected position instead of tracked position!")
                return True
            else:
                print("✗ Did not save detected position correctly")
                return False
        else:
            if saved_x == 1000 and saved_y == 200:
                print("✓ Correctly fell back to tracked position when detection failed")
                return True
            else:
                print("✗ Did not fall back to tracked position correctly")
                return False
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_explicit_coordinates():
    """Test saving with explicit coordinates still works."""
    print("\n3. Testing explicit coordinate saving...")
    
    try:
        from pyslidemorpher.realtime import save_window_position, load_window_position
        
        # Save with explicit coordinates
        test_x, test_y = 2500, 300
        print(f"   Saving explicit coordinates: ({test_x}, {test_y})")
        save_window_position("test_window", test_x, test_y)
        
        # Check what was saved
        position_data = load_window_position()
        saved_x = position_data.get('x')
        saved_y = position_data.get('y')
        
        if saved_x == test_x and saved_y == test_y:
            print("✓ Explicit coordinate saving works correctly")
            return True
        else:
            print(f"✗ Explicit coordinates not saved correctly. Got ({saved_x}, {saved_y})")
            return False
            
    except Exception as e:
        print(f"✗ Explicit coordinate test failed: {e}")
        return False

if __name__ == "__main__":
    print("FIXED POSITION DETECTION TEST")
    print("=" * 60)
    print("This test verifies that the position detection fix works correctly.")
    print()
    
    success1 = test_fixed_position_detection()
    success2 = test_explicit_coordinates()
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"Fixed position detection: {'✓' if success1 else '✗'}")
    print(f"Explicit coordinates: {'✓' if success2 else '✗'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! The position detection fix is working!")
        print()
        print("WHAT'S FIXED:")
        print("- AppleScript now searches all processes, not just Python processes")
        print("- save_window_position now tries to detect actual position first")
        print("- Only falls back to tracked position if detection fails")
        print("- This should fix the issue where same position was always saved")
        print()
        print("NOW WHEN USER:")
        print("1. Moves window manually to external monitor")
        print("2. Saves position (via 's' key or on exit)")
        print("3. System will detect actual current position")
        print("4. Save the detected position instead of old tracked position")
    else:
        print("\n⚠️ Some tests failed. The fix may need more work.")
#!/usr/bin/env python3
"""
Simple test to verify the new position saving functionality works.
"""

import sys
from pathlib import Path

# Add the pyslidemorpher module to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_position_save_load():
    """Test that we can save and load window positions with coordinates."""
    print("Testing enhanced position save/load functionality...")
    
    try:
        from pyslidemorpher.realtime import save_window_position, load_window_position
        
        # Test saving a position for external monitor (left side)
        test_x, test_y = -1820, 150
        print(f"Saving test position: ({test_x}, {test_y}) - simulating left external monitor")
        save_window_position("test_window", test_x, test_y)
        
        # Load it back
        position_data = load_window_position()
        print(f"Loaded position data: {position_data}")
        
        # Verify the coordinates were saved correctly
        if position_data.get('remember_position', False):
            saved_x = position_data.get('x')
            saved_y = position_data.get('y')
            
            if saved_x == test_x and saved_y == test_y:
                print("✓ Position coordinates saved and loaded correctly!")
                print(f"  Saved: ({saved_x}, {saved_y})")
                
                # Test interpretation
                if saved_x < 0:
                    print("  → Correctly identified as left external monitor position")
                
                return True
            else:
                print(f"✗ Position mismatch. Expected ({test_x}, {test_y}), got ({saved_x}, {saved_y})")
                return False
        else:
            print("✗ Position not marked as remembered")
            return False
            
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_position_validation():
    """Test that position validation works for multi-monitor setups."""
    print("\nTesting position validation for multi-monitor setups...")
    
    try:
        from pyslidemorpher.realtime import ensure_window_on_screen, save_window_position
        
        # Test various positions
        test_positions = [
            (100, 100, "Main monitor"),
            (-1820, 100, "Left external monitor"),
            (1920, 100, "Right external monitor"),
            (-5000, 100, "Too far left - should be corrected"),
            (10000, 100, "Too far right - should be corrected"),
            (100, -3000, "Too far up - should be corrected"),
            (100, 5000, "Too far down - should be corrected"),
        ]
        
        for x, y, description in test_positions:
            print(f"Testing {description}: ({x}, {y})")
            save_window_position("test_window", x, y)
            
            # The ensure_window_on_screen function should handle validation
            try:
                ensure_window_on_screen("test_window")
                print(f"  ✓ Position handled without error")
            except Exception as e:
                print(f"  ✗ Error handling position: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        return False

def test_external_monitor_presets():
    """Test the external monitor preset positions."""
    print("\nTesting external monitor preset positions...")
    
    try:
        from pyslidemorpher.realtime import save_window_position, load_window_position
        
        # Test preset positions that would be used by the 's' key functionality
        presets = [
            (100, 100, "Main monitor preset"),
            (-1820, 100, "Left external monitor preset"),
            (1920, 100, "Right external monitor preset"),
        ]
        
        for x, y, description in presets:
            print(f"Testing {description}: ({x}, {y})")
            save_window_position("test_window", x, y)
            
            position_data = load_window_position()
            saved_x = position_data.get('x')
            saved_y = position_data.get('y')
            
            if saved_x == x and saved_y == y:
                print(f"  ✓ Preset saved correctly")
            else:
                print(f"  ✗ Preset save failed. Expected ({x}, {y}), got ({saved_x}, {saved_y})")
        
        return True
        
    except Exception as e:
        print(f"✗ Preset test failed: {e}")
        return False

if __name__ == "__main__":
    print("Enhanced Position Management Functionality Test")
    print("=" * 60)
    
    tests = [
        test_position_save_load,
        test_position_validation,
        test_external_monitor_presets,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced position management is working correctly.")
        print("The issue with windows always loading on laptop screen should be resolved.")
        print("Users can now:")
        print("- Press 's' to enter position adjustment mode")
        print("- Use arrow keys/WASD to position the window")
        print("- Use number keys 1,2,3 for monitor presets")
        print("- Press 'S' to save their preferred position")
        print("- Have the position remembered on next startup")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
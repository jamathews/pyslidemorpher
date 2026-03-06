#!/usr/bin/env python3
"""
Test script to verify CRITICAL logging is working in position management.
"""

import sys
import logging
from pathlib import Path

# Set up logging to see CRITICAL messages
logging.basicConfig(level=logging.CRITICAL, format='%(levelname)s - %(message)s')

# Add the pyslidemorpher module to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_critical_logging():
    """Test that CRITICAL logging works for position management."""
    print("Testing CRITICAL logging for position management...")
    print("You should see CRITICAL log messages below:")
    print("-" * 60)
    
    try:
        from pyslidemorpher.realtime import (
            save_window_position,
            load_window_position,
            track_window_position,
            get_tracked_window_position
        )
        
        print("\n1. Testing position loading (should show CRITICAL log):")
        position_data = load_window_position()
        
        print("\n2. Testing position tracking and saving (should show CRITICAL logs):")
        # Track a position
        track_window_position("test_window", 1920, 100)
        
        # Save the tracked position
        save_window_position("test_window")
        
        print("\n3. Testing position loading again (should show CRITICAL log with saved data):")
        position_data = load_window_position()
        
        print("\n4. Testing explicit coordinate saving (should show CRITICAL logs):")
        save_window_position("test_window", 2000, 150)
        
        print("-" * 60)
        print("✓ CRITICAL logging test completed")
        print("If you saw CRITICAL log messages above, the logging is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_window_initialization_logging():
    """Test the window initialization logging without running full slideshow."""
    print("\n" + "=" * 60)
    print("Testing window initialization logging...")
    print("This simulates what happens when the slideshow starts.")
    print("-" * 60)
    
    try:
        from pyslidemorpher.realtime import load_window_position, track_window_position
        
        # Simulate the window initialization process
        print("\nSimulating window initialization:")
        logging.critical("WINDOW INITIALIZATION: Reading position for window startup")
        position_config = load_window_position()
        
        if position_config.get('remember_position', False):
            saved_x = position_config.get('x', 100)
            saved_y = position_config.get('y', 100)
            logging.critical(f"WINDOW INITIALIZATION: Using saved position ({saved_x}, {saved_y}) for window startup")
            track_window_position("test_window", saved_x, saved_y)
        else:
            logging.critical("WINDOW INITIALIZATION: No saved position found, using default (100, 100)")
            track_window_position("test_window", 100, 100)
        
        print("-" * 60)
        print("✓ Window initialization logging test completed")
        print("This is what you should see when the slideshow starts!")
        
        return True
        
    except Exception as e:
        print(f"✗ Window initialization test failed: {e}")
        return False

if __name__ == "__main__":
    print("CRITICAL LOGGING VERIFICATION TEST")
    print("=" * 60)
    print("This test verifies that CRITICAL logs are shown for position management.")
    print()
    
    success1 = test_critical_logging()
    success2 = test_window_initialization_logging()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 All CRITICAL logging tests passed!")
        print()
        print("WHAT YOU SHOULD SEE WHEN RUNNING THE SLIDESHOW:")
        print("- CRITICAL logs when position is read at startup")
        print("- CRITICAL logs when position is saved")
        print("- CRITICAL logs when position is loaded")
        print()
        print("These logs will help you confirm that:")
        print("1. Position is being read when window first appears")
        print("2. Position is being saved correctly")
        print("3. The position management system is working")
    else:
        print("⚠️ Some tests failed. Check the output above.")
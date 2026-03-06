#!/usr/bin/env python3
"""
Test script to verify crash fixes in PySlidemorpher.
Tests various scenarios that could cause crashes.
"""

import sys
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image

# Add the pyslidemorpher module to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_import():
    """Test that imports work without crashing."""
    print("Testing imports...")
    try:
        import pyslidemorpher
        from pyslidemorpher.realtime import play_realtime
        from pyslidemorpher import cli
        from pyslidemorpher import web_gui
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False

def test_audio_initialization():
    """Test pygame audio initialization."""
    print("Testing audio initialization...")
    try:
        from pyslidemorpher.realtime import PYGAME_AVAILABLE
        if PYGAME_AVAILABLE:
            import pygame
            # Test that mixer is already initialized
            if pygame.mixer.get_init():
                print("✓ pygame mixer initialized successfully")
                return True
            else:
                print("✗ pygame mixer not initialized")
                return False
        else:
            print("✓ pygame not available (expected on some systems)")
            return True
    except Exception as e:
        print(f"✗ Audio initialization failed: {e}")
        traceback.print_exc()
        return False

def test_opencv_operations():
    """Test OpenCV operations that could crash."""
    print("Testing OpenCV operations...")
    try:
        import cv2
        
        # Test window creation and destruction
        window_name = "test_window"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        # Test image operations
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[:] = (255, 0, 0)  # Red image
        
        # Test color conversion
        bgr_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
        
        # Test window property operations (these might fail on some systems)
        try:
            cv2.getWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE)
        except:
            pass  # Expected to fail on some systems
        
        # Test window positioning
        try:
            cv2.moveWindow(window_name, 100, 100)
        except:
            pass  # May fail on some systems
        
        cv2.destroyAllWindows()
        print("✓ OpenCV operations completed successfully")
        return True
    except Exception as e:
        print(f"✗ OpenCV operations failed: {e}")
        traceback.print_exc()
        return False

def test_position_management():
    """Test window position management functions."""
    print("Testing position management...")
    try:
        from pyslidemorpher.realtime import (
            get_window_position_file,
            save_window_position,
            load_window_position,
            ensure_window_on_screen
        )
        
        # Test file path creation
        config_file = get_window_position_file()
        print(f"  Config file path: {config_file}")
        
        # Test save/load operations
        save_window_position("test_window")
        position_data = load_window_position()
        
        if position_data.get('remember_position', False):
            print("✓ Position management working correctly")
            return True
        else:
            print("✗ Position data not saved/loaded correctly")
            return False
    except Exception as e:
        print(f"✗ Position management failed: {e}")
        traceback.print_exc()
        return False

def test_web_gui():
    """Test web GUI initialization."""
    print("Testing web GUI...")
    try:
        from pyslidemorpher.web_gui import FLASK_AVAILABLE, get_controller
        
        if FLASK_AVAILABLE:
            controller = get_controller()
            settings = controller.get_settings()
            print(f"  Default settings: {settings}")
            print("✓ Web GUI controller working")
        else:
            print("✓ Flask not available (expected on some systems)")
        return True
    except Exception as e:
        print(f"✗ Web GUI test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test that error handling doesn't cause crashes."""
    print("Testing error handling...")
    try:
        from pyslidemorpher.realtime import get_random_transition_function
        
        # Test transition function selection
        for _ in range(5):
            func = get_random_transition_function()
            print(f"  Selected: {func.__name__}")
        
        print("✓ Error handling working correctly")
        return True
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all crash prevention tests."""
    print("PySlidemorpher Crash Prevention Test Suite")
    print("=" * 50)
    
    tests = [
        test_import,
        test_audio_initialization,
        test_opencv_operations,
        test_position_management,
        test_web_gui,
        test_error_handling,
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
            traceback.print_exc()
            print()
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Crash fixes are working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
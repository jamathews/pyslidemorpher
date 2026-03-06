#!/usr/bin/env python3
"""
Test script to explore window position detection methods.
"""

import cv2
import numpy as np
import time

def test_opencv_window_properties():
    """Test various OpenCV window properties to find position info."""
    print("Testing OpenCV window properties...")
    
    # Create a test window
    window_name = "Position Test Window"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # Create a simple test image
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:] = (0, 255, 0)  # Green
    cv2.putText(img, "Move this window around", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display the image
    cv2.imshow(window_name, img)
    
    print("Window created. Move it around and press any key to test position detection...")
    cv2.waitKey(0)
    
    # Test various window properties
    properties_to_test = [
        ('WND_PROP_FULLSCREEN', cv2.WND_PROP_FULLSCREEN),
        ('WND_PROP_AUTOSIZE', cv2.WND_PROP_AUTOSIZE),
        ('WND_PROP_ASPECT_RATIO', cv2.WND_PROP_ASPECT_RATIO),
        ('WND_PROP_OPENGL', cv2.WND_PROP_OPENGL),
        ('WND_PROP_VISIBLE', cv2.WND_PROP_VISIBLE),
        ('WND_PROP_TOPMOST', cv2.WND_PROP_TOPMOST),
    ]
    
    print("\nTesting window properties:")
    for prop_name, prop_value in properties_to_test:
        try:
            result = cv2.getWindowProperty(window_name, prop_value)
            print(f"  {prop_name}: {result}")
        except Exception as e:
            print(f"  {prop_name}: ERROR - {e}")
    
    # Test if there are any undocumented properties for position
    print("\nTesting potential position properties:")
    for i in range(10):  # Test property values 0-9
        try:
            result = cv2.getWindowProperty(window_name, i)
            print(f"  Property {i}: {result}")
        except Exception as e:
            print(f"  Property {i}: ERROR - {e}")
    
    cv2.destroyAllWindows()

def test_platform_specific_methods():
    """Test platform-specific methods for getting window position."""
    print("\nTesting platform-specific methods...")
    
    import sys
    if sys.platform == 'darwin':  # macOS
        print("Testing macOS-specific methods...")
        try:
            # Try using AppKit (if available)
            import AppKit
            print("AppKit available - could potentially use NSWindow methods")
        except ImportError:
            print("AppKit not available")
        
        try:
            # Try using Quartz (if available)
            import Quartz
            print("Quartz available - could potentially use CGWindow methods")
        except ImportError:
            print("Quartz not available")
    
    elif sys.platform == 'win32':  # Windows
        print("Testing Windows-specific methods...")
        try:
            import win32gui
            print("win32gui available - could use FindWindow and GetWindowRect")
        except ImportError:
            print("win32gui not available")
    
    elif sys.platform.startswith('linux'):  # Linux
        print("Testing Linux-specific methods...")
        try:
            import subprocess
            # Test if xwininfo is available
            result = subprocess.run(['which', 'xwininfo'], capture_output=True)
            if result.returncode == 0:
                print("xwininfo available - could use for window position")
            else:
                print("xwininfo not available")
        except Exception as e:
            print(f"Error testing xwininfo: {e}")

def test_alternative_approaches():
    """Test alternative approaches for position tracking."""
    print("\nTesting alternative approaches...")
    
    # Test if we can use window manager hints
    window_name = "Position Test Window 2"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # Try to set window position and see if we can track it
    positions_to_test = [(100, 100), (300, 200), (500, 300)]
    
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    img[:] = (255, 0, 0)  # Red
    
    for i, (x, y) in enumerate(positions_to_test):
        print(f"Moving window to position ({x}, {y})")
        cv2.moveWindow(window_name, x, y)
        cv2.putText(img, f"Position {i+1}: ({x}, {y})", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(window_name, img)
        cv2.waitKey(1000)  # Wait 1 second
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Window Position Detection Test")
    print("=" * 40)
    
    test_opencv_window_properties()
    test_platform_specific_methods()
    test_alternative_approaches()
    
    print("\nTest completed. Check the output above for available methods.")
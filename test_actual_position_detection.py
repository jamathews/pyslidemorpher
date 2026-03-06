#!/usr/bin/env python3
"""
Test script to verify that actual window position detection is working.
"""

import sys
import time
from pathlib import Path
import numpy as np
import cv2

# Add the pyslidemorpher module to the path
sys.path.insert(0, str(Path(__file__).parent))

from pyslidemorpher.realtime import get_actual_window_position, save_window_position, load_window_position

def test_position_detection():
    """Test the actual window position detection."""
    print("Testing actual window position detection...")
    print("This test will:")
    print("1. Create a test window")
    print("2. Move it to different positions")
    print("3. Try to detect the actual position")
    print("4. Save and verify the position")
    print()

    # Create a test window
    window_name = "Position Detection Test"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # Create a simple test image
    img = np.zeros((300, 500, 3), dtype=np.uint8)
    img[:] = (0, 255, 0)  # Green
    cv2.putText(img, "Position Detection Test", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "Move this window around", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, "Press 't' to test position detection", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "Press 's' to save position", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "Press 'q' to quit", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Test different positions
    test_positions = [
        (100, 100, "Main monitor top-left"),
        (500, 200, "Main monitor center"),
        (-1820, 100, "Left external monitor (if available)"),
        (1920, 100, "Right external monitor (if available)"),
    ]
    
    print("Testing programmatic positioning and detection...")
    for i, (x, y, description) in enumerate(test_positions):
        print(f"\nTest {i+1}: {description}")
        print(f"Moving window to ({x}, {y})")
        
        try:
            cv2.moveWindow(window_name, x, y)
            cv2.imshow(window_name, img)
            cv2.waitKey(500)  # Wait for window to move
            
            # Try to detect the actual position
            detected_x, detected_y = get_actual_window_position(window_name)
            if detected_x is not None and detected_y is not None:
                print(f"✓ Detected position: ({detected_x}, {detected_y})")
                
                # Check if detection is reasonably close to expected
                if abs(detected_x - x) < 50 and abs(detected_y - y) < 50:
                    print("✓ Detection appears accurate")
                else:
                    print(f"⚠ Detection differs from expected by ({detected_x - x}, {detected_y - y})")
            else:
                print("✗ Could not detect window position")
                
        except Exception as e:
            print(f"✗ Error testing position ({x}, {y}): {e}")
    
    # Interactive testing
    print("\n" + "="*60)
    print("Interactive Testing:")
    print("Now you can manually move the window around and test position detection.")
    print("Controls:")
    print("- 't': Test current position detection")
    print("- 's': Save current position")
    print("- 'l': Load and show saved position")
    print("- 'q': Quit")
    print("="*60)
    
    cv2.imshow(window_name, img)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('t'):
            print("\nTesting current position detection...")
            detected_x, detected_y = get_actual_window_position(window_name)
            if detected_x is not None and detected_y is not None:
                print(f"✓ Current window position: ({detected_x}, {detected_y})")
                
                # Provide interpretation
                if detected_x < 0:
                    print("  → Window appears to be on left external monitor")
                elif detected_x > 1500:
                    print("  → Window appears to be on right external monitor")
                else:
                    print("  → Window appears to be on main monitor")
            else:
                print("✗ Could not detect current window position")
                
        elif key == ord('s'):
            print("\nSaving current window position...")
            save_window_position(window_name)
            print("✓ Position saved")
            
        elif key == ord('l'):
            print("\nLoading saved position...")
            position_data = load_window_position()
            print(f"Saved position data: {position_data}")
            
            if position_data.get('remember_position', False):
                x = position_data.get('x', 'Not set')
                y = position_data.get('y', 'Not set')
                print(f"Saved position: ({x}, {y})")
            else:
                print("No position saved")
    
    cv2.destroyAllWindows()
    print("\nPosition detection test completed.")

if __name__ == "__main__":
    test_position_detection()
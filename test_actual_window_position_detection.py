#!/usr/bin/env python3
"""
Test to verify position detection works when an actual slideshow window is open.
This addresses the issue: "make sure it ALWAYS determines the exact position of the open window"
"""

import sys
import time
import threading
from pathlib import Path
import numpy as np
import cv2

# Add the pyslidemorpher module to the path
sys.path.insert(0, str(Path(__file__).parent))

def create_simple_test_image():
    """Create a simple test image for the window."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:] = (0, 150, 255)  # Orange background
    
    # Add text instructions
    cv2.putText(img, "POSITION DETECTION TEST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "This window is open for position detection", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "Move this window around", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "Press 't' to test position detection", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "Press 's' to save position", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "Press 'q' to quit", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return img

def test_position_detection_with_open_window():
    """Test position detection with an actual open slideshow window."""
    print("TESTING POSITION DETECTION WITH ACTUAL OPEN WINDOW")
    print("=" * 60)
    print("This test will:")
    print("1. Open a real PySlidemorpher window")
    print("2. Test position detection while the window is open")
    print("3. Show whether detection works or fails")
    print("4. Allow you to move the window and test detection")
    print()
    
    try:
        from pyslidemorpher.realtime import (
            get_actual_window_position,
            save_window_position,
            load_window_position,
            track_window_position
        )
        
        # Create the exact same window name as used in the slideshow
        window_name = "PySlidemorpher - Realtime Slideshow"
        
        print(f"Creating window: '{window_name}'")
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        # Create and display test image
        test_img = create_simple_test_image()
        cv2.imshow(window_name, test_img)
        
        # Move window to a known position for testing
        initial_x, initial_y = 200, 150
        cv2.moveWindow(window_name, initial_x, initial_y)
        track_window_position(window_name, initial_x, initial_y)
        
        print(f"Window created and moved to initial position: ({initial_x}, {initial_y})")
        print("Waiting 2 seconds for window to stabilize...")
        cv2.waitKey(2000)  # Wait 2 seconds for window to stabilize
        
        # Test initial position detection
        print("\nTesting initial position detection...")
        detected_x, detected_y = get_actual_window_position(window_name)
        
        if detected_x is not None and detected_y is not None:
            print(f"✓ SUCCESS: Detected position: ({detected_x}, {detected_y})")
            print(f"  Expected position: ({initial_x}, {initial_y})")
            
            # Check if detection is reasonably accurate
            x_diff = abs(detected_x - initial_x)
            y_diff = abs(detected_y - initial_y)
            
            if x_diff <= 10 and y_diff <= 10:
                print("✓ Detection is accurate (within 10 pixels)")
            else:
                print(f"⚠ Detection differs by ({x_diff}, {y_diff}) pixels")
        else:
            print("✗ FAILED: Could not detect window position")
            print("This means the position detection system is not working properly")
        
        # Interactive testing
        print("\n" + "="*60)
        print("INTERACTIVE TESTING:")
        print("The window is now open. You can:")
        print("- Move the window around manually")
        print("- Press 't' to test position detection")
        print("- Press 's' to save current position (will test detection)")
        print("- Press 'q' to quit")
        print("="*60)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('t'):
                print("\nTesting current position detection...")
                detected_x, detected_y = get_actual_window_position(window_name)
                
                if detected_x is not None and detected_y is not None:
                    print(f"✓ Current detected position: ({detected_x}, {detected_y})")
                    
                    # Provide interpretation
                    if detected_x < 0:
                        print("  → Window appears to be on left external monitor")
                    elif detected_x > 1500:
                        print("  → Window appears to be on right external monitor")
                    else:
                        print("  → Window appears to be on main monitor")
                else:
                    print("✗ Could not detect current window position")
                    print("  → This indicates the position detection system is failing")
                    
            elif key == ord('s'):
                print("\nTesting position saving (which uses detection)...")
                save_window_position(window_name)
                
                # Check what was saved
                position_data = load_window_position()
                if position_data.get('remember_position', False):
                    saved_x = position_data.get('x')
                    saved_y = position_data.get('y')
                    print(f"Position saved: ({saved_x}, {saved_y})")
                    
                    # Check the logs to see if detection or tracking was used
                    print("Check the console output above to see if it says:")
                    print("  'Saving detected window position' (detection worked)")
                    print("  OR 'Saving tracked window position' (detection failed)")
                else:
                    print("✗ Position saving failed")
        
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_position_detection_with_open_window()
    
    print("\n" + "=" * 60)
    if success:
        print("TEST COMPLETED")
        print()
        print("WHAT THIS TEST SHOWS:")
        print("- Whether position detection works when a real window is open")
        print("- If detection fails, the system falls back to tracked position")
        print("- The exact behavior of the position saving system")
        print()
        print("IF DETECTION WORKED:")
        print("- You should see 'Saving detected window position' in logs")
        print("- The system can determine the exact position of open windows")
        print()
        print("IF DETECTION FAILED:")
        print("- You should see 'Saving tracked window position' in logs")
        print("- This means the AppleScript/platform detection is not working")
        print("- The system falls back to the last programmatic position")
    else:
        print("⚠️ Test had issues, but this helps identify the problem")
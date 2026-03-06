import sys
import os
import time
import cv2
import numpy as np
sys.path.insert(0, os.path.abspath('.'))

from pyslidemorpher.realtime import (
    get_tracked_window_position,
    track_window_position,
    get_actual_window_position,
    save_window_position,
    load_window_position,
    ensure_window_on_screen
)

print("Testing realtime position functionality...")

# Create a simple test window
window_name = "PySlidemorpher - Position Test"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

# Create a simple test image
test_image = np.zeros((400, 600, 3), dtype=np.uint8)
test_image[:] = (50, 100, 150)  # Fill with a color
cv2.putText(test_image, "Position Test Window", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(test_image, "Move this window and press 's' to test", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
cv2.putText(test_image, "Press 'q' to quit", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

# Test initial positioning
print("Testing initial positioning...")
position_config = load_window_position()
if position_config.get('remember_position', False):
    saved_x = position_config.get('x', 100)
    saved_y = position_config.get('y', 100)
    print(f"Using saved position: ({saved_x}, {saved_y})")
    ensure_window_on_screen(window_name)
else:
    print("No saved position, using default (100, 100)")
    cv2.moveWindow(window_name, 100, 100)
    track_window_position(window_name, 100, 100)

# Display the window
cv2.imshow(window_name, test_image)

print("Window displayed. You can:")
print("1. Move the window to test position detection")
print("2. Press 's' to save current position")
print("3. Press 'd' to detect current position")
print("4. Press 'q' to quit")

last_position_check = time.time()
position_check_interval = 2.0

try:
    while True:
        # Periodic position checking
        current_time = time.time()
        if current_time - last_position_check > position_check_interval:
            try:
                actual_x, actual_y = get_actual_window_position(window_name)
                if actual_x is not None and actual_y is not None:
                    tracked_x, tracked_y = get_tracked_window_position()
                    if actual_x != tracked_x or actual_y != tracked_y:
                        print(f"Position changed: ({tracked_x}, {tracked_y}) -> ({actual_x}, {actual_y})")
                        track_window_position(window_name, actual_x, actual_y)
                else:
                    print("Could not detect window position")
            except Exception as e:
                print(f"Position detection error: {e}")
            last_position_check = current_time

        # Handle keyboard input
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print("Saving current position...")
            save_window_position(window_name)
            position_data = load_window_position()
            if position_data.get('remember_position', False):
                x = position_data.get('x', 100)
                y = position_data.get('y', 100)
                print(f"Position saved: ({x}, {y})")
            else:
                print("Position save failed")
        elif key == ord('d'):
            print("Detecting current position...")
            actual_x, actual_y = get_actual_window_position(window_name)
            tracked_x, tracked_y = get_tracked_window_position()
            print(f"Detected position: ({actual_x}, {actual_y})")
            print(f"Tracked position: ({tracked_x}, {tracked_y})")

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # Save position on exit
    print("Saving position on exit...")
    save_window_position(window_name)
    cv2.destroyAllWindows()
    print("Test completed")
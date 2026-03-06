import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from pyslidemorpher.realtime import (
    get_tracked_window_position,
    track_window_position,
    get_actual_window_position,
    save_window_position,
    load_window_position
)

print("Testing position tracking...")

# Test initial tracked position
x, y = get_tracked_window_position()
print(f"Initial tracked position: ({x}, {y})")

# Test loading saved position
position_data = load_window_position()
print(f"Loaded position data: {position_data}")

# Test tracking a new position
track_window_position("test", 500, 300)
x, y = get_tracked_window_position()
print(f"After tracking (500, 300): ({x}, {y})")

# Test actual position detection (this might fail without a real window)
try:
    actual_x, actual_y = get_actual_window_position("PySlidemorpher - Realtime Slideshow")
    print(f"Actual position detection: ({actual_x}, {actual_y})")
except Exception as e:
    print(f"Actual position detection failed: {e}")
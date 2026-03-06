import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from pyslidemorpher.realtime import load_window_position, validate_window_position

print('Testing position validation...')
print('Valid position (100, 100):', validate_window_position(100, 100))
print('Valid position (-1000, 200):', validate_window_position(-1000, 200))
print('Invalid position (1500, 200):', validate_window_position(1500, 200))
print('Invalid position (10000, 100):', validate_window_position(10000, 100))

print('\nTesting position loading...')
position_data = load_window_position()
print('Loaded position data:', position_data)
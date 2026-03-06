#!/usr/bin/env python3
"""
Test the position management functions directly.
"""

import sys
from pathlib import Path

# Add the pyslidemorpher module to the path
sys.path.insert(0, str(Path(__file__).parent))

from pyslidemorpher.realtime import (
    get_window_position_file,
    save_window_position,
    load_window_position,
    ensure_window_on_screen
)

def test_position_functions():
    """Test the position management functions."""
    print("Testing position management functions...")
    
    # Test getting the config file path
    config_file = get_window_position_file()
    print(f"Config file path: {config_file}")
    
    # Test saving position (with a dummy window name)
    print("Testing save_window_position...")
    save_window_position("test_window")
    
    # Check if file was created
    if config_file.exists():
        print("✓ Position file created successfully")
        
        # Test loading position
        print("Testing load_window_position...")
        position_data = load_window_position()
        print(f"Loaded position data: {position_data}")
        
        if position_data.get('remember_position', False):
            print("✓ Position data loaded correctly")
        else:
            print("✗ Position data not loaded correctly")
    else:
        print("✗ Position file was not created")
    
    # Test ensure_window_on_screen (this will just test that it doesn't crash)
    print("Testing ensure_window_on_screen...")
    try:
        ensure_window_on_screen("test_window")
        print("✓ ensure_window_on_screen completed without error")
    except Exception as e:
        print(f"✗ ensure_window_on_screen failed: {e}")
    
    print("Position function tests completed.")

if __name__ == "__main__":
    test_position_functions()
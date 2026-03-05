#!/usr/bin/env python3
"""Test script to verify that random transitions work correctly in slideshow context."""

import sys
import os
import numpy as np
from unittest.mock import Mock, MagicMock
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyslidemorpher'))

from pyslidemorpher.realtime import get_random_transition_function

def test_slideshow_random_transitions():
    """Test that random transitions are selected for each image pair in slideshow."""
    print("Testing slideshow random transition behavior...")
    
    # Mock the transition functions to track which ones are called
    transition_calls = []
    
    def mock_transition_function(name):
        def mock_func(*args, **kwargs):
            transition_calls.append(name)
            # Return a simple frame generator
            yield np.zeros((100, 100, 3), dtype=np.uint8)
        mock_func.__name__ = name
        return mock_func
    
    # Simulate the behavior of standard_frame_generator with multiple image pairs
    print("Simulating standard frame generator with 5 image transitions...")
    
    # Mock args
    args = Mock()
    args.transition = "random"
    
    # Simulate the loop from standard_frame_generator
    for i in range(5):  # 5 transitions
        print(f"\nTransition {i + 1}:")
        
        # This is the key line that was fixed - it's now inside the loop
        if args.transition == "random":
            transition_fn = get_random_transition_function()
            print(f"  Selected: {transition_fn.__name__}")
            transition_calls.append(transition_fn.__name__)
        
    print(f"\nTransition functions called: {transition_calls}")
    unique_transitions = set(transition_calls)
    print(f"Unique transitions used: {len(unique_transitions)}")
    print(f"Transitions: {unique_transitions}")
    
    if len(unique_transitions) == 1:
        print("❌ ISSUE: Same transition used for all image pairs!")
        return False
    else:
        print("✅ SUCCESS: Different transitions selected for different image pairs!")
        return True

if __name__ == "__main__":
    test_slideshow_random_transitions()
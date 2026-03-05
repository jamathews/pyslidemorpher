#!/usr/bin/env python3
"""Test script to reproduce the random transition issue."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pyslidemorpher'))

from pyslidemorpher.realtime import get_random_transition_function

def test_random_transition_selection():
    """Test that get_random_transition_function returns different functions."""
    print("Testing random transition selection...")
    
    # Call the function multiple times and collect the results
    functions = []
    for i in range(10):
        func = get_random_transition_function()
        functions.append(func.__name__)
        print(f"Call {i+1}: {func.__name__}")
    
    # Check if we got different functions
    unique_functions = set(functions)
    print(f"\nUnique functions selected: {len(unique_functions)}")
    print(f"Functions: {unique_functions}")
    
    if len(unique_functions) == 1:
        print("❌ ISSUE CONFIRMED: Same function selected every time!")
        return False
    else:
        print("✅ Random selection working correctly")
        return True

if __name__ == "__main__":
    test_random_transition_selection()
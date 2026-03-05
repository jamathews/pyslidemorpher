#!/usr/bin/env python3
"""
Test script to verify that get_random_transition_function() never selects 
the same transition function twice in a row.
"""

import sys
import os

# Add the current directory to the path so we can import pyslidemorpher
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyslidemorpher import get_random_transition_function

def test_no_consecutive_duplicates():
    """Test that get_random_transition_function never returns the same function twice in a row."""
    print("Testing get_random_transition_function() for consecutive duplicates...")
    
    # Reset the function's state by deleting the attribute if it exists
    if hasattr(get_random_transition_function, '_last_selected'):
        delattr(get_random_transition_function, '_last_selected')
    
    # Test with multiple calls
    num_tests = 100
    consecutive_duplicates = 0
    previous_function = None
    
    for i in range(num_tests):
        current_function = get_random_transition_function()
        
        if previous_function is not None and current_function == previous_function:
            consecutive_duplicates += 1
            print(f"ERROR: Consecutive duplicate found at call {i}: {current_function.__name__}")
        
        print(f"Call {i+1}: {current_function.__name__}")
        previous_function = current_function
    
    print(f"\nTest Results:")
    print(f"Total calls: {num_tests}")
    print(f"Consecutive duplicates found: {consecutive_duplicates}")
    
    if consecutive_duplicates == 0:
        print("✅ SUCCESS: No consecutive duplicates found!")
        return True
    else:
        print("❌ FAILURE: Consecutive duplicates were found!")
        return False

def test_function_variety():
    """Test that the function returns different transition functions over multiple calls."""
    print("\nTesting function variety...")
    
    # Reset the function's state
    if hasattr(get_random_transition_function, '_last_selected'):
        delattr(get_random_transition_function, '_last_selected')
    
    # Collect unique functions over many calls
    unique_functions = set()
    num_calls = 50
    
    for i in range(num_calls):
        func = get_random_transition_function()
        unique_functions.add(func.__name__)
    
    print(f"Unique functions found over {num_calls} calls: {len(unique_functions)}")
    print(f"Functions: {sorted(unique_functions)}")
    
    # We should see multiple different functions (at least 2 since we have 7 total)
    if len(unique_functions) >= 2:
        print("✅ SUCCESS: Multiple different functions were selected!")
        return True
    else:
        print("❌ FAILURE: Not enough variety in function selection!")
        return False

if __name__ == "__main__":
    print("Testing the modified get_random_transition_function()...\n")
    
    test1_passed = test_no_consecutive_duplicates()
    test2_passed = test_function_variety()
    
    print(f"\n{'='*50}")
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("💥 SOME TESTS FAILED!")
        sys.exit(1)
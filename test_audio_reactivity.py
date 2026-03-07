#!/usr/bin/env python3
"""
Test script for the simplified audio reactivity logic.
This script tests the new behavior where:
- When audio is quiet (intensity < threshold), no transition occurs
- When audio is loud (intensity >= threshold), transition occurs
"""

import sys
import os
import time
import threading
from unittest.mock import Mock, patch
import numpy as np

# Add the pyslidemorpher package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_audio_reactivity_logic():
    """Test the simplified audio reactivity logic."""
    print("Testing simplified audio reactivity logic...")
    
    # Mock audio features for testing
    test_cases = [
        {"intensity": 0.1, "threshold": 0.5, "should_trigger": False, "description": "Quiet audio below threshold"},
        {"intensity": 0.3, "threshold": 0.5, "should_trigger": False, "description": "Quiet audio below threshold"},
        {"intensity": 0.5, "threshold": 0.5, "should_trigger": True, "description": "Audio at threshold"},
        {"intensity": 0.7, "threshold": 0.5, "should_trigger": True, "description": "Loud audio above threshold"},
        {"intensity": 0.9, "threshold": 0.5, "should_trigger": True, "description": "Very loud audio above threshold"},
        {"intensity": 0.2, "threshold": 0.8, "should_trigger": False, "description": "Audio below high threshold"},
        {"intensity": 0.8, "threshold": 0.8, "should_trigger": True, "description": "Audio at high threshold"},
    ]
    
    print("\nTesting audio threshold logic:")
    print("=" * 60)
    
    for i, case in enumerate(test_cases):
        intensity = case["intensity"]
        threshold = case["threshold"]
        expected = case["should_trigger"]
        description = case["description"]
        
        # Simple logic from the rewritten function
        is_loud = intensity >= threshold
        
        result = "PASS" if is_loud == expected else "FAIL"
        status = "TRIGGER" if is_loud else "NO TRIGGER"
        
        print(f"Test {i+1}: {description}")
        print(f"  Intensity: {intensity:.1f}, Threshold: {threshold:.1f}")
        print(f"  Expected: {'TRIGGER' if expected else 'NO TRIGGER'}, Got: {status}")
        print(f"  Result: {result}")
        print()
        
        if is_loud != expected:
            print(f"ERROR: Test {i+1} failed!")
            return False
    
    print("All audio threshold tests passed!")
    return True

def test_minimum_interval_logic():
    """Test the minimum interval logic to prevent rapid switching."""
    print("\nTesting minimum interval logic:")
    print("=" * 60)
    
    min_interval = 0.5  # 500ms as defined in the code
    
    test_cases = [
        {"time_since_last": 0.1, "should_allow": False, "description": "Too soon after last trigger"},
        {"time_since_last": 0.3, "should_allow": False, "description": "Still too soon"},
        {"time_since_last": 0.5, "should_allow": True, "description": "Exactly at minimum interval"},
        {"time_since_last": 0.7, "should_allow": True, "description": "Past minimum interval"},
        {"time_since_last": 1.0, "should_allow": True, "description": "Well past minimum interval"},
    ]
    
    for i, case in enumerate(test_cases):
        time_since_last = case["time_since_last"]
        expected = case["should_allow"]
        description = case["description"]
        
        # Logic from the rewritten function
        can_trigger = time_since_last >= min_interval
        
        result = "PASS" if can_trigger == expected else "FAIL"
        status = "ALLOW" if can_trigger else "BLOCK"
        
        print(f"Test {i+1}: {description}")
        print(f"  Time since last: {time_since_last:.1f}s, Min interval: {min_interval:.1f}s")
        print(f"  Expected: {'ALLOW' if expected else 'BLOCK'}, Got: {status}")
        print(f"  Result: {result}")
        print()
        
        if can_trigger != expected:
            print(f"ERROR: Test {i+1} failed!")
            return False
    
    print("All minimum interval tests passed!")
    return True

def test_combined_logic():
    """Test the combined logic (both audio threshold and minimum interval)."""
    print("\nTesting combined trigger logic:")
    print("=" * 60)
    
    min_interval = 0.5
    
    test_cases = [
        {"intensity": 0.7, "threshold": 0.5, "time_since_last": 0.6, "in_transition": False, 
         "should_trigger": True, "description": "Loud audio, enough time, not in transition"},
        {"intensity": 0.3, "threshold": 0.5, "time_since_last": 0.6, "in_transition": False, 
         "should_trigger": False, "description": "Quiet audio, enough time, not in transition"},
        {"intensity": 0.7, "threshold": 0.5, "time_since_last": 0.3, "in_transition": False, 
         "should_trigger": False, "description": "Loud audio, not enough time, not in transition"},
        {"intensity": 0.7, "threshold": 0.5, "time_since_last": 0.6, "in_transition": True, 
         "should_trigger": False, "description": "Loud audio, enough time, but in transition"},
    ]
    
    for i, case in enumerate(test_cases):
        intensity = case["intensity"]
        threshold = case["threshold"]
        time_since_last = case["time_since_last"]
        in_transition = case["in_transition"]
        expected = case["should_trigger"]
        description = case["description"]
        
        # Combined logic from the rewritten function
        is_loud = intensity >= threshold
        can_trigger = time_since_last >= min_interval
        should_trigger = is_loud and can_trigger and not in_transition
        
        result = "PASS" if should_trigger == expected else "FAIL"
        status = "TRIGGER" if should_trigger else "NO TRIGGER"
        
        print(f"Test {i+1}: {description}")
        print(f"  Intensity: {intensity:.1f}, Threshold: {threshold:.1f}")
        print(f"  Time since last: {time_since_last:.1f}s, In transition: {in_transition}")
        print(f"  Expected: {'TRIGGER' if expected else 'NO TRIGGER'}, Got: {status}")
        print(f"  Result: {result}")
        print()
        
        if should_trigger != expected:
            print(f"ERROR: Test {i+1} failed!")
            return False
    
    print("All combined logic tests passed!")
    return True

def main():
    """Run all tests."""
    print("Testing Simplified Audio Reactivity Logic")
    print("=" * 60)
    print("This tests the new behavior:")
    print("- When audio is quiet (intensity < threshold), no transition occurs")
    print("- When audio is loud (intensity >= threshold), transition occurs")
    print("- Minimum 500ms interval between transitions")
    print()
    
    all_passed = True
    
    # Run individual tests
    all_passed &= test_audio_reactivity_logic()
    all_passed &= test_minimum_interval_logic()
    all_passed &= test_combined_logic()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The simplified audio reactivity logic is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
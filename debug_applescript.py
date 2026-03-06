#!/usr/bin/env python3
"""
Debug AppleScript for window position detection.
"""

import subprocess
import sys

def test_applescript():
    """Test AppleScript directly."""
    
    # Test basic AppleScript functionality
    print("Testing basic AppleScript...")
    script1 = 'tell application "System Events" to return name of every process'
    
    try:
        result = subprocess.run(['osascript', '-e', script1], 
                              capture_output=True, text=True, timeout=5)
        print(f"Process list result: {result.returncode}")
        if result.stdout:
            print(f"Processes found: {len(result.stdout.split(','))}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")
    except Exception as e:
        print(f"Error testing processes: {e}")
    
    print("\nTesting window detection...")
    script2 = '''
    tell application "System Events"
        set windowList to every window of every process
        return count of windowList
    end tell
    '''
    
    try:
        result = subprocess.run(['osascript', '-e', script2], 
                              capture_output=True, text=True, timeout=5)
        print(f"Window count result: {result.returncode}")
        print(f"Window count: {result.stdout.strip()}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")
    except Exception as e:
        print(f"Error testing windows: {e}")
    
    print("\nTesting Python process detection...")
    script3 = '''
    tell application "System Events"
        set pythonProcesses to every process whose name contains "python"
        return count of pythonProcesses
    end tell
    '''
    
    try:
        result = subprocess.run(['osascript', '-e', script3], 
                              capture_output=True, text=True, timeout=5)
        print(f"Python process result: {result.returncode}")
        print(f"Python processes: {result.stdout.strip()}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")
    except Exception as e:
        print(f"Error testing Python processes: {e}")

if __name__ == "__main__":
    test_applescript()
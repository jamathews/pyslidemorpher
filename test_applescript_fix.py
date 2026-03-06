#!/usr/bin/env python3
"""
Test improved AppleScript for window position detection.
"""

import subprocess
import sys

def test_improved_applescript():
    """Test improved AppleScript that looks for windows more broadly."""
    print("Testing improved AppleScript...")
    
    # First, let's see what processes are actually running
    script1 = '''
    tell application "System Events"
        set processList to {}
        repeat with proc in every process
            try
                if (count of windows of proc) > 0 then
                    set end of processList to (name of proc)
                end if
            end try
        end repeat
        return processList
    end tell
    '''
    
    try:
        print("Finding all processes with windows...")
        result = subprocess.run(['osascript', '-e', script1], 
                              capture_output=True, text=True, timeout=10)
        print(f"Processes with windows: {result.stdout.strip()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Now let's look for any window with "PySlidemorpher" in the name
    script2 = '''
    tell application "System Events"
        set windowFound to false
        set windowX to 0
        set windowY to 0
        set foundProcess to ""
        
        repeat with proc in every process
            try
                repeat with win in every window of proc
                    set winName to name of win
                    if winName contains "PySlidemorpher" then
                        set windowPos to position of win
                        set windowX to item 1 of windowPos
                        set windowY to item 2 of windowPos
                        set windowFound to true
                        set foundProcess to name of proc
                        exit repeat
                    end if
                end repeat
                if windowFound then exit repeat
            end try
        end repeat
        
        if windowFound then
            return "FOUND:" & foundProcess & ":" & (windowX as string) & "," & (windowY as string)
        else
            return "NOT_FOUND"
        end if
    end tell
    '''
    
    try:
        print("Looking for PySlidemorpher window...")
        result = subprocess.run(['osascript', '-e', script2], 
                              capture_output=True, text=True, timeout=10)
        output = result.stdout.strip()
        print(f"Window search result: {output}")
        
        if output.startswith("FOUND:"):
            parts = output.split(":")
            process_name = parts[1]
            coords = parts[2].split(",")
            x, y = int(coords[0]), int(coords[1])
            print(f"✓ Found window in process '{process_name}' at position ({x}, {y})")
            return True
        else:
            print("✗ Window not found")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_with_any_window():
    """Test by looking for any window to see if the script works at all."""
    print("\nTesting with any window...")
    
    script = '''
    tell application "System Events"
        repeat with proc in every process
            try
                if (count of windows of proc) > 0 then
                    set firstWin to window 1 of proc
                    set winName to name of firstWin
                    set winPos to position of firstWin
                    set winX to item 1 of winPos
                    set winY to item 2 of winPos
                    return "FOUND:" & (name of proc) & ":" & winName & ":" & (winX as string) & "," & (winY as string)
                end if
            end try
        end repeat
        return "NO_WINDOWS"
    end tell
    '''
    
    try:
        result = subprocess.run(['osascript', '-e', script], 
                              capture_output=True, text=True, timeout=10)
        output = result.stdout.strip()
        print(f"Any window result: {output}")
        
        if output.startswith("FOUND:"):
            parts = output.split(":")
            process_name = parts[1]
            window_name = parts[2]
            coords = parts[3].split(",")
            x, y = int(coords[0]), int(coords[1])
            print(f"✓ Found window '{window_name}' in process '{process_name}' at ({x}, {y})")
            return True
        else:
            print("✗ No windows found at all")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("APPLESCRIPT FIX TEST")
    print("=" * 40)
    
    success1 = test_improved_applescript()
    success2 = test_with_any_window()
    
    print("\n" + "=" * 40)
    print("RESULTS:")
    print(f"Improved PySlidemorpher detection: {'✓' if success1 else '✗'}")
    print(f"Any window detection: {'✓' if success2 else '✗'}")
    
    if success2 and not success1:
        print("\nAppleScript works, but no PySlidemorpher window is currently open.")
        print("You need to run the slideshow first, then test position detection.")
    elif not success2:
        print("\nAppleScript is not working at all. There might be permission issues.")
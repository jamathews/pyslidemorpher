#!/usr/bin/env python3
"""
Comprehensive explanation and test of how position detection works in PySlidemorpher.
This addresses the user's question: "How are you determining the position? I think you're still getting it wrong."
"""

import sys
import logging
from pathlib import Path

# Set up logging to see all messages
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')

# Add the pyslidemorpher module to the path
sys.path.insert(0, str(Path(__file__).parent))

def explain_position_detection_system():
    """Explain exactly how the position detection system works."""
    print("HOW POSITION DETECTION WORKS IN PYSLIDEMORPHER")
    print("=" * 60)
    print()
    
    print("OVERVIEW:")
    print("The system uses a multi-layered approach to determine window position:")
    print("1. Platform-specific detection (tries to get actual current position)")
    print("2. Position tracking (tracks programmatic moves)")
    print("3. Fallback to saved position")
    print()
    
    print("PLATFORM-SPECIFIC DETECTION METHODS:")
    print()
    
    print("macOS (Darwin):")
    print("- Uses AppleScript to query System Events")
    print("- Searches all processes for windows containing 'PySlidemorpher'")
    print("- Gets position using: 'set windowPos to position of win'")
    print("- Timeout: 3 seconds")
    print("- Returns: (x, y) coordinates or (None, None) if failed")
    print()
    
    print("Windows:")
    print("- Uses win32gui.EnumWindows() to enumerate all windows")
    print("- Looks for visible windows with 'PySlidemorpher' in title")
    print("- Gets position using: win32gui.GetWindowRect()")
    print("- Returns: (x, y) coordinates or (None, None) if failed")
    print()
    
    print("Linux:")
    print("- Uses xwininfo command-line tool")
    print("- Runs: xwininfo -name 'PySlidemorpher'")
    print("- Parses output for 'Absolute upper-left X:' and 'Y:' lines")
    print("- Timeout: 2 seconds")
    print("- Returns: (x, y) coordinates or (None, None) if failed")
    print()
    
    print("POSITION SAVING LOGIC (save_window_position function):")
    print("1. If explicit coordinates provided → use them")
    print("2. Else: try get_actual_window_position()")
    print("3. If detection succeeds → save detected position")
    print("4. If detection fails → fall back to tracked position")
    print()
    
    print("POSITION TRACKING SYSTEM:")
    print("- Tracks every programmatic window move (cv2.moveWindow calls)")
    print("- Stores last known position in global variable")
    print("- Used as fallback when actual detection fails")
    print()

def test_current_platform_detection():
    """Test the position detection on the current platform."""
    print("TESTING CURRENT PLATFORM DETECTION:")
    print("-" * 40)
    
    try:
        from pyslidemorpher.realtime import (
            get_actual_window_position,
            get_window_position_macos,
            get_window_position_windows,
            get_window_position_linux
        )
        
        import sys
        platform = sys.platform
        print(f"Current platform: {platform}")
        
        window_name = "PySlidemorpher - Realtime Slideshow"
        
        if platform == 'darwin':
            print("Testing macOS AppleScript detection...")
            x, y = get_window_position_macos(window_name)
            print(f"macOS detection result: ({x}, {y})")
            
            if x is None and y is None:
                print("ISSUE: AppleScript detection failed")
                print("Possible reasons:")
                print("- No PySlidemorpher window is currently open")
                print("- AppleScript permissions not granted")
                print("- AppleScript timeout (3 seconds)")
                print("- Window title doesn't contain 'PySlidemorpher'")
            else:
                print("✓ AppleScript detection working!")
                
        elif platform == 'win32':
            print("Testing Windows win32gui detection...")
            x, y = get_window_position_windows(window_name)
            print(f"Windows detection result: ({x}, {y})")
            
            if x is None and y is None:
                print("ISSUE: win32gui detection failed")
                print("Possible reasons:")
                print("- No PySlidemorpher window is currently open")
                print("- win32gui module not available")
                print("- Window title doesn't contain 'PySlidemorpher'")
            else:
                print("✓ Windows detection working!")
                
        elif platform.startswith('linux'):
            print("Testing Linux xwininfo detection...")
            x, y = get_window_position_linux(window_name)
            print(f"Linux detection result: ({x}, {y})")
            
            if x is None and y is None:
                print("ISSUE: xwininfo detection failed")
                print("Possible reasons:")
                print("- No PySlidemorpher window is currently open")
                print("- xwininfo command not available")
                print("- xwininfo timeout (2 seconds)")
                print("- Window name doesn't match 'PySlidemorpher'")
            else:
                print("✓ Linux detection working!")
        
        # Test the generic function
        print(f"\nTesting generic get_actual_window_position...")
        x, y = get_actual_window_position(window_name)
        print(f"Generic detection result: ({x}, {y})")
        
        return x is not None and y is not None
        
    except Exception as e:
        print(f"Error testing detection: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_position_saving_logic():
    """Test the position saving logic to show how it works."""
    print("\nTESTING POSITION SAVING LOGIC:")
    print("-" * 40)
    
    try:
        from pyslidemorpher.realtime import (
            save_window_position,
            load_window_position,
            track_window_position,
            get_tracked_window_position
        )
        
        # Set up a tracked position
        print("1. Setting tracked position to (1500, 200)...")
        track_window_position("test_window", 1500, 200)
        tracked_x, tracked_y = get_tracked_window_position()
        print(f"   Tracked position: ({tracked_x}, {tracked_y})")
        
        # Test saving without explicit coordinates
        print("\n2. Calling save_window_position() without coordinates...")
        print("   This will:")
        print("   - Try to detect actual window position first")
        print("   - Fall back to tracked position if detection fails")
        
        save_window_position("test_window")
        
        # Check what was saved
        position_data = load_window_position()
        saved_x = position_data.get('x')
        saved_y = position_data.get('y')
        print(f"\n3. Position saved: ({saved_x}, {saved_y})")
        
        # Analyze the result
        if saved_x == tracked_x and saved_y == tracked_y:
            print("   → Used tracked position (detection likely failed)")
        else:
            print("   → Used detected position (detection succeeded)")
            
        return True
        
    except Exception as e:
        print(f"Error testing saving logic: {e}")
        return False

def explain_common_issues():
    """Explain common issues with position detection."""
    print("\nCOMMON ISSUES AND LIMITATIONS:")
    print("-" * 40)
    
    print("1. NO WINDOW OPEN:")
    print("   - Position detection only works when a PySlidemorpher window is open")
    print("   - If no window exists, detection returns (None, None)")
    print("   - System falls back to tracked position")
    print()
    
    print("2. PERMISSION ISSUES (macOS):")
    print("   - AppleScript may need accessibility permissions")
    print("   - System Preferences → Security & Privacy → Accessibility")
    print("   - Add Terminal or Python to allowed apps")
    print()
    
    print("3. TIMING ISSUES:")
    print("   - AppleScript: 3-second timeout")
    print("   - xwininfo: 2-second timeout")
    print("   - If system is slow, detection may timeout")
    print()
    
    print("4. WINDOW TITLE MATCHING:")
    print("   - Detection looks for 'PySlidemorpher' in window title")
    print("   - Exact title: 'PySlidemorpher - Realtime Slideshow'")
    print("   - If title changes, detection may fail")
    print()
    
    print("5. MISSING DEPENDENCIES:")
    print("   - Windows: requires win32gui module")
    print("   - Linux: requires xwininfo command")
    print("   - macOS: uses built-in AppleScript")
    print()

def provide_solution_summary():
    """Provide a summary of how the position system works and what to expect."""
    print("\nHOW THE COMPLETE SYSTEM WORKS:")
    print("-" * 40)
    
    print("WHEN USER STARTS SLIDESHOW:")
    print("1. System loads saved position from ~/.pyslidemorpher/window_position.json")
    print("2. If position exists, moves window there using cv2.moveWindow()")
    print("3. Tracks this programmatic move in memory")
    print()
    
    print("WHEN USER MANUALLY MOVES WINDOW:")
    print("1. System doesn't automatically detect the move (disabled for reliability)")
    print("2. User can press 's' key to save current position")
    print("3. Or position is saved automatically on exit")
    print()
    
    print("WHEN SAVING POSITION:")
    print("1. Try to detect actual current position using platform-specific method")
    print("2. If detection succeeds: save the detected position")
    print("3. If detection fails: save the last tracked position")
    print("4. Write position to JSON file with timestamp")
    print()
    
    print("RELIABILITY APPROACH:")
    print("- Real-time detection was disabled due to timeouts and reliability issues")
    print("- System relies on exit-time detection + user manual saving")
    print("- Provides clear feedback via CRITICAL logs")
    print("- Falls back gracefully when detection fails")
    print()

if __name__ == "__main__":
    explain_position_detection_system()
    
    detection_works = test_current_platform_detection()
    test_position_saving_logic()
    
    explain_common_issues()
    provide_solution_summary()
    
    print("\n" + "=" * 60)
    print("SUMMARY FOR USER:")
    print(f"Position detection currently: {'WORKING' if detection_works else 'NOT WORKING'}")
    
    if not detection_works:
        print("\nWHY DETECTION MIGHT BE FAILING:")
        print("- No PySlidemorpher window is currently open")
        print("- Platform-specific tools/permissions not available")
        print("- Window title doesn't match expected pattern")
        print("\nWHAT HAPPENS WHEN DETECTION FAILS:")
        print("- System falls back to tracked position (last programmatic move)")
        print("- This is why you might see the same position saved repeatedly")
        print("- The tracked position is updated when window is moved programmatically")
        print("- But not when user drags the window manually")
    else:
        print("\n✓ Position detection is working on your system!")
        print("- The system should detect actual window position")
        print("- Manual window moves should be detected when saving")
    
    print("\nTO TEST IF IT'S WORKING:")
    print("1. Run a slideshow: python -m pyslidemorpher images --realtime")
    print("2. Drag the window to a different position")
    print("3. Press 's' then 'S' to save position")
    print("4. Look for CRITICAL log: 'Saving detected window position' vs 'Saving tracked window position'")
    print("5. If you see 'detected', the system is working correctly")
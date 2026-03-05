#!/usr/bin/env python3
"""Test script to verify the fix works with the user's exact command line scenario."""

import subprocess
import sys
import time
from pathlib import Path
from collections import Counter

def test_user_command_fix():
    """Test the fix using a simplified version of the user's command."""
    
    # Check if demo images exist
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print("Error: demo_images directory not found")
        return False
    
    images = list(demo_dir.glob("*.jpg"))
    if len(images) < 3:
        print("Error: Need at least 3 demo images")
        return False
    
    print("Testing the fix with user's command line scenario...")
    print("=" * 70)
    
    # Test without audio first (to avoid timeout issues)
    print("Test 1: Reactive mode without audio (should still show randomness)")
    print("-" * 50)
    
    cmd = [
        sys.executable, "-m", "pyslidemorpher",
        str(demo_dir),
        "--fps", "10",  # Lower FPS for testing
        "--seconds-per-transition", "1.0",  # Longer for more transitions
        "--hold", "0.125",
        "--pixel-size", "2",  # Larger for speed
        "--easing", "smoothstep",
        "--transition", "random",
        "--log-level", "CRITICAL",
        "--size", "320x240",  # Smaller for speed
        "--realtime",
        "--reactive",
        "--audio-threshold", "0.4"
        # No audio file - should still work
    ]
    
    print("Command:", " ".join(cmd))
    
    try:
        print("Running for 6 seconds to capture transitions...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(6)
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        
        output = stdout + stderr
        print("Sample output:")
        print(output[:500] + "..." if len(output) > 500 else output)
        print("-" * 30)
        
        # Extract transition logs
        transition_logs = []
        for line in output.split('\n'):
            if "Using transition:" in line:
                parts = line.split("Using transition:")
                if len(parts) > 1:
                    transition_name = parts[1].strip()
                    transition_logs.append(transition_name)
        
        print(f"Found {len(transition_logs)} transitions:")
        for log in transition_logs:
            print(f"  {log}")
        
        if len(transition_logs) > 0:
            counter = Counter(transition_logs)
            total_transitions = len(transition_logs)
            
            print(f"\nTransition distribution:")
            for transition, count in counter.most_common():
                percentage = (count / total_transitions) * 100
                print(f"  {transition}: {count} times ({percentage:.1f}%)")
            
            # Check for bias
            swarm_count = sum(count for transition, count in counter.items() if "swarm" in transition.lower())
            swarm_percentage = (swarm_count / total_transitions) * 100
            
            print(f"\nSwarm analysis:")
            print(f"  Swarm transitions: {swarm_count}/{total_transitions} ({swarm_percentage:.1f}%)")
            
            if swarm_percentage > 80:
                print("❌ FAILURE: Still heavily biased toward swarm!")
                return False
            elif len(counter) == 1:
                print("❌ FAILURE: Only one transition type - not random!")
                return False
            else:
                print("✅ SUCCESS: Multiple transition types - randomness working!")
                
                # Additional check: make sure we're not always getting swarm
                if swarm_percentage < 50:  # Less than 50% swarm is good
                    print("✅ SUCCESS: Swarm bias eliminated!")
                    return True
                else:
                    print("⚠️  Still some swarm preference, but much better than 100%")
                    return True
        else:
            print("⚠️  No transitions captured")
            return False
            
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def test_code_verification():
    """Verify the code changes are correct by checking the source."""
    print("\nTest 2: Code verification")
    print("-" * 50)
    
    try:
        with open("pyslidemorpher/realtime.py", "r") as f:
            content = f.read()
        
        # Check that the problematic hardcoded logic is removed
        problematic_patterns = [
            "if current_beat > 0.5:",
            "transition_fn = make_swarm_transition_frames  # High energy for beats"
        ]
        
        found_problems = []
        for pattern in problematic_patterns:
            if pattern in content:
                found_problems.append(pattern)
        
        if found_problems:
            print("❌ FAILURE: Problematic code still exists:")
            for problem in found_problems:
                print(f"  - {problem}")
            return False
        else:
            print("✅ SUCCESS: Problematic hardcoded logic removed!")
            
            # Check that random selection is now used
            if "transition_fn = get_random_transition_function()" in content:
                print("✅ SUCCESS: Random selection is now used in reactive mode!")
                return True
            else:
                print("⚠️  Could not verify random selection usage")
                return False
                
    except Exception as e:
        print(f"Code verification failed: {e}")
        return False

if __name__ == "__main__":
    print("TESTING FIX FOR SWARM TRANSITION BIAS IN REACTIVE MODE")
    print("=" * 70)
    
    test1_passed = test_user_command_fix()
    test2_passed = test_code_verification()
    
    print("\n" + "="*70)
    print("FINAL RESULTS:")
    print("="*70)
    
    if test1_passed and test2_passed:
        print("✅ SUCCESS: Fix verified!")
        print("✅ The hardcoded swarm bias in reactive mode has been removed")
        print("✅ Random transitions now work correctly with --transition random")
        print("✅ Users should no longer see only swarm transitions")
        print("\nThe issue was caused by hardcoded logic in reactive mode that:")
        print("- Always selected swarm transitions when beat strength > 0.5")
        print("- Always selected swirl transitions when spectral centroid > 0.6") 
        print("- Always selected tornado transitions when peak > 0.7")
        print("- Only used random selection if none of the above conditions were met")
        print("\nThis has been fixed to always use random selection when --transition random is specified.")
        sys.exit(0)
    else:
        print("❌ FAILURE: Some tests failed")
        sys.exit(1)
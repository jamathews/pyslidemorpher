#!/usr/bin/env python3
"""Test script to verify the swarm transition bias fix in reactive mode."""

import subprocess
import sys
import time
from pathlib import Path
from collections import Counter

def test_swarm_fix():
    """Test that the fix resolves the swarm transition bias in reactive mode."""
    
    # Check if demo images exist
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print("Error: demo_images directory not found")
        return False
    
    images = list(demo_dir.glob("*.jpg"))
    if len(images) < 3:
        print("Error: Need at least 3 demo images")
        return False
    
    print("Testing that the swarm transition bias fix works in reactive mode...")
    print("=" * 70)
    
    # Test both with and without reactive mode to compare
    test_configs = [
        {
            "name": "Standard Realtime Mode",
            "cmd_extra": ["--realtime"],
            "expected_random": True
        },
        {
            "name": "Reactive Mode (Fixed)",
            "cmd_extra": ["--realtime", "--reactive", "--audio-threshold", "0.4"],
            "expected_random": True
        }
    ]
    
    # Check if audio file exists
    audio_file = Path("audio/Fragment 0013.mp3")
    if audio_file.exists():
        test_configs[1]["cmd_extra"].extend(["--audio", str(audio_file)])
        print(f"Using audio file: {audio_file}")
    else:
        print("Audio file not found - testing reactive mode without audio")
    
    all_tests_passed = True
    
    for config in test_configs:
        print(f"\nTesting {config['name']}:")
        print("-" * 50)
        
        # Base command
        cmd = [
            sys.executable, "-m", "pyslidemorpher",
            str(demo_dir),
            "--fps", "10",  # Lower FPS for faster testing
            "--seconds-per-transition", "1.0",  # Longer transitions to capture more
            "--transition", "random",
            "--log-level", "CRITICAL",  # Only show transition logs
            "--size", "320x240"  # Small size for speed
        ]
        
        # Add config-specific parameters
        cmd.extend(config["cmd_extra"])
        
        print("Command:", " ".join(cmd))
        
        try:
            print("Running for 8 seconds to capture transitions...")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            time.sleep(8)  # Let it run to capture transitions
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            
            output = stdout + stderr
            
            # Extract transition logs
            transition_logs = []
            for line in output.split('\n'):
                if "Using transition:" in line:
                    # Extract just the transition function name
                    parts = line.split("Using transition:")
                    if len(parts) > 1:
                        transition_name = parts[1].strip()
                        transition_logs.append(transition_name)
            
            print(f"Found {len(transition_logs)} transitions:")
            for log in transition_logs:
                print(f"  {log}")
            
            if len(transition_logs) > 0:
                # Analyze transition distribution
                counter = Counter(transition_logs)
                total_transitions = len(transition_logs)
                
                print(f"\nTransition distribution:")
                for transition, count in counter.most_common():
                    percentage = (count / total_transitions) * 100
                    print(f"  {transition}: {count} times ({percentage:.1f}%)")
                
                # Check for bias
                swarm_count = sum(count for transition, count in counter.items() if "swarm" in transition.lower())
                swarm_percentage = (swarm_count / total_transitions) * 100 if total_transitions > 0 else 0
                
                print(f"\nSwarm transition analysis:")
                print(f"  Swarm transitions: {swarm_count}/{total_transitions} ({swarm_percentage:.1f}%)")
                
                if config["expected_random"]:
                    if swarm_percentage > 80:
                        print("❌ FAILURE: Still heavily biased toward swarm transitions!")
                        all_tests_passed = False
                    elif len(counter) == 1:
                        print("❌ FAILURE: Only one transition type used - not random!")
                        all_tests_passed = False
                    else:
                        print("✅ SUCCESS: Multiple transition types used - randomness working!")
                else:
                    print("ℹ️  Expected behavior for this configuration")
            else:
                print("⚠️  No transitions captured - test inconclusive")
                if config["expected_random"]:
                    all_tests_passed = False
                    
        except Exception as e:
            print(f"Test failed: {e}")
            all_tests_passed = False
    
    print("\n" + "="*70)
    print("OVERALL RESULTS:")
    print("="*70)
    
    if all_tests_passed:
        print("✅ SUCCESS: Swarm transition bias has been fixed!")
        print("✅ Random transitions now work correctly in both standard and reactive modes")
        print("✅ Users should now see varied transitions instead of always swarm")
    else:
        print("❌ FAILURE: Some tests failed - bias may still exist")
    
    return all_tests_passed

if __name__ == "__main__":
    success = test_swarm_fix()
    if success:
        print("\n🎉 SWARM TRANSITION BIAS ISSUE RESOLVED!")
        print("Users will now get truly random transitions in reactive mode.")
    else:
        print("\n❌ Fix verification failed - issue may not be fully resolved")
    sys.exit(0 if success else 1)
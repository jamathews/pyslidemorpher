#!/usr/bin/env python3
"""Test script to check for bias in transition selection and logging visibility."""

import subprocess
import sys
import time
from pathlib import Path
from collections import Counter

def test_transition_bias_and_logging():
    """Test for bias in transition selection and logging at different levels."""
    
    # Check if demo images exist
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print("Error: demo_images directory not found")
        return False
    
    images = list(demo_dir.glob("*.jpg"))
    if len(images) < 3:
        print("Error: Need at least 3 demo images")
        return False
    
    print("Testing transition selection bias and logging visibility...")
    print("-" * 70)
    
    # Test 1: Check for bias in transition selection
    print("TEST 1: Checking for bias in transition selection")
    print("Running multiple slideshows to collect transition statistics...")
    
    all_transitions = []
    
    for run in range(10):  # More runs for better statistics
        print(f"Run {run + 1}/10...", end=" ")
        
        cmd = [
            sys.executable, "-m", "pyslidemorpher",
            str(demo_dir),
            "--transition", "random",
            "--fps", "30",
            "--seconds-per-transition", "0.3",  # Very short for speed
            "--size", "160x120",  # Very small for speed
            "--out", f"test_bias_run_{run + 1}.mp4",
            "--log-level", "CRITICAL"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            
            # Extract transition names from logs
            output = result.stdout + result.stderr
            transitions = []
            for line in output.split('\n'):
                if "Randomly selected transition:" in line:
                    parts = line.split("Randomly selected transition:")
                    if len(parts) > 1:
                        transition_name = parts[1].strip()
                        transitions.append(transition_name)
            
            all_transitions.extend(transitions)
            print(f"Got {len(transitions)} transitions")
            
            # Clean up
            output_file = Path(f"test_bias_run_{run + 1}.mp4")
            if output_file.exists():
                output_file.unlink()
                
        except Exception as e:
            print(f"Failed: {e}")
    
    # Analyze bias
    print(f"\nTransition selection statistics (total: {len(all_transitions)}):")
    if all_transitions:
        counter = Counter(all_transitions)
        for transition, count in counter.most_common():
            percentage = (count / len(all_transitions)) * 100
            print(f"  {transition}: {count} times ({percentage:.1f}%)")
        
        # Check if distribution is reasonably even
        expected_per_transition = len(all_transitions) / len(counter)
        max_deviation = max(abs(count - expected_per_transition) for count in counter.values())
        deviation_percentage = (max_deviation / expected_per_transition) * 100
        
        print(f"\nExpected per transition: {expected_per_transition:.1f}")
        print(f"Max deviation: {max_deviation:.1f} ({deviation_percentage:.1f}%)")
        
        if deviation_percentage > 50:  # More than 50% deviation suggests bias
            print("⚠️  POTENTIAL BIAS: Some transitions are selected much more often than others")
        else:
            print("✅ Distribution looks reasonably even")
    else:
        print("❌ No transitions captured!")
    
    # Test 2: Check logging visibility at different levels
    print("\n" + "="*70)
    print("TEST 2: Checking logging visibility at different levels")
    print("="*70)
    
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    
    for log_level in log_levels:
        print(f"\nTesting with log level: {log_level}")
        
        cmd = [
            sys.executable, "-m", "pyslidemorpher",
            str(demo_dir),
            "--transition", "random",
            "--fps", "30",
            "--seconds-per-transition", "0.3",
            "--size", "160x120",
            "--out", f"test_log_{log_level.lower()}.mp4",
            "--log-level", log_level
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            output = result.stdout + result.stderr
            
            # Count different types of log messages
            transition_logs = len([line for line in output.split('\n') if "Randomly selected transition:" in line])
            using_logs = len([line for line in output.split('\n') if "Using transition:" in line])
            other_logs = len([line for line in output.split('\n') if ' - ' in line and 'transition' not in line.lower()])
            
            print(f"  Transition selection logs: {transition_logs}")
            print(f"  'Using transition' logs: {using_logs}")
            print(f"  Other log messages: {other_logs}")
            
            if transition_logs == 0 and using_logs == 0:
                print(f"  ❌ NO TRANSITION LOGS visible at {log_level} level!")
            else:
                print(f"  ✅ Transition logs visible at {log_level} level")
            
            # Clean up
            output_file = Path(f"test_log_{log_level.lower()}.mp4")
            if output_file.exists():
                output_file.unlink()
                
        except Exception as e:
            print(f"  Failed: {e}")
    
    print("\n" + "="*70)
    print("CONCLUSIONS:")
    print("="*70)
    print("1. If bias is detected, the random selection algorithm may need improvement")
    print("2. If transition logs are not visible at INFO/WARNING levels, that's the user's issue")
    print("3. Users may need to use --log-level CRITICAL to see transition selections")
    
    return True

if __name__ == "__main__":
    test_transition_bias_and_logging()
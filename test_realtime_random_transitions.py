#!/usr/bin/env python3
"""Test script to check random transitions in realtime mode."""

import subprocess
import sys
import time
from pathlib import Path

def test_realtime_random_transitions():
    """Test random transitions in realtime mode."""
    
    # Check if demo images exist
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print("Error: demo_images directory not found")
        return False
    
    images = list(demo_dir.glob("*.jpg"))
    if len(images) < 3:
        print("Error: Need at least 3 demo images")
        return False
    
    print("Testing random transitions in REALTIME mode...")
    print("This will run realtime mode multiple times and capture transition logs")
    print("-" * 70)
    
    transitions_per_run = []
    
    # Run realtime mode multiple times
    for run in range(3):  # Fewer runs since realtime mode needs manual termination
        print(f"\nRun {run + 1}/3:")
        
        # Run slideshow in realtime mode with random transitions
        cmd = [
            sys.executable, "-m", "pyslidemorpher",
            str(demo_dir),
            "--realtime",
            "--transition", "random",
            "--fps", "10",
            "--seconds-per-transition", "1.0",  # Short transitions
            "--size", "320x240",  # Small size
            "--log-level", "CRITICAL"  # Only show transition selection logs
        ]
        
        try:
            # Start the process
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Let it run for a few seconds to capture some transitions
            time.sleep(5)
            
            # Terminate the process
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            
            # Extract transition names from logs
            output = stdout + stderr
            transitions = []
            for line in output.split('\n'):
                if "Randomly selected transition function:" in line:
                    # Extract transition name
                    parts = line.split("Randomly selected transition function:")
                    if len(parts) > 1:
                        transition_name = parts[1].strip()
                        transitions.append(transition_name)
                        print(f"  Selected: {transition_name}")
            
            transitions_per_run.append(transitions)
                
        except subprocess.TimeoutExpired:
            print(f"  Run {run + 1} timed out")
            transitions_per_run.append([])
        except Exception as e:
            print(f"  Run {run + 1} failed: {e}")
            transitions_per_run.append([])
    
    # Analyze results
    print("\n" + "="*70)
    print("REALTIME MODE ANALYSIS:")
    print("="*70)
    
    all_same = True
    if len(transitions_per_run) > 1:
        first_run = transitions_per_run[0]
        for i, run_transitions in enumerate(transitions_per_run[1:], 2):
            if run_transitions != first_run:
                all_same = False
                break
    
    print(f"Total runs: {len(transitions_per_run)}")
    for i, transitions in enumerate(transitions_per_run, 1):
        print(f"Run {i}: {transitions}")
    
    if all_same and len(transitions_per_run[0]) > 0:
        print("\n❌ ISSUE CONFIRMED: Same transitions selected in every realtime run!")
        print("This indicates the random seed is not being properly initialized in realtime mode.")
        return False
    elif len(transitions_per_run[0]) == 0:
        print("\n⚠️  No transition logs captured - test inconclusive")
        print("This might indicate that realtime mode isn't logging transitions properly")
        return False
    else:
        print("\n✅ Random transitions working correctly in realtime mode")
        return True

if __name__ == "__main__":
    success = test_realtime_random_transitions()
    sys.exit(0 if success else 1)
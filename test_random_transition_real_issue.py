#!/usr/bin/env python3
"""Test script to reproduce the actual random transition issue in slideshow context."""

import subprocess
import sys
import time
from pathlib import Path

def test_random_transitions_in_slideshow():
    """Test random transitions in actual slideshow runs."""
    
    # Check if demo images exist
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print("Error: demo_images directory not found")
        return False
    
    images = list(demo_dir.glob("*.jpg"))
    if len(images) < 3:
        print("Error: Need at least 3 demo images")
        return False
    
    print("Testing random transitions in actual slideshow runs...")
    print("This will run the slideshow multiple times and capture the transition logs")
    print("-" * 70)
    
    transitions_per_run = []
    
    # Run slideshow multiple times
    for run in range(5):
        print(f"\nRun {run + 1}/5:")
        
        # Run slideshow with random transitions
        cmd = [
            sys.executable, "-m", "pyslidemorpher",
            str(demo_dir),
            "--transition", "random",
            "--fps", "30",
            "--seconds-per-transition", "0.5",  # Short transitions for quick test
            "--size", "320x240",  # Small size for speed
            "--out", f"test_random_run_{run + 1}.mp4",
            "--log-level", "CRITICAL"  # Only show transition selection logs
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Extract transition names from logs
            output = result.stdout + result.stderr
            transitions = []
            for line in output.split('\n'):
                if "Randomly selected transition:" in line:
                    # Extract transition name
                    parts = line.split("Randomly selected transition:")
                    if len(parts) > 1:
                        transition_name = parts[1].strip()
                        transitions.append(transition_name)
                        print(f"  Selected: {transition_name}")
            
            transitions_per_run.append(transitions)
            
            # Clean up output file
            output_file = Path(f"test_random_run_{run + 1}.mp4")
            if output_file.exists():
                output_file.unlink()
                
        except subprocess.TimeoutExpired:
            print(f"  Run {run + 1} timed out")
            transitions_per_run.append([])
        except Exception as e:
            print(f"  Run {run + 1} failed: {e}")
            transitions_per_run.append([])
    
    # Analyze results
    print("\n" + "="*70)
    print("ANALYSIS:")
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
        print("\n❌ ISSUE CONFIRMED: Same transitions selected in every run!")
        print("This indicates the random seed is not being properly initialized.")
        return False
    elif len(transitions_per_run[0]) == 0:
        print("\n⚠️  No transition logs captured - test inconclusive")
        return False
    else:
        print("\n✅ Random transitions working correctly - different selections per run")
        return True

if __name__ == "__main__":
    success = test_random_transitions_in_slideshow()
    sys.exit(0 if success else 1)
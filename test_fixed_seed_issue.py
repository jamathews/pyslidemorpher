#!/usr/bin/env python3
"""Test script to check if fixed seed causes the random transition issue."""

import subprocess
import sys
import time
from pathlib import Path

def test_fixed_seed_randomness():
    """Test random transitions with fixed seed to see if that's the issue."""
    
    # Check if demo images exist
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print("Error: demo_images directory not found")
        return False
    
    images = list(demo_dir.glob("*.jpg"))
    if len(images) < 3:
        print("Error: Need at least 3 demo images")
        return False
    
    print("Testing random transitions with FIXED SEED...")
    print("This will test if using the same seed causes the same transitions")
    print("-" * 70)
    
    # Test with fixed seed (default is 123)
    print("\nTesting with fixed seed (123):")
    transitions_with_seed = []
    
    for run in range(3):
        print(f"\nRun {run + 1}/3 with seed=123:")
        
        cmd = [
            sys.executable, "-m", "pyslidemorpher",
            str(demo_dir),
            "--transition", "random",
            "--seed", "123",  # Fixed seed
            "--fps", "30",
            "--seconds-per-transition", "0.5",
            "--size", "320x240",
            "--out", f"test_seed_run_{run + 1}.mp4",
            "--log-level", "CRITICAL"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Extract transition names from logs
            output = result.stdout + result.stderr
            transitions = []
            for line in output.split('\n'):
                if "Randomly selected transition:" in line:
                    parts = line.split("Randomly selected transition:")
                    if len(parts) > 1:
                        transition_name = parts[1].strip()
                        transitions.append(transition_name)
                        print(f"  Selected: {transition_name}")
            
            transitions_with_seed.append(transitions)
            
            # Clean up
            output_file = Path(f"test_seed_run_{run + 1}.mp4")
            if output_file.exists():
                output_file.unlink()
                
        except Exception as e:
            print(f"  Run {run + 1} failed: {e}")
            transitions_with_seed.append([])
    
    # Test without specifying seed (uses default)
    print("\n" + "-"*50)
    print("Testing with DEFAULT seed behavior:")
    transitions_default = []
    
    for run in range(3):
        print(f"\nRun {run + 1}/3 with default seed:")
        
        cmd = [
            sys.executable, "-m", "pyslidemorpher",
            str(demo_dir),
            "--transition", "random",
            # No --seed specified, uses default
            "--fps", "30",
            "--seconds-per-transition", "0.5",
            "--size", "320x240",
            "--out", f"test_default_run_{run + 1}.mp4",
            "--log-level", "CRITICAL"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Extract transition names from logs
            output = result.stdout + result.stderr
            transitions = []
            for line in output.split('\n'):
                if "Randomly selected transition:" in line:
                    parts = line.split("Randomly selected transition:")
                    if len(parts) > 1:
                        transition_name = parts[1].strip()
                        transitions.append(transition_name)
                        print(f"  Selected: {transition_name}")
            
            transitions_default.append(transitions)
            
            # Clean up
            output_file = Path(f"test_default_run_{run + 1}.mp4")
            if output_file.exists():
                output_file.unlink()
                
        except Exception as e:
            print(f"  Run {run + 1} failed: {e}")
            transitions_default.append([])
    
    # Analyze results
    print("\n" + "="*70)
    print("FIXED SEED ANALYSIS:")
    print("="*70)
    
    print("With fixed seed (123):")
    for i, transitions in enumerate(transitions_with_seed, 1):
        print(f"  Run {i}: {transitions}")
    
    print("\nWith default seed:")
    for i, transitions in enumerate(transitions_default, 1):
        print(f"  Run {i}: {transitions}")
    
    # Check if fixed seed causes identical results
    seed_all_same = True
    if len(transitions_with_seed) > 1:
        first_run = transitions_with_seed[0]
        for run_transitions in transitions_with_seed[1:]:
            if run_transitions != first_run:
                seed_all_same = False
                break
    
    # Check if default seed causes identical results
    default_all_same = True
    if len(transitions_default) > 1:
        first_run = transitions_default[0]
        for run_transitions in transitions_default[1:]:
            if run_transitions != first_run:
                default_all_same = False
                break
    
    print("\nCONCLUSIONS:")
    if seed_all_same and len(transitions_with_seed[0]) > 0:
        print("❌ ISSUE FOUND: Fixed seed causes identical transition sequences!")
        print("   This explains why users see the same transitions every time.")
    else:
        print("✅ Fixed seed test: Different sequences even with same seed")
    
    if default_all_same and len(transitions_default[0]) > 0:
        print("❌ ISSUE FOUND: Default seed behavior causes identical sequences!")
        print("   Users get the same transitions because default seed is always 123.")
    else:
        print("✅ Default seed test: Different sequences with default behavior")
    
    return not (seed_all_same or default_all_same)

if __name__ == "__main__":
    success = test_fixed_seed_randomness()
    sys.exit(0 if success else 1)
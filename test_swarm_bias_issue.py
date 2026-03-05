#!/usr/bin/env python3
"""Test script to reproduce the swarm transition bias issue in reactive mode."""

import subprocess
import sys
import time
from pathlib import Path

def test_swarm_bias_issue():
    """Test the exact command line that always produces swarm transitions."""
    
    # Check if demo images exist
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print("Error: demo_images directory not found")
        return False
    
    images = list(demo_dir.glob("*.jpg"))
    if len(images) < 3:
        print("Error: Need at least 3 demo images")
        return False
    
    # Check if audio file exists
    audio_file = Path("audio/Fragment 0013.mp3")
    if not audio_file.exists():
        print("Error: Audio file not found - using demo without audio")
        audio_file = None
    
    print("Testing the exact command line that causes swarm transition bias...")
    print("=" * 70)
    
    # Reproduce the user's exact command line (simplified for testing)
    cmd = [
        sys.executable, "-m", "pyslidemorpher",
        str(demo_dir),
        "--fps", "30",
        "--seconds-per-transition", "0.5",
        "--hold", "0.125", 
        "--pixel-size", "1",
        "--easing", "smoothstep",
        "--transition", "random",
        "--log-level", "WARNING",
        "--size", "1024x768",
        "--realtime",
        "--reactive",
        "--audio-threshold", "0.4"
    ]
    
    # Add audio if available
    if audio_file:
        cmd.extend(["--audio", str(audio_file)])
        print(f"Using audio file: {audio_file}")
    else:
        print("Running without audio (may not reproduce the exact issue)")
    
    print("Command:", " ".join(cmd))
    print("-" * 50)
    
    try:
        print("Starting realtime reactive mode for 10 seconds to capture transitions...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(10)  # Let it run longer to capture more transitions
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        
        output = stdout + stderr
        print("Raw output:")
        print(output)
        print("-" * 40)
        
        # Look for transition logs
        transition_logs = []
        for line in output.split('\n'):
            if "Using transition:" in line:
                transition_logs.append(line.strip())
        
        print(f"Found {len(transition_logs)} transition logs:")
        for log in transition_logs:
            print(f"  {log}")
        
        # Count swarm transitions
        swarm_count = sum(1 for log in transition_logs if "swarm" in log.lower())
        total_transitions = len(transition_logs)
        
        if total_transitions > 0:
            swarm_percentage = (swarm_count / total_transitions) * 100
            print(f"\nTransition analysis:")
            print(f"  Total transitions: {total_transitions}")
            print(f"  Swarm transitions: {swarm_count}")
            print(f"  Swarm percentage: {swarm_percentage:.1f}%")
            
            if swarm_percentage > 80:  # More than 80% swarm indicates bias
                print("❌ ISSUE CONFIRMED: Swarm transition bias detected!")
                print("   Reactive mode is heavily biased toward swarm transitions")
                return False
            elif swarm_percentage > 50:
                print("⚠️  POTENTIAL ISSUE: High swarm transition usage")
                return False
            else:
                print("✅ Random transitions appear to be working correctly")
                return True
        else:
            print("⚠️  No transition logs captured - test inconclusive")
            return False
            
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_swarm_bias_issue()
    if not success:
        print("\n🔍 ISSUE ANALYSIS:")
        print("The reactive mode has hardcoded logic that selects specific transitions")
        print("based on audio characteristics, overriding random selection:")
        print("- Beat strength > 0.5 → Always swarm transition")
        print("- Spectral centroid > 0.6 → Always swirl transition") 
        print("- Peak > 0.7 → Always tornado transition")
        print("- Only if none of these → Random selection")
        print("\nThis explains why users see the same transition repeatedly!")
    sys.exit(0 if success else 1)
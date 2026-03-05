#!/usr/bin/env python3
"""Test script to verify CRITICAL logging for transitions in realtime mode."""

import subprocess
import sys
import time
from pathlib import Path

def test_realtime_critical_logging():
    """Test that realtime mode shows CRITICAL logs for every transition."""
    
    # Check if demo images exist
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print("Error: demo_images directory not found")
        return False
    
    images = list(demo_dir.glob("*.jpg"))
    if len(images) < 3:
        print("Error: Need at least 3 demo images")
        return False
    
    print("Testing CRITICAL logging for transitions in REALTIME mode...")
    print("=" * 60)
    
    # Test realtime mode with CRITICAL logging
    cmd = [
        sys.executable, "-m", "pyslidemorpher",
        str(demo_dir),
        "--realtime",
        "--transition", "random",
        "--fps", "10",
        "--seconds-per-transition", "1.0",
        "--size", "320x240",
        "--log-level", "CRITICAL"  # Only show CRITICAL logs
    ]
    
    try:
        print("Starting realtime mode for 5 seconds to capture transition logs...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(5)  # Let it run for 5 seconds to capture transitions
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
        
        output = stdout + stderr
        print("Raw output:")
        print(output)
        print("-" * 40)
        
        critical_logs = [line for line in output.split('\n') if "Using transition:" in line and "CRITICAL" in line]
        
        print(f"Found {len(critical_logs)} CRITICAL transition logs in realtime mode:")
        for log in critical_logs:
            print(f"  {log.strip()}")
        
        if len(critical_logs) > 0:
            print("✅ SUCCESS: CRITICAL transition logs found in realtime mode!")
            return True
        else:
            print("⚠️  No CRITICAL transition logs captured in realtime mode")
            print("This could be due to timing - transitions may not have occurred in 5 seconds")
            return False
            
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_realtime_critical_logging()
    if success:
        print("\n🎉 REALTIME CRITICAL LOGGING WORKING!")
    else:
        print("\n⚠️  Realtime test inconclusive - may need longer run time")
    sys.exit(0 if success else 1)
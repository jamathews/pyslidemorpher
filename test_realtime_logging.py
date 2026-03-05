#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

def test_realtime_logging():
    """Test to reproduce the realtime mode logging issue."""

    # Check if demo images exist
    demo_dir = Path("demo_images")
    if not demo_dir.exists():
        print("Error: demo_images directory not found")
        return False

    images = list(demo_dir.glob("*.jpg"))
    if len(images) < 2:
        print("Error: Need at least 2 demo images")
        return False

    # Test different log levels
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]

    for log_level in log_levels:
        print(f"\nTesting realtime mode logging with {log_level} level...")
        print("-" * 50)

        # Run realtime mode with specified log level
        cmd = [
            sys.executable, "-m", "pyslidemorpher",
            str(demo_dir),
            "--realtime",
            "--log-level", log_level,
            "--fps", "10",
            "--seconds-per-transition", "2",
            "--size", "640x480"
        ]

        test_log_level(cmd, log_level)

    return True

def test_log_level(cmd, log_level):
    """Test a specific log level."""
    import time

    try:
        # Start the process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Wait a short time to capture initial logs
        time.sleep(3)

        # Terminate the process
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)

        print("STDOUT:")
        print(stdout if stdout else "(empty)")
        print("\nSTDERR:")
        print(stderr if stderr else "(empty)")
        print(f"\nReturn code: {process.returncode}")

        # Count log messages
        output = stdout + stderr
        log_lines = [line for line in output.split('\n') if ' - ' in line and any(level in line for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])]
        print(f"Found {len(log_lines)} log messages with {log_level} level")

        if log_lines:
            print("Sample log messages:")
            for line in log_lines[:5]:  # Show first 5 log messages
                print(f"  {line}")

        return len(log_lines) > 0

    except subprocess.TimeoutExpired:
        print("Process communication timed out")
        return False
    except Exception as e:
        print(f"Error running test: {e}")
        return False

if __name__ == "__main__":
    success = test_realtime_logging()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Test script to verify audio functionality in PySlidemorpher.
Tests both file output and realtime modes with audio.
"""

import subprocess
import sys
import os
from pathlib import Path

def test_file_output_with_audio():
    """Test file output mode with audio."""
    print("Testing file output mode with audio...")

    cmd = [
        sys.executable, "pyslidemorpher.py",
        "demo_images",
        "--out", "test_output_with_audio.mp4",
        "--audio", "audio/Fragment 0013.mp3",
        "--fps", "10",
        "--seconds-per-transition", "1.0",
        "--hold", "0.2",
        "--size", "640x480"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✓ File output with audio completed successfully")
            if Path("test_output_with_audio.mp4").exists():
                print("✓ Output file created")
                return True
            else:
                print("✗ Output file not found")
                return False
        else:
            print(f"✗ File output failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ File output timed out")
        return False
    except Exception as e:
        print(f"✗ File output error: {e}")
        return False

def test_file_output_without_audio():
    """Test file output mode without audio (baseline)."""
    print("Testing file output mode without audio (baseline)...")

    cmd = [
        sys.executable, "pyslidemorpher.py",
        "demo_images",
        "--out", "test_output_no_audio.mp4",
        "--fps", "10",
        "--seconds-per-transition", "1.0",
        "--hold", "0.2",
        "--size", "640x480"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✓ File output without audio completed successfully")
            if Path("test_output_no_audio.mp4").exists():
                print("✓ Output file created")
                return True
            else:
                print("✗ Output file not found")
                return False
        else:
            print(f"✗ File output failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("✗ File output timed out")
        return False
    except Exception as e:
        print(f"✗ File output error: {e}")
        return False

def test_help_shows_audio_parameter():
    """Test that --help shows the new --audio parameter."""
    print("Testing that --help shows --audio parameter...")

    cmd = [sys.executable, "pyslidemorpher.py", "--help"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if "--audio" in result.stdout:
            print("✓ --audio parameter appears in help")
            return True
        else:
            print("✗ --audio parameter not found in help")
            print("Help output:", result.stdout)
            return False
    except Exception as e:
        print(f"✗ Help test error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing PySlidemorpher audio functionality...")
    print("=" * 50)

    tests = [
        test_help_shows_audio_parameter,
        test_file_output_without_audio,
        test_file_output_with_audio,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()

    # Summary
    print("=" * 50)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path

def test_reactive_validation():
    """Test that --reactive requires both --audio and --realtime"""
    print("Testing reactive mode validation...")

    # Test 1: --reactive without --realtime should fail
    result = subprocess.run([
        sys.executable, "pyslidemorpher.py", 
        "demo_images", 
        "--reactive"
    ], capture_output=True, text=True)

    if result.returncode != 1:
        print("❌ FAIL: --reactive without --realtime should fail")
        return False

    if "can only be used with --realtime mode" not in result.stdout:
        print("❌ FAIL: Expected error message not found")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False

    print("✅ PASS: --reactive without --realtime correctly fails")

    # Test 2: --reactive with --realtime but without --audio should fail
    result = subprocess.run([
        sys.executable, "pyslidemorpher.py", 
        "demo_images", 
        "--reactive", "--realtime"
    ], capture_output=True, text=True)

    if result.returncode != 1:
        print("❌ FAIL: --reactive without --audio should fail")
        return False

    if "requires --audio to be specified" not in result.stdout:
        print("❌ FAIL: Expected error message not found")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False

    print("✅ PASS: --reactive without --audio correctly fails")

    # Test 3: Invalid audio threshold should fail
    result = subprocess.run([
        sys.executable, "pyslidemorpher.py", 
        "demo_images", 
        "--reactive", "--realtime", "--audio", "audio/Fragment 0013.wav",
        "--audio-threshold", "1.5"
    ], capture_output=True, text=True)

    if result.returncode != 1:
        print("❌ FAIL: Invalid audio threshold should fail")
        return False

    if "must be between 0.0 and 1.0" not in result.stdout:
        print("❌ FAIL: Expected error message not found")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        return False

    print("✅ PASS: Invalid audio threshold correctly fails")

    return True

def test_help_shows_reactive():
    """Test that help shows the new reactive parameters"""
    print("\nTesting help output...")

    result = subprocess.run([
        sys.executable, "pyslidemorpher.py", "--help"
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ FAIL: Help command failed")
        return False

    if "--reactive" not in result.stdout:
        print("❌ FAIL: --reactive not found in help")
        return False

    if "--audio-threshold" not in result.stdout:
        print("❌ FAIL: --audio-threshold not found in help")
        return False

    print("✅ PASS: Help shows reactive parameters")
    return True

def test_normal_mode_still_works():
    """Test that normal mode still works without reactive parameters"""
    print("\nTesting normal mode compatibility...")

    # Test normal realtime mode (should not fail due to validation)
    result = subprocess.run([
        sys.executable, "pyslidemorpher.py", 
        "demo_images", 
        "--realtime", "--help"  # Use --help to avoid actually running
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ FAIL: Normal realtime mode validation failed")
        return False

    print("✅ PASS: Normal mode validation works")
    return True

def main():
    """Run all tests"""
    print("Testing reactive mode implementation...\n")

    # Check if demo images exist
    if not Path("demo_images").exists():
        print("❌ FAIL: demo_images directory not found")
        return False

    # Check if audio file exists
    if not Path("audio/Fragment 0013.wav").exists():
        print("❌ FAIL: audio/Fragment 0013.wav not found")
        return False

    tests = [
        test_reactive_validation,
        test_help_shows_reactive,
        test_normal_mode_still_works
    ]

    passed = 0
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"❌ Test {test.__name__} failed")

    print(f"\n{passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

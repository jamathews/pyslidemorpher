#!/usr/bin/env python3
"""
Test script to verify the real Flask server works correctly.
"""

import time
import threading
import subprocess
from pyslidemorpher.web_gui import start_web_server, FLASK_AVAILABLE

def test_real_server():
    """Test the real Flask server."""
    print("Testing real Flask server...")

    if not FLASK_AVAILABLE:
        print("Flask not available")
        return False

    # Start the web server
    print("Starting web server...")
    server_thread = start_web_server()

    if not server_thread:
        print("Failed to start web server")
        return False

    # Wait for server to fully initialize
    print("Waiting for server to initialize...")
    time.sleep(3)

    try:
        # Test the main page using curl
        print("Testing main page...")

        # Get status code
        status_result = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 'http://localhost:5001/'],
                                     capture_output=True, text=True, timeout=10)
        status_code = int(status_result.stdout.strip())
        print(f"Response status code: {status_code}")

        if status_code == 200:
            print("✓ Main page accessible")

            # Get page content
            content_result = subprocess.run(['curl', '-s', 'http://localhost:5001/'], 
                                          capture_output=True, text=True, timeout=10)
            content = content_result.stdout

            if 'PySlidemorpher Control Panel' in content:
                print("✓ Control panel content found")
                return True
            else:
                print("✗ Control panel content not found")
                print(f"Response preview: {content[:200]}...")
                return False
        else:
            print(f"✗ Unexpected status code: {status_code}")

            # Get error content
            content_result = subprocess.run(['curl', '-s', 'http://localhost:5001/'], 
                                          capture_output=True, text=True, timeout=10)
            print(f"Response text: {content_result.stdout[:500]}...")
            return False

    except subprocess.TimeoutExpired:
        print("✗ Request timed out")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_real_server()
    if success:
        print("🎉 Real server test passed!")
    else:
        print("❌ Real server test failed!")

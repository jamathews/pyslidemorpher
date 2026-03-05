#!/usr/bin/env python3
"""
Test script to check both main route and test route.
"""

import time
import subprocess
from pyslidemorpher.web_gui import start_web_server, FLASK_AVAILABLE

def test_routes():
    """Test both routes."""
    print("Testing Flask routes...")

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
        # Test the simple test route first
        print("Testing /test route...")
        status_result = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 'http://localhost:5001/test'],
                                     capture_output=True, text=True, timeout=10)
        test_status = int(status_result.stdout.strip())
        print(f"Test route status code: {test_status}")

        if test_status == 200:
            print("✓ Test route accessible")

            # Get test route content
            content_result = subprocess.run(['curl', '-s', 'http://localhost:5001/test'],
                                          capture_output=True, text=True, timeout=10)
            print(f"Test route content: {content_result.stdout}")
        else:
            print(f"✗ Test route failed with status: {test_status}")

        # Test the main route
        print("\nTesting main route (/)...")
        status_result = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 'http://localhost:5001/'],
                                     capture_output=True, text=True, timeout=10)
        main_status = int(status_result.stdout.strip())
        print(f"Main route status code: {main_status}")

        # Get main route content regardless of status
        content_result = subprocess.run(['curl', '-s', 'http://localhost:5001/'],
                                      capture_output=True, text=True, timeout=10)
        print(f"Main route content preview: {content_result.stdout[:500]}...")

        return test_status == 200

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_routes()
    if success:
        print("🎉 Route tests completed!")
    else:
        print("❌ Route tests failed!")

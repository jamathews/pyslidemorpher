#!/usr/bin/env python3
"""
Test script for the new stop button functionality.
This script tests that the stop command is properly handled by the web GUI.
"""

import sys
from pathlib import Path

def test_stop_command_api():
    """Test that the stop command is accepted by the web API."""
    print("Testing stop command API...")

    # Test that stop command is accepted
    try:
        # Import the web GUI module to test the controller directly
        from pyslidemorpher.web_gui import RealtimeController

        controller = RealtimeController()

        # Test sending stop command
        controller.send_command('stop')

        # Check that the command was queued
        if not controller.command_queue.empty():
            command = controller.command_queue.get()
            if command == 'stop':
                print("✓ Stop command successfully queued in controller")
                return True
            else:
                print(f"✗ Wrong command in queue: {command}")
                return False
        else:
            print("✗ Command queue is empty")
            return False

    except Exception as e:
        print(f"✗ Controller test failed: {e}")
        return False

def test_flask_api():
    """Test the Flask API endpoint for stop command."""
    print("\nTesting Flask API endpoint...")

    try:
        from pyslidemorpher.web_gui import create_web_app, FLASK_AVAILABLE

        if not FLASK_AVAILABLE:
            print("⚠ Flask not available, skipping Flask API test")
            return True

        app = create_web_app()
        if app is None:
            print("✗ Failed to create Flask app")
            return False

        # Test the app in test mode
        with app.test_client() as client:
            # Test stop command
            response = client.post('/api/command', 
                                 json={'command': 'stop'},
                                 content_type='application/json')

            if response.status_code == 200:
                data = response.get_json()
                if data.get('status') == 'success' and data.get('command') == 'stop':
                    print("✓ Flask API accepts stop command")
                    return True
                else:
                    print(f"✗ Unexpected response: {data}")
                    return False
            else:
                print(f"✗ API returned status code: {response.status_code}")
                return False

    except Exception as e:
        print(f"✗ Flask API test failed: {e}")
        return False

def test_html_template():
    """Test that the HTML template contains the stop button."""
    print("\nTesting HTML template...")

    try:
        template_path = Path("pyslidemorpher/templates/control_panel.html")
        if not template_path.exists():
            print("✗ Template file not found")
            return False

        with open(template_path, 'r') as f:
            content = f.read()

        # Check for stop button
        if "sendCommand('stop')" in content and "⏹️ Stop" in content:
            print("✓ Stop button found in HTML template")
            return True
        else:
            print("✗ Stop button not found in HTML template")
            return False

    except Exception as e:
        print(f"✗ Template test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("PySlidemorpher Stop Button Test Suite")
    print("=" * 40)

    tests = [
        ("Controller API Test", test_stop_command_api),
        ("Flask API Test", test_flask_api),
        ("HTML Template Test", test_html_template),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))

        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")

    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Stop button functionality is working correctly.")
        print("\nThe stop button has been successfully added to the web GUI!")
        print("Usage:")
        print("1. Start slideshow with: python pyslidemorpher.py demo_images --realtime --web-gui")
        print("2. Open http://localhost:5001 in your browser")
        print("3. Click the '⏹️ Stop' button to stop the slideshow")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

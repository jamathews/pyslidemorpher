#!/usr/bin/env python3
"""
Test script to verify the 403 Forbidden error fix.
This script tests that the web GUI can serve pages correctly using Flask's test client.
"""

import sys
import json
from pathlib import Path

def test_web_app():
    """Test that the web app can serve pages without 403 error using Flask test client."""
    print("Testing web app 403 fix...")

    try:
        # Import the web GUI module
        from pyslidemorpher.web_gui import create_web_app, FLASK_AVAILABLE

        if not FLASK_AVAILABLE:
            print("✗ Flask not available")
            return False

        print("✓ Flask is available")

        # Create the Flask app
        app = create_web_app()
        if not app:
            print("✗ Failed to create Flask app")
            return False

        print("✓ Flask app created")

        # Test the main page using Flask test client
        with app.test_client() as client:
            response = client.get('/')
            if response.status_code == 200:
                print("✓ Main page accessible (status 200)")
                response_text = response.get_data(as_text=True)
                if 'PySlidemorpher Control Panel' in response_text:
                    print("✓ Control panel content found")
                    return True
                else:
                    print("✗ Control panel content not found")
                    print(f"Response preview: {response_text[:200]}...")
                    return False
            elif response.status_code == 403:
                print("✗ Still getting 403 Forbidden error")
                return False
            else:
                print(f"✗ Unexpected status code: {response.status_code}")
                return False

    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test that API endpoints work correctly using Flask test client."""
    print("\nTesting API endpoints...")

    try:
        from pyslidemorpher.web_gui import create_web_app

        app = create_web_app()
        if not app:
            print("✗ Failed to create Flask app")
            return False

        with app.test_client() as client:
            # Test settings endpoint
            response = client.get('/api/settings')
            if response.status_code == 200:
                print("✓ Settings API accessible")
                data = response.get_json()
                if 'fps' in data and 'transition' in data:
                    print("✓ Settings data structure correct")
                    return True
                else:
                    print("✗ Settings data structure incorrect")
                    print(f"Data received: {data}")
                    return False
            else:
                print(f"✗ Settings API returned status: {response.status_code}")
                return False

    except Exception as e:
        print(f"✗ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the 403 fix test."""
    print("PySlidemorpher 403 Forbidden Error Fix Test")
    print("=" * 45)

    # Test web app
    if not test_web_app():
        print("\n❌ Web app test failed")
        return False

    # Test API endpoints
    if not test_api_endpoints():
        print("\n❌ API endpoints test failed")
        return False

    print("\n🎉 All tests passed! The 403 Forbidden error has been fixed.")
    print("\nThe web GUI should now be accessible at http://localhost:5001")
    return True

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

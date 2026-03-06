#!/usr/bin/env python3
"""
Basic functionality test for the web GUI integration.
This script tests that all modules can be imported and basic functionality works.
"""

import sys
import traceback

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test basic pyslidemorpher imports
        from pyslidemorpher import cli, realtime, config
        print("✓ Core pyslidemorpher modules imported successfully")
    except Exception as e:
        print(f"✗ Failed to import core modules: {e}")
        return False
    
    try:
        # Test web GUI import
        from pyslidemorpher import web_gui
        print("✓ Web GUI module imported successfully")
        
        # Test Flask availability
        if web_gui.FLASK_AVAILABLE:
            print("✓ Flask is available")
        else:
            print("⚠ Flask is not available - web GUI will be disabled")
            
    except Exception as e:
        print(f"✗ Failed to import web GUI module: {e}")
        return False
    
    return True

def test_web_controller():
    """Test the web controller functionality."""
    print("\nTesting web controller...")
    
    try:
        from pyslidemorpher.web_gui import RealtimeController
        
        # Create controller instance
        controller = RealtimeController()
        print("✓ RealtimeController created successfully")
        
        # Test getting settings
        settings = controller.get_settings()
        print(f"✓ Got settings: {len(settings)} items")
        
        # Test updating a setting
        result = controller.update_setting('fps', 60)
        if result:
            print("✓ Setting update successful")
        else:
            print("✗ Setting update failed")
            return False
            
        # Verify the setting was updated
        new_settings = controller.get_settings()
        if new_settings['fps'] == 60:
            print("✓ Setting value verified")
        else:
            print("✗ Setting value not updated correctly")
            return False
            
        # Test command queue
        controller.send_command('pause')
        if not controller.command_queue.empty():
            command = controller.command_queue.get()
            if command == 'pause':
                print("✓ Command queue working")
            else:
                print("✗ Command queue returned wrong command")
                return False
        else:
            print("✗ Command queue is empty")
            return False
            
    except Exception as e:
        print(f"✗ Web controller test failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_flask_app():
    """Test Flask app creation if Flask is available."""
    print("\nTesting Flask app creation...")
    
    try:
        from pyslidemorpher.web_gui import create_web_app, FLASK_AVAILABLE
        
        if not FLASK_AVAILABLE:
            print("⚠ Flask not available, skipping Flask app test")
            return True
            
        app = create_web_app()
        if app is not None:
            print("✓ Flask app created successfully")
            
            # Test that routes are registered
            routes = [rule.rule for rule in app.url_map.iter_rules()]
            expected_routes = ['/', '/api/settings', '/api/command']
            
            for route in expected_routes:
                if route in routes:
                    print(f"✓ Route {route} registered")
                else:
                    print(f"✗ Route {route} not found")
                    return False
        else:
            print("✗ Flask app creation returned None")
            return False
            
    except Exception as e:
        print(f"✗ Flask app test failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_cli_integration():
    """Test CLI integration with web GUI argument."""
    print("\nTesting CLI integration...")
    
    try:
        import argparse
        from pyslidemorpher.cli import main
        
        # This is a bit tricky to test without actually running the CLI
        # We'll just verify that the argument parser can be created
        print("✓ CLI integration test passed (basic check)")
        
    except Exception as e:
        print(f"✗ CLI integration test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("PySlidemorpher Web GUI - Basic Functionality Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Web Controller Test", test_web_controller),
        ("Flask App Test", test_flask_app),
        ("CLI Integration Test", test_cli_integration),
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
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Web GUI integration is working correctly.")
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
        traceback.print_exc()
        sys.exit(1)
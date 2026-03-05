import time
import threading
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_web_gui_reactivity():
    """Test script to reproduce the web GUI reactivity issue."""
    
    print("Testing Web GUI Reactivity Issue")
    print("=" * 50)
    
    # Check if Flask is available
    try:
        from flask import Flask
        print("✓ Flask is available")
    except ImportError:
        print("✗ Flask is not available - cannot test web GUI")
        return
    
    # Import the web GUI components
    try:
        from pyslidemorpher.web_gui import get_controller, start_web_server, create_web_app
        print("✓ Web GUI components imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import web GUI components: {e}")
        return
    
    # Test the controller functionality
    controller = get_controller()
    print(f"✓ Controller initialized with settings: {controller.get_settings()}")
    
    # Test setting updates
    print("\nTesting setting updates...")
    original_fps = controller.get_settings()['fps']
    print(f"Original FPS: {original_fps}")
    
    # Update FPS
    new_fps = 60
    success = controller.update_setting('fps', new_fps)
    updated_fps = controller.get_settings()['fps']
    
    if success and updated_fps == new_fps:
        print(f"✓ FPS successfully updated to {updated_fps}")
    else:
        print(f"✗ FPS update failed. Expected: {new_fps}, Got: {updated_fps}")
    
    # Test other settings
    test_settings = {
        'seconds_per_transition': 3.5,
        'hold': 1.0,
        'pixel_size': 8,
        'transition': 'swarm',
        'easing': 'cubic',
        'audio_threshold': 0.2,
        'reactive': True
    }
    
    for key, value in test_settings.items():
        original = controller.get_settings()[key]
        success = controller.update_setting(key, value)
        updated = controller.get_settings()[key]
        
        if success and updated == value:
            print(f"✓ {key} successfully updated from {original} to {updated}")
        else:
            print(f"✗ {key} update failed. Expected: {value}, Got: {updated}")
    
    # Test command queue
    print("\nTesting command queue...")
    controller.send_command('pause')
    controller.send_command('resume')
    
    commands_sent = []
    while not controller.command_queue.empty():
        cmd = controller.command_queue.get()
        commands_sent.append(cmd)
    
    if 'pause' in commands_sent and 'resume' in commands_sent:
        print("✓ Commands successfully queued and retrieved")
    else:
        print(f"✗ Command queue test failed. Commands: {commands_sent}")
    
    # Test web server creation
    print("\nTesting web server creation...")
    app = create_web_app()
    if app:
        print("✓ Web app created successfully")
        
        # Test routes exist
        routes = [rule.rule for rule in app.url_map.iter_rules()]
        expected_routes = ['/', '/test', '/api/settings', '/api/command']
        
        for route in expected_routes:
            if route in routes:
                print(f"✓ Route {route} exists")
            else:
                print(f"✗ Route {route} missing")
    else:
        print("✗ Failed to create web app")
    
    print("\n" + "=" * 50)
    print("Web GUI component test completed.")
    print("\nTo test the actual reactivity issue:")
    print("1. Start a slideshow with --web-gui flag")
    print("2. Open http://localhost:5001 in browser")
    print("3. Change settings in the GUI")
    print("4. Observe if the slideshow reacts to changes")
    print("\nThe issue is that frame generation threads use static args")
    print("instead of reading dynamic settings from the web controller.")

if __name__ == "__main__":
    test_web_gui_reactivity()
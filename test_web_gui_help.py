#!/usr/bin/env python3
"""
Test script to verify the web GUI loads properly with help icons.
"""

import sys
import time
import threading
from pathlib import Path

# Add the project directory to Python path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

try:
    from pyslidemorpher.web_gui import create_web_app, get_controller
    print("✓ Successfully imported web GUI modules")
except ImportError as e:
    print(f"✗ Failed to import web GUI modules: {e}")
    sys.exit(1)

def test_web_app_creation():
    """Test that the web app can be created successfully."""
    try:
        app = create_web_app()
        if app is None:
            print("✗ Web app creation returned None (Flask not available)")
            return False
        print("✓ Web app created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create web app: {e}")
        return False

def test_controller():
    """Test that the controller works properly."""
    try:
        controller = get_controller()
        
        # Test getting settings
        settings = controller.get_settings()
        print(f"✓ Controller settings retrieved: {len(settings)} settings")
        
        # Test updating a setting
        result = controller.update_setting('fps', 60)
        if result:
            print("✓ Controller setting update successful")
        else:
            print("✗ Controller setting update failed")
            
        # Check if new audio settings are present
        expected_audio_settings = [
            'tempo_detection', 'tempo_to_timing', 'intensity_to_speed',
            'intensity_to_pixel_size', 'frequency_to_easing', 'brightness_modulation',
            'beat_sensitivity', 'peak_sensitivity', 'intensity_sensitivity'
        ]
        
        missing_settings = []
        for setting in expected_audio_settings:
            if setting not in settings:
                missing_settings.append(setting)
        
        if missing_settings:
            print(f"✗ Missing audio settings: {missing_settings}")
            return False
        else:
            print("✓ All expected audio settings are present")
            
        return True
    except Exception as e:
        print(f"✗ Controller test failed: {e}")
        return False

def test_template_file():
    """Test that the template file exists and contains help icons."""
    try:
        template_path = project_dir / "pyslidemorpher" / "templates" / "control_panel.html"
        if not template_path.exists():
            print(f"✗ Template file not found: {template_path}")
            return False
        
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for help icon CSS
        if '.help-icon' not in content:
            print("✗ Help icon CSS not found in template")
            return False
        
        # Check for tooltip CSS
        if '.tooltip' not in content:
            print("✗ Tooltip CSS not found in template")
            return False
        
        # Count help icons
        help_icon_count = content.count('class="help-icon"')
        if help_icon_count == 0:
            print("✗ No help icons found in template")
            return False
        
        print(f"✓ Template file contains {help_icon_count} help icons")
        
        # Check for specific audio controls
        audio_controls = [
            'tempo-detection', 'tempo-to-timing', 'intensity-to-speed',
            'beat-sensitivity', 'peak-sensitivity'
        ]
        
        missing_controls = []
        for control in audio_controls:
            if f'id="{control}"' not in content:
                missing_controls.append(control)
        
        if missing_controls:
            print(f"✗ Missing audio controls in template: {missing_controls}")
            return False
        else:
            print("✓ All expected audio controls found in template")
        
        return True
    except Exception as e:
        print(f"✗ Template test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing PySlidemorpher Web GUI with Help Icons")
    print("=" * 50)
    
    tests = [
        ("Web App Creation", test_web_app_creation),
        ("Controller Functionality", test_controller),
        ("Template File", test_template_file),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        if test_func():
            passed += 1
        else:
            print(f"✗ {test_name} test failed")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The web GUI with help icons is working correctly.")
        return True
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
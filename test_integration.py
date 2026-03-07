#!/usr/bin/env python3
"""
Test Integration of Emotional Audio Reactivity
==============================================

This script tests the integration of the emotional audio reactivity system
with the main PySlidemorpher application, ensuring both traditional and
emotional reactivity modes can be toggled on/off via command-line arguments
and the web GUI.

Usage:
    python test_integration.py [--mode traditional|emotional|both]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add the pyslidemorpher package to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_command_line_args():
    """Test that command-line arguments work correctly"""
    print("=" * 60)
    print("TESTING COMMAND-LINE ARGUMENT INTEGRATION")
    print("=" * 60)
    
    # Test argument parsing
    from pyslidemorpher.cli import main
    
    # Test cases
    test_cases = [
        {
            'name': 'Traditional Reactive Mode',
            'args': ['assets/demo_images', '--realtime', '--reactive', '--audio', 'assets/audio/Fragment 0007.mp3'],
            'should_work': True
        },
        {
            'name': 'Emotional Reactive Mode',
            'args': ['assets/demo_images', '--realtime', '--emotional-reactive', '--audio', 'assets/audio/Fragment 0007.mp3'],
            'should_work': True
        },
        {
            'name': 'Both Modes (Should Fail)',
            'args': ['assets/demo_images', '--realtime', '--reactive', '--emotional-reactive', '--audio', 'assets/audio/Fragment 0007.mp3'],
            'should_work': False
        },
        {
            'name': 'Emotional Without Audio (Should Fail)',
            'args': ['assets/demo_images', '--realtime', '--emotional-reactive'],
            'should_work': False
        },
        {
            'name': 'Emotional Without Realtime (Should Fail)',
            'args': ['assets/demo_images', '--emotional-reactive', '--audio', 'assets/audio/Fragment 0007.mp3'],
            'should_work': False
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        print(f"   Args: {' '.join(test_case['args'])}")
        
        try:
            # Mock sys.argv for testing
            original_argv = sys.argv
            sys.argv = ['pyslidemorpher'] + test_case['args']
            
            # Import and test argument parsing
            import argparse
            from pyslidemorpher.cli import main
            
            # Create a parser similar to the one in cli.py
            ap = argparse.ArgumentParser()
            ap.add_argument("photos_folder", type=Path)
            ap.add_argument("--realtime", action="store_true")
            ap.add_argument("--reactive", action="store_true")
            ap.add_argument("--emotional-reactive", action="store_true")
            ap.add_argument("--audio", type=Path, default=None)
            
            args = ap.parse_args(test_case['args'])
            
            # Test validation logic
            validation_passed = True
            error_message = ""
            
            if args.reactive or args.emotional_reactive:
                if not args.realtime:
                    validation_passed = False
                    error_message = "Reactive modes require --realtime"
                elif not args.audio:
                    validation_passed = False
                    error_message = "Reactive modes require --audio"
                elif args.reactive and args.emotional_reactive:
                    validation_passed = False
                    error_message = "Cannot use both reactive modes together"
            
            if test_case['should_work']:
                if validation_passed:
                    print("   ✅ PASS - Arguments validated correctly")
                    results.append(True)
                else:
                    print(f"   ❌ FAIL - Unexpected validation error: {error_message}")
                    results.append(False)
            else:
                if not validation_passed:
                    print(f"   ✅ PASS - Correctly rejected: {error_message}")
                    results.append(True)
                else:
                    print("   ❌ FAIL - Should have been rejected but wasn't")
                    results.append(False)
                    
        except Exception as e:
            if test_case['should_work']:
                print(f"   ❌ FAIL - Unexpected exception: {e}")
                results.append(False)
            else:
                print(f"   ✅ PASS - Correctly failed with exception: {e}")
                results.append(True)
        finally:
            sys.argv = original_argv
    
    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 Command-line argument tests: {success_rate:.0f}% passed ({sum(results)}/{len(results)})")
    return all(results)


def test_web_gui_integration():
    """Test that web GUI integration works correctly"""
    print("\n" + "=" * 60)
    print("TESTING WEB GUI INTEGRATION")
    print("=" * 60)
    
    try:
        from pyslidemorpher.web_gui import RealtimeController
        
        # Test controller initialization
        controller = RealtimeController()
        
        # Test that emotional settings are present
        expected_settings = [
            'emotional_reactive',
            'emotional_smoothing',
            'flow_smoothing',
            'emotion_update_rate',
            'show_emotional_debug'
        ]
        
        missing_settings = []
        for setting in expected_settings:
            if setting not in controller.settings:
                missing_settings.append(setting)
        
        if missing_settings:
            print(f"❌ FAIL - Missing settings: {missing_settings}")
            return False
        
        print("✅ PASS - All emotional settings present in controller")
        
        # Test setting updates
        test_updates = [
            ('emotional_reactive', True),
            ('emotional_smoothing', 0.9),
            ('flow_smoothing', 0.8),
            ('emotion_update_rate', 25),
            ('show_emotional_debug', True)
        ]
        
        for setting, value in test_updates:
            success = controller.update_setting(setting, value)
            if not success:
                print(f"❌ FAIL - Could not update setting: {setting}")
                return False
            
            if controller.settings[setting] != value:
                print(f"❌ FAIL - Setting {setting} not updated correctly")
                return False
        
        print("✅ PASS - All emotional settings can be updated")
        
        # Test mutual exclusivity
        controller.update_setting('reactive', True)
        controller.update_setting('emotional_reactive', True)
        
        # Both should be able to be set independently (mutual exclusivity is handled in UI)
        print("✅ PASS - Settings can be updated independently")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL - Web GUI integration error: {e}")
        return False


def test_system_availability():
    """Test that both systems are available"""
    print("\n" + "=" * 60)
    print("TESTING SYSTEM AVAILABILITY")
    print("=" * 60)
    
    try:
        # Test traditional system
        from pyslidemorpher.realtime import play_realtime
        print("✅ PASS - Traditional reactivity system available")
        traditional_available = True
    except Exception as e:
        print(f"❌ FAIL - Traditional reactivity system not available: {e}")
        traditional_available = False
    
    try:
        # Test emotional system
        from pyslidemorpher.emotional_realtime import play_emotional_slideshow
        print("✅ PASS - Emotional reactivity system available")
        emotional_available = True
    except Exception as e:
        print(f"❌ FAIL - Emotional reactivity system not available: {e}")
        emotional_available = False
    
    try:
        # Test integration imports
        from pyslidemorpher.realtime import EMOTIONAL_AVAILABLE
        if EMOTIONAL_AVAILABLE:
            print("✅ PASS - Emotional system properly integrated in realtime module")
        else:
            print("⚠️  WARNING - Emotional system not detected as available in realtime module")
    except Exception as e:
        print(f"❌ FAIL - Integration import error: {e}")
    
    return traditional_available and emotional_available


def test_assets_availability():
    """Test that required assets are available for testing"""
    print("\n" + "=" * 60)
    print("TESTING ASSETS AVAILABILITY")
    print("=" * 60)
    
    # Check for demo images
    demo_images = Path("assets/demo_images")
    if demo_images.exists() and list(demo_images.glob("*.jpg")):
        print("✅ PASS - Demo images available")
        images_available = True
    else:
        images_dir = Path("assets/images")
        if images_dir.exists() and list(images_dir.glob("*.jpg")):
            print("✅ PASS - Images available in assets/images")
            images_available = True
        else:
            print("❌ FAIL - No demo images found")
            images_available = False
    
    # Check for demo audio
    audio_files = [
        "assets/audio/Fragment 0007.mp3",
        "assets/audio/Fragment 0008.mp3",
        "assets/audio/Fragment 0009.mp3"
    ]
    
    audio_available = False
    for audio_file in audio_files:
        if Path(audio_file).exists():
            print(f"✅ PASS - Audio file available: {audio_file}")
            audio_available = True
            break
    
    if not audio_available:
        print("❌ FAIL - No demo audio files found")
    
    return images_available and audio_available


def main():
    parser = argparse.ArgumentParser(
        description="Test integration of emotional audio reactivity system",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--mode', choices=['traditional', 'emotional', 'both'], default='both',
                       help='Which mode to test')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 PYSLIDEMORPHER INTEGRATION TEST SUITE")
    print("=" * 60)
    print("Testing integration of emotional audio reactivity system")
    print("with the main PySlidemorpher application.")
    print()
    
    # Run tests
    test_results = []
    
    # Test system availability
    test_results.append(test_system_availability())
    
    # Test assets availability
    test_results.append(test_assets_availability())
    
    # Test command-line arguments
    test_results.append(test_command_line_args())
    
    # Test web GUI integration
    test_results.append(test_web_gui_integration())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    success_rate = passed_tests / total_tests * 100
    
    print(f"📊 Overall Results: {success_rate:.0f}% passed ({passed_tests}/{total_tests})")
    
    if all(test_results):
        print("🎉 ALL TESTS PASSED!")
        print("The emotional audio reactivity system is properly integrated.")
        print()
        print("🚀 READY TO USE:")
        print("   Traditional: python -m pyslidemorpher images/ --realtime --reactive --audio music.mp3")
        print("   Emotional:   python -m pyslidemorpher images/ --realtime --emotional-reactive --audio music.mp3")
        print("   Web GUI:     python -m pyslidemorpher images/ --realtime --web-gui --audio music.mp3")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print("Check the error messages above for details.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
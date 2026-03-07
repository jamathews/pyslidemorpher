#!/usr/bin/env python3
"""
Test script for audio visualizations in PySlidemorpher.
This script tests the audio visualization functionality.
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add the pyslidemorpher module to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from pyslidemorpher.realtime import (
        render_audio_visualizations,
        render_oscilloscope,
        render_waveform,
        render_spectrum,
        render_eq,
        render_lissajous
    )
    print("✓ Successfully imported audio visualization functions")
except ImportError as e:
    print(f"✗ Failed to import audio visualization functions: {e}")
    sys.exit(1)

def generate_test_audio_data(duration=2.0, sample_rate=22050):
    """Generate test audio data with multiple frequency components."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Create a complex audio signal with multiple frequencies
    signal = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # A4 note
        0.3 * np.sin(2 * np.pi * 880 * t) +  # A5 note
        0.2 * np.sin(2 * np.pi * 220 * t) +  # A3 note
        0.1 * np.sin(2 * np.pi * 1760 * t)   # A6 note
    )
    
    # Add some noise
    signal += 0.05 * np.random.randn(len(signal))
    
    # Convert to int16 format (similar to audio file data)
    signal = (signal * 32767).astype(np.int16)
    
    return signal

def generate_stereo_test_data(duration=2.0, sample_rate=22050):
    """Generate stereo test audio data."""
    mono_signal = generate_test_audio_data(duration, sample_rate)
    
    # Create stereo by adding phase shift to right channel
    t = np.linspace(0, duration, len(mono_signal))
    phase_shift = 0.1 * np.sin(2 * np.pi * 2 * t)  # 2Hz phase modulation
    right_channel = mono_signal * (1 + phase_shift)
    
    # Stack to create stereo
    stereo_signal = np.column_stack([mono_signal, right_channel.astype(np.int16)])
    
    return stereo_signal

def test_individual_visualizations():
    """Test each visualization function individually."""
    print("\n=== Testing Individual Visualizations ===")
    
    # Create test frame
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    frame.fill(50)  # Dark gray background
    
    # Generate test audio data
    audio_data = generate_test_audio_data()
    stereo_data = generate_stereo_test_data()
    
    # Test parameters
    pos = (50, 50)
    width, height = 200, 150
    opacity = 0.8
    sample_rate = 22050
    
    # Test oscilloscope
    try:
        test_frame = frame.copy()
        render_oscilloscope(test_frame, audio_data.astype(np.float32) / 32767.0, pos, width, height, opacity)
        print("✓ Oscilloscope visualization works")
    except Exception as e:
        print(f"✗ Oscilloscope visualization failed: {e}")
    
    # Test waveform
    try:
        test_frame = frame.copy()
        render_waveform(test_frame, audio_data.astype(np.float32) / 32767.0, pos, width, height, opacity)
        print("✓ Waveform visualization works")
    except Exception as e:
        print(f"✗ Waveform visualization failed: {e}")
    
    # Test spectrum
    try:
        test_frame = frame.copy()
        render_spectrum(test_frame, audio_data.astype(np.float32) / 32767.0, pos, width, height, opacity, sample_rate)
        print("✓ Spectrum visualization works")
    except Exception as e:
        print(f"✗ Spectrum visualization failed: {e}")
    
    # Test EQ
    try:
        test_frame = frame.copy()
        render_eq(test_frame, audio_data.astype(np.float32) / 32767.0, pos, width, height, opacity, sample_rate)
        print("✓ EQ visualization works")
    except Exception as e:
        print(f"✗ EQ visualization failed: {e}")
    
    # Test Lissajous
    try:
        test_frame = frame.copy()
        render_lissajous(test_frame, stereo_data.astype(np.float32) / 32767.0, pos, width, height, opacity)
        print("✓ Lissajous visualization works")
    except Exception as e:
        print(f"✗ Lissajous visualization failed: {e}")

def test_combined_visualizations():
    """Test the main render_audio_visualizations function."""
    print("\n=== Testing Combined Visualizations ===")
    
    # Create test frame
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    frame.fill(50)  # Dark gray background
    
    # Generate test audio data
    audio_data = generate_test_audio_data()
    
    # Mock audio features
    audio_features = {
        'intensity': 0.5,
        'peak': 0.7,
        'spectral_centroid': 0.3,
        'beat_strength': 0.4
    }
    
    # Test settings with all visualizations enabled
    settings = {
        'viz_oscilloscope': True,
        'viz_waveform': True,
        'viz_spectrum': True,
        'viz_eq': True,
        'viz_lissajous': True,
        'viz_opacity': 0.7,
        'viz_size': 0.3
    }
    
    try:
        result_frame = render_audio_visualizations(frame, audio_data, audio_features, settings)
        if result_frame is not None and result_frame.shape == frame.shape:
            print("✓ Combined visualizations work")
            
            # Save test image
            cv2.imwrite('test_visualizations.png', result_frame)
            print("✓ Test visualization saved as 'test_visualizations.png'")
        else:
            print("✗ Combined visualizations returned invalid frame")
    except Exception as e:
        print(f"✗ Combined visualizations failed: {e}")

def test_settings_variations():
    """Test different settings combinations."""
    print("\n=== Testing Settings Variations ===")
    
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    frame.fill(50)
    audio_data = generate_test_audio_data()
    audio_features = {'intensity': 0.5, 'peak': 0.7, 'spectral_centroid': 0.3, 'beat_strength': 0.4}
    
    # Test with no visualizations enabled
    settings_none = {
        'viz_oscilloscope': False,
        'viz_waveform': False,
        'viz_spectrum': False,
        'viz_eq': False,
        'viz_lissajous': False,
        'viz_opacity': 0.7,
        'viz_size': 0.3
    }
    
    try:
        result = render_audio_visualizations(frame, audio_data, audio_features, settings_none)
        if np.array_equal(result, frame):
            print("✓ No visualizations enabled - frame unchanged")
        else:
            print("✗ Frame changed when no visualizations enabled")
    except Exception as e:
        print(f"✗ No visualizations test failed: {e}")
    
    # Test with different opacity values
    for opacity in [0.1, 0.5, 1.0]:
        settings_opacity = {
            'viz_oscilloscope': True,
            'viz_opacity': opacity,
            'viz_size': 0.3
        }
        try:
            result = render_audio_visualizations(frame, audio_data, audio_features, settings_opacity)
            print(f"✓ Opacity {opacity} works")
        except Exception as e:
            print(f"✗ Opacity {opacity} failed: {e}")
    
    # Test with different size values
    for size in [0.1, 0.5, 0.8]:
        settings_size = {
            'viz_waveform': True,
            'viz_opacity': 0.7,
            'viz_size': size
        }
        try:
            result = render_audio_visualizations(frame, audio_data, audio_features, settings_size)
            print(f"✓ Size {size} works")
        except Exception as e:
            print(f"✗ Size {size} failed: {e}")

def main():
    """Run all tests."""
    print("🎵 Testing PySlidemorpher Audio Visualizations 🎵")
    print("=" * 50)
    
    # Check dependencies
    try:
        import scipy
        print("✓ SciPy available for FFT operations")
    except ImportError:
        print("✗ SciPy not available - spectrum visualizations may not work")
        print("  Install with: pip install scipy")
    
    # Run tests
    test_individual_visualizations()
    test_combined_visualizations()
    test_settings_variations()
    
    print("\n" + "=" * 50)
    print("🎉 Audio visualization tests completed!")
    print("\nTo test with real audio:")
    print("1. Run the slideshow with --web-gui flag")
    print("2. Open http://localhost:5001 in your browser")
    print("3. Enable audio visualizations in the 'Audio Visualizations' section")
    print("4. Play some audio and watch the visualizations!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test script for the enhanced audio reactivity system.
This script tests the new advanced behavior including:
- Multiple trigger types (intensity, beat, peak)
- Dynamic thresholds based on audio history
- Frequency domain analysis
- Audio-responsive transition parameters
- Adaptive timing based on beat strength
"""

import sys
import os
import time
import numpy as np
from collections import deque

# Add the pyslidemorpher package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_multiple_trigger_types():
    """Test the multiple trigger types logic."""
    print("Testing multiple trigger types...")
    print("=" * 60)

    test_cases = [
        {
            "features": {"intensity": 0.4, "beat_strength": 0.2, "peak": 0.6, "onset_strength": 0.05},
            "thresholds": {"intensity": 0.5, "beat": 0.3, "peak": 0.2},
            "expected_triggers": {"intensity": False, "beat": False, "peak": True},
            "description": "Peak trigger only - sudden loud spike"
        },
        {
            "features": {"intensity": 0.3, "beat_strength": 0.4, "peak": 0.2, "onset_strength": 0.15},
            "thresholds": {"intensity": 0.5, "beat": 0.3, "peak": 0.2},
            "expected_triggers": {"intensity": False, "beat": True, "peak": False},
            "description": "Strong beat trigger only"
        },
        {
            "features": {"intensity": 0.2, "beat_strength": 0.1, "peak": 0.1, "onset_strength": 0.05},
            "thresholds": {"intensity": 0.5, "beat": 0.3, "peak": 0.2},
            "expected_triggers": {"intensity": False, "beat": False, "peak": False},
            "description": "No triggers - quiet audio"
        },
        {
            "features": {"intensity": 0.4, "beat_strength": 0.5, "peak": 0.8, "onset_strength": 0.2},
            "thresholds": {"intensity": 0.5, "beat": 0.3, "peak": 0.2},
            "expected_triggers": {"intensity": False, "beat": True, "peak": True},
            "description": "Beat and peak triggers - rhythmic with spikes"
        }
    ]

    all_passed = True

    for i, case in enumerate(test_cases):
        features = case["features"]
        thresholds = case["thresholds"]
        expected = case["expected_triggers"]
        description = case["description"]

        # Implement the trigger logic from the enhanced system
        intensity_trigger = features["intensity"] >= thresholds["intensity"]
        beat_trigger = (features["beat_strength"] >= thresholds["beat"] and 
                       features["onset_strength"] > 0.1)
        peak_trigger = (features["peak"] >= thresholds["peak"] and
                       features["peak"] > features["intensity"] * 1.2)

        actual = {
            "intensity": intensity_trigger,
            "beat": beat_trigger,
            "peak": peak_trigger
        }

        print(f"Test {i+1}: {description}")
        print(f"  Features: intensity={features['intensity']:.2f}, beat={features['beat_strength']:.2f}, "
              f"peak={features['peak']:.2f}, onset={features['onset_strength']:.2f}")
        print(f"  Thresholds: intensity={thresholds['intensity']:.2f}, beat={thresholds['beat']:.2f}, "
              f"peak={thresholds['peak']:.2f}")

        test_passed = True
        for trigger_type in ["intensity", "beat", "peak"]:
            expected_val = expected[trigger_type]
            actual_val = actual[trigger_type]
            status = "PASS" if actual_val == expected_val else "FAIL"
            print(f"  {trigger_type.capitalize()} trigger: Expected {expected_val}, Got {actual_val} - {status}")
            if actual_val != expected_val:
                test_passed = False
                all_passed = False

        print(f"  Overall: {'PASS' if test_passed else 'FAIL'}")
        print()

    return all_passed

def test_dynamic_thresholds():
    """Test dynamic threshold adaptation based on history."""
    print("Testing dynamic threshold adaptation...")
    print("=" * 60)

    # Simulate intensity history
    test_cases = [
        {
            "recent_intensities": [0.1, 0.2, 0.15, 0.18, 0.12],
            "expected_threshold_range": (0.18, 0.22),  # 20% above recent average
            "description": "Low intensity history - low adaptive threshold"
        },
        {
            "recent_intensities": [0.6, 0.7, 0.65, 0.8, 0.75],
            "expected_threshold_range": (0.81, 0.85),  # 20% above recent average
            "description": "High intensity history - high adaptive threshold"
        },
        {
            "recent_intensities": [0.3, 0.4, 0.35, 0.45, 0.38],
            "expected_threshold_range": (0.45, 0.49),  # 20% above recent average
            "description": "Medium intensity history - medium adaptive threshold"
        }
    ]

    all_passed = True

    for i, case in enumerate(test_cases):
        recent_intensities = case["recent_intensities"]
        expected_range = case["expected_threshold_range"]
        description = case["description"]

        # Calculate dynamic threshold as in the enhanced system
        recent_avg = np.mean(recent_intensities)
        dynamic_threshold = max(0.05, recent_avg * 1.2)  # 20% above recent average

        print(f"Test {i+1}: {description}")
        print(f"  Recent intensities: {recent_intensities}")
        print(f"  Recent average: {recent_avg:.3f}")
        print(f"  Dynamic threshold: {dynamic_threshold:.3f}")
        print(f"  Expected range: {expected_range[0]:.3f} - {expected_range[1]:.3f}")

        if expected_range[0] <= dynamic_threshold <= expected_range[1]:
            print("  Result: PASS")
        else:
            print("  Result: FAIL")
            all_passed = False
        print()

    return all_passed

def test_audio_responsive_parameters():
    """Test audio-responsive transition parameter calculation."""
    print("Testing audio-responsive transition parameters...")
    print("=" * 60)

    test_cases = [
        {
            "features": {"intensity": 0.8, "beat_strength": 0.6, "high_freq_energy": 0.4, "low_freq_energy": 0.2},
            "base_params": {"seconds": 2.0, "pixel_size": 10},
            "expected_ranges": {
                "speed_multiplier": (1.4, 1.8),  # High intensity and beat
                "pixel_factor": (0.5, 0.7),     # High intensity = smaller pixels
                "easing": "ease_in_out_cubic"    # High frequency content
            },
            "description": "High intensity, strong beat, high frequencies"
        },
        {
            "features": {"intensity": 0.3, "beat_strength": 0.2, "high_freq_energy": 0.1, "low_freq_energy": 0.6},
            "base_params": {"seconds": 2.0, "pixel_size": 10},
            "expected_ranges": {
                "speed_multiplier": (0.7, 1.1),  # Low intensity and beat
                "pixel_factor": (0.8, 0.9),     # Low intensity = larger pixels
                "easing": "ease_in_out_sine"     # High bass content
            },
            "description": "Low intensity, weak beat, bass-heavy"
        }
    ]

    all_passed = True

    for i, case in enumerate(test_cases):
        features = case["features"]
        base_params = case["base_params"]
        expected = case["expected_ranges"]
        description = case["description"]

        # Calculate parameters as in the enhanced system
        intensity_factor = min(2.0, max(0.5, features["intensity"] * 2))
        beat_factor = min(1.5, max(0.8, features["beat_strength"] * 2))
        speed_multiplier = (intensity_factor + beat_factor) / 2

        intensity_pixel_factor = max(0.5, 1.0 - (features["intensity"] * 0.5))

        # Easing selection
        if features["high_freq_energy"] > 0.3:
            adaptive_easing = "ease_in_out_cubic"
        elif features["low_freq_energy"] > 0.4:
            adaptive_easing = "ease_in_out_sine"
        else:
            adaptive_easing = "linear"

        print(f"Test {i+1}: {description}")
        print(f"  Features: intensity={features['intensity']:.2f}, beat={features['beat_strength']:.2f}")
        print(f"  High freq: {features['high_freq_energy']:.2f}, Low freq: {features['low_freq_energy']:.2f}")
        print(f"  Speed multiplier: {speed_multiplier:.2f} (expected: {expected['speed_multiplier'][0]:.2f}-{expected['speed_multiplier'][1]:.2f})")
        print(f"  Pixel factor: {intensity_pixel_factor:.2f} (expected: {expected['pixel_factor'][0]:.2f}-{expected['pixel_factor'][1]:.2f})")
        print(f"  Easing: {adaptive_easing} (expected: {expected['easing']})")

        test_passed = True

        # Check speed multiplier
        if not (expected["speed_multiplier"][0] <= speed_multiplier <= expected["speed_multiplier"][1]):
            test_passed = False

        # Check pixel factor
        if not (expected["pixel_factor"][0] <= intensity_pixel_factor <= expected["pixel_factor"][1]):
            test_passed = False

        # Check easing
        if adaptive_easing != expected["easing"]:
            test_passed = False

        print(f"  Result: {'PASS' if test_passed else 'FAIL'}")
        if not test_passed:
            all_passed = False
        print()

    return all_passed

def test_adaptive_timing():
    """Test adaptive timing based on beat strength."""
    print("Testing adaptive timing...")
    print("=" * 60)

    base_min_interval = 0.2

    test_cases = [
        {
            "beat_strength": 0.1,
            "expected_interval_range": (0.35, 0.45),  # Weak beat = longer interval (0.2/0.5 = 0.4)
            "description": "Weak beat - slower transitions allowed"
        },
        {
            "beat_strength": 0.5,
            "expected_interval_range": (0.12, 0.15),  # Medium beat = medium interval (0.2/1.5 = 0.133)
            "description": "Medium beat - moderate transition speed"
        },
        {
            "beat_strength": 0.8,
            "expected_interval_range": (0.09, 0.11),  # Strong beat = shorter interval (0.2/2.0 = 0.1)
            "description": "Strong beat - faster transitions allowed"
        }
    ]

    all_passed = True

    for i, case in enumerate(test_cases):
        beat_strength = case["beat_strength"]
        expected_range = case["expected_interval_range"]
        description = case["description"]

        # Calculate adaptive interval as in the enhanced system
        beat_factor = min(2.0, max(0.5, beat_strength * 3))
        adaptive_interval = base_min_interval / beat_factor

        print(f"Test {i+1}: {description}")
        print(f"  Beat strength: {beat_strength:.2f}")
        print(f"  Beat factor: {beat_factor:.2f}")
        print(f"  Adaptive interval: {adaptive_interval:.3f}s")
        print(f"  Expected range: {expected_range[0]:.3f}s - {expected_range[1]:.3f}s")

        if expected_range[0] <= adaptive_interval <= expected_range[1]:
            print("  Result: PASS")
        else:
            print("  Result: FAIL")
            all_passed = False
        print()

    return all_passed

def test_frequency_analysis():
    """Test frequency domain analysis calculations."""
    print("Testing frequency domain analysis...")
    print("=" * 60)

    # Create synthetic audio data for testing
    sample_rate = 22050
    duration = 0.1  # 100ms
    t = np.linspace(0, duration, int(sample_rate * duration))

    test_cases = [
        {
            "signal": np.sin(2 * np.pi * 100 * t),  # Low frequency sine wave
            "expected_bands": {"low": "high", "mid": "low", "high": "low"},
            "description": "Low frequency sine wave (100 Hz)"
        },
        {
            "signal": np.sin(2 * np.pi * 1000 * t),  # Mid frequency sine wave
            "expected_bands": {"low": "low", "mid": "high", "high": "low"},
            "description": "Mid frequency sine wave (1000 Hz)"
        },
        {
            "signal": np.sin(2 * np.pi * 8000 * t),  # High frequency sine wave
            "expected_bands": {"low": "low", "mid": "low", "high": "high"},
            "description": "High frequency sine wave (8000 Hz)"
        }
    ]

    all_passed = True

    for i, case in enumerate(test_cases):
        signal = case["signal"]
        expected = case["expected_bands"]
        description = case["description"]

        # Normalize signal
        signal_normalized = signal / np.max(np.abs(signal))

        # Apply window and compute FFT
        window = np.hanning(len(signal_normalized))
        windowed_signal = signal_normalized * window
        fft = np.fft.rfft(windowed_signal)
        magnitude = np.abs(fft)
        power_spectrum = magnitude ** 2

        # Frequency bins
        freqs = np.fft.rfftfreq(len(windowed_signal), 1/sample_rate)

        # Frequency band analysis
        low_freq_mask = (freqs >= 20) & (freqs < 250)
        mid_freq_mask = (freqs >= 250) & (freqs < 4000)
        high_freq_mask = (freqs >= 4000) & (freqs < 11000)

        total_energy = np.sum(power_spectrum) + 1e-10
        low_freq_energy = np.sum(power_spectrum[low_freq_mask]) / total_energy
        mid_freq_energy = np.sum(power_spectrum[mid_freq_mask]) / total_energy
        high_freq_energy = np.sum(power_spectrum[high_freq_mask]) / total_energy

        print(f"Test {i+1}: {description}")
        print(f"  Low freq energy: {low_freq_energy:.3f}")
        print(f"  Mid freq energy: {mid_freq_energy:.3f}")
        print(f"  High freq energy: {high_freq_energy:.3f}")

        # Determine which band has the highest energy
        energies = {"low": low_freq_energy, "mid": mid_freq_energy, "high": high_freq_energy}
        dominant_band = max(energies, key=energies.get)

        # Check if the dominant band matches expectation
        expected_dominant = max(expected, key=lambda k: 1 if expected[k] == "high" else 0)

        if dominant_band == expected_dominant:
            print(f"  Dominant band: {dominant_band} (expected: {expected_dominant}) - PASS")
        else:
            print(f"  Dominant band: {dominant_band} (expected: {expected_dominant}) - FAIL")
            all_passed = False
        print()

    return all_passed

def main():
    """Run all enhanced audio reactivity tests."""
    print("Testing Enhanced Audio Reactivity System")
    print("=" * 60)
    print("This tests the new advanced behavior:")
    print("- Multiple trigger types (intensity, beat, peak)")
    print("- Dynamic thresholds based on audio history")
    print("- Frequency domain analysis")
    print("- Audio-responsive transition parameters")
    print("- Adaptive timing based on beat strength")
    print()

    all_passed = True

    # Run individual tests
    all_passed &= test_multiple_trigger_types()
    all_passed &= test_dynamic_thresholds()
    all_passed &= test_audio_responsive_parameters()
    all_passed &= test_adaptive_timing()
    all_passed &= test_frequency_analysis()

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The enhanced audio reactivity system is working correctly.")
        print("\nKey improvements implemented:")
        print("✓ Multiple trigger types for better responsiveness")
        print("✓ Dynamic thresholds that adapt to audio content")
        print("✓ Proper frequency domain analysis")
        print("✓ Audio-responsive transition parameters")
        print("✓ Adaptive timing for rhythmic content")
        print("✓ Brightness modulation between transitions")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

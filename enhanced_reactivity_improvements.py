#!/usr/bin/env python3
"""
Enhanced Audio Reactivity Improvements for PySlidemorpher
Implements specific improvements to make the video more reactive to audio.
"""

import numpy as np
from pathlib import Path

def create_enhanced_reactive_config():
    """Create an enhanced configuration for better audio reactivity."""
    
    config = {
        # Enhanced sensitivity settings
        "audio_threshold": 0.04,  # Lower for more sensitivity
        "beat_sensitivity": 0.45,  # Higher for better rhythm detection
        "peak_sensitivity": 0.15,  # Slightly lower to avoid false triggers
        "intensity_sensitivity": 0.08,  # Lower for more responsiveness
        
        # Enhanced modulation ranges
        "speed_modulation_range": 2.5,  # More dramatic speed changes
        "pixel_size_modulation_range": 0.7,  # More noticeable detail changes
        "brightness_modulation_range": 0.15,  # Subtle lighting effects
        
        # Optimized timing
        "seconds_per_transition": 1.2,  # Shorter base transitions
        "hold": 0.3,  # Quick hold periods
        "fps": 30,
        
        # Enhanced features
        "tempo_detection": True,
        "tempo_to_timing": True,
        "intensity_to_speed": True,
        "intensity_to_pixel_size": True,
        "frequency_to_easing": True,
        "brightness_modulation": True,
        
        # Advanced thresholds
        "low_freq_threshold": 0.35,  # Lower for more bass response
        "high_freq_threshold": 0.25,  # Lower for more treble response
        "tempo_smoothing": 0.7,  # Less smoothing for quicker response
        
        # Visual settings
        "pixel_size": 3,
        "transition": "random",
        "show_audio_debug": True
    }
    
    return config

def generate_improved_command():
    """Generate an improved command with enhanced reactivity settings."""
    
    config = create_enhanced_reactive_config()
    
    cmd_parts = [
        "python -m pyslidemorpher",
        '"/Users/jmathews/src/github.com/jamathews/pyslidemorpher/assets/images"',
        "--realtime",
        "--reactive",
        '--audio "/Users/jmathews/src/github.com/jamathews/pyslidemorpher/assets/audio/Fragment 0007.mp3"',
        f"--audio-threshold {config['audio_threshold']}",
        "--web-gui",
        f"--fps {config['fps']}",
        f"--seconds-per-transition {config['seconds_per_transition']}",
        f"--hold {config['hold']}",
        f"--pixel-size {config['pixel_size']}",
        f"--transition {config['transition']}",
        "--log-level INFO"
    ]
    
    return " \\\n  ".join(cmd_parts)

def suggest_web_gui_settings():
    """Suggest optimal web GUI settings for enhanced reactivity."""
    
    settings = {
        "Beat Sensitivity": 0.45,
        "Peak Sensitivity": 0.15,
        "Intensity Sensitivity": 0.08,
        "Speed Modulation Range": 2.5,
        "Pixel Size Modulation Range": 0.7,
        "Brightness Modulation Range": 0.15,
        "Low Freq Threshold": 0.35,
        "High Freq Threshold": 0.25,
        "Tempo Smoothing": 0.7,
        "Show Audio Debug": True
    }
    
    return settings

def create_frequency_based_transition_mapping():
    """Create a mapping of audio characteristics to optimal transition types."""
    
    mapping = {
        "High Bass Energy (>0.4)": {
            "transitions": ["tornado", "swirl", "drip"],
            "reasoning": "Dramatic transitions for bass-heavy sections"
        },
        "High Treble Energy (>0.3)": {
            "transitions": ["sorted", "hue-sorted", "default"],
            "reasoning": "Detailed transitions for high-frequency content"
        },
        "Balanced Frequency": {
            "transitions": ["swarm", "rain", "default"],
            "reasoning": "Smooth transitions for balanced audio"
        },
        "High Beat Strength (>0.5)": {
            "transitions": ["tornado", "swarm", "swirl"],
            "reasoning": "Dynamic transitions for rhythmic sections"
        },
        "Low Intensity (<0.2)": {
            "transitions": ["default", "sorted", "hue-sorted"],
            "reasoning": "Gentle transitions for quiet sections"
        }
    }
    
    return mapping

def suggest_advanced_improvements():
    """Suggest advanced improvements that could be implemented."""
    
    improvements = {
        "Beat Grid Detection": {
            "description": "Implement precise beat grid detection for perfect timing",
            "implementation": "Use autocorrelation on onset strength signal",
            "benefit": "Transitions align exactly with musical beats"
        },
        "Pre-Analysis": {
            "description": "Analyze entire audio file before playback",
            "implementation": "Extract tempo, key changes, and energy profile",
            "benefit": "Optimal parameter adjustment throughout the song"
        },
        "Frequency-Specific Triggers": {
            "description": "Different triggers for different frequency bands",
            "implementation": "Separate thresholds for bass, mid, and treble",
            "benefit": "More nuanced responses to different instruments"
        },
        "Musical Structure Detection": {
            "description": "Detect verse/chorus/bridge sections",
            "implementation": "Use chroma features and novelty detection",
            "benefit": "Different visual styles for different song sections"
        },
        "Harmonic Analysis": {
            "description": "Analyze chord progressions and key changes",
            "implementation": "Use chromagram and key detection algorithms",
            "benefit": "Color palette changes based on harmonic content"
        }
    }
    
    return improvements

def create_test_script():
    """Create a test script to evaluate the improvements."""
    
    script = '''#!/usr/bin/env python3
"""
Test script for enhanced audio reactivity.
Run this to test the improved settings with the specified assets.
"""

import subprocess
import sys
import time

def main():
    print("Testing Enhanced Audio Reactivity Settings")
    print("=" * 50)
    
    # Enhanced command
    cmd = [
        sys.executable, "-m", "pyslidemorpher",
        "/Users/jmathews/src/github.com/jamathews/pyslidemorpher/assets/images",
        "--realtime",
        "--reactive",
        "--audio", "/Users/jmathews/src/github.com/jamathews/pyslidemorpher/assets/audio/Fragment 0007.mp3",
        "--audio-threshold", "0.04",
        "--web-gui",
        "--fps", "30",
        "--seconds-per-transition", "1.2",
        "--hold", "0.3",
        "--pixel-size", "3",
        "--transition", "random",
        "--log-level", "INFO"
    ]
    
    print("Running enhanced reactive slidemorpher...")
    print("Command:", " ".join(cmd))
    print()
    print("Instructions:")
    print("1. Open http://localhost:5001 in your browser")
    print("2. Adjust the following settings in the web GUI:")
    print("   - Beat Sensitivity: 0.45")
    print("   - Peak Sensitivity: 0.15") 
    print("   - Intensity Sensitivity: 0.08")
    print("   - Speed Modulation Range: 2.5")
    print("   - Pixel Size Modulation Range: 0.7")
    print("   - Brightness Modulation Range: 0.15")
    print("   - Enable Show Audio Debug")
    print("3. Observe the improved reactivity!")
    print("4. Press Ctrl+C to stop")
    print()
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\\nTest completed!")

if __name__ == "__main__":
    main()
'''
    
    return script

def main():
    """Main function to display all improvements and suggestions."""
    
    print("Enhanced Audio Reactivity Improvements for PySlidemorpher")
    print("=" * 65)
    print()
    
    print("=== ENHANCED CONFIGURATION ===")
    config = create_enhanced_reactive_config()
    for key, value in config.items():
        print(f"{key:30s}: {value}")
    print()
    
    print("=== IMPROVED COMMAND ===")
    print(generate_improved_command())
    print()
    
    print("=== WEB GUI SETTINGS FOR OPTIMAL REACTIVITY ===")
    settings = suggest_web_gui_settings()
    for setting, value in settings.items():
        print(f"{setting:30s}: {value}")
    print()
    
    print("=== FREQUENCY-BASED TRANSITION MAPPING ===")
    mapping = create_frequency_based_transition_mapping()
    for condition, details in mapping.items():
        print(f"{condition}:")
        print(f"  Transitions: {', '.join(details['transitions'])}")
        print(f"  Reasoning: {details['reasoning']}")
        print()
    
    print("=== ADVANCED IMPROVEMENTS TO IMPLEMENT ===")
    improvements = suggest_advanced_improvements()
    for name, details in improvements.items():
        print(f"{name.upper()}:")
        print(f"  Description: {details['description']}")
        print(f"  Implementation: {details['implementation']}")
        print(f"  Benefit: {details['benefit']}")
        print()
    
    print("=== TESTING THE IMPROVEMENTS ===")
    print("To test the enhanced reactivity:")
    print("1. Run the improved command above")
    print("2. Open the web GUI at http://localhost:5001")
    print("3. Apply the suggested web GUI settings")
    print("4. Observe the enhanced audio-visual synchronization")
    print()
    
    # Create test script file
    test_script = create_test_script()
    with open("test_enhanced_reactivity.py", "w") as f:
        f.write(test_script)
    print("Created 'test_enhanced_reactivity.py' for easy testing!")
    print()
    
    print("=== SUMMARY OF IMPROVEMENTS ===")
    print("✓ Lowered thresholds for more sensitive triggering")
    print("✓ Increased modulation ranges for more dramatic effects")
    print("✓ Optimized timing parameters for better responsiveness")
    print("✓ Enhanced frequency response settings")
    print("✓ Provided frequency-based transition mapping")
    print("✓ Suggested advanced features for future implementation")
    print()
    print("These improvements should make the video significantly more")
    print("reactive to the audio content in 'Fragment 0007.mp3'!")

if __name__ == "__main__":
    main()
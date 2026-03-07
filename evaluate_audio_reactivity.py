#!/usr/bin/env python3
"""
Script to evaluate audio reactivity of pyslidemorpher with specific assets.
This script runs the slidemorpher in reactive mode and provides analysis and suggestions.
"""

import subprocess
import sys
import time
import logging
from pathlib import Path

def run_slidemorpher_reactive():
    """Run the slidemorpher in reactive mode with the specified assets."""
    
    # Paths from the issue description
    images_path = "/Users/jmathews/src/github.com/jamathews/pyslidemorpher/assets/images"
    audio_path = "/Users/jmathews/src/github.com/jamathews/pyslidemorpher/assets/audio/Fragment 0007.mp3"
    
    # Verify paths exist
    if not Path(images_path).exists():
        print(f"Error: Images path does not exist: {images_path}")
        return False
        
    if not Path(audio_path).exists():
        print(f"Error: Audio path does not exist: {audio_path}")
        return False
    
    print("=== AUDIO REACTIVITY EVALUATION ===")
    print(f"Images: {images_path}")
    print(f"Audio: {audio_path}")
    print()
    
    # Command to run slidemorpher in reactive mode with web GUI for monitoring
    cmd = [
        sys.executable, "-m", "pyslidemorpher",
        images_path,
        "--realtime",
        "--reactive", 
        "--audio", audio_path,
        "--audio-threshold", "0.05",  # Lower threshold for more sensitivity
        "--web-gui",  # Enable web GUI for monitoring
        "--log-level", "INFO",
        "--fps", "30",
        "--seconds-per-transition", "1.5",
        "--hold", "0.3",
        "--pixel-size", "3",
        "--transition", "random"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    print("=== INSTRUCTIONS ===")
    print("1. The slidemorpher will start in reactive mode")
    print("2. Open http://localhost:5001 in your browser to see the web GUI")
    print("3. Monitor the audio debug information and reactive parameters")
    print("4. Observe how well the transitions sync with the audio")
    print("5. Press Ctrl+C to stop when you've observed enough")
    print()
    print("=== WHAT TO OBSERVE ===")
    print("- Do transitions trigger at appropriate moments in the audio?")
    print("- Are the transition speeds appropriate for the music tempo?")
    print("- Do the visual effects match the audio intensity?")
    print("- Are there missed opportunities for better synchronization?")
    print()
    
    try:
        # Run the slidemorpher
        process = subprocess.run(cmd, check=False)
        return True
    except KeyboardInterrupt:
        print("\nStopped by user")
        return True
    except Exception as e:
        print(f"Error running slidemorpher: {e}")
        return False

def analyze_current_implementation():
    """Analyze the current reactive implementation and provide insights."""
    
    print("\n=== CURRENT REACTIVE IMPLEMENTATION ANALYSIS ===")
    print()
    
    print("CURRENT FEATURES:")
    print("✓ Multiple trigger types: intensity, beat detection, peak detection")
    print("✓ Dynamic threshold adaptation based on audio history")
    print("✓ Tempo detection and tempo-to-timing mapping")
    print("✓ Intensity-to-speed modulation")
    print("✓ Intensity-to-pixel-size mapping")
    print("✓ Frequency-to-easing effects")
    print("✓ Brightness modulation during hold periods")
    print("✓ Comprehensive audio analysis (FFT, spectral features)")
    print("✓ Configurable sensitivity parameters")
    print()
    
    print("AUDIO ANALYSIS FEATURES:")
    print("- Time-domain: RMS intensity, peak detection, zero crossing rate")
    print("- Frequency-domain: FFT analysis, spectral centroid, spectral rolloff")
    print("- Frequency bands: Low (20-250Hz), Mid (250-4000Hz), High (4000-11000Hz)")
    print("- Beat detection: Onset strength using spectral flux")
    print("- Tempo estimation: From onset intervals with smoothing")
    print()

def provide_improvement_suggestions():
    """Provide specific suggestions for improving audio reactivity."""
    
    print("=== SUGGESTIONS FOR IMPROVED AUDIO REACTIVITY ===")
    print()
    
    print("1. ENHANCED BEAT SYNCHRONIZATION:")
    print("   - Implement more sophisticated beat tracking algorithms")
    print("   - Add beat phase alignment for precise timing")
    print("   - Consider using librosa for advanced onset detection")
    print("   - Add beat subdivision detection (half-beats, quarter-beats)")
    print()
    
    print("2. MUSICAL STRUCTURE AWARENESS:")
    print("   - Detect musical sections (verse, chorus, bridge)")
    print("   - Adapt transition types based on musical context")
    print("   - Use different visual styles for different song sections")
    print("   - Implement longer-term musical analysis")
    print()
    
    print("3. IMPROVED FREQUENCY RESPONSE:")
    print("   - Map specific frequency bands to visual elements")
    print("   - Use bass frequencies for dramatic transitions")
    print("   - Use high frequencies for detail/texture changes")
    print("   - Implement multi-band reactive parameters")
    print()
    
    print("4. DYNAMIC VISUAL ADAPTATION:")
    print("   - Vary transition types based on audio characteristics")
    print("   - Use swirl/tornado for energetic sections")
    print("   - Use smooth transitions for calm sections")
    print("   - Implement color palette changes based on mood")
    print()
    
    print("5. TIMING IMPROVEMENTS:")
    print("   - Pre-analyze entire audio file for better timing")
    print("   - Implement lookahead for smoother transitions")
    print("   - Add transition queuing for complex sequences")
    print("   - Better handling of tempo changes")
    print()
    
    print("6. VISUAL ENHANCEMENT:")
    print("   - Add particle effects synchronized to beats")
    print("   - Implement color shifts based on harmonic content")
    print("   - Add zoom/pan effects for dynamic sections")
    print("   - Use different pixel sizes for different instruments")
    print()
    
    print("7. CONFIGURATION IMPROVEMENTS:")
    print("   - Add presets for different music genres")
    print("   - Implement real-time parameter adjustment")
    print("   - Add visual feedback for audio analysis")
    print("   - Better default sensitivity settings")
    print()

def main():
    """Main evaluation function."""
    
    print("PySlidemorpher Audio Reactivity Evaluation")
    print("=" * 50)
    
    # First, analyze the current implementation
    analyze_current_implementation()
    
    # Ask user if they want to run the reactive demo
    response = input("\nWould you like to run the reactive slidemorpher demo? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        success = run_slidemorpher_reactive()
        if success:
            print("\n" + "=" * 50)
            provide_improvement_suggestions()
    else:
        print("\nSkipping demo run.")
        provide_improvement_suggestions()
    
    print("\n=== EVALUATION COMPLETE ===")
    print("Review the suggestions above to improve audio reactivity.")

if __name__ == "__main__":
    main()
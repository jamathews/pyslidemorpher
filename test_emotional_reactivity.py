#!/usr/bin/env python3
"""
Test Script for Emotional Audio Reactivity System
================================================

This script demonstrates and tests the new emotional audio reactivity system
that focuses on the aesthetic and emotional qualities of music rather than
purely technical analysis.

Usage:
    python test_emotional_reactivity.py [--demo] [--audio AUDIO_FILE] [--images IMAGE_DIR]

Features tested:
- Emotional state analysis from audio
- Visual parameter mapping based on emotions
- Integration with existing transition system
- Real-time slideshow with emotional reactivity
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add the pyslidemorpher package to the path
sys.path.insert(0, str(Path(__file__).parent))

from pyslidemorpher.audio_emotion import EmotionalAudioReactivity, EmotionalState
from pyslidemorpher.emotional_realtime import play_emotional_slideshow


class SimpleArgs:
    """Simple argument container for testing"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_emotional_analysis(audio_file):
    """Test the emotional analysis system without visual output"""
    print("=" * 60)
    print("TESTING EMOTIONAL AUDIO ANALYSIS")
    print("=" * 60)
    
    if not Path(audio_file).exists():
        print(f"❌ Audio file not found: {audio_file}")
        return False
    
    print(f"🎵 Audio file: {audio_file}")
    print("🧠 Starting emotional analysis...")
    print("📊 This will show how the system interprets music emotionally")
    print("⏱️  Running for 20 seconds...")
    print()
    
    # Create emotional reactivity system
    reactivity = EmotionalAudioReactivity(audio_file)
    
    # Track emotional changes
    emotion_log = []
    
    def emotion_callback(visual_params, emotion):
        emotion_log.append((time.time(), emotion, visual_params))
        
        # Print current emotional state
        print(f"🎭 Emotion: Energy={emotion.energy:.2f} Valence={emotion.valence:.2f} "
              f"Texture={emotion.texture:.2f} Flow={emotion.flow:.2f}")
        print(f"🎨 Visual: {visual_params.get('transition_type', 'default')} "
              f"({visual_params.get('duration', 0):.1f}s) "
              f"px={visual_params.get('pixel_size', 4)} "
              f"ease={visual_params.get('easing', 'linear')}")
        print(f"🌈 Color: warmth={visual_params.get('color_shift', (0,0,0))[0]:.2f} "
              f"brightness={visual_params.get('color_shift', (0,0,0))[1]:.2f}")
        print("-" * 50)
    
    try:
        reactivity.start(emotion_callback)
        time.sleep(20)  # Run for 20 seconds
        
        print("\n📈 ANALYSIS SUMMARY:")
        if emotion_log:
            # Calculate averages
            avg_energy = sum(e[1].energy for e in emotion_log) / len(emotion_log)
            avg_valence = sum(e[1].valence for e in emotion_log) / len(emotion_log)
            avg_texture = sum(e[1].texture for e in emotion_log) / len(emotion_log)
            avg_flow = sum(e[1].flow for e in emotion_log) / len(emotion_log)
            
            print(f"📊 Average Emotional State:")
            print(f"   Energy: {avg_energy:.2f} ({'High' if avg_energy > 0.6 else 'Medium' if avg_energy > 0.3 else 'Low'})")
            print(f"   Valence: {avg_valence:.2f} ({'Happy' if avg_valence > 0.6 else 'Neutral' if avg_valence > 0.4 else 'Sad'})")
            print(f"   Texture: {avg_texture:.2f} ({'Rough' if avg_texture > 0.6 else 'Medium' if avg_texture > 0.3 else 'Smooth'})")
            print(f"   Flow: {avg_flow:.2f} ({'Dynamic' if avg_flow > 0.6 else 'Medium' if avg_flow > 0.3 else 'Static'})")
            
            # Most common transition types
            transitions = [e[2].get('transition_type', 'default') for e in emotion_log]
            transition_counts = {}
            for t in transitions:
                transition_counts[t] = transition_counts.get(t, 0) + 1
            
            print(f"🎬 Most Common Transitions:")
            for transition, count in sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"   {transition}: {count} times")
            
            print("✅ Emotional analysis test completed successfully!")
            return True
        else:
            print("❌ No emotional data collected")
            return False
            
    except Exception as e:
        print(f"❌ Error during emotional analysis: {e}")
        return False
    finally:
        reactivity.stop()


def test_slideshow_integration(audio_file, images_dir):
    """Test the full slideshow with emotional reactivity"""
    print("\n" + "=" * 60)
    print("TESTING EMOTIONAL SLIDESHOW INTEGRATION")
    print("=" * 60)
    
    # Find images
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_paths.extend(Path(images_dir).glob(ext))
    
    if not image_paths:
        print(f"❌ No images found in: {images_dir}")
        return False
    
    print(f"🖼️  Found {len(image_paths)} images in: {images_dir}")
    print(f"🎵 Audio file: {audio_file}")
    print()
    print("🎮 CONTROLS:")
    print("   Q or ESC: Quit")
    print("   SPACE: Force transition")
    print("   N: Next image")
    print("   P: Previous image")
    print()
    print("🎭 WHAT TO OBSERVE:")
    print("   - Transitions should feel natural and match the music's mood")
    print("   - High-energy music → faster, more dynamic transitions")
    print("   - Low-energy music → slower, gentler transitions")
    print("   - Color shifts based on emotional content")
    print("   - Image selection intelligence (random vs sequential)")
    print()
    input("Press ENTER to start the emotional slideshow...")
    
    # Create arguments for slideshow
    args = SimpleArgs(
        audio=audio_file,
        fps=30,
        height=720,
        fullscreen=False,
        web_gui=False,
        port=5001
    )
    
    try:
        play_emotional_slideshow(image_paths, args)
        print("✅ Slideshow test completed!")
        return True
    except Exception as e:
        print(f"❌ Error during slideshow: {e}")
        return False


def run_demo():
    """Run a complete demo with default assets"""
    print("🎬 EMOTIONAL AUDIO REACTIVITY DEMO")
    print("=" * 60)
    print("This demo showcases a completely new approach to audio reactivity")
    print("that focuses on emotional and aesthetic qualities rather than")
    print("purely technical analysis.")
    print()
    
    # Find demo assets
    audio_file = None
    for audio_path in [
        "assets/audio/Fragment 0007.mp3",
        "assets/audio/Fragment 0008.mp3",
        "assets/audio/Fragment 0009.mp3"
    ]:
        if Path(audio_path).exists():
            audio_file = audio_path
            break
    
    if not audio_file:
        print("❌ No demo audio files found in assets/audio/")
        return False
    
    images_dir = None
    for img_dir in ["assets/demo_images", "assets/images"]:
        if Path(img_dir).exists() and list(Path(img_dir).glob("*.jpg")):
            images_dir = img_dir
            break
    
    if not images_dir:
        print("❌ No demo images found in assets/")
        return False
    
    print(f"🎵 Using audio: {audio_file}")
    print(f"🖼️  Using images: {images_dir}")
    print()
    
    # Run tests
    success = True
    
    # Test 1: Emotional analysis
    if not test_emotional_analysis(audio_file):
        success = False
    
    # Test 2: Full slideshow
    if success:
        if not test_slideshow_integration(audio_file, images_dir):
            success = False
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("The emotional audio reactivity system is working correctly.")
        print()
        print("🆚 COMPARISON WITH TRADITIONAL SYSTEM:")
        print("   Traditional: Beat detection, FFT analysis, technical thresholds")
        print("   Emotional: Mood interpretation, aesthetic mapping, organic flow")
        print()
        print("🎨 KEY INNOVATIONS:")
        print("   • Emotion-based transition selection")
        print("   • Color psychology integration")
        print("   • Organic timing (breathing with music)")
        print("   • Intelligent image sequencing")
        print("   • Aesthetic-focused parameter mapping")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Check the error messages above for details.")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Test the new emotional audio reactivity system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_emotional_reactivity.py --demo
  python test_emotional_reactivity.py --audio assets/audio/Fragment0007.mp3 --images assets/demo_images
  python test_emotional_reactivity.py --analysis-only --audio my_music.mp3
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run complete demo with default assets')
    parser.add_argument('--audio', type=str,
                       help='Audio file to use for testing')
    parser.add_argument('--images', type=str,
                       help='Directory containing images for slideshow')
    parser.add_argument('--analysis-only', action='store_true',
                       help='Only test emotional analysis, no slideshow')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run demo if requested
    if args.demo:
        return run_demo()
    
    # Validate arguments
    if not args.audio:
        print("❌ Audio file is required (use --audio or --demo)")
        return False
    
    if not Path(args.audio).exists():
        print(f"❌ Audio file not found: {args.audio}")
        return False
    
    success = True
    
    # Test emotional analysis
    if not test_emotional_analysis(args.audio):
        success = False
    
    # Test slideshow if not analysis-only
    if not args.analysis_only and success:
        if not args.images:
            print("❌ Images directory is required for slideshow test (use --images)")
            return False
        
        if not Path(args.images).exists():
            print(f"❌ Images directory not found: {args.images}")
            return False
        
        if not test_slideshow_integration(args.audio, args.images):
            success = False
    
    return success


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
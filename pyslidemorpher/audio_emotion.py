"""
Audio Emotion-Based Reactivity System
====================================

A completely new approach to audio reactivity that focuses on the emotional
and aesthetic qualities of music rather than purely technical analysis.

This system aims to create more intuitive and artistically pleasing
visual responses by:

1. Emotion Detection: Analyzing music for emotional qualities (happy, sad, energetic, calm)
2. Musical Texture: Understanding the "texture" of sound (smooth, rough, sparkly, dense)
3. Narrative Flow: Treating the slideshow as a visual story that follows the music's journey
4. Color Psychology: Using color theory to match visuals to musical moods
5. Organic Timing: Using natural, breathing-like rhythms instead of rigid beat detection

Core Philosophy:
- Music is emotional, not just mathematical
- Visuals should "breathe" with the music
- Transitions should tell a story
- Less is often more - subtle changes can be more powerful than dramatic ones
"""

import numpy as np
import cv2
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import logging

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


@dataclass
class EmotionalState:
    """Represents the current emotional interpretation of the music"""
    energy: float  # 0.0 (calm) to 1.0 (energetic)
    valence: float  # 0.0 (sad) to 1.0 (happy)
    texture: float  # 0.0 (smooth) to 1.0 (rough/complex)
    density: float  # 0.0 (sparse) to 1.0 (dense/full)
    brightness: float  # 0.0 (dark/warm) to 1.0 (bright/cool)
    flow: float  # 0.0 (static) to 1.0 (flowing/dynamic)
    
    def __post_init__(self):
        """Ensure all values are clamped between 0 and 1"""
        for field in ['energy', 'valence', 'texture', 'density', 'brightness', 'flow']:
            value = getattr(self, field)
            setattr(self, field, max(0.0, min(1.0, value)))


class MusicEmotionAnalyzer:
    """
    Analyzes audio to extract emotional and aesthetic qualities
    
    Unlike traditional technical analysis, this focuses on perceptual qualities
    that humans naturally associate with emotions and visual aesthetics.
    """
    
    def __init__(self, sample_rate=22050, buffer_size=1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.history_length = 100  # Keep 5 seconds of history at 20Hz
        
        # Emotional state history
        self.emotion_history = deque(maxlen=self.history_length)
        self.current_emotion = EmotionalState(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        
        # Audio analysis buffers
        self.audio_buffer = deque(maxlen=buffer_size * 4)
        self.rms_history = deque(maxlen=50)
        self.spectral_history = deque(maxlen=50)
        
        # Smoothing factors for emotional stability
        self.emotion_smoothing = 0.85
        self.flow_smoothing = 0.7
        
    def analyze_audio_chunk(self, audio_data: np.ndarray) -> EmotionalState:
        """
        Analyze a chunk of audio and return emotional interpretation
        
        This uses a more intuitive approach than traditional audio analysis:
        - Energy comes from overall dynamics and rhythm
        - Valence from harmonic content and brightness
        - Texture from spectral complexity
        - Density from frequency distribution
        - Brightness from high-frequency content
        - Flow from temporal changes
        """
        if len(audio_data) == 0:
            return self.current_emotion
            
        # Basic audio metrics
        rms = np.sqrt(np.mean(audio_data ** 2))
        self.rms_history.append(rms)
        
        # FFT for frequency analysis
        fft = np.fft.rfft(audio_data)
        magnitude = np.abs(fft)
        
        # Frequency bands (more intuitive than technical)
        bass_range = magnitude[1:50]  # Deep, foundational sounds
        warmth_range = magnitude[50:200]  # Warm, rich sounds
        presence_range = magnitude[200:800]  # Human voice, main instruments
        brilliance_range = magnitude[800:2000]  # Sparkle, clarity
        air_range = magnitude[2000:]  # Airiness, space
        
        # Calculate emotional dimensions
        energy = self._calculate_energy(rms, bass_range, presence_range)
        valence = self._calculate_valence(warmth_range, brilliance_range, air_range)
        texture = self._calculate_texture(magnitude)
        density = self._calculate_density(magnitude)
        brightness = self._calculate_brightness(brilliance_range, air_range)
        flow = self._calculate_flow(rms)
        
        # Create new emotional state
        new_emotion = EmotionalState(energy, valence, texture, density, brightness, flow)
        
        # Smooth the emotional transition
        self.current_emotion = self._smooth_emotion_transition(self.current_emotion, new_emotion)
        self.emotion_history.append(self.current_emotion)
        
        return self.current_emotion
    
    def _calculate_energy(self, rms: float, bass: np.ndarray, presence: np.ndarray) -> float:
        """Calculate energy based on overall dynamics and low-frequency content"""
        # Energy comes from both overall loudness and bass presence
        rms_energy = min(rms * 10, 1.0)  # Scale RMS to 0-1
        bass_energy = min(np.mean(bass) * 0.001, 1.0)  # Bass adds energy
        presence_energy = min(np.mean(presence) * 0.0005, 1.0)  # Presence adds punch
        
        # Combine with emphasis on dynamics
        energy = (rms_energy * 0.5 + bass_energy * 0.3 + presence_energy * 0.2)
        
        # Add dynamic component - sudden changes increase energy
        if len(self.rms_history) > 1:
            rms_change = abs(rms - self.rms_history[-2])
            energy += min(rms_change * 20, 0.3)  # Sudden changes boost energy
            
        return min(energy, 1.0)
    
    def _calculate_valence(self, warmth: np.ndarray, brilliance: np.ndarray, air: np.ndarray) -> float:
        """Calculate emotional valence (happy vs sad) from harmonic content"""
        # Happy music tends to have more high-frequency content and brightness
        warmth_val = min(np.mean(warmth) * 0.0008, 0.4)  # Warm sounds are pleasant
        brilliance_val = min(np.mean(brilliance) * 0.001, 0.4)  # Bright sounds are happy
        air_val = min(np.mean(air) * 0.002, 0.3)  # Airy sounds are uplifting
        
        # Base valence starts neutral, gets modified by frequency content
        valence = 0.4 + warmth_val + brilliance_val + air_val
        
        return min(valence, 1.0)
    
    def _calculate_texture(self, magnitude: np.ndarray) -> float:
        """Calculate texture based on spectral complexity"""
        # Texture is about how "rough" or "smooth" the sound is
        # More complex spectra = rougher texture
        
        # Calculate spectral spread
        freqs = np.arange(len(magnitude))
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / (np.sum(magnitude) + 1e-10))
        
        # Normalize to 0-1 range
        texture = min(spectral_spread / 1000, 1.0)
        
        # Add noise component - more noise = rougher texture
        noise_level = np.std(magnitude) / (np.mean(magnitude) + 1e-10)
        texture += min(noise_level * 0.1, 0.3)
        
        return min(texture, 1.0)
    
    def _calculate_density(self, magnitude: np.ndarray) -> float:
        """Calculate density based on frequency distribution"""
        # Density is about how "full" the sound is across frequencies
        
        # Count significant frequency bins
        threshold = np.max(magnitude) * 0.1
        active_bins = np.sum(magnitude > threshold)
        density = active_bins / len(magnitude)
        
        # Add overall energy component
        total_energy = np.sum(magnitude)
        energy_density = min(total_energy * 0.00001, 0.5)
        
        return min(density + energy_density, 1.0)
    
    def _calculate_brightness(self, brilliance: np.ndarray, air: np.ndarray) -> float:
        """Calculate brightness from high-frequency content"""
        # Brightness is about high-frequency energy
        brilliance_bright = min(np.mean(brilliance) * 0.001, 0.6)
        air_bright = min(np.mean(air) * 0.002, 0.4)
        
        return min(brilliance_bright + air_bright, 1.0)
    
    def _calculate_flow(self, rms: float) -> float:
        """Calculate flow based on temporal changes"""
        if len(self.rms_history) < 3:
            return 0.5
            
        # Flow is about how much the music is changing over time
        recent_rms = list(self.rms_history)[-10:]  # Last 0.5 seconds
        rms_variance = np.var(recent_rms)
        
        # More variance = more flow
        flow = min(rms_variance * 1000, 1.0)
        
        # Smooth flow changes
        if hasattr(self, '_last_flow'):
            flow = self._last_flow * self.flow_smoothing + flow * (1 - self.flow_smoothing)
        self._last_flow = flow
        
        return flow
    
    def _smooth_emotion_transition(self, current: EmotionalState, new: EmotionalState) -> EmotionalState:
        """Smooth emotional transitions to avoid jarring changes"""
        alpha = 1 - self.emotion_smoothing
        
        return EmotionalState(
            energy=current.energy * self.emotion_smoothing + new.energy * alpha,
            valence=current.valence * self.emotion_smoothing + new.valence * alpha,
            texture=current.texture * self.emotion_smoothing + new.texture * alpha,
            density=current.density * self.emotion_smoothing + new.density * alpha,
            brightness=current.brightness * self.emotion_smoothing + new.brightness * alpha,
            flow=current.flow * self.flow_smoothing + new.flow * (1 - self.flow_smoothing)
        )
    
    def get_emotional_trend(self, window_size: int = 20) -> Optional[EmotionalState]:
        """Get the emotional trend over the last window_size samples"""
        if len(self.emotion_history) < window_size:
            return None
            
        recent_emotions = list(self.emotion_history)[-window_size:]
        
        # Calculate average emotional state
        avg_energy = np.mean([e.energy for e in recent_emotions])
        avg_valence = np.mean([e.valence for e in recent_emotions])
        avg_texture = np.mean([e.texture for e in recent_emotions])
        avg_density = np.mean([e.density for e in recent_emotions])
        avg_brightness = np.mean([e.brightness for e in recent_emotions])
        avg_flow = np.mean([e.flow for e in recent_emotions])
        
        return EmotionalState(avg_energy, avg_valence, avg_texture, avg_density, avg_brightness, avg_flow)


class EmotionalVisualMapper:
    """
    Maps emotional states to visual parameters
    
    This is where the magic happens - translating emotions into visual choices
    that feel natural and aesthetically pleasing.
    """
    
    def __init__(self):
        # Color palettes for different emotional states
        self.color_palettes = {
            'energetic_happy': [(255, 200, 100), (255, 150, 50), (255, 100, 150)],  # Warm, vibrant
            'energetic_sad': [(100, 150, 200), (150, 100, 200), (200, 100, 150)],   # Cool, intense
            'calm_happy': [(200, 255, 200), (150, 255, 150), (255, 255, 150)],      # Soft, warm
            'calm_sad': [(150, 150, 200), (100, 100, 150), (150, 100, 100)],       # Muted, cool
        }
        
        # Transition preferences for different emotional states
        self.transition_preferences = {
            'high_energy': ['tornado', 'swarm', 'swirl'],
            'medium_energy': ['drip', 'rainfall', 'default'],
            'low_energy': ['sorted', 'hue_sorted', 'default'],
            'smooth_texture': ['default', 'sorted', 'hue_sorted'],
            'rough_texture': ['tornado', 'swirl', 'drip'],
            'flowing': ['swarm', 'rainfall', 'swirl'],
            'static': ['sorted', 'hue_sorted', 'default']
        }
    
    def map_emotion_to_visual_params(self, emotion: EmotionalState) -> Dict[str, Any]:
        """
        Convert emotional state to visual parameters
        
        Returns a dictionary of visual parameters that can be used
        to modify transitions and effects.
        """
        params = {}
        
        # Transition timing based on energy and flow
        base_duration = 2.0  # Base transition duration
        energy_factor = 0.3 + (1.0 - emotion.energy) * 1.7  # High energy = faster
        flow_factor = 0.8 + emotion.flow * 0.4  # High flow = slightly faster
        params['duration'] = base_duration * energy_factor / flow_factor
        
        # Pixel size based on texture and density
        base_pixel_size = 4
        texture_factor = 0.5 + emotion.texture * 0.5  # Rough texture = smaller pixels
        density_factor = 0.7 + emotion.density * 0.3  # Dense = smaller pixels
        params['pixel_size'] = int(base_pixel_size * texture_factor * density_factor)
        
        # Transition type based on emotional qualities
        params['transition_type'] = self._select_transition_type(emotion)
        
        # Color modulation based on valence and brightness
        params['color_shift'] = self._calculate_color_shift(emotion)
        
        # Easing function based on texture and flow
        params['easing'] = self._select_easing_function(emotion)
        
        # Hold time based on energy (low energy = longer holds)
        params['hold_time'] = 0.1 + (1.0 - emotion.energy) * 0.8
        
        # Brightness modulation
        params['brightness_mod'] = 0.9 + emotion.brightness * 0.2
        
        return params
    
    def _select_transition_type(self, emotion: EmotionalState) -> str:
        """Select transition type based on emotional state"""
        # High energy prefers dynamic transitions
        if emotion.energy > 0.7:
            if emotion.texture > 0.6:
                return np.random.choice(self.transition_preferences['rough_texture'])
            else:
                return np.random.choice(self.transition_preferences['high_energy'])
        
        # Medium energy
        elif emotion.energy > 0.3:
            if emotion.flow > 0.6:
                return np.random.choice(self.transition_preferences['flowing'])
            else:
                return np.random.choice(self.transition_preferences['medium_energy'])
        
        # Low energy prefers gentle transitions
        else:
            if emotion.texture < 0.4:
                return np.random.choice(self.transition_preferences['smooth_texture'])
            else:
                return np.random.choice(self.transition_preferences['low_energy'])
    
    def _calculate_color_shift(self, emotion: EmotionalState) -> Tuple[float, float, float]:
        """Calculate color shift based on emotional state"""
        # Map valence and brightness to color temperature
        # High valence + high brightness = warm, bright colors
        # Low valence + low brightness = cool, dark colors
        
        warmth = emotion.valence * 0.3 - 0.15  # -0.15 to +0.15
        brightness_shift = emotion.brightness * 0.2 - 0.1  # -0.1 to +0.1
        saturation = emotion.energy * 0.3  # 0 to 0.3
        
        return (warmth, brightness_shift, saturation)
    
    def _select_easing_function(self, emotion: EmotionalState) -> str:
        """Select easing function based on emotional texture and flow"""
        if emotion.texture > 0.7:
            return 'cubic'  # Sharp, dramatic
        elif emotion.flow > 0.7:
            return 'sine'   # Smooth, flowing
        elif emotion.energy > 0.7:
            return 'quad'   # Punchy
        else:
            return 'linear' # Gentle, steady


class EmotionalAudioReactivity:
    """
    Main class that orchestrates the emotion-based audio reactivity system
    
    This replaces the traditional technical approach with an intuitive,
    emotion-driven system that creates more natural and aesthetically
    pleasing visual responses to music.
    """
    
    def __init__(self, audio_file: str):
        self.audio_file = audio_file
        self.analyzer = MusicEmotionAnalyzer()
        self.mapper = EmotionalVisualMapper()
        
        # Threading components
        self.audio_thread = None
        self.running = False
        self.current_params = {}
        
        # Pygame audio setup
        if PYGAME_AVAILABLE:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
        
        # Callback for visual parameter updates
        self.param_update_callback = None
        
        logging.info("Emotional Audio Reactivity system initialized")
    
    def start(self, param_update_callback=None):
        """Start the emotional audio analysis"""
        self.param_update_callback = param_update_callback
        self.running = True
        
        if PYGAME_AVAILABLE:
            # Start audio playback
            pygame.mixer.music.load(self.audio_file)
            pygame.mixer.music.play()
            
            # Start analysis thread
            self.audio_thread = threading.Thread(target=self._audio_analysis_loop, daemon=True)
            self.audio_thread.start()
            
            logging.info("Emotional audio reactivity started")
        else:
            logging.warning("Pygame not available - emotional reactivity disabled")
    
    def stop(self):
        """Stop the emotional audio analysis"""
        self.running = False
        
        if PYGAME_AVAILABLE and pygame.mixer.get_init():
            pygame.mixer.music.stop()
        
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        
        logging.info("Emotional audio reactivity stopped")
    
    def _audio_analysis_loop(self):
        """Main audio analysis loop running in separate thread"""
        # This is a simplified version - in a real implementation,
        # we would need to capture audio data from the playing stream
        # For now, we'll simulate emotional analysis
        
        while self.running:
            try:
                # Simulate audio analysis (in real implementation, capture audio data)
                # For demonstration, create varying emotional states
                t = time.time()
                simulated_emotion = EmotionalState(
                    energy=0.3 + 0.4 * np.sin(t * 0.5) + 0.3 * np.random.random(),
                    valence=0.4 + 0.3 * np.cos(t * 0.3) + 0.3 * np.random.random(),
                    texture=0.2 + 0.6 * np.sin(t * 0.7) + 0.2 * np.random.random(),
                    density=0.3 + 0.4 * np.cos(t * 0.4) + 0.3 * np.random.random(),
                    brightness=0.4 + 0.4 * np.sin(t * 0.6) + 0.2 * np.random.random(),
                    flow=0.2 + 0.6 * np.cos(t * 0.8) + 0.2 * np.random.random()
                )
                
                # Map emotion to visual parameters
                visual_params = self.mapper.map_emotion_to_visual_params(simulated_emotion)
                self.current_params = visual_params
                
                # Notify callback if provided
                if self.param_update_callback:
                    self.param_update_callback(visual_params, simulated_emotion)
                
                # Log emotional state periodically
                if int(t) % 5 == 0:  # Every 5 seconds
                    logging.info(f"Emotional State - Energy: {simulated_emotion.energy:.2f}, "
                               f"Valence: {simulated_emotion.valence:.2f}, "
                               f"Texture: {simulated_emotion.texture:.2f}")
                
                time.sleep(0.05)  # 20Hz update rate
                
            except Exception as e:
                logging.error(f"Error in emotional audio analysis: {e}")
                time.sleep(0.1)
    
    def get_current_params(self) -> Dict[str, Any]:
        """Get the current visual parameters based on emotional analysis"""
        return self.current_params.copy()
    
    def get_current_emotion(self) -> EmotionalState:
        """Get the current emotional state"""
        return self.analyzer.current_emotion
    
    def should_trigger_transition(self) -> bool:
        """
        Determine if a transition should be triggered based on emotional flow
        
        Unlike beat-based triggering, this uses emotional "breathing" -
        natural moments where the music calls for a visual change.
        """
        emotion = self.analyzer.current_emotion
        
        # High flow or significant energy changes suggest transition moments
        flow_trigger = emotion.flow > 0.7
        energy_trigger = emotion.energy > 0.8
        
        # Get emotional trend to detect significant changes
        trend = self.analyzer.get_emotional_trend()
        if trend:
            energy_change = abs(emotion.energy - trend.energy) > 0.3
            valence_change = abs(emotion.valence - trend.valence) > 0.4
            change_trigger = energy_change or valence_change
        else:
            change_trigger = False
        
        return flow_trigger or energy_trigger or change_trigger


# Example usage and integration functions
def create_emotional_slideshow(images_path: str, audio_file: str, **kwargs):
    """
    Create an emotionally reactive slideshow
    
    This is the main entry point for the new emotional reactivity system.
    It can be integrated into the existing pyslidemorpher framework.
    """
    reactivity = EmotionalAudioReactivity(audio_file)
    
    def on_param_update(visual_params, emotion):
        # This callback would be used to update the slideshow parameters
        # in real-time based on the emotional analysis
        logging.debug(f"Visual params updated: {visual_params}")
    
    reactivity.start(on_param_update)
    
    # The slideshow would run here, using the emotional parameters
    # to guide transitions and visual effects
    
    return reactivity


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # This would be integrated with the main slideshow system
    audio_file = "assets/audio/Fragment 0007.mp3"
    reactivity = EmotionalAudioReactivity(audio_file)
    
    def demo_callback(params, emotion):
        print(f"Emotion: E={emotion.energy:.2f} V={emotion.valence:.2f} T={emotion.texture:.2f}")
        print(f"Visual: duration={params.get('duration', 0):.1f}s, "
              f"transition={params.get('transition_type', 'default')}")
        print("-" * 50)
    
    try:
        reactivity.start(demo_callback)
        time.sleep(30)  # Run for 30 seconds
    finally:
        reactivity.stop()
"""
Emotional Realtime Slideshow
============================

A new realtime slideshow implementation that uses the emotional audio reactivity
system instead of the traditional technical analysis approach.

This module integrates the EmotionalAudioReactivity system with the existing
transition and rendering infrastructure to create a more intuitive and
aesthetically pleasing audio-reactive slideshow experience.
"""

import json
import logging
import random
import threading
import time
from pathlib import Path
from queue import Queue
from collections import deque
import cv2
import numpy as np

from .audio_emotion import EmotionalAudioReactivity, EmotionalState
from .transitions import (
    make_transition_frames,
    make_swarm_transition_frames,
    make_tornado_transition_frames,
    make_swirl_transition_frames,
    make_drip_transition_frames,
    make_rainfall_transition_frames,
    make_sorted_transition_frames,
    make_hue_sorted_transition_frames
)

try:
    from .web_gui import get_controller, start_web_server, FLASK_AVAILABLE
except ImportError:
    FLASK_AVAILABLE = False
    logging.warning("Flask not available - web GUI will be disabled")


class EmotionalSlideshow:
    """
    Main class for running an emotionally reactive slideshow
    
    This replaces the complex technical analysis in the original realtime.py
    with a more intuitive emotional approach that focuses on the aesthetic
    and emotional qualities of music.
    """
    
    def __init__(self, images, audio_file, args):
        self.images = images
        self.audio_file = audio_file
        self.args = args
        
        # Initialize emotional reactivity system
        self.emotional_system = EmotionalAudioReactivity(audio_file)
        
        # Slideshow state
        self.current_image_idx = 0
        self.frame_queue = Queue(maxsize=30)
        self.running = False
        
        # Transition state
        self.transition_in_progress = False
        self.last_transition_time = 0
        self.min_transition_interval = 0.5  # Minimum time between transitions
        
        # Visual parameters (updated by emotional system)
        self.current_visual_params = {
            'duration': 2.0,
            'pixel_size': 4,
            'transition_type': 'default',
            'easing': 'linear',
            'hold_time': 0.3,
            'brightness_mod': 1.0
        }
        
        # Threading components
        self.frame_thread = None
        self.display_thread = None
        
        # Web GUI integration
        self.web_controller = None
        if FLASK_AVAILABLE and args.web_gui:
            self.web_controller = get_controller()
            
        logging.info("Emotional slideshow initialized")
    
    def start(self):
        """Start the emotional slideshow"""
        self.running = True
        
        # Start emotional audio analysis
        self.emotional_system.start(self._on_emotional_update)
        
        # Start frame generation thread
        self.frame_thread = threading.Thread(target=self._frame_generation_loop, daemon=True)
        self.frame_thread.start()
        
        # Start web server if enabled
        if self.web_controller:
            web_thread = threading.Thread(
                target=start_web_server, 
                args=(self.args.port,), 
                daemon=True
            )
            web_thread.start()
            logging.info(f"Web GUI available at http://localhost:{self.args.port}")
        
        # Main display loop
        self._display_loop()
    
    def stop(self):
        """Stop the emotional slideshow"""
        self.running = False
        self.emotional_system.stop()
        
        if self.frame_thread and self.frame_thread.is_alive():
            self.frame_thread.join(timeout=1.0)
            
        logging.info("Emotional slideshow stopped")
    
    def _on_emotional_update(self, visual_params, emotion):
        """Callback for when emotional analysis updates visual parameters"""
        self.current_visual_params.update(visual_params)
        
        # Log emotional changes periodically
        if hasattr(self, '_last_log_time'):
            if time.time() - self._last_log_time > 3.0:  # Every 3 seconds
                self._log_emotional_state(emotion, visual_params)
                self._last_log_time = time.time()
        else:
            self._last_log_time = time.time()
    
    def _log_emotional_state(self, emotion, params):
        """Log current emotional state and visual parameters"""
        logging.info(
            f"Emotion: Energy={emotion.energy:.2f} Valence={emotion.valence:.2f} "
            f"Texture={emotion.texture:.2f} Flow={emotion.flow:.2f} | "
            f"Visual: {params.get('transition_type', 'default')} "
            f"({params.get('duration', 0):.1f}s, px={params.get('pixel_size', 4)})"
        )
    
    def _should_trigger_transition(self):
        """Determine if a transition should be triggered"""
        current_time = time.time()
        
        # Respect minimum interval
        if current_time - self.last_transition_time < self.min_transition_interval:
            return False
        
        # Don't trigger if already in transition
        if self.transition_in_progress:
            return False
        
        # Use emotional system to determine transition timing
        return self.emotional_system.should_trigger_transition()
    
    def _get_next_image_index(self):
        """Get the next image index, with some intelligence based on emotion"""
        emotion = self.emotional_system.get_current_emotion()
        
        # For high-energy music, prefer more random jumps
        # For low-energy music, prefer sequential progression
        if emotion.energy > 0.7:
            # High energy - more random selection
            available_indices = [i for i in range(len(self.images)) if i != self.current_image_idx]
            return random.choice(available_indices)
        elif emotion.energy < 0.3:
            # Low energy - sequential with occasional skips
            if random.random() < 0.8:  # 80% chance of sequential
                return (self.current_image_idx + 1) % len(self.images)
            else:
                return (self.current_image_idx + random.randint(2, 5)) % len(self.images)
        else:
            # Medium energy - balanced approach
            if random.random() < 0.6:  # 60% chance of sequential
                return (self.current_image_idx + 1) % len(self.images)
            else:
                available_indices = [i for i in range(len(self.images)) if i != self.current_image_idx]
                return random.choice(available_indices)
    
    def _get_transition_function(self, transition_name):
        """Get the appropriate transition function"""
        transition_map = {
            'default': make_transition_frames,
            'swarm': make_swarm_transition_frames,
            'tornado': make_tornado_transition_frames,
            'swirl': make_swirl_transition_frames,
            'drip': make_drip_transition_frames,
            'rainfall': make_rainfall_transition_frames,
            'sorted': make_sorted_transition_frames,
            'hue_sorted': make_hue_sorted_transition_frames,
        }
        return transition_map.get(transition_name, make_transition_frames)
    
    def _apply_emotional_color_effects(self, frame, emotion):
        """Apply color effects based on emotional state"""
        if not hasattr(self.current_visual_params, 'color_shift'):
            return frame
        
        warmth, brightness_shift, saturation = self.current_visual_params.get('color_shift', (0, 0, 0))
        
        # Convert to HSV for easier color manipulation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Apply warmth (hue shift)
        if abs(warmth) > 0.01:
            hsv[:, :, 0] = (hsv[:, :, 0] + warmth * 30) % 180
        
        # Apply brightness
        if abs(brightness_shift) > 0.01:
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + brightness_shift), 0, 255)
        
        # Apply saturation
        if abs(saturation) > 0.01:
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + saturation), 0, 255)
        
        # Convert back to BGR
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _frame_generation_loop(self):
        """Main frame generation loop running in separate thread"""
        current_frame = None
        
        # Load initial image
        if self.images:
            img_path = self.images[self.current_image_idx]
            current_frame = cv2.imread(str(img_path))
            if current_frame is not None:
                # Resize to target resolution
                target_height = self.args.height or 720
                aspect_ratio = current_frame.shape[1] / current_frame.shape[0]
                target_width = int(target_height * aspect_ratio)
                current_frame = cv2.resize(current_frame, (target_width, target_height))
        
        while self.running:
            try:
                # Check if we should trigger a transition
                if self._should_trigger_transition() and len(self.images) > 1:
                    self._generate_transition(current_frame)
                    current_frame = self._load_current_image()
                
                # Apply emotional color effects to current frame
                if current_frame is not None:
                    emotion = self.emotional_system.get_current_emotion()
                    enhanced_frame = self._apply_emotional_color_effects(current_frame, emotion)
                    
                    # Apply brightness modulation
                    brightness_mod = self.current_visual_params.get('brightness_mod', 1.0)
                    if abs(brightness_mod - 1.0) > 0.01:
                        enhanced_frame = cv2.convertScaleAbs(enhanced_frame, alpha=brightness_mod)
                    
                    # Add frame to queue
                    if not self.frame_queue.full():
                        self.frame_queue.put(enhanced_frame.copy())
                
                time.sleep(1.0 / self.args.fps)  # Control frame rate
                
            except Exception as e:
                logging.error(f"Error in frame generation: {e}")
                time.sleep(0.1)
    
    def _generate_transition(self, current_frame):
        """Generate transition frames between images"""
        if current_frame is None:
            return
        
        self.transition_in_progress = True
        self.last_transition_time = time.time()
        
        try:
            # Get next image
            next_idx = self._get_next_image_index()
            next_img_path = self.images[next_idx]
            next_frame = cv2.imread(str(next_img_path))
            
            if next_frame is None:
                self.transition_in_progress = False
                return
            
            # Resize next frame to match current frame
            next_frame = cv2.resize(next_frame, (current_frame.shape[1], current_frame.shape[0]))
            
            # Get transition parameters from emotional system
            params = self.current_visual_params
            transition_type = params.get('transition_type', 'default')
            duration = params.get('duration', 2.0)
            pixel_size = params.get('pixel_size', 4)
            easing = params.get('easing', 'linear')
            hold_time = params.get('hold_time', 0.3)
            
            # Get transition function
            transition_func = self._get_transition_function(transition_type)
            
            # Generate transition frames
            transition_frames = transition_func(
                current_frame, next_frame,
                pixel_size=pixel_size,
                fps=self.args.fps,
                seconds=duration,
                hold=hold_time,
                ease_name=easing,
                seed=random.randint(0, 1000000)
            )
            
            # Add transition frames to queue
            for frame in transition_frames:
                if not self.running:
                    break
                
                # Apply emotional color effects
                emotion = self.emotional_system.get_current_emotion()
                enhanced_frame = self._apply_emotional_color_effects(frame, emotion)
                
                if not self.frame_queue.full():
                    self.frame_queue.put(enhanced_frame)
                else:
                    # If queue is full, wait a bit
                    time.sleep(0.01)
            
            # Update current image index
            self.current_image_idx = next_idx
            
            logging.debug(f"Transition completed: {transition_type} to image {next_idx}")
            
        except Exception as e:
            logging.error(f"Error generating transition: {e}")
        finally:
            self.transition_in_progress = False
    
    def _load_current_image(self):
        """Load and resize the current image"""
        if not self.images:
            return None
        
        img_path = self.images[self.current_image_idx]
        frame = cv2.imread(str(img_path))
        
        if frame is not None:
            # Resize to target resolution
            target_height = self.args.height or 720
            aspect_ratio = frame.shape[1] / frame.shape[0]
            target_width = int(target_height * aspect_ratio)
            frame = cv2.resize(frame, (target_width, target_height))
        
        return frame
    
    def _display_loop(self):
        """Main display loop"""
        cv2.namedWindow('Emotional Slideshow', cv2.WINDOW_NORMAL)
        
        if self.args.fullscreen:
            cv2.setWindowProperty('Emotional Slideshow', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.running:
            try:
                # Get frame from queue
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    
                    # Display frame
                    cv2.imshow('Emotional Slideshow', frame)
                    
                    # FPS counter
                    fps_counter += 1
                    if fps_counter % 60 == 0:  # Every 60 frames
                        elapsed = time.time() - fps_start_time
                        actual_fps = 60 / elapsed
                        logging.debug(f"Display FPS: {actual_fps:.1f}")
                        fps_start_time = time.time()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord(' '):  # Space - force transition
                    if not self.transition_in_progress:
                        self.last_transition_time = 0  # Reset to allow immediate transition
                elif key == ord('n'):  # 'n' - next image
                    self.current_image_idx = (self.current_image_idx + 1) % len(self.images)
                elif key == ord('p'):  # 'p' - previous image
                    self.current_image_idx = (self.current_image_idx - 1) % len(self.images)
                
                time.sleep(0.001)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                logging.error(f"Error in display loop: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()


def play_emotional_slideshow(images, args):
    """
    Main entry point for the emotional slideshow
    
    This replaces the traditional play_realtime function with an emotion-based approach.
    """
    if not args.audio:
        logging.error("Audio file is required for emotional slideshow")
        return
    
    if not Path(args.audio).exists():
        logging.error(f"Audio file not found: {args.audio}")
        return
    
    if not images:
        logging.error("No images provided for slideshow")
        return
    
    logging.info(f"Starting emotional slideshow with {len(images)} images")
    logging.info(f"Audio: {args.audio}")
    logging.info("Press 'q' or ESC to quit, SPACE to force transition, 'n'/'p' for next/previous")
    
    # Create and start slideshow
    slideshow = EmotionalSlideshow(images, args.audio, args)
    
    try:
        slideshow.start()
    except KeyboardInterrupt:
        logging.info("Slideshow interrupted by user")
    except Exception as e:
        logging.error(f"Error running slideshow: {e}")
    finally:
        slideshow.stop()


if __name__ == "__main__":
    # Example usage for testing
    import argparse
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)
    
    # Simple test setup
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', default='assets/audio/Fragment 0007.mp3')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fullscreen', action='store_true')
    parser.add_argument('--web-gui', action='store_true')
    parser.add_argument('--port', type=int, default=5001)
    
    args = parser.parse_args()
    
    # Find demo images
    demo_images = list(Path('assets/demo_images').glob('*.jpg'))
    if not demo_images:
        demo_images = list(Path('assets/images').glob('*.jpg'))
    
    if demo_images and Path(args.audio).exists():
        play_emotional_slideshow(demo_images, args)
    else:
        print("Demo images or audio file not found")
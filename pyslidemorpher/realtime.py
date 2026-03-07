
"""
Real-time playback functionality for PySlide Morpher.
Handles live slideshow display using OpenCV.
"""

import json
import logging
import random
import threading
import time
from os import environ
from pathlib import Path
from queue import Queue

import cv2
import numpy as np

# Try to import scipy for advanced FFT operations, fallback to numpy
try:
    from scipy import signal
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Use numpy's FFT as fallback
    from numpy.fft import fft, fftfreq
    logging.warning("scipy not available - using numpy FFT for audio visualizations")

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

try:
    import pygame

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("pygame not available - audio playback will be disabled in realtime mode")

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
    WEB_GUI_AVAILABLE = FLASK_AVAILABLE
except ImportError:
    WEB_GUI_AVAILABLE = False
    logging.warning("Web GUI not available - Flask or web_gui module not found")


def get_random_transition_function():
    """Randomly select a transition function from available options.

    Ensures that the same transition function is never selected twice in a row.
    """
    transition_functions = [
        make_transition_frames,  # default
        make_swarm_transition_frames,  # swarm
        make_tornado_transition_frames,  # tornado
        make_swirl_transition_frames,  # swirl
        make_drip_transition_frames,  # drip
        # make_rainfall_transition_frames, # rain
        make_sorted_transition_frames,  # sorted
        make_hue_sorted_transition_frames,  # hue-sorted
    ]

    # Initialize the last selected function attribute if it doesn't exist
    if not hasattr(get_random_transition_function, '_last_selected'):
        get_random_transition_function._last_selected = None

    # If this is the first call or there's only one function, just return a random choice
    if get_random_transition_function._last_selected is None or len(transition_functions) <= 1:
        selected = random.choice(transition_functions)
        get_random_transition_function._last_selected = selected
        logging.info(f"Randomly selected transition function: {selected.__name__}")
        return selected

    # Create a list of available functions excluding the last selected one
    available_functions = [func for func in transition_functions
                           if func != get_random_transition_function._last_selected]

    # Select from the available functions
    selected = random.choice(available_functions)
    get_random_transition_function._last_selected = selected
    logging.info(f"Randomly selected transition function: {selected.__name__}")
    return selected


def render_audio_visualizations(frame, audio_data, audio_features, settings, sample_rate=22050):
    """Render audio visualizations on the frame based on current settings."""
    if audio_data is None or len(audio_data) == 0:
        return frame

    height, width = frame.shape[:2]
    viz_size = settings.get('viz_size', 0.3)
    viz_opacity = settings.get('viz_opacity', 0.7)

    # Calculate visualization dimensions
    viz_width = int(width * viz_size)
    viz_height = int(height * viz_size)

    # Position calculations for different visualizations
    positions = {
        'top_left': (20, 20),
        'top_right': (width - viz_width - 20, 20),
        'bottom_left': (20, height - viz_height - 20),
        'bottom_right': (width - viz_width - 20, height - viz_height - 20),
        'center': (width // 2 - viz_width // 2, height // 2 - viz_height // 2)
    }

    viz_count = 0
    position_keys = list(positions.keys())

    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1:
        mono_data = np.mean(audio_data, axis=1)
    else:
        mono_data = audio_data

    # Get recent audio data for visualizations (last 2048 samples)
    recent_samples = min(2048, len(mono_data))
    recent_data = mono_data[-recent_samples:].astype(np.float32)

    # Normalize audio data
    if np.max(np.abs(recent_data)) > 0:
        recent_data = recent_data / np.max(np.abs(recent_data))

    # Oscilloscope visualization
    if settings.get('viz_oscilloscope', False) and viz_count < len(position_keys):
        pos = positions[position_keys[viz_count]]
        render_oscilloscope(frame, recent_data, pos, viz_width, viz_height, viz_opacity)
        viz_count += 1

    # Waveform visualization
    if settings.get('viz_waveform', False) and viz_count < len(position_keys):
        pos = positions[position_keys[viz_count]]
        render_waveform(frame, recent_data, pos, viz_width, viz_height, viz_opacity)
        viz_count += 1

    # Spectrum visualization
    if settings.get('viz_spectrum', False) and viz_count < len(position_keys):
        pos = positions[position_keys[viz_count]]
        render_spectrum(frame, recent_data, pos, viz_width, viz_height, viz_opacity, sample_rate)
        viz_count += 1

    # EQ visualization
    if settings.get('viz_eq', False) and viz_count < len(position_keys):
        pos = positions[position_keys[viz_count]]
        render_eq(frame, recent_data, pos, viz_width, viz_height, viz_opacity, sample_rate)
        viz_count += 1

    # Lissajous visualization (requires stereo data)
    if settings.get('viz_lissajous', False) and viz_count < len(position_keys):
        pos = positions[position_keys[viz_count]]
        if len(audio_data.shape) > 1:
            render_lissajous(frame, audio_data[-recent_samples:], pos, viz_width, viz_height, viz_opacity)
        else:
            # Create pseudo-stereo for mono data
            delayed_data = np.roll(recent_data, recent_samples // 4)
            stereo_data = np.column_stack([recent_data, delayed_data])
            render_lissajous(frame, stereo_data, pos, viz_width, viz_height, viz_opacity)
        viz_count += 1

    return frame


def render_oscilloscope(frame, audio_data, pos, width, height, opacity):
    """Render oscilloscope visualization."""
    x, y = pos

    # Create overlay
    overlay = frame.copy()

    # Draw background
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (100, 100, 100), 2)

    # Draw center line
    center_y = y + height // 2
    cv2.line(overlay, (x, center_y), (x + width, center_y), (50, 50, 50), 1)

    # Draw waveform
    if len(audio_data) > 1:
        points = []
        for i in range(min(width, len(audio_data))):
            sample_idx = int(i * len(audio_data) / width)
            sample_y = int(center_y + audio_data[sample_idx] * height * 0.4)
            sample_y = max(y, min(y + height, sample_y))
            points.append((x + i, sample_y))

        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(overlay, points[i], points[i + 1], (0, 255, 0), 2)

    # Blend with original frame
    cv2.addWeighted(frame, 1 - opacity, overlay, opacity, 0, frame)


def render_waveform(frame, audio_data, pos, width, height, opacity):
    """Render waveform visualization."""
    x, y = pos

    # Create overlay
    overlay = frame.copy()

    # Draw background
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (100, 100, 100), 2)

    # Draw waveform as bars
    if len(audio_data) > 0:
        bar_width = max(1, width // 64)  # 64 bars
        for i in range(0, width, bar_width):
            sample_idx = int(i * len(audio_data) / width)
            if sample_idx < len(audio_data):
                amplitude = abs(audio_data[sample_idx])
                bar_height = int(amplitude * height * 0.8)
                bar_y = y + height - bar_height

                # Color based on amplitude
                color = (0, int(255 * amplitude), int(255 * (1 - amplitude)))
                cv2.rectangle(overlay, (x + i, bar_y), (x + i + bar_width - 1, y + height), color, -1)

    # Blend with original frame
    cv2.addWeighted(frame, 1 - opacity, overlay, opacity, 0, frame)


def render_spectrum(frame, audio_data, pos, width, height, opacity, sample_rate):
    """Render spectrum analyzer visualization."""
    x, y = pos

    # Create overlay
    overlay = frame.copy()

    # Draw background
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (100, 100, 100), 2)

    if len(audio_data) > 0:
        # Apply window function to reduce spectral leakage
        if SCIPY_AVAILABLE:
            windowed_data = audio_data * np.hanning(len(audio_data))
        else:
            # Use numpy's hanning window as fallback
            windowed_data = audio_data * np.hanning(len(audio_data))

        # Compute FFT
        fft_data = np.abs(fft(windowed_data))
        freqs = fftfreq(len(windowed_data), 1/sample_rate)

        # Take only positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft_data[:len(fft_data)//2]

        # Focus on audible range (20Hz to 20kHz)
        audible_mask = (positive_freqs >= 20) & (positive_freqs <= 20000)
        audible_freqs = positive_freqs[audible_mask]
        audible_fft = positive_fft[audible_mask]

        if len(audible_fft) > 0:
            # Logarithmic frequency scaling
            log_freqs = np.log10(audible_freqs + 1)

            # Draw spectrum bars
            bar_width = max(1, width // 64)
            for i in range(0, width, bar_width):
                freq_idx = int(i * len(audible_fft) / width)
                if freq_idx < len(audible_fft):
                    magnitude = audible_fft[freq_idx]
                    # Logarithmic magnitude scaling
                    log_magnitude = np.log10(magnitude + 1) / 6  # Normalize
                    bar_height = int(log_magnitude * height * 0.8)
                    bar_y = y + height - bar_height

                    # Color based on frequency (blue for low, red for high)
                    freq_ratio = i / width
                    color = (int(255 * freq_ratio), int(255 * (1 - freq_ratio)), 255)
                    cv2.rectangle(overlay, (x + i, bar_y), (x + i + bar_width - 1, y + height), color, -1)

    # Blend with original frame
    cv2.addWeighted(frame, 1 - opacity, overlay, opacity, 0, frame)


def render_eq(frame, audio_data, pos, width, height, opacity, sample_rate):
    """Render equalizer-style visualization with frequency bands."""
    x, y = pos

    # Create overlay
    overlay = frame.copy()

    # Draw background
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (100, 100, 100), 2)

    if len(audio_data) > 0:
        # Define frequency bands (similar to graphic equalizer)
        bands = [60, 170, 310, 600, 1000, 3000, 6000, 12000, 14000, 16000]  # Hz

        # Compute FFT
        windowed_data = audio_data * np.hanning(len(audio_data))
        fft_data = np.abs(fft(windowed_data))
        freqs = fftfreq(len(windowed_data), 1/sample_rate)

        # Take only positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft_data[:len(fft_data)//2]

        # Calculate energy in each band
        band_energies = []
        for i in range(len(bands) - 1):
            low_freq = bands[i]
            high_freq = bands[i + 1]

            # Find frequency indices
            low_idx = np.argmin(np.abs(positive_freqs - low_freq))
            high_idx = np.argmin(np.abs(positive_freqs - high_freq))

            # Calculate average energy in this band
            if high_idx > low_idx:
                band_energy = np.mean(positive_fft[low_idx:high_idx])
                band_energies.append(band_energy)
            else:
                band_energies.append(0)

        # Draw EQ bars
        if band_energies:
            bar_width = width // len(band_energies)
            max_energy = max(band_energies) if max(band_energies) > 0 else 1

            for i, energy in enumerate(band_energies):
                normalized_energy = energy / max_energy
                bar_height = int(normalized_energy * height * 0.8)
                bar_x = x + i * bar_width
                bar_y = y + height - bar_height

                # Color gradient from green (low) to red (high)
                color = (0, int(255 * (1 - normalized_energy)), int(255 * normalized_energy))
                cv2.rectangle(overlay, (bar_x + 2, bar_y), (bar_x + bar_width - 2, y + height), color, -1)

                # Draw frequency label
                freq_label = f"{bands[i]}"
                cv2.putText(overlay, freq_label, (bar_x + 2, y + height - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Blend with original frame
    cv2.addWeighted(frame, 1 - opacity, overlay, opacity, 0, frame)


def render_lissajous(frame, stereo_data, pos, width, height, opacity):
    """Render Lissajous curve visualization."""
    x, y = pos

    # Create overlay
    overlay = frame.copy()

    # Draw background
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (100, 100, 100), 2)

    # Draw center lines
    center_x = x + width // 2
    center_y = y + height // 2
    cv2.line(overlay, (center_x, y), (center_x, y + height), (50, 50, 50), 1)
    cv2.line(overlay, (x, center_y), (x + width, center_y), (50, 50, 50), 1)

    if len(stereo_data) > 1 and stereo_data.shape[1] >= 2:
        # Normalize stereo channels
        left = stereo_data[:, 0].astype(np.float32)
        right = stereo_data[:, 1].astype(np.float32)

        if np.max(np.abs(left)) > 0:
            left = left / np.max(np.abs(left))
        if np.max(np.abs(right)) > 0:
            right = right / np.max(np.abs(right))

        # Create Lissajous points
        points = []
        step = max(1, len(left) // 500)  # Limit number of points for performance

        for i in range(0, len(left), step):
            point_x = int(center_x + left[i] * width * 0.4)
            point_y = int(center_y + right[i] * height * 0.4)
            point_x = max(x, min(x + width, point_x))
            point_y = max(y, min(y + height, point_y))
            points.append((point_x, point_y))

        # Draw Lissajous curve
        if len(points) > 1:
            for i in range(len(points) - 1):
                # Color fade effect
                alpha = i / len(points)
                color = (int(255 * alpha), int(255 * (1 - alpha)), 255)
                cv2.line(overlay, points[i], points[i + 1], color, 2)

    # Blend with original frame
    cv2.addWeighted(frame, 1 - opacity, overlay, opacity, 0, frame)




def play_realtime(imgs, args):
    """Play slideshow in realtime using OpenCV display with optimized performance."""
    W, H = args.size
    frame_time = 1.0 / args.fps  # Time per frame in seconds

    # Initialize web GUI controller if requested and available
    web_controller = None
    web_server_thread = None
    if hasattr(args, 'web_gui') and args.web_gui and WEB_GUI_AVAILABLE:
        try:
            web_controller = get_controller()
            # Initialize controller with current args
            web_controller.update_setting('fps', args.fps)
            web_controller.update_setting('seconds_per_transition', args.seconds_per_transition)
            web_controller.update_setting('hold', args.hold)
            web_controller.update_setting('pixel_size', args.pixel_size)
            web_controller.update_setting('transition', args.transition)
            web_controller.update_setting('easing', args.easing)
            web_controller.update_setting('audio_threshold', args.audio_threshold)
            web_controller.update_setting('reactive', args.reactive)

            # Start web server
            web_server_thread = start_web_server()
            if web_server_thread:
                logging.warning("Web GUI available at http://localhost:5001")
            else:
                logging.warning("Failed to start web server")
                web_controller = None
        except Exception as e:
            logging.error(f"Failed to initialize web GUI: {e}")
            web_controller = None
    elif hasattr(args, 'web_gui') and args.web_gui and not WEB_GUI_AVAILABLE:
        logging.error("Web GUI requested but Flask is not available. Install Flask to use web GUI.")

    # Track previous settings for change detection
    previous_settings = None

    def get_current_settings():
        """Get current settings from web controller or fallback to args."""
        if web_controller:
            settings = web_controller.get_settings()
            # Create a namespace object similar to args for compatibility
            class SettingsNamespace:
                def __init__(self, settings_dict, original_args):
                    # Copy all original args first
                    for key, value in vars(original_args).items():
                        setattr(self, key, value)
                    # Override with web controller settings
                    self.fps = settings_dict['fps']
                    self.seconds_per_transition = settings_dict['seconds_per_transition']
                    self.hold = settings_dict['hold']
                    self.pixel_size = settings_dict['pixel_size']
                    self.transition = settings_dict['transition']
                    self.easing = settings_dict['easing']
                    self.audio_threshold = settings_dict['audio_threshold']
                    self.reactive = settings_dict['reactive']
            return SettingsNamespace(settings, args)
        return args

    def check_and_log_settings_changes():
        """Check if settings have changed and log the entire settings JSON if they have."""
        nonlocal previous_settings

        current_settings = get_current_settings()

        # Convert current settings to a comparable dictionary
        current_dict = {}
        if web_controller:
            current_dict = web_controller.get_settings().copy()
        else:
            # Extract relevant settings from args
            current_dict = {
                'fps': current_settings.fps,
                'seconds_per_transition': current_settings.seconds_per_transition,
                'hold': current_settings.hold,
                'pixel_size': current_settings.pixel_size,
                'transition': current_settings.transition,
                'easing': current_settings.easing,
                'audio_threshold': current_settings.audio_threshold,
                'reactive': current_settings.reactive,
            }

        # Compare with previous settings
        if previous_settings is None:
            # First time - just store current settings without logging
            previous_settings = current_dict.copy()
        elif previous_settings != current_dict:
            # Settings have changed - log the entire settings JSON
            settings_json = json.dumps(current_dict, indent=2)
            logging.warning(f"Settings changed:\n{settings_json}")
            previous_settings = current_dict.copy()

        return current_settings

    # Log realtime mode start at WARNING level so it's always visible
    logging.warning(f"Starting realtime slideshow with {len(imgs)-1} images at {args.fps} fps ({W}x{H})")

    # Initialize audio if provided
    audio_initialized = False
    audio_data = None
    if args.audio and PYGAME_AVAILABLE:
        if args.audio.exists():
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                pygame.mixer.music.load(str(args.audio))
                audio_initialized = True
                logging.warning(f"Audio loaded: {args.audio}")  # WARNING level so always visible

                # For reactive mode, we need to load audio data for analysis
                if args.reactive:
                    try:
                        # Load audio data using pygame's Sound for analysis
                        sound = pygame.mixer.Sound(str(args.audio))
                        audio_data = pygame.sndarray.array(sound)
                        logging.info(f"Audio data loaded for reactive mode. Shape: {audio_data.shape}")
                    except Exception as e:
                        logging.error(f"Failed to load audio data for reactive mode: {e}")
                        args.reactive = False  # Fall back to time-based mode

            except Exception as e:
                logging.error(f"Failed to load audio file {args.audio}: {e}")
        else:
            logging.error(f"Audio file not found: {args.audio}")
    elif args.audio and not PYGAME_AVAILABLE:
        logging.warning("Audio file specified but pygame is not available")

    # Create OpenCV window with optimizations
    window_name = "PySlidemorpher - Realtime Slideshow"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)


    # Enable GPU acceleration if available
    try:
        cv2.setUseOptimized(True)
        if cv2.useOptimized():
            logging.info("OpenCV optimizations enabled")
    except:
        pass

    # Frame buffer for smoother playback
    frame_buffer = Queue(maxsize=args.fps * 2)  # Buffer 2 seconds worth of frames

    # Audio intensity monitoring for reactive mode
    audio_features = {'intensity': 0.0, 'peak': 0.0, 'spectral_centroid': 0.0, 'beat_strength': 0.0}
    audio_features_lock = threading.Lock()

    def get_audio_intensity():
        """Get current audio intensity and additional audio features from playing audio."""
        if not audio_initialized or not args.reactive or audio_data is None:
            return {'intensity': 0.0, 'peak': 0.0, 'spectral_centroid': 0.0, 'beat_strength': 0.0}

        try:
            # Get current playback position
            pos = pygame.mixer.music.get_pos()  # Position in milliseconds
            if pos == -1:  # Music not playing
                return {'intensity': 0.0, 'peak': 0.0, 'spectral_centroid': 0.0, 'beat_strength': 0.0}

            # Convert position to sample index
            sample_rate = 22050
            samples_per_ms = sample_rate / 1000.0
            sample_index = int(pos * samples_per_ms)

            # Analyze multiple windows for better audio analysis
            short_window = int(sample_rate * 0.05)  # 50ms for immediate response
            long_window = int(sample_rate * 0.2)  # 200ms for beat detection

            # Short window for immediate intensity
            start_idx = max(0, sample_index - short_window // 2)
            end_idx = min(len(audio_data), start_idx + short_window)

            if start_idx >= end_idx:
                return {'intensity': 0.0, 'peak': 0.0, 'spectral_centroid': 0.0, 'beat_strength': 0.0}

            # Get short window data
            short_data = audio_data[start_idx:end_idx]
            if len(short_data.shape) > 1:  # Stereo audio
                short_data = np.mean(short_data, axis=1)  # Convert to mono

            # Calculate RMS intensity
            rms = np.sqrt(np.mean(short_data.astype(np.float64) ** 2))
            normalized_intensity = min(1.0, rms / 32767.0)

            # Calculate peak amplitude
            peak = np.max(np.abs(short_data.astype(np.float64))) / 32767.0

            # Long window for beat detection
            long_start = max(0, sample_index - long_window // 2)
            long_end = min(len(audio_data), long_start + long_window)
            long_data = audio_data[long_start:long_end]
            if len(long_data.shape) > 1:
                long_data = np.mean(long_data, axis=1)

            # Simple beat detection using energy variance
            if len(long_data) > 0:
                # Split into smaller chunks and calculate energy variance
                chunk_size = len(long_data) // 8
                if chunk_size > 0:
                    chunks = [long_data[i:i + chunk_size] for i in range(0, len(long_data), chunk_size)]
                    energies = [np.mean(chunk.astype(np.float64) ** 2) for chunk in chunks if len(chunk) > 0]
                    if len(energies) > 1:
                        beat_strength = np.std(energies) / (np.mean(energies) + 1e-10)
                        beat_strength = min(1.0, beat_strength * 10)  # Scale and clamp
                    else:
                        beat_strength = 0.0
                else:
                    beat_strength = 0.0
            else:
                beat_strength = 0.0

            # Simple spectral centroid approximation
            if len(short_data) > 1:
                # Use high-frequency content as proxy for spectral centroid
                diff = np.diff(short_data.astype(np.float64))
                spectral_centroid = np.mean(np.abs(diff)) / (np.mean(np.abs(short_data.astype(np.float64))) + 1e-10)
                spectral_centroid = min(1.0, spectral_centroid)
            else:
                spectral_centroid = 0.0

            return {
                'intensity': normalized_intensity,
                'peak': peak,
                'spectral_centroid': spectral_centroid,
                'beat_strength': beat_strength
            }

        except Exception as e:
            logging.debug(f"Error calculating audio intensity: {e}")
            return {'intensity': 0.0, 'peak': 0.0, 'spectral_centroid': 0.0, 'beat_strength': 0.0}

    def audio_monitor():
        """Monitor audio features in a separate thread."""
        nonlocal audio_features
        while audio_initialized:
            try:
                # Check current settings to see if reactive mode is still enabled
                current_settings = check_and_log_settings_changes()
                if not current_settings.reactive:
                    time.sleep(0.1)  # Sleep longer when not in reactive mode
                    continue

                features = get_audio_intensity()
                with audio_features_lock:
                    audio_features = features
                time.sleep(0.01)  # Update every 10ms
            except Exception as e:
                logging.debug(f"Error in audio monitoring: {e}")
                break

    def frame_generator():
        """Generate frames in a separate thread for better performance."""
        try:
            # Check settings and log changes if any
            current_settings = check_and_log_settings_changes()
            if current_settings.reactive:
                # Reactive mode: generate frames on demand based on audio intensity
                reactive_frame_generator()
            else:
                # Standard time-based mode
                standard_frame_generator()
        except Exception as e:
            logging.error(f"Error in frame generation: {e}")
            frame_buffer.put(None)

    def standard_frame_generator():
        """Standard time-based frame generation."""
        # Get current settings (may be updated from web GUI)
        current_settings = check_and_log_settings_changes()

        # Initial hold frames
        init_hold = int(round(current_settings.hold * current_settings.fps))
        logging.debug(f"Displaying initial hold frames: {init_hold}")
        for _ in range(init_hold // 4):
            # Convert RGB to BGR for OpenCV display
            frame_bgr = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR)
            frame_buffer.put(frame_bgr)

        # Process transitions
        for i in range(len(imgs) - 1):
            # Get fresh settings for each transition to allow real-time updates
            current_settings = check_and_log_settings_changes()

            logging.info(f"Processing transition {i + 1}/{len(imgs) - 1}")
            a, b = imgs[i], imgs[i + 1]
            pair_seed = (current_settings.seed or 0) + i * random.randint(1, 10000)

            # Select transition function (moved inside loop for random transitions)
            transition_fn = get_transition_function(current_settings.transition)

            # Log which transition is being used for this image pair
            logging.critical(f"Using transition: {transition_fn.__name__}")

            # Generate transition frames with current settings
            for frame in transition_fn(
                    a, b,
                    pixel_size=current_settings.pixel_size,
                    fps=current_settings.fps,
                    seconds=current_settings.seconds_per_transition,
                    hold=0.0,
                    ease_name=current_settings.easing,
                    seed=pair_seed,
            ):
                # Convert RGB to BGR for OpenCV display
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_buffer.put(frame_bgr)

            # Hold frames with current settings
            current_settings = check_and_log_settings_changes()  # Refresh again for hold duration
            updated_hold = int(round(current_settings.hold * current_settings.fps))
            for _ in range(updated_hold // 4):
                frame_bgr = cv2.cvtColor(b, cv2.COLOR_RGB2BGR)
                frame_buffer.put(frame_bgr)

        # Signal end of frames
        frame_buffer.put(None)

    def reactive_frame_generator():
        """Simple reactive mode: generate frames based on audio loudness threshold."""
        current_img_idx = 0
        current_frame_bgr = cv2.cvtColor(imgs[current_img_idx], cv2.COLOR_RGB2BGR)

        # Simple state tracking
        in_transition = False
        transition_frames = []
        transition_frame_idx = 0
        in_hold = False
        hold_frames_remaining = 0
        last_trigger_time = time.time()

        # Minimum interval between transitions to prevent rapid switching
        min_interval = 0.5  # 500ms minimum between transitions

        # Put initial frame
        frame_buffer.put(current_frame_bgr)

        # Get initial settings
        current_settings = check_and_log_settings_changes()
        logging.info(f"Simple reactive mode started. Audio threshold: {current_settings.audio_threshold}")

        while True:
            try:
                # Get current settings for this iteration (allows real-time updates)
                current_settings = check_and_log_settings_changes()

                # Get current audio intensity (loudness)
                with audio_features_lock:
                    current_intensity = audio_features['intensity']

                current_time = time.time()
                time_since_last = current_time - last_trigger_time

                # Simple trigger logic: loud audio triggers transition, quiet audio doesn't
                is_loud = current_intensity >= current_settings.audio_threshold
                can_trigger = time_since_last >= min_interval

                should_trigger = is_loud and can_trigger and not in_transition and not in_hold

                if should_trigger:
                    # Trigger new transition
                    next_img_idx = (current_img_idx + 1) % len(imgs)
                    if next_img_idx == 0:
                        next_img_idx = 1 if len(imgs) > 1 else 0

                    logging.info(f"Audio trigger! Loudness: {current_intensity:.3f} >= threshold: {current_settings.audio_threshold:.3f}")
                    logging.info(f"Transitioning from image {current_img_idx} to {next_img_idx}")

                    a, b = imgs[current_img_idx], imgs[next_img_idx]
                    pair_seed = (current_settings.seed or 0) + current_img_idx * random.randint(1, 10000)

                    # Choose transition type
                    if current_settings.transition == "random":
                        transition_fn = get_random_transition_function()
                    else:
                        transition_fn = get_transition_function(current_settings.transition)

                    # Log which transition is being used for this image pair
                    logging.info(f"Using transition: {transition_fn.__name__}")

                    # Generate transition frames with standard parameters
                    transition_frames = list(transition_fn(
                        a, b,
                        pixel_size=current_settings.pixel_size,
                        fps=current_settings.fps,
                        seconds=current_settings.seconds_per_transition,
                        hold=0.0,  # We handle hold separately in reactive mode
                        ease_name=current_settings.easing,
                        seed=pair_seed,
                    ))

                    in_transition = True
                    transition_frame_idx = 0
                    current_img_idx = next_img_idx
                    last_trigger_time = current_time

                # Handle frame output
                if in_transition and transition_frames:
                    if transition_frame_idx < len(transition_frames):
                        frame = transition_frames[transition_frame_idx]
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frame_buffer.put(frame_bgr)
                        transition_frame_idx += 1
                    else:
                        # Transition complete, start hold period
                        in_transition = False
                        transition_frames = []
                        current_frame_bgr = cv2.cvtColor(imgs[current_img_idx], cv2.COLOR_RGB2BGR)

                        # Calculate hold frames based on current settings
                        hold_frames_remaining = int(round(current_settings.hold * current_settings.fps))
                        if hold_frames_remaining > 0:
                            in_hold = True
                            logging.debug(f"Starting hold period: {hold_frames_remaining} frames ({current_settings.hold:.2f}s)")

                        frame_buffer.put(current_frame_bgr)
                elif in_hold:
                    # In hold period, display current image
                    frame_buffer.put(current_frame_bgr)
                    hold_frames_remaining -= 1
                    if hold_frames_remaining <= 0:
                        in_hold = False
                        logging.debug("Hold period complete")
                else:
                    # No transition, no hold, display current image
                    frame_buffer.put(current_frame_bgr)

                # Use current frame time from settings
                current_frame_time = 1.0 / current_settings.fps
                time.sleep(current_frame_time)  # Maintain frame rate

            except Exception as e:
                logging.error(f"Error in reactive frame generation: {e}")
                break

    def get_transition_function(transition_name):
        """Get the appropriate transition function based on name."""
        if transition_name == "random":
            return get_random_transition_function()
        elif transition_name == "swarm":
            return make_swarm_transition_frames
        elif transition_name == "tornado":
            return make_tornado_transition_frames
        elif transition_name == "swirl":
            return make_swirl_transition_frames
        elif transition_name == "drip":
            return make_drip_transition_frames
        elif transition_name == "rain":
            return make_rainfall_transition_frames
        elif transition_name == "sorted":
            return make_sorted_transition_frames
        elif transition_name == "hue-sorted":
            return make_hue_sorted_transition_frames
        else:
            return make_transition_frames

    # Start frame generation in separate thread
    generator_thread = threading.Thread(target=frame_generator, daemon=True)
    generator_thread.start()

    # Start audio monitoring thread for reactive mode
    audio_monitor_thread = None
    current_settings = check_and_log_settings_changes()
    if current_settings.reactive and audio_initialized:
        audio_monitor_thread = threading.Thread(target=audio_monitor, daemon=True)
        audio_monitor_thread.start()
        logging.info("Audio monitoring started for reactive mode")

    if current_settings.reactive:
        logging.warning("Starting reactive slideshow. Press 'q' to quit, 'p' to pause/resume, 'r' to restart.")
        logging.warning(f"Audio threshold: {current_settings.audio_threshold:.3f}")
    else:
        logging.warning("Starting realtime playback. Press 'q' to quit, 'p' to pause/resume, 'r' to restart.")

    # Start audio playback if initialized
    if audio_initialized:
        pygame.mixer.music.play(-1)  # Loop indefinitely
        logging.warning("Audio playback started")  # WARNING level so always visible

    paused = False
    stop_requested = False
    start_time = time.time()
    frame_count = 0


    try:
        while True:
            if not paused:
                # Get current settings for dynamic frame timing
                current_settings = check_and_log_settings_changes()
                current_frame_time = 1.0 / current_settings.fps

                # Get next frame from buffer
                try:
                    frame = frame_buffer.get(timeout=1.0)
                    if frame is None:  # End of frames
                        logging.info("Slideshow completed. Restarting...")
                        # Restart audio if initialized
                        if audio_initialized:
                            pygame.mixer.music.stop()
                            pygame.mixer.music.play(-1)  # Loop indefinitely
                        # Restart by creating new generator thread
                        generator_thread = threading.Thread(target=frame_generator, daemon=True)
                        generator_thread.start()
                        continue

                    # Apply audio visualizations if any are enabled
                    if web_controller:
                        settings = web_controller.get_settings()
                        # Check if any visualizations are enabled
                        viz_enabled = any([
                            settings.get('viz_oscilloscope', False),
                            settings.get('viz_lissajous', False),
                            settings.get('viz_spectrum', False),
                            settings.get('viz_waveform', False),
                            settings.get('viz_eq', False)
                        ])

                        if viz_enabled and audio_initialized and audio_data is not None:
                            with audio_features_lock:
                                current_audio_features = audio_features.copy()
                            frame = render_audio_visualizations(
                                frame, audio_data, current_audio_features, settings
                            )

                    # Display frame
                    cv2.imshow(window_name, frame)
                    frame_count += 1


                    # Calculate timing for consistent FPS using current settings
                    expected_time = start_time + (frame_count * current_frame_time)
                    current_time = time.time()
                    sleep_time = expected_time - current_time

                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    elif sleep_time < -current_frame_time:  # If we're more than one frame behind, reset timing
                        start_time = current_time
                        frame_count = 0

                except:
                    # If buffer is empty, just wait a bit
                    time.sleep(current_frame_time)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                if audio_initialized:
                    if paused:
                        pygame.mixer.music.pause()
                    else:
                        pygame.mixer.music.unpause()
                if not paused:
                    # Reset timing when resuming
                    start_time = time.time()
                    frame_count = 0
                logging.info(f"Playback {'paused' if paused else 'resumed'}")
            elif key == ord('r'):
                # Restart slideshow
                logging.info("Restarting slideshow...")
                # Restart audio if initialized
                if audio_initialized:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.play(-1)  # Loop indefinitely
                # Clear buffer
                while not frame_buffer.empty():
                    try:
                        frame_buffer.get_nowait()
                    except:
                        break
                # Start new generator thread
                generator_thread = threading.Thread(target=frame_generator, daemon=True)
                generator_thread.start()
                start_time = time.time()
                frame_count = 0
                paused = False

            # Handle web GUI commands and setting updates
            if web_controller:
                # Check for commands from web interface
                try:
                    while not web_controller.command_queue.empty():
                        command = web_controller.command_queue.get_nowait()
                        if command == 'pause':
                            paused = True
                            if audio_initialized:
                                pygame.mixer.music.pause()
                            logging.info("Paused via web interface")
                        elif command == 'resume':
                            paused = False
                            if audio_initialized:
                                pygame.mixer.music.unpause()
                            start_time = time.time()
                            frame_count = 0
                            logging.info("Resumed via web interface")
                        elif command == 'restart':
                            logging.info("Restarting via web interface...")
                            if audio_initialized:
                                pygame.mixer.music.stop()
                                pygame.mixer.music.play(-1)
                            while not frame_buffer.empty():
                                try:
                                    frame_buffer.get_nowait()
                                except:
                                    break
                            generator_thread = threading.Thread(target=frame_generator, daemon=True)
                            generator_thread.start()
                            start_time = time.time()
                            frame_count = 0
                            paused = False
                        elif command == 'next':
                            # Skip to next image by clearing buffer
                            while not frame_buffer.empty():
                                try:
                                    frame_buffer.get_nowait()
                                except:
                                    break
                            logging.info("Skipped to next image via web interface")
                        elif command == 'stop':
                            logging.info("Stopping slideshow via web interface...")
                            stop_requested = True
                except:
                    pass  # Ignore queue errors

                # Update args with new settings from web interface
                current_settings = web_controller.get_settings()
                settings_changed = False

                if args.fps != current_settings['fps']:
                    args.fps = current_settings['fps']
                    frame_time = 1.0 / args.fps
                    settings_changed = True

                if args.seconds_per_transition != current_settings['seconds_per_transition']:
                    args.seconds_per_transition = current_settings['seconds_per_transition']
                    settings_changed = True

                if args.hold != current_settings['hold']:
                    args.hold = current_settings['hold']
                    settings_changed = True

                if args.pixel_size != current_settings['pixel_size']:
                    args.pixel_size = current_settings['pixel_size']
                    settings_changed = True

                if args.transition != current_settings['transition']:
                    args.transition = current_settings['transition']
                    settings_changed = True

                if args.easing != current_settings['easing']:
                    args.easing = current_settings['easing']
                    settings_changed = True

                if args.audio_threshold != current_settings['audio_threshold']:
                    args.audio_threshold = current_settings['audio_threshold']
                    settings_changed = True

                if args.reactive != current_settings['reactive']:
                    args.reactive = current_settings['reactive']
                    settings_changed = True

                # Update paused state from web interface
                if paused != current_settings['paused']:
                    paused = current_settings['paused']
                    if audio_initialized:
                        if paused:
                            pygame.mixer.music.pause()
                        else:
                            pygame.mixer.music.unpause()
                            start_time = time.time()
                            frame_count = 0

                if settings_changed:
                    logging.info("Settings updated via web interface")

            # Check if stop was requested via web interface
            if stop_requested:
                break

    finally:
        # Stop audio if initialized
        if audio_initialized:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            logging.warning("Audio playback stopped")  # WARNING level so always visible
        cv2.destroyAllWindows()
        logging.warning("Realtime playback ended.")  # WARNING level so always visible


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
from collections import deque

import cv2
import numpy as np

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
                    # Enhanced audio reactivity settings
                    self.tempo_detection = settings_dict.get('tempo_detection', True)
                    self.tempo_to_timing = settings_dict.get('tempo_to_timing', True)
                    self.intensity_to_speed = settings_dict.get('intensity_to_speed', True)
                    self.intensity_to_pixel_size = settings_dict.get('intensity_to_pixel_size', True)
                    self.frequency_to_easing = settings_dict.get('frequency_to_easing', True)
                    self.brightness_modulation = settings_dict.get('brightness_modulation', True)
                    self.beat_sensitivity = settings_dict.get('beat_sensitivity', 0.3)
                    self.peak_sensitivity = settings_dict.get('peak_sensitivity', 0.2)
                    self.intensity_sensitivity = settings_dict.get('intensity_sensitivity', 0.1)
                    self.speed_modulation_range = settings_dict.get('speed_modulation_range', 2.0)
                    self.pixel_size_modulation_range = settings_dict.get('pixel_size_modulation_range', 0.5)
                    self.brightness_modulation_range = settings_dict.get('brightness_modulation_range', 0.1)
                    self.low_freq_threshold = settings_dict.get('low_freq_threshold', 0.4)
                    self.high_freq_threshold = settings_dict.get('high_freq_threshold', 0.3)
                    self.tempo_smoothing = settings_dict.get('tempo_smoothing', 0.8)
                    self.show_audio_debug = settings_dict.get('show_audio_debug', False)
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

    # Enhanced audio analysis for reactive mode
    audio_features = {
        'intensity': 0.0, 
        'peak': 0.0, 
        'spectral_centroid': 0.0, 
        'beat_strength': 0.0,
        'low_freq_energy': 0.0,
        'mid_freq_energy': 0.0,
        'high_freq_energy': 0.0,
        'spectral_rolloff': 0.0,
        'zero_crossing_rate': 0.0,
        'onset_strength': 0.0,
        'estimated_tempo': 0.0
    }
    audio_features_lock = threading.Lock()

    # History tracking for adaptive behavior
    audio_history = deque(maxlen=50)  # Keep last 0.5 seconds at 100Hz update rate
    intensity_history = deque(maxlen=100)  # Keep last 1 second for dynamic thresholds
    beat_history = deque(maxlen=20)  # Keep last 0.2 seconds for beat detection

    # Tempo detection variables
    tempo_history = deque(maxlen=10)  # Keep last 10 tempo estimates for smoothing
    onset_times = deque(maxlen=50)  # Keep last 50 onset times for tempo calculation
    last_tempo_update = time.time()
    current_tempo = 0.0

    def get_audio_intensity():
        """Enhanced audio analysis with frequency domain features and history tracking."""
        if not audio_initialized or not args.reactive or audio_data is None:
            return {
                'intensity': 0.0, 'peak': 0.0, 'spectral_centroid': 0.0, 'beat_strength': 0.0,
                'low_freq_energy': 0.0, 'mid_freq_energy': 0.0, 'high_freq_energy': 0.0,
                'spectral_rolloff': 0.0, 'zero_crossing_rate': 0.0, 'onset_strength': 0.0,
                'estimated_tempo': 0.0
            }

        try:
            # Get current playback position
            pos = pygame.mixer.music.get_pos()  # Position in milliseconds
            if pos == -1:  # Music not playing
                return {
                    'intensity': 0.0, 'peak': 0.0, 'spectral_centroid': 0.0, 'beat_strength': 0.0,
                    'low_freq_energy': 0.0, 'mid_freq_energy': 0.0, 'high_freq_energy': 0.0,
                    'spectral_rolloff': 0.0, 'zero_crossing_rate': 0.0, 'onset_strength': 0.0,
                    'estimated_tempo': 0.0
                }

            # Convert position to sample index
            sample_rate = 22050
            samples_per_ms = sample_rate / 1000.0
            sample_index = int(pos * samples_per_ms)

            # Analysis windows
            short_window = int(sample_rate * 0.05)  # 50ms for immediate response
            long_window = int(sample_rate * 0.2)   # 200ms for beat detection
            fft_window = int(sample_rate * 0.1)    # 100ms for frequency analysis

            # Get short window data for time-domain analysis
            start_idx = max(0, sample_index - short_window // 2)
            end_idx = min(len(audio_data), start_idx + short_window)

            if start_idx >= end_idx:
                return {
                    'intensity': 0.0, 'peak': 0.0, 'spectral_centroid': 0.0, 'beat_strength': 0.0,
                    'low_freq_energy': 0.0, 'mid_freq_energy': 0.0, 'high_freq_energy': 0.0,
                    'spectral_rolloff': 0.0, 'zero_crossing_rate': 0.0, 'onset_strength': 0.0,
                    'estimated_tempo': 0.0
                }

            # Get audio data and convert to mono if needed
            short_data = audio_data[start_idx:end_idx]
            if len(short_data.shape) > 1:  # Stereo audio
                short_data = np.mean(short_data, axis=1)  # Convert to mono

            # Normalize to float
            short_data_float = short_data.astype(np.float64) / 32767.0

            # Basic time-domain features
            rms = np.sqrt(np.mean(short_data_float ** 2))
            intensity = min(1.0, rms)
            peak = np.max(np.abs(short_data_float))

            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(short_data_float)) != 0)
            zero_crossing_rate = zero_crossings / len(short_data_float)

            # FFT analysis for frequency domain features
            fft_start = max(0, sample_index - fft_window // 2)
            fft_end = min(len(audio_data), fft_start + fft_window)
            fft_data = audio_data[fft_start:fft_end]

            if len(fft_data.shape) > 1:
                fft_data = np.mean(fft_data, axis=1)

            fft_data_float = fft_data.astype(np.float64) / 32767.0

            # Apply window function to reduce spectral leakage
            if len(fft_data_float) > 0:
                window = np.hanning(len(fft_data_float))
                windowed_data = fft_data_float * window

                # Compute FFT
                fft = np.fft.rfft(windowed_data)
                magnitude = np.abs(fft)
                power_spectrum = magnitude ** 2

                # Frequency bins
                freqs = np.fft.rfftfreq(len(windowed_data), 1/sample_rate)

                # Frequency band analysis
                low_freq_mask = (freqs >= 20) & (freqs < 250)    # Bass
                mid_freq_mask = (freqs >= 250) & (freqs < 4000)  # Mids
                high_freq_mask = (freqs >= 4000) & (freqs < 11000) # Highs

                total_energy = np.sum(power_spectrum) + 1e-10
                low_freq_energy = np.sum(power_spectrum[low_freq_mask]) / total_energy
                mid_freq_energy = np.sum(power_spectrum[mid_freq_mask]) / total_energy
                high_freq_energy = np.sum(power_spectrum[high_freq_mask]) / total_energy

                # Spectral centroid (brightness)
                if total_energy > 1e-10:
                    spectral_centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
                    spectral_centroid = min(1.0, spectral_centroid / (sample_rate / 2))
                else:
                    spectral_centroid = 0.0

                # Spectral rolloff (85% of energy)
                cumulative_energy = np.cumsum(power_spectrum)
                rolloff_threshold = 0.85 * total_energy
                rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
                if len(rolloff_idx) > 0:
                    spectral_rolloff = freqs[rolloff_idx[0]] / (sample_rate / 2)
                    spectral_rolloff = min(1.0, spectral_rolloff)
                else:
                    spectral_rolloff = 1.0
            else:
                low_freq_energy = mid_freq_energy = high_freq_energy = 0.0
                spectral_centroid = spectral_rolloff = 0.0

            # Enhanced beat detection using onset strength
            long_start = max(0, sample_index - long_window // 2)
            long_end = min(len(audio_data), long_start + long_window)
            long_data = audio_data[long_start:long_end]
            if len(long_data.shape) > 1:
                long_data = np.mean(long_data, axis=1)

            long_data_float = long_data.astype(np.float64) / 32767.0

            # Calculate onset strength using spectral flux
            onset_strength = 0.0
            beat_strength = 0.0

            if len(long_data_float) > 512:  # Need enough samples for analysis
                # Split into overlapping frames for onset detection
                frame_size = 512
                hop_size = 256
                frames = []

                for i in range(0, len(long_data_float) - frame_size, hop_size):
                    frame = long_data_float[i:i + frame_size]
                    window = np.hanning(frame_size)
                    windowed_frame = frame * window
                    fft_frame = np.fft.rfft(windowed_frame)
                    magnitude_frame = np.abs(fft_frame)
                    frames.append(magnitude_frame)

                if len(frames) > 1:
                    # Calculate spectral flux (onset strength)
                    flux = []
                    for i in range(1, len(frames)):
                        diff = frames[i] - frames[i-1]
                        # Only positive differences (increases in energy)
                        positive_diff = np.maximum(0, diff)
                        flux.append(np.sum(positive_diff))

                    if len(flux) > 0:
                        onset_strength = np.mean(flux)
                        onset_strength = min(1.0, onset_strength * 100)  # Scale appropriately

                        # Beat strength from onset strength variance
                        if len(flux) > 1:
                            beat_strength = np.std(flux) / (np.mean(flux) + 1e-10)
                            beat_strength = min(1.0, beat_strength * 5)

            # Tempo detection using onset strength
            nonlocal current_tempo, last_tempo_update, onset_times, tempo_history
            estimated_tempo = current_tempo

            current_time = time.time()

            # Detect onsets (significant increases in onset strength)
            if onset_strength > 0.2 and beat_strength > 0.3:  # Threshold for onset detection
                onset_times.append(current_time)

                # Calculate tempo every 2 seconds or when we have enough onsets
                if (current_time - last_tempo_update > 2.0 and len(onset_times) >= 4) or len(onset_times) >= 20:
                    # Calculate intervals between onsets
                    intervals = []
                    for i in range(1, len(onset_times)):
                        interval = onset_times[i] - onset_times[i-1]
                        if 0.2 < interval < 2.0:  # Filter reasonable intervals (30-300 BPM)
                            intervals.append(interval)

                    if len(intervals) >= 3:
                        # Calculate tempo from median interval
                        median_interval = np.median(intervals)
                        tempo_bpm = 60.0 / median_interval

                        # Filter reasonable tempo range
                        if 60 <= tempo_bpm <= 200:
                            tempo_history.append(tempo_bpm)

                            # Smooth tempo using history
                            if len(tempo_history) > 0:
                                current_settings = get_current_settings()
                                smoothing = current_settings.tempo_smoothing if hasattr(current_settings, 'tempo_smoothing') else 0.8
                                if len(tempo_history) == 1:
                                    estimated_tempo = tempo_bpm
                                else:
                                    estimated_tempo = smoothing * current_tempo + (1 - smoothing) * tempo_bpm
                                current_tempo = estimated_tempo

                    last_tempo_update = current_time

            # Store in history for adaptive behavior
            current_features = {
                'intensity': intensity,
                'peak': peak,
                'spectral_centroid': spectral_centroid,
                'beat_strength': beat_strength,
                'low_freq_energy': low_freq_energy,
                'mid_freq_energy': mid_freq_energy,
                'high_freq_energy': high_freq_energy,
                'spectral_rolloff': spectral_rolloff,
                'zero_crossing_rate': zero_crossing_rate,
                'onset_strength': onset_strength,
                'estimated_tempo': estimated_tempo
            }

            # Update history
            audio_history.append(current_features)
            intensity_history.append(intensity)
            beat_history.append(beat_strength)

            return current_features

        except Exception as e:
            logging.debug(f"Error in enhanced audio analysis: {e}")
            return {
                'intensity': 0.0, 'peak': 0.0, 'spectral_centroid': 0.0, 'beat_strength': 0.0,
                'low_freq_energy': 0.0, 'mid_freq_energy': 0.0, 'high_freq_energy': 0.0,
                'spectral_rolloff': 0.0, 'zero_crossing_rate': 0.0, 'onset_strength': 0.0,
                'estimated_tempo': 0.0
            }

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

                # Share audio features with web controller for debug display
                if web_controller:
                    web_controller.audio_features = features
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
        """Enhanced reactive mode with multiple trigger types and audio-responsive parameters."""
        current_img_idx = 0
        current_frame_bgr = cv2.cvtColor(imgs[current_img_idx], cv2.COLOR_RGB2BGR)

        # Enhanced state tracking
        in_transition = False
        transition_frames = []
        transition_frame_idx = 0
        in_hold = False
        hold_frames_remaining = 0
        last_trigger_time = time.time()

        # Dynamic threshold tracking
        dynamic_intensity_threshold = 0.1
        dynamic_beat_threshold = 0.3
        dynamic_peak_threshold = 0.2

        # Adaptive timing
        base_min_interval = 0.2  # Base minimum interval (200ms)
        current_min_interval = 0.5  # Current adaptive interval

        # Put initial frame
        frame_buffer.put(current_frame_bgr)

        # Get initial settings
        current_settings = check_and_log_settings_changes()
        logging.info(f"Enhanced reactive mode started. Base audio threshold: {current_settings.audio_threshold}")

        while True:
            try:
                # Get current settings for this iteration (allows real-time updates)
                current_settings = check_and_log_settings_changes()

                # Get current audio features
                with audio_features_lock:
                    current_features = audio_features.copy()

                current_time = time.time()
                time_since_last = current_time - last_trigger_time

                # Update dynamic thresholds based on recent history and user settings
                if len(intensity_history) > 10:
                    recent_avg = np.mean(list(intensity_history)[-20:])  # Last 0.2 seconds
                    dynamic_intensity_threshold = max(current_settings.intensity_sensitivity, recent_avg * 1.2)
                else:
                    dynamic_intensity_threshold = current_settings.intensity_sensitivity

                if len(beat_history) > 5:
                    recent_beat_avg = np.mean(list(beat_history)[-10:])  # Last 0.1 seconds
                    dynamic_beat_threshold = max(current_settings.beat_sensitivity, recent_beat_avg * 1.5)
                else:
                    dynamic_beat_threshold = current_settings.beat_sensitivity

                # Use user-configurable peak sensitivity
                dynamic_peak_threshold = current_settings.peak_sensitivity

                # Calculate adaptive minimum interval based on beat strength
                beat_factor = min(2.0, max(0.5, current_features['beat_strength'] * 3))
                current_min_interval = base_min_interval / beat_factor

                # Multiple trigger types with user-configurable sensitivities
                intensity_trigger = (current_features['intensity'] >= 
                                   max(current_settings.audio_threshold, dynamic_intensity_threshold))

                beat_trigger = (current_features['beat_strength'] >= dynamic_beat_threshold and
                               current_features['onset_strength'] > 0.1)

                peak_trigger = (current_features['peak'] >= dynamic_peak_threshold and
                               current_features['peak'] > current_features['intensity'] * 1.2)

                # Combine triggers with timing constraints
                can_trigger = time_since_last >= current_min_interval
                should_trigger = (intensity_trigger or beat_trigger or peak_trigger) and can_trigger and not in_transition and not in_hold

                if should_trigger:
                    # Determine trigger type for logging
                    trigger_type = "intensity" if intensity_trigger else ("beat" if beat_trigger else "peak")

                    # Trigger new transition
                    next_img_idx = (current_img_idx + 1) % len(imgs)
                    if next_img_idx == 0:
                        next_img_idx = 1 if len(imgs) > 1 else 0

                    logging.info(f"Audio trigger ({trigger_type})! Features: intensity={current_features['intensity']:.3f}, "
                               f"beat={current_features['beat_strength']:.3f}, peak={current_features['peak']:.3f}")
                    logging.info(f"Transitioning from image {current_img_idx} to {next_img_idx}")

                    a, b = imgs[current_img_idx], imgs[next_img_idx]
                    pair_seed = (current_settings.seed or 0) + current_img_idx * random.randint(1, 10000)

                    # Choose transition type
                    if current_settings.transition == "random":
                        transition_fn = get_random_transition_function()
                    else:
                        transition_fn = get_transition_function(current_settings.transition)

                    # Audio-responsive transition parameters with user-configurable mappings

                    # Initialize with base settings
                    adaptive_seconds = current_settings.seconds_per_transition
                    adaptive_pixel_size = current_settings.pixel_size
                    adaptive_easing = current_settings.easing

                    # Tempo-based timing (if enabled)
                    if current_settings.tempo_detection and current_settings.tempo_to_timing:
                        if current_features['estimated_tempo'] > 0:
                            # Map tempo to transition timing (faster tempo = faster transitions)
                            tempo_factor = current_features['estimated_tempo'] / 120.0  # Normalize to 120 BPM
                            tempo_factor = min(2.0, max(0.5, tempo_factor))  # Clamp to reasonable range
                            adaptive_seconds = current_settings.seconds_per_transition / tempo_factor

                    # Speed modulation based on intensity (if enabled)
                    if current_settings.intensity_to_speed:
                        max_range = current_settings.speed_modulation_range
                        min_range = 1.0 / max_range

                        intensity_factor = min(max_range, max(min_range, current_features['intensity'] * max_range))
                        beat_factor = min(max_range * 0.75, max(min_range * 1.25, current_features['beat_strength'] * max_range * 0.75))
                        speed_multiplier = (intensity_factor + beat_factor) / 2

                        adaptive_seconds = adaptive_seconds / speed_multiplier

                    # Clamp adaptive seconds to reasonable range
                    adaptive_seconds = max(0.2, min(10.0, adaptive_seconds))

                    # Detail level: higher intensity = smaller pixels (if enabled)
                    if current_settings.intensity_to_pixel_size:
                        modulation_range = current_settings.pixel_size_modulation_range
                        intensity_pixel_factor = max(1.0 - modulation_range, 1.0 - (current_features['intensity'] * modulation_range))
                        adaptive_pixel_size = int(current_settings.pixel_size * intensity_pixel_factor)
                        adaptive_pixel_size = max(1, min(50, adaptive_pixel_size))  # Clamp to reasonable range

                    # Frequency-based easing effects (if enabled)
                    if current_settings.frequency_to_easing:
                        high_threshold = current_settings.high_freq_threshold
                        low_threshold = current_settings.low_freq_threshold

                        if current_features['high_freq_energy'] > high_threshold:
                            # Use sharper easing for high-frequency content
                            adaptive_easing = "ease_in_out_cubic" if current_settings.easing == "linear" else current_settings.easing
                        elif current_features['low_freq_energy'] > low_threshold:
                            # Use smoother easing for bass-heavy content
                            adaptive_easing = "ease_in_out_sine" if current_settings.easing == "linear" else current_settings.easing

                    # Log which transition is being used with adaptive parameters
                    tempo_info = f", tempo={current_features['estimated_tempo']:.1f}BPM" if current_settings.tempo_detection else ""
                    logging.info(f"Using transition: {transition_fn.__name__} with adaptive params: "
                               f"seconds={adaptive_seconds:.2f}, pixel_size={adaptive_pixel_size}, "
                               f"easing={adaptive_easing}{tempo_info}")

                    # Generate transition frames with audio-responsive parameters
                    transition_frames = list(transition_fn(
                        a, b,
                        pixel_size=adaptive_pixel_size,
                        fps=current_settings.fps,
                        seconds=adaptive_seconds,
                        hold=0.0,  # We handle hold separately in reactive mode
                        ease_name=adaptive_easing,
                        seed=pair_seed,
                    ))

                    in_transition = True
                    transition_frame_idx = 0
                    current_img_idx = next_img_idx
                    last_trigger_time = current_time

                # Handle frame output with brightness modulation
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
                    # In hold period, display current image with configurable brightness modulation
                    frame = imgs[current_img_idx].copy()

                    # Configurable brightness modulation based on audio intensity
                    if current_settings.brightness_modulation:
                        modulation_range = current_settings.brightness_modulation_range
                        brightness_factor = 1.0 + (current_features['intensity'] * modulation_range)
                        max_brightness = 1.0 + modulation_range
                        min_brightness = 1.0 - modulation_range * 0.5  # Less dimming than brightening
                        brightness_factor = min(max_brightness, max(min_brightness, brightness_factor))

                        if brightness_factor != 1.0:
                            frame = frame.astype(np.float32)
                            frame *= brightness_factor
                            frame = np.clip(frame, 0, 255).astype(np.uint8)

                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_buffer.put(frame_bgr)
                    hold_frames_remaining -= 1
                    if hold_frames_remaining <= 0:
                        in_hold = False
                        logging.debug("Hold period complete")
                else:
                    # No transition, no hold, display current image with configurable brightness modulation
                    frame = imgs[current_img_idx].copy()

                    # Configurable brightness modulation based on audio intensity
                    if current_settings.brightness_modulation:
                        modulation_range = current_settings.brightness_modulation_range
                        brightness_factor = 1.0 + (current_features['intensity'] * modulation_range)
                        max_brightness = 1.0 + modulation_range
                        min_brightness = 1.0 - modulation_range * 0.5  # Less dimming than brightening
                        brightness_factor = min(max_brightness, max(min_brightness, brightness_factor))

                        if brightness_factor != 1.0:
                            frame = frame.astype(np.float32)
                            frame *= brightness_factor
                            frame = np.clip(frame, 0, 255).astype(np.uint8)

                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_buffer.put(frame_bgr)

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

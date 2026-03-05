"""
Real-time playback functionality for PySlide Morpher.
Handles live slideshow display using OpenCV.
"""

import logging
import random
import threading
import time
from os import environ
from queue import Queue

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
        while audio_initialized and args.reactive:
            try:
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
            if args.reactive:
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
        # Initial hold frames
        init_hold = int(round(args.hold * args.fps))
        logging.debug(f"Displaying initial hold frames: {init_hold}")
        for _ in range(init_hold // 4):
            # Convert RGB to BGR for OpenCV display
            frame_bgr = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR)
            frame_buffer.put(frame_bgr)

        # Process transitions
        for i in range(len(imgs) - 1):
            logging.info(f"Processing transition {i + 1}/{len(imgs) - 1}")
            a, b = imgs[i], imgs[i + 1]
            pair_seed = (args.seed or 0) + i * random.randint(1, 10000)

            # Select transition function (moved inside loop for random transitions)
            transition_fn = get_transition_function(args.transition)

            # Log which transition is being used for this image pair
            logging.critical(f"Using transition: {transition_fn.__name__}")

            # Generate transition frames
            for frame in transition_fn(
                    a, b,
                    pixel_size=args.pixel_size,
                    fps=args.fps,
                    seconds=args.seconds_per_transition,
                    hold=0.0,
                    ease_name=args.easing,
                    seed=pair_seed,
            ):
                # Convert RGB to BGR for OpenCV display
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_buffer.put(frame_bgr)

            # Hold frames
            for _ in range(init_hold // 4):
                frame_bgr = cv2.cvtColor(b, cv2.COLOR_RGB2BGR)
                frame_buffer.put(frame_bgr)

        # Signal end of frames
        frame_buffer.put(None)

    def reactive_frame_generator():
        """Enhanced reactive mode: generate frames based on comprehensive audio analysis."""
        current_img_idx = 0
        current_frame_bgr = cv2.cvtColor(imgs[current_img_idx], cv2.COLOR_RGB2BGR)

        # Enhanced state tracking
        in_transition = False
        transition_frames = []
        transition_frame_idx = 0
        last_trigger_time = time.time()

        # Adaptive parameters based on audio
        base_min_interval = 0.5  # Reduced minimum interval for more responsiveness
        intensity_history = []
        beat_history = []

        # Put initial frame
        frame_buffer.put(current_frame_bgr)
        logging.info(f"Enhanced reactive mode started. Audio threshold: {args.audio_threshold}")

        while True:
            try:
                # Get current audio features
                with audio_features_lock:
                    current_features = audio_features.copy()

                current_intensity = current_features['intensity']
                current_peak = current_features['peak']
                current_beat = current_features['beat_strength']
                current_spectral = current_features['spectral_centroid']

                # Update history for adaptive behavior
                intensity_history.append(current_intensity)
                beat_history.append(current_beat)
                if len(intensity_history) > 50:  # Keep last 0.5 seconds of history
                    intensity_history.pop(0)
                    beat_history.pop(0)

                current_time = time.time()
                time_since_last = current_time - last_trigger_time

                # Calculate adaptive threshold and timing
                avg_intensity = np.mean(intensity_history) if intensity_history else 0.0
                avg_beat = np.mean(beat_history) if beat_history else 0.0

                # Dynamic threshold based on recent audio activity
                adaptive_threshold = max(args.audio_threshold, avg_intensity * 1.2)

                # Dynamic minimum interval based on beat strength
                min_interval = base_min_interval * (1.0 - avg_beat * 0.5)  # Faster transitions with stronger beats
                min_interval = max(0.2, min_interval)  # Never go below 0.2 seconds

                # Enhanced trigger conditions
                intensity_trigger = current_intensity > adaptive_threshold
                beat_trigger = current_beat > 0.3 and time_since_last > min_interval * 0.5
                peak_trigger = current_peak > args.audio_threshold * 1.5 and time_since_last > min_interval * 0.3

                should_trigger = (intensity_trigger or beat_trigger or peak_trigger) and time_since_last > min_interval

                if not in_transition and should_trigger:
                    # Trigger new transition
                    next_img_idx = (current_img_idx + 1) % len(imgs)
                    if next_img_idx == 0:
                        next_img_idx = 1 if len(imgs) > 1 else 0

                    trigger_type = "intensity" if intensity_trigger else ("beat" if beat_trigger else "peak")
                    logging.info(
                        f"Audio trigger ({trigger_type})! I:{current_intensity:.3f} B:{current_beat:.3f} P:{current_peak:.3f}")
                    logging.info(f"Transitioning from image {current_img_idx} to {next_img_idx}")

                    a, b = imgs[current_img_idx], imgs[next_img_idx]
                    pair_seed = (args.seed or 0) + current_img_idx * random.randint(1, 10000)

                    # Audio-responsive transition parameters
                    # Scale transition speed with audio intensity (higher intensity = faster transitions)
                    speed_multiplier = 0.5 + current_intensity * 1.5  # 0.5x to 2.0x speed
                    adaptive_seconds = args.seconds_per_transition / speed_multiplier
                    adaptive_seconds = max(0.3, min(3.0, adaptive_seconds))  # Clamp between 0.3 and 3.0 seconds

                    # Choose transition type - always use random selection when requested
                    if args.transition == "random":
                        transition_fn = get_random_transition_function()
                        # Log audio characteristics for debugging but don't override random selection
                        logging.debug(f"Audio characteristics - Beat: {current_beat:.3f}, Spectral: {current_spectral:.3f}, Peak: {current_peak:.3f}")
                    else:
                        transition_fn = get_transition_function(args.transition)

                    # Log which transition is being used for this image pair
                    logging.info(f"Using transition: {transition_fn.__name__}")

                    # Audio-responsive pixel size (higher intensity = smaller pixels for more detail)
                    adaptive_pixel_size = max(2, int(args.pixel_size * (1.5 - current_intensity)))

                    # Generate transition frames with audio-responsive parameters
                    transition_frames = list(transition_fn(
                        a, b,
                        pixel_size=adaptive_pixel_size,
                        fps=args.fps,
                        seconds=adaptive_seconds,
                        hold=0.0,
                        ease_name=args.easing,
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
                        # Transition complete
                        in_transition = False
                        transition_frames = []
                        current_frame_bgr = cv2.cvtColor(imgs[current_img_idx], cv2.COLOR_RGB2BGR)
                        frame_buffer.put(current_frame_bgr)
                else:
                    # No transition, display current image with subtle audio-reactive effects
                    if current_intensity > args.audio_threshold * 0.5:
                        # Add subtle brightness modulation based on audio
                        brightness_factor = 1.0 + (current_intensity - args.audio_threshold * 0.5) * 0.2
                        brightness_factor = min(1.3, brightness_factor)

                        enhanced_frame = (imgs[current_img_idx].astype(np.float32) * brightness_factor).clip(0,
                                                                                                             255).astype(
                            np.uint8)
                        frame_bgr = cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)
                        frame_buffer.put(frame_bgr)
                    else:
                        frame_buffer.put(current_frame_bgr)

                time.sleep(frame_time)  # Maintain frame rate

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
    if args.reactive and audio_initialized:
        audio_monitor_thread = threading.Thread(target=audio_monitor, daemon=True)
        audio_monitor_thread.start()
        logging.info("Audio monitoring started for reactive mode")

    if args.reactive:
        logging.warning("Starting reactive slideshow. Press 'q' to quit, 'p' to pause/resume, 'r' to restart.")
        logging.warning(f"Audio threshold: {args.audio_threshold:.3f}")
    else:
        logging.warning("Starting realtime playback. Press 'q' to quit, 'p' to pause/resume, 'r' to restart.")

    # Start audio playback if initialized
    if audio_initialized:
        pygame.mixer.music.play(-1)  # Loop indefinitely
        logging.warning("Audio playback started")  # WARNING level so always visible

    paused = False
    start_time = time.time()
    frame_count = 0

    try:
        while True:
            if not paused:
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

                    # Calculate timing for consistent FPS
                    expected_time = start_time + (frame_count * frame_time)
                    current_time = time.time()
                    sleep_time = expected_time - current_time

                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    elif sleep_time < -frame_time:  # If we're more than one frame behind, reset timing
                        start_time = current_time
                        frame_count = 0

                except:
                    # If buffer is empty, just wait a bit
                    time.sleep(frame_time)

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

    finally:
        # Stop audio if initialized
        if audio_initialized:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            logging.warning("Audio playback stopped")  # WARNING level so always visible
        cv2.destroyAllWindows()
        logging.warning("Realtime playback ended.")  # WARNING level so always visible

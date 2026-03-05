"""
Real-time playback functionality for PySlide Morpher.
Handles live slideshow display using OpenCV.
"""

import logging
import random
import time
import threading
from queue import Queue
import cv2
import numpy as np

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
        make_transition_frames,        # default
        make_swarm_transition_frames,  # swarm
        make_tornado_transition_frames, # tornado
        make_swirl_transition_frames,  # swirl
        make_drip_transition_frames,   # drip
        # make_rainfall_transition_frames, # rain
        make_sorted_transition_frames, # sorted
        make_hue_sorted_transition_frames, # hue-sorted
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

    # Initialize audio if provided
    audio_initialized = False
    audio_data = None
    if args.audio and PYGAME_AVAILABLE:
        if args.audio.exists():
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                pygame.mixer.music.load(str(args.audio))
                audio_initialized = True
                logging.info(f"Audio loaded: {args.audio}")

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
    audio_intensity = 0.0
    audio_intensity_lock = threading.Lock()

    def get_audio_intensity():
        """Get current audio intensity from playing audio."""
        if not audio_initialized or not args.reactive or audio_data is None:
            return 0.0

        try:
            # Get current playback position
            pos = pygame.mixer.music.get_pos()  # Position in milliseconds
            if pos == -1:  # Music not playing
                return 0.0

            # Convert position to sample index
            sample_rate = 22050
            samples_per_ms = sample_rate / 1000.0
            sample_index = int(pos * samples_per_ms)

            # Analyze a small window of audio data around current position
            window_size = int(sample_rate * 0.1)  # 100ms window
            start_idx = max(0, sample_index - window_size // 2)
            end_idx = min(len(audio_data), start_idx + window_size)

            if start_idx >= end_idx:
                return 0.0

            # Calculate RMS (Root Mean Square) intensity
            window_data = audio_data[start_idx:end_idx]
            if len(window_data.shape) > 1:  # Stereo audio
                window_data = np.mean(window_data, axis=1)  # Convert to mono

            rms = np.sqrt(np.mean(window_data.astype(np.float64) ** 2))
            # Normalize to 0-1 range (assuming 16-bit audio)
            normalized_intensity = min(1.0, rms / 32767.0)

            return normalized_intensity

        except Exception as e:
            logging.debug(f"Error calculating audio intensity: {e}")
            return 0.0

    def audio_monitor():
        """Monitor audio intensity in a separate thread."""
        nonlocal audio_intensity
        while audio_initialized and args.reactive:
            try:
                intensity = get_audio_intensity()
                with audio_intensity_lock:
                    audio_intensity = intensity
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

            # Select transition function
            transition_fn = get_transition_function(args.transition)

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
        """Reactive mode: generate frames based on audio intensity."""
        current_img_idx = 0
        current_frame_bgr = cv2.cvtColor(imgs[current_img_idx], cv2.COLOR_RGB2BGR)

        # Put initial frame
        frame_buffer.put(current_frame_bgr)

        logging.info(f"Reactive mode started. Audio threshold: {args.audio_threshold}")

        last_trigger_time = time.time()
        min_transition_interval = 1.0  # Minimum 1 second between transitions

        while True:
            try:
                # Get current audio intensity
                with audio_intensity_lock:
                    current_intensity = audio_intensity

                # Check if intensity exceeds threshold and enough time has passed
                current_time = time.time()
                if (current_intensity > args.audio_threshold and 
                    current_time - last_trigger_time > min_transition_interval):

                    # Trigger transition to next image
                    next_img_idx = (current_img_idx + 1) % len(imgs)
                    if next_img_idx == 0:  # Skip the duplicate first image at the end
                        next_img_idx = 1 if len(imgs) > 1 else 0

                    logging.info(f"Audio trigger detected! Intensity: {current_intensity:.3f} > {args.audio_threshold:.3f}")
                    logging.info(f"Transitioning from image {current_img_idx} to {next_img_idx}")

                    a, b = imgs[current_img_idx], imgs[next_img_idx]
                    pair_seed = (args.seed or 0) + current_img_idx * random.randint(1, 10000)

                    # Select transition function
                    transition_fn = get_transition_function(args.transition)

                    # Generate and queue transition frames
                    for frame in transition_fn(
                            a, b,
                            pixel_size=args.pixel_size,
                            fps=args.fps,
                            seconds=args.seconds_per_transition,
                            hold=0.0,
                            ease_name=args.easing,
                            seed=pair_seed,
                    ):
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frame_buffer.put(frame_bgr)

                    current_img_idx = next_img_idx
                    current_frame_bgr = cv2.cvtColor(imgs[current_img_idx], cv2.COLOR_RGB2BGR)
                    last_trigger_time = current_time
                else:
                    # No trigger, just display current image
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
        logging.info("Starting reactive slideshow. Press 'q' to quit, 'p' to pause/resume, 'r' to restart.")
        logging.info(f"Audio threshold: {args.audio_threshold:.3f}")
    else:
        logging.info("Starting realtime playback. Press 'q' to quit, 'p' to pause/resume, 'r' to restart.")

    # Start audio playback if initialized
    if audio_initialized:
        pygame.mixer.music.play(-1)  # Loop indefinitely
        logging.info("Audio playback started")

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
            logging.info("Audio playback stopped")
        cv2.destroyAllWindows()
        logging.info("Realtime playback ended.")

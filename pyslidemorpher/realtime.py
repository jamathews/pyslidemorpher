
"""
Real-time playback functionality for PySlide Morpher.
Handles live slideshow display using OpenCV.
"""

import json
import logging
import math
import random
import subprocess
import threading
import time
from os import environ
from pathlib import Path
from queue import Queue

import cv2
import numpy as np

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

try:
    import pygame
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


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


def _play_audio_loop(audio_file):
    """Play audio file in a loop for the duration of the slideshow."""
    try:
        # Load and play the audio file
        pygame.mixer.music.load(str(audio_file))
        pygame.mixer.music.play(-1)  # -1 means loop indefinitely
        logging.debug(f"Audio loop started for: {audio_file}")

        # Keep the thread alive while audio is playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    except Exception as e:
        logging.error(f"Error in audio playback: {e}")


def _build_audio_envelope(audio_file, envelope_fps):
    """Build a normalized RMS envelope using ffmpeg-decoded PCM data."""
    if envelope_fps <= 0:
        return None

    try:
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", str(audio_file),
            "-ac", "1",
            "-ar", "44100",
            "-f", "f32le",
            "-"
        ]
        result = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError:
        logging.warning("Reactive mode requested, but ffmpeg is not installed.")
        return None
    except subprocess.CalledProcessError as err:
        stderr = err.stderr.decode("utf-8", errors="ignore").strip()
        logging.warning(f"Could not analyze audio for reactive mode: {stderr}")
        return None

    samples = np.frombuffer(result.stdout, dtype=np.float32)
    if samples.size == 0:
        logging.warning("Reactive mode audio analysis produced no samples.")
        return None

    sample_rate = 44100
    window = max(1, int(sample_rate / envelope_fps))
    usable = (samples.size // window) * window
    if usable == 0:
        return None

    samples = samples[:usable].reshape(-1, window)
    rms = np.sqrt(np.mean(samples * samples, axis=1))

    low = float(np.percentile(rms, 5))
    high = float(np.percentile(rms, 98))
    if high <= low:
        normalized = np.clip(rms, 0.0, 1.0)
    else:
        normalized = np.clip((rms - low) / (high - low), 0.0, 1.0)

    duration = samples.shape[0] / envelope_fps
    return {
        "values": normalized.astype(np.float32),
        "fps": float(envelope_fps),
        "duration": float(duration),
    }


def _current_audio_level(audio_envelope, audio_start_time):
    """Get current normalized audio level based on elapsed looped playback time."""
    if not audio_envelope or audio_start_time is None:
        return 0.0

    duration = audio_envelope["duration"]
    if duration <= 0:
        return 0.0

    elapsed = (time.time() - audio_start_time) % duration
    idx = int(elapsed * audio_envelope["fps"])
    idx = max(0, min(idx, len(audio_envelope["values"]) - 1))
    return float(audio_envelope["values"][idx])


def _apply_audio_reactive_effect(frame_bgr, level, elapsed_time):
    """Apply a dramatic audio-reactive effect stack for gallery-style playback."""
    if level <= 0:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    drive = float(np.clip(level, 0.0, 1.0)) ** 1.35

    # Strong pulse zoom and rhythmic rotation to make beats physically visible.
    zoom = 1.0 + 0.20 * drive + 0.06 * math.sin(elapsed_time * 2.4)
    if zoom > 1.001:
        crop_w = max(2, int(w / zoom))
        crop_h = max(2, int(h / zoom))
        x0 = (w - crop_w) // 2
        y0 = (h - crop_h) // 2
        pulsed = cv2.resize(frame_bgr[y0:y0 + crop_h, x0:x0 + crop_w], (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        pulsed = frame_bgr

    angle = (12.0 * drive) * math.sin(elapsed_time * 3.8)
    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    pulsed = cv2.warpAffine(pulsed, rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Chroma drift for broad spectral movement.
    hsv = cv2.cvtColor(pulsed, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_shift = 35.0 * drive + 16.0 * math.sin(elapsed_time * 1.25)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.15 + 0.85 * drive), 0.0, 255.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + 0.22 * drive), 0.0, 255.0)
    shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # RGB channel split gives a kinetic prism/ghosting artifact on loud passages.
    split_px = max(1, int(3 + (24 * drive)))
    b, g, r = cv2.split(shifted)
    b = np.roll(b, -split_px, axis=1)
    r = np.roll(r, split_px, axis=0)
    split = cv2.merge((b, g, r))

    # Bloom and strobe flash for dramatic impact peaks.
    sigma = 2.0 + (14.0 * drive)
    flash = max(0.0, drive - 0.66) * 2.8
    if flash > 0:
        split = cv2.convertScaleAbs(split, alpha=1.0 + flash, beta=65.0 * flash)
    bloom = cv2.GaussianBlur(shifted, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = cv2.addWeighted(split, 1.0, bloom, 0.34 + (0.7 * drive), 0.0)

    # Pulsing vignette to focus energy into the center and add tunnel-like depth.
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt(((xx - cx) / max(1, cx)) ** 2 + ((yy - cy) / max(1, cy)) ** 2)
    vignette = np.clip(1.22 - (0.55 + 0.35 * drive) * dist, 0.45, 1.45)
    out = np.clip(out.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)

    return out


def play_realtime(imgs, args):
    """Play slideshow in realtime using OpenCV display with optimized performance."""
    W, H = args.size
    frame_time = 1.0 / args.fps  # Time per frame in seconds

    # Initialize audio if provided
    audio_thread = None
    audio_start_time = None
    reactive_enabled = bool(getattr(args, "reactive", False))
    audio_envelope = None
    if hasattr(args, 'audio') and args.audio and args.audio.exists():
        if reactive_enabled:
            audio_envelope = _build_audio_envelope(args.audio, envelope_fps=max(args.fps, 24))
            if audio_envelope is None:
                reactive_enabled = False
                logging.warning("Reactive mode disabled: audio analysis unavailable for this file/environment.")
            else:
                logging.info("Reactive mode enabled: audio-driven pulse, chroma drift, and bloom are active.")
        if AUDIO_AVAILABLE:
            try:
                pygame.mixer.init()
                audio_thread = threading.Thread(target=_play_audio_loop, args=(args.audio,), daemon=True)
                audio_thread.start()
                audio_start_time = time.time()
                logging.info(f"Started audio playback: {args.audio}")
            except Exception as e:
                logging.error(f"Failed to initialize audio: {e}")
                if reactive_enabled:
                    reactive_enabled = False
                    logging.warning("Reactive mode disabled: audio playback could not be started.")
        else:
            logging.warning("Audio file provided but pygame is not available. Install pygame for audio support.")
            if reactive_enabled:
                reactive_enabled = False
                logging.warning("Reactive mode disabled: pygame is required for synced audio playback.")
    elif hasattr(args, 'audio') and args.audio and not args.audio.exists():
        logging.error(f"Audio file not found: {args.audio}")

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



    def frame_generator():
        """Generate frames in a separate thread for better performance."""
        try:
            # Check settings and log changes if any
            current_settings = check_and_log_settings_changes()
            # Standard mode: generate frames based on time
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

    current_settings = check_and_log_settings_changes()
    logging.warning("Starting realtime playback. Press 'q' to quit, 'p' to pause/resume, 'r' to restart.")

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
                        # Restart by creating new generator thread
                        generator_thread = threading.Thread(target=frame_generator, daemon=True)
                        generator_thread.start()
                        continue

                    # Display frame
                    if reactive_enabled:
                        reactive_level = _current_audio_level(audio_envelope, audio_start_time)
                        frame = _apply_audio_reactive_effect(frame, reactive_level, time.time() - start_time)
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
                if not paused:
                    # Reset timing when resuming
                    start_time = time.time()
                    frame_count = 0
                logging.info(f"Playback {'paused' if paused else 'resumed'}")
            elif key == ord('r'):
                # Restart slideshow
                logging.info("Restarting slideshow...")
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
                            logging.info("Paused via web interface")
                        elif command == 'resume':
                            paused = False
                            start_time = time.time()
                            frame_count = 0
                            logging.info("Resumed via web interface")
                        elif command == 'restart':
                            logging.info("Restarting via web interface...")
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


                # Update paused state from web interface
                if paused != current_settings['paused']:
                    paused = current_settings['paused']
                    if not paused:
                        start_time = time.time()
                        frame_count = 0

                if settings_changed:
                    logging.info("Settings updated via web interface")

            # Check if stop was requested via web interface
            if stop_requested:
                break

    finally:
        # Stop audio if it was playing
        if audio_thread and AUDIO_AVAILABLE:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                logging.debug("Audio playback stopped")
            except Exception as e:
                logging.error(f"Error stopping audio: {e}")

        cv2.destroyAllWindows()
        logging.warning("Realtime playback ended.")  # WARNING level so always visible

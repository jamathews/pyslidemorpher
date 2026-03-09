"""Real-time playback orchestration for PySlide Morpher."""

import json
import logging
import platform
import random
import threading
import time
from os import environ
from pathlib import Path
from queue import Queue

import cv2

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

try:
    import pygame
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    sd = None
    SOUNDDEVICE_AVAILABLE = False

try:
    from .web_gui import get_controller, start_web_server, FLASK_AVAILABLE
    WEB_GUI_AVAILABLE = FLASK_AVAILABLE
except ImportError:
    WEB_GUI_AVAILABLE = False
    logging.warning("Web GUI not available - Flask or web_gui module not found")

from .realtime_audio import (
    _build_audio_envelope,
    _current_audio_features,
    FfmpegLiveAudioAnalyzer,
    LiveAudioAnalyzer,
)
from .realtime_effects import _apply_audio_reactive_effect, _resolve_reactive_controls
from .realtime_runtime import (
    _handle_web_command,
    _restart_playback,
    _start_frame_generator,
    _sync_args_from_web_settings,
    SettingsNamespace,
)
from .realtime_shared import DEFAULT_REACTIVE_CONTROLS
from .realtime_transitions import get_random_transition_function, get_transition_function


def _play_audio_loop(audio_file):
    """Play audio file in a loop for the duration of the slideshow."""
    try:
        pygame.mixer.music.load(str(audio_file))
        pygame.mixer.music.play(-1)
        logging.debug(f"Audio loop started for: {audio_file}")

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

    except Exception as e:
        logging.error(f"Error in audio playback: {e}")

def play_realtime(imgs, args):
    """Play slideshow in realtime using OpenCV display with optimized performance."""
    W, H = args.size

    # Initialize audio if provided
    audio_start_time = None
    reactive_enabled = bool(getattr(args, "reactive", False))
    active_audio_source = "none"
    live_audio_analyzer = None
    current_audio_device = getattr(args, "audio_device", None)
    audio_envelope = None
    audio_file = None
    if hasattr(args, 'audio') and args.audio:
        candidate_path = Path(str(args.audio))
        if candidate_path.exists() and candidate_path.is_file():
            audio_file = candidate_path

    if current_audio_device and str(current_audio_device).strip() == "__file__" and not audio_file:
        current_audio_device = None

    if reactive_enabled and current_audio_device and str(current_audio_device).strip() != "__file__":
        if SOUNDDEVICE_AVAILABLE:
            try:
                live_audio_analyzer = LiveAudioAnalyzer(
                    current_audio_device,
                    sd,
                    envelope_fps=max(args.fps, 24),
                )
                live_audio_analyzer.start()
                active_audio_source = "device"
                logging.info(f"Reactive source: live input device ({current_audio_device})")
            except Exception as e:
                logging.error(f"Failed to start live audio input device {current_audio_device}: {e}")
                active_audio_source = "none"
        elif platform.system() == "Darwin":
            try:
                live_audio_analyzer = FfmpegLiveAudioAnalyzer(current_audio_device, envelope_fps=max(args.fps, 24))
                live_audio_analyzer.start()
                active_audio_source = "device"
                logging.info(f"Reactive source: ffmpeg/AVFoundation input ({current_audio_device})")
            except Exception as e:
                logging.error(f"Failed to start ffmpeg live audio input {current_audio_device}: {e}")
                active_audio_source = "none"
        else:
            logging.warning("Reactive device mode requested but sounddevice is not available.")
            active_audio_source = "none"

    def stop_audio_file_playback():
        nonlocal audio_start_time
        if AUDIO_AVAILABLE:
            try:
                if pygame.mixer.get_init():
                    pygame.mixer.music.stop()
            except Exception:
                pass
        audio_start_time = None

    def start_audio_file_playback(file_path):
        nonlocal audio_start_time
        if not AUDIO_AVAILABLE or file_path is None:
            return
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(str(file_path))
            pygame.mixer.music.play(-1)
            audio_start_time = time.time()
            logging.info(f"Started audio playback: {file_path}")
        except Exception as e:
            logging.error(f"Failed to initialize audio playback for {file_path}: {e}")

    def sync_audio_file_setting(settings_obj):
        """Reload file-track audio if the selected file path changed."""
        nonlocal audio_file, audio_envelope, active_audio_source
        desired_audio = getattr(settings_obj, "audio", getattr(args, "audio", "")) or ""
        desired_audio = str(desired_audio).strip()
        current_audio = str(audio_file) if audio_file is not None else ""
        if desired_audio == current_audio:
            return

        stop_audio_file_playback()
        if desired_audio:
            cand = Path(desired_audio)
            if cand.exists() and cand.is_file():
                audio_file = cand
                audio_envelope = None
                if reactive_enabled and str(current_audio_device).strip() == "__file__":
                    audio_envelope = _build_audio_envelope(audio_file, envelope_fps=max(args.fps, 24))
                if str(current_audio_device).strip() == "__file__":
                    start_audio_file_playback(audio_file)
                    active_audio_source = "file"
                logging.info(f"Loaded audio file track: {audio_file}")
            else:
                audio_file = None
                if str(current_audio_device).strip() == "__file__":
                    active_audio_source = "none"
                logging.warning(f"Configured audio file does not exist: {desired_audio}")
        else:
            audio_file = None
            audio_envelope = None
            if str(current_audio_device).strip() == "__file__":
                active_audio_source = "none"

    if audio_file:
        if reactive_enabled:
            if active_audio_source != "device":
                audio_envelope = _build_audio_envelope(audio_file, envelope_fps=max(args.fps, 24))
                if audio_envelope is None:
                    active_audio_source = "none"
                    logging.warning("Reactive file-source analysis unavailable for this file/environment.")
                else:
                    active_audio_source = "file"
                    logging.info("Reactive mode enabled: stackable pulse/warp/color/glow/strobe/trails are active.")
        if AUDIO_AVAILABLE:
            start_audio_file_playback(audio_file)
        else:
            logging.warning("Audio file provided but pygame is not available. Install pygame for audio support.")
    elif hasattr(args, 'audio') and args.audio and not current_audio_device:
        logging.error(f"Audio file not found: {args.audio}")

    # Initialize web GUI controller if requested and available
    web_controller = None
    web_server_thread = None
    if hasattr(args, 'web_gui') and args.web_gui and WEB_GUI_AVAILABLE:
        try:
            web_controller = get_controller()
            # Apply only explicit CLI overrides so persisted settings remain the startup baseline.
            cli_overrides = set(getattr(args, "_cli_overrides", set()))
            base_keys = [
                'fps', 'seconds_per_transition', 'hold', 'pixel_size',
                'transition', 'easing', 'reactive_style', 'reactive_enabled',
                'audio', 'audio_device',
                'window_width', 'window_height', 'window_x', 'window_y',
            ]
            for key in base_keys:
                if key in cli_overrides and hasattr(args, key):
                    web_controller.update_setting(key, getattr(args, key))
            if 'reactive' in cli_overrides:
                web_controller.update_setting('reactive_enabled', bool(getattr(args, 'reactive', False)))
            for key, value in DEFAULT_REACTIVE_CONTROLS.items():
                if key in cli_overrides:
                    web_controller.update_setting(key, getattr(args, key, value))

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
                'reactive_enabled': getattr(current_settings, 'reactive_enabled', getattr(args, 'reactive', False)),
                'reactive_style': getattr(current_settings, 'reactive_style', 'dramatic'),
                'reactive_master_gain': getattr(current_settings, 'reactive_master_gain', 1.0),
                'audio': getattr(current_settings, 'audio', ''),
                'audio_device': getattr(current_settings, 'audio_device', ''),
                'window_width': getattr(current_settings, 'window_width', W),
                'window_height': getattr(current_settings, 'window_height', H),
                'window_x': getattr(current_settings, 'window_x', 80),
                'window_y': getattr(current_settings, 'window_y', 80),
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
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


    # Enable GPU acceleration if available
    try:
        cv2.setUseOptimized(True)
        if cv2.useOptimized():
            logging.info("OpenCV optimizations enabled")
    except:
        pass

    # Frame buffer for smoother playback
    frame_buffer = Queue(maxsize=args.fps * 2)  # Buffer 2 seconds worth of frames
    last_window_state = None

    def apply_window_settings(settings_obj):
        """Apply OpenCV window geometry from current settings if changed."""
        nonlocal last_window_state
        width = int(getattr(settings_obj, "window_width", W) or W)
        height = int(getattr(settings_obj, "window_height", H) or H)
        x = int(getattr(settings_obj, "window_x", 80) or 80)
        y = int(getattr(settings_obj, "window_y", 80) or 80)
        state = (width, height, x, y)
        if state == last_window_state:
            return
        try:
            cv2.resizeWindow(window_name, width, height)
            cv2.moveWindow(window_name, x, y)
            last_window_state = state
        except Exception:
            pass

    def sync_reactive_audio_source(settings_obj):
        """Switch reactive source between file and input device when requested."""
        nonlocal current_audio_device, live_audio_analyzer, active_audio_source, audio_envelope, reactive_enabled
        desired_enabled = bool(getattr(settings_obj, "reactive_enabled", getattr(args, "reactive", False)))
        if desired_enabled != reactive_enabled:
            reactive_enabled = desired_enabled
            if not reactive_enabled:
                if live_audio_analyzer is not None:
                    live_audio_analyzer.stop()
                    live_audio_analyzer = None
                active_audio_source = "none"
                return

        if not reactive_enabled:
            return
        sync_audio_file_setting(settings_obj)
        desired = getattr(settings_obj, "audio_device", current_audio_device)
        desired = (str(desired).strip() if desired is not None else "")
        if desired == "":
            desired = "__file__" if audio_file else "__default__"
        if desired == "__file__" and audio_file is None:
            desired = "__default__"
        if desired == current_audio_device:
            return

        if live_audio_analyzer is not None:
            live_audio_analyzer.stop()
            live_audio_analyzer = None

        current_audio_device = desired
        if desired == "__file__":
            if audio_file is None:
                active_audio_source = "none"
                logging.warning("Audio source set to file, but no audio file is loaded.")
            else:
                if audio_envelope is None:
                    audio_envelope = _build_audio_envelope(audio_file, envelope_fps=max(args.fps, 24))
                active_audio_source = "file" if audio_envelope is not None else "none"
                start_audio_file_playback(audio_file)
                logging.info("Reactive source switched to audio file track.")
            return

        stop_audio_file_playback()

        if not SOUNDDEVICE_AVAILABLE:
            if platform.system() != "Darwin":
                active_audio_source = "none"
                logging.warning("Reactive source switch ignored: sounddevice not available.")
                return
            try:
                live_audio_analyzer = FfmpegLiveAudioAnalyzer(desired, envelope_fps=max(args.fps, 24))
                live_audio_analyzer.start()
                active_audio_source = "device"
                logging.info(f"Reactive source switched to ffmpeg/AVFoundation input ({desired}).")
                return
            except Exception as e:
                active_audio_source = "none"
                live_audio_analyzer = None
                logging.error(f"Failed to switch to ffmpeg input {desired}: {e}")
                return

        try:
            live_audio_analyzer = LiveAudioAnalyzer(
                desired,
                sd,
                envelope_fps=max(args.fps, 24),
            )
            live_audio_analyzer.start()
            active_audio_source = "device"
            logging.info(f"Reactive source switched to live input device ({desired}).")
        except Exception as e:
            active_audio_source = "none"
            live_audio_analyzer = None
            logging.error(f"Failed to switch to audio device {desired}: {e}")



    def frame_generator():
        """Generate frames in a separate thread for better performance."""
        try:
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


    # Start frame generation in separate thread
    generator_thread = _start_frame_generator(frame_generator)

    current_settings = check_and_log_settings_changes()
    apply_window_settings(current_settings)
    sync_audio_file_setting(current_settings)
    sync_reactive_audio_source(current_settings)
    logging.warning("Starting realtime playback. Press 'q' to quit, 'p' to pause/resume, 'r' to restart.")

    paused = False
    stop_requested = False
    start_time = time.time()
    frame_count = 0
    previous_reactive_frame = None


    try:
        while True:
            if not paused:
                # Get current settings for dynamic frame timing
                current_settings = check_and_log_settings_changes()
                apply_window_settings(current_settings)
                sync_audio_file_setting(current_settings)
                sync_reactive_audio_source(current_settings)
                current_frame_time = 1.0 / current_settings.fps

                # Get next frame from buffer
                try:
                    frame = frame_buffer.get(timeout=1.0)
                    if frame is None:  # End of frames
                        logging.info("Slideshow completed. Restarting...")
                        generator_thread = _start_frame_generator(frame_generator)
                        previous_reactive_frame = None
                        continue

                    # Display frame
                    if reactive_enabled:
                        if active_audio_source == "device" and live_audio_analyzer is not None:
                            reactive_features = live_audio_analyzer.get_features()
                        else:
                            reactive_features = _current_audio_features(audio_envelope, audio_start_time)
                        reactive_controls = _resolve_reactive_controls(current_settings)
                        frame = _apply_audio_reactive_effect(
                            frame,
                            reactive_features,
                            time.time() - start_time,
                            style=getattr(current_settings, "reactive_style", "dramatic"),
                            previous_frame=previous_reactive_frame,
                            controls=reactive_controls,
                        )
                        previous_reactive_frame = frame.copy()
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
                generator_thread = _restart_playback(frame_buffer, frame_generator)
                start_time = time.time()
                frame_count = 0
                paused = False
                previous_reactive_frame = None

            # Handle web GUI commands and setting updates
            if web_controller:
                # Check for commands from web interface
                try:
                    while not web_controller.command_queue.empty():
                        command = web_controller.command_queue.get_nowait()
                        (
                            paused,
                            start_time,
                            frame_count,
                            previous_reactive_frame,
                            cmd_stop_requested,
                            cmd_generator_thread,
                        ) = _handle_web_command(
                            command,
                            frame_buffer,
                            frame_generator,
                            paused,
                            start_time,
                            frame_count,
                            previous_reactive_frame,
                        )
                        stop_requested = stop_requested or cmd_stop_requested
                        if cmd_generator_thread is not None:
                            generator_thread = cmd_generator_thread
                except:
                    pass  # Ignore queue errors

                # Update args with new settings from web interface
                current_settings = web_controller.get_settings()
                settings_changed = _sync_args_from_web_settings(args, current_settings)


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
        if live_audio_analyzer is not None:
            live_audio_analyzer.stop()

        # Stop audio if it was playing
        if AUDIO_AVAILABLE:
            try:
                if pygame.mixer.get_init():
                    pygame.mixer.music.stop()
                    pygame.mixer.quit()
                logging.debug("Audio playback stopped")
            except Exception as e:
                logging.error(f"Error stopping audio: {e}")

        cv2.destroyAllWindows()
        logging.warning("Realtime playback ended.")  # WARNING level so always visible

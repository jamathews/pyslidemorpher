"""Runtime helpers for realtime playback loop."""

import logging
import threading
import time

from .realtime_shared import DEFAULT_REACTIVE_CONTROLS

WEB_SYNC_KEYS = [
    "fps",
    "seconds_per_transition",
    "hold",
    "pixel_size",
    "transition",
    "easing",
    "window_width",
    "window_height",
    "window_x",
    "window_y",
    "reactive_style",
    "audio_device",
    "audio",
]


class SettingsNamespace:
    """Namespace built from args + optional web-controller overrides."""

    def __init__(self, settings_dict, original_args):
        for key, value in vars(original_args).items():
            setattr(self, key, value)
        for key, value in settings_dict.items():
            setattr(self, key, value)


def _drain_queue_nowait(queue_obj):
    """Best-effort non-blocking queue drain."""
    while not queue_obj.empty():
        try:
            queue_obj.get_nowait()
        except Exception:
            break


def _start_frame_generator(frame_generator):
    """Start and return a daemon frame-generator thread."""
    generator_thread = threading.Thread(target=frame_generator, daemon=True)
    generator_thread.start()
    return generator_thread


def _sync_args_from_web_settings(args, current_settings):
    """Copy dynamic web settings into args namespace for compatibility."""
    settings_changed = False

    for key in WEB_SYNC_KEYS:
        new_value = current_settings.get(key)
        if getattr(args, key, None) != new_value:
            setattr(args, key, new_value)
            settings_changed = True

    reactive_enabled = bool(current_settings.get("reactive_enabled", getattr(args, "reactive", False)))
    if getattr(args, "reactive", False) != reactive_enabled:
        args.reactive = reactive_enabled
        settings_changed = True

    for key, default_value in DEFAULT_REACTIVE_CONTROLS.items():
        current_value = current_settings.get(key, default_value)
        if getattr(args, key, default_value) != current_value:
            setattr(args, key, current_value)
            settings_changed = True

    return settings_changed


def _restart_playback(frame_buffer, frame_generator):
    """Clear pending frames and restart generation thread."""
    _drain_queue_nowait(frame_buffer)
    return _start_frame_generator(frame_generator)


def _handle_web_command(command, frame_buffer, frame_generator, paused, start_time, frame_count, previous_reactive_frame):
    """Apply a single web command and return updated playback state."""
    stop_requested = False
    generator_thread = None

    if command == "pause":
        paused = True
        logging.info("Paused via web interface")
    elif command == "resume":
        paused = False
        start_time = time.time()
        frame_count = 0
        logging.info("Resumed via web interface")
    elif command == "restart":
        logging.info("Restarting via web interface...")
        generator_thread = _restart_playback(frame_buffer, frame_generator)
        start_time = time.time()
        frame_count = 0
        paused = False
        previous_reactive_frame = None
    elif command == "next":
        _drain_queue_nowait(frame_buffer)
        logging.info("Skipped to next image via web interface")
    elif command == "stop":
        logging.info("Stopping slideshow via web interface...")
        stop_requested = True

    return (
        paused,
        start_time,
        frame_count,
        previous_reactive_frame,
        stop_requested,
        generator_thread,
    )

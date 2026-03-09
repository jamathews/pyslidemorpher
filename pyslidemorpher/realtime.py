
"""
Real-time playback functionality for PySlide Morpher.
Handles live slideshow display using OpenCV.
"""

import json
import logging
import math
import platform
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

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


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

REACTIVE_BANDS = ["sub", "bass", "low_mid", "high_mid", "treble", "air"]
EFFECT_KEYS = ["pulse", "warp", "color", "glow", "strobe", "trails"]
DEFAULT_REACTIVE_CONTROLS = {
    "reactive_master_gain": 1.0,
    "pulse_enabled": True,
    "pulse_strength": 1.0,
    "pulse_band": "bass",
    "warp_enabled": True,
    "warp_strength": 1.0,
    "warp_band": "low_mid",
    "color_enabled": True,
    "color_strength": 1.0,
    "color_band": "high_mid",
    "glow_enabled": True,
    "glow_strength": 1.0,
    "glow_band": "treble",
    "strobe_enabled": True,
    "strobe_strength": 1.0,
    "strobe_band": "bass",
    "trails_enabled": True,
    "trails_strength": 1.0,
    "trails_band": "sub",
}
for _band in REACTIVE_BANDS:
    DEFAULT_REACTIVE_CONTROLS[f"eq_{_band}_gain"] = 1.0


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


def _normalize_series(values):
    """Normalize array to 0..1 with robust percentile scaling."""
    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 98))
    if high <= low:
        return np.clip(values, 0.0, 1.0)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _build_audio_envelope(audio_file, envelope_fps):
    """Build normalized overall + per-band envelopes using ffmpeg-decoded PCM."""
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
    normalized = _normalize_series(rms)

    spectrum = np.abs(np.fft.rfft(samples, axis=1))
    freqs = np.fft.rfftfreq(window, d=1.0 / sample_rate)
    band_ranges = {
        "sub": (20, 80),
        "bass": (80, 250),
        "low_mid": (250, 1000),
        "high_mid": (1000, 4000),
        "treble": (4000, 12000),
        "air": (12000, 20000),
    }

    band_envelopes = {}
    for band_name, (f_lo, f_hi) in band_ranges.items():
        mask = (freqs >= f_lo) & (freqs < f_hi)
        if not np.any(mask):
            band_energy = np.zeros((samples.shape[0],), dtype=np.float32)
        else:
            band_energy = np.mean(spectrum[:, mask], axis=1)
        band_envelopes[band_name] = _normalize_series(band_energy).astype(np.float32)

    duration = samples.shape[0] / envelope_fps
    return {
        "values": normalized.astype(np.float32),
        "bands": band_envelopes,
        "fps": float(envelope_fps),
        "duration": float(duration),
    }


def _parse_audio_device_spec(audio_device):
    """Parse audio device selector into sounddevice-compatible spec."""
    if not audio_device:
        return None
    device = str(audio_device).strip()
    if device in {"__default__", "default"}:
        return None
    if device.startswith("index:"):
        try:
            return int(device.split(":", 1)[1])
        except Exception:
            return None
    return device


class LiveAudioAnalyzer:
    """Capture live input audio and expose normalized overall + band features."""

    def __init__(self, device, envelope_fps=30, sample_rate=44100):
        self.device = _parse_audio_device_spec(device)
        self.sample_rate = int(sample_rate)
        self.envelope_fps = max(1, int(envelope_fps))
        self.block_size = max(256, int(self.sample_rate / self.envelope_fps))
        self.stream = None
        self._lock = threading.Lock()
        self._overall = 0.0
        self._bands = {band: 0.0 for band in REACTIVE_BANDS}

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass
        if indata is None or len(indata) == 0:
            return
        mono = np.asarray(indata[:, 0], dtype=np.float32)
        self._update_features(mono)

    def _update_features(self, mono):
        if mono.size <= 0:
            return
        rms = float(np.sqrt(np.mean(mono * mono)))
        rms_norm = float(np.clip(rms * 6.5, 0.0, 1.0))

        spec = np.abs(np.fft.rfft(mono))
        freqs = np.fft.rfftfreq(mono.size, d=1.0 / self.sample_rate)
        band_ranges = {
            "sub": (20, 80),
            "bass": (80, 250),
            "low_mid": (250, 1000),
            "high_mid": (1000, 4000),
            "treble": (4000, 12000),
            "air": (12000, 20000),
        }
        bands = {}
        for band_name, (f_lo, f_hi) in band_ranges.items():
            mask = (freqs >= f_lo) & (freqs < f_hi)
            if not np.any(mask):
                val = 0.0
            else:
                val = float(np.mean(spec[mask]))
            bands[band_name] = float(np.clip(val * 0.09, 0.0, 1.0))

        with self._lock:
            alpha = 0.25
            self._overall = ((1.0 - alpha) * self._overall) + (alpha * rms_norm)
            for band_name in REACTIVE_BANDS:
                self._bands[band_name] = ((1.0 - alpha) * self._bands[band_name]) + (alpha * bands[band_name])

    def start(self):
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice is not installed")
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def get_features(self):
        with self._lock:
            return {
                "overall": float(self._overall),
                "bands": {k: float(v) for k, v in self._bands.items()},
            }


class FfmpegLiveAudioAnalyzer(LiveAudioAnalyzer):
    """Capture live input audio via ffmpeg (macOS AVFoundation fallback)."""

    def __init__(self, device, envelope_fps=30, sample_rate=44100):
        super().__init__(device, envelope_fps=envelope_fps, sample_rate=sample_rate)
        self._reader_thread = None
        self._stop_event = threading.Event()
        self._process = None

    def _resolve_avfoundation_device(self):
        raw = str(self.device) if self.device is not None else "0"
        if raw in {"__default__", "default", "None"}:
            return "0"
        if raw.startswith("avf:"):
            return raw.split(":", 1)[1]
        if raw.startswith("index:"):
            return raw.split(":", 1)[1]
        if raw.isdigit():
            return raw
        return "0"

    def start(self):
        if platform.system() != "Darwin":
            raise RuntimeError("ffmpeg live analyzer fallback currently supports macOS only")

        audio_index = self._resolve_avfoundation_device()
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "avfoundation",
            "-i", f":{audio_index}",
            "-ac", "1",
            "-ar", str(self.sample_rate),
            "-f", "f32le",
            "-"
        ]
        self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._stop_event.clear()

        def _reader():
            block = self.block_size
            need_bytes = block * 4  # f32le mono
            while not self._stop_event.is_set() and self._process and self._process.stdout:
                chunk = self._process.stdout.read(need_bytes)
                if not chunk:
                    break
                mono = np.frombuffer(chunk, dtype=np.float32)
                if mono.size > 0:
                    self._update_features(mono)

        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()

    def stop(self):
        self._stop_event.set()
        if self._process is not None:
            try:
                self._process.terminate()
            except Exception:
                pass
            self._process = None
        self._reader_thread = None


def _current_audio_features(audio_envelope, audio_start_time):
    """Get current normalized overall + per-band levels for looped playback time."""
    if not audio_envelope or audio_start_time is None:
        return {"overall": 0.0, "bands": {band: 0.0 for band in REACTIVE_BANDS}}

    duration = audio_envelope["duration"]
    if duration <= 0:
        return {"overall": 0.0, "bands": {band: 0.0 for band in REACTIVE_BANDS}}

    elapsed = (time.time() - audio_start_time) % duration
    idx = int(elapsed * audio_envelope["fps"])
    idx = max(0, min(idx, len(audio_envelope["values"]) - 1))
    bands = {}
    for band in REACTIVE_BANDS:
        series = audio_envelope.get("bands", {}).get(band)
        bands[band] = float(series[idx]) if series is not None and len(series) > idx else 0.0
    return {"overall": float(audio_envelope["values"][idx]), "bands": bands}


def _resolve_reactive_controls(current_settings):
    """Build effective reactive controls from settings with defaults."""
    controls = DEFAULT_REACTIVE_CONTROLS.copy()
    if current_settings is None:
        return controls
    for key in list(controls.keys()):
        if hasattr(current_settings, key):
            controls[key] = getattr(current_settings, key)
    return controls


def _apply_audio_reactive_effect(frame_bgr, audio_features, elapsed_time, style="dramatic", previous_frame=None, controls=None):
    """Apply stackable audio-reactive effects using per-band routing controls."""
    controls = controls or DEFAULT_REACTIVE_CONTROLS
    overall = float(audio_features.get("overall", 0.0))
    bands = audio_features.get("bands", {})
    if overall <= 0:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    drive = float(np.clip(overall, 0.0, 1.0)) ** 1.35
    style = style if style in {"subtle", "dramatic", "extreme"} else "dramatic"

    if style == "subtle":
        strength = 0.55
    elif style == "extreme":
        strength = 2.2
    else:
        strength = 1.0

    master_gain = float(max(0.0, controls.get("reactive_master_gain", 1.0)))

    def effect_drive(effect_key, fallback_band):
        if not controls.get(f"{effect_key}_enabled", True):
            return 0.0
        band = str(controls.get(f"{effect_key}_band", fallback_band))
        band_level = float(bands.get(band, overall))
        eq = float(max(0.0, controls.get(f"eq_{band}_gain", 1.0)))
        eff_gain = float(max(0.0, controls.get(f"{effect_key}_strength", 1.0)))
        return float(np.clip((band_level * eq * eff_gain * master_gain), 0.0, 2.5))

    pulse_drive = effect_drive("pulse", "bass")
    warp_drive = effect_drive("warp", "low_mid")
    color_drive = effect_drive("color", "high_mid")
    glow_drive = effect_drive("glow", "treble")
    strobe_drive = effect_drive("strobe", "bass")
    trails_drive = effect_drive("trails", "sub")

    # Strong pulse zoom and rotation.
    zoom = 1.0 + (0.24 * strength) * pulse_drive + (0.08 * strength) * math.sin(elapsed_time * 2.8)
    if pulse_drive > 0 and zoom > 1.001:
        crop_w = max(2, int(w / zoom))
        crop_h = max(2, int(h / zoom))
        x0 = (w - crop_w) // 2
        y0 = (h - crop_h) // 2
        pulsed = cv2.resize(frame_bgr[y0:y0 + crop_h, x0:x0 + crop_w], (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        pulsed = frame_bgr

    angle = (18.0 * strength * pulse_drive) * math.sin(elapsed_time * 4.5)
    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    pulsed = cv2.warpAffine(pulsed, rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Color drift and channel split.
    hsv = cv2.cvtColor(pulsed, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_shift = (42.0 * strength) * color_drive + (22.0 * strength) * math.sin(elapsed_time * 1.5)
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.10 + (0.9 * strength) * color_drive), 0.0, 255.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + (0.32 * strength) * color_drive), 0.0, 255.0)
    shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    split_px = max(1, int(2 + (32 * strength * color_drive)))
    b, g, r = cv2.split(shifted)
    b = np.roll(b, -split_px, axis=1)
    r = np.roll(r, split_px, axis=0)
    out = cv2.merge((b, g, r)) if color_drive > 0 else shifted

    # Glow.
    sigma = 1.0 + (18.0 * strength * glow_drive)
    bloom = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = cv2.addWeighted(out, 1.0, bloom, (0.15 + 0.95 * glow_drive), 0.0)

    # Strobe with more punch in extreme mode.
    strobe_threshold = 0.80 if style == "subtle" else 0.64 if style == "dramatic" else 0.40
    flash = max(0.0, strobe_drive - strobe_threshold) * (2.2 + 1.3 * strength)
    if flash > 0:
        out = cv2.convertScaleAbs(out, alpha=1.0 + flash, beta=85.0 * flash)

    # Pulsing vignette to focus energy into the center and add tunnel-like depth.
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt(((xx - cx) / max(1, cx)) ** 2 + ((yy - cy) / max(1, cy)) ** 2)
    vignette = np.clip(1.22 - (0.45 + (0.45 * strength) * max(pulse_drive, warp_drive)) * dist, 0.35, 1.65)
    out = np.clip(out.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)

    # Radial warp.
    amp = (0.010 + 0.065 * strength) * warp_drive
    if amp > 0.0005:
        y, x = np.indices((h, w), dtype=np.float32)
        dx = x - cx
        dy = y - cy
        radius = np.sqrt(dx * dx + dy * dy) + 1e-6
        wave = np.sin((radius / max(1.0, min(w, h) * 0.12)) - (elapsed_time * 10.0))
        displacement = amp * min(w, h) * wave
        map_x = (x + (dx / radius) * displacement).astype(np.float32)
        map_y = (y + (dy / radius) * displacement).astype(np.float32)
        out = cv2.remap(out, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Neon edges piggyback on glow drive.
    if glow_drive > 0:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 160)
        edge_color = cv2.applyColorMap(edges, cv2.COLORMAP_TURBO)
        out = cv2.addWeighted(out, 1.0, edge_color, 0.06 + (0.52 * glow_drive * strength), 0.0)

    # Temporal trails.
    if trails_drive > 0 and previous_frame is not None and previous_frame.shape == out.shape:
        trail_mix = np.clip(0.08 + 0.46 * trails_drive * strength, 0.0, 0.7)
        out = cv2.addWeighted(out, 1.0 - trail_mix, previous_frame, trail_mix, 0.0)

    return out


def play_realtime(imgs, args):
    """Play slideshow in realtime using OpenCV display with optimized performance."""
    W, H = args.size
    frame_time = 1.0 / args.fps  # Time per frame in seconds

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
                live_audio_analyzer = LiveAudioAnalyzer(current_audio_device, envelope_fps=max(args.fps, 24))
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
            # Create a namespace object similar to args for compatibility
            class SettingsNamespace:
                def __init__(self, settings_dict, original_args):
                    # Copy all original args first
                    for key, value in vars(original_args).items():
                        setattr(self, key, value)
                    # Override with web controller settings (all keys for dynamic controls).
                    for key, value in settings_dict.items():
                        setattr(self, key, value)
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
            live_audio_analyzer = LiveAudioAnalyzer(desired, envelope_fps=max(args.fps, 24))
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
                        # Restart by creating new generator thread
                        generator_thread = threading.Thread(target=frame_generator, daemon=True)
                        generator_thread.start()
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
                previous_reactive_frame = None

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
                            previous_reactive_frame = None
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

                for key in ['window_width', 'window_height', 'window_x', 'window_y']:
                    if getattr(args, key, None) != current_settings.get(key):
                        setattr(args, key, current_settings.get(key))
                        settings_changed = True

                if getattr(args, 'reactive_style', 'dramatic') != current_settings.get('reactive_style', 'dramatic'):
                    args.reactive_style = current_settings.get('reactive_style', 'dramatic')
                    settings_changed = True

                if getattr(args, 'reactive', False) != bool(current_settings.get('reactive_enabled', getattr(args, 'reactive', False))):
                    args.reactive = bool(current_settings.get('reactive_enabled', args.reactive))
                    settings_changed = True

                if getattr(args, 'audio_device', '') != current_settings.get('audio_device', ''):
                    args.audio_device = current_settings.get('audio_device', '')
                    settings_changed = True

                if getattr(args, 'audio', '') != current_settings.get('audio', ''):
                    args.audio = current_settings.get('audio', '')
                    settings_changed = True

                for key, default_value in DEFAULT_REACTIVE_CONTROLS.items():
                    current_value = current_settings.get(key, default_value)
                    if getattr(args, key, default_value) != current_value:
                        setattr(args, key, current_value)
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

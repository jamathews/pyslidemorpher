"""Audio feature analysis for realtime reactive playback."""

import logging
import platform
import subprocess
import threading
import time

import numpy as np

from .realtime_shared import REACTIVE_BANDS


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
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError:
        logging.warning("Reactive mode requested, but ffmpeg is not installed.")
        return None
    except subprocess.CalledProcessError as err:
        stderr = err.stderr.decode("utf-8", errors="ignore").strip()
        logging.warning("Could not analyze audio for reactive mode: %s", stderr)
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

    def __init__(self, device, sounddevice_module, envelope_fps=30, sample_rate=44100):
        self._sd = sounddevice_module
        self.device = _parse_audio_device_spec(device)
        self.sample_rate = int(sample_rate)
        self.envelope_fps = max(1, int(envelope_fps))
        self.block_size = max(256, int(self.sample_rate / self.envelope_fps))
        self.stream = None
        self._lock = threading.Lock()
        self._overall = 0.0
        self._bands = {band: 0.0 for band in REACTIVE_BANDS}

    def _callback(self, indata, frames, time_info, status):
        del frames, time_info
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
        self.stream = self._sd.InputStream(
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
        super().__init__(device, sounddevice_module=None, envelope_fps=envelope_fps, sample_rate=sample_rate)
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
            "-",
        ]
        self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._stop_event.clear()

        def _reader():
            block = self.block_size
            need_bytes = block * 4
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

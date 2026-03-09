"""Audio-reactive visual effects for realtime playback."""

import math

import cv2
import numpy as np

from .realtime_shared import DEFAULT_REACTIVE_CONTROLS


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

    sigma = 1.0 + (18.0 * strength * glow_drive)
    bloom = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = cv2.addWeighted(out, 1.0, bloom, (0.15 + 0.95 * glow_drive), 0.0)

    strobe_threshold = 0.80 if style == "subtle" else 0.64 if style == "dramatic" else 0.40
    flash = max(0.0, strobe_drive - strobe_threshold) * (2.2 + 1.3 * strength)
    if flash > 0:
        out = cv2.convertScaleAbs(out, alpha=1.0 + flash, beta=85.0 * flash)

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt(((xx - cx) / max(1, cx)) ** 2 + ((yy - cy) / max(1, cy)) ** 2)
    vignette = np.clip(1.22 - (0.45 + (0.45 * strength) * max(pulse_drive, warp_drive)) * dist, 0.35, 1.65)
    out = np.clip(out.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)

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

    if glow_drive > 0:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 160)
        edge_color = cv2.applyColorMap(edges, cv2.COLORMAP_TURBO)
        out = cv2.addWeighted(out, 1.0, edge_color, 0.06 + (0.52 * glow_drive * strength), 0.0)

    if trails_drive > 0 and previous_frame is not None and previous_frame.shape == out.shape:
        trail_mix = np.clip(0.08 + 0.46 * trails_drive * strength, 0.0, 0.7)
        out = cv2.addWeighted(out, 1.0 - trail_mix, previous_frame, trail_mix, 0.0)

    return out

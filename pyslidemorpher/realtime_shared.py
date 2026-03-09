"""Shared constants for realtime playback."""

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

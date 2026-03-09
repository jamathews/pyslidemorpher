"""
Web-based GUI for controlling PySlidemorpher realtime settings.
Provides a browser interface to tweak slideshow parameters in real-time.
"""

import json
import logging
import platform
import re
import subprocess
import threading
import time
from pathlib import Path
from queue import Queue
from uuid import uuid4

try:
    from flask import Flask, render_template, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

SETTINGS_FILE_NAME = "pyslidemorpher_web_settings.json"
UPLOADS_DIR_NAME = ".pyslidemorpher_uploads"


def get_settings_file_path():
    """Get the settings file path in the current working directory."""
    return Path.cwd() / SETTINGS_FILE_NAME


def get_uploads_dir():
    """Get uploads directory path, creating it if missing."""
    uploads = Path.cwd() / UPLOADS_DIR_NAME
    uploads.mkdir(parents=True, exist_ok=True)
    return uploads


def load_persisted_settings():
    """Load persisted settings from disk if they exist and are valid."""
    settings_file = get_settings_file_path()
    if not settings_file.exists():
        return {}
    try:
        with settings_file.open('r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logging.warning(f"Failed to load settings file {settings_file}: {e}")
        return {}


def list_audio_input_devices():
    """Return audio input devices for web GUI selection."""
    devices = [{"id": "__file__", "name": "Audio File Track"}]
    if not SOUNDDEVICE_AVAILABLE:
        if platform.system() == "Darwin":
            devices.append({"id": "__default__", "name": "System Default Input"})
            try:
                cmd = [
                    "ffmpeg",
                    "-hide_banner",
                    "-f", "avfoundation",
                    "-list_devices", "true",
                    "-i", ""
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                text = (proc.stderr or "") + "\n" + (proc.stdout or "")
                in_audio = False
                for line in text.splitlines():
                    if "AVFoundation audio devices" in line:
                        in_audio = True
                        continue
                    if "AVFoundation video devices" in line:
                        in_audio = False
                    if not in_audio:
                        continue
                    m = re.search(r"\[(\d+)\]\s+(.+)$", line.strip())
                    if m:
                        idx = m.group(1)
                        name = m.group(2).strip()
                        devices.append({"id": f"avf:{idx}", "name": name})
            except Exception as e:
                logging.warning(f"Could not enumerate AVFoundation devices via ffmpeg: {e}")
        else:
            devices.append({"id": "__default__", "name": "Default Input (sounddevice unavailable)"})
        return devices
    devices.append({"id": "__default__", "name": "System Default Input"})
    try:
        queried = sd.query_devices()
        for idx, dev in enumerate(queried):
            try:
                max_in = int(dev.get('max_input_channels', 0))
            except Exception:
                max_in = 0
            if max_in > 0:
                name = str(dev.get('name', f'Input {idx}'))
                devices.append({"id": f"index:{idx}", "name": name})
    except Exception as e:
        logging.warning(f"Could not enumerate audio devices: {e}")
    return devices


class RealtimeController:
    """Controller class to manage realtime slideshow settings."""

    def __init__(self):
        self.settings = {
            'fps': 30,
            'seconds_per_transition': 2.0,
            'hold': 0.5,
            'pixel_size': 4,
            'transition': 'default',
            'easing': 'smoothstep',
            'audio': '',
            'reactive_enabled': False,
            'reactive_style': 'dramatic',
            'audio_device': '__file__',
            'reactive_master_gain': 1.0,
            'pulse_enabled': True,
            'pulse_strength': 1.0,
            'pulse_band': 'bass',
            'warp_enabled': True,
            'warp_strength': 1.0,
            'warp_band': 'low_mid',
            'color_enabled': True,
            'color_strength': 1.0,
            'color_band': 'high_mid',
            'glow_enabled': True,
            'glow_strength': 1.0,
            'glow_band': 'treble',
            'strobe_enabled': True,
            'strobe_strength': 1.0,
            'strobe_band': 'bass',
            'trails_enabled': True,
            'trails_strength': 1.0,
            'trails_band': 'sub',
            'eq_sub_gain': 1.0,
            'eq_bass_gain': 1.0,
            'eq_low_mid_gain': 1.0,
            'eq_high_mid_gain': 1.0,
            'eq_treble_gain': 1.0,
            'eq_air_gain': 1.0,
            'window_width': 1280,
            'window_height': 720,
            'window_x': 80,
            'window_y': 80,
            'paused': False
        }
        persisted = load_persisted_settings()
        if persisted:
            self.settings.update({k: v for k, v in persisted.items() if k in self.settings})
        self.settings_lock = threading.Lock()
        self.command_queue = Queue()

    def save_settings(self):
        """Persist settings to disk."""
        settings_file = get_settings_file_path()
        try:
            with settings_file.open('w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to persist settings to {settings_file}: {e}")

    def get_settings(self):
        """Get current settings thread-safely."""
        with self.settings_lock:
            return self.settings.copy()

    def update_setting(self, key, value):
        """Update a single setting thread-safely."""
        with self.settings_lock:
            if key in self.settings:
                # Type conversion based on expected types
                if key == 'fps':
                    value = int(value)
                elif key in ['seconds_per_transition', 'hold']:
                    value = float(value)
                elif key == 'pixel_size':
                    value = int(value)
                elif key in ['paused', 'reactive_enabled']:
                    value = bool(value)
                elif key == 'reactive_style':
                    value = str(value)
                    if value not in ['subtle', 'dramatic', 'extreme']:
                        value = 'dramatic'
                elif key == 'audio':
                    value = str(value or '')
                elif key == 'audio_device':
                    value = str(value)
                elif key.endswith('_enabled'):
                    if isinstance(value, bool):
                        pass
                    elif isinstance(value, str):
                        value = value.lower() in ['1', 'true', 'on', 'yes']
                    else:
                        value = bool(value)
                elif key.endswith('_strength') or key.startswith('eq_') or key == 'reactive_master_gain':
                    value = float(value)
                elif key.endswith('_band'):
                    value = str(value)
                    if value not in ['sub', 'bass', 'low_mid', 'high_mid', 'treble', 'air']:
                        value = 'bass'
                elif key in ['window_width', 'window_height', 'window_x', 'window_y']:
                    value = int(value)
                    if key == 'window_width':
                        value = max(320, min(7680, value))
                    elif key == 'window_height':
                        value = max(240, min(4320, value))
                    elif key in ['window_x', 'window_y']:
                        value = max(-4000, min(4000, value))

                self.settings[key] = value
                self.save_settings()
                logging.info(f"Updated setting {key} to {value}")
                return True
        return False

    def send_command(self, command):
        """Send a command to the slideshow (pause, resume, restart, etc.)."""
        self.command_queue.put(command)
        logging.info(f"Command sent: {command}")


# Global controller instance
controller = RealtimeController()


def create_web_app():
    """Create and configure the Flask web application."""
    if not FLASK_AVAILABLE:
        return None

    # Create Flask app with default settings first
    app = Flask(__name__)

    # Set template folder after app creation
    current_dir = Path(__file__).resolve().parent
    template_dir = current_dir / 'templates'
    static_dir = current_dir / 'static'

    # Only set template folder if it exists
    if template_dir.exists():
        app.template_folder = str(template_dir.resolve())
        logging.info(f"Template folder set to: {app.template_folder}")

    # Only set static folder if it exists
    if static_dir.exists():
        app.static_folder = str(static_dir.resolve())
        logging.info(f"Static folder set to: {app.static_folder}")

    @app.route('/')
    def index():
        """Main control panel page."""
        try:
            return render_template('control_panel.html')
        except Exception as e:
            # If template rendering fails, return error info
            return f"Template rendering failed: {str(e)}<br>Template folder: {app.template_folder}<br>Template exists: {Path(app.template_folder, 'control_panel.html').exists()}", 500

    @app.route('/test')
    def test():
        """Simple test route."""
        return "Flask server is working!"

    @app.route('/api/settings', methods=['GET'])
    def get_settings():
        """API endpoint to get current settings."""
        return jsonify(controller.get_settings())

    @app.route('/api/audio-devices', methods=['GET'])
    def get_audio_devices():
        """API endpoint to list available audio input devices."""
        return jsonify({'devices': list_audio_input_devices()})

    @app.route('/api/settings', methods=['POST'])
    def update_settings():
        """API endpoint to update settings."""
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        updated = {}
        for key, value in data.items():
            if controller.update_setting(key, value):
                updated[key] = value

        return jsonify({'updated': updated})

    @app.route('/api/audio-file', methods=['POST'])
    def upload_audio_file():
        """Upload an audio file from the web UI and activate file-track source."""
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file_obj = request.files['file']
        if not file_obj or not file_obj.filename:
            return jsonify({'error': 'No file selected'}), 400

        safe_name = Path(file_obj.filename).name
        ext = Path(safe_name).suffix.lower()
        allowed = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg'}
        if ext not in allowed:
            return jsonify({'error': f'Unsupported audio format: {ext}'}), 400

        dest = get_uploads_dir() / f"{uuid4().hex}_{safe_name}"
        try:
            file_obj.save(str(dest))
        except Exception as e:
            return jsonify({'error': f'Failed to save file: {e}'}), 500

        controller.update_setting('audio', str(dest))
        controller.update_setting('audio_device', '__file__')
        return jsonify({'status': 'ok', 'audio': str(dest), 'filename': safe_name})

    @app.route('/api/command', methods=['POST'])
    def send_command():
        """API endpoint to send commands."""
        data = request.get_json()
        if not data or 'command' not in data:
            return jsonify({'error': 'No command provided'}), 400

        command = data['command']
        if command in ['pause', 'resume', 'restart', 'next', 'previous', 'stop']:
            controller.send_command(command)
            return jsonify({'status': 'success', 'command': command})
        else:
            return jsonify({'error': 'Invalid command'}), 400

    return app


def start_web_server(host='localhost', port=5001):
    """Start the web server in a separate thread."""
    if not FLASK_AVAILABLE:
        logging.error("Cannot start web server: Flask is not available")
        return None

    app = create_web_app()
    if app is None:
        return None

    def run_server():
        app.run(host=host, port=port, debug=True, use_reloader=False, threaded=True)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    logging.info(f"Web GUI started at http://{host}:{port}")
    return server_thread


def get_controller():
    """Get the global controller instance."""
    return controller

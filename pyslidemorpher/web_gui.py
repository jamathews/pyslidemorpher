"""
Web-based GUI for controlling PySlidemorpher realtime settings.
Provides a browser interface to tweak slideshow parameters in real-time.
"""

import json
import logging
import threading
import time
from pathlib import Path
from queue import Queue

try:
    from flask import Flask, render_template, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

SETTINGS_FILE_NAME = "pyslidemorpher_web_settings.json"


def get_settings_file_path():
    """Get the settings file path in the current working directory."""
    return Path.cwd() / SETTINGS_FILE_NAME


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
            'reactive_style': 'dramatic',
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
                elif key in ['paused']:
                    value = bool(value)
                elif key == 'reactive_style':
                    value = str(value)
                    if value not in ['subtle', 'dramatic', 'extreme']:
                        value = 'dramatic'
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

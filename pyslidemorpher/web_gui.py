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
            'audio_threshold': 0.1,
            'reactive': False,
            'paused': False,
            # Enhanced audio reactivity settings
            'tempo_detection': True,
            'tempo_to_timing': True,
            'intensity_to_speed': True,
            'intensity_to_pixel_size': True,
            'frequency_to_easing': True,
            'brightness_modulation': True,
            'beat_sensitivity': 0.3,
            'peak_sensitivity': 0.2,
            'intensity_sensitivity': 0.1,
            'speed_modulation_range': 2.0,
            'pixel_size_modulation_range': 0.5,
            'brightness_modulation_range': 0.1,
            'low_freq_threshold': 0.4,
            'high_freq_threshold': 0.3,
            'tempo_smoothing': 0.8,
            'show_audio_debug': False
        }
        self.settings_lock = threading.Lock()
        self.command_queue = Queue()

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
                elif key in ['seconds_per_transition', 'hold', 'audio_threshold', 'beat_sensitivity', 
                           'peak_sensitivity', 'intensity_sensitivity', 'speed_modulation_range',
                           'pixel_size_modulation_range', 'brightness_modulation_range', 
                           'low_freq_threshold', 'high_freq_threshold', 'tempo_smoothing']:
                    value = float(value)
                elif key == 'pixel_size':
                    value = int(value)
                elif key in ['reactive', 'paused', 'tempo_detection', 'tempo_to_timing', 
                           'intensity_to_speed', 'intensity_to_pixel_size', 'frequency_to_easing',
                           'brightness_modulation', 'show_audio_debug']:
                    value = bool(value)

                self.settings[key] = value
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

    @app.route('/api/audio-debug', methods=['GET'])
    def get_audio_debug():
        """API endpoint to get real-time audio analysis data."""
        # This will be populated by the realtime module when available
        if hasattr(controller, 'audio_features'):
            return jsonify(controller.audio_features)
        else:
            return jsonify({
                'intensity': 0.0, 'peak': 0.0, 'spectral_centroid': 0.0, 'beat_strength': 0.0,
                'low_freq_energy': 0.0, 'mid_freq_energy': 0.0, 'high_freq_energy': 0.0,
                'spectral_rolloff': 0.0, 'zero_crossing_rate': 0.0, 'onset_strength': 0.0,
                'estimated_tempo': 0.0
            })

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

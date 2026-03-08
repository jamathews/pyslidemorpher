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
            'reactive_style': 'dramatic',
            'paused': False
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

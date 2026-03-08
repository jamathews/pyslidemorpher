# PySlidemorpher Web GUI

This document describes the new web-based GUI feature for PySlidemorpher that allows real-time control of slideshow settings through a browser interface.

## Overview

The web GUI provides a modern, intuitive interface for controlling PySlidemorpher slideshows in real-time. Instead of being limited to keyboard shortcuts, users can now adjust settings like FPS, transition types, timing, and visual effects through a responsive web interface.

## Features

### Real-time Settings Control
- **Performance Settings**: Adjust FPS and pixel size on the fly
- **Timing Controls**: Modify transition duration and hold times
- **Visual Effects**: Change transition types and easing functions
- **Audio Settings**: Control reactive mode and audio threshold

### Playback Controls
- Pause/Resume slideshow
- Restart slideshow
- Skip to next/previous image
- Real-time status updates

### Modern Interface
- Responsive design that works on desktop and mobile
- Beautiful gradient background with glassmorphism effects
- Real-time value displays for all sliders
- Status bar showing connection and slideshow state

## Installation

### Requirements
The web GUI requires Flask to be installed:

```bash
# Using pip
pip install flask

# Using conda
conda install flask

# Or add to your Pipfile
flask = "*"
```

### Verification
You can verify the installation by running the test script:

```bash
python test_basic_functionality.py
```

## Usage

### Basic Usage
To enable the web GUI, add the `--web-gui` flag to your realtime slideshow command:

```bash
python pyslidemorpher.py demo_images --realtime --web-gui
```

This will:
1. Start the slideshow in realtime mode
2. Launch a web server on `http://localhost:5001`
3. Display a message with the web interface URL

### Advanced Usage Examples

#### High-performance slideshow with web control:
```bash
python pyslidemorpher.py demo_images --realtime --web-gui --fps 60 --pixel-size 2
```

#### Audio-reactive slideshow with web control:
```bash
python pyslidemorpher.py demo_images --realtime --web-gui --audio music.mp3 --reactive
```

#### Custom resolution with web control:
```bash
python pyslidemorpher.py demo_images --realtime --web-gui --size 1920x1080
```

### Web Interface Controls

Once the slideshow is running, open `http://localhost:5001` in your browser to access the control panel:

#### Performance Section
- **FPS Slider**: Adjust frames per second (10-120 FPS)
- **Pixel Size Slider**: Control processing detail level (1-20 pixels)

#### Timing Section
- **Seconds Per Transition**: How long each transition takes (0.5-10 seconds)
- **Hold Time**: How long to display each image (0-5 seconds)

#### Visual Effects Section
- **Transition Type**: Choose from various transition effects:
  - Default: Standard pixel morphing
  - Swarm: Particle swarm effect
  - Tornado: Spiral tornado effect
  - Swirl: Swirling motion
  - Drip: Dripping paint effect
  - Rain: Rainfall effect
  - Sorted: Color-sorted transitions
  - Hue Sorted: Hue-based sorting
  - Random: Randomly selected transitions

- **Easing Function**: Control transition smoothness:
  - Linear: Constant speed
  - Smooth Step: Smooth acceleration/deceleration
  - Cubic: Cubic curve easing

#### Audio Section (when audio is provided)
- **Reactive Mode**: Toggle audio-reactive transitions
- **Reactive Style**: Switch between `subtle`, `dramatic`, and `extreme` live
- **Audio Threshold**: Sensitivity for audio triggers (0.0-1.0)

#### Playback Controls
- **Pause/Resume**: Control slideshow playback
- **Restart**: Restart the slideshow from the beginning
- **Next/Previous**: Skip between images

## Technical Details

### Architecture
The web GUI consists of several components:

1. **RealtimeController**: Thread-safe controller for managing settings and commands
2. **Flask Web Server**: Serves the web interface and API endpoints
3. **Integration Layer**: Connects the web controller to the realtime slideshow

### API Endpoints
- `GET /`: Main control panel interface
- `GET /api/settings`: Retrieve current settings
- `POST /api/settings`: Update settings
- `POST /api/command`: Send playback commands

### Thread Safety
All setting updates and command handling are thread-safe, ensuring smooth operation even with multiple concurrent web interface users.

## Troubleshooting

### Flask Not Available
If you see the error "Flask not available - web GUI will be disabled":
```bash
pip install flask
```

### Web Server Won't Start
If the web server fails to start:
- Check if port 5001 is already in use
- Ensure Flask is properly installed
- Check the console for error messages

### Settings Not Updating
If settings changes don't take effect:
- Refresh the web page
- Check the browser console for JavaScript errors
- Verify the slideshow is running in realtime mode

### Connection Issues
If the web interface shows "Disconnected":
- Ensure the slideshow is still running
- Check if the web server is still active
- Try refreshing the page

## Examples and Testing

### Demo Script
Run the updated demo script to see web GUI examples:
```bash
python demo_realtime.py
```

### Test Scripts
Two test scripts are provided:

1. **Basic Functionality Test**:
   ```bash
   python test_basic_functionality.py
   ```

2. **Full Web GUI Test**:
   ```bash
   python test_web_gui.py
   ```

## Browser Compatibility

The web interface is compatible with modern browsers:
- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

Mobile browsers are also supported for touch-friendly control.

## Security Notes

- The web server runs on localhost by default for security
- No authentication is required for local access
- For remote access, consider using SSH tunneling or VPN

## Future Enhancements

Potential future improvements:
- Custom port configuration
- Remote access authentication
- Preset saving/loading
- Real-time performance metrics
- Multi-user support

## Files Added/Modified

### New Files
- `pyslidemorpher/web_gui.py`: Web GUI controller and Flask app
- `pyslidemorpher/templates/control_panel.html`: Web interface template
- `test_basic_functionality.py`: Basic functionality tests
- `test_web_gui.py`: Full web GUI test suite
- `WEB_GUI_README.md`: This documentation

### Modified Files
- `pyslidemorpher/realtime.py`: Added web GUI integration
- `pyslidemorpher/cli.py`: Added --web-gui command line argument
- `demo_realtime.py`: Added web GUI usage examples

## Support

For issues or questions about the web GUI feature:
1. Check this documentation
2. Run the test scripts to verify installation
3. Check the console output for error messages
4. Ensure all dependencies are properly installed

# PySlidemorpher

A powerful Python application that creates stunning morphing videos from a folder of images using particle-based transitions. PySlidemorpher transforms static photos into dynamic, visually captivating slideshows with smooth morphing effects between images.

## What is PySlidemorpher?

PySlidemorpher is a pixel-morph slideshow generator that treats image pixels as particles, creating fluid transitions between photos. Unlike traditional crossfade transitions, PySlidemorpher uses advanced particle animation algorithms to morph one image into another, creating mesmerizing visual effects.

## Key Features

### 🎬 Multiple Transition Types
- **Default**: Classic particle-based morphing with shuffled pixel positions
- **Swarm**: Particles move in coordinated swarm-like patterns
- **Tornado**: Spiral tornado-like particle movements
- **Swirl**: Elegant swirling particle animations
- **Drip**: Gravity-based dripping particle effects
- **Rainfall**: Rain-like particle cascades
- **Sorted**: Color-sorted particle transitions
- **Hue-Sorted**: Hue-based color sorting transitions
- **Random**: Randomly selects different transitions for each image pair

### 🎵 Audio Integration
- **Audio Synchronization**: Add background music to generated videos
- **Audio-Reactive Mode**: Transitions triggered by audio intensity in real-time
- **Configurable Thresholds**: Adjust sensitivity for audio-reactive transitions

### ⚡ Performance Options
- **CPU Rendering**: NumPy-based rendering for broad compatibility
- **GPU Acceleration**: Optional PyTorch GPU acceleration for faster processing
- **Configurable Quality**: Adjustable pixel size, FPS, and video quality settings

### 🎮 Interactive Modes
- **Real-time Playback**: Live slideshow with interactive controls
- **Web GUI**: Browser-based control panel for real-time parameter adjustment
- **Command-line Interface**: Full-featured CLI for batch processing

## How It Works

PySlidemorpher uses a particle-based rendering system:

1. **Image Processing**: Each image is loaded and resized to fit the target resolution
2. **Particle Generation**: Image pixels are treated as individual particles with position and color data
3. **Transition Calculation**: Particles are animated from their positions in image A to their target positions in image B
4. **Frame Rendering**: Each frame of the transition is rendered by drawing particles at their interpolated positions
5. **Easing Functions**: Smooth animation curves (linear, smoothstep, cubic) control particle movement timing
6. **Video Output**: Frames are compiled into MP4 video with optional audio synchronization

## Installation

### Basic Installation
```bash
pip install imageio[ffmpeg] opencv-python
```

### Optional Dependencies
```bash
# For PyTorch GPU acceleration
pip install torch

# For real-time mode with audio
pip install pygame

# For web GUI
pip install flask
```

### System Requirements
- **FFmpeg**: Required for video output and audio processing
- **Python 3.7+**: Compatible with modern Python versions

## Usage Examples

### Basic Video Generation
```bash
python pyslidemorpher.py /path/to/photos --out slideshow.mp4
```

### Advanced Video with Custom Settings
```bash
python pyslidemorpher.py /path/to/photos \
    --out morphing_slideshow.mp4 \
    --fps 60 \
    --seconds-per-transition 3.0 \
    --pixel-size 2 \
    --size 1920x1080 \
    --transition swarm \
    --easing smoothstep \
    --audio background_music.mp3
```

### Real-time Interactive Mode
```bash
python pyslidemorpher.py /path/to/photos --realtime
```

### Audio-Reactive Real-time Mode
```bash
python pyslidemorpher.py /path/to/photos \
    --realtime \
    --reactive \
    --audio music.mp3 \
    --reactive-style extreme
```

### Web GUI Control
```bash
python pyslidemorpher.py /path/to/photos --realtime --web-gui
```
Then open http://localhost:5001 in your browser for real-time control.

### GPU Acceleration
```bash
python pyslidemorpher.py /path/to/photos \
    --out slideshow.mp4 \
    --use-pytorch
```

## Command-line Options

| Option | Description | Default |
|--------|-------------|---------|
| `photos_folder` | Folder containing images | Required |
| `--out` | Output video filename | Auto-generated |
| `--fps` | Frames per second | 30 |
| `--seconds-per-transition` | Duration of each transition | 2.0 |
| `--hold` | Hold time on each image | 0.5 |
| `--pixel-size` | Particle size (lower = higher detail) | 4 |
| `--size` | Output resolution (WxH) | 1920x1080 |
| `--transition` | Transition type | default |
| `--easing` | Animation easing function | smoothstep |
| `--audio` | Audio file to include | None |
| `--realtime` | Real-time playback mode | False |
| `--reactive` | Audio-reactive transitions | False |
| `--reactive-style` | Reactive visual style (`subtle`, `dramatic`, `extreme`) | dramatic |
| `--audio-threshold` | Audio sensitivity (0.0-1.0) | 0.1 |
| `--use-pytorch` | Enable GPU acceleration | False |
| `--web-gui` | Enable web control interface | False |
| `--seed` | Random seed for reproducible results | Random |

## Technical Details

### Supported Image Formats
- JPEG, PNG, BMP, TIFF, and other OpenCV-supported formats
- Automatic letterboxing to maintain aspect ratios
- Color space conversion and normalization

### Video Output
- **Codec**: H.264 (libx264)
- **Format**: MP4 container
- **Quality**: Configurable CRF (Constant Rate Factor)
- **Audio**: AAC encoding for audio tracks

### Performance Considerations
- **Pixel Size**: Lower values create higher detail but require more processing
- **Resolution**: Higher resolutions increase rendering time exponentially
- **GPU Acceleration**: Significant speedup for large images and complex transitions
- **Memory Usage**: Scales with image resolution and number of particles

## Project Structure

```
pyslidemorpher/
├── pyslidemorpher.py          # Main entry point
├── pyslidemorpher/
│   ├── cli.py                 # Command-line interface
│   ├── realtime.py            # Real-time playback engine
│   ├── transitions.py         # Transition algorithms
│   ├── rendering.py           # Core rendering functions
│   ├── web_gui.py             # Web-based control interface
│   ├── utils.py               # Utility functions
│   └── config.py              # Configuration settings
├── assets/                    # Demo images and audio
└── tests/                     # Test suite
```

## Contributing

PySlidemorpher is designed with modularity in mind. New transition types can be added by implementing functions in `transitions.py` that follow the established pattern of particle animation between two images.

## License

See LICENSE file for details.

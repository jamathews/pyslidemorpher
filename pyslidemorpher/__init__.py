"""
PySlide Morpher - Pixel-based slideshow video generator.

This package provides functionality for creating smooth transitions between images
using particle-based animations with various effects like swarm, tornado, swirl, etc.
"""

# Import main CLI function
from .cli import main

# Import all transition functions for programmatic use
from .transitions import (
    make_transition_frames,
    make_swarm_transition_frames,
    make_tornado_transition_frames,
    make_swirl_transition_frames,
    make_drip_transition_frames,
    make_rainfall_transition_frames,
    make_sorted_transition_frames,
    make_hue_sorted_transition_frames,
)

# Import utility functions
from .utils import (
    parse_size,
    list_images,
    fit_letterbox,
    downsample_for_particles,
    easing_fn,
)

# Import rendering functions
from .rendering import (
    prepare_transition,
    render_frame,
    render_frame_torch,
    render_frame_optimized,
    create_smooth_blend_frame,
)

# Import realtime functionality
from .realtime import (
    play_realtime,
    get_random_transition_function,
)

# Import configuration
from .config import (
    PYTORCH_AVAILABLE,
    DEVICE,
    USE_PYTORCH,
)

# Package metadata
__version__ = "1.0.0"
__author__ = "PySlide Morpher Contributors"
__description__ = "Pixel-based slideshow video generator with smooth transitions"

# Define what gets imported with "from pyslidemorpher import *"
__all__ = [
    # Main entry point
    'main',
    
    # Transition functions
    'make_transition_frames',
    'make_swarm_transition_frames',
    'make_tornado_transition_frames',
    'make_swirl_transition_frames',
    'make_drip_transition_frames',
    'make_rainfall_transition_frames',
    'make_sorted_transition_frames',
    'make_hue_sorted_transition_frames',
    
    # Utility functions
    'parse_size',
    'list_images',
    'fit_letterbox',
    'downsample_for_particles',
    'easing_fn',
    
    # Rendering functions
    'prepare_transition',
    'render_frame',
    'render_frame_torch',
    'render_frame_optimized',
    'create_smooth_blend_frame',
    
    # Realtime functionality
    'play_realtime',
    'get_random_transition_function',
    
    # Configuration
    'PYTORCH_AVAILABLE',
    'DEVICE',
    'USE_PYTORCH',
]
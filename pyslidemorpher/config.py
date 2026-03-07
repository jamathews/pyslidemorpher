"""
Configuration module for PySlide Morpher.
Contains global constants and PyTorch setup.
"""

import logging

# PyTorch imports for GPU acceleration
try:
    import torch
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
    # Check for CUDA availability
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logging.info(f"PyTorch GPU acceleration available on {torch.cuda.get_device_name()}")
    else:
        logging.info("PyTorch CPU acceleration available")
except ImportError:
    PYTORCH_AVAILABLE = False
    DEVICE = None
    logging.info("PyTorch not available, using NumPy/OpenCV only")

# Global variable to control PyTorch usage
USE_PYTORCH = False
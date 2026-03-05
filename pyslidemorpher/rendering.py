"""
Core rendering functions for PySlide Morpher.
Contains particle preparation, frame rendering, and blending functionality.
"""

import logging
import random
import numpy as np
import cv2

from .config import PYTORCH_AVAILABLE, DEVICE, USE_PYTORCH

# Import torch conditionally
if PYTORCH_AVAILABLE:
    import torch


def prepare_transition(a_low, b_low, seed=None):
    """Prepare transition data by creating particle positions and colors."""
    lh, lw = a_low.shape[:2]
    n = lh * lw
    a_cols = a_low.reshape(-1, 3).astype(np.float32)
    b_cols = b_low.reshape(-1, 3).astype(np.float32)
    y, x = np.indices((lh, lw))
    a_pos = np.stack([x.flatten(), y.flatten()], axis=1).astype(np.float32)
    b_pos = a_pos.copy()
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    perm = np.array(perm, dtype=np.int32)
    src_idx = perm
    tgt_idx = np.arange(n)
    logging.debug(f"Transition prepared with grid_shape: {lh}x{lw}, seed: {seed}")
    return a_pos[src_idx], b_pos[tgt_idx], a_cols[src_idx], b_cols[tgt_idx], (lh, lw)


def render_frame(pos, cols, grid_shape):
    """Render frame using NumPy (fallback implementation)."""
    lh, lw = grid_shape
    xi = np.rint(pos[:, 0]).astype(np.int32)
    yi = np.rint(pos[:, 1]).astype(np.int32)
    np.clip(xi, 0, lw - 1, out=xi)
    np.clip(yi, 0, lh - 1, out=yi)
    canvas = np.zeros((lh, lw, 3), dtype=np.float32)
    canvas[yi, xi] = cols
    logging.debug("Rendered a single frame (NumPy)")
    return canvas.clip(0, 255).astype(np.uint8)


def render_frame_torch(pos, cols, grid_shape):
    """Render frame using PyTorch with GPU acceleration."""
    if not PYTORCH_AVAILABLE:
        return render_frame(pos, cols, grid_shape)

    lh, lw = grid_shape

    # Convert to PyTorch tensors with consistent dtypes and move to GPU
    pos_tensor = torch.from_numpy(pos.astype(np.float32)).to(DEVICE)
    cols_tensor = torch.from_numpy(cols.astype(np.float32)).to(DEVICE)

    # Round and clamp positions
    xi = torch.round(pos_tensor[:, 0]).long()
    yi = torch.round(pos_tensor[:, 1]).long()
    xi = torch.clamp(xi, 0, lw - 1)
    yi = torch.clamp(yi, 0, lh - 1)

    # Create canvas on GPU
    canvas = torch.zeros((lh, lw, 3), dtype=torch.float32, device=DEVICE)

    # Use advanced indexing to set pixel values
    canvas[yi, xi] = cols_tensor

    # Clamp values and convert back to numpy
    canvas = torch.clamp(canvas, 0, 255)
    result = canvas.cpu().numpy().astype(np.uint8)

    logging.debug("Rendered a single frame (PyTorch GPU)")
    return result


def render_frame_optimized(pos, cols, grid_shape):
    """Choose the best rendering method based on availability and user preference."""
    global USE_PYTORCH
    if PYTORCH_AVAILABLE and USE_PYTORCH and len(pos) > 100:  # Use PyTorch if requested and available
        return render_frame_torch(pos, cols, grid_shape)
    else:
        return render_frame(pos, cols, grid_shape)


def create_smooth_blend_frame(solid_img, pixelated_img, blend_factor):
    """
    Create a smooth blend between a solid image and its pixelated version.

    Args:
        solid_img: Full resolution solid image
        pixelated_img: Pixelated version of the image
        blend_factor: Float between 0.0 (fully solid) and 1.0 (fully pixelated)

    Returns:
        Blended frame
    """
    # Ensure both images have the same dimensions
    if solid_img.shape != pixelated_img.shape:
        H, W = solid_img.shape[:2]
        pixelated_img = cv2.resize(pixelated_img, (W, H), interpolation=cv2.INTER_NEAREST)

    # Apply smooth blending
    blend_factor = np.clip(blend_factor, 0.0, 1.0)
    blended = (1.0 - blend_factor) * solid_img.astype(np.float32) + blend_factor * pixelated_img.astype(np.float32)
    return blended.clip(0, 255).astype(np.uint8)
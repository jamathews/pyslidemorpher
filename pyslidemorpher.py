#!/usr/bin/env python3
"""
Pixel-Morph Slideshow (fixed for imageio v2 API)
------------------------------------------------
Transitions between photos by moving “pixels” from their positions in one image
to new positions in the next image.

Dependencies:
    pip install imageio[ffmpeg] opencv-python
"""

import argparse
import logging
import random
import time  # Add at the top of the file if not already imported
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue

import cv2
import imageio  # v2 API for get_writer
import numpy as np

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


def parse_size(s: str):
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("Size must look like WIDTHxHEIGHT, e.g. 1920x1080")


def list_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    files = sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])
    if not files:
        logging.error(f"No images found in: {folder}")
        raise SystemExit(1)
    logging.debug(f"Found {len(files)} valid images in folder: {folder}")
    return files


def fit_letterbox(img, out_wh):
    out_w, out_h = out_wh
    h, w = img.shape[:2]
    scale = min(out_w / w, out_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x0 = (out_w - new_w) // 2
    y0 = (out_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    logging.debug(f"Image resized to fit letterbox of {out_wh}, new size: {new_w}x{new_h}")
    return canvas


def downsample_for_particles(img, pixel_size):
    H, W = img.shape[:2]
    lw, lh = W // pixel_size, H // pixel_size
    low = cv2.resize(img, (lw, lh), interpolation=cv2.INTER_AREA)
    logging.debug(f"Downsampled image for particles with pixel size {pixel_size}, new dimensions: {lw}x{lh}")
    return low, (lh, lw)


def easing_fn(name):
    def linear(t):
        return t

    def smoothstep(t):
        return t * t * (3 - 2 * t)

    def cubic(t):
        return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2

    name = (name or "smoothstep").lower()
    if name == "linear": return linear
    if name in ("smoothstep", "smooth"): return smoothstep
    if name in ("cubic", "ease", "easeinoutcubic"): return cubic
    return smoothstep


def prepare_transition(a_low, b_low, seed=None):
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


def make_transition_frames(a_img, b_img, *, pixel_size, fps, seconds, hold, ease_name, seed):
    total_hold_frames = int(round(hold * fps))
    logging.debug(f"Generating hold frames: {total_hold_frames}")
    for _ in range(total_hold_frames):
        yield a_img

    a_low, _ = downsample_for_particles(a_img, pixel_size)
    b_low, grid_shape = downsample_for_particles(b_img, pixel_size)
    a_pos, b_pos, a_cols, b_cols, grid_shape = prepare_transition(a_low, b_low, seed=seed)
    n_frames = max(2, int(round(seconds * fps)))
    ease = easing_fn(ease_name)
    H, W = a_img.shape[:2]

    for f in range(n_frames):
        t = f / (n_frames - 1) if n_frames > 1 else 1.0
        s = ease(t)
        pos = (1.0 - s) * a_pos + s * b_pos
        cols = (1.0 - s) * a_cols + s * b_cols
        low_frame = render_frame_optimized(pos, cols, grid_shape)
        frame = cv2.resize(low_frame, (W, H), interpolation=cv2.INTER_NEAREST)
        logging.debug(f"Generated transition frame {f + 1}/{n_frames}")
        yield frame

    for _ in range(total_hold_frames):
        yield b_img


def make_swarm_transition_frames(a_img, b_img, *, pixel_size, fps, seconds, hold, ease_name, seed):
    """Create a transition where pixels 'swarm' intensely like birds before settling into the next image."""
    total_hold_frames = int(round(hold * fps))
    logging.debug(f"Generating hold frames: {total_hold_frames}")
    for _ in range(total_hold_frames):
        yield a_img

    a_low, _ = downsample_for_particles(a_img, pixel_size)
    b_low, grid_shape = downsample_for_particles(b_img, pixel_size)
    a_pos, b_pos, a_cols, b_cols, grid_shape = prepare_transition(a_low, b_low, seed=seed)
    n_frames = max(2, int(round(seconds * fps)))
    ease = easing_fn(ease_name)
    H, W = a_img.shape[:2]

    # Increase randomness to exaggerate the swarming effect
    rng = np.random.default_rng(seed)
    velocities = rng.uniform(-4, 4, size=(len(a_pos), 2))  # Faster initial random velocities
    accelerations = rng.uniform(-0.5, 0.5, size=(len(a_pos), 2))  # Larger random accelerations

    for f in range(n_frames):
        t = f / (n_frames - 1) if n_frames > 1 else 1.0
        s = ease(t)

        # Update velocities and positions
        velocities += accelerations * (1 - s)  # Amplify acceleration effects as the transition begins
        velocities = np.clip(velocities, -6, 6)  # Cap velocity magnitudes for stability
        swarming_pos = a_pos + velocities * (1 - s) * 4  # Scale velocity effect for exaggerated swarming

        # Introduce a swirling motion
        angles = rng.uniform(0, 2 * np.pi, size=(len(a_pos),))  # Random rotation angles
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        # Create a rotation matrix for each particle
        swirling_offsets = np.column_stack((
            velocities[:, 0] * cos_angles - velocities[:, 1] * sin_angles,
            velocities[:, 0] * sin_angles + velocities[:, 1] * cos_angles
        )) * (1 - s)

        # Combine swirling and swarming movements
        swarming_pos += swirling_offsets

        # Blend between swarming positions and the final positions
        pos = (1.0 - s) * swarming_pos + s * b_pos
        cols = (1.0 - s) * a_cols + s * b_cols

        # Render frame
        low_frame = render_frame_optimized(pos, cols, grid_shape)
        frame = cv2.resize(low_frame, (W, H), interpolation=cv2.INTER_NEAREST)
        logging.debug(f"Generated intensified swarm transition frame {f + 1}/{n_frames}")
        yield frame

    for _ in range(total_hold_frames):
        yield b_img


def make_tornado_transition_frames(a_img, b_img, *, pixel_size, fps, seconds, hold, ease_name, seed):
    """Create a tornado-style transition between a_img and b_img."""
    total_hold_frames = int(round(hold * fps))
    logging.debug(f"Generating hold frames: {total_hold_frames}")
    for _ in range(total_hold_frames):
        yield a_img

    H, W = a_img.shape[:2]
    a_low, _ = downsample_for_particles(a_img, pixel_size)
    b_low, grid_shape = downsample_for_particles(b_img, pixel_size)
    a_pos, b_pos, a_cols, b_cols, grid_shape = prepare_transition(a_low, b_low, seed=seed)

    n_frames = max(2, int(round(seconds * fps)))
    ease = easing_fn(ease_name)
    center = np.array([W / 2.0 / pixel_size, H / 2.0 / pixel_size])  # Center of the tornado

    for f in range(n_frames):
        t = f / (n_frames - 1) if n_frames > 1 else 1.0
        s = ease(t)

        # Tornado effect: calculate radii and spiral angles
        offset_from_center = a_pos - center
        radii = np.linalg.norm(offset_from_center, axis=1)
        angles = np.arctan2(offset_from_center[:, 1], offset_from_center[:, 0])

        # Add swirling motion that decreases over time
        swirliness = (1 - s) * np.pi * 4  # Swirl intensity
        angles += swirliness

        # Recalculate positions based on spiral motion
        swirling_offsets = np.column_stack((
            radii * np.cos(angles),
            radii * np.sin(angles)
        ))

        # Combine swirling with final transition position
        swirly_pos = center + swirling_offsets
        pos = (1.0 - s) * swirly_pos + s * b_pos
        cols = (1.0 - s) * a_cols + s * b_cols

        # Render and yield the frame
        low_frame = render_frame_optimized(pos, cols, grid_shape)
        frame = cv2.resize(low_frame, (W, H), interpolation=cv2.INTER_NEAREST)
        logging.debug(f"Generated tornado transition frame {f + 1}/{n_frames}")
        yield frame

    for _ in range(total_hold_frames):
        yield b_img


def make_drip_transition_frames(a_img, b_img, *, pixel_size, fps, seconds, hold, ease_name, seed):
    """Create a transition where darkest pixels fall to the bottom and brightest pixels rise to the top."""
    total_hold_frames = int(round(hold * fps))
    logging.debug(f"Generating hold frames: {total_hold_frames}")
    for _ in range(total_hold_frames):
        yield a_img

    H, W = a_img.shape[:2]
    a_low, _ = downsample_for_particles(a_img, pixel_size)
    b_low, grid_shape = downsample_for_particles(b_img, pixel_size)
    a_pos, b_pos, a_cols, b_cols, grid_shape = prepare_transition(a_low, b_low, seed=seed)

    n_frames = max(2, int(round(seconds * fps)))
    ease = easing_fn(ease_name)

    # Calculate pixel brightness for sorting
    brightness = np.linalg.norm(a_cols, axis=1)  # Euclidean norm to approximate brightness
    sorted_indices = np.argsort(brightness)  # Sort pixels by brightness (dark to light)

    # Sort positions and colors by brightness
    a_pos = a_pos[sorted_indices]
    b_pos = b_pos[sorted_indices]
    a_cols = a_cols[sorted_indices]
    b_cols = b_cols[sorted_indices]

    for f in range(n_frames):
        t = f / (n_frames - 1) if n_frames > 1 else 1.0
        s = ease(t)

        # Calculate movement (falling and rising)
        darkening_offsets = np.zeros_like(a_pos)
        lightening_offsets = np.zeros_like(a_pos)

        # For the darkest pixels, move them downward
        darkening_offsets[:, 1] = (1 - s) * H / pixel_size  # Move downward

        # For the brightest pixels, move them upward
        lightening_offsets[:, 1] = -(1 - s) * H / pixel_size  # Move upward

        # Blend offsets based on pixel brightness
        blended_offsets = np.zeros_like(a_pos)
        midpoint = len(a_pos) // 2  # Rough midpoint to determine separation between dark and light pixels
        blended_offsets[:midpoint] += darkening_offsets[:midpoint]  # Apply falling to dark pixels
        blended_offsets[midpoint:] += lightening_offsets[midpoint:]  # Apply rising to bright pixels

        # Combine positions with the offsets and mix into target positions
        drip_pos = a_pos + blended_offsets
        pos = (1.0 - s) * drip_pos + s * b_pos
        cols = (1.0 - s) * a_cols + s * b_cols

        # Render and yield the frame
        low_frame = render_frame_optimized(pos, cols, grid_shape)
        frame = cv2.resize(low_frame, (W, H), interpolation=cv2.INTER_NEAREST)
        logging.debug(f"Generated drip transition frame {f + 1}/{n_frames}")
        yield frame

    for _ in range(total_hold_frames):
        yield b_img


def make_rainfall_transition_frames(a_img, b_img, *, pixel_size, fps, seconds, hold, ease_name, seed):
    """
    Create a transition where b_img falls from above, replacing pixels of a_img until frame fills with b_img.
    """
    total_hold_frames = int(round(hold * fps))
    logging.debug(f"Generating hold frames: {total_hold_frames}")
    for _ in range(total_hold_frames):
        yield a_img

    H, W = a_img.shape[:2]
    a_low, _ = downsample_for_particles(a_img, pixel_size)
    b_low, grid_shape = downsample_for_particles(b_img, pixel_size)
    _, b_pos, _, b_cols, grid_shape = prepare_transition(a_low, b_low, seed=seed)

    n_frames = max(2, int(round(seconds * fps)))
    ease = easing_fn(ease_name)

    # Begin with b_img pixels above the frame
    initial_y_offset = -H // pixel_size
    b_pos[:, 1] += initial_y_offset  # Shift all starting b_img pixels above the top of the frame

    for f in range(n_frames):
        t = f / (n_frames - 1) if n_frames > 1 else 1.0
        s = ease(t)

        # Blend a_img and b_img pixels based on vertical falling progress
        current_b_pos = b_pos.copy()
        current_b_pos[:, 1] += s * H / pixel_size  # Bring b_img pixels down into the frame

        # Composite b_img pixels over a_img pixels
        mask = current_b_pos[:, 1] >= 0  # Only render b_img pixels once they "enter" the visible frame

        pos = np.where(mask[:, None], current_b_pos, b_pos)  # Use b_pos only if b_img hasn't entered
        cols = np.where(mask[:, None], b_cols, a_low.reshape(-1, 3))  # Blend colors of a_img and b_img

        # Prepare and render the frame
        low_frame = render_frame_optimized(pos, cols, grid_shape)
        frame = cv2.resize(low_frame, (W, H), interpolation=cv2.INTER_NEAREST)
        logging.debug(f"Generated rainfall transition frame {f + 1}/{n_frames}")
        yield frame

    # Final hold frames with b_img unmodified
    for _ in range(total_hold_frames):
        yield b_img


def make_sorted_transition_frames(a_img, b_img, *, pixel_size, fps, seconds, hold, ease_name, seed):
    """
     Create a transition where pixels are gradually sorted by luminosity, starting from a_img,
    through intermediate sorting stages for both a_img and b_img, and ending with an unmodified b_img.
    """
    total_hold_frames = int(round(hold * fps))
    logging.debug(f"Generating hold frames: {total_hold_frames}")
    for _ in range(total_hold_frames):
        yield a_img

    H, W = a_img.shape[:2]
    a_low, _ = downsample_for_particles(a_img, pixel_size)
    b_low, grid_shape = downsample_for_particles(b_img, pixel_size)

    # Prepare transitions based on pixel positions and colors
    a_pos, b_pos, a_cols, b_cols, grid_shape = prepare_transition(a_low, b_low, seed=seed)

    # Calculate pixel brightness (luminosity) for sorting
    a_luminosity = np.linalg.norm(a_cols, axis=1)  # Approximate brightness for a_img
    b_luminosity = np.linalg.norm(b_cols, axis=1)  # Approximate brightness for b_img

    # Sort all pixels by luminosity (independently for a_img and b_img)
    a_sorted_indices = np.argsort(a_luminosity)
    b_sorted_indices = np.argsort(b_luminosity)

    a_pos_sorted = a_pos[a_sorted_indices]
    a_cols_sorted = a_cols[a_sorted_indices]
    b_pos_sorted = b_pos[b_sorted_indices]
    b_cols_sorted = b_cols[b_sorted_indices]

    n_frames = max(2, int(round(seconds * fps)))
    ease = easing_fn(ease_name)

    for f in range(n_frames):
        t = f / (n_frames - 1) if n_frames > 1 else 1.0
        s = ease(t)

        # Intermediate sorting progress
        sorted_fraction = int(s * len(a_pos))  # Fraction of pixels to sort at the current frame

        # Gradually sort a_img pixels
        a_current_indices = np.concatenate([
            a_sorted_indices[:sorted_fraction],  # Sorted pixels
            a_sorted_indices[sorted_fraction:]  # Unsorted pixels
        ])
        a_pos_intermediate = a_pos[a_current_indices]
        a_cols_intermediate = a_cols[a_current_indices]

        # Gradually sort b_img pixels
        b_current_indices = np.concatenate([
            b_sorted_indices[:sorted_fraction],  # Sorted pixels
            b_sorted_indices[sorted_fraction:]  # Unsorted pixels
        ])
        b_pos_intermediate = b_pos[b_current_indices]
        b_cols_intermediate = b_cols[b_current_indices]

        # Interpolate between a_img and b_img during transition
        pos = (1.0 - s) * a_pos_intermediate + s * b_pos_intermediate
        cols = (1.0 - s) * a_cols_intermediate + s * b_cols_intermediate

        # Render and resize frame
        low_frame = render_frame_optimized(pos, cols, grid_shape)
        frame = cv2.resize(low_frame, (W, H), interpolation=cv2.INTER_NEAREST)
        logging.debug(f"Generated intermediate sorted transition frame {f + 1}/{n_frames}")
        yield frame

    # Final hold frames with b_img unmodified
    for _ in range(total_hold_frames):
        yield b_img


def make_hue_sorted_transition_frames(a_img, b_img, *, pixel_size, fps, seconds, hold, ease_name, seed):
    """
    Create a transition where pixels are gradually sorted by hue instead of luminosity, starting from a_img,
    through intermediate sorting stages for both a_img and b_img, and ending with an unmodified b_img.
    """
    total_hold_frames = int(round(hold * fps))
    logging.debug(f"Generating hold frames: {total_hold_frames}")
    for _ in range(total_hold_frames):
        yield a_img

    H, W = a_img.shape[:2]
    a_low, _ = downsample_for_particles(a_img, pixel_size)
    b_low, grid_shape = downsample_for_particles(b_img, pixel_size)

    # Prepare transitions based on pixel positions and colors
    a_pos, b_pos, a_cols, b_cols, grid_shape = prepare_transition(a_low, b_low, seed=seed)

    # Convert colors to HSV format for hue calculation
    a_hsv = cv2.cvtColor(a_cols.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    b_hsv = cv2.cvtColor(b_cols.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)

    # Extract the hue channel for sorting
    a_hue = a_hsv[:, 0].astype(np.float32)  # Hue of a_img
    b_hue = b_hsv[:, 0].astype(np.float32)  # Hue of b_img

    # Sort all pixels by hue (independently for a_img and b_img)
    a_sorted_indices = np.argsort(a_hue)
    b_sorted_indices = np.argsort(b_hue)

    a_pos_sorted = a_pos[a_sorted_indices]
    a_cols_sorted = a_cols[a_sorted_indices]
    b_pos_sorted = b_pos[b_sorted_indices]
    b_cols_sorted = b_cols[b_sorted_indices]

    n_frames = max(2, int(round(seconds * fps)))
    ease = easing_fn(ease_name)

    for f in range(n_frames):
        t = f / (n_frames - 1) if n_frames > 1 else 1.0
        s = ease(t)

        # Intermediate sorting progress
        sorted_fraction = int(s * len(a_pos))  # Fraction of pixels to sort at the current frame

        # Gradually sort a_img pixels
        a_current_indices = np.concatenate([
            a_sorted_indices[:sorted_fraction],  # Sorted pixels
            a_sorted_indices[sorted_fraction:]  # Unsorted pixels
        ])
        a_pos_intermediate = a_pos[a_current_indices]
        a_cols_intermediate = a_cols[a_current_indices]

        # Gradually sort b_img pixels
        b_current_indices = np.concatenate([
            b_sorted_indices[:sorted_fraction],  # Sorted pixels
            b_sorted_indices[sorted_fraction:]  # Unsorted pixels
        ])
        b_pos_intermediate = b_pos[b_current_indices]
        b_cols_intermediate = b_cols[b_current_indices]

        # Interpolate between a_img and b_img during transition
        pos = (1.0 - s) * a_pos_intermediate + s * b_pos_intermediate
        cols = (1.0 - s) * a_cols_intermediate + s * b_cols_intermediate

        # Render and resize frame
        low_frame = render_frame_optimized(pos, cols, grid_shape)
        frame = cv2.resize(low_frame, (W, H), interpolation=cv2.INTER_NEAREST)
        logging.debug(f"Generated intermediate hue-sorted transition frame {f + 1}/{n_frames}")
        yield frame

    # Final hold frames with b_img unmodified
    for _ in range(total_hold_frames):
        yield b_img


def play_realtime(imgs, args):
    """Play slideshow in realtime using OpenCV display with optimized performance."""
    W, H = args.size
    frame_time = 1.0 / args.fps  # Time per frame in seconds

    # Create OpenCV window with optimizations
    window_name = "PySlidemorpher - Realtime Slideshow"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Enable GPU acceleration if available
    try:
        cv2.setUseOptimized(True)
        if cv2.useOptimized():
            logging.info("OpenCV optimizations enabled")
    except:
        pass

    # Frame buffer for smoother playback
    frame_buffer = Queue(maxsize=args.fps * 2)  # Buffer 2 seconds worth of frames

    def frame_generator():
        """Generate frames in a separate thread for better performance."""
        try:
            # Initial hold frames
            init_hold = int(round(args.hold * args.fps))
            logging.debug(f"Displaying initial hold frames: {init_hold}")
            for _ in range(init_hold // 4):
                # Convert RGB to BGR for OpenCV display
                frame_bgr = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2BGR)
                frame_buffer.put(frame_bgr)

            # Process transitions
            for i in range(len(imgs) - 1):
                logging.info(f"Processing transition {i + 1}/{len(imgs) - 1}")
                a, b = imgs[i], imgs[i + 1]
                pair_seed = (args.seed or 0) + i * random.randint(1, 10000)

                # Select transition function
                if args.transition == "swarm":
                    transition_fn = make_swarm_transition_frames
                elif args.transition == "tornado":
                    transition_fn = make_tornado_transition_frames
                elif args.transition == "drip":
                    transition_fn = make_drip_transition_frames
                elif args.transition == "rain":
                    transition_fn = make_rainfall_transition_frames
                elif args.transition == "sorted":
                    transition_fn = make_sorted_transition_frames
                elif args.transition == "hue-sorted":
                    transition_fn = make_hue_sorted_transition_frames
                else:
                    transition_fn = make_transition_frames

                # Generate transition frames
                for frame in transition_fn(
                        a, b,
                        pixel_size=args.pixel_size,
                        fps=args.fps,
                        seconds=args.seconds_per_transition,
                        hold=0.0,
                        ease_name=args.easing,
                        seed=pair_seed,
                ):
                    # Convert RGB to BGR for OpenCV display
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_buffer.put(frame_bgr)

                # Hold frames
                for _ in range(init_hold // 4):
                    frame_bgr = cv2.cvtColor(b, cv2.COLOR_RGB2BGR)
                    frame_buffer.put(frame_bgr)

            # Signal end of frames
            frame_buffer.put(None)

        except Exception as e:
            logging.error(f"Error in frame generation: {e}")
            frame_buffer.put(None)

    # Start frame generation in separate thread
    generator_thread = threading.Thread(target=frame_generator, daemon=True)
    generator_thread.start()

    logging.info("Starting realtime playback. Press 'q' to quit, 'p' to pause/resume, 'r' to restart.")

    paused = False
    start_time = time.time()
    frame_count = 0

    try:
        while True:
            if not paused:
                # Get next frame from buffer
                try:
                    frame = frame_buffer.get(timeout=1.0)
                    if frame is None:  # End of frames
                        logging.info("Slideshow completed. Restarting...")
                        # Restart by creating new generator thread
                        generator_thread = threading.Thread(target=frame_generator, daemon=True)
                        generator_thread.start()
                        continue

                    # Display frame
                    cv2.imshow(window_name, frame)
                    frame_count += 1

                    # Calculate timing for consistent FPS
                    expected_time = start_time + (frame_count * frame_time)
                    current_time = time.time()
                    sleep_time = expected_time - current_time

                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    elif sleep_time < -frame_time:  # If we're more than one frame behind, reset timing
                        start_time = current_time
                        frame_count = 0

                except:
                    # If buffer is empty, just wait a bit
                    time.sleep(frame_time)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                if not paused:
                    # Reset timing when resuming
                    start_time = time.time()
                    frame_count = 0
                logging.info(f"Playback {'paused' if paused else 'resumed'}")
            elif key == ord('r'):
                # Restart slideshow
                logging.info("Restarting slideshow...")
                # Clear buffer
                while not frame_buffer.empty():
                    try:
                        frame_buffer.get_nowait()
                    except:
                        break
                # Start new generator thread
                generator_thread = threading.Thread(target=frame_generator, daemon=True)
                generator_thread.start()
                start_time = time.time()
                frame_count = 0
                paused = False

    finally:
        cv2.destroyAllWindows()
        logging.info("Realtime playback ended.")


def main():
    ap = argparse.ArgumentParser(description="Pixel-morph slideshow video generator")
    ap.add_argument("photos_folder", type=Path, help="Folder containing images")
    ap.add_argument("--out", default=None, help="Output video filename")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second")
    ap.add_argument("--seconds-per-transition", type=float, default=2.0)
    ap.add_argument("--hold", type=float, default=0.5)
    ap.add_argument("--pixel-size", type=int, default=4)
    ap.add_argument("--size", type=parse_size, default="1920x1080")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--easing", default="smoothstep",
                    choices=["linear", "smoothstep", "cubic"])
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--transition", type=str, default="default",
                    choices=["default", "swarm", "tornado", "drip", "rain", "sorted", "hue-sorted"],
                    help="Select the type of transition")
    ap.add_argument("--realtime", action="store_true",
                    help="Play slideshow in realtime instead of writing to file")
    ap.add_argument("--use-pytorch", action="store_true",
                    help="Enable PyTorch GPU acceleration (requires PyTorch installation)")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    help="Set the log level for the script")
    args = ap.parse_args()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), None),
                        format="%(asctime)s - %(levelname)s - %(message)s")

    # Set PyTorch usage based on command-line argument
    global USE_PYTORCH
    USE_PYTORCH = args.use_pytorch
    if USE_PYTORCH and PYTORCH_AVAILABLE:
        logging.info(f"PyTorch acceleration enabled (using {DEVICE})")
    elif USE_PYTORCH and not PYTORCH_AVAILABLE:
        logging.warning("PyTorch acceleration requested but PyTorch is not available")
        USE_PYTORCH = False

    # Construct default filename if --out is not specified and not in realtime mode
    if not args.realtime and args.out is None:
        args.out = (f"slideshow_{args.size[0]}x{args.size[1]}_{args.fps}fps_"
                    f"{args.pixel_size}px_{args.seconds_per_transition}s_"
                    f"{args.hold}hold_{args.easing}_"
                    f"{args.transition}_{args.preset}.mp4")
        logging.info(f"No --out specified. Using default filename: {args.out}")

    if args.realtime:
        logging.info("Starting Pixel-Morph Slideshow in realtime mode.")
    else:
        logging.info("Starting Pixel-Morph Slideshow generator.")

    files = list_images(args.photos_folder)
    W, H = args.size
    logging.info(f"Found {len(files)} images. Output: {W}x{H} at {args.fps} fps")

    imgs = []
    for idx, p in enumerate(files):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            logging.warning(f"Could not read {p}, skipping.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = fit_letterbox(img, (W, H))
        logging.debug(f"Processed image {idx + 1}/{len(files)}: {p.name}")
        imgs.append(img)
    imgs.append(imgs[0])  # Append the first image to the end to create a loop

    if len(imgs) < 2:
        logging.error("Need at least 2 readable images.")
        raise SystemExit(1)

    # If realtime mode is enabled, use realtime playback instead of file writing
    if args.realtime:
        play_realtime(imgs, args)
        return

    writer = imageio.get_writer(
        args.out,
        fps=args.fps,
        codec="libx264",
        quality=None,
        ffmpeg_params=[
            "-pix_fmt", "yuv420p",
            "-preset", args.preset,
            "-crf", str(args.crf),
        ],
    )

    try:
        # Before the loop that processes transitions, create a start time
        start_time = time.time()

        init_hold = int(round(args.hold * args.fps))
        logging.debug(f"Writing initial hold frames: {init_hold}")
        for _ in range(init_hold // 4):
            writer.append_data(imgs[0])

        # Update the loop to calculate and log transition times and estimated remaining time
        for i in range(len(imgs) - 1):
            logging.info(f"Processing transition {i + 1}/{len(imgs) - 1}")
            a, b = imgs[i], imgs[i + 1]
            pair_seed = (args.seed or 0) + i * random.randint(1, 10000)

            if args.transition == "swarm":
                transition_fn = make_swarm_transition_frames
            elif args.transition == "tornado":
                transition_fn = make_tornado_transition_frames
            elif args.transition == "drip":
                transition_fn = make_drip_transition_frames
            elif args.transition == "rain":
                transition_fn = make_rainfall_transition_frames
            elif args.transition == "sorted":
                transition_fn = make_sorted_transition_frames
            elif args.transition == "hue-sorted":
                transition_fn = make_hue_sorted_transition_frames
            else:
                transition_fn = make_transition_frames

            transition_start = time.time()  # Track the start time of the current transition

            for frame in transition_fn(
                    a, b,
                    pixel_size=args.pixel_size,
                    fps=args.fps,
                    seconds=args.seconds_per_transition,
                    hold=0.0,
                    ease_name=args.easing,
                    seed=pair_seed,
            ):
                writer.append_data(frame)

            transition_end = time.time()  # Track the end time of the current transition
            elapsed_time = transition_end - transition_start

            # Estimate total remaining time
            transitions_completed = i + 1
            avg_time_per_transition = (transition_end - start_time) / transitions_completed
            remaining_transitions = (len(imgs) - 1) - transitions_completed
            estimated_remaining_time = avg_time_per_transition * remaining_transitions

            logging.info(f"Finished transition {i + 1}/{len(imgs) - 1} "
                         f"in {elapsed_time:.2f} seconds. Estimated remaining time: "
                         f"{estimated_remaining_time:.2f} seconds.")

            for _ in range(init_hold // 4):
                writer.append_data(b)

    finally:
        writer.close()

    logging.info(f"Done. Wrote: {args.out}")


if __name__ == "__main__":
    main()

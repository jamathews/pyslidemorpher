"""
Transition effects for PySlide Morpher.
Contains all the different transition animations between images.
"""

import logging
import random
import numpy as np
import cv2

from .utils import downsample_for_particles, easing_fn
from .rendering import prepare_transition, render_frame_optimized, create_smooth_blend_frame


def make_transition_frames(a_img, b_img, *, pixel_size, fps, seconds, hold, ease_name, seed):
    """Default transition with smooth blending between pixelated and solid images."""
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

    # Calculate smooth blend frames (about 10% of transition at start and end)
    blend_frames = max(1, int(n_frames * 0.1))
    core_frames = n_frames - 2 * blend_frames

    # Create initial pixelated versions for blending
    initial_pos = a_pos.copy()
    initial_cols = a_cols.copy()
    initial_low_frame = render_frame_optimized(initial_pos, initial_cols, grid_shape)
    initial_pixelated = cv2.resize(initial_low_frame, (W, H), interpolation=cv2.INTER_NEAREST)

    final_pos = b_pos.copy()
    final_cols = b_cols.copy()
    final_low_frame = render_frame_optimized(final_pos, final_cols, grid_shape)
    final_pixelated = cv2.resize(final_low_frame, (W, H), interpolation=cv2.INTER_NEAREST)

    frame_idx = 0

    # Start blend frames: solid a_img → pixelated
    for f in range(blend_frames):
        blend_factor = (f + 1) / blend_frames  # 0 to 1
        frame = create_smooth_blend_frame(a_img, initial_pixelated, blend_factor)
        logging.debug(f"Generated start blend frame {frame_idx + 1}/{n_frames}")
        yield frame
        frame_idx += 1

    # Core transition frames: pixelated a → pixelated b
    for f in range(core_frames):
        t = f / (core_frames - 1) if core_frames > 1 else 1.0
        s = ease(t)
        pos = (1.0 - s) * a_pos + s * b_pos
        cols = (1.0 - s) * a_cols + s * b_cols
        low_frame = render_frame_optimized(pos, cols, grid_shape)
        frame = cv2.resize(low_frame, (W, H), interpolation=cv2.INTER_NEAREST)
        logging.debug(f"Generated core transition frame {frame_idx + 1}/{n_frames}")
        yield frame
        frame_idx += 1

    # End blend frames: pixelated → solid b_img
    for f in range(blend_frames):
        blend_factor = 1.0 - (f + 1) / blend_frames  # 1 to 0
        frame = create_smooth_blend_frame(b_img, final_pixelated, blend_factor)
        logging.debug(f"Generated end blend frame {frame_idx + 1}/{n_frames}")
        yield frame
        frame_idx += 1

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

    # Calculate smooth blend frames (about 10% of transition at start and end)
    blend_frames = max(1, int(n_frames * 0.1))
    core_frames = n_frames - 2 * blend_frames

    # Create initial pixelated versions for blending
    initial_pos = a_pos.copy()
    initial_cols = a_cols.copy()
    initial_low_frame = render_frame_optimized(initial_pos, initial_cols, grid_shape)
    initial_pixelated = cv2.resize(initial_low_frame, (W, H), interpolation=cv2.INTER_NEAREST)

    final_pos = b_pos.copy()
    final_cols = b_cols.copy()
    final_low_frame = render_frame_optimized(final_pos, final_cols, grid_shape)
    final_pixelated = cv2.resize(final_low_frame, (W, H), interpolation=cv2.INTER_NEAREST)

    # Increase randomness to exaggerate the swarming effect
    rng = np.random.default_rng(seed)
    velocities = rng.uniform(-4, 4, size=(len(a_pos), 2))  # Faster initial random velocities
    accelerations = rng.uniform(-0.5, 0.5, size=(len(a_pos), 2))  # Larger random accelerations

    frame_idx = 0

    # Start blend frames: solid a_img → pixelated
    for f in range(blend_frames):
        blend_factor = (f + 1) / blend_frames  # 0 to 1
        frame = create_smooth_blend_frame(a_img, initial_pixelated, blend_factor)
        logging.debug(f"Generated swarm start blend frame {frame_idx + 1}/{n_frames}")
        yield frame
        frame_idx += 1

    # Core swarm transition frames: pixelated a → pixelated b with swarming
    for f in range(core_frames):
        t = f / (core_frames - 1) if core_frames > 1 else 1.0
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
        logging.debug(f"Generated swarm core transition frame {frame_idx + 1}/{n_frames}")
        yield frame
        frame_idx += 1

    # End blend frames: pixelated → solid b_img
    for f in range(blend_frames):
        blend_factor = 1.0 - (f + 1) / blend_frames  # 1 to 0
        frame = create_smooth_blend_frame(b_img, final_pixelated, blend_factor)
        logging.debug(f"Generated swarm end blend frame {frame_idx + 1}/{n_frames}")
        yield frame
        frame_idx += 1

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

    # Calculate smooth blend frames (about 10% of transition at start and end)
    blend_frames = max(1, int(n_frames * 0.1))
    core_frames = n_frames - 2 * blend_frames

    # Create initial pixelated versions for blending
    initial_pos = a_pos.copy()
    initial_cols = a_cols.copy()
    initial_low_frame = render_frame_optimized(initial_pos, initial_cols, grid_shape)
    initial_pixelated = cv2.resize(initial_low_frame, (W, H), interpolation=cv2.INTER_NEAREST)

    final_pos = b_pos.copy()
    final_cols = b_cols.copy()
    final_low_frame = render_frame_optimized(final_pos, final_cols, grid_shape)
    final_pixelated = cv2.resize(final_low_frame, (W, H), interpolation=cv2.INTER_NEAREST)

    frame_idx = 0

    # Start blend frames: solid a_img → pixelated
    for f in range(blend_frames):
        blend_factor = (f + 1) / blend_frames  # 0 to 1
        frame = create_smooth_blend_frame(a_img, initial_pixelated, blend_factor)
        logging.debug(f"Generated tornado start blend frame {frame_idx + 1}/{n_frames}")
        yield frame
        frame_idx += 1

    # Core tornado transition frames: pixelated a → pixelated b with tornado effect
    for f in range(core_frames):
        t = f / (core_frames - 1) if core_frames > 1 else 1.0
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
        logging.debug(f"Generated tornado core transition frame {frame_idx + 1}/{n_frames}")
        yield frame
        frame_idx += 1

    # End blend frames: pixelated → solid b_img
    for f in range(blend_frames):
        blend_factor = 1.0 - (f + 1) / blend_frames  # 1 to 0
        frame = create_smooth_blend_frame(b_img, final_pixelated, blend_factor)
        logging.debug(f"Generated tornado end blend frame {frame_idx + 1}/{n_frames}")
        yield frame
        frame_idx += 1

    for _ in range(total_hold_frames):
        yield b_img


def make_swirl_transition_frames(a_img, b_img, *, pixel_size, fps, seconds, hold, ease_name, seed):
    """Create a swirl-style transition with smooth three-phase flow: swirl current image -> transition -> unswirl next image."""
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
    center = np.array([W / 2.0 / pixel_size, H / 2.0 / pixel_size])  # Center of the swirl

    # Calculate swirl properties for both images
    def calculate_swirl_positions(base_pos, swirl_intensity_factor):
        offset_from_center = base_pos - center
        radii = np.linalg.norm(offset_from_center, axis=1)
        angles = np.arctan2(offset_from_center[:, 1], offset_from_center[:, 0])

        # Calculate maximum radius for normalization
        max_radius = np.max(radii) if len(radii) > 0 else 1.0
        normalized_radii = radii / max_radius
        center_bias = 1.0 - normalized_radii  # 1.0 at center, 0.0 at perimeter

        # Apply swirl intensity
        swirl_intensity = swirl_intensity_factor * np.pi * 4 * (1.0 + center_bias * 2.0)
        swirled_angles = angles + swirl_intensity

        # Calculate swirled positions
        swirled_offsets = np.column_stack((
            radii * np.cos(swirled_angles),
            radii * np.sin(swirled_angles)
        ))
        return center + swirled_offsets

    for f in range(n_frames):
        t = f / (n_frames - 1) if n_frames > 1 else 1.0
        s = ease(t)

        # Three-phase transition:
        # Phase 1 (0.0 - 0.33): Swirl the current image
        # Phase 2 (0.33 - 0.67): Transition from swirled current to swirled next
        # Phase 3 (0.67 - 1.0): Unswirl the next image

        if s <= 0.33:
            # Phase 1: Swirl the current image
            phase_progress = s / 0.33  # 0 to 1 within this phase
            swirl_factor = phase_progress  # Increase swirl intensity

            pos = (1.0 - phase_progress) * a_pos + phase_progress * calculate_swirl_positions(a_pos, swirl_factor)
            cols = a_cols  # Keep original colors

        elif s <= 0.67:
            # Phase 2: Transition between swirled images
            phase_progress = (s - 0.33) / 0.34  # 0 to 1 within this phase

            # Both images are fully swirled during this phase
            swirled_a_pos = calculate_swirl_positions(a_pos, 1.0)
            swirled_b_pos = calculate_swirl_positions(b_pos, 1.0)

            pos = (1.0 - phase_progress) * swirled_a_pos + phase_progress * swirled_b_pos
            cols = (1.0 - phase_progress) * a_cols + phase_progress * b_cols

        else:
            # Phase 3: Unswirl the next image
            phase_progress = (s - 0.67) / 0.33  # 0 to 1 within this phase
            swirl_factor = 1.0 - phase_progress  # Decrease swirl intensity

            pos = (1.0 - phase_progress) * calculate_swirl_positions(b_pos, swirl_factor) + phase_progress * b_pos
            cols = b_cols  # Keep target colors

        # Render and yield the frame
        low_frame = render_frame_optimized(pos, cols, grid_shape)
        frame = cv2.resize(low_frame, (W, H), interpolation=cv2.INTER_NEAREST)
        logging.debug(f"Generated swirl transition frame {f + 1}/{n_frames}")
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
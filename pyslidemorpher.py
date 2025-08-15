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

import cv2
import imageio  # v2 API for get_writer
import numpy as np


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
    lh, lw = grid_shape
    xi = np.rint(pos[:, 0]).astype(np.int32)
    yi = np.rint(pos[:, 1]).astype(np.int32)
    np.clip(xi, 0, lw - 1, out=xi)
    np.clip(yi, 0, lh - 1, out=yi)
    canvas = np.zeros((lh, lw, 3), dtype=np.float32)
    canvas[yi, xi] = cols
    logging.debug("Rendered a single frame")
    return canvas.clip(0, 255).astype(np.uint8)


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
        low_frame = render_frame(pos, cols, grid_shape)
        frame = cv2.resize(low_frame, (W, H), interpolation=cv2.INTER_NEAREST)
        logging.debug(f"Generated transition frame {f + 1}/{n_frames}")
        yield frame

    for _ in range(total_hold_frames):
        yield b_img


def make_swarm_transition_frames(a_img, b_img, *, pixel_size, fps, seconds, hold, ease_name, seed):
    """Create a transition where pixels 'swarm' like birds before settling into the next image."""
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

    # Random initial velocity and acceleration for the swarm effect
    rng = np.random.default_rng(seed)
    velocities = rng.uniform(-2, 2, size=(len(a_pos), 2))  # Initial random velocities
    accelerations = rng.uniform(-0.2, 0.2, size=(len(a_pos), 2))  # Random accelerations

    for f in range(n_frames):
        t = f / (n_frames - 1) if n_frames > 1 else 1.0
        s = ease(t)

        # Update velocities and positions to mimic a bird swarm effect
        velocities += accelerations  # Alter velocity by applying acceleration
        swarming_pos = a_pos + velocities * (1 - s)  # Positions spread based on velocity

        # Blend between swarming positions and the final positions
        pos = (1.0 - s) * swarming_pos + s * b_pos
        cols = (1.0 - s) * a_cols + s * b_cols

        # Render frame
        low_frame = render_frame(pos, cols, grid_shape)
        frame = cv2.resize(low_frame, (W, H), interpolation=cv2.INTER_NEAREST)
        logging.debug(f"Generated swarm transition frame {f + 1}/{n_frames}")
        yield frame

    for _ in range(total_hold_frames):
        yield b_img


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
    ap.add_argument("--transition", default="default",
                    choices=["default", "swarm"], help="Select the type of transition")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    help="Set the log level for the script")
    args = ap.parse_args()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), None),
                        format="%(asctime)s - %(levelname)s - %(message)s")

    # Construct default filename if --out is not specified
    if args.out is None:
        args.out = (f"slideshow_{args.size[0]}x{args.size[1]}_{args.fps}fps_"
                    f"{args.pixel_size}px_{args.seconds_per_transition}s_"
                    f"{args.hold}hold_{args.easing}_"
                    f"{args.transition}_{args.preset}.mp4")
        logging.info(f"No --out specified. Using default filename: {args.out}")

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

    if len(imgs) < 2:
        logging.error("Need at least 2 readable images.")
        raise SystemExit(1)

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
        for _ in range(init_hold):
            writer.append_data(imgs[0])

        # Update the loop to calculate and log transition times and estimated remaining time
        for i in range(len(imgs) - 1):
            logging.info(f"Processing transition {i + 1}/{len(imgs) - 1}")
            a, b = imgs[i], imgs[i + 1]
            pair_seed = (args.seed or 0) + i * 1337

            if args.transition == "swarm":
                transition_fn = make_swarm_transition_frames
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

            for _ in range(init_hold):
                writer.append_data(b)

    finally:
        writer.close()

    logging.info(f"Done. Wrote: {args.out}")


if __name__ == "__main__":
    main()

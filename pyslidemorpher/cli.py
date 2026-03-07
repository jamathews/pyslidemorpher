"""
Command-line interface for PySlide Morpher.
Contains the main function and argument parsing logic.
"""

import argparse
import logging
import random
import time
import subprocess
from pathlib import Path
import cv2
import imageio

from .config import USE_PYTORCH, PYTORCH_AVAILABLE, DEVICE
from .utils import parse_size, list_images, fit_letterbox
from .realtime import play_realtime, get_random_transition_function
from .transitions import (
    make_transition_frames,
    make_swarm_transition_frames,
    make_tornado_transition_frames,
    make_swirl_transition_frames,
    make_drip_transition_frames,
    make_rainfall_transition_frames,
    make_sorted_transition_frames,
    make_hue_sorted_transition_frames
)


def main():
    """Main entry point for the PySlide Morpher application."""
    ap = argparse.ArgumentParser(description="Pixel-morph slideshow video generator")
    ap.add_argument("photos_folder", type=Path, help="Folder containing images")
    ap.add_argument("--out", default=None, help="Output video filename")
    ap.add_argument("--fps", type=int, default=30, help="Frames per second")
    ap.add_argument("--seconds-per-transition", type=float, default=2.0)
    ap.add_argument("--hold", type=float, default=0.5)
    ap.add_argument("--pixel-size", type=int, default=4)
    ap.add_argument("--size", type=parse_size, default="1920x1080")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducible results. If not specified, uses current time for true randomness.")
    ap.add_argument("--easing", default="smoothstep",
                    choices=["linear", "smoothstep", "cubic"])
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--transition", type=str, default="default",
                    choices=["default", "swarm", "tornado", "swirl", "drip", "rain", "sorted", "hue-sorted", "random"],
                    help="Select the type of transition. Use 'random' to randomly choose a different transition for each frame transition.")
    ap.add_argument("--audio", type=Path, default=None,
                    help="Audio file to include in the output video")
    ap.add_argument("--realtime", action="store_true",
                    help="Play slideshow in realtime instead of writing to file")
    ap.add_argument("--use-pytorch", action="store_true",
                    help="Enable PyTorch GPU acceleration (requires PyTorch installation)")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    help="Set the log level for the script")
    ap.add_argument("--web-gui", action="store_true",
                    help="Enable web-based GUI for realtime control (requires --realtime and Flask)")
    args = ap.parse_args()


    # Validate web GUI requirements
    if args.web_gui:
        if not args.realtime:
            print("Error: --web-gui can only be used with --realtime mode")
            raise SystemExit(1)

    # Configure logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        force=True)  # Force reconfiguration of logging

    # Initialize random seed for better randomness
    # If no seed is provided, use current time for true randomness
    if args.seed is None:
        import time
        seed = int(time.time() * 1000) % (2**32)  # Use current time as seed
        random.seed(seed)
        logging.debug(f"Using time-based random seed: {seed}")
    else:
        random.seed(args.seed)
        logging.debug(f"Using provided random seed: {args.seed}")

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

    # Check if audio file exists if provided
    if args.audio and not args.audio.exists():
        logging.error(f"Audio file not found: {args.audio}")
        raise SystemExit(1)

    # Determine output filename - use temporary name if audio will be added
    if args.audio:
        temp_video_out = args.out.replace('.mp4', '_temp_no_audio.mp4')
        logging.info(f"Creating temporary video without audio: {temp_video_out}")
    else:
        temp_video_out = args.out

    # Parameters for video only (audio will be added separately if needed)
    ffmpeg_params = [
        "-pix_fmt", "yuv420p",
        "-preset", args.preset,
        "-crf", str(args.crf),
    ]

    writer = imageio.get_writer(
        temp_video_out,
        fps=args.fps,
        codec="libx264",
        quality=None,
        ffmpeg_params=ffmpeg_params,
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
            # Generate a unique seed for each transition pair
            base_seed = args.seed if args.seed is not None else 0
            pair_seed = base_seed + i * random.randint(1, 10000)

            if args.transition == "random":
                transition_fn = get_random_transition_function()
                logging.info(f"Randomly selected transition: {transition_fn.__name__}")
            elif args.transition == "swarm":
                transition_fn = make_swarm_transition_frames
            elif args.transition == "tornado":
                transition_fn = make_tornado_transition_frames
            elif args.transition == "swirl":
                transition_fn = make_swirl_transition_frames
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

            # Log which transition is being used for this image pair
            logging.info(f"Using transition: {transition_fn.__name__}")

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

    # If audio was provided, combine the temporary video with audio
    if args.audio:
        logging.info(f"Combining video with audio from: {args.audio}")
        try:
            # Use ffmpeg to combine video and audio
            ffmpeg_cmd = [
                "ffmpeg", "-y",  # -y to overwrite output file
                "-i", temp_video_out,  # Video input
                "-i", str(args.audio),  # Audio input
                "-c:v", "copy",  # Copy video stream (no re-encoding)
                "-c:a", "aac",   # Encode audio to AAC
                "-shortest",     # Stop when shortest stream ends
                args.out         # Final output
            ]

            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logging.info(f"Successfully combined video with audio: {args.out}")
                # Remove temporary video file
                Path(temp_video_out).unlink()
                logging.debug(f"Removed temporary file: {temp_video_out}")
            else:
                logging.error(f"Failed to combine video with audio: {result.stderr}")
                logging.info(f"Temporary video saved as: {temp_video_out}")
                raise SystemExit(1)

        except FileNotFoundError:
            logging.error("ffmpeg not found. Please install ffmpeg to use audio functionality.")
            logging.info(f"Video without audio saved as: {temp_video_out}")
            raise SystemExit(1)
        except Exception as e:
            logging.error(f"Error combining video with audio: {e}")
            logging.info(f"Video without audio saved as: {temp_video_out}")
            raise SystemExit(1)

    logging.info(f"Done. Wrote: {args.out}")

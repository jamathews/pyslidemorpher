"""Command-line interface for PySlide Morpher."""

import argparse
import logging
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import imageio

from . import config
from .config import DEVICE, PYTORCH_AVAILABLE
from .realtime import get_random_transition_function, play_realtime
from .transitions import (
    make_drip_transition_frames,
    make_hue_sorted_transition_frames,
    make_rainfall_transition_frames,
    make_sorted_transition_frames,
    make_swarm_transition_frames,
    make_swirl_transition_frames,
    make_tornado_transition_frames,
    make_transition_frames,
)
from .utils import fit_letterbox, list_images, parse_size

_TRANSITION_FUNCTIONS = {
    "default": make_transition_frames,
    "swarm": make_swarm_transition_frames,
    "tornado": make_tornado_transition_frames,
    "swirl": make_swirl_transition_frames,
    "drip": make_drip_transition_frames,
    "rain": make_rainfall_transition_frames,
    "sorted": make_sorted_transition_frames,
    "hue-sorted": make_hue_sorted_transition_frames,
}


def _get_cli_overrides(argv):
    """Return argument names explicitly provided on the command line."""
    overrides = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        flag = token.split("=", 1)[0]
        overrides.add(flag[2:].replace("-", "_"))
    return overrides


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Pixel-morph slideshow video generator"
    )
    parser.add_argument("photos_folder", type=Path, help="Folder containing images")
    parser.add_argument("--out", default=None, help="Output video filename")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--seconds-per-transition", type=float, default=2.0)
    parser.add_argument("--hold", type=float, default=0.5)
    parser.add_argument("--pixel-size", type=int, default=4)
    parser.add_argument("--size", type=parse_size, default="1920x1080")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for reproducible results. If not specified, uses current "
            "time for true randomness."
        ),
    )
    parser.add_argument(
        "--easing", default="smoothstep", choices=["linear", "smoothstep", "cubic"]
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument(
        "--transition",
        type=str,
        default="default",
        choices=[
            "default",
            "swarm",
            "tornado",
            "swirl",
            "drip",
            "rain",
            "sorted",
            "hue-sorted",
            "random",
        ],
        help=(
            "Select the type of transition. Use 'random' to randomly choose a "
            "different transition for each frame transition."
        ),
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Audio source: file path or (with --realtime) input device name",
    )
    parser.add_argument(
        "--audio-device",
        type=str,
        default=None,
        help="Realtime reactive input device (default, index:N, or device name)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Play slideshow in realtime instead of writing to file",
    )
    parser.add_argument(
        "--reactive",
        action="store_true",
        help="Enable immersive audio-reactive visuals (requires --realtime and --audio)",
    )
    parser.add_argument(
        "--reactive-style",
        type=str,
        default="dramatic",
        choices=["subtle", "dramatic", "extreme"],
        help="Reactive visual style when --reactive is enabled",
    )
    parser.add_argument(
        "--use-pytorch",
        action="store_true",
        help="Enable PyTorch GPU acceleration (requires PyTorch installation)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the log level for the script",
    )
    parser.add_argument(
        "--web-gui",
        action="store_true",
        help="Enable web-based GUI for realtime control (requires --realtime and Flask)",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=None,
        help="Realtime window width (overrides saved web GUI setting)",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=None,
        help="Realtime window height (overrides saved web GUI setting)",
    )
    parser.add_argument(
        "--window-x",
        type=int,
        default=None,
        help="Realtime window X position (overrides saved web GUI setting)",
    )
    parser.add_argument(
        "--window-y",
        type=int,
        default=None,
        help="Realtime window Y position (overrides saved web GUI setting)",
    )
    return parser


def _apply_persisted_web_gui_settings(args, cli_overrides):
    try:
        from .web_gui import load_persisted_settings

        persisted = load_persisted_settings()
    except Exception:
        persisted = {}

    if isinstance(persisted, dict):
        for key, value in persisted.items():
            if hasattr(args, key) and key not in cli_overrides:
                setattr(args, key, value)


def _apply_window_defaults(args):
    if args.window_width is None:
        args.window_width = args.size[0]
    if args.window_height is None:
        args.window_height = args.size[1]
    if args.window_x is None:
        args.window_x = 80
    if args.window_y is None:
        args.window_y = 80


def _normalize_realtime_audio_args(args, cli_overrides):
    audio_candidate = Path(args.audio) if args.audio else None
    if (
        args.realtime
        and args.audio
        and not (audio_candidate.exists() and audio_candidate.is_file())
        and not args.audio_device
    ):
        # Convenience: allow --audio <device-name> in realtime mode.
        args.audio_device = args.audio
        args.audio = None
        cli_overrides.add("audio_device")


def _validate_args(args):
    if args.web_gui and not args.realtime:
        print("Error: --web-gui can only be used with --realtime mode")
        raise SystemExit(1)

    if args.reactive and not args.realtime:
        print("Error: --reactive can only be used with --realtime mode")
        raise SystemExit(1)

    if args.audio_device and not args.realtime:
        print("Error: --audio-device can only be used with --realtime mode")
        raise SystemExit(1)


def _parse_args(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    raw_argv = sys.argv[1:] if argv is None else argv
    cli_overrides = _get_cli_overrides(raw_argv)
    args._cli_overrides = cli_overrides

    _apply_persisted_web_gui_settings(args, cli_overrides)
    _apply_window_defaults(args)
    _normalize_realtime_audio_args(args, cli_overrides)
    _validate_args(args)

    return args


def _configure_logging(log_level):
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def _configure_random_seed(seed):
    if seed is None:
        generated_seed = int(time.time() * 1000) % (2**32)
        random.seed(generated_seed)
        logging.debug(f"Using time-based random seed: {generated_seed}")
        return

    random.seed(seed)
    logging.debug(f"Using provided random seed: {seed}")


def _configure_pytorch(use_pytorch):
    config.USE_PYTORCH = use_pytorch
    if config.USE_PYTORCH and PYTORCH_AVAILABLE:
        logging.info(f"PyTorch acceleration enabled (using {DEVICE})")
    elif config.USE_PYTORCH and not PYTORCH_AVAILABLE:
        logging.warning("PyTorch acceleration requested but PyTorch is not available")
        config.USE_PYTORCH = False


def _ensure_output_path(args):
    if args.realtime or args.out is not None:
        return

    args.out = (
        f"slideshow_{args.size[0]}x{args.size[1]}_{args.fps}fps_"
        f"{args.pixel_size}px_{args.seconds_per_transition}s_"
        f"{args.hold}hold_{args.easing}_"
        f"{args.transition}_{args.preset}.mp4"
    )
    logging.info(f"No --out specified. Using default filename: {args.out}")


def _load_images(folder, size):
    files = list_images(folder)
    width, height = size

    imgs = []
    for idx, path in enumerate(files):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            logging.warning(f"Could not read {path}, skipping.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = fit_letterbox(img, (width, height))
        logging.debug(f"Processed image {idx + 1}/{len(files)}: {path.name}")
        imgs.append(img)

    if not imgs:
        logging.error("Need at least 2 readable images.")
        raise SystemExit(1)

    imgs.append(imgs[0])
    if len(imgs) < 2:
        logging.error("Need at least 2 readable images.")
        raise SystemExit(1)

    return imgs, len(files)


def _require_render_mode_constraints(args):
    if args.audio_device:
        logging.error("--audio-device is only supported in --realtime mode.")
        raise SystemExit(1)

    if args.audio and not Path(args.audio).exists():
        logging.error(f"Audio file not found: {args.audio}")
        raise SystemExit(1)


def _get_transition_function(transition_name):
    if transition_name == "random":
        transition_fn = get_random_transition_function()
        logging.info(f"Randomly selected transition: {transition_fn.__name__}")
        return transition_fn
    return _TRANSITION_FUNCTIONS[transition_name]


def _get_video_output_path(args):
    if not args.audio:
        return args.out
    temp_video_out = args.out.replace(".mp4", "_temp_no_audio.mp4")
    logging.info(f"Creating temporary video without audio: {temp_video_out}")
    return temp_video_out


def _write_video(imgs, args, output_path):
    ffmpeg_params = ["-pix_fmt", "yuv420p", "-preset", args.preset, "-crf", str(args.crf)]
    writer = imageio.get_writer(
        output_path,
        fps=args.fps,
        codec="libx264",
        quality=None,
        ffmpeg_params=ffmpeg_params,
    )

    try:
        start_time = time.time()
        hold_frames = int(round(args.hold * args.fps))
        logging.debug(f"Writing initial hold frames: {hold_frames}")

        for _ in range(hold_frames // 4):
            writer.append_data(imgs[0])

        for i in range(len(imgs) - 1):
            logging.info(f"Processing transition {i + 1}/{len(imgs) - 1}")
            a, b = imgs[i], imgs[i + 1]
            base_seed = args.seed if args.seed is not None else 0
            pair_seed = base_seed + i * random.randint(1, 10000)

            transition_fn = _get_transition_function(args.transition)
            logging.info(f"Using transition: {transition_fn.__name__}")

            transition_start = time.time()
            for frame in transition_fn(
                a,
                b,
                pixel_size=args.pixel_size,
                fps=args.fps,
                seconds=args.seconds_per_transition,
                hold=0.0,
                ease_name=args.easing,
                seed=pair_seed,
            ):
                writer.append_data(frame)

            elapsed_time = time.time() - transition_start
            transitions_completed = i + 1
            avg_time_per_transition = (time.time() - start_time) / transitions_completed
            remaining_transitions = (len(imgs) - 1) - transitions_completed
            estimated_remaining_time = avg_time_per_transition * remaining_transitions

            logging.info(
                f"Finished transition {i + 1}/{len(imgs) - 1} "
                f"in {elapsed_time:.2f} seconds. Estimated remaining time: "
                f"{estimated_remaining_time:.2f} seconds."
            )

            for _ in range(hold_frames // 4):
                writer.append_data(b)
    finally:
        writer.close()


def _combine_video_with_audio(temp_video_out, audio_in, out_path):
    logging.info(f"Combining video with audio from: {audio_in}")
    try:
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_video_out,
            "-i",
            str(audio_in),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            out_path,
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logging.info(f"Successfully combined video with audio: {out_path}")
            Path(temp_video_out).unlink()
            logging.debug(f"Removed temporary file: {temp_video_out}")
            return

        logging.error(f"Failed to combine video with audio: {result.stderr}")
        logging.info(f"Temporary video saved as: {temp_video_out}")
        raise SystemExit(1)
    except FileNotFoundError:
        logging.error("ffmpeg not found. Please install ffmpeg to use audio functionality.")
        logging.info(f"Video without audio saved as: {temp_video_out}")
        raise SystemExit(1)
    except Exception as exc:
        logging.error(f"Error combining video with audio: {exc}")
        logging.info(f"Video without audio saved as: {temp_video_out}")
        raise SystemExit(1)


def main():
    """Main entry point for the PySlide Morpher application."""
    args = _parse_args()

    _configure_logging(args.log_level)
    _configure_random_seed(args.seed)
    _configure_pytorch(args.use_pytorch)
    _ensure_output_path(args)

    if args.realtime:
        logging.info("Starting Pixel-Morph Slideshow in realtime mode.")
    else:
        logging.info("Starting Pixel-Morph Slideshow generator.")

    imgs, file_count = _load_images(args.photos_folder, args.size)
    logging.info(
        f"Found {file_count} images. Output: {args.size[0]}x{args.size[1]} at {args.fps} fps"
    )

    if args.realtime:
        play_realtime(imgs, args)
        return

    _require_render_mode_constraints(args)
    temp_video_out = _get_video_output_path(args)
    _write_video(imgs, args, temp_video_out)

    if args.audio:
        _combine_video_with_audio(temp_video_out, args.audio, args.out)

    logging.info(f"Done. Wrote: {args.out}")

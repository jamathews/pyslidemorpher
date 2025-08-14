import argparse
import logging
import os
from glob import glob

import cv2
import numpy as np


def load_images_sorted(folder):
    files = sorted(glob(os.path.join(folder, "*.*")))
    images = []
    for f in files:
        logging.debug(f"Loading image: {f}")
        img = cv2.imread(f)
        if img is not None:
            images.append(img)
        else:
            logging.warning(f"Failed to load image: {f}")
    return images


def pad_to_size(img, width, height):
    h, w = img.shape[:2]
    top = (height - h) // 2
    bottom = height - h - top
    left = (width - w) // 2
    right = width - w - left
    logging.debug(f"Padding image from {w}x{h} to {width}x{height}")
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def bubble_sort_frames(img):
    """Return a list of frames showing bubble sort progress."""
    pixels = img.reshape(-1, 3)
    frames = []
    n = len(pixels)
    swapped = True
    logging.debug(f"Starting bubble sort on {n} pixels")
    while swapped:
        swapped = False
        for i in range(n - 1):
            if np.sum(pixels[i]) > np.sum(pixels[i + 1]):
                pixels[i], pixels[i + 1] = np.copy(pixels[i + 1]), np.copy(pixels[i])
                swapped = True
        frame = pixels.reshape(img.shape)
        frames.append(np.copy(frame))
    logging.debug(f"Bubble sort complete, generated {len(frames)} frames")
    return frames


def cross_dissolve(vid1, vid2, steps=15):
    """Generate frames blending vid1 last frame to vid2 first frame."""
    frames = []
    last1 = vid1[-1]
    first2 = vid2[0]
    logging.debug(f"Generating {steps} cross-dissolve frames")
    for i in range(steps):
        alpha = i / (steps - 1)
        blended = cv2.addWeighted(last1, 1 - alpha, first2, alpha, 0)
        frames.append(blended)
    return frames


def main():
    parser = argparse.ArgumentParser(description="Bubble sort photo video generator.")
    parser.add_argument("folder", help="Folder containing photos")
    parser.add_argument("--output", default="output.mp4", help="Output video file name")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Processing images from folder: {args.folder}")
    images = load_images_sorted(args.folder)
    if not images:
        logging.error("No images found.")
        return

    # Get max dimensions
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    logging.info(f"Maximum dimensions: {max_w}x{max_h}")

    # Pad all images
    logging.info("Padding images to uniform size")
    images = [pad_to_size(img, max_w, max_h) for img in images]

    all_frames = []
    for idx, img in enumerate(images):
        logging.info(f"Processing image {idx + 1}/{len(images)}")
        sort_frames = bubble_sort_frames(img)
        rev_frames = list(reversed(sort_frames))
        short_vid = rev_frames + sort_frames
        if idx > 0:
            all_frames.extend(cross_dissolve(all_frames, short_vid))
        all_frames.extend(short_vid)

    # Write to video
    logging.info(f"Writing video to {args.output}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (max_w, max_h))
    for frame in all_frames:
        out.write(frame)
    out.release()
    logging.info(f"Video saved to {args.output}")


if __name__ == "__main__":
    main()

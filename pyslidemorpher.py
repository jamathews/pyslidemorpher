import os
import argparse
from PIL import Image
import numpy as np
import cv2
import math

def jpeg_to_int_list(path):
    img = Image.open(path).convert("RGB")
    data = np.array(img)
    return data, data.flatten().tolist()

def int_list_to_image(int_list, shape):
    arr = np.array(int_list, dtype=np.uint8).reshape(shape)
    return arr

def bubble_sort_frames(int_list, shape, highlight_swaps=False, swap_flash=False):
    frames = []
    frames.append(int_list_to_image(int_list, shape))
    n = len(int_list)
    width = shape[1]

    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if int_list[j] > int_list[j + 1]:
                int_list[j], int_list[j + 1] = int_list[j + 1], int_list[j]
                swapped = True

                if highlight_swaps:
                    frame_img = int_list_to_image(int_list, shape).copy()
                    px_coords = [(j // 3 // width, (j // 3) % width),
                                 ((j+1) // 3 // width, ((j+1) // 3) % width)]
                    for y, x in px_coords:
                        frame_img[y, x] = [255, 0, 0]
                    frames.append(frame_img)

                    if swap_flash:
                        frames.append(int_list_to_image(int_list, shape))
                else:
                    frames.append(int_list_to_image(int_list, shape))

        if not swapped:
            break
    return frames

def transition_frames(start_frame, end_frame, pixels_per_frame):
    start_list = start_frame.flatten().astype(np.int16)
    end_list = end_frame.flatten().astype(np.int16)
    shape = start_frame.shape
    frames = []

    indices = np.arange(len(start_list))
    np.random.shuffle(indices)

    for i in range(0, len(indices), pixels_per_frame):
        batch_indices = indices[i:i + pixels_per_frame]
        for idx in batch_indices:
            if start_list[idx] != end_list[idx]:
                avg_val = (start_list[idx] + end_list[idx]) / 2
                if end_list[idx] > start_list[idx]:
                    start_list[idx] = math.ceil(avg_val)
                else:
                    start_list[idx] = math.floor(avg_val)
        frames.append(int_list_to_image(start_list, shape))

    return frames

def collect_all_frames(folder_path, pixels_per_frame, highlight_swaps, swap_flash):
    all_frames = []
    sorted_frames_per_image = []

    files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg")))

    for file in files:
        path = os.path.join(folder_path, file)
        print(f"Processing {path}...")
        img_data, int_list = jpeg_to_int_list(path)
        frames = bubble_sort_frames(int_list, img_data.shape, highlight_swaps, swap_flash)

        sorted_frames_per_image.append(frames[-1])  # Save final sorted frame

        # Add reverse-forward sequence for this image
        all_frames.extend(reversed(frames))
        all_frames.extend(frames)

    # Add dissolve transitions between sorted frames
    for i in range(len(sorted_frames_per_image) - 1):
        print(f"Creating transition from image {i} to {i+1}...")
        transition = transition_frames(
            sorted_frames_per_image[i],
            sorted_frames_per_image[i + 1],
            pixels_per_frame
        )
        all_frames.extend(transition)

    return all_frames

def write_video(frames, output_path, fps=60):
    if not frames:
        print("No frames to write.")
        return
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bubble sort image video with dissolve transitions.")
    parser.add_argument("folder", help="Folder containing JPEGs")
    parser.add_argument("--pixels-per-frame", type=int, default=None,
                        help="Number of pixels to change per frame during transitions (default: calculated from --transition-duration)")
    parser.add_argument("--transition-duration", type=float, default=2.0,
                        help="Duration in seconds for each dissolve transition (default: 2.0)")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second for output video")
    parser.add_argument("--output", default="bubble_sort_with_dissolve.mp4", help="Output video filename")
    parser.add_argument("--highlight-swaps", action="store_true",
                        help="Highlight swapped pixels in red during bubble sort")
    parser.add_argument("--swap-flash", action="store_true",
                        help="If highlighting swaps, only flash red for one frame instead of keeping red permanently")
    args = parser.parse_args()

    # Determine pixels per frame if not manually specified
    sample_image = next((f for f in os.listdir(args.folder) if f.lower().endswith((".jpg", ".jpeg"))), None)
    if not sample_image:
        raise ValueError("No JPEG images found in the folder.")
    sample_data = Image.open(os.path.join(args.folder, sample_image)).convert("RGB")
    total_pixels = np.prod(np.array(sample_data).shape[:2]) * 3  # Total RGB values
    if args.pixels_per_frame is None:
        frames_needed = args.transition_duration * args.fps
        args.pixels_per_frame = max(1, int(total_pixels / frames_needed))
        print(f"Calculated pixels-per-frame={args.pixels_per_frame} based on transition duration {args.transition_duration}s at {args.fps} FPS.")

    frames = collect_all_frames(args.folder, args.pixels_per_frame, args.highlight_swaps, args.swap_flash)
    write_video(frames, args.output, fps=args.fps)

    print(f"Video saved to {args.output} with {len(frames)} frames.")

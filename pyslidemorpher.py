import os
import argparse
from PIL import Image
import numpy as np
import cv2
import math

def jpeg_to_array(path):
    """Convert JPEG to numpy array"""
    img = Image.open(path).convert("RGB")
    return np.array(img)

def bubble_sort_frames(img_array, highlight_swaps=False, swap_flash=False):
    frames = []
    height, width, _ = img_array.shape
    img_array = img_array.copy()
    frames.append(img_array.copy())
    
    for i in range(height):
        for channel in range(3):
            swapped = False
            for j in range(width - 1):
                for k in range(width - j - 1):
                    if img_array[i, k, channel] > img_array[i, k + 1, channel]:
                        # Swap pixels
                        img_array[i, k, channel], img_array[i, k + 1, channel] = \
                            img_array[i, k + 1, channel], img_array[i, k, channel]
                        swapped = True
                        
                        if highlight_swaps:
                            frame = img_array.copy()
                            # Highlight swapped pixels in red
                            frame[i, k] = [255, 0, 0]
                            frame[i, k + 1] = [255, 0, 0]
                            frames.append(frame)
                            
                            if swap_flash:
                                frames.append(img_array.copy())
                        else:
                            frames.append(img_array.copy())
                            
            if not swapped:
                break
    
    return frames

def transition_frames(start_frame, end_frame, pixels_per_frame):
    frames = []
    current = start_frame.astype(np.int16)
    
    # Create index pairs for all pixels
    height, width, _ = start_frame.shape
    indices = [(i, j) for i in range(height) for j in range(width)]
    np.random.shuffle(indices)
    
    for i in range(0, len(indices), pixels_per_frame):
        batch_indices = indices[i:i + pixels_per_frame]
        for y, x in batch_indices:
            for c in range(3):
                if current[y, x, c] != end_frame[y, x, c]:
                    avg_val = (int(current[y, x, c]) + int(end_frame[y, x, c])) / 2
                    if end_frame[y, x, c] > current[y, x, c]:
                        current[y, x, c] = math.ceil(avg_val)
                    else:
                        current[y, x, c] = math.floor(avg_val)
        
        frames.append(current.astype(np.uint8).copy())
    
    return frames

def collect_all_frames(folder_path, pixels_per_frame, highlight_swaps, swap_flash):
    all_frames = []
    sorted_frames_per_image = []
    
    files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg")))
    
    for file in files:
        path = os.path.join(folder_path, file)
        print(f"Processing {path}...")
        img_array = jpeg_to_array(path)
        frames = bubble_sort_frames(img_array, highlight_swaps, swap_flash)
        
        sorted_frames_per_image.append(frames[-1])
        
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
    total_pixels = np.prod(np.array(sample_data).shape[:2])  # Total pixels (not RGB values)
    if args.pixels_per_frame is None:
        frames_needed = args.transition_duration * args.fps
        args.pixels_per_frame = max(1, int(total_pixels / frames_needed))
        print(f"Calculated pixels-per-frame={args.pixels_per_frame} based on transition duration {args.transition_duration}s at {args.fps} FPS.")

    frames = collect_all_frames(args.folder, args.pixels_per_frame, args.highlight_swaps, args.swap_flash)
    write_video(frames, args.output, fps=args.fps)
    
    print(f"Video saved to {args.output} with {len(frames)} frames.")
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
    parser.add_argument("--output", default=None, help="Output video file name")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    parser.add_argument("--sort-algorithm",
                        choices=["bubble", "quick", "counting", "radix", "bucket", "col", "hue", "corner", "diag"],
                        default="diag",
                        help="Sorting algorithm to use (default: quick)")
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

    # Get max dimensions for final output
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    logging.info(f"Maximum dimensions: {max_w}x{max_h}")

    sort_functions = {
        "bubble": bubble_sort_frames,
        "quick": quicksort_frames,
        "counting": counting_sort_frames,
        "radix": radix_sort_frames,
        "bucket": bucket_sort_frames_hue,
        "col": col_sort_frames,
        "hue": sort_columns_by_hue,
        "corner": corner_sort,
        "diag": diag_sort,
    }
    sort_function = sort_functions[args.sort_algorithm]

    all_frames = []
    for idx, img in enumerate(images):
        logging.info(f"Processing image {idx + 1}/{len(images)}")

        # Create sorting animation at original image size
        sort_frames = sort_function(img)

        # Add 5 seconds of unmodified image (24 fps * 5 seconds = 120 frames)
        unmodified_frame = sort_frames[0]
        static_frames = []
        for _ in range(len(sort_frames)):
            static_frames.append(unmodified_frame)

        sorted_frame = sort_frames[-1]
        sorted_static_frames = []
        for _ in range(len(sort_frames)//2):
            sorted_static_frames.append(sorted_frame)
        rev_frames = list(reversed(sort_frames))
        short_vid = sorted_static_frames + rev_frames + static_frames + sort_frames + sorted_static_frames

        # Pad the frames of this video segment to match max dimensions
        padded_vid = [pad_to_size(frame, max_w, max_h) for frame in short_vid]

        if idx > 0:
            # Create cross-dissolve between padded videos
            transition = cross_diag_dissolve(all_frames, padded_vid, steps=len(sort_frames))
            all_frames.extend(transition)

        all_frames.extend(padded_vid)

    # Write to video
    if not args.output:
        args.output = f"{args.sort_algorithm}.mp4"
    logging.info(f"Writing video to {args.output}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (max_w, max_h))
    for frame in all_frames:
        out.write(frame)
    out.release()
    logging.info(f"Video saved to {args.output}")


def quicksort_frames(img):
    """Return a list of frames showing quicksort progress."""
    pixels = img.reshape(-1, 3)
    frames = []
    n = len(pixels)
    logging.debug(f"Starting quicksort on {n} pixels")

    def partition(low, high):
        pivot_idx = high
        pivot = np.sum(pixels[pivot_idx])
        i = low - 1

        for j in range(low, high):
            if np.sum(pixels[j]) <= pivot:
                i += 1
                pixels[i], pixels[j] = np.copy(pixels[j]), np.copy(pixels[i])

        pixels[i + 1], pixels[high] = np.copy(pixels[high]), np.copy(pixels[i + 1])

        # Add frame after each partition
        frame = pixels.reshape(img.shape)
        frames.append(np.copy(frame))

        return i + 1

    def quicksort(low, high):
        if low < high:
            pi = partition(low, high)
            quicksort(low, pi - 1)
            quicksort(pi + 1, high)

    quicksort(0, n - 1)
    logging.debug(f"Quicksort complete, generated {len(frames)} frames")
    return frames


def counting_sort_frames(img):
    """Return a list of frames showing counting sort progress."""
    pixels = img.reshape(-1, 3)
    frames = []
    n = len(pixels)
    logging.debug(f"Starting counting sort on {n} pixels")

    # Convert to intensity values
    intensities = np.sum(pixels, axis=1)
    max_val = int(np.max(intensities))
    min_val = int(np.min(intensities))
    range_val = max_val - min_val + 1

    # Initialize counting array
    count = np.zeros(range_val, dtype=int)

    # Count occurrences
    for i in range(n):
        count[int(intensities[i]) - min_val] += 1

    # Calculate cumulative count
    for i in range(1, len(count)):
        count[i] += count[i - 1]

    # Build output array
    output = np.zeros_like(pixels)
    for i in range(n - 1, -1, -1):
        intensity = int(intensities[i]) - min_val
        index = count[intensity] - 1
        output[index] = pixels[i]
        count[intensity] -= 1

        if i % (n // 50 + 1) == 0:  # Capture frames periodically
            frame = output.reshape(img.shape)
            frames.append(np.copy(frame))

    logging.debug(f"Counting sort complete, generated {len(frames)} frames")
    return frames


def radix_sort_frames(img):
    """Return a list of frames showing radix sort progress."""
    pixels = img.reshape(-1, 3)
    frames = []
    n = len(pixels)
    logging.debug(f"Starting radix sort on {n} pixels")

    def counting_sort_exp(exp):
        output = np.zeros_like(pixels)
        count = np.zeros(256, dtype=int)  # 256 possible values for each byte

        # Calculate intensities
        intensities = np.sum(pixels, axis=1)

        # Count occurrences
        for i in range(n):
            idx = int((intensities[i] / exp) % 256)
            count[idx] += 1

        # Calculate cumulative count
        for i in range(1, 256):
            count[i] += count[i - 1]

        # Build output array
        for i in range(n - 1, -1, -1):
            idx = int((intensities[i] / exp) % 256)
            output[count[idx] - 1] = pixels[i]
            count[idx] -= 1

        pixels[:] = output[:]
        frame = pixels.reshape(img.shape)
        frames.append(np.copy(frame))

    # Perform counting sort for each byte
    max_val = int(np.max(np.sum(pixels, axis=1)))
    exp = 1
    while max_val // exp > 0:
        counting_sort_exp(exp)
        exp *= 256

    logging.debug(f"Radix sort complete, generated {len(frames)} frames")
    return frames


def bucket_sort_frames(img):
    """Return a list of frames showing bucket sort progress."""
    pixels = img.reshape(-1, 3)
    frames = []
    n = len(pixels)
    logging.debug(f"Starting bucket sort on {n} pixels")

    # Calculate intensities
    intensities = np.sum(pixels, axis=1)
    max_val = np.max(intensities)
    min_val = np.min(intensities)

    # Create buckets
    # num_buckets = min(n // 50, 100)  # Reasonable number of buckets
    num_buckets = 100000
    buckets = [[] for _ in range(num_buckets)]

    # Distribute elements into buckets
    for i in range(n):
        bucket_idx = int((intensities[i] - min_val) * (num_buckets - 1) / (max_val - min_val))
        buckets[bucket_idx].append(pixels[i])

    # Sort buckets and collect elements
    index = 0
    for bucket in buckets:
        if bucket:
            # Sort bucket using numpy's sort
            bucket_intensities = np.array([np.sum(x) for x in bucket])
            sorted_indices = np.argsort(bucket_intensities)
            bucket_array = np.array(bucket)
            sorted_bucket = bucket_array[sorted_indices]

            # Place sorted elements back
            for pixel in sorted_bucket:
                pixels[index] = pixel
                index += 1

                # Capture frame periodically
                if index % (n // 50 + 1) == 0:
                    frame = pixels.reshape(img.shape)
                    frames.append(np.copy(frame))

    # Ensure we have at least the final frame
    frame = pixels.reshape(img.shape)
    frames.append(np.copy(frame))

    logging.debug(f"Bucket sort complete, generated {len(frames)} frames")
    return frames


def bucket_sort_frames_luminosity(img):
    """Return a list of frames showing bucket sort progress, sorting by perceived luminosity."""
    pixels = img.reshape(-1, 3)
    frames = []
    n = len(pixels)
    logging.debug(f"Starting luminosity-based bucket sort on {n} pixels")

    # Create a copy of original pixels to preserve exact values
    original_pixels = np.copy(pixels)

    # Calculate luminosity using standard coefficients
    # Using BT.709 coefficients: R: 0.2126, G: 0.7152, B: 0.0722
    luminosity = (original_pixels[..., 2] * 0.2126 +  # Red channel
                  original_pixels[..., 1] * 0.7152 +  # Green channel
                  original_pixels[..., 0] * 0.0722)  # Blue channel (OpenCV uses BGR)

    max_val = np.max(luminosity)
    min_val = np.min(luminosity)

    # Create buckets
    num_buckets = min(n // 50, 100)  # Reasonable number of buckets
    buckets = [[] for _ in range(num_buckets)]

    # Store indices and luminosity values for sorting
    pixel_data = [(i, luminosity[i]) for i in range(n)]

    # Distribute elements into buckets
    for i, lum in pixel_data:
        bucket_idx = int((lum - min_val) * (num_buckets - 1) / (max_val - min_val))
        buckets[bucket_idx].append((i, lum))

    # Sort buckets and collect elements
    output_index = 0
    output_pixels = np.zeros_like(pixels)

    for bucket in buckets:
        if bucket:
            # Sort bucket by luminosity while keeping track of original indices
            sorted_bucket = sorted(bucket, key=lambda x: x[1])

            # Place original pixels in sorted order
            for orig_idx, _ in sorted_bucket:
                output_pixels[output_index] = original_pixels[orig_idx]
                output_index += 1

                # Capture frame periodically
                if output_index % (n // 50 + 1) == 0:
                    frame = output_pixels.reshape(img.shape)
                    frames.append(np.copy(frame))

    # Ensure we have at least the final frame
    frame = output_pixels.reshape(img.shape)
    frames.append(np.copy(frame))

    logging.debug(f"Luminosity-based bucket sort complete, generated {len(frames)} frames")
    return frames


def bucket_sort_frames_hue(img):
    """Return a list of frames showing bucket sort progress, sorting by hue."""
    pixels = img.reshape(-1, 3)
    frames = []
    n = len(pixels)
    logging.debug(f"Starting hue-based bucket sort on {n} pixels")

    # Create a copy of original pixels to preserve exact values
    original_pixels = np.copy(pixels)

    # Reshape for OpenCV color conversion (needs 2D image)
    pixels_2d = original_pixels.reshape(1, -1, 3)

    # Convert to HSV color space
    hsv = cv2.cvtColor(pixels_2d, cv2.COLOR_BGR2HSV)
    hue = hsv[0, :, 0]  # Extract hue channel (0-180 in OpenCV)

    # Create buckets (180 buckets for OpenCV's hue range)
    num_buckets = 180
    buckets = [[] for _ in range(num_buckets)]

    # Store indices and hue values for sorting
    pixel_data = [(i, hue[i]) for i in range(n)]

    # Distribute elements into buckets
    for i, h in pixel_data:
        bucket_idx = int(h)
        buckets[bucket_idx].append((i, h))

    # Sort buckets and collect elements
    output_index = 0
    output_pixels = np.zeros_like(pixels)

    for bucket in buckets:
        if bucket:
            # Get original indices from current bucket
            bucket_indices = [i for i, _ in bucket]

            # Get HSV values for pixels in this bucket
            bucket_pixels_hsv = hsv[0, bucket_indices]

            # Create sort keys based on saturation and value
            sort_keys = [(sat, val, idx) for idx, (sat, val) in
                         enumerate(zip(bucket_pixels_hsv[:, 1], bucket_pixels_hsv[:, 2]))]

            # Sort by saturation and value, then get the original indices in sorted order
            sorted_bucket_indices = [bucket_indices[x[2]] for x in sorted(sort_keys)]

            # Place original pixels in sorted order
            for orig_idx in sorted_bucket_indices:
                output_pixels[output_index] = original_pixels[orig_idx]
                output_index += 1

                # Capture frame periodically
                if output_index % (n // 50 + 1) == 0:
                    frame = output_pixels.reshape(img.shape)
                    frames.append(np.copy(frame))

    # Ensure we have at least the final frame
    frame = output_pixels.reshape(img.shape)
    frames.append(np.copy(frame))

    logging.debug(f"Hue-based bucket sort complete, generated {len(frames)} frames")
    return frames


def col_sort_frames(img):
    """Return a list of frames showing column-wise hue sorting progress using mergesort.
    
    Args:
        img: Input image in BGR format (OpenCV default)
        
    Returns:
        List of frames showing the sorting progress
    """
    frames = []
    result = np.copy(img)
    height, width = img.shape[:2]
    logging.debug(f"Starting column-wise mergesort on {width} columns")

    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def merge(arr_bgr, arr_hsv, left, mid, right, col):
        """Merge two sorted subarrays and update frames."""
        # Create temporary arrays for BGRs and HSVs
        L_bgr = arr_bgr[left:mid]
        R_bgr = arr_bgr[mid:right]
        L_hsv = arr_hsv[left:mid]
        R_hsv = arr_hsv[mid:right]

        i = j = 0
        k = left

        while i < len(L_hsv) and j < len(R_hsv):
            # Compare using hue, then saturation, then value
            L_key = (L_hsv[i][0], L_hsv[i][1], L_hsv[i][2])
            R_key = (R_hsv[j][0], R_hsv[j][1], R_hsv[j][2])

            if L_key <= R_key:
                arr_bgr[k] = L_bgr[i]
                arr_hsv[k] = L_hsv[i]
                i += 1
            else:
                arr_bgr[k] = R_bgr[j]
                arr_hsv[k] = R_hsv[j]
                j += 1
            k += 1

            # Update the result image and capture frame
            result[:, col] = arr_bgr
            frames.append(np.copy(result))

        # Copy remaining elements
        while i < len(L_hsv):
            arr_bgr[k] = L_bgr[i]
            arr_hsv[k] = L_hsv[i]
            k += 1
            i += 1
            result[:, col] = arr_bgr
            frames.append(np.copy(result))

        while j < len(R_hsv):
            arr_bgr[k] = R_bgr[j]
            arr_hsv[k] = R_hsv[j]
            k += 1
            j += 1
            result[:, col] = arr_bgr
            frames.append(np.copy(result))

    def mergesort(arr_bgr, arr_hsv, left, right, col):
        """Recursive mergesort implementation."""
        if right - left > 1:
            mid = (left + right) // 2
            mergesort(arr_bgr, arr_hsv, left, mid, col)
            mergesort(arr_bgr, arr_hsv, mid, right, col)
            merge(arr_bgr, arr_hsv, left, mid, right, col)

    # Process each column
    for col in range(width):
        # Get the column's pixels
        column_bgr = result[:, col].copy()
        column_hsv = hsv[:, col].copy()

        # Sort the column using mergesort
        mergesort(column_bgr, column_hsv, 0, height, col)

        # Ensure the final state of the column is captured
        result[:, col] = column_bgr
        frames.append(np.copy(result))

    logging.debug(f"Column-wise mergesort complete, generated {len(frames)} frames")
    return frames


def sort_columns_by_hue(img):
    """Generate frames showing columns sorting by hue in random order.
    
    Args:
        img: Input image in BGR format (OpenCV default)
        
    Returns:
        List of frames for a 5-second animation at 24 fps
    """
    target_frames = 2 * 24

    # Create initial frame
    result = np.copy(img)
    frames = [np.copy(result)]
    height, width = img.shape[:2]

    # Convert to HSV for hue-based sorting
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Create list of columns and shuffle them
    columns = list(range(width))
    np.random.shuffle(columns)

    # Process one column at a time
    columns_per_frame = max(1, width // target_frames)

    for col_start in range(0, width, columns_per_frame):
        col_end = min(col_start + columns_per_frame, width)

        # Process each column in this shuffled batch
        for col_idx in range(col_start, col_end):
            col = columns[col_idx]  # Use shuffled column index

            # Get column pixels in both BGR and HSV
            column_bgr = result[:, col].copy()
            column_hsv = hsv[:, col].copy()

            # Create sorting indices based on HSV values
            sort_keys = [(v, s, h, i) for i, (h, s, v) in enumerate(column_hsv)]
            sorted_indices = [i for h, s, v, i in reversed(sorted(sort_keys))]

            # Apply sorting to the column
            result[:, col] = column_bgr[sorted_indices]

        # Add frame after processing this batch of columns
        frames.append(np.copy(result))

    # If we have too few frames, interpolate
    if len(frames) < target_frames:
        final_frames = []
        for i in range(target_frames):
            # Calculate fractional index into original frames
            idx = i * (len(frames) - 1) / (target_frames - 1)
            frame1_idx = int(idx)
            frame2_idx = min(frame1_idx + 1, len(frames) - 1)
            frac = idx - frame1_idx

            # Interpolate between frames
            frame = cv2.addWeighted(
                frames[frame1_idx], 1 - frac,
                frames[frame2_idx], frac,
                0
            )
            final_frames.append(frame)
        frames = final_frames

    return frames


def cross_col_dissolve(vid1, vid2, steps=15):
    """Generate frames blending vid1 last frame to vid2 first frame by randomly swapping columns.
    
    Args:
        vid1: First video (list of frames)
        vid2: Second video (list of frames)
        steps: Number of transition frames to generate
        
    Returns:
        List of frames showing the column-swap transition
    """
    frames = []
    last1 = vid1[-1].copy()
    first2 = vid2[0].copy()
    height, width = last1.shape[:2]

    # Create list of column indices and shuffle
    cols = list(range(width))
    np.random.shuffle(cols)

    # Calculate how many columns to swap per frame
    cols_per_step = max(1, width // steps)

    # Generate transition frames
    result = last1.copy()
    for i in range(steps):
        # Calculate range of columns to swap in this step
        start_idx = i * cols_per_step
        end_idx = min(start_idx + cols_per_step, width)

        # Swap the columns
        for col_idx in cols[start_idx:end_idx]:
            result[:, col_idx] = first2[:, col_idx]

        frames.append(result.copy())

    return frames


def corner_sort(img):
    """Generate frames by first sorting columns by hue, then sorting the resulting frames by rows.
    
    Args:
        img: Input image in BGR format (OpenCV default)
        
    Returns:
        List of frames showing the progressive sorting effect
    """
    target_frames = 2 * 24  # 2 seconds at 24fps

    # Phase 1: Get frames from column-wise sorting
    result = np.copy(img)
    frames = [np.copy(result)]
    height, width = img.shape[:2]
    
    # Convert to HSV for hue-based sorting
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Sort columns first
    columns = list(range(width))
    np.random.shuffle(columns)
    columns_per_frame = max(1, width // target_frames)
    
    for col_start in range(0, width, columns_per_frame):
        col_end = min(col_start + columns_per_frame, width)
        
        for col_idx in range(col_start, col_end):
            col = columns[col_idx]
            
            column_bgr = result[:, col].copy()
            column_hsv = hsv[:, col].copy()
            
            sort_keys = [(v, s, h, i) for i, (h, s, v) in enumerate(column_hsv)]
            sorted_indices = [i for h, s, v, i in reversed(sorted(sort_keys))]
            
            result[:, col] = column_bgr[sorted_indices]
        
        frames.append(np.copy(result))

    # Phase 2: Sort the frames by rows
    sorted_frames = []
    last_frame = frames[-1]  # Keep the final frame from column sorting
    
    # For each row, create a new frame by progressively replacing rows
    # from top to bottom with the corresponding row from the last frame
    height = last_frame.shape[0]
    for row in range(height):
        new_frame = np.copy(frames[0])  # Start with first frame
        # Replace all rows up to current row with rows from final frame
        new_frame[:row+1] = last_frame[:row+1]
        sorted_frames.append(new_frame)
    
    # Combine both phases
    all_frames = frames + sorted_frames
    
    # Interpolate if needed
    target_total_frames = target_frames * 2
    if len(all_frames) < target_total_frames:
        final_frames = []
        for i in range(target_total_frames):
            idx = i * (len(all_frames) - 1) / (target_total_frames - 1)
            frame1_idx = int(idx)
            frame2_idx = min(frame1_idx + 1, len(all_frames) - 1)
            frac = idx - frame1_idx
            
            frame = cv2.addWeighted(
                all_frames[frame1_idx], 1 - frac,
                all_frames[frame2_idx], frac,
                0
            )
            final_frames.append(frame)
        all_frames = final_frames
    
    return all_frames

def diag_sort(img):
    """Generate frames showing diagonal sorting by hue.
    
    Args:
        img: Input image in BGR format (OpenCV default)
        
    Returns:
        List of frames showing diagonal sorting animation
    """
    target_frames = 2 * 24  # 2 seconds at 24fps
    frames = [np.copy(img)]
    result = np.copy(img)
    height, width = img.shape[:2]
    
    # Convert to HSV for hue-based sorting
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Get all diagonal indices
    total_diags = width + height - 1
    
    # Create list of diagonals and shuffle them
    diags = list(range(total_diags))
    np.random.shuffle(diags)
    
    diags_per_frame = max(1, total_diags // target_frames)
    
    for diag_start in range(0, total_diags, diags_per_frame):
        diag_end = min(diag_start + diags_per_frame, total_diags)
        
        # Process each diagonal in this batch
        for diag_idx in range(diag_start, diag_end):
            diag = diags[diag_idx]
            
            # Calculate points on this diagonal
            diag_points = []
            # Start point of diagonal
            x = max(0, diag - height + 1)
            y = max(0, height - 1 - diag)
            
            # Collect all points on the diagonal
            while x < width and y < height:
                if 0 <= x < width and 0 <= y < height:
                    diag_points.append((y, x))
                x += 1
                y += 1
            
            if diag_points:
                # Get diagonal pixels in both BGR and HSV
                diag_bgr = np.array([result[y, x] for y, x in diag_points])
                diag_hsv = np.array([hsv[y, x] for y, x in diag_points])
                
                # Create sorting indices based on HSV values
                sort_keys = [(v, s, h, i) for i, (h, s, v) in enumerate(diag_hsv)]
                sorted_indices = [i for h, s, v, i in reversed(sorted(sort_keys))]
                
                # Apply sorting to the diagonal
                sorted_bgr = diag_bgr[sorted_indices]
                for idx, (y, x) in enumerate(diag_points):
                    result[y, x] = sorted_bgr[idx]
        
        frames.append(np.copy(result))
    
    # Interpolate if we have too few frames
    if len(frames) < target_frames:
        final_frames = []
        for i in range(target_frames):
            idx = i * (len(frames) - 1) / (target_frames - 1)
            frame1_idx = int(idx)
            frame2_idx = min(frame1_idx + 1, len(frames) - 1)
            frac = idx - frame1_idx
            
            frame = cv2.addWeighted(
                frames[frame1_idx], 1 - frac,
                frames[frame2_idx], frac,
                0
            )
            final_frames.append(frame)
        frames = final_frames
    
    return frames


def cross_diag_dissolve(vid1, vid2, steps=15):
    """Generate frames blending vid1 last frame to vid2 first frame by randomly swapping diagonals.

    Args:
        vid1: First video (list of frames)
        vid2: Second video (list of frames)
        steps: Number of transition frames to generate

    Returns:
        List of frames showing the diagonal-swap transition
    """
    frames = []
    last1 = vid1[-1].copy()
    first2 = vid2[0].copy()
    height, width = last1.shape[:2]

    # Calculate total number of diagonals
    total_diags = width + height - 1

    # Create list of diagonal indices and shuffle
    diags = list(range(total_diags))
    np.random.shuffle(diags)

    # Calculate how many diagonals to swap per frame
    diags_per_step = max(1, total_diags // steps)

    # Generate transition frames
    result = last1.copy()
    for i in range(steps):
        # Calculate range of diagonals to swap in this step
        start_idx = i * diags_per_step
        end_idx = min(start_idx + diags_per_step, total_diags)

        # Swap the diagonals
        for diag_idx in diags[start_idx:end_idx]:
            # Calculate points on this diagonal
            x = max(0, diag_idx - height + 1)
            y = max(0, height - 1 - diag_idx)

            # Copy all points on the diagonal
            while x < width and y < height:
                if 0 <= x < width and 0 <= y < height:
                    result[y, x] = first2[y, x]
                x += 1
                y += 1

        frames.append(result.copy())

    return frames

if __name__ == "__main__":
    main()
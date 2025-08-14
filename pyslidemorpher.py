import os
import argparse
from PIL import Image
import numpy as np
import cv2
import math
import multiprocessing as mp
from tempfile import mkdtemp
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor
import tempfile
import logging
import subprocess
import atexit
import signal

# Initialize logger at module level
logger = logging.getLogger()

# Global ProcessPoolExecutor instance
_process_pool = None

def cleanup_processes():
    """Cleanup multiprocessing resources properly"""
    global _process_pool
    if _process_pool is not None:
        _process_pool.shutdown(wait=True)
        _process_pool = None
    
    # Clean up any remaining child processes
    for process in mp.active_children():
        process.terminate()
        process.join()

# Register cleanup function
atexit.register(cleanup_processes)

# Handle SIGTERM gracefully
signal.signal(signal.SIGTERM, lambda signo, frame: cleanup_processes())

def setup_logging(log_level):
    """Configure logging based on a specified log level"""
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    global logger
    logger = logging.getLogger()
    return logger

def get_max_dimensions(folder_path):
    """Scan through all images to find maximum dimensions"""
    max_height = 0
    max_width = 0
    
    logger.debug("Starting to scan images for maximum dimensions")
    image_count = 0
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg')):
            path = os.path.join(folder_path, file)
            with Image.open(path) as img:
                width, height = img.size
                max_width = max(max_width, width)
                max_height = max(max_height, height)
                image_count += 1
                logger.debug(f"Processed image {file}: {width}x{height}")
    
    logger.debug(f"Finished scanning {image_count} images. Max dimensions: {max_width}x{max_height}")
    return max_height, max_width

def center_pad_image(img_array, target_height, target_width):
    """Center the image in a black canvas of target size"""
    height, width = img_array.shape[:2]
    
    pad_height = target_height - height
    pad_width = target_width - width
    
    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad
    
    padded_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    padded_img[top_pad:top_pad+height, left_pad:left_pad+width] = img_array
    
    return padded_img

def jpeg_to_array(path, target_height, target_width):
    """Convert JPEG to numpy array with specified dimensions"""
    img = Image.open(path).convert("RGB")
    img_array = np.array(img)
    return center_pad_image(img_array, target_height, target_width)

# def bubble_sort_frames(img_array, highlight_swaps=False, swap_flash=False):
#     frames = []
#     height, width, _ = img_array.shape
#     img_array = img_array.copy()
#     frames.append(img_array.copy())
#
#     logger.debug(f"Starting bubble sort on image of size {width}x{height}")
#     total_swaps = 0
#
#     for i in range(height):
#         logger.debug(f"Processing row {i}/{height}")
#         for channel in range(3):
#             logger.debug(f"Processing channel {channel} in row {i}")
#             swapped = False
#             row_swaps = 0
#
#             for j in range(width - 1):
#                 for k in range(width - j - 1):
#                     if img_array[i, k, channel] > img_array[i, k + 1, channel]:
#                         img_array[i, k, channel], img_array[i, k + 1, channel] = \
#                             img_array[i, k + 1, channel], img_array[i, k, channel]
#                         swapped = True
#                         row_swaps += 1
#
#                         if highlight_swaps:
#                             frame = img_array.copy()
#                             frame[i, k] = [255, 0, 0]
#                             frame[i, k + 1] = [255, 0, 0]
#                             frames.append(frame)
#
#                             if swap_flash:
#                                 frames.append(img_array.copy())
#                         else:
#                             frames.append(img_array.copy())
#
#             if not swapped:
#                 logger.debug(f"Row {i}, channel {channel} already sorted")
#                 break
#             else:
#                 logger.debug(f"Row {i}, channel {channel} completed with {row_swaps} swaps")
#                 total_swaps += row_swaps
#
#     logger.debug(f"Bubble sort completed with {total_swaps} total swaps and {len(frames)} frames generated")
#     return frames

def write_frames_to_video(frames, output_path, fps=60):
    """Write frames to a video file"""
    if not frames:
        logger.warning("No frames to write to video")
        return None
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    logger.debug(f"Writing {len(frames)} frames to video at {fps} FPS")
    for i, frame in enumerate(frames):
        if i % 100 == 0:  # Log progress every 100 frames
            logger.debug(f"Writing frame {i}/{len(frames)}")
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()
    logger.debug(f"Video writing completed: {output_path}")
    return output_path

def process_single_image(args):
    """Process a single image and return path to temporary video file"""
    image_path, temp_dir, max_height, max_width, highlight_swaps, swap_flash, fps, frame_interval, log_level = args

    # Setup logging with the specified level
    logger = setup_logging(log_level)
    temp_video = None

    try:
        # Create unique temporary file for this process
        temp_video = os.path.join(temp_dir, f"{uuid.uuid4()}.mp4")
        
        # Process image
        img_array = jpeg_to_array(image_path, max_height, max_width)
        
        # Generate and save frames to disk
        frames_dir = None
        try:
            frames_dir, frame_count, last_frame = bubble_sort_frames_buffered(
                img_array, temp_dir, highlight_swaps, swap_flash, frame_interval
            )
            
            # Create forward-reverse sequence
            forward_video = os.path.join(temp_dir, f"forward_{uuid.uuid4()}.mp4")
            reverse_video = os.path.join(temp_dir, f"reverse_{uuid.uuid4()}.mp4")
            
            try:
                write_frames_to_video_from_disk(frames_dir, frame_count, forward_video, fps)
                write_frames_to_video_from_disk(frames_dir, frame_count, reverse_video, fps, reverse=True)
                
                # Combine forward and reverse videos
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write(f"file '{forward_video}'\n")
                    f.write(f"file '{reverse_video}'\n")
                    temp_list = f.name
                
                try:
                    subprocess.run([
                        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                        '-i', temp_list, '-c', 'copy', temp_video
                    ], check=True)
                finally:
                    if os.path.exists(temp_list):
                        os.unlink(temp_list)
                return {
                    'video_path': temp_video,
                    'last_frame': last_frame,
                    'original_path': image_path
                }
            finally:
                pass
            #     # Cleanup intermediate videos
            #     for video in [forward_video, reverse_video]:
            #         if os.path.exists(video):
            #             os.unlink(video)
        finally:
            pass
        #     # Cleanup frames directory
        #     if frames_dir and os.path.exists(frames_dir):
        #         shutil.rmtree(frames_dir)
        

    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        if temp_video and os.path.exists(temp_video):
            os.unlink(temp_video)
        return None

def create_transition_video(start_frame, end_frame, temp_dir, pixels_per_frame, fps):
    """Create a video for transition between two frames"""
    frames = []
    current = start_frame.astype(np.int16)
    
    height, width, _ = start_frame.shape
    indices = [(i, j) for i in range(height) for j in range(width)]
    np.random.shuffle(indices)
    
    total_pixels = len(indices)
    frames_count = (total_pixels + pixels_per_frame - 1) // pixels_per_frame
    logger.debug(f"Creating transition video with {frames_count} frames ({total_pixels} pixels, {pixels_per_frame} per frame)")
    
    for i in range(0, len(indices), pixels_per_frame):
        batch_indices = indices[i:i + pixels_per_frame]
        frame_number = i // pixels_per_frame + 1
        pixels_processed = min(i + pixels_per_frame, total_pixels)
        
        logger.debug(f"Processing transition frame {frame_number}/{frames_count} ({pixels_processed}/{total_pixels} pixels)")
        
        for y, x in batch_indices:
            for c in range(3):
                if current[y, x, c] != end_frame[y, x, c]:
                    avg_val = (int(current[y, x, c]) + int(end_frame[y, x, c])) / 2
                    current[y, x, c] = math.ceil(avg_val) if end_frame[y, x, c] > current[y, x, c] else math.floor(avg_val)
        
        frames.append(current.astype(np.uint8).copy())
    
    # Write transition to temporary file
    temp_video = os.path.join(temp_dir, f"transition_{uuid.uuid4()}.mp4")
    logger.debug(f"Writing transition video with {len(frames)} frames")
    write_frames_to_video(frames, temp_video, fps)
    return temp_video

def combine_videos(video_list, output_path):
    """Combine multiple video files into one"""
    import subprocess
    
    # Create file list for ffmpeg
    list_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    for video_path in video_list:
        list_file.write(f"file '{video_path}'\n")
    list_file.close()
    
    # Combine videos using ffmpeg
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', list_file.name,
        '-c', 'copy',
        output_path
    ]
    
    subprocess.run(cmd, check=True)
    os.unlink(list_file.name)

def main():
    parser = argparse.ArgumentParser(description="Parallel bubble sort image video with dissolve transitions.")
    parser.add_argument("folder", help="Folder containing JPEGs")
    parser.add_argument("--pixels-per-frame", type=int, default=None,
                        help="Number of pixels to change per frame during transitions")
    parser.add_argument("--transition-duration", type=float, default=2.0,
                        help="Duration in seconds for each dissolve transition (default: 2.0)")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second for output video")
    parser.add_argument("--output", default="bubble_sort_with_dissolve.mp4", help="Output video filename")
    parser.add_argument("--highlight-swaps", action="store_true",
                        help="Highlight swapped pixels in red during bubble sort")
    parser.add_argument("--swap-flash", action="store_true",
                        help="If highlighting swaps, only flash red for one frame")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: CPU count - 1)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO)")
    parser.add_argument("--frame-interval", type=int, default=100,
                        help="Number of swaps between captured frames (default: 100)")
    args = parser.parse_args()

    # Setup logging with the specified level
    logger = setup_logging(args.log_level)

    try:
        # Create a temporary directory for intermediate files
        temp_dir = mkdtemp()
        
        # Get maximum dimensions
        max_height, max_width = get_max_dimensions(args.folder)
        logger.info(f"Maximum dimensions: {max_width}x{max_height}")
        
        # Calculate pixels per frame if not specified
        total_pixels = max_height * max_width
        if args.pixels_per_frame is None:
            frames_needed = args.transition_duration * args.fps
            args.pixels_per_frame = max(1, int(total_pixels / frames_needed))
            logger.debug(f"Calculated pixels per frame: {args.pixels_per_frame}")
        
        # Prepare list of images to process
        image_files = sorted(
            os.path.join(args.folder, f) 
            for f in os.listdir(args.folder) 
            if f.lower().endswith(('.jpg', '.jpeg'))
        )
        logger.debug(f"Found {len(image_files)} images to process")
        
        # Prepare arguments for parallel processing
        process_args = [
            (path, temp_dir, max_height, max_width, args.highlight_swaps, 
             args.swap_flash, args.fps, args.frame_interval, args.log_level)
            for path in image_files
        ]
        
        logger.info("Processing images in parallel...")
        
        # Initialize the global process pool
        global _process_pool
        if not args.workers:
            max_workers = max(1, mp.cpu_count() // 2)
        else:
            max_workers = args.workers
        logger.debug(f"Using {max_workers} worker processes")
        
        _process_pool = ProcessPoolExecutor(max_workers=max_workers)
        
        # Use the global process pool
        results = []
        for i, result in enumerate(_process_pool.map(process_single_image, process_args)):
            logger.debug(f"Completed processing image {i+1}/{len(process_args)}")
            results.append(result)
            
        # Filter out any failed processes
        results = [r for r in results if r is not None]
        logger.debug(f"Successfully processed {len(results)} images")
        
        # Create transitions between images
        logger.info("Creating transitions...")
        video_sequence = []
        for i in range(len(results)):
            logger.debug(f"Adding video segment {i+1}/{len(results)}")
            video_sequence.append(results[i]['video_path'])
            if i < len(results) - 1:
                logger.debug(f"Creating transition between segments {i+1} and {i+2}")
                transition = create_transition_video(
                    results[i]['last_frame'],
                    results[i + 1]['last_frame'],
                    temp_dir,
                    args.pixels_per_frame,
                    args.fps
                )
                video_sequence.append(transition)
        
        # Combine all videos
        logger.info(f"Combining {len(video_sequence)} video segments...")
        combine_videos(video_sequence, args.output)
        
        logger.info(f"Video saved to {args.output}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup processes before cleaning up files
        cleanup_processes()
        logger.debug("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()

def bubble_sort_frames_buffered(img_array, temp_dir, highlight_swaps=False, swap_flash=False, frame_interval=100):
    """Create frames while bubble sorting, buffering frames to disk to save memory"""
    height, width, _ = img_array.shape
    img_array = img_array.copy()
    frame_count = 0
    frames_dir = os.path.join(temp_dir, f"frames_{uuid.uuid4()}")
    os.makedirs(frames_dir, exist_ok=True)
    
    logger.debug(f"Starting bubble sort on image of size {width}x{height}")
    logger.debug(f"Buffering frames to {frames_dir}")
    
    # Save initial frame
    cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_count:08d}.png"), 
                cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    frame_count += 1
    total_swaps = 0
    swap_count = 0
    
    for i in range(height):
        # logger.debug(f"Processing row {i}/{height}")
        for channel in range(3):
            # logger.debug(f"Processing channel {channel} in row {i}")
            swapped = False
            row_swaps = 0
            
            for j in range(width - 1):
                # logger.debug(f"Processing row {i}/{height} col {j}/{width}")
                for k in range(width - j - 1):
                    logger.debug(f"Processing row {i}/{height}\tchannel {channel}/3\tcol {j}/{width}\tprogress {k}/{width - j - 1}")
                    if img_array[i, k, channel] > img_array[i, k + 1, channel]:
                        img_array[i, k, channel], img_array[i, k + 1, channel] = \
                            img_array[i, k + 1, channel], img_array[i, k, channel]
                        swapped = True
                        row_swaps += 1
                        swap_count += 1
                        
                        # Only save a frame every frame_interval swaps
                        if swap_count % frame_interval == 0:
                            if highlight_swaps:
                                frame = img_array.copy()
                                frame[i, k] = [255, 0, 0]
                                frame[i, k + 1] = [255, 0, 0]
                                cv2.imwrite(
                                    os.path.join(frames_dir, f"frame_{frame_count:08d}.png"),
                                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                )
                                frame_count += 1
                                
                                if swap_flash:
                                    cv2.imwrite(
                                        os.path.join(frames_dir, f"frame_{frame_count:08d}.png"),
                                        cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                                    )
                                    frame_count += 1
                            else:
                                cv2.imwrite(
                                    os.path.join(frames_dir, f"frame_{frame_count:08d}.png"),
                                    cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                                )
                                frame_count += 1
                            
            if not swapped:
                logger.debug(f"Row {i}, channel {channel} already sorted")
                break
            else:
                logger.debug(f"Row {i}, channel {channel} completed with {row_swaps} swaps")
                total_swaps += row_swaps
    
    # Save final frame if needed
    final_frame_path = os.path.join(frames_dir, f"frame_{frame_count-1:08d}.png")
    if not os.path.exists(final_frame_path):
        cv2.imwrite(
            final_frame_path,
            cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        )
    
    logger.debug(f"Bubble sort completed with {total_swaps} total swaps and {frame_count} frames generated")
    return frames_dir, frame_count, img_array

def write_frames_to_video_from_disk(frames_dir, frame_count, output_path, fps=60, reverse=False):
    """Create video from frames stored on disk"""
    if frame_count == 0:
        logger.warning("No frames to write to video")
        return None
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, "frame_00000000.png"))
    height, width = first_frame.shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    logger.debug(f"Writing {frame_count} frames to video at {fps} FPS")
    frame_range = range(frame_count-1, -1, -1) if reverse else range(frame_count)
    
    for i in frame_range:
        if i % 100 == 0:  # Log progress periodically
            logger.debug(f"Writing frame {i}/{frame_count}")
        frame = cv2.imread(os.path.join(frames_dir, f"frame_{i:08d}.png"))
        if frame is not None:
            out.write(frame)
        else:
            logger.error(f"Failed to read frame {i}")
    
    out.release()
    logger.debug(f"Video writing completed: {output_path}")
    return output_path
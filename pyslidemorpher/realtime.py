"""
Real-time playback functionality for PySlide Morpher.
Handles live slideshow display using OpenCV.
"""

import logging
import random
import time
import threading
from queue import Queue
import cv2

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


def get_random_transition_function():
    """Randomly select a transition function from available options.

    Ensures that the same transition function is never selected twice in a row.
    """
    transition_functions = [
        make_transition_frames,        # default
        make_swarm_transition_frames,  # swarm
        make_tornado_transition_frames, # tornado
        make_swirl_transition_frames,  # swirl
        make_drip_transition_frames,   # drip
        # make_rainfall_transition_frames, # rain
        make_sorted_transition_frames, # sorted
        make_hue_sorted_transition_frames, # hue-sorted
    ]

    # Initialize the last selected function attribute if it doesn't exist
    if not hasattr(get_random_transition_function, '_last_selected'):
        get_random_transition_function._last_selected = None

    # If this is the first call or there's only one function, just return a random choice
    if get_random_transition_function._last_selected is None or len(transition_functions) <= 1:
        selected = random.choice(transition_functions)
        get_random_transition_function._last_selected = selected
        logging.info(f"Randomly selected transition function: {selected.__name__}")
        return selected

    # Create a list of available functions excluding the last selected one
    available_functions = [func for func in transition_functions 
                          if func != get_random_transition_function._last_selected]

    # Select from the available functions
    selected = random.choice(available_functions)
    get_random_transition_function._last_selected = selected
    logging.info(f"Randomly selected transition function: {selected.__name__}")
    return selected


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
                if args.transition == "random":
                    transition_fn = get_random_transition_function()
                    logging.debug(f"Randomly selected transition: {transition_fn.__name__}")
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
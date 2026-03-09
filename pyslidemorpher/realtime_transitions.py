"""Transition selection helpers for realtime mode."""

import logging
import random

from .transitions import (
    make_transition_frames,
    make_swarm_transition_frames,
    make_tornado_transition_frames,
    make_swirl_transition_frames,
    make_drip_transition_frames,
    make_rainfall_transition_frames,
    make_sorted_transition_frames,
    make_hue_sorted_transition_frames,
)

TRANSITION_FUNCTIONS = {
    "swarm": make_swarm_transition_frames,
    "tornado": make_tornado_transition_frames,
    "swirl": make_swirl_transition_frames,
    "drip": make_drip_transition_frames,
    "rain": make_rainfall_transition_frames,
    "sorted": make_sorted_transition_frames,
    "hue-sorted": make_hue_sorted_transition_frames,
}


def get_random_transition_function():
    """Randomly select a transition function from available options.

    Ensures that the same transition function is never selected twice in a row.
    """
    transition_functions = [
        make_transition_frames,
        make_swarm_transition_frames,
        make_tornado_transition_frames,
        make_swirl_transition_frames,
        make_drip_transition_frames,
        make_sorted_transition_frames,
        make_hue_sorted_transition_frames,
    ]

    if not hasattr(get_random_transition_function, "_last_selected"):
        get_random_transition_function._last_selected = None

    if get_random_transition_function._last_selected is None or len(transition_functions) <= 1:
        selected = random.choice(transition_functions)
        get_random_transition_function._last_selected = selected
        logging.info("Randomly selected transition function: %s", selected.__name__)
        return selected

    available_functions = [
        func for func in transition_functions if func != get_random_transition_function._last_selected
    ]

    selected = random.choice(available_functions)
    get_random_transition_function._last_selected = selected
    logging.info("Randomly selected transition function: %s", selected.__name__)
    return selected


def get_transition_function(transition_name):
    """Get transition function for a transition name."""
    if transition_name == "random":
        return get_random_transition_function()
    return TRANSITION_FUNCTIONS.get(transition_name, make_transition_frames)

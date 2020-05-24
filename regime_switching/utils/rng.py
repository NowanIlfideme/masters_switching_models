from copy import deepcopy
from typing import Union

from numpy.random import Generator, default_rng

AnyRandomState = Union[int, Generator, None]


def fix_rng(random_state: AnyRandomState = None) -> Generator:
    """Helper function for copying a random state."""

    if isinstance(random_state, Generator):
        random_state = deepcopy(random_state)
    else:
        random_state = default_rng(random_state)
    return random_state

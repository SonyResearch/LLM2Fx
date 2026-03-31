import math
from typing import Any


def is_number(something: Any, finite: bool = True) -> bool:
    """Check whether something is convertible to a float

    Args:
        something: something to test for float casting.

    Returns:
        Whether or not it's a number
    """
    try:
        f = float(something)
        if finite and not math.isfinite(f):
            return False
        return True
    except ValueError:
        return False

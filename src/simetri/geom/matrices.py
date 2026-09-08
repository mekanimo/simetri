"""Small matrix helpers shared across geometry and rendering.

Examples:
    >>> import simetri.graphics as sg
    >>> matrix = sg.identity_matrix()
    >>> points = sg.homogenize([[1, 2], [3, 4]])
    >>> print(points @ matrix)
    [[1. 2. 1.]
     [3. 4. 1.]]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["identity_matrix"]


def identity_matrix() -> NDArray:
    """Return the 3x3 identity matrix.

    Returns:
        np.ndarray: ``[[1, 0, 0], [0, 1, 0], [0, 0, 1]]``.
    """
    return np.identity(3)
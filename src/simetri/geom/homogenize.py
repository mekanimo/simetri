"""Convert Cartesian points to homogeneous coordinates."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from simetri.base.common import PointType


def homogenize(points: Sequence[PointType]) -> NDArray:
    """Convert a list of points to homogeneous coordinates.

    Args:
        points: Sequence of ``(x, y)`` points (extra coords ignored).

    Returns:
        NDArray: Homogeneous coordinates with a trailing 1 column.

    Examples:
        >>> from simetri.geom.homogenize import homogenize
        >>> homogenize([(20, 40), (60, 80)]).tolist()
        [[20.0, 40.0, 1.0], [60.0, 80.0, 1.0]]
"""
    try:
        xy_array = np.array(points, dtype=float)
    except ValueError:
        xy_array = np.array([p[:2] for p in points], dtype=float)
    n_rows, n_cols = xy_array.shape
    if n_cols > 2:
        xy_array = xy_array[:, :2]
    ones = np.ones((n_rows, 1), dtype=float)
    homogeneous_array = np.append(xy_array, ones, axis=1)

    return homogeneous_array

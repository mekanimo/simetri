from simetri.base.common import PointType


import numpy as np
from numpy.typing import NDArray


from collections.abc import Sequence


def homogenize(points: Sequence[PointType]) -> NDArray:
    """Convert a list of points to homogeneous coordinates.

    Args:
        points: Sequence of ``(x, y)`` points (extra coords ignored).

    Returns:
        NDArray: Homogeneous coordinates with a trailing 1 column.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.homogenize([(1, 2), (3, 4)])
        array([[1., 2., 1.],
               [3., 4., 1.]])
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

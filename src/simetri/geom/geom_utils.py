"""Dependency-free geometry primitives.

This module is intentionally a **leaf**: it must not import from
``points``, ``segments``, ``polygons``, ``nonlinear``, or ``geometry``.
Other geometry modules may import from here to avoid circular imports.
"""

from collections.abc import Sequence
from math import cos, sin, sqrt

from simetri.base.common import PointType


def r_polar(a: float, b: float, theta: float) -> float:
    """Return the ellipse radius at the given angle.

    Args:
        a: Semi-axis along x.
        b: Semi-axis along y.
        theta: Angle in radians.

    Returns:
        Radius of the ellipse at ``theta``.
    """
    return (a * b) / sqrt((b * cos(theta)) ** 2 + (a * sin(theta)) ** 2)


def distance2(p1: PointType, p2: PointType) -> float:
    """Return the squared distance between two points.

    Args:
        p1: First point.
        p2: Second point.

    Returns:
        Squared Euclidean distance.
    """
    return (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2


def close_points2(p1: PointType, p2: PointType, dist2: float = 0.01) -> bool:
    """Return True if two points are within squared distance ``dist2``.

    Args:
        p1: First point.
        p2: Second point.
        dist2: Squared distance threshold. Defaults to 0.01.

    Returns:
        True if the points are close enough.
    """
    return distance2(p1, p2) <= dist2


def offset_point_from_start(
    p1: PointType, p2: PointType, offset: float
) -> PointType:
    """Return the point on the line through ``p1``–``p2`` at distance ``offset`` from ``p1``.

    Args:
        p1: Start point on the line.
        p2: Second point defining the line direction.
        offset: Distance from ``p1`` along the line.

    Returns:
        Point on the line at the given offset.
    """
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    dx, dy = x2 - x1, y2 - y1
    d = (dx**2 + dy**2) ** 0.5
    if d == 0:
        return p1
    return (x1 + offset * dx / d, y1 + offset * dy / d)


def midpoint(p1: PointType, p2: PointType) -> PointType:
    """Return the midpoint of two points.

    Args:
        p1: First point.
        p2: Second point.

    Returns:
        Midpoint ``((x1+x2)/2, (y1+y2)/2)``.
    """
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def offset_point(point: PointType, dx: float = 0, dy: float = 0) -> PointType:
    """Return ``point`` translated by ``(dx, dy)``.

    Args:
        point: Input point.
        dx: Offset in x. Defaults to 0.
        dy: Offset in y. Defaults to 0.

    Returns:
        Translated point.
    """
    x, y = point[:2]
    return (x + dx, y + dy)


def connected_pairs(items: Sequence, closed: bool = False) -> list[tuple]:
    """Return consecutive pairs from ``items``.

    ``[a, b, c]`` → ``[(a, b), (b, c)]``. With ``closed=True``, also
    appends ``(c, a)``.

    Args:
        items: Sequence of items.
        closed: If True, connect last item to first. Defaults to False.

    Returns:
        List of adjacent pairs.
    """
    pairs = list(zip(items, items[1:]))
    if closed and items:
        pairs.append((items[-1], items[0]))
    return pairs


def turning_function(curve):
    """
    Compute the turning function for a planar curve.
    curve: Nx2 array-like of (x, y) points (ordered).
    Returns: angles (cumulative turning angle at each point), arc_lengths
    """
    curve = np.asarray(curve)
    if curve.shape[0] < 2:
        return np.zeros(0), np.zeros(0)
    # Compute tangent vectors
    tangents = np.diff(curve, axis=0)
    # Compute angles between consecutive segments
    angles = np.arctan2(tangents[:, 1], tangents[:, 0])
    # Compute turning angles (difference between consecutive angles)
    turning_angles = np.diff(angles)
    # Unwrap to avoid jumps at -pi/pi
    turning_angles = np.unwrap(turning_angles)
    # Cumulative sum gives the turning function
    cumulative_turn = np.concatenate([[0], np.cumsum(turning_angles)])
    # Arc length parameterization
    arc_lengths = np.concatenate(
        [[0], np.cumsum(np.linalg.norm(tangents, axis=1))]
    )
    return cumulative_turn, arc_lengths


def turning_function_metric(curve1, curve2, num_samples=100):
    """
    Computes the turning function metric (L1 distance between turning functions).
    curve1, curve2: Nx2 arrays of (x, y) points.
    Returns: scalar metric value.
    """
    tf1, s1 = turning_function(curve1)
    tf2, s2 = turning_function(curve2)
    # Resample both turning functions to a common arc length grid
    s_min = min(s1[0], s2[0])
    s_max = min(s1[-1], s2[-1])
    s_common = np.linspace(s_min, s_max, num_samples)
    tf1_interp = np.interp(s_common, s1, tf1)
    tf2_interp = np.interp(s_common, s2, tf2)
    # L1 distance
    metric = np.sum(np.abs(tf1_interp - tf2_interp)) * (
        s_common[1] - s_common[0]
    )
    return metric

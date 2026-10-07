"""Point utilities: distance, rounding, midpoints, and related helpers."""

from __future__ import annotations

from collections.abc import Sequence
from math import atan2, hypot, isclose, sqrt
from typing import Any

import numpy as np
from numpy import array
from numpy.typing import NDArray

from simetri.base.all_enums import Types
from simetri.base.common import (
    LineType,
    PointType,
    get_defaults,
    resolve_tol,
)
from simetri.config.settings import runtime_defaults
from simetri.geom.affine import rotate_point
from simetri.geom.geom_utils import close_points_square
from simetri.geom.vectors import (
    cross_product_sense3,
    perp_unit_vector,
)
from simetri.helpers.utilities import lerp


def distance(p1: PointType, p2: PointType) -> float:
    """Return the Euclidean distance between two points.

    Args:
        p1: First point.
        p2: Second point.

    Returns:
        float: Distance between the two points.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.distance((0, 0), (60, 80))
        100.0
"""
    return hypot(p2[0] - p1[0], p2[1] - p1[1])


def equal_points(
    point1: PointType,
    point2: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if two points are within ``abs_tol`` of each other.

    Args:
        point1: First point.
        point2: Second point.
        rel_tol: Relative tolerance. Defaults to ``runtime_defaults["rel_tol"]``.
        abs_tol: Absolute tolerance in points. Defaults to ``runtime_defaults["abs_tol"]``.

    Returns:
        bool: True if the points are within the given distance.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.equal_points((0, 0), (0.0005, 0))
        True
        >>> sg.equal_points((0, 0), (40, 0))
        False
"""
    _, abs_tol = resolve_tol(rel_tol, abs_tol)

    return distance(point1, point2) <= abs_tol


def congruent_points(
    point1: PointType,
    point2: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Alias for ``equal_points``.

    Args:
        point1: First point.
        point2: Second point.
        rel_tol: Relative tolerance. Defaults to ``runtime_defaults["rel_tol"]``.
        abs_tol: Absolute tolerance in points. Defaults to ``runtime_defaults["abs_tol"]``.

    Returns:
        bool: True if the points are within the given distance.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.congruent_points((0, 0), (0.0005, 0))
        True
        >>> sg.congruent_points((0, 0), (40, 0))
        False
"""
    return equal_points(
        point1,
        point2,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def offset_point_on_line(
    point: PointType, line: LineType, offset: float
) -> PointType:
    """Return a point on a line that is offset from the given point.

    Args:
        point (PointType): Input point.
        line (LineType): Input line.
        offset (float): Offset distance.

    Returns:
        PointType: Offset point on the line.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.offset_point_on_line((0, 0), [(0, 0), (40, 0)], 2)
        (2.0, 0.0)
"""
    x, y = point[:2]
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    dx = x2 - x1
    dy = y2 - y1
    # normalize the vector
    mag = (dx * dx + dy * dy) ** 0.5
    dx = dx / mag
    dy = dy / mag
    return x + dx * offset, y + dy * offset


def perp_offset_point(
    point: PointType, line: LineType, offset: float
) -> PointType:
    """Return a point that is offset from the given point in the perpendicular direction to the given line.

    Args:
        point (PointType): Input point.
        line (LineType): Input line.
        offset (float): Offset distance.

    Returns:
        PointType: Perpendicular offset point.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.perp_offset_point((0, 0), [(0, 0), (40, 0)], 1)
        [0.0, 1.0]
"""
    unit_vec = perp_unit_vector(line)
    dx = unit_vec[0] * offset
    dy = unit_vec[1] * offset
    x, y = point[:2]
    return [x + dx, y + dy]


def fix_degen_points(
    points: list[PointType],
    loop: bool = False,
    closed: bool = False,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
    check_collinear: bool = True,
) -> list[PointType]:
    """Return points with duplicates and collinear middles removed.

    Args:
        points: Point list (mutated in place).
        loop: Whether to treat the list as a loop. Defaults to False.
        closed: Whether the polyline is closed. Defaults to False.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.
        check_collinear (bool, optional): Whether to check for collinear points. Defaults to True.

    Returns:
        list[PointType]: List of points with duplicate and collinear points removed.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.fix_degen_points(
        ...     [(0, 0), (0, 0), (40, 0), (80, 0)],
        ...     check_collinear=False,
        ... )
        [(0, 0), (40, 0), (80, 0)]
"""
    rel_tol, abs_tol = resolve_tol(rel_tol, abs_tol)
    abs_tol2 = abs_tol * abs_tol
    new_points = []
    for i, point in enumerate(points):
        if i == 0:
            new_points.append(point)
        else:
            if not close_points_square(point, new_points[-1], dist2=abs_tol2):
                new_points.append(point)
    if loop and close_points_square(
        new_points[0], new_points[-1], dist2=abs_tol2
    ):
        new_points.pop(-1)

    if check_collinear:
        # Check for collinear points and remove the middle one.
        from simetri.geom.segments.line_utils import (
            merge_consecutive_collinear_edges,
        )

        new_points = merge_consecutive_collinear_edges(
            new_points,
            closed,
            rel_tol,
            abs_tol,
        )

    return new_points


def round_point(point: list[float], n_digits: int = 2) -> list[float]:
    """
    Round a point (x, y) to a given precision.

    Args:
        point (list[float]): Input point.
        n_digits (int, optional): Number of decimal places to round to. Defaults to 2.

    Returns:
        list[float]: Rounded point.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.round_point([1.234, 5.678], 2)
        (1.23, 5.68)
"""
    x, y = point[:2]
    x = round(x, n_digits)
    y = round(y, n_digits)
    return (x, y)


def round_points(points: list[PointType], n_digits: int = 2) -> list[PointType]:
    """
    Round a list of points to a given precision.

    Args:
        points (list[PointType]): Input point list.
        n_digits (int, optional): Number of decimal places to round to. Defaults to 2.

    Returns:
        list[PointType]: Rounded points list.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.round_points([(1.234, 5.678)], 1)
        [(1.2, 5.7)]
"""

    return [round_point(p, n_digits) for p in points]


def direction3(p: PointType, q: PointType, r: PointType) -> float:
    """Return the signed orientation of three points.

    Args:
        p: First point.
        q: Second point.
        r: Third point.

    Returns:
        Zero if collinear, positive if counter-clockwise, negative if clockwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.direction3((0, 0), (40, 0), (40, 40))
        -1600
        >>> sg.direction3((0, 0), (40, 0), (40, -40))
        1600
"""
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])


def between3(a: PointType, b: PointType, c: PointType) -> bool:
    """Return True if ``c`` lies on segment ``ab`` (collinear and in range).

    Args:
        a: Segment start.
        b: Segment end.
        c: Query point.

    Returns:
        True if ``c`` is between ``a`` and ``b``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.between3((0, 0), (80, 0), (40, 0))
        True
        >>> sg.between3((0, 0), (40, 0), (80, 0))
        False
"""
    from simetri.geom.segments.line_utils import collinear3

    if not collinear3(a, b, c):
        res = False
    elif a[0] != b[0]:
        res = ((a[0] <= c[0]) and (c[0] <= b[0])) or (
            (a[0] >= c[0]) and (c[0] >= b[0])
        )
    else:
        res = ((a[1] <= c[1]) and (c[1] <= b[1])) or (
            (a[1] >= c[1]) and (c[1] >= b[1])
        )
    return res


def check_consecutive_duplicates(
    points: Sequence[PointType] | NDArray[np.floating],
    rel_tol: float = 0,
    abs_tol: float | None = None,
) -> bool:
    """Return True if any consecutive vertices match within tolerance.

    Args:
        points: Points to check.
        rel_tol: Relative tolerance. Defaults to 0.
        abs_tol: Absolute tolerance. Defaults to ``runtime_defaults['abs_tol']``.

    Returns:
        bool: True if consecutive duplicate points are found, False otherwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.check_consecutive_duplicates([(0, 0), (0, 0), (40, 0)], abs_tol=0.001)
        True
        >>> sg.check_consecutive_duplicates([(0, 0), (40, 0)], abs_tol=0.001)
        False
"""
    if abs_tol is None:
        abs_tol = runtime_defaults["abs_tol"]
    if isinstance(points, np.ndarray):
        points = points.tolist()
    if points and len(points) > 1:
        for i, pnt in enumerate(points[:-1]):
            next_pnt = points[i + 1]
            val1 = pnt[0] + pnt[1]
            val2 = next_pnt[0] + next_pnt[1]
            if isclose(
                val1, val2, rel_tol=rel_tol, abs_tol=abs_tol
            ) and np.allclose(pnt, next_pnt, rtol=0, atol=abs_tol):
                return True

    return False


def left3(a: PointType, b: PointType, c: PointType) -> bool:
    """
    Check if point c is left of line ab.
    Args:
        a (PointType): The first point defining the line.
        b (PointType): The second point defining the line.
        c (PointType): The point to test.
    Returns:
        bool: True if point c is left of line ab, False otherwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.left3((0, 0), (40, 0), (0, 40))
        True
        >>> sg.left3((0, 0), (40, 0), (0, -40))
        False
"""

    ax, ay = a[:2]
    bx, by = b[:2]
    cx, cy = c[:2]
    return (bx - ax) * (cy - ay) - (cx - ax) * (by - ay) > 0


def remove_duplicate_points(
    points: list[PointType],
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> list[PointType]:
    """
    Return a list of points with duplicate points removed.

    Args:
        points (list[PointType]): List of points.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        list[PointType]: List of points with duplicate points removed.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.remove_duplicate_points([(0, 0), (0, 0), (40, 0)], abs_tol=0.001)
        [(0, 0), (40, 0)]
"""
    _, abs_tol = resolve_tol(rel_tol, abs_tol)
    abs_tol2 = abs_tol * abs_tol
    new_points = []
    for i, point in enumerate(points):
        if i == 0:
            new_points.append(point)
        else:
            if not close_points_square(point, new_points[-1], dist2=abs_tol2):
                new_points.append(point)
    return new_points


def remove_collinear_points(
    points: list[PointType],
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> list[PointType]:
    """
    Return a list of points with collinear points removed.

    Args:
        points (list[PointType]): List of points.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        list[PointType]: List of points with collinear points removed.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.remove_collinear_points([(0, 0), (40, 0), (80, 0)], abs_tol=0.001)
        [(0, 0), (80, 0)]
"""
    from simetri.geom.segments.line_utils import collinear3

    rel_tol, abs_tol = resolve_tol(rel_tol, abs_tol)
    n = len(points)
    closed = n > 2 and points[0] == points[-1]
    new_points = []
    for i, point in enumerate(points):
        if i == 0:
            new_points.append(point)
            continue
        if not closed and i == n - 1:
            new_points.append(point)
            continue
        next_index = (i + 1) % n if closed else i + 1
        if not collinear3(
            new_points[-1],
            point,
            points[next_index],
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        ):
            new_points.append(point)
    return new_points


def clockwise3(p: PointType, q: PointType, r: PointType) -> bool:
    """Return 1 if the points p, q, and r are in clockwise order,
    return -1 if the points are in counter-clockwise order,
    return 0 if the points are collinear

    Args:
        p (PointType): First point.
        q (PointType): Second point.
        r (PointType): Third point.

    Returns:
        int: 1 if the points are in clockwise order, -1 if counter-clockwise, 0 if collinear.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.clockwise3((0, 0), (40, 0), (0, 40))
        -1
        >>> sg.clockwise3((0, 0), (0, 40), (40, 0))
        1
"""
    px, py = p[:2]
    qx, qy = q[:2]
    rx, ry = r[:2]
    area_ = (qx - px) * (ry - py) - (rx - px) * (qy - py)
    if area_ > 0:
        res = -1
    elif area_ < 0:
        res = 1
    else:
        res = 0

    return res


def on_segment(
    a: PointType, b: PointType, p: PointType, eps: float = 1e-12
) -> bool:
    """Return True if point ``p`` lies on segment ``ab`` within ``eps``.

    Args:
        a: Segment start point.
        b: Segment end point.
        p: Query point.
        eps: Numeric tolerance. Defaults to ``1e-12``.

    Returns:
        bool: True if ``p`` is collinear with ``ab`` and inside its bbox.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.on_segment((0, 0), (80, 0), (40, 0))
        True
        >>> sg.on_segment((0, 0), (40, 0), (80, 0))
        False
"""

    # check collinear + within bbox
    def cross(ax: float, ay: float, bx: float, by: float) -> float:
        return ax * by - ay * bx

    def orient3(a: PointType, b: PointType, c: PointType) -> float:
        # cross((b-a),(c-a))
        return cross(b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1])

    if abs(orient3(a, b, p)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def lerp_point(p1: PointType, p2: PointType, t: float) -> PointType:
    """Linear interpolation of two points.

    Args:
        p1 (PointType): First point.
        p2 (PointType): Second point.
        t (float): Interpolation parameter. t = 0 => p1, t = 1 => p2.

    Returns:
        PointType: Interpolated point.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.lerp_point((0, 0), (40, 0), 0.5)
        (20.0, 0.0)
"""
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    return (lerp(x1, x2, t), lerp(y1, y2, t))


def angle(point: PointType) -> float:
    """Return the angle of a line drawn from the given point to the origin in radians.

    Args:
        point (PointType): Input point.

    Returns:
        float: Angle of the point in radians.

    Examples:
        >>> import simetri.graphics as sg
        >>> round(sg.angle((40, 0)), 10)
        0.0
        >>> round(sg.angle((0, 1)), 10)
        1.5707963268
"""
    return atan2(point[1], point[0])


def point_on_line(
    point: PointType,
    line: LineType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if the given point is on the given line

    Args:
        point (PointType): Input point.
        line (LineType): Input line.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        bool: True if the point is on the line, False otherwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.point_on_line((40, 40), [(0, 0), (80, 80)], abs_tol=0.001)
        True
"""
    from simetri.geom.segments.line_utils import slope

    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    p1, p2 = line
    return isclose(
        slope(p1, point), slope(point, p2), rel_tol=rel_tol, abs_tol=abs_tol
    )


def point_on_line_segment(
    point: PointType,
    line: LineType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if the given point is on the given line segment

    Args:
        point (PointType): Input point.
        line (LineType): Input line segment.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        bool: True if the point is on the line segment, False otherwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.point_on_line_segment((40, 0), [(0, 0), (80, 0)], abs_tol=0.001)
        True
        >>> sg.point_on_line_segment((100, 0), [(0, 0), (80, 0)], abs_tol=0.001)
        False
"""
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    p1, p2 = line
    return isclose(
        (distance(p1, point) + distance(p2, point)),
        distance(p1, p2),
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def point_to_line_distance(point: PointType, line: LineType) -> float:
    """Return the distance between a line and a point.

    Args:
        point (PointType): Input point.
        line (LineType): Input line.

    Returns:
        float: Distance from the point to the line.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.point_to_line_distance((0, 40), [(0, 0), (40, 0)])
        40.0
"""
    x0, y0 = point
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    dx = x2 - x1
    dy = y2 - y1
    return abs(dx * (y1 - y0) - (x1 - x0) * dy) / sqrt(dx**2 + dy**2)


def point_to_line_seg_distance(
    p: PointType, lp1: PointType, lp2: PointType
) -> float | bool:
    """Given a point p and a line segment defined by boundary points
    lp1 and lp2, returns the distance between the line segment and the point.
    If the point is not located in the perpendicular area between the
    boundary points, returns False.

    Args:
        p (PointType): Input point.
        lp1 (PointType): First boundary point of the line segment.
        lp2 (PointType): Second boundary point of the line segment.

    Returns:
        float: Distance between the point and the line segment, or False if the point is not in the perpendicular area.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.point_to_line_seg_distance((40, 40), (0, 0), (80, 0))
        40.0
        >>> sg.point_to_line_seg_distance((100, 0), (0, 0), (80, 0))
        False
"""
    if lp1[:2] == lp2[:2]:
        msg = "Error! Line is ill defined. Start and end points are coincident."
        raise ValueError(msg)
    x3, y3 = p[:2]
    x1, y1 = lp1[:2]
    x2, y2 = lp2[:2]

    u = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / distance(
        lp1, lp2
    ) ** 2
    if 0 <= u <= 1:
        x = x1 + u * (x2 - x1)
        y = y1 + u * (y2 - y1)
        res = distance((x, y), p)
    else:
        res = False  # p is not between lp1 and lp2

    return res


def flat_points(connected_segments: Sequence[LineType]) -> list[PointType]:
    """Return a list of points from a list of connected pairs of points.

    Args:
        connected_segments (list[tuple]): List of connected pairs of points.

    Returns:
        list[PointType]: List of points.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.flat_points([((0, 0), (40, 0)), ((40, 0), (40, 40))])
        [(0, 0), (40, 0), (40, 40)]
"""
    points = [line[0] for line in connected_segments]
    points.append(connected_segments[-1][1])
    return points


def point_in_quad(point: PointType, quad: list[PointType]) -> bool:
    """Return True if the point is inside the quad.

    Args:
        point (PointType): Input point.
        quad (list[PointType]): List of points representing the quad.

    Returns:
        bool: True if the point is inside the quad, False otherwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> quad = [(0, 0), (80, 0), (80, 80), (0, 80)]
        >>> sg.point_in_quad((40, 40), quad)
        True
        >>> sg.point_in_quad((100, 100), quad)
        False
"""
    x, y = point[:2]
    x1, y1 = quad[0][:2]
    x2, y2 = quad[1][:2]
    x3, y3 = quad[2][:2]
    x4, y4 = quad[3][:2]
    xs = [x1, x2, x3, x4]
    ys = [y1, y2, y3, y4]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    return min_x <= x <= max_x and min_y <= y <= max_y


def remove_bad_points(points: list[PointType]) -> list[PointType]:
    """Remove redundant and collinear points from a list of points.

    Args:
        points (list[PointType]): List of points.

    Returns:
        list[PointType]: List of points with redundant and collinear points removed.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.remove_bad_points([(0, 0), (0, 0), (40, 0), (80, 0)])
        [(0, 0), (40, 0), (80, 0)]
"""
    EPSILON = 1e-16
    n_points = len(points)
    # check for redundant points
    for i, p in enumerate(points[:]):
        for j in range(i + 1, n_points - 1):
            if p == points[j]:  # then remove the redundant point
                # maybe we should display a warning message here indicating
                # that redundant point is removed!!!
                points.remove(p)

    n_points = len(points)
    # check for three consecutive points on a line
    lin_points = []
    for i in range(2, n_points - 1):
        first_point = points[i - 2][:2]
        second_point = points[i - 1][:2]
        third_point = points[i][:2]
        signed_area = (
            first_point[0] * second_point[1]
            + second_point[0] * third_point[1]
            + third_point[0] * first_point[1]
            - second_point[0] * first_point[1]
            - third_point[0] * second_point[1]
            - first_point[0] * third_point[1]
        )
        if EPSILON > abs(signed_area) / 2.0 > -EPSILON:
            lin_points.append(points[i - 1])

    if len(points) > 2 and points[0] == points[-1]:
        first_point = points[-2][:2]
        second_point = points[-1][:2]
        third_point = points[0][:2]
        signed_area = (
            first_point[0] * second_point[1]
            + second_point[0] * third_point[1]
            + third_point[0] * first_point[1]
            - second_point[0] * first_point[1]
            - third_point[0] * second_point[1]
            - first_point[0] * third_point[1]
        )
        if EPSILON > abs(signed_area) / 2.0 > -EPSILON:
            lin_points.append(points[-1])

    for p in lin_points:
        # maybe we should display a warning message here indicating that linear
        # point is removed!!!
        points.remove(p)

    return points


class Vertex(list):
    """A 3D vertex.

    Examples:
        >>> from simetri.geom.points.point_utils import Vertex
        >>> v = Vertex(40, 80, 60)
        >>> v.coords
        (40, 80, 60)
        >>> v.copy().coords
        (40, 80, 60)
"""

    def __init__(self, x: float, y: float, z: float = 0) -> None:
        """Create a vertex at ``(x, y, z)``.

        Args:
            x: X coordinate.
            y: Y coordinate.
            z: Z coordinate. Defaults to 0.

        Examples:
            >>> from simetri.geom.points.point_utils import Vertex
            >>> Vertex(40, 80).coords
            (40, 80, 0)
        """
        super().__init__((x, y, z))
        self.x = x
        self.y = y
        self.z = z
        self.type = Types.VERTEX

    def __repr__(self) -> str:
        return f"Vertex({self.x}, {self.y}, {self.z})"

    def __eq__(self, other: object) -> bool:
        return (
            self[0] == other[0] and self[1] == other[1] and self[2] == other[2]
        )

    def copy(self) -> Vertex:
        """Return a new ``Vertex`` with the same coordinates.

        Returns:
            Vertex: Copy of this vertex.

        Examples:
            >>> from simetri.geom.points.point_utils import Vertex
            >>> Vertex(40, 80).copy().coords
            (40, 80, 0)
"""
        return Vertex(self.x, self.y, self.z)

    def __add__(self, other: Vertex) -> Vertex:
        return Vertex(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vertex) -> Vertex:
        return Vertex(self.x - other.x, self.y - other.y, self.z - other.z)

    @property
    def coords(self) -> tuple[float, float, float]:
        """Return the coordinates as a tuple.

        Examples:
            >>> from simetri.geom.points.point_utils import Vertex
            >>> Vertex(60, 80).coords
            (60, 80, 0)
"""
        return (self.x, self.y, self.z)

    @property
    def array(self) -> NDArray[np.floating]:
        """Homogeneous coordinates as a numpy array.

        Examples:
            >>> from simetri.geom.points.point_utils import Vertex
            >>> Vertex(80, 60).array.tolist()
            [80.0, 60.0, 1.0]
"""
        return array([self.x, self.y, 1.0], dtype=float)

    def v_tuple(self) -> tuple[float, float, float]:
        """Return the vertex as a tuple.

        Examples:
            >>> from simetri.geom.points.point_utils import Vertex
            >>> Vertex(80, 60).v_tuple()
            (80, 60, 0)
"""
        return (self.x, self.y, self.z)

    def below(self, other: Vertex) -> bool:
        """This is for 2D points only

        Args:
            other (Vertex): Other vertex.

        Returns:
            bool: True if this vertex is below the other vertex, False otherwise.

        Examples:
            >>> from simetri.geom.points.point_utils import Vertex
            >>> Vertex(0, 0).below(Vertex(0, 40))
            True
"""
        res = False
        if self.y < other.y or self.y == other.y and self.x > other.x:
            res = True
        return res

    def above(self, other: Vertex) -> bool:
        """This is for 2D points only

        Args:
            other (Vertex): Other vertex.

        Returns:
            bool: True if this vertex is above the other vertex, False otherwise.

        Examples:
            >>> from simetri.geom.points.point_utils import Vertex
            >>> Vertex(0, 40).above(Vertex(0, 0))
            True
"""
        if self.y > other.y or self.y == other.y and self.x < other.x:
            res = True
        else:
            res = False

        return res


def set_vertices(points: list[Vertex]) -> None:
    """Set the next and previous vertices of a list of vertices.

    Args:
        points (list[Vertex]): List of vertices.

    Examples:
        >>> from simetri.geom.points.point_utils import Vertex, set_vertices
        >>> verts = [Vertex(0, 0), Vertex(40, 0), Vertex(0, 40)]
        >>> set_vertices(verts)
        >>> [(v.next.coords[:2], v.prev.coords[:2]) for v in verts]
        [((40, 0), (0, 40)), ((0, 40), (0, 0)), ((0, 0), (40, 0))]
"""
    if not isinstance(points[0], Vertex):
        points = [Vertex(*p[:]) for p in points]
    n_points = len(points)
    for i, p in enumerate(points):
        if i == 0:
            p.prev = points[-1]
            p.next = points[i + 1]
        elif i == (n_points - 1):
            p.prev = points[i - 1]
            p.next = points[0]
        else:
            p.prev = points[i - 1]
            p.next = points[i + 1]
        p.angle = cross_product_sense3(p.prev, p, p.next)


def project_point_on_line(point: Vertex, line: tuple[Vertex, Vertex]) -> Vertex:
    """Project ``point`` onto the segment ``line``.

    Args:
        point: Query vertex.
        line: Segment as ``(start, end)`` vertices.

    Returns:
        Closest point on the segment.

    Examples:
        >>> from simetri.geom.points.point_utils import Vertex, project_point_on_line
        >>> v = Vertex(40, 40)
        >>> p = project_point_on_line(v, (Vertex(0, 0), Vertex(80, 0)))
        >>> (p.x, p.y)
        (40.0, 0.0)
"""
    v = point
    a, b = line

    av = v - a
    ab = b - a
    ab2 = ab.x * ab.x + ab.y * ab.y
    if ab2 == 0.0:
        return a.copy()
    t = (av.x * ab.x + av.y * ab.y) / ab2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return Vertex(a.x + ab.x * t, a.y + ab.y * t, a.z + ab.z * t)

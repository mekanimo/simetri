"""Polygon utilities: area, winding, simplicity, and related helpers."""

from collections.abc import Sequence
from math import isclose, pi, sin

import numpy as np

from simetri.base.all_enums import Connection
from simetri.base.common import PointType, get_defaults
from simetri.geom.points.point_utils import (
    close_points_square,
    remove_bad_points,
)
from simetri.geom.segments.line_utils import (
    intersection,
    check_intersection,
)
from simetri.geom.vectors import cross_product_sense3, distance
from simetri.helpers.utilities import reg_poly_points
from simetri.config.settings import defaults


def right_handed(polygon: Sequence[PointType], dist_tol=None) -> float:
    """Return True if the polygon walk is counter-clockwise.

    The test is the sign of the shoelace sum. An unclosed ring is closed
    on a copy for that sum.

    Args:
        polygon (Sequence[PointType]): Vertices in walk order.
        dist_tol (float, optional): Distance used to decide whether the
            ring is already closed. Defaults to ``defaults["dist_tol"]``.

    Returns:
        bool: True if the walk is counter-clockwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.right_handed([(0, 0), (1, 0), (1, 1), (0, 1)])
        True
        >>> sg.right_handed([(0, 0), (0, 1), (1, 1), (1, 0)])
        False
        >>> sg.right_handed([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        True
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    dist_tol2 = dist_tol * dist_tol
    if close_points_square(polygon[0], polygon[-1], dist2=dist_tol2):
        poly = polygon
    else:
        poly = list(polygon) + [polygon[0]]
    area_ = 0
    for i, point in enumerate(poly[:-1]):
        x1, y1 = point[:2]
        x2, y2 = poly[i + 1][:2]
        area_ += x1 * y2 - x2 * y1
    return area_ > 0


def is_simple(
    polygon,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if ``polygon`` does not cross or overlap itself.

    A shared endpoint of consecutive edges is allowed. A proper crossing,
    a collinear overlap, or a Y-joint is not. The vertices are copied
    before the ring is closed.

    Args:
        polygon: Ordered polygon vertices.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        bool: True if the polygon is simple, False otherwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.geom.polygons.polygon_utils import is_simple
        >>> is_simple([(0, 0), (2, 0), (2, 2), (0, 2)])
        True
        >>> is_simple([(0, 0), (2, 2), (2, 0), (0, 2)])
        False
        >>> square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        >>> is_simple(square)
        True
        >>> square
        [(0, 0), (1, 0), (1, 1), (0, 1)]
    """
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])

    vertices = [point[:2] for point in polygon]
    if not close_points_square(vertices[0], vertices[-1]):
        vertices.append(vertices[0])
    segments = [
        [vertices[i], vertices[i + 1]] for i in range(len(vertices) - 1)
    ]

    segment_coords = []
    for segment in segments:
        segment_coords.append(
            [segment[0][0], segment[0][1], segment[1][0], segment[1][1]]
        )
    seg_arr = np.array(segment_coords)
    n_rows = seg_arr.shape[0]
    xmin = np.minimum(seg_arr[:, 0], seg_arr[:, 2]).reshape(n_rows, 1)
    xmax = np.maximum(seg_arr[:, 0], seg_arr[:, 2]).reshape(n_rows, 1)
    ymin = np.minimum(seg_arr[:, 1], seg_arr[:, 3]).reshape(n_rows, 1)
    ymax = np.maximum(seg_arr[:, 1], seg_arr[:, 3]).reshape(n_rows, 1)
    id_ = np.arange(n_rows).reshape(n_rows, 1)
    seg_arr = np.concatenate((seg_arr, xmin, ymin, xmax, ymax, id_), 1)
    seg_arr = seg_arr[seg_arr[:, 4].argsort()]
    i_xmin, i_ymin, i_xmax, i_ymax, i_id = range(4, 9)

    s_processed = set()
    for i in range(n_rows):
        x1, y1, x2, y2, sl_xmin, sl_ymin, sl_xmax, sl_ymax, id1 = seg_arr[i, :]
        id1 = int(id1)
        segment = [x1, y1, x2, y2]
        start = i + 1
        candidates = seg_arr[start:, :][
            (
                (
                    (seg_arr[start:, i_xmax] >= sl_xmin)
                    & (seg_arr[start:, i_xmin] <= sl_xmax)
                )
                & (
                    (seg_arr[start:, i_ymax] >= sl_ymin)
                    & (seg_arr[start:, i_ymin] <= sl_ymax)
                )
            )
        ]
        for cand in candidates:
            id2 = int(cand[i_id])
            pair = frozenset((id1, id2))
            if pair in s_processed:
                continue
            s_processed.add(pair)
            seg2 = cand[:4]
            x1, y1, x2, y2 = segment
            x3, y3, x4, y4 = seg2
            res = check_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
            if res[0] == Connection.COLL_CHAIN:
                length1 = distance((x1, y1), (x2, y2))
                length2 = distance((x3, y3), (x4, y4))
                p1, p2 = res[1][0], res[1][2]
                chain_length = distance(p1, p2)
                if not isclose(
                    length1 + length2,
                    chain_length,
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                ):
                    return False
                continue
            if res[0] in (Connection.CHAIN, Connection.PARALLEL):
                continue
            if res[0] != Connection.DISJOINT:
                return False

    return True


def get_polygon_grid_point(n, line1, line2, circumradius=100):
    """Return the intersection of two chords of a regular polygon.

    ``line1`` and ``line2`` are index pairs into the polygon vertices.
    The polygon is centered at the origin with the given circumradius.

    Args:
        n (int): Number of sides.
        line1: ``(start_index, end_index)`` of the first chord.
        line2: ``(start_index, end_index)`` of the second chord.
        circumradius (float, optional): Circumradius. Defaults to 100.

    Returns:
        PointType: Intersection of the two chords.

    Examples:
        >>> from simetri.geom.polygons.polygon_utils import get_polygon_grid_point
        >>> get_polygon_grid_point(4, (0, 2), (1, 3))
        (0.0, 0.0)
        >>> get_polygon_grid_point(4, (0, 1), (1, 2), circumradius=100)[1]
        100.0
    """
    s = circumradius * 2 * sin(pi / n)  # side length
    points = reg_poly_points(0, 0, n, s)[:-1]
    p1 = points[line1[0]]
    p2 = points[line1[1]]
    p3 = points[line2[0]]
    p4 = points[line2[1]]

    return intersection((p1, p2), (p3, p4))[1]


def is_ccw(vertices, *, eps=0.0):
    """Return True if polygon vertices are in counter-clockwise order.

    The test is the sign of the shoelace sum. ``eps`` is accepted and
    not used.

    Args:
        vertices: Vertices in walk order.
        eps: Accepted and not used.

    Returns:
        bool: True if the shoelace sum is positive.

    Raises:
        ValueError: If fewer than 3 vertices are provided.

    Examples:
        >>> from simetri.geom.polygons.polygon_utils import is_ccw
        >>> is_ccw([(0, 0), (1, 0), (1, 1), (0, 1)])
        True
        >>> is_ccw([(0, 0), (0, 1), (1, 1), (1, 0)])
        False
        >>> is_ccw([(0, 0), (1, 0)])
        Traceback (most recent call last):
            ...
        ValueError: Need at least 3 vertices
    """
    n = len(vertices)
    if n < 3:
        raise ValueError("Need at least 3 vertices")

    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1  # 2 * signed area

    return area > 0


def calc_area(points):
    """Return the absolute area and whether the walk is counter-clockwise.

    The second value is True when the shoelace sum is positive. That is
    a counter-clockwise walk, not a clockwise one.

    Args:
        points: Vertices in walk order. The ring need not repeat the first
            vertex.

    Returns:
        tuple[float, bool]: Absolute area, and True if the walk is
        counter-clockwise.

    Examples:
        >>> from simetri.geom.polygons.polygon_utils import calc_area
        >>> calc_area([(0, 0), (1, 0), (1, 1), (0, 1)])
        (1.0, True)
        >>> calc_area([(0, 0), (0, 1), (1, 1), (1, 0)])
        (1.0, False)
    """
    area_ = 0
    n_points = len(points)
    for i in range(n_points):
        v = points[i]
        vnext = points[(i + 1) % n_points]
        area_ += v[0] * vnext[1] - vnext[0] * v[1]
    clockwise = area_ > 0

    return (abs(area_ / 2.0), clockwise)


def is_convex(points):
    """Return True if the polygon is convex.

    This calls ``remove_bad_points``, which removes collinear and
    repeated points from ``points`` in place. The polygon is convex when
    every turn has the same sense.

    Args:
        points (list[PointType]): Polygon vertices (mutated).

    Returns:
        bool: True if every turn has the same sense.

    Examples:
        >>> from simetri.geom.polygons.polygon_utils import is_convex
        >>> is_convex([(0, 0), (1, 0), (1, 1), (0, 1)])
        True
        >>> is_convex([(0, 0), (3, 0), (1, 1), (0, 3)])
        False
    """
    points = remove_bad_points(points)
    n_checks = len(points)
    points = points + [points[0]]
    senses = []
    for i in range(n_checks):
        if i == (n_checks - 1):
            senses.append(cross_product_sense3(points[i], points[0], points[1]))
        else:
            senses.append(
                cross_product_sense3(points[i], points[i + 1], points[i + 2])
            )
    s = set(senses)
    return len(s) == 1

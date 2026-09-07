"""Polygon utilities: area, winding, simplicity, and related helpers."""

from collections.abc import Sequence
from math import isclose, pi, sin

import numpy as np

from simetri.base.all_enums import Connection
from simetri.base.common import PointType, get_defaults
from simetri.geom.points.point_utils import close_points2, remove_bad_points
from simetri.geom.segments.line_utils import (
    intersection,
    intersection3,
    sorted_edges,
)
from simetri.geom.vectors import cross_product_sense, distance, sin
from simetri.helpers.utilities import reg_poly_points
from simetri.config.settings import defaults


def right_handed(polygon: Sequence[PointType], dist_tol=None) -> float:
    """If polygon is counter-clockwise, return True

    Args:
        polygon (Sequence[PointType]): List of points representing the polygon.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        bool: True if the polygon is counter-clockwise, False otherwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.right_handed([(0, 0), (1, 0), (1, 1), (0, 1)])
        True
        >>> sg.right_handed([(0, 0), (0, 1), (1, 1), (1, 0)])
        False
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    dist_tol2 = dist_tol * dist_tol
    if close_points2(polygon[0], polygon[-1], dist2=dist_tol2):
        poly = polygon
    else:
        poly = list(polygon) + [polygon[0]]
    area_ = 0
    for i, point in enumerate(poly[:-1]):
        x1, y1 = point[:2]
        x2, y2 = poly[i + 1][:2]
        area_ += x1 * y2 - x2 * y1
    return area_ > 0


def is_simple2(
    polygon,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """
    Return True if the polygon is simple.

    Args:
        polygon (list): List of points representing the polygon.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        bool: True if the polygon is simple, False otherwise.
    """
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])

    if not close_points2(polygon[0], polygon[-1]):
        polygon.append(polygon[0])
    segments = [[polygon[i], polygon[i + 1]] for i in range(len(polygon) - 1)]

    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    segment_coords = []
    for segment in segments:
        segment_coords.append(
            [segment[0][0], segment[0][1], segment[1][0], segment[1][1]]
        )
    seg_arr = np.array(segment_coords)  # segments array
    n_rows = seg_arr.shape[0]
    xmin = np.minimum(seg_arr[:, 0], seg_arr[:, 2]).reshape(n_rows, 1)
    xmax = np.maximum(seg_arr[:, 0], seg_arr[:, 2]).reshape(n_rows, 1)
    ymin = np.minimum(seg_arr[:, 1], seg_arr[:, 3]).reshape(n_rows, 1)
    ymax = np.maximum(seg_arr[:, 1], seg_arr[:, 3]).reshape(n_rows, 1)
    id_ = np.arange(n_rows).reshape(n_rows, 1)
    seg_arr = np.concatenate((seg_arr, xmin, ymin, xmax, ymax, id_), 1)
    seg_arr = seg_arr[seg_arr[:, 4].argsort()]
    i_xmin, i_ymin, i_xmax, i_ymax, i_id = range(4, 9)  # column indices

    s_processed = set()  # set of processed segment pairs
    for i in range(n_rows):
        x1, y1, x2, y2, sl_xmin, sl_ymin, sl_xmax, sl_ymax, id1 = seg_arr[i, :]
        id1 = int(id1)
        segment = [x1, y1, x2, y2]
        start = i + 1  # keep pushing the sweep line forward
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
            res = intersection3(x1, y1, x2, y2, x3, y3, x4, y4)
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
                else:
                    continue
            if res[0] in (Connection.CHAIN, Connection.PARALLEL):
                continue
            if res[0] != Connection.DISJOINT:
                return False

    return True


def is_simple(polygon):
    """Return True if ``polygon`` is simple (non-self-intersecting).

    Uses a sweep-line style test. Polygons with more than 100 vertices
    are delegated to ``is_simple2``.

    Args:
        polygon: Ordered polygon vertices.

    Returns:
        bool: True if the polygon does not self-intersect.
    """
    if len(polygon) > 100:
        return is_simple2(polygon)

    queue = []
    points = []
    edges = sorted_edges(polygon)

    points = [tuple(p) for edge in edges for p in edge]
    points = list(set(points))
    points.sort()

    for p in points:
        for edge in edges:
            if p == tuple(edge[0]):
                for e in queue:
                    if e[0] in edge or e[1] in edge:
                        continue
                    res = intersection(e, edge)

                    if res[0] == Connection.INTERSECT:
                        return False
                queue.append(edge)
            elif p == tuple(edge[1]):
                queue.remove(edge)

    return True


def get_polygon_grid_point(n, line1, line2, circumradius=100):
    """See chapter ??? for explanation of this function.

    Args:
        n (int): Number of sides.
        line1 (LineType): First line.
        line2 (LineType): Second line.
        circumradius (float, optional): Circumradius of the polygon. Defaults to 100.

    Returns:
        PointType: Grid point of the polygon.
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

    Args:
        vertices: List of ``(x, y)`` without repeating the first vertex at
            the end.
        eps: Tolerance; if ``abs(signed_area) <= eps``, returns False
            (degenerate).

    Returns:
        bool: True for CCW orientation.

    Raises:
        ValueError: If fewer than 3 vertices are provided.
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
    """Calculate the area of a simple polygon (given by a list of its vertices).

    Args:
        points (list[PointType]): List of points representing the polygon.

    Returns:
        tuple: Area of the polygon and whether it is clockwise.
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

    Args:
        points (list[PointType]): List of points representing the polygon.

    Returns:
        bool: True if the polygon is convex, False otherwise.
    """
    points = remove_bad_points(points)
    n_checks = len(points)
    points = points + [points[0]]
    senses = []
    for i in range(n_checks):
        if i == (n_checks - 1):
            senses.append(cross_product_sense(points[i], points[0], points[1]))
        else:
            senses.append(
                cross_product_sense(points[i], points[i + 1], points[i + 2])
            )
    s = set(senses)
    return len(s) == 1

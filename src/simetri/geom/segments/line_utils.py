"""Line and segment utilities: intersections, offsets, fillets, and angles."""

from collections.abc import Sequence
from functools import cmp_to_key
from math import acos, atan2, cos, isclose, pi, sin, tan

import numpy as np
from numpy import array

from simetri.base.all_enums import Connection, Types
from simetri.base.common import LineType, PointType, get_defaults
from simetri.geom.geometry import (
    positive_angle,
)
from simetri.geom.geom_utils import connected_pairs
from simetri.geom.geometry import double_area, bbox_overlap
from simetri.geom.geom_utils import (
    close_points2,
    midpoint,
    offset_point_from_start,
)
from simetri.geom.points.point_utils import (
    Vertex,
    between,
    clockwise,
    direction,
    distance,
    equal_points,
    point_on_line_segment,
    round_point,
)
from simetri.geom.vectors import (
    LineType,
    PointType,
    Sequence,
    Vector,
    i_vec,
    j_vec,
    acos,
    atan2,
    cos,
    distance,
    line_vector,
    perp_unit_vector,
    sin,
    v_cross,
    v_from_points,
    v_mul,
)
from simetri.config.settings import defaults


def equal_edges(edge1: LineType, edge2: LineType, dist_tol=0.001) -> bool:
    """Return True if two edges have matching endpoints (either orientation).

    Args:
        edge1: First edge ``(p1, p2)``.
        edge2: Second edge ``(p3, p4)``.
        dist_tol: Endpoint distance tolerance. Defaults to 0.001.

    Returns:
        bool: True if endpoints match within ``dist_tol``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.equal_edges([(0, 0), (1, 0)], [(1, 0), (0, 0)])
        True
        >>> sg.equal_edges([(0, 0), (1, 0)], [(0, 0), (1, 1)])
        False
    """
    p1, p2 = edge1
    p3, p4 = edge2

    return (
        equal_points(p1, p3, dist_tol) and equal_points(p2, p4, dist_tol)
    ) or (equal_points(p1, p4, dist_tol) and equal_points(p2, p3, dist_tol))


# alias for equal_edges
def equal_segments(edge1: LineType, edge2: LineType, dist_tol=0.001) -> bool:
    """Alias for ``equal_edges``.

    Args:
        edge1: First segment.
        edge2: Second segment.
        dist_tol: Endpoint distance tolerance. Defaults to 0.001.

    Returns:
        bool: True if endpoints match within ``dist_tol``.
    """

    return equal_edges(edge1, edge2, dist_tol=dist_tol)


# alias for equal_edges
def congruent_edges(edge1: LineType, edge2: LineType, dist_tol=0.001) -> bool:
    """Alias for ``equal_edges``.

    Args:
        edge1: First edge.
        edge2: Second edge.
        dist_tol: Endpoint distance tolerance. Defaults to 0.001.

    Returns:
        bool: True if endpoints match within ``dist_tol``.
    """

    return equal_edges(edge1, edge2, dist_tol=dist_tol)


# alias for equal_edges
def congruent_segments(
    edge1: LineType, edge2: LineType, dist_tol=0.001
) -> bool:
    """Alias for ``equal_edges``.

    Args:
        edge1: First segment.
        edge2: Second segment.
        dist_tol: Endpoint distance tolerance. Defaults to 0.001.

    Returns:
        bool: True if endpoints match within ``dist_tol``.
    """

    return equal_edges(edge1, edge2, dist_tol=dist_tol)


def line_angle(start_point: PointType, end_point: PointType) -> float:
    """Return the orientation angle (in radians) of a line given by start and end points.
    Order makes a difference.

    Args:
        start_point (PointType): Start point of the line.
        end_point (PointType): End point of the line.

    Returns:
        float: Orientation angle of the line in radians.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.line_angle((0, 0), (1, 0))
        0.0
        >>> sg.line_angle((0, 0), (0, 1)) == sg.pi / 2
        True
    """
    return positive_angle(
        atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
    )


def angled_line(line: LineType, theta: float) -> LineType:
    """
    Given a line find another line with theta radians between them.

    Args:
        line (LineType): Input line.
        theta (float): Angle in radians.

    Returns:
        LineType: New line with the given angle.

    Examples:
        >>> import simetri.graphics as sg
        >>> p0, p1 = sg.angled_line([(0, 0), (1, 0)], sg.pi / 2)
        >>> p0, (round(p1[0], 10), round(p1[1], 10))
        ((0, 0), (0.0, 1.0))
    """
    # find the angle of the line
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    theta1 = atan2(y2 - y1, x2 - x1)
    theta2 = theta1 + theta
    # find the length of the line
    dx = x2 - x1
    dy = y2 - y1
    length_ = (dx**2 + dy**2) ** 0.5
    # find the new line
    x3 = x1 + length_ * cos(theta2)
    y3 = y1 + length_ * sin(theta2)

    return [(x1, y1), (x3, y3)]


def offset_line(
    line: Sequence[PointType], offset: float
) -> Sequence[PointType]:
    """Return an offset line from a given line.

    Args:
        line (Sequence[PointType]): Input line.
        offset (float): Offset distance.

    Returns:
        Sequence[PointType]: Offset line.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.offset_line([(0, 0), (1, 0)], 1)
        [[0.0, 1.0], [1.0, 1.0]]
    """
    unit_vec = perp_unit_vector(line)
    dx = unit_vec[0] * offset
    dy = unit_vec[1] * offset
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    return [[x1 + dx, y1 + dy], [x2 + dx, y2 + dy]]


def offset_lines(
    polylines: Sequence[LineType], offset: float = 1
) -> list[LineType]:
    """Return a list of offset lines from a list of lines.

    Args:
        polylines (Sequence[LineType]): List of input lines.
        offset (float, optional): Offset distance. Defaults to 1.

    Returns:
        list[LineType]: List of offset lines.
    """

    def stitch_(polyline):
        res = []
        line1 = polyline[0]
        for i, _ in enumerate(polyline):
            if i == len(polyline) - 1:
                break
            line2 = polyline[i + 1]
            line1, line2 = stitch_lines(line1, line2)
            res.extend(line1)
            line1 = line2
        res.append(line2[-1])
        return res

    poly = []
    for line in polylines:
        poly.append(offset_line(line, offset))
    poly = stitch_(poly)
    return poly


def parallel_line(line: LineType, point: PointType) -> LineType:
    """Return a parallel line to the given line that goes through the given point

    Args:
        line (LineType): Input line.
        point (PointType): PointType through which the parallel line passes.

    Returns:
        LineType: Parallel line.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.parallel_line([(0, 0), (1, 0)], (0, 2))
        [[0, 2], [1, 2]]
    """
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    x3, y3 = point
    dx = x2 - x1
    dy = y2 - y1
    return [[x3, y3], [x3 + dx, y3 + dy]]


def perp_bisector(line: LineType) -> LineType:
    """Return the perpendicular bisector of a line

    Args:
        line (LineType): Input line.

    Returns:
        LineType: Perpendicular bisector of the line.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.perp_bisector([(0, 0), (2, 0)])
        [(1.0, 0.0), [1.0, 2.0]]
    """
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    mid = midpoint(line[0], line[1])
    dx = x2 - x1
    dy = y2 - y1
    return [mid, [mid[0] - dy, mid[1] + dx]]


def collinear(a, b, c, area_tol=None):
    """Return True if a, b, and c are collinear.

    Args:
        a (PointType): First point.
        b (PointType): Second point.
        c (PointType): Third point.
        area_rtol (float, optional): Relative tolerance for area. Defaults to None.
        area_atol (float, optional): Absolute tolerance for area. Defaults to None.

    Returns:
        bool: True if the points are collinear, False otherwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.collinear((0, 0), (1, 1), (2, 2))
        True
        >>> sg.collinear((0, 0), (1, 0), (0, 1))
        False
    """
    if area_tol is None:
        area_tol = defaults["area_tol"]

    return abs(double_area(a, b, c)) <= area_tol


def merge_consecutive_collinear_edges(
    points, closed=False, area_rtol=None, area_atol=None
):
    """Remove the middle points from collinear edges.

    Args:
        points (list[PointType]): List of points.
        closed (bool, optional): Whether the points form a closed shape. Defaults to False.
        area_rtol (float, optional): Relative tolerance for area. Defaults to None.
        area_atol (float, optional): Absolute tolerance for area. Defaults to None.

    Returns:
        list[PointType]: List of points with collinear points removed.
    """
    area_rtol, area_atol = get_defaults(
        ["area_rtol", "area_atol"], [area_rtol, area_atol]
    )
    points = points[:]

    while True:
        cyc = cycle(points)
        a = next(cyc)
        b = next(cyc)
        c = next(cyc)
        looping = False
        n = len(points) - 1
        if closed:
            n += 1
        discarded = []
        for _ in range(n - 1):
            if collinear(a, b, c, area_rtol=area_rtol, area_atol=area_atol):
                discarded.append(b)
                looping = True
                break
            a = b
            b = c
            c = next(cyc)
        for point in discarded:
            points.remove(point)
        if not looping or len(points) < 3:
            break

    return points


def round_segment(segment: Sequence[PointType], n_digits: int = 2):
    """Round a segment to a given precision.

    Args:
        segment (Sequence[PointType]): Input segment.
        n_digits (int, optional): Number of decimal places to round to. Defaults to 2.

    Returns:
        Sequence[PointType]: Rounded segment.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.round_segment([(1.26, 2.34), (3.56, 4.78)], 1)
        [(1.3, 2.3), (3.6, 4.8)]
    """
    p1 = round_point(segment[0], n_digits)
    p2 = round_point(segment[1], n_digits)

    return [p1, p2]


def line_segment_bbox(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float, float]:
    """
    Return the bounding box of a line segment.

    Args:
        x1 (float): Segment start point x-coordinate.
        y1 (float): Segment start point y-coordinate.
        x2 (float): Segment end point x-coordinate.
        y2 (float): Segment end point y-coordinate.

    Returns:
        tuple: Bounding box as (min_x, min_y, max_x, max_y).

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.line_segment_bbox(1, 3, 2, 0)
        (1, 0, 2, 3)
    """
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def line_segment_bbox_check(seg1: LineType, seg2: LineType) -> bool:
    """
    Given two line segments, return True if their bounding boxes overlap.

    Args:
        seg1 (LineType): First line segment.
        seg2 (LineType): Second line segment.

    Returns:
        bool: True if the bounding boxes overlap, False otherwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.line_segment_bbox_check([(0, 0), (2, 0)], [(1, -1), (1, 1)])
        True
    """
    x1, y1 = seg1[0][:2]
    x2, y2 = seg1[1][:2]
    x3, y3 = seg2[0][:2]
    x4, y4 = seg2[1][:2]
    return bbox_overlap(
        *line_segment_bbox(x1, y1, x2, y2), *line_segment_bbox(x3, y3, x4, y4)
    )


def intersect2(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> PointType:
    """Return the intersection point of two lines.
    line1: (x1, y1), (x2, y2)
    line2: (x3, y3), (x4, y4)
    To find the intersection point of two line segments use the
    "intersection" function

    Args:
        x1 (float): x-coordinate of the first point of the first line.
        y1 (float): y-coordinate of the first point of the first line.
        x2 (float): x-coordinate of the second point of the first line.
        y2 (float): y-coordinate of the second point of the first line.
        x3 (float): x-coordinate of the first point of the second line.
        y3 (float): y-coordinate of the first point of the second line.
        x4 (float): x-coordinate of the second point of the second line.
        y4 (float): y-coordinate of the second point of the second line.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        PointType: Intersection point of the two lines.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.intersect2(0, 0, 2, 2, 0, 2, 2, 0)
        (1.0, 1.0)
    """
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    x1_x2 = x1 - x2
    y1_y2 = y1 - y2
    x3_x4 = x3 - x4
    y3_y4 = y3 - y4

    denom = (x1_x2) * (y3_y4) - (y1_y2) * (x3_x4)
    if isclose(denom, 0, rel_tol=rel_tol, abs_tol=abs_tol):
        res = None  # parallel lines
    else:
        x = (
            (x1 * y2 - y1 * x2) * (x3_x4) - (x1_x2) * (x3 * y4 - y3 * x4)
        ) / denom
        y = (
            (x1 * y2 - y1 * x2) * (y3_y4) - (y1_y2) * (x3 * y4 - y3 * x4)
        ) / denom
        res = (x, y)

    return res


def intersect(line1: LineType, line2: LineType) -> PointType:
    """Return the intersection point of two lines.
    line1: [(x1, y1), (x2, y2)]
    line2: [(x3, y3), (x4, y4)]
    To find the intersection point of two line segments use the
    "intersection" function

    Args:
        line1 (LineType): First line.
        line2 (LineType): Second line.

    Returns:
        PointType: Intersection point of the two lines.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.intersect([(0, 0), (2, 2)], [(0, 2), (2, 0)])
        (1.0, 1.0)
    """
    x1, y1 = line1[0][:2]
    x2, y2 = line1[1][:2]
    x3, y3 = line2[0][:2]
    x4, y4 = line2[1][:2]
    return intersect2(x1, y1, x2, y2, x3, y3, x4, y4)


def intersection2(x1, y1, x2, y2, x3, y3, x4, y4, rel_tol=None, abs_tol=None):
    """Check the intersection of two line segments. See the documentation

    Args:
        x1 (float): x-coordinate of the first point of the first line segment.
        y1 (float): y-coordinate of the first point of the first line segment.
        x2 (float): x-coordinate of the second point of the first line segment.
        y2 (float): y-coordinate of the second point of the first line segment.
        x3 (float): x-coordinate of the first point of the second line segment.
        y3 (float): y-coordinate of the first point of the second line segment.
        x4 (float): x-coordinate of the second point of the second line segment.
        y4 (float): y-coordinate of the second point of the second line segment.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        tuple: Connection type and intersection point.

    Examples:
        >>> import simetri.graphics as sg
        >>> kind, point = sg.intersection2(0, 0, 2, 2, 0, 2, 2, 0)
        >>> kind == sg.Connection.INTERSECT, point
        (True, (1.0, 1.0))
        >>> kind, _ = sg.intersection2(0, 0, 1, 0, 0, 1, 1, 1)
        >>> kind == sg.Connection.PARALLEL
        True
    """
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    x2_x1 = x2 - x1
    y2_y1 = y2 - y1
    x4_x3 = x4 - x3
    y4_y3 = y4 - y3
    denom = (y4_y3) * (x2_x1) - (x4_x3) * (y2_y1)
    if isclose(denom, 0, rel_tol=rel_tol, abs_tol=abs_tol):  # parallel
        return Connection.PARALLEL, None
    x1_x3 = x1 - x3
    y1_y3 = y1 - y3
    ua = ((x4_x3) * (y1_y3) - (y4_y3) * (x1_x3)) / denom
    if ua < 0 or ua > 1:
        return Connection.DISJOINT, None
    ub = ((x2_x1) * (y1_y3) - (y2_y1) * (x1_x3)) / denom
    if ub < 0 or ub > 1:
        return Connection.DISJOINT, None
    x = x1 + ua * (x2_x1)
    y = y1 + ua * (y2_y1)
    return Connection.INTERSECT, (x, y)


def collinear_segments(segment1, segment2, rel_tol=None, abs_tol=None):
    """
    Checks if two line segments (a1, b1) and (a2, b2) are collinear.

    Args:
        segment1 (LineType): First line segment.
        segment2 (LineType): Second line segment.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        bool: True if the segments are collinear, False otherwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.collinear_segments([(0, 0), (2, 0)], [(1, 0), (3, 0)])
        True
        >>> sg.collinear_segments([(0, 0), (2, 0)], [(0, 1), (2, 1)])
        False
    """
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    a1, b1 = segment1
    a2, b2 = segment2

    return isclose(
        direction(a1, b1, a2), 0, rel_tol=rel_tol, abs_tol=abs_tol
    ) and isclose(direction(a1, b1, b2), 0, rel_tol=rel_tol, abs_tol=abs_tol)


def intersection3(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
    dist_tol: float | None = None,
    area_atol: float | None = None,
) -> tuple[Connection, list]:
    """Check the intersection of two line segments. See the documentation
    for more details.

    Args:
        x1 (float): x-coordinate of the first point of the first line segment.
        y1 (float): y-coordinate of the first point of the first line segment.
        x2 (float): x-coordinate of the second point of the first line segment.
        y2 (float): y-coordinate of the second point of the first line segment.
        x3 (float): x-coordinate of the first point of the second line segment.
        y3 (float): y-coordinate of the first point of the second line segment.
        x4 (float): x-coordinate of the second point of the second line segment.
        y4 (float): y-coordinate of the second point of the second line segment.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.
        dist_tol (float, optional): Distance tolerance. Defaults to None.
        area_atol (float, optional): Absolute tolerance for area. Defaults to None.

    Returns:
        tuple: Connection type and intersection result.

    Examples:
        >>> import simetri.graphics as sg
        >>> kind, point = sg.intersection3(0, 0, 2, 2, 0, 2, 2, 0)
        >>> kind == sg.Connection.INTERSECT, point
        (True, (1.0, 1.0))
    """
    # collinear check uses area_atol

    # s1: start1 = (x1, y1)
    # e1: end1 = (x2, y2)
    # s2: start2 = (x3, y3)
    # e2: end2 = (x4, y4)
    # s1s2: start1 and start2 is connected
    # s1e2: start1 and end2 is connected
    # e1s2: end1 and start2 is connected
    # e1e2: end1 and end2 is connected
    rel_tol, abs_tol, dist_tol, area_atol = get_defaults(
        ["rel_tol", "abs_tol", "dist_tol", "area_atol"],
        [rel_tol, abs_tol, dist_tol, area_atol],
    )

    s1 = (x1, y1)
    e1 = (x2, y2)
    s2 = (x3, y3)
    e2 = (x4, y4)
    segment1 = [(x1, y1), (x2, y2)]
    segment2 = [(x3, y3), (x4, y4)]

    # check if the segments' bounding boxes overlap
    if not line_segment_bbox_check(segment1, segment2):
        return (Connection.DISJOINT, None)

    # Check if the segments are parallel
    x2_x1 = x2 - x1
    y2_y1 = y2 - y1
    x4_x3 = x4 - x3
    y4_y3 = y4 - y3
    denom = (y4_y3) * (x2_x1) - (x4_x3) * (y2_y1)
    parallel = isclose(denom, 0, rel_tol=rel_tol, abs_tol=abs_tol)
    # angle1 = atan2(y2 - y1, x2 - x1) % pi
    # angle2 = atan2(y4 - y3, x4 - x3) % pi
    # parallel = close_angles(angle1, angle2, angtol=defaults['angtol'])

    # Coincident end points
    dist_tol2 = dist_tol * dist_tol
    s1s2 = close_points2(s1, s2, dist2=dist_tol2)
    s1e2 = close_points2(s1, e2, dist2=dist_tol2)
    e1s2 = close_points2(e1, s2, dist2=dist_tol2)
    e1e2 = close_points2(e1, e2, dist2=dist_tol2)
    connected = s1s2 or s1e2 or e1s2 or e1e2
    if parallel:
        length1 = distance((x1, y1), (x2, y2))
        length2 = distance((x3, y3), (x4, y4))
        min_x = min(x1, x2, x3, x4)
        max_x = max(x1, x2, x3, x4)
        min_y = min(y1, y2, y3, y4)
        max_y = max(y1, y2, y3, y4)
        total_length = distance((min_x, min_y), (max_x, max_y))
        l1_eq_l2 = isclose(length1, length2, rel_tol=rel_tol, abs_tol=abs_tol)
        l1_eq_total = isclose(
            length1, total_length, rel_tol=rel_tol, abs_tol=abs_tol
        )
        l2_eq_total = isclose(
            length2, total_length, rel_tol=rel_tol, abs_tol=abs_tol
        )
        if connected:
            if l1_eq_l2 and l1_eq_total:
                return Connection.CONGRUENT, segment1

            if l1_eq_total:
                return Connection.CONTAINS, segment1
            if l2_eq_total:
                return Connection.WITHIN, segment2
            if isclose(
                length1 + length2,
                total_length,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            ):
                # chained and collienar
                if s1s2:
                    return Connection.COLL_CHAIN, (e1, s1, e2)
                if s1e2:
                    return Connection.COLL_CHAIN, (e1, s1, s2)
                if e1s2:
                    return Connection.COLL_CHAIN, (s1, s2, e2)
                if e1e2:
                    return Connection.COLL_CHAIN, (s1, e1, s2)
        else:
            if total_length < length1 + length2 and collinear_segments(
                segment1, segment2, abs_tol
            ):
                p1 = (min_x, min_y)
                p2 = (max_x, max_y)
                seg = [p1, p2]
                return Connection.OVERLAPS, seg

            return intersection2(
                x1, y1, x2, y2, x3, y3, x4, y4, rel_tol, abs_tol
            )
    else:
        if connected:
            if s1s2:
                return Connection.CHAIN, (e1, s1, e2)
            if s1e2:
                return Connection.CHAIN, (e1, s1, s2)
            if e1s2:
                return Connection.CHAIN, (s1, s2, e2)
            if e1e2:
                return Connection.CHAIN, (s1, e1, s2)
        else:
            if between(s1, e1, e2):
                return Connection.YJOINT, e1
            if between(s1, e1, s2):
                return Connection.YJOINT, s1
            if between(s2, e2, e1):
                return Connection.YJOINT, e2
            if between(s2, e2, s1):
                return Connection.YJOINT, s2

            return intersection2(
                x1, y1, x2, y2, x3, y3, x4, y4, rel_tol, abs_tol
            )
    return (Connection.DISJOINT, None)


def inclination_angle(start_point: PointType, end_point: PointType) -> float:
    """Return the inclination angle (in radians) of a line given by start and end points.
    Inclination angle is always between zero and pi.
    Order makes no difference.

    Args:
        start_point (PointType): Start point of the line.
        end_point (PointType): End point of the line.

    Returns:
        float: Inclination angle of the line in radians.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.inclination_angle((0, 0), (1, 1)) == sg.pi / 4
        True
        >>> sg.inclination_angle((1, 1), (0, 0)) == sg.pi / 4
        True
    """
    return line_angle(start_point, end_point) % pi


def clip_line_to_rect(point, direction, lower_left, upper_right):
    """Clip an infinite line against an axis-aligned rectangle.

    The line is defined by a point and a direction vector.

    Args:
        point: A point on the line ``(x, y)``.
        direction: Direction vector ``(dx, dy)``.
        lower_left: Rectangle lower-left corner.
        upper_right: Rectangle upper-right corner.

    Returns:
        tuple | None: Clipped segment ``((x1, y1), (x2, y2))``, or None if
        the line misses the rectangle.
    """
    x_min, y_min = lower_left[:2]
    x_max, y_max = upper_right[:2]
    rectangle = (x_min, y_min, x_max, y_max)
    t_min, t_max = -float("inf"), float("inf")

    # Iterate over x and y dimensions
    for i in range(2):
        if direction[i] == 0:
            if point[i] < rectangle[i] or point[i] > rectangle[i + 2]:
                return None
        else:
            t1 = (rectangle[i] - point[i]) / direction[i]
            t2 = (rectangle[i + 2] - point[i]) / direction[i]
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))

    if t_min <= t_max:
        start = (
            point[0] + t_min * direction[0],
            point[1] + t_min * direction[1],
        )
        end = (point[0] + t_max * direction[0], point[1] + t_max * direction[1])
        return (start, end)
    return None


def sorted_edges(polygon):
    """Return polygon edges sorted for sweep-line processing.

    Edges are oriented left-to-right, then sorted by increasing start ``x``
    and ``y``. Used by simplicity tests.

    Args:
        polygon: Sequence of polygon vertices.

    Returns:
        list: Oriented and sorted edges ``[(p1, p2), ...]``.
    """

    # order the edges:increasing x coordinates then increasing y coordinates for the start points
    # this is used for line sweep algorithm to check if the polygon is simple
    def get_edges(polygon):
        edges = []
        for i, p in enumerate(polygon[:-1]):
            np = polygon[i + 1]  # next point
            edges.append((p, np))
        p = polygon[-1]
        np = polygon[0]
        edges.append((p, np))
        return edges

    def compare_edges(edge1, edge2):
        x1 = edge1[0][0]
        x2 = edge2[0][0]
        y1 = edge1[0][1]
        y2 = edge2[0][1]
        if x1 < x2:
            return -1
        elif x1 > x2:
            return 1
        else:
            if y1 < y2:
                return -1
            elif y1 > y2:
                return 1
            else:
                return 0

    edges = get_edges(polygon)
    oriented_edges = []
    for edge in edges:
        start_x, start_y = edge[0][:2]
        end_x, end_y = edge[1][:2]
        if start_x > end_x or (start_x == end_x and start_y > end_y):
            oriented_edges.append((edge[1], edge[0]))
        else:
            oriented_edges.append(edge)
    oriented_edges.sort(key=cmp_to_key(compare_edges))

    return oriented_edges


def all_intersections(
    edges: list[tuple[PointType, PointType]],
    rel_tol: float | None = None,
    abs_tol: float | None = None,
    return_points_list: bool = False,
) -> tuple[dict, list[tuple]]:
    """Return all proper intersections between the given edges.

    Bounding-box candidates are collected into one NumPy array, then
    line-segment intersections are computed in bulk.

    Args:
        edges: Line segments to test.
        rel_tol: Relative tolerance. Defaults to settings default.
        abs_tol: Absolute tolerance. Defaults to settings default.
        return_points_list: If True, include a flat list of intersection
            points in the return value.

    Returns:
        tuple: ``(intersection_map, points)`` when ``return_points_list``
        is True; otherwise an intersection map keyed by edge id.
    """

    relative_tolerance, absolute_tolerance = get_defaults(
        ["rel_tol", "abs_tol"], [rel_tol, abs_tol]
    )
    edge_coordinates = []
    for edge in edges:
        start_point, end_point = edge
        start_x, start_y = start_point[:2]
        end_x, end_y = end_point[:2]
        edge_coordinates.append([start_x, start_y, end_x, end_y])

    edge_array = np.asarray(edge_coordinates, dtype=float)
    edge_count = edge_array.shape[0]
    edge_min_x = np.minimum(edge_array[:, 0], edge_array[:, 2])
    edge_min_y = np.minimum(edge_array[:, 1], edge_array[:, 3])
    edge_max_x = np.maximum(edge_array[:, 0], edge_array[:, 2])
    edge_max_y = np.maximum(edge_array[:, 1], edge_array[:, 3])
    edge_ids = np.arange(edge_count)
    sort_order = edge_min_x.argsort()
    edge_array = edge_array[sort_order]
    edge_min_x = edge_min_x[sort_order]
    edge_min_y = edge_min_y[sort_order]
    edge_max_x = edge_max_x[sort_order]
    edge_max_y = edge_max_y[sort_order]
    edge_ids = edge_ids[sort_order]

    candidate_starts = np.arange(edge_count) + 1
    candidate_ends = np.searchsorted(edge_min_x, edge_max_x, side="right")
    candidate_counts = candidate_ends - candidate_starts
    candidate_count = candidate_counts.sum()
    if not candidate_count:
        if return_points_list:
            return []
        return {}, []

    candidate_offsets = np.cumsum(candidate_counts) - candidate_counts
    candidate_rows = np.arange(candidate_count)
    first_indices = np.repeat(np.arange(edge_count), candidate_counts)
    candidate_offsets = np.repeat(candidate_offsets, candidate_counts)
    candidate_starts = np.repeat(candidate_starts, candidate_counts)
    second_indices = candidate_rows - candidate_offsets + candidate_starts
    y_overlap_mask = (
        edge_min_y[first_indices] <= edge_max_y[second_indices]
    ) & (edge_max_y[first_indices] >= edge_min_y[second_indices])
    first_indices = first_indices[y_overlap_mask]
    second_indices = second_indices[y_overlap_mask]
    candidate_array = np.hstack(
        (
            edge_array[first_indices],
            edge_array[second_indices],
            edge_ids[first_indices, None],
            edge_ids[second_indices, None],
        )
    )
    first_edges = candidate_array[:, :4]
    second_edges = candidate_array[:, 4:8]
    first_delta_x = first_edges[:, 2] - first_edges[:, 0]
    first_delta_y = first_edges[:, 3] - first_edges[:, 1]
    second_delta_x = second_edges[:, 2] - second_edges[:, 0]
    second_delta_y = second_edges[:, 3] - second_edges[:, 1]
    denominator = (
        second_delta_y * first_delta_x - second_delta_x * first_delta_y
    )

    parallel_mask = np.abs(denominator) <= np.maximum(
        absolute_tolerance, relative_tolerance * np.abs(denominator)
    )
    candidate_array = candidate_array[~parallel_mask]
    first_edges = first_edges[~parallel_mask]
    second_edges = second_edges[~parallel_mask]
    first_delta_x = first_delta_x[~parallel_mask]
    first_delta_y = first_delta_y[~parallel_mask]
    second_delta_x = second_delta_x[~parallel_mask]
    second_delta_y = second_delta_y[~parallel_mask]
    denominator = denominator[~parallel_mask]
    first_to_second_x = first_edges[:, 0] - second_edges[:, 0]
    first_to_second_y = first_edges[:, 1] - second_edges[:, 1]
    first_parameter = (
        second_delta_x * first_to_second_y - second_delta_y * first_to_second_x
    ) / denominator
    second_parameter = (
        first_delta_x * first_to_second_y - first_delta_y * first_to_second_x
    ) / denominator
    intersecting_mask = (
        (first_parameter >= 0)
        & (first_parameter <= 1)
        & (second_parameter >= 0)
        & (second_parameter <= 1)
    )
    intersection_x = first_edges[:, 0] + first_parameter * first_delta_x
    intersection_y = first_edges[:, 1] + first_parameter * first_delta_y
    intersection_rows = candidate_array[intersecting_mask]

    if return_points_list:
        res = [
            ((x_coordinate, y_coordinate), (int(first_id), int(second_id)))
            for x_coordinate, y_coordinate, first_id, second_id in zip(
                intersection_x[intersecting_mask],
                intersection_y[intersecting_mask],
                intersection_rows[:, 8],
                intersection_rows[:, 9],
            )
        ]
    else:
        d_results = {}
        points = []
        for x_coordinate, y_coordinate, first_id, second_id in zip(
            intersection_x[intersecting_mask],
            intersection_y[intersecting_mask],
            intersection_rows[:, 8],
            intersection_rows[:, 9],
        ):
            point = (x_coordinate, y_coordinate)
            first_id = int(first_id)
            second_id = int(second_id)
            if first_id not in d_results:
                d_results[first_id] = []
            if second_id not in d_results:
                d_results[second_id] = []
            d_results[first_id].append((point, second_id))
            d_results[second_id].append((point, first_id))
            points.append(point)
        res = (d_results, points)

    return res


def all_segments_sorted(
    edges: list[tuple[PointType, PointType]],
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> list[tuple[PointType]]:
    """Split edges at intersection points, ordered along each edge.

    Intersection points on an edge are sorted by distance from the edge's
    first endpoint. The result is a flat list of consecutive sub-segments.

    Args:
        edges: Input line segments.
        rel_tol: Relative tolerance for intersections.
        abs_tol: Absolute tolerance for intersections.

    Returns:
        list: Sub-segments covering each input edge in order.
    """
    intersection_map, _ = all_intersections(edges, rel_tol, abs_tol)

    intersections_by_edge = {edge_id: [] for edge_id in range(len(edges))}
    for edge_id, intersections in intersection_map.items():
        intersections_by_edge[edge_id] = intersections

    sorted_segments = []
    for edge_id, edge in enumerate(edges):
        start_point, end_point = edge
        intersections = sorted(
            intersections_by_edge[edge_id],
            key=lambda item: distance(start_point, item[0]),
        )
        points_on_edge = []
        point_coordinates = {tuple(start_point[:2]), tuple(end_point[:2])}
        for point, _ in intersections:
            coordinates = tuple(point[:2])
            if coordinates not in point_coordinates:
                points_on_edge.append(point)
                point_coordinates.add(coordinates)
        edge_points = [start_point] + points_on_edge + [end_point]
        sorted_segments.extend(connected_pairs(edge_points))
        # sorted_segments.append(
        #     tuple([start_point] + points_on_edge + [end_point])

    return sorted_segments


def angle_between_lines2(
    point1: PointType, point2: PointType, point3: PointType
) -> float:
    """
    Given line1 as point1 and point2, and line2 as point2 and point3
    return the angle between two lines
    (point2 is the corner point)

    Args:
        point1 (PointType): First point of the first line.
        point2 (PointType): Second point of the first line and first point of the second line.
        point3 (PointType): Second point of the second line.

    Returns:
        float: Signed angle between the two lines in radians (-π to π).
    """
    vec1 = v_from_points(point2, point1)
    vec2 = v_from_points(point2, point3)
    cross = v_cross(vec1, vec2)
    dot = v_mul(vec1, vec2)
    return atan2(cross, dot)


def trim_right(line, x_value):
    """
    Trim a line to the right at a given x-coordinate.

    Args:
        line (LineType): The line to trim.
        x_value (float): The x-coordinate to trim the line at.

    Returns:
        LineType: The trimmed line.
    """
    reverse = False
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        reverse = True

    if x1 >= x_value:
        res = None
    elif x2 >= x_value:
        intersection_ = intersect(line, [(x_value, 0), (x_value, 1)])
        res = [(x1, y1), intersection_]
    else:
        res = line
        reverse = False
    if res and reverse:
        res = [res[1], res[0]]  # Reverse the order if we swapped x1 and x2

    return res


def trim_left(line, x_value):
    """
    Trim a line to the left at a given x-coordinate.

    Args:
        line (LineType): The line to trim.
        x_value (float): The x-coordinate to trim the line at.

    Returns:
        LineType: The trimmed line.
    """
    reverse = False
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        reverse = True

    if x1 >= x_value:
        res = line
        reverse = False
    elif x2 >= x_value:
        intersection_ = intersect(line, [(x_value, 0), (x_value, 1)])
        res = [intersection_, (x2, y2)]
    else:
        res = None

    if res and reverse:
        res = [res[1], res[0]]  # Reverse the order if we swapped x1 and x2

    return res


def trim_top(line, y_value):
    """
    Trim a line to the top at a given y-coordinate.

    Args:
        line (LineType): The line to trim.
        y_value (float): The y-coordinate to trim the line at.

    Returns:
        LineType: The trimmed line.
    """
    reverse = False
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    if y1 > y2:
        y1, y2 = y2, y1
        x1, x2 = x2, x1
        reverse = True

    if y1 >= y_value:
        res = None
    elif y2 >= y_value:
        intersection_ = intersect(line, [(0, y_value), (1, y_value)])
        res = [(x1, y1), intersection_]
    else:
        res = line
        reverse = False

    if res and reverse:
        res = [res[1], res[0]]

    return res


def trim_bottom(line, y_value):
    """
    Trim a line to the bottom at a given y-coordinate.

    Args:
        line (LineType): The line to trim.
        y_value (float): The y-coordinate to trim the line at.

    Returns:
        LineType: The trimmed line.
    """
    reverse = False
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    if y1 > y2:
        y1, y2 = y2, y1
        x1, x2 = x2, x1
        reverse = True

    if y1 >= y_value:
        res = line
        reverse = False
    elif y2 >= y_value:
        intersection_ = intersect(line, [(0, y_value), (1, y_value)])
        res = [intersection_, (x2, y2)]
    else:
        res = None

    if res and reverse:
        res = [res[1], res[0]]

    return res


def stitch(
    lines: list[LineType],
    closed: bool = True,
    return_points: bool = True,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> list[PointType]:
    """
    Stitches a list of lines together.

    Args:
        lines (list[LineType]): List of lines to stitch.
        closed (bool, optional): Whether the lines form a closed shape. Defaults to True.
        return_points (bool, optional): Whether to return points or lines. Defaults to True.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        list[PointType]: Stitched list of points or lines.
    """
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    if closed:
        points = []
    else:
        points = [lines[0][0]]
    for i, line in enumerate(lines[:-1]):
        x1, y1 = line[0][:2]
        x2, y2 = line[1][:2]
        x3, y3 = lines[i + 1][0][:2]
        x4, y4 = lines[i + 1][1][:2]
        x_point = intersect2(x1, y1, x2, y2, x3, y3, x4, y4)
        if x_point:
            points.append(x_point)
    if closed:
        x1, y1 = lines[-1][0][:2]
        x2, y2 = lines[-1][1][:2]
        x3, y3 = lines[0][0][:2]
        x4, y4 = lines[0][1][:2]
        final_x = intersect2(x1, y1, x2, y2, x3, y3, x4, y4)
        if final_x:
            points.insert(0, final_x)
            points.append(final_x)
    else:
        points.append(lines[-1][1])
    if return_points:
        res = points
    else:
        res = connected_pairs(points)

    return res


def equal_lines(
    line1: LineType, line2: LineType, dist_tol: float | None = None
) -> bool:
    """
    Return True if two lines are close enough.

    Args:
        line1 (LineType): First line.
        line2 (LineType): Second line.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        bool: True if the lines are close enough, False otherwise.
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    dist_tol2 = dist_tol * dist_tol
    p1, p2 = line1
    p3, p4 = line2
    return (
        close_points2(p1, p3, dist2=dist_tol2)
        and close_points2(p2, p4, dist2=dist_tol2)
    ) or (
        close_points2(p1, p4, dist2=dist_tol2)
        and close_points2(p2, p3, dist2=dist_tol2)
    )


def length(line: LineType) -> float:
    """Return the length of a line.

    Args:
        line (LineType): Input line.

    Returns:
        float: Length of the line.
    """
    p1, p2 = line
    return distance(p1, p2)


def extended_line(dist: float, line: LineType, extend_both=False) -> LineType:
    """
    Given a line ((x1, y1), (x2, y2)) and a distance,
    the given line is extended by distance units.
    Return a new line ((x1, y1), (x2', y2')).

    Args:
        dist (float): Distance to extend the line.
        line (LineType): Input line.
        extend_both (bool, optional): Whether to extend both ends of the line. Defaults to False.

    Returns:
        LineType: Extended line.
    """

    def extend(dist, line):
        # p = (1-t)*p1 + t*p2 : parametric equation of a line segment (p1, p2)
        line_length = length(line)
        t = (line_length + dist) / line_length
        p1, p2 = line
        x1, y1 = p1[:2][:2]
        x2, y2 = p2[:2][:2]
        c = 1 - t

        return [(x1, y1), (c * x1 + t * x2, c * y1 + t * y2)]

    if extend_both:
        p1, p2 = extend(dist, line)
        p1, p2 = extend(dist, [p2, p1])
        res = [p2, p1]
    else:
        res = extend(dist, line)

    return res


def line_through_point_angle(
    point: PointType, angle: float, length_: float, both_sides=False
) -> LineType:
    """
    Return a line that passes through the given point
    with the given angle and length.
    If both_side is True, the line is extended on both sides by the given
    length.

    Args:
        point (PointType): PointType through which the line passes.
        angle (float): Angle of the line in radians.
        length_ (float): Length of the line.
        both_sides (bool, optional): Whether to extend the line on both sides. Defaults to False.

    Returns:
        LineType: Line passing through the given point with the given angle and length.
    """
    x, y = point[:2]
    line = [(x, y), (x + length_ * cos(angle), y + length_ * sin(angle))]
    if both_sides:
        p1, p2 = line
        line = extended_line(length_, [p2, p1])

    return line


def split_segment(segment: LineType, point: PointType):
    """Split a segment into two pieces at ``point``.

    Args:
        segment: Line segment ``(p1, p2)``.
        point: Split point (must lie on the segment, not at an endpoint).

    Returns:
        tuple | None: ``((p1, point), (point, p2))``, or None if ``point``
        is an endpoint or not on the segment.
    """
    p1, p2 = segment
    if close_points2(point, p1) or close_points2(point, p2):
        return None
    if not point_on_line_segment(point, segment):
        return None

    return [(p1, point), (point, p2)]


def multi_split_segment(segment: LineType, points: Sequence, dist_tol=0.1):
    """Split a segment into multiple pieces at the given points.

    Split points are ordered by distance from the segment start.

    Args:
        segment: Line segment ``(p1, p2)``.
        points: Points that lie on the segment.
        dist_tol: Unused legacy tolerance parameter. Defaults to 0.1.

    Returns:
        list: Consecutive sub-segments from start to end.
    """
    p1, p2 = segment
    distances = []
    for i, pnt in enumerate(points):
        dist = distance(p1, pnt)
        distances.append((dist, i))
    distances.sort()
    points = [points[ind] for (_, ind) in distances]

    if len(points) == 2:
        close_p1 = close_points2(points[0], p1)
        close_p2 = close_points2(points[1], p2)
        if close_p1 and close_p2:
            return [segment]

    for i, pnt in enumerate(points):
        if close_points2(p1, pnt):
            continue
        if not point_on_line_segment(pnt, segment):
            print("point not on line")
            return None

    segments = []
    start = p1
    for point in points:
        if distance(start, point) < dist_tol:
            continue
        segments.append((start, point))
        start = point

    return segments


def intersects(seg1, seg2):
    """Checks if the line segments intersect.
    If they are chained together, they are considered as intersecting.
    Returns True if the segments intersect, False otherwise.

    Args:
        seg1 (LineType): First line segment.
        seg2 (LineType): Second line segment.

    Returns:
        bool: True if the segments intersect, False otherwise.
    """
    p1, q1 = seg1
    p2, q2 = seg2
    o1 = clockwise(p1, q1, p2)
    o2 = clockwise(p1, q1, q2)
    o3 = clockwise(p2, q2, p1)
    o4 = clockwise(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and between(p1, p2, q1):
        return True
    if o2 == 0 and between(p1, q2, q1):
        return True
    if o3 == 0 and between(p2, p1, q2):
        return True
    return bool(o4 == 0 and between(p2, q1, q2))


def is_chained(seg1, seg2):
    """Checks if the line segments are chained together.

    Args:
        seg1 (LineType): First line segment.
        seg2 (LineType): Second line segment.

    Returns:
        bool: True if the segments are chained together, False otherwise.
    """
    p1, q1 = seg1
    p2, q2 = seg2
    return bool(
        close_points2(p1, p2)
        or close_points2(p1, q2)
        or close_points2(q1, p2)
        or close_points2(q1, q2)
    )


def stitch_lines(line1: LineType, line2: LineType) -> Sequence[LineType]:
    """if the lines intersect, trim the lines
    if the lines don't intersect, extend the lines

    Args:
        line1 (LineType): First line.
        line2 (LineType): Second line.

    Returns:
        Sequence[LineType]: Trimmed or extended lines.
    """
    intersection_ = intersect(line1, line2)
    res = None
    if intersection_:
        p1, _ = line1
        _, p2 = line2
        line1 = [p1, intersection_]
        line2 = [intersection_, p2]

        res = (line1, line2)

    return res


def intersection(
    line1: LineType, line2: LineType, rel_tol: float | None = None
) -> int:
    """return the intersection point of two line segments.
    segment1: ((x1, y1), (x2, y2))
    segment2: ((x3, y3), (x4, y4))
    To find the intersection point of two lines use the "intersect" function

    Args:
        line1 (LineType): First line segment.
        line2 (LineType): Second line segment.
        rel_tol (float, optional): Relative tolerance. Defaults to None.

    Returns:
        int: Intersection type.
    """
    if rel_tol is None:
        rel_tol = defaults["rel_tol"]
    x1, y1 = line1[0][:2]
    x2, y2 = line1[1][:2]
    x3, y3 = line2[0][:2]
    x4, y4 = line2[1][:2]
    return intersection2(x1, y1, x2, y2, x3, y3, x4, y4)


def merge_segments(
    seg1: Sequence[PointType], seg2: Sequence[PointType]
) -> Sequence[PointType]:
    """Merge two segments into one if they overlap or chain.

    Order of endpoints does not matter. Unconnected segments are not merged.

    Args:
        seg1: First segment ``(p1, p2)``.
        seg2: Second segment ``(p3, p4)``.

    Returns:
        Sequence[PointType]: Merged segment endpoints, or the originals if
        they cannot be merged.
    """

    # """Merge two segments into one segment if they are connected.
    # They need to be overlapping or simply connected to each other,
    # otherwise they will not be merged. Order doesn't matter.

    # Args:
    #     seg1 (Sequence[PointType]): First segment.
    #     seg2 (Sequence[PointType]): Second segment.

    # Returns:
    #     Sequence[PointType]: Merged segment.
    # """
    Conn = Connection
    p1, p2 = seg1
    p3, p4 = seg2

    res = all_intersections([(p1, p2), (p3, p4)], use_intersection3=True)
    if res:
        conn_type = next(iter(res.values()))[0][0]
        verts = next(iter(res.values()))[0][1]
        if conn_type in (Conn.OVERLAPS, Conn.CONGRUENT, Conn.CHAIN):
            res = verts
        elif conn_type == Conn.COLL_CHAIN:
            res = (verts[0], verts[1])
        else:
            res = None
    else:
        res = None  # need this to avoid returning an empty dict

    return res


def is_horizontal(line: LineType, eps: float = 0.0001) -> bool:
    """Return True if the line is horizontal.

    Args:
        line (LineType): Input line.
        eps (float, optional): Tolerance. Defaults to 0.0001.

    Returns:
        bool: True if the line is horizontal, False otherwise.
    """
    return abs(j_vec.dot(line_vector(line))) <= eps


def is_vertical(line: LineType, eps: float = 0.0001) -> bool:
    """Return True if the line is vertical.

    Args:
        line (LineType): Input line.
        eps (float, optional): Tolerance. Defaults to 0.0001.

    Returns:
        bool: True if the line is vertical, False otherwise.
    """
    return abs(i_vec.dot(line_vector(line))) <= eps


def slope(
    start_point: PointType, end_point: PointType, rel_tol=None, abs_tol=None
) -> float:
    """Return the slope of a line given by two points.
    Order makes a difference.

    Args:
        start_point (PointType): Start point of the line.
        end_point (PointType): End point of the line.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        float: Slope of the line.
    """
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    x1, y1 = start_point[:2]
    x2, y2 = end_point[:2]
    if isclose(x1, x2, rel_tol=rel_tol, abs_tol=abs_tol):
        res = defaults["INF"]
    else:
        res = (y2 - y1) / (x2 - x1)

    return res


def segmentize_line(line: LineType, segment_length: float) -> list[LineType]:
    """Return a list of points that would form segments with the given length.

    Args:
        line (LineType): Input line.
        segment_length (float): Length of each segment.

    Returns:
        list[LineType]: List of segments.
    """
    length_ = distance(line[0], line[1])
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    increments = int(length_ / segment_length)
    x_segments = np.linspace(x1, x2, increments)
    y_segments = np.linspace(y1, y2, increments)

    return list(zip(x_segments, y_segments))


def line_through_point_and_angle(
    point: PointType, angle: float, length_: float = 100
) -> LineType:
    """Return a line through the given point with the given angle and length

    Args:
        point (PointType): PointType through which the line passes.
        angle (float): Angle of the line in radians.
        length_ (float, optional): Length of the line. Defaults to 100.

    Returns:
        LineType: Line passing through the given point with the given angle and length.
    """
    x, y = point[:2]
    dx = length_ * cos(angle)
    dy = length_ * sin(angle)
    return [[x, y], [x + dx, y + dy]]


def translate_line(dx: float, dy: float, line: LineType) -> LineType:
    """Return a translated line by dx and dy

    Args:
        dx (float): Translation distance in x-direction.
        dy (float): Translation distance in y-direction.
        line (LineType): Input line.

    Returns:
        LineType: Translated line.
    """
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    return [[x1 + dx, y1 + dy], [x2 + dx, y2 + dy]]


def trim_line(line1: LineType, line2: LineType) -> LineType:
    """Trim line1 to the intersection of line1 and line2.
    Extend it if necessary.

    Args:
        line1 (LineType): First line.
        line2 (LineType): Second line.

    Returns:
        LineType: Trimmed line.
    """
    intersection_ = intersection(line1, line2)
    return [line1[0], intersection_]


def angle_between_two_lines(line1, line2):
    """Return the angle between two lines in radians.

    Args:
        line1 (LineType): First line.
        line2 (LineType): Second line.

    Returns:
        float: Angle between the two lines in radians.
    """
    alpha1 = line_angle(*line1)
    alpha2 = line_angle(*line2)
    return abs(alpha1 - alpha2)


def bisector_line(a: PointType, b: PointType, c: PointType) -> LineType:
    """
    Given three points that form two lines [a, b] and [b, c]
    return the bisector line between them.

    Args:
        a (PointType): First point.
        b (PointType): Second point.
        c (PointType): Third point.

    Returns:
        LineType: Bisector line.
    """
    d = midpoint(a, c)

    return [d, b]


def fillet(
    a: PointType, b: PointType, c: PointType, radius: float
) -> tuple[LineType, LineType, PointType]:
    """
    Given three points that form two lines [a, b] and [b, c]
    return the clipped lines [a, d], [e, c], center point
    of the radius circle (tangent to both lines), and the arc
    angle of the formed fillet.

    Args:
        a (PointType): First point.
        b (PointType): Second point.
        c (PointType): Third point.
        radius (float): Radius of the fillet.

    Returns:
        tuple: Clipped lines [a, d], [e, c], center point of the radius circle, and the arc angle.
    """
    alpha2 = angle_between_lines2(a, b, c) / 2
    sin_alpha2 = sin(alpha2)
    cos_alpha2 = cos(alpha2)
    clip_length = radius * cos_alpha2 / sin_alpha2
    d = offset_point_from_start(b, a, clip_length)
    e = offset_point_from_start(b, c, clip_length)
    mp = midpoint(a, c)  # [b, mp] is the bisector line
    center = offset_point_from_start(b, mp, radius / sin_alpha2)
    arc_angle = angle_between_lines2(e, center, d)

    return [a, d], [e, c], center, arc_angle


def fillet_points(
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
    radius: float,
    n: int,
    *,
    clamp_radius: bool = False,
    eps: float = 1e-12,
) -> list[PointType]:
    """Sample points along a circular fillet at vertex ``p2``.

    The fillet is between segments ``p1→p2`` and ``p2→p3`` (2D). When
    ``n >= 2``, samples include tangency endpoints. When ``n == 1``, returns
    the arc midpoint. When ``n <= 0``, returns an empty list.

    Args:
        p1: First point ``(x, y)``.
        p2: Corner vertex ``(x, y)``.
        p3: Third point ``(x, y)``.
        radius: Desired fillet radius (must be > 0).
        n: Number of sample points along the arc.
        clamp_radius: If True, reduce ``radius`` to the maximal feasible
            value when it is too large for the segments.
        eps: Numeric tolerance.

    Returns:
        list[PointType]: Sampled points along the fillet arc.

    Raises:
        ValueError: If geometry is degenerate or radius is infeasible
            (unless ``clamp_radius`` is True).
    """
    if radius <= 0:
        raise ValueError("radius must be > 0")

    p1 = p1[:2]
    p2 = p2[:2]
    p3 = p3[:2]

    # Convert to Vector objects
    v1 = Vector(float(p1[0]), float(p1[1]))
    v2 = Vector(float(p2[0]), float(p2[1]))
    v3 = Vector(float(p3[0]), float(p3[1]))

    # Direction vectors along the polyline
    v_in = (v2 - v1).normalize()  # direction into p2
    v_out = (v3 - v2).normalize()  # direction out of p2

    # Rays from the corner along each segment
    u1 = -v_in  # from p2 toward p1
    u2 = v_out  # from p2 toward p3

    # Interior angle between u1 and u2
    c = max(-1.0, min(1.0, u1.dot(u2)))
    theta = acos(c)
    if theta < eps or abs(theta - pi) < eps:
        raise ValueError(
            "Points are collinear or angle too close to 0/180 degrees."
        )

    # Distances and feasibility of the radius
    L1 = (v2 - v1).mag()
    L2 = (v3 - v2).mag()
    # Tangency distance along each leg
    t = radius / tan(theta / 2.0)
    # Max radius allowed by each leg: r_max_i = Li * tan(theta/2)
    r_max = min(L1, L2) * tan(theta / 2.0)

    if t > L1 + eps or t > L2 + eps:
        if clamp_radius:
            # Clamp radius to feasible value (slightly inside to avoid degeneracy)
            radius = max(0.0, min(radius, r_max * (1.0 - 1e-9)))
            t = radius / tan(theta / 2.0)
        else:
            raise ValueError(
                f"Radius too large for given segments. Max feasible ~ {r_max:.6f}"
            )

    # Tangency points on each leg
    A = v2 + u1 * t
    B = v2 + u2 * t

    # Bisector direction (inside the angle)
    bis = u1 + u2
    if bis.mag() < eps:
        raise ValueError("Angle too close to 180°, bisector undefined.")
    w_hat = bis.normalize()

    # Center of the fillet circle
    center_dist = radius / sin(theta / 2.0)
    C = v2 + w_hat * center_dist

    # Angles of tangency points around center
    a1 = atan2(A.y - C.y, A.x - C.x)
    a2 = atan2(B.y - C.y, B.x - C.x)

    # Determine sweep direction based on the turn (left/CCW or right/CW)
    turn = v_in.cross(v_out)  # >0 => left turn (CCW), <0 => right turn (CW)
    delta = a2 - a1
    # Normalize delta to follow the turn direction along the minor arc
    if turn > 0:  # CCW
        if delta < 0:
            delta += 2.0 * pi
    else:  # CW
        if delta > 0:
            delta -= 2.0 * pi

    # Generate points on the arc
    if n <= 0:
        return []
    if n == 1:
        ang = a1 + 0.5 * delta
        pt = C + Vector(radius * cos(ang), radius * sin(ang))
        return [(pt.x, pt.y)]
    # n >= 2: include both tangency endpoints
    pts: list[PointType] = []
    for i in range(n):
        t_frac = i / (n - 1)
        ang = a1 + t_frac * delta
        pt = C + Vector(radius * cos(ang), radius * sin(ang))
        pts.append((pt.x, pt.y))
    return pts


def fillet_corners(
    vertices: Sequence[PointType], d_vert_radius: dict[int, float], n: int = 12
) -> Sequence[PointType]:
    """Return a new vertex list with selected corners filleted.

    Args:
        vertices: Original polygon/polyline vertices.
        d_vert_radius: Map from vertex index to fillet radius.
        n: Number of sample points per fillet. Defaults to 12.

    Returns:
        Sequence[PointType]: Vertices including fillet arc samples.
    """
    count = len(vertices)

    # Build set of indices to fillet for quick lookup
    indices = d_vert_radius.keys()
    fillet_set = set(indices)

    # Build new vertex list
    new_vertices = []
    fillet_count = 0
    for i in range(count):
        if i not in fillet_set:
            # Keep original vertex
            new_vertices.append(vertices[i][:2])
        else:
            # Replace with fillet arc
            prev_idx = (i - 1) % count
            next_idx = (i + 1) % count

            p1 = vertices[prev_idx][:2]
            p2 = vertices[i][:2]
            p3 = vertices[next_idx][:2]

            # Generate fillet points
            radius = d_vert_radius[i]
            arc_points = fillet_points(p1, p2, p3, radius, n, clamp_radius=True)
            new_vertices.extend(arc_points)
            fillet_count += 1

    # Create new shape with same properties
    return new_vertices


def line_by_point_angle_length(point, angle, length_):
    """
    Given a point, an angle, and a length, return the line
    that starts at the point and has the given angle and length.

    Args:
        point (PointType): Start point of the line.
        angle (float): Angle of the line in radians.
        length_ (float): Length of the line.

    Returns:
        LineType: Line with the given angle and length.
    """
    x, y = point[:2]
    dx = length_ * cos(angle)
    dy = length_ * sin(angle)

    return [(x, y), (x + dx, y + dy)]


class Edge:
    """A 2D edge."""

    def __init__(
        self,
        start_point: PointType | Vertex,
        end_point: PointType | Vertex,
    ):
        """Create an edge from ``start_point`` to ``end_point``.

        Args:
            start_point: Start as ``PointType`` or ``Vertex``.
            end_point: End as ``PointType`` or ``Vertex``.

        Raises:
            TypeError: If either endpoint is not a point or ``Vertex``.
        """
        if isinstance(start_point, PointType):
            start = Vertex(*start_point)
        elif isinstance(end_point, Vertex):
            start = start_point
        else:
            raise TypeError(
                "Start point should be a PointType or Vertex instance."
            )

        if isinstance(end_point, PointType):
            end = Vertex(*end_point)
        elif isinstance(end_point, Vertex):
            end = end_point
        else:
            raise TypeError(
                "End point should be a PointType or Vertex instance."
            )

        self.start = start
        self.end = end
        self.type = Types.EDGE

    def __repr__(self):
        return str(f"Edge({self.start}, {self.end})")

    def __str__(self):
        return str(f"Edge({self.start.point}, {self.end.point})")

    def __eq__(self, other):
        start = other.start.point
        end = other.end.point

        return (
            isclose(
                self.start.point,
                start,
                rel_tol=defaults["rel_tol"],
                abs_tol=defaults["abs_tol"],
            )
            and isclose(
                self.end.point,
                end,
                rel_tol=defaults["rel_tol"],
                abs_tol=defaults["abs_tol"],
            )
        ) or (
            isclose(
                self.start.point,
                end,
                rel_tol=defaults["rel_tol"],
                abs_tol=defaults["abs_tol"],
            )
            and isclose(
                self.end.point,
                start,
                rel_tol=defaults["rel_tol"],
                abs_tol=defaults["abs_tol"],
            )
        )

    def __getitem__(self, subscript):
        vertices = self.vertices
        if isinstance(subscript, slice):
            res = vertices[subscript.start : subscript.stop : subscript.step]
        elif isinstance(subscript, int):
            res = vertices[subscript]
        else:
            raise TypeError("Invalid subscript.")
        return res

    def __setitem__(self, subscript, value):
        vertices = self.vertices
        if isinstance(subscript, slice):
            vertices[subscript.start : subscript.stop : subscript.step] = value
        else:
            isinstance(subscript, int)
            vertices[subscript] = value

    @property
    def slope(self):
        """Line slope. The slope of the line passing through the start and end points."""
        return (self.y2 - self.y1) / (self.x2 - self.x1)

    @property
    def angle(self):
        """Line angle. Angle between the line and the x-axis."""
        return atan2(self.y2 - self.y1, self.x2 - self.x1)

    @property
    def inclination(self):
        """Inclination angle. Angle between the line and the x-axis converted to
        a value between zero and pi."""
        return self.angle % pi

    @property
    def length(self):
        """Length of the line segment."""
        return distance(self.start.point, self.end.point)

    @property
    def x1(self):
        """x-coordinate of the start point."""
        return self.start.x

    @property
    def y1(self):
        """y-coordinate of the start point."""
        return self.start.y

    @property
    def x2(self):
        """x-coordinate of the end point."""
        return self.end.x

    @property
    def y2(self):
        """y-coordinate of the end point."""
        return self.end.y

    @property
    def points(self):
        """Start and end"""
        return [self.start.point, self.end.point]

    @property
    def vertices(self):
        """Start and end vertices."""
        return [self.start, self.end]

    @property
    def array(self):
        """Homogeneous coordinates as a numpy array."""
        return array([self.start.array, self.end.array])

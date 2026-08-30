from collections.abc import Sequence
from math import atan2, cos, hypot, sin
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..geometry.vectors import perp_unit_vector
from ..graphics.all_enums import Connection
from ..graphics.common import LineType, PointType, get_defaults
from ..settings.settings import defaults

around = np.around


def homogenize(points: Sequence[PointType]) -> NDArray:
    """
    Convert a list of points to homogeneous coordinates.

    Args:
        points (Sequence[PointType]): List of points.

    Returns:
        np.ndarray: Homogeneous coordinates.
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


def is_number(x: Any) -> bool:
    """
    Return True if x is a number.

    Args:
        x (Any): The input value to check.

    Returns:
        bool: True if x is a number, False otherwise.
    """
    return isinstance(x, (int, float, complex)) and not isinstance(x, bool)


def is_point(pnt: Any) -> bool:
    """Return True if the input is a point.

    Args:
        pnt (Any): Input value.

    Returns:
        bool: True if the input is a point, False otherwise.
    """
    try:
        x, y = pnt[:2]
        return is_number(x) and is_number(y)
    except:
        return False


def is_line(line_: Any) -> bool:
    """Return True if the input is a line.

    Args:
        line_ (Any): Input value.

    Returns:
        bool: True if the input is a line, False otherwise.
    """
    try:
        p1, p2 = line_
        return is_point(p1) and is_point(p2)
    except:
        return False


def positive_angle(angle, radians=True, rel_tol=None, abs_tol=None):
    """Return the positive angle in radians or degrees.

    Args:
        angle (float): Input angle.
        radians (bool, optional): Whether the angle is in radians. Defaults to True.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        float: Positive angle.
    """
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    if radians:
        if angle < 0:
            angle += 2 * pi
    else:
        if angle < 0:
            angle += 360

    return angle


def line_angle(start_point: PointType, end_point: PointType) -> float:
    """Return the orientation angle (in radians) of a line given by start and end points.
    Order makes a difference.

    Args:
        start_point (PointType): Start point of the line.
        end_point (PointType): End point of the line.

    Returns:
        float: Orientation angle of the line in radians.
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


def offset_point(point: PointType, dx: float = 0, dy: float = 0) -> PointType:
    """Return an offset point from a given point.

    Args:
        point (PointType): Input point.
        dx (float, optional): Offset distance in x-direction. Defaults to 0.
        dy (float, optional): Offset distance in y-direction. Defaults to 0.

    Returns:
        PointType: Offset point.
    """
    x, y = point[:2]
    return x + dx, y + dy


def parallel_line(line: LineType, point: PointType) -> LineType:
    """Return a parallel line to the given line that goes through the given point

    Args:
        line (LineType): Input line.
        point (PointType): PointType through which the parallel line passes.

    Returns:
        LineType: Parallel line.
    """
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    x3, y3 = point
    dx = x2 - x1
    dy = y2 - y1
    return [[x3, y3], [x3 + dx, y3 + dy]]


def midpoint(p1: PointType, p2: PointType) -> PointType:
    """Return the mid point of two points.

    Args:
        p1 (PointType): First point.
        p2 (PointType): Second point.

    Returns:
        PointType: Mid point of the two points.
    """
    x = (p2[0] + p1[0]) / 2
    y = (p2[1] + p1[1]) / 2
    return (x, y)


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
    """
    unit_vec = perp_unit_vector(line)
    dx = unit_vec[0] * offset
    dy = unit_vec[1] * offset
    x, y = point[:2]
    return [x + dx, y + dy]


def area(a, b, c):
    """Return the area of a triangle given its vertices.

    Args:
        a (PointType): First vertex.
        b (PointType): Second vertex.
        c (PointType): Third vertex.

    Returns:
        float: Area of the triangle.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])


def perp_bisector(line: LineType) -> LineType:
    """Return the perpendicular bisector of a line

    Args:
        line (LineType): Input line.

    Returns:
        LineType: Perpendicular bisector of the line.
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
    """
    if area_tol is None:
        area_tol = defaults["area_tol"]

    return abs(area(a, b, c)) <= area_tol


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


def fix_degen_points(
    points: list[PointType],
    loop=False,
    closed=False,
    dist_tol: float | None = None,
    area_rtol: float | None = None,
    area_atol: float | None = None,
    check_collinear=True,
) -> list[PointType]:
    """
    Return a list of points with duplicate points removed.
    Remove the middle point from the collinear points.

    Args:
        points (list[PointType]): List of points.
        loop (bool, optional): Whether to loop the points. Defaults to False.
        closed (bool, optional): Whether the points form a closed shape. Defaults to False.
        dist_tol (float, optional): Distance tolerance. Defaults to None.
        area_rtol (float, optional): Relative tolerance for area. Defaults to None.
        area_atol (float, optional): Absolute tolerance for area. Defaults to None.
        check_collinear (bool, optional): Whether to check for collinear points. Defaults to True.

    Returns:
        list[PointType]: List of points with duplicate and collinear points removed.
    """
    dist_tol, area_rtol, area_atol = get_defaults(
        ["dist_tol", "area_rtol", "area_atol"], [dist_tol, area_rtol, area_atol]
    )
    dist_tol2 = dist_tol * dist_tol
    new_points = []
    for i, point in enumerate(points):
        if i == 0:
            new_points.append(point)
        else:
            if not close_points2(point, new_points[-1], dist2=dist_tol2):
                new_points.append(point)
    if loop and close_points2(new_points[0], new_points[-1], dist2=dist_tol2):
        new_points.pop(-1)

    if check_collinear:
        # Check for collinear points and remove the middle one.
        new_points = merge_consecutive_collinear_edges(
            new_points, closed, area_rtol, area_atol
        )

    return new_points


def get_polygons(
    nested_points: Sequence[PointType],
    n_round_digits: int = 2,
    dist_tol: float | None = None,
) -> list:
    """Convert points to clean polygons. Points are vertices of polygons.

    Args:
        nested_points (Sequence[PointType]): List of nested points.
        n_round_digits (int, optional): Number of decimal places to round to. Defaults to 2.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        list: List of clean polygons.
    """
    from ..helpers.graph import sanitize_graph_edges

    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    from ..helpers.graph import get_cycles

    nested_rounded_points = []
    for points in nested_points:
        rounded_points = []
        for point in points:
            rounded_point = (around(point, n_round_digits)).tolist()
            rounded_points.append(tuple(rounded_point))
        nested_rounded_points.append(rounded_points)

    s_points = set()
    d_id__point = {}
    d_point__id = {}
    for points in nested_rounded_points:
        for point in points:
            s_points.add(point)

    for i, fs_point in enumerate(s_points):
        d_id__point[i] = fs_point  # we need a bidirectional dictionary
        d_point__id[fs_point] = i

    nested_point_ids = []
    for points in nested_rounded_points:
        point_ids = []
        for point in points:
            point_ids.append(d_point__id[point])
        nested_point_ids.append(point_ids)

    graph_edges = []
    for point_ids in nested_point_ids:
        graph_edges.extend(connected_pairs(point_ids))
    polygons = []
    graph_edges = sanitize_graph_edges(graph_edges)
    cycles = get_cycles(graph_edges)
    if cycles is None:
        return []
    for cycle_ in cycles:
        nodes = cycle_
        points = [d_id__point[i] for i in nodes]
        points = fix_degen_points(points, closed=True, dist_tol=dist_tol)
        polygons.append(points)

    return polygons


def round_point(point: list[float], n_digits: int = 2) -> list[float]:
    """
    Round a point (x, y) to a given precision.

    Args:
        point (list[float]): Input point.
        n_digits (int, optional): Number of decimal places to round to. Defaults to 2.

    Returns:
        list[float]: Rounded point.
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
    """

    return [round_point(p, n_digits) for p in points]


def round_segment(segment: Sequence[PointType], n_digits: int = 2):
    """Round a segment to a given precision.

    Args:
        segment (Sequence[PointType]): Input segment.
        n_digits (int, optional): Number of decimal places to round to. Defaults to 2.

    Returns:
        Sequence[PointType]: Rounded segment.
    """
    p1 = round_point(segment[0], n_digits)
    p2 = round_point(segment[1], n_digits)

    return [p1, p2]


def connected_pairs(items):
    """Return a list of connected pair tuples corresponding to the items.
    [a, b, c] -> [(a, b), (b, c)]

    Args:
        items (list): List of items.

    Returns:
        list[tuple]: List of connected pair tuples.
    """
    return list(zip(items, items[1:]))


def close_points2(p1: PointType, p2: PointType, dist2: float = 0.01) -> bool:
    """
    Return True if two points are close to each other.

    Args:
        p1 (PointType): First point.
        p2 (PointType): Second point.
        dist2 (float, optional): Square of the threshold distance. Defaults to 0.01.

    Returns:
        bool: True if the points are close to each other, False otherwise.
    """
    return distance2(p1, p2) <= dist2


def distance2(p1: PointType, p2: PointType) -> float:
    """
    Return the squared distance between two points.
    Useful for comparing distances without the need to
    compute the square root.

    Args:
        p1 (PointType): First point.
        p2 (PointType): Second point.

    Returns:
        float: Squared distance between the two points.
    """
    return (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2


def distance(p1: PointType, p2: PointType) -> float:
    """
    Return the distance between two points.

    Args:
        p1 (PointType): First point.
        p2 (PointType): Second point.

    Returns:
        float: Distance between the two points.
    """
    return hypot(p2[0] - p1[0], p2[1] - p1[1])


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
    """
    x1, y1 = line1[0][:2]
    x2, y2 = line1[1][:2]
    x3, y3 = line2[0][:2]
    x4, y4 = line2[1][:2]
    return intersect2(x1, y1, x2, y2, x3, y3, x4, y4)


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


def polar_to_cartesian(r, theta, center=(0, 0)):
    """Convert polar coordinates to cartesian coordinates.

    Args:
        r (float): Radius.
        theta (float): Angle in radians.

    Returns:
        PointType: Cartesian coordinates.
    """
    dx, dy = center
    return (r * cos(theta) + dx, r * sin(theta) + dy)


def cartesian_to_polar(x, y, center=(0, 0)):
    """Convert cartesian coordinates to polar coordinates.

    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.

    Returns:
        tuple: Polar coordinates (r, theta).
    """
    dx, dy = center
    x -= dx
    y -= dy
    r = hypot(x, y)
    theta = positive_angle(atan2(y, x))
    return r, theta


def right_handed(polygon: Sequence[PointType], dist_tol=None) -> float:
    """If polygon is counter-clockwise, return True

    Args:
        polygon (Sequence[PointType]): List of points representing the polygon.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        bool: True if the polygon is counter-clockwise, False otherwise.
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    dist_tol2 = dist_tol * dist_tol
    added_point = False
    if not close_points2(polygon[0], polygon[-1], dist2=dist_tol2):
        polygon.append(polygon[0])
        added_point = True
    area_ = 0
    for i, point in enumerate(polygon[:-1]):
        x1, y1 = point[:2]
        x2, y2 = polygon[i + 1][:2]
        area_ += x1 * y2 - x2 * y1
    if added_point:
        polygon.pop()
    return area_ > 0


def inclination_angle(start_point: PointType, end_point: PointType) -> float:
    """Return the inclination angle (in radians) of a line given by start and end points.
    Inclination angle is always between zero and pi.
    Order makes no difference.

    Args:
        start_point (PointType): Start point of the line.
        end_point (PointType): End point of the line.

    Returns:
        float: Inclination angle of the line in radians.
    """
    return line_angle(start_point, end_point) % pi

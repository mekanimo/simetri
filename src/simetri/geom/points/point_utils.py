"""Point related utility functions."""

from collections.abc import Sequence
from math import atan2, hypot, isclose, sqrt
from typing import Any

import numpy as np
from numpy import array
from numpy.typing import NDArray

from simetri.base.all_enums import Types
from simetri.base.common import LineType, PointType, get_defaults
from simetri.geom.affine import rotate_point
from simetri.geom.geom_utils import (
    close_points2,
    distance2,
    midpoint,
    offset_point,
    offset_point_from_start,
)
from simetri.geom.vectors import (
    LineType,
    PointType,
    Sequence,
    atan2,
    cross_product_sense,
    distance,
    perp_unit_vector,
    sqrt,
)
from simetri.helpers.utilities import lerp
from simetri.helpers.validation import is_number, is_point
from simetri.config.settings import defaults


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


def distance(p1: PointType, p2: PointType) -> float:
    """Return the Euclidean distance between two points.

    Args:
        p1: First point.
        p2: Second point.

    Returns:
        float: Distance between the two points.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.distance((0, 0), (3, 4))
        5.0
    """
    return hypot(p2[0] - p1[0], p2[1] - p1[1])


def equal_points(point1: PointType, point2: PointType, dist_tol=0.001) -> bool:
    """Return True if two points are within ``dist_tol`` of each other.

    Args:
        point1: First point.
        point2: Second point.
        dist_tol: Maximum allowed distance. Defaults to 0.001.

    Returns:
        bool: True if the points are within the given distance.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.equal_points((0, 0), (0.0005, 0))
        True
        >>> sg.equal_points((0, 0), (1, 0))
        False
    """

    return distance(point1, point2) <= dist_tol


def congruent_points(
    point1: PointType, point2: PointType, dist_tol=0.001
) -> bool:
    """Alias for ``equal_points``.

    Args:
        point1: First point.
        point2: Second point.
        dist_tol: Maximum allowed distance. Defaults to 0.001.

    Returns:
        bool: True if the points are within the given distance.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.congruent_points((0, 0), (0.0005, 0))
        True
        >>> sg.congruent_points((0, 0), (1, 0))
        False
    """

    return equal_points(point1, point2, dist_tol=dist_tol)


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
        >>> sg.offset_point_on_line((0, 0), [(0, 0), (1, 0)], 2)
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
        >>> sg.perp_offset_point((0, 0), [(0, 0), (1, 0)], 1)
        [0.0, 1.0]
    """
    unit_vec = perp_unit_vector(line)
    dx = unit_vec[0] * offset
    dy = unit_vec[1] * offset
    x, y = point[:2]
    return [x + dx, y + dy]


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

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.fix_degen_points(
        ...     [(0, 0), (0, 0), (1, 0), (2, 0)],
        ...     check_collinear=False,
        ... )
        [(0, 0), (1, 0), (2, 0)]
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
        from simetri.geom.segments.line_utils import (
            merge_consecutive_collinear_edges,
        )

        new_points = merge_consecutive_collinear_edges(
            new_points, closed, area_rtol, area_atol
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


def direction(p, q, r):
    """
    Checks the orientation of three points (p, q, r).

    Args:
        p (PointType): First point.
        q (PointType): Second point.
        r (PointType): Third point.

    Returns:
        int: 0 if collinear, >0 if counter-clockwise, <0 if clockwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.direction((0, 0), (1, 0), (1, 1))
        -1
        >>> sg.direction((0, 0), (1, 0), (1, -1))
        1
    """
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])


def between(a, b, c):
    """Return True if c is between a and b.

    Args:
        a (PointType): First point.
        b (PointType): Second point.
        c (PointType): Third point.

    Returns:
        bool: True if c is between a and b, False otherwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.between((0, 0), (2, 0), (1, 0))
        True
        >>> sg.between((0, 0), (2, 0), (3, 0))
        False
    """
    from simetri.geom.segments.line_utils import collinear

    if not collinear(a, b, c):
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


def check_consecutive_duplicates(points, rel_tol=0, abs_tol=None) -> bool:
    """Check for consecutive duplicate points in a list of points.

    Args:
        points (list): List of points to check.
        rel_tol (float, optional): Relative tolerance. Defaults to 0.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        bool: True if consecutive duplicate points are found, False otherwise.
    """
    if abs_tol is None:
        abs_tol = defaults["abs_tol"]
    if isinstance(points, np.ndarray):
        points = points.tolist()
    if points and len(points) > 1:
        for i, pnt in enumerate(points[:-1]):
            next_pnt = points[i + 1]
            val1 = pnt[0] + pnt[1]
            val2 = next_pnt[0] + next_pnt[1]
            if isclose(val1, val2, rel_tol=0, abs_tol=abs_tol) and np.allclose(
                pnt, next_pnt, rtol=0, atol=abs_tol
            ):
                return True

    return False


def left(a: PointType, b: PointType, c: PointType) -> bool:
    """
    Check if point c is left of line ab.
    Args:
        a (PointType): The first point defining the line.
        b (PointType): The second point defining the line.
        c (PointType): The point to test.
    Returns:
        bool: True if point c is left of line ab, False otherwise.
    """

    ax, ay = a[:2]
    bx, by = b[:2]
    cx, cy = c[:2]
    return (bx - ax) * (cy - ay) - (cx - ax) * (by - ay) > 0


def remove_duplicate_points(
    points: list[PointType], dist_tol=None
) -> list[PointType]:
    """
    Return a list of points with duplicate points removed.

    Args:
        points (list[PointType]): List of points.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        list[PointType]: List of points with duplicate points removed.
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    new_points = []
    for i, point in enumerate(points):
        if i == 0:
            new_points.append(point)
        else:
            dist_tol2 = dist_tol * dist_tol
            if not close_points2(point, new_points[-1], dist2=dist_tol2):
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
    """
    from simetri.geom.segments.line_utils import collinear

    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    new_points = []
    for i, point in enumerate(points):
        if i == 0:
            new_points.append(point)
        else:
            if not collinear(
                new_points[-1],
                point,
                points[(i + 1) % len(points)],
                rel_tol,
                abs_tol,
            ):
                new_points.append(point)
    return new_points


def clockwise(p: PointType, q: PointType, r: PointType) -> bool:
    """Return 1 if the points p, q, and r are in clockwise order,
    return -1 if the points are in counter-clockwise order,
    return 0 if the points are collinear

    Args:
        p (PointType): First point.
        q (PointType): Second point.
        r (PointType): Third point.

    Returns:
        int: 1 if the points are in clockwise order, -1 if counter-clockwise, 0 if collinear.
    """
    px, py = p[:2]
    qx, qy = q[:2]
    rx, ry = r[:2]
    area_ = (qx - px) * (ry - py) - (rx - px) * (qy - py)
    if area_ > 0:
        res = 1
    elif area_ < 0:
        res = -1
    else:
        res = 0

    return res


def _homogenize(coordinates: Sequence[float]) -> NDArray:
    """Internal use only. API provides a homogenize function.
    Given a sequence of coordinates(x1, y1, x2, y2, ... xn, yn),
    return a numpy array of points array(((x1, y1, 1.),
    (x2, y2, 1.), ... (xn, yn, 1.))).

    Args:
        coordinates (Sequence[float]): Sequence of coordinates.

    Returns:
        np.ndarray: Homogeneous coordinates.
    """
    xy_array = np.array(
        list(zip(coordinates[0::2], coordinates[1::2])), dtype=float
    )
    n_rows = xy_array.shape[0]
    ones = np.ones((n_rows, 1), dtype=float)
    homogeneous_array = np.append(xy_array, ones, axis=1)

    return homogeneous_array


def on_segment(a, b, p, eps=1e-12):
    """Return True if point ``p`` lies on segment ``ab`` within ``eps``.

    Args:
        a: Segment start point.
        b: Segment end point.
        p: Query point.
        eps: Numeric tolerance. Defaults to ``1e-12``.

    Returns:
        bool: True if ``p`` is collinear with ``ab`` and inside its bbox.
    """

    # check collinear + within bbox
    def cross(ax, ay, bx, by):
        return ax * by - ay * bx

    def orient(a, b, c):
        # cross((b-a),(c-a))
        return cross(b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1])

    if abs(orient(a, b, p)) > eps:
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
    """
    x0, y0 = point
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    dx = x2 - x1
    dy = y2 - y1
    return abs(dx * (y1 - y0) - (x1 - x0) * dy) / sqrt(dx**2 + dy**2)


def point_to_line_seg_distance(p, lp1, lp2):
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


def flat_points(connected_segments):
    """Return a list of points from a list of connected pairs of points.

    Args:
        connected_segments (list[tuple]): List of connected pairs of points.

    Returns:
        list[PointType]: List of points.
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


def remove_bad_points(points):
    """Remove redundant and collinear points from a list of points.

    Args:
        points (list[PointType]): List of points.

    Returns:
        list[PointType]: List of points with redundant and collinear points removed.
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
    """A 3D vertex."""

    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.type = Types.VERTEX

    def __repr__(self):
        return f"Vertex({self.x}, {self.y}, {self.z})"

    def __eq__(self, other):
        return (
            self[0] == other[0] and self[1] == other[1] and self[2] == other[2]
        )

    def copy(self):
        """Return a new ``Vertex`` with the same coordinates.

        Returns:
            Vertex: Copy of this vertex.
        """
        return Vertex(self.x, self.y, self.z)

    def __add__(self, other):
        return Vertex(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vertex(self.x - other.x, self.y - other.y, self.z - other.z)

    @property
    def coords(self):
        """Return the coordinates as a tuple."""
        return (self.x, self.y, self.z)

    @property
    def array(self):
        """Homogeneous coordinates as a numpy array."""
        return array([self.x, self.y, 1])

    def v_tuple(self):
        """Return the vertex as a tuple."""
        return (self.x, self.y, self.z)

    def below(self, other):
        """This is for 2D points only

        Args:
            other (Vertex): Other vertex.

        Returns:
            bool: True if this vertex is below the other vertex, False otherwise.
        """
        res = False
        if self.y < other.y or self.y == other.y and self.x > other.x:
            res = True
        return res

    def above(self, other):
        """This is for 2D points only

        Args:
            other (Vertex): Other vertex.

        Returns:
            bool: True if this vertex is above the other vertex, False otherwise.
        """
        if self.y > other.y or self.y == other.y and self.x < other.x:
            res = True
        else:
            res = False

        return res


def set_vertices(points):
    """Set the next and previous vertices of a list of vertices.

    Args:
        points (list[Vertex]): List of vertices.
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
        p.angle = cross_product_sense(p.prev, p, p.next)


def get_interior_points(start, end, n_points):
    """Given start and end points and number of interior points
    returns the positions of the interior points

    Args:
        start (PointType): Start point.
        end (PointType): End point.
        n_points (int): Number of interior points.

    Returns:
        list[PointType]: List of interior points.
    """
    from simetri.geom.segments.line_utils import line_angle

    rot_angle = line_angle(start, end)
    length_ = distance(start, end)
    seg_length = length_ / (n_points + 1.0)
    points = []
    for i in range(n_points):
        points.append(
            rotate_point(
                [start[0] + seg_length * (i + 1), start[1]], start, rot_angle
            )
        )
    return points


def project_point_on_line(point: "Vertex", line: "Edge"):
    """Given a point and a line, returns the projection of the point on the line

    Args:
        point (Vertex): Input point.
        line (Edge): Input line.

    Returns:
        Vertex: Projection of the point on the line.
    """
    v = point
    a, b = line

    av = v - a
    ab = b - a
    t = (av * ab) / (ab * ab)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return a + ab * t

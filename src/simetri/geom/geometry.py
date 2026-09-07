"""Core 2D geometry operations used across simetri.

Includes line clipping, intersections, polygon simplicity tests, trimming,
fillets, and related utilities. Many helpers are also re-exported via
``import simetri.graphics as sg``.
"""

# To do: Clean up this module.

from __future__ import annotations

import re
from collections.abc import Callable
from math import (
    acos,
    atan2,
    cos,
    floor,
    hypot,
    isclose,
    pi,
    sin,
    sqrt,
)
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import around, array
from numpy.typing import NDArray

from ..base.common import (
    PointType,
    get_defaults,
)
from ..config.settings import defaults
from .geom_utils import close_points2, connected_pairs
from .vectors import *

if TYPE_CHECKING:
    from ..shapes.shape import Shape

tau = 2 * pi  # 360 degrees


def positive_angle(angle, radians=True, rel_tol=None, abs_tol=None):
    """Return the positive angle in radians or degrees.

    Args:
        angle (float): Input angle.
        radians (bool, optional): Whether the angle is in radians. Defaults to True.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        float: Positive angle.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.positive_angle(-sg.pi / 2) == 1.5 * sg.pi
        True
        >>> sg.positive_angle(-90, radians=False)
        270
    """
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    if radians:
        if angle < 0:
            angle += 2 * pi
    else:
        if angle < 0:
            angle += 360

    return angle


def equal_angles(
    angle1: float,
    angle2: float,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if two angles are equal within tolerance.

    Negative angles are converted to positive values before comparison.

    Args:
        angle1: First angle in radians.
        angle2: Second angle in radians.
        rel_tol: Relative tolerance. Defaults to ``defaults[\"rel_tol\"]``.
        abs_tol: Absolute tolerance. Defaults to ``defaults[\"abs_tol\"]``.

    Returns:
        bool: True if the angles match within tolerance.
    """
    if rel_tol is None:
        rel_tol = defaults["rel_tol"]

    if abs_tol is None:
        abs_tol = defaults["abs_tol"]

    angle1 = positive_angle(angle1)
    angle2 = positive_angle(angle2)

    return isclose(angle1, angle2, rel_tol=rel_tol, abs_tol=abs_tol)


def triangle_centroid(p1, p2, p3):
    """Return the centroid of a triangle given its three vertices.

    Args:
        p1: First vertex ``(x, y)``.
        p2: Second vertex ``(x, y)``.
        p3: Third vertex ``(x, y)``.

    Returns:
        tuple: Centroid ``(cx, cy)``.
    """

    cx = (p1[0] + p2[0] + p3[0]) / 3
    cy = (p1[1] + p2[1] + p3[1]) / 3

    return (cx, cy)


def triangle_angles_from_sides(
    a: float, b: float, c: float
) -> tuple[float, float, float]:
    """
    Calculates the inner angles of a triangle given its side lengths.

    Args:
        a float: Length of side a.
        b float: Length of side b.
        c float: Length of side c.

    Returns:
        tuple[float, float, float]: A tuple containing the angles (A, B, C) in degrees.
    """
    a2 = a * a
    b2 = b * b
    c2 = c * c
    A = acos((b2 + c2 - a2) / (2 * b * c))
    B = acos((a2 + c2 - b2) / (2 * a * c))
    C = acos((a2 + b2 - c2) / (2 * a * b))

    return A, B, C


def close_angles(angle1: float, angle2: float, angtol=None) -> bool:
    """
    Return True if two angles are close to each other.

    Args:
        angle1 (float): First angle in radians.
        angle2 (float): Second angle in radians.
        angtol (float, optional): Angle tolerance. Defaults to None.

    Returns:
        bool: True if the angles are close to each other, False otherwise.
    """
    if angtol is None:
        angtol = defaults["angtol"]

    return (abs(angle1 - angle2) % (2 * pi)) < angtol


def connect2(
    poly_point1: list[PointType],
    poly_point2: list[PointType],
    dist_tol: float | None = None,
    rel_tol: float | None = None,
) -> list[PointType]:
    """
    Connect two polypoints together.

    Args:
        poly_point1 (list[PointType]): First list of points.
        poly_point2 (list[PointType]): Second list of points.
        dist_tol (float, optional): Distance tolerance. Defaults to None.
        rel_tol (float, optional): Relative tolerance. Defaults to None.

    Returns:
        list[PointType]: Connected list of points.
    """
    rel_tol, dist_tol = get_defaults(
        ["rel_tol", "dist_tol"], [rel_tol, dist_tol]
    )
    dist_tol2 = dist_tol * dist_tol
    start1, end1 = poly_point1[0], poly_point1[-1]
    start2, end2 = poly_point2[0], poly_point2[-1]
    pp1 = poly_point1[:]
    pp2 = poly_point2[:]
    points = []
    if close_points2(end1, start2, dist2=dist_tol2):
        points.extend(pp1)
        points.extend(pp2[1:])
    elif close_points2(end1, end2, dist2=dist_tol2):
        points.extend(pp1)
        pp2.reverse()
        points.extend(pp2[1:])
    elif close_points2(start1, start2, dist2=dist_tol2):
        pp1.reverse()
        points.extend(pp1)
        points.extend(pp2[1:])
    elif close_points2(start1, end2, dist2=dist_tol2):
        pp1.reverse()
        points.extend(pp1)
        pp2.reverse()
        points.extend(pp2[1:])

    return points


def trim_shape(shape: Shape, trim_func: Callable, value: float):
    """
    Trim a shape using a specified trim function and value.

    Args:
        shape (Shape): The shape to trim.
        trim_func (function): The trimming function to use. trim_righ, trim_left,
                                                        trim_top, or trim_bottom.
        value (float): The value at which to trim the shape.

    Returns:
        Shape: The trimmed shape.
    """
    new_shape = shape.copy()
    edges = []
    for edge in shape.edges:
        trimmed = trim_func(edge, value)
        if trimmed:
            edges.append(trimmed)

    points = []
    for edge in edges:
        if edge:
            p1, p2 = edge
            if points:
                if points[-1] != p1:
                    points.append(p1)
            else:
                points.append(p1)
            points.append(p2)
    from .points.point_utils import remove_duplicate_points

    points = remove_duplicate_points(points)
    new_shape[:] = points
    return new_shape


def global_to_local(
    x: float, y: float, xi: float, yi: float, theta: float = 0
) -> PointType:
    """Given a point(x, y) in global coordinates
    and local CS position and orientation,
    return a point(ksi, eta) in local coordinates

    Args:
        x (float): Global x-coordinate.
        y (float): Global y-coordinate.
        xi (float): Local x-coordinate.
        yi (float): Local y-coordinate.
        theta (float, optional): Angle in radians. Defaults to 0.

    Returns:
        PointType: Local coordinates (ksi, eta).
    """
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    ksi = (x - xi) * cos_theta + (y - yi) * sin_theta
    eta = (y - yi) * cos_theta - (x - xi) * sin_theta
    return (ksi, eta)


def get_quadrant(x: float, y: float) -> int:
    """quadrants:
    +x, +y = 1st
    +x, -y = 2nd
    -x, -y = 3rd
    +x, -y = 4th

    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.

    Returns:
        int: Quadrant number.
    """
    return int(floor((atan2(y, x) % (tau)) / (pi / 2)) + 1)


def get_quadrant_from_deg_angle(deg_angle: float) -> int:
    """quadrants:
    (0, 90) = 1st
    (90, 180) = 2nd
    (180, 270) = 3rd
    (270, 360) = 4th

    Args:
        deg_angle (float): Angle in degrees.

    Returns:
        int: Quadrant number.
    """
    return int(floor(deg_angle / 90.0) % 4 + 1)

    # return radius**2 * (p - center) / dist


def ndarray_to_xy_list(arr: NDArray) -> Sequence[PointType]:
    """Convert a numpy array to a list of points.

    Args:
        arr (np.ndarray): Input numpy array.

    Returns:
        Sequence[PointType]: List of points.
    """
    return arr[:, :2].tolist()


def radius_to_side_len(n: int, radius: float) -> float:
    """Given a radius and the number of sides, return the side length
    of an n-sided regular polygon with the given radius

    Args:
        n (int): Number of sides.
        radius (float): Radius of the polygon.

    Returns:
        float: Side length of the polygon.
    """
    return 2 * radius * sin(pi / n)


def tokenize_svg_path(path: str) -> list[str]:
    """Tokenize an SVG path string.

    Args:
        path (str): SVG path string.

    Returns:
        list[str]: List of tokens.
    """
    return re.findall(r"[a-zA-Z]|[-+]?\d*\.\d+|\d+", path)


def law_of_cosines(a: float, b: float, c: float) -> float:
    """Return the angle of a triangle given the three sides.
    Returns the angle of A in radians. A is the angle between
    sides b and c.
    cos(A) = (b^2 + c^2 - a^2) / (2 * b * c)

    Args:
        a (float): Length of side a.
        b (float): Length of side b.
        c (float): Length of side c.

    Returns:
        float: Angle of A in radians.
    """
    return acos((b**2 + c**2 - a**2) / (2 * b * c))


def side_len_to_radius(n: int, side_len: float) -> float:
    """Given a side length and the number of sides, return the radius
    of an n-sided regular polygon with the given side_len length

    Args:
        n (int): Number of sides.
        side_len (float): Side length of the polygon.

    Returns:
        float: Radius of the polygon.
    """
    return side_len / (2 * sin(pi / n))


def tri_to_cart(points):
    """
    Convert a list of points from triangular to cartesian coordinates.

    Args:
        points (list[PointType]): List of points in triangular coordinates.

    Returns:
        np.ndarray: List of points in cartesian coordinates.
    """
    u = [1, 0]
    v = cos(pi / 3), sin(pi / 3)
    convert = array([u, v])

    return array(points) @ convert


def cart_to_tri(points):
    """
    Convert a list of points from cartesian to triangular coordinates.

    Args:
        points (list[PointType]): List of points in cartesian coordinates.

    Returns:
        np.ndarray: List of points in triangular coordinates.
    """
    u = [1, 0]
    v = cos(pi / 3), sin(pi / 3)
    convert = np.linalg.inv(array([u, v]))

    return array(points) @ convert


def triangle_area(a: float, b: float, c: float) -> float:
    """
    Given side lengths a, b and c, return the area of the triangle.

    Args:
        a (float): Length of side a.
        b (float): Length of side b.
        c (float): Length of side c.

    Returns:
        float: Area of the triangle.
    """
    a_b = a - b
    return sqrt((a + (b + c)) * (c - (a_b)) * (c + (a_b)) * (a + (b - c))) / 4


def bbox_overlap(
    min_x1: float,
    min_y1: float,
    max_x2: float,
    max_y2: float,
    min_x3: float,
    min_y3: float,
    max_x4: float,
    max_y4: float,
) -> bool:
    """
    Given two bounding boxes, return True if they overlap.

    Args:
        min_x1 (float): Minimum x-coordinate of the first bounding box.
        min_y1 (float): Minimum y-coordinate of the first bounding box.
        max_x2 (float): Maximum x-coordinate of the first bounding box.
        max_y2 (float): Maximum y-coordinate of the first bounding box.
        min_x3 (float): Minimum x-coordinate of the second bounding box.
        min_y3 (float): Minimum y-coordinate of the second bounding box.
        max_x4 (float): Maximum x-coordinate of the second bounding box.
        max_y4 (float): Maximum y-coordinate of the second bounding box.

    Returns:
        bool: True if the bounding boxes overlap, False otherwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.bbox_overlap(0, 0, 2, 2, 1, 1, 3, 3)
        True
        >>> sg.bbox_overlap(0, 0, 1, 1, 2, 2, 3, 3)
        False
    """
    return not (
        max_x2 < min_x3 or max_x4 < min_x1 or max_y2 < min_y3 or max_y4 < min_y1
    )


def polar_to_cartesian(r, theta, center=(0, 0)):
    """Convert polar coordinates to cartesian coordinates.

    Args:
        r (float): Radius.
        theta (float): Angle in radians.

    Returns:
        PointType: Cartesian coordinates.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.polar_to_cartesian(1, 0)
        (1.0, 0.0)
        >>> x, y = sg.polar_to_cartesian(1, sg.pi / 2)
        >>> round(x, 10), round(y, 10)
        (0.0, 1.0)
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

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.cartesian_to_polar(1, 0)
        (1.0, 0.0)
        >>> r, theta = sg.cartesian_to_polar(0, 1)
        >>> r, theta == sg.pi / 2
        (1.0, True)
    """
    dx, dy = center
    x -= dx
    y -= dy
    r = hypot(x, y)
    theta = positive_angle(atan2(y, x))
    return r, theta


def double_area(a, b, c):
    """Return twice the signed area of triangle ``abc``.

    Computes the 2D cross product of ``AB`` and ``AC``:

    ``(b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)``.

    That value is the signed area of the parallelogram spanned by those
    vectors, i.e. **twice** the signed triangle area. Positive when
    ``a → b → c`` is counterclockwise, negative when clockwise, and near
    zero when the points are collinear. Kept as ``2 * area`` so orientation
    and collinearity tests can compare against ``area_tol`` without an
    extra multiply/divide.

    Args:
        a (PointType): First vertex.
        b (PointType): Second vertex.
        c (PointType): Third vertex.

    Returns:
        float: Twice the signed triangle area (parallelogram cross product).

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.double_area((0, 0), (1, 0), (0, 1))
        1
        >>> sg.double_area((0, 0), (1, 0), (0, 1)) / 2  # geometric triangle area
        0.5
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])

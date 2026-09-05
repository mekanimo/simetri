"""Circle geometry: tangents, Apollonius, Steiner chains, and related helpers."""

import cmath
from dataclasses import dataclass
from math import acos, atan, atan2, cos, pi, sin, sqrt

import numpy as np

from simetri.geom.affine import rotate_point
from simetri.geom.vectors import atan2, cos, distance, sin, sqrt

from ...shapes.geom_items import Circle
from ..affine import rotate, rotation_matrix, scale_matrix
from ..geom_utils import offset_point_from_start, r_polar
from ..geometry import side_len_to_radius
from ..homogenize import homogenize
from ..points.point_utils import distance
from ..segments.line_utils import angle_between_lines2, angle_between_two_lines

array = np.array
dot = np.dot
linalg = np.linalg


@dataclass
class Circle_:
    """Lightweight circle with center and radius (not a drawable Shape).

    Attributes:
        center: Center point ``(x, y)``.
        radius: Circle radius.
    """

    center: tuple
    radius: float


def tangent_points_from_point(circle, point):
    """Return the two tangent points from an external point to a circle.

    Args:
        circle: Object with ``center`` and ``radius`` attributes (e.g. ``Circle_``).
        point: External point ``(x, y)``.

    Returns:
        tuple: Two tangent points ``(p1, p2)`` on the circle.
    """

    x, y = point[:2]
    cx, cy = circle.center[:2]
    r = circle.radius

    d = distance(circle.center, point)

    theta = acos(r / d)

    phi = atan2(y - cy, x - cx)

    # tangent point 1
    p1 = (cx + r * cos(phi + theta), cy + r * sin(phi + theta))

    # tangent point 2
    p2 = (cx + r * cos(phi - theta), cy + r * sin(phi - theta))

    return (p1, p2)


def circle_tangent_to_3_circles(c1, r1, c2, r2, c3, r3, s1=-1, s2=-1, s3=-1):
    """Given the centers and radii of 3 circles, return the center and radius
    of a circle that is tangent to all 3 circles.

    Args:
        c1 (tuple): Center of the first circle.
        r1 (float): Radius of the first circle.
        c2 (tuple): Center of the second circle.
        r2 (float): Radius of the second circle.
        c3 (tuple): Center of the third circle.
        r3 (float): Radius of the third circle.
        s1 (int, optional): Sign for the first circle. Defaults to -1.
        s2 (int, optional): Sign for the second circle. Defaults to -1.
        s3 (int, optional): Sign for the third circle. Defaults to -1.

    Returns:
        tuple: Center (x, y) and radius of the tangent circle.
    """

    x1, y1 = c1
    x2, y2 = c2
    x3, y3 = c3

    v11 = 2 * x2 - 2 * x1
    v12 = 2 * y2 - 2 * y1
    v13 = x1 * x1 - x2 * x2 + y1 * y1 - y2 * y2 - r1 * r1 + r2 * r2
    v14 = 2 * s2 * r2 - 2 * s1 * r1

    v21 = 2 * x3 - 2 * x2
    v22 = 2 * y3 - 2 * y2
    v23 = x2 * x2 - x3 * x3 + y2 * y2 - y3 * y3 - r2 * r2 + r3 * r3
    v24 = 2 * s3 * r3 - 2 * s3 * r2

    w12 = v12 / v11
    w13 = v13 / v11
    w14 = v14 / v11

    w22 = v22 / v21 - w12
    w23 = v23 / v21 - w13
    w24 = v24 / v21 - w14

    P = -w23 / w22
    Q = w24 / w22
    M = -w12 * P - w13
    N = w14 - w12 * Q

    a = N * N + Q * Q - 1
    b = 2 * M * N - 2 * N * x1 + 2 * P * Q - 2 * Q * y1 + 2 * s1 * r1
    c = x1 * x1 + M * M - 2 * M * x1 + P * P + y1 * y1 - 2 * P * y1 - r1 * r1

    # Find a root of a quadratic equation.
    # This requires the circle centers not to be collinear
    D = b * b - 4 * a * c
    rs = (-b - sqrt(D)) / (2 * a)

    xs = M + N * rs
    ys = P + Q * rs

    return (xs, ys, rs)


def apollonius(r1, r2, r3, z1, z2, z3, plus_minus=1):
    """Solves the Problem of Apollonius using Descartes' Theorem.

    Args:
        r1 (float): Radius of the first circle.
        r2 (float): Radius of the second circle.
        r3 (float): Radius of the third circle.
        z1 (complex): Center of the first circle.
        z2 (complex): Center of the second circle.
        z3 (complex): Center of the third circle.
        plus_minus (int, optional): +1 for outer tangent circle, -1 for inner tangent circle. Defaults to 1.

    Returns:
        tuple: Radius and center coordinates (x, y) of the tangent circle, or None if no solution is found.
    """
    k1, k2, k3 = 1 / r1, 1 / r2, 1 / r3

    # Applying Descartes' Theorem
    k4_values = (k1 + k2 + k3) + plus_minus * 2 * sqrt(
        k1 * k2 + k2 * k3 + k3 * k1
    )

    # Handle cases where no solution exists (e.g., division by zero)
    if k4_values == 0:
        return None

    r4 = 1 / k4_values
    z4 = (
        k1 * z1
        + k2 * z2
        + k3 * z3
        + plus_minus
        * 2
        * cmath.sqrt(k1 * k2 * z1 * z2 + k2 * k3 * z2 * z3 + k3 * k1 * z3 * z1)
    ) / k4_values

    return r4, z4


def circle_tangent_to_2_circles(c1, r1, c2, r2, r):
    """Given the centers and radii of 2 circles, return the center
    of a circle with radius r that is tangent to both circles.

    Args:
        c1 (tuple): Center of the first circle.
        r1 (float): Radius of the first circle.
        c2 (tuple): Center of the second circle.
        r2 (float): Radius of the second circle.
        r (float): Radius of the tangent circle.

    Returns:
        tuple: Centers (x1, y1) and (x2, y2) of the tangent circle.
    """
    x1, y1 = c1
    x2, y2 = c2

    r12 = r1**2
    r12y1 = r12 * y1
    r12y2 = r12 * y2
    r1r2 = r1 * r2
    r22 = r2**2
    r22y2 = r22 * y2
    r_2 = r**2
    rr1 = r * r1
    rr1y1 = rr1 * y1
    rr1y2 = rr1 * y2
    rr2 = r * r2
    rr2y1 = rr2 * y1
    rr2y2 = rr2 * y2
    x12 = x1**2
    x12y1 = x12 * y1
    x12y2 = x12 * y2
    x1_x2 = x1 - x2
    x1x2 = x1 * x2
    x1x2y1 = x1x2 * y1
    x1x2y2 = x1x2 * y2
    x22 = x2**2
    x22y1 = x22 * y1
    x22y2 = x22 * y2
    y12 = y1**2
    y12y2 = y12 * y2
    y13 = y1**3
    y1y2 = y1 * y2
    y22 = y2**2
    y1y22 = y1 * y22
    y23 = y2**3

    x_1 = (
        -(y1 - y2)
        * (
            -2 * rr1y1
            + 2 * rr1y2
            + 2 * rr2y1
            - 2 * rr2y2
            - r12y1
            + r12y2
            + r22 * y1
            - r22y2
            + x12y1
            + x12y2
            - 2 * x1x2y1
            - 2 * x1x2y2
            + x22y1
            + x22y2
            + y13
            - y12y2
            - y1y22
            + y23
            + sqrt(
                (
                    -r12
                    + 2 * r1r2
                    - r22
                    + x12
                    - 2 * x1x2
                    + x22
                    + y12
                    - 2 * y1y2
                    + y22
                )
                * (
                    4 * r_2
                    + 4 * rr1
                    + 4 * rr2
                    + r12
                    + 2 * r1r2
                    + r22
                    - x12
                    + 2 * x1x2
                    - x22
                    - y12
                    + 2 * y1y2
                    - y22
                )
            )
            * (-x1 + x2)
        )
        - (x12 - 2 * x1x2 + x22 + y12 - 2 * y1y2 + y22)
        * (2 * rr1 - 2 * rr2 + r12 - r22 - x12 + x22 - y12 + y22)
    ) / (2 * (x1_x2) * (x12 - 2 * x1x2 + x22 + y12 - 2 * y1y2 + y22))

    y_1 = (
        -2 * rr1y1
        + 2 * rr1y2
        + 2 * rr2y1
        - 2 * rr2y2
        - r12y1
        + r12y2
        + r22 * y1
        - r22y2
        + x12y1
        + x12y2
        - 2 * x1x2y1
        - 2 * x1x2y2
        + x22y1
        + x22y2
        + y13
        - y12y2
        - y1y22
        + y23
        + sqrt(
            (
                -r12
                + 2 * r1r2
                - r22
                + x12
                - 2 * x1x2
                + x22
                + y12
                - 2 * y1y2
                + y22
            )
            * (
                4 * r_2
                + 4 * rr1
                + 4 * rr2
                + r12
                + 2 * r1r2
                + r22
                - x12
                + 2 * x1x2
                - x22
                - y12
                + 2 * y1y2
                - y22
            )
        )
        * (-x1 + x2)
    ) / (2 * (x12 - 2 * x1x2 + x22 + y12 - 2 * y1y2 + y22))

    x_2 = (
        -(y1 - y2)
        * (
            -2 * rr1y1
            + 2 * rr1y2
            + 2 * rr2y1
            - 2 * rr2y2
            - r12y1
            + r12y2
            + r22 * y1
            - r22y2
            + x12y1
            + x12y2
            - 2 * x1x2y1
            - 2 * x1x2y2
            + x22y1
            + x22y2
            + y13
            - y12y2
            - y1y22
            + y23
            + sqrt(
                (
                    -r12
                    + 2 * r1r2
                    - r22
                    + x12
                    - 2 * x1x2
                    + x22
                    + y12
                    - 2 * y1y2
                    + y22
                )
                * (
                    4 * r_2
                    + 4 * rr1
                    + 4 * rr2
                    + r12
                    + 2 * r1r2
                    + r22
                    - x12
                    + 2 * x1x2
                    - x22
                    - y12
                    + 2 * y1y2
                    - y22
                )
            )
            * (x1_x2)
        )
        - (x12 - 2 * x1x2 + x22 + y12 - 2 * y1y2 + y22)
        * (2 * rr1 - 2 * rr2 + r12 - r22 - x12 + x22 - y12 + y22)
    ) / (2 * (x1_x2) * (x12 - 2 * x1x2 + x22 + y12 - 2 * y1y2 + y22))

    y_2 = (
        -2 * rr1y1
        + 2 * rr1y2
        + 2 * rr2y1
        - 2 * rr2y2
        - r12y1
        + r12y2
        + r22 * y1
        - r22y2
        + x12y1
        + x12y2
        - 2 * x1x2y1
        - 2 * x1x2y2
        + x22y1
        + x22y2
        + y13
        - y12y2
        - y1y22
        + y23
        + sqrt(
            (
                -r12
                + 2 * r1r2
                - r22
                + x12
                - 2 * x1x2
                + x22
                + y12
                - 2 * y1y2
                + y22
            )
            * (
                4 * r_2
                + 4 * rr1
                + 4 * rr2
                + r12
                + 2 * r1r2
                + r22
                - x12
                + 2 * x1x2
                - x22
                - y12
                + 2 * y1y2
                - y22
            )
        )
        * (x1_x2)
    ) / (2 * (x12 - 2 * x1x2 + x22 + y12 - 2 * y1y2 + y22))

    return ((x_1, y_1), (x_2, y_2))


def tangent_points(center1, radius, center2, radius2, cross=False):
    """Returns the tangent points (p1, p2, p3, p4) in world coordinates.

    Args:
        center1 (tuple): Center of the first circle.
        radius (float): Radius of the first circle.
        center2 (tuple): Center of the second circle.
        radius2 (float): Radius of the second circle.
        cross (bool, optional): Whether to calculate crossing tangents. Defaults to False.

    Returns:
        tuple: Tangent points (p1, p2, p3, p4) in world coordinates.
    """
    c1 = Circle_(center1, radius)
    c2 = Circle_(center2, radius2)
    if radius < radius2:
        c1, c2 = c2, c1
    pos = c1.center
    dist = distance(pos, c2.center)
    r1 = c1.radius
    r2 = c2.radius

    if cross:
        dr = r1 + r2
    else:
        dr = r1 - r2

    x = sqrt(dist**2 - dr**2)
    y = pos[1] + r1
    p1 = [pos[0], y]
    p2 = [pos[0] + x, y]
    points = homogenize([p1, p2])
    alpha = angle_between_lines2((pos[0] + x, pos[1] + dr), pos, c2.center)
    tp1w, tp2w = rotate(points, alpha, pos)

    if x == 0:
        beta = 0
    else:
        beta = pi / 2 - atan(dr / x)
    tp3w = rotate([tp1w], -2 * beta, pos)[0]
    tp4w = rotate([tp2w], -2 * beta, c2.center)[0]

    return (tp1w, tp2w, tp3w, tp4w)


def circle_area(rad):
    """Given the radius of a circle, return the area of the circle.

    Args:
        rad (float): Radius of the circle.

    Returns:
        float: Area of the circle.
    """
    return pi * rad**2


def circle_circumference(rad):
    """Given the radius of a circle, return the circumference of the circle.

    Args:
        rad (float): Radius of the circle.

    Returns:
        float: Circumference of the circle.
    """
    return 2 * pi * rad


def flower_angle(r1, r2, r3):
    """Given the radii of 3 circles forming an interstice, return the angle between
    the lines connecting circles' centers to center of the circle with r1 radius.

    Args:
        r1 (float): Radius of the first circle.
        r2 (float): Radius of the second circle.
        r3 (float): Radius of the third circle.

    Returns:
        float: Angle between the lines connecting circles' centers.
    """
    angle = acos(
        ((r1 + r2) ** 2 + (r1 + r3) ** 2 - (r2 + r3) ** 2)
        / (2 * (r1 + r2) * (r1 + r3))
    )

    return angle


ratios = {
    8: 0.4974,
    9: 0.5394,
    10: 0.575,
    11: 0.6056,
    12: 0.6321,
    13: 0.6553,
    14: 0.6757,
    15: 0.6939,
    16: 0.7101,
    17: 0.7248,
    18: 0.738,
    19: 0.75,
    20: 0.7609,
}


def circle_flower(n, radius=25, layers=6, ratio=None):
    """Return a Steiner-chain style circle flower pattern.

    Args:
        n: Number of circles around the ring (must be >= 8).
        radius: Circle radius. Defaults to 25.
        layers: Number of scaled/rotated layers. Defaults to 6.
        ratio: Scale factor between layers; if None, uses a tabulated or
            fitted value for ``n``.

    Returns:
        Group: Circles forming a flower-like pattern after layered transforms.

    Raises:
        ValueError: If ``n`` is less than 8.

    Examples:
        ::

            import simetri.graphics as sg

            flowers = sg.circle_flower(n=8, radius=20, layers=4)
            canvas = sg.Canvas()
            canvas.draw(flowers)
    """
    if n < 8:
        raise ValueError("n must be greater than 7")
    if ratio is None:
        if n < 21:
            ratio = ratios[n]
        else:
            ratio = (
                -0.000000089767 * n**4
                + 0.000015821834 * n**3
                + -0.001100867708 * n**2
                + 0.038096046379 * n
                + 0.327363569038
            )

    r1 = side_len_to_radius(n, 2 * radius)
    circles = Circle((r1, 0), radius).rotate(pi / (n / 2), (0, 0), reps=n - 1)
    xform = scale_matrix(ratio) @ rotation_matrix(pi / n)

    return circles.transform(xform_matrix=xform, reps=layers)


def circle_inversion(point, center, radius):
    """
    Inverts a point with respect to a circle.

    Args:
        point (tuple): The point to invert, represented as a tuple (x, y).
        center (tuple): The center of the circle, represented as a tuple (x, y).
        radius (float): The radius of the circle.

    Returns:
        tuple: The inverted point, represented as a tuple (x, y).
    """
    x, y = point[:2]
    cx, cy = center[:2]
    # Calculate the distance from the point to the center of the circle
    dist = sqrt((x - cx) ** 2 + (y - cy) ** 2)
    # If the point is at the center of the circle, return the point at infinity
    if dist == 0:
        return float("inf"), float("inf")
    # Calculate the distance from the inverted point to the center of the circle
    inv_dist = radius**2 / dist
    # Calculate the inverted point
    inv_x = cx + inv_dist * (x - cx) / dist
    inv_y = cy + inv_dist * (y - cy) / dist
    return inv_x, inv_y


def circle_tangent_to2lines(line1, line2, intersection_, radius):
    """Given two lines, their intersection point and a radius,
    return the center of the circle tangent to both lines and
    with the given radius.

    Args:
        line1 (LineType): First line.
        line2 (LineType): Second line.
        intersection_ (PointType): Intersection point of the lines.
        radius (float): Radius of the circle.

    Returns:
        tuple: Center of the circle, start and end points of the tangent lines.
    """
    alpha = angle_between_two_lines(line1, line2)
    dist = radius / sin(alpha / 2)
    start = offset_point_from_start(intersection_, line1.p1, dist)
    center = rotate_point(start, intersection_, alpha / 2)
    end = offset_point_from_start(intersection_, line2.p1, dist)

    return center, start, end


def circle_circle_intersections(point1, radius1, point2, radius2):
    """Return the intersection points of two circles.

    Args:
        point1 (PointType): Center of the first circle.
        radius1 (float): Radius of the first circle.
        point2 (PointType): Center of the second circle.
        radius2 (float): Radius of the second circle.

    Returns:
        tuple: Intersection points of the two circles.
    """
    # taken from https://stackoverflow.com/questions/55816902/finding-the-
    # intersection-of-two-circles
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    x0, y0 = point1[:2]
    x1, y1 = point2[:2]
    r0 = radius1
    r1 = radius2

    d = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # non intersecting
    if d > r0 + r1 or d < abs(r0 - r1) or d == 0 and r0 == r1:
        res = None
    else:
        a = (r0**2 - r1**2 + d**2) / (2 * d)
        h = sqrt(r0**2 - a**2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (x1 - x0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d
        x4 = x2 - h * (y1 - y0) / d
        y4 = y2 + h * (x1 - x0) / d

        res = ((x3, y3), (x4, y4))

    return res


def tfl_by_sides(point1, point2, side1, side2):
    """Triangle from a line segment and two side lengths.

    Returns the third vertex candidates (circle-circle intersections) for the
    triangle given by two points and the two adjacent side lengths.

    Args:
        point1: First point of the line segment.
        point2: Second point of the line segment.
        side1: Length of the side from ``point1`` to the third vertex.
        side2: Length of the side from ``point2`` to the third vertex.

    Returns:
        Intersection points of the two circles, or ``None`` if none exist.
    """
    c = sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    if c == 0:
        raise ValueError("Error! Points are coincident.")

    if side1 + side2 < c:
        raise ValueError(
            "Error! The sum of the sides is less than the distance between the points."
        )

    if side1 + c < side2:
        raise ValueError(
            "Error! The sum of the first side and the distance between the points is less than the second side."
        )

    if side2 + c < side1:
        raise ValueError(
            "Error! The sum of the second side and the distance between the points is less than the first side."
        )

    if side1 == 0 or side2 == 0:
        raise ValueError("Error! One of the sides is zero.")

    if side1 < 0 or side2 < 0:
        raise ValueError("Error! One of the sides is negative.")

    return circle_circle_intersections(point1, side1, point2, side2)


def circle_segment_intersection(circle, p1, p2):
    """Return True if the circle and the line segment intersect.

    Args:
        circle (Circle): Input circle.
        p1 (PointType): First point of the line segment.
        p2 (PointType): Second point of the line segment.

    Returns:
        bool: True if the circle and the line segment intersect, False otherwise.
    """
    # if line seg and circle intersects returns true, false otherwise
    # c: circle
    # p1 and p2 are the endpoints of the line segment

    x3, y3 = circle.pos[:2]
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    if (
        distance(p1, circle.pos) < circle.radius
        or distance(p2, circle.pos) < circle.radius
    ):
        return True
    u = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / (
        (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
    )
    res = False
    if 0 <= u <= 1:
        x = x1 + u * (x2 - x1)
        y = y1 + u * (y2 - y1)
        if distance((x, y), circle.pos) < circle.radius:
            res = True

    return res  # p is not between lp1 and lp2


def ellipse_line_intersection(a, b, point):
    """Return the intersection points of an ellipse and a line segment
    connecting the given point to the ellipse center at (0, 0).

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        point (PointType): PointType on the line segment.

    Returns:
        list[PointType]: Intersection points of the ellipse and the line segment.
    """
    # adapted from http://mathworld.wolfram.com/Ellipse-LineIntersection.html
    # a, b is the ellipse width/2 and height/2 and (x_0, y_0) is the point

    x_0, y_0 = point[:2]
    x = ((a * b) / (sqrt(a**2 * y_0**2 + b**2 * x_0**2))) * x_0
    y = ((a * b) / (sqrt(a**2 * y_0**2 + b**2 * x_0**2))) * y_0

    return [(x, y), (-x, -y)]


def ellipse_tangent(a, b, x, y, tol=0.001):
    """Calculates the slope of the tangent line to an ellipse at the point (x, y).
    If point is not on the ellipse, return False.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.
        tol (float, optional): Tolerance. Defaults to 0.001.

    Returns:
        float: Slope of the tangent line, or False if the point is not on the ellipse.
    """
    if abs((x**2 / a**2) + (y**2 / b**2) - 1) > tol:
        res = False
    else:
        res = -(b**2 * x) / (a**2 * y)

    return res


def elliptic_arclength(t_0, t_1, a, b):
    """Return the arclength of an ellipse between the given parametric angles.
    The ellipse has semi-major axis a and semi-minor axis b.

    Args:
        t_0 (float): Start parametric angle in radians.
        t_1 (float): End parametric angle in radians.
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.

    Returns:
        float: Arclength of the ellipse between the given parametric angles.
    """
    # from: https://www.johndcook.com/blog/2022/11/02/elliptic-arc-length/
    from scipy.special import ellipeinc  # this takes too long to import!!!

    m = 1 - (b / a) ** 2
    t1 = ellipeinc(t_1 - 0.5 * pi, m)
    t0 = ellipeinc(t_0 - 0.5 * pi, m)
    return a * (t1 - t0)


def central_to_parametric_angle(a, b, phi):
    """
    Converts a central angle to a parametric angle on an ellipse.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        phi (float): Central angle in radians.

    Returns:
        float: Parametric angle in radians.
    """
    t = atan2((a / b) * sin(phi), cos(phi))
    if t < 0:
        t += 2 * pi

    return t


def parametric_to_central_angle(a, b, t):
    """
    Converts a parametric angle on an ellipse to a central angle.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        t (float): Parametric angle in radians.

    Returns:
        float: Central angle in radians.
    """
    phi = atan2((b / a) * sin(t), cos(t))
    if phi < 0:
        phi += 2 * pi

    return phi


def ellipse_points(center, a, b, n_points):
    """Generate points on an ellipse.

    Args:
        center (tuple): (x, y) coordinates of the ellipse center.
        a (float): Length of the semi-major axis.
        b (float): Length of the semi-minor axis.
        n_points (int): Number of points to generate.

    Returns:
        np.ndarray: Array of (x, y) coordinates of the ellipse points.
    """
    t = np.linspace(0, 2 * np.pi, n_points)
    x = center[0] + a * np.cos(t)
    y = center[1] + b * np.sin(t)

    return np.column_stack((x, y))


def ellipse_point(a, b, angle):
    """Return a point on an ellipse with the given a=width/2, b=height/2, and angle.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        angle (float): Angle in radians.

    Returns:
        PointType: PointType on the ellipse.
    """
    r = r_polar(a, b, angle)

    return (r * cos(angle), r * sin(angle))


def circle_line_intersection(c, p1, p2):
    """Return the intersection points of a circle and a line segment.

    Args:
        c (Circle): Input circle.
        p1 (PointType): First point of the line segment.
        p2 (PointType): Second point of the line segment.

    Returns:
        tuple: Intersection points of the circle and the line segment.
    """

    # adapted from http://mathworld.wolfram.com/Circle-LineIntersection.html
    # c is the circle and p1 and p2 are the line points
    def sgn(num):
        if num < 0:
            res = -1
        else:
            res = 1
        return res

    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    r = c.radius
    x, y = c.pos[:2]

    x1 -= x
    x2 -= x
    y1 -= y
    y2 -= y

    dx = x2 - x1
    dy = y2 - y1
    dr = sqrt(dx**2 + dy**2)
    d = x1 * y2 - x2 * y1
    d2 = d**2
    r2 = r**2
    dr2 = dr**2

    discriminant = r2 * dr2 - d2

    if discriminant > 0:
        ddy = d * dy
        ddx = d * dx
        sqrterm = sqrt(r2 * dr2 - d2)
        temp = sgn(dy) * dx * sqrterm

        a = (ddy + temp) / dr2
        b = (-ddx + abs(dy) * sqrterm) / dr2
        if discriminant == 0:
            res = (a + x, b + y)
        else:
            c = (ddy - temp) / dr2
            d = (-ddx - abs(dy) * sqrterm) / dr2
            res = ((a + x, b + y), (c + x, d + y))

    else:
        res = False

    return res


def circle_poly_intersection(circle, polygon):
    """Return True if the circle and the polygon intersect.

    Args:
        circle (Circle): Input circle.
        polygon (Polygon): Input polygon.

    Returns:
        bool: True if the circle and the polygon intersect, False otherwise.
    """
    points = polygon.vertices
    n = len(points)
    res = False
    for i in range(n):
        x = points[i][0]
        y = points[i][1]
        x1 = points[(i + 1) % n][0]
        y1 = points[(i + 1) % n][1]
        if circle_segment_intersection(circle, (x, y), (x1, y1)):
            res = True
            break
    return res


def point_to_circle_distance(point, center, radius):
    """Given a point, center point, and radius, returns distance
    between the given point and the circle

    Args:
        point (PointType): Input point.
        center (PointType): Center of the circle.
        radius (float): Radius of the circle.

    Returns:
        float: Distance between the point and the circle.
    """
    return abs(distance(center, point) - radius)


def circle_3point(point1, point2, point3):
    """Given three points, returns the center point and radius

    Args:
        point1 (PointType): First point.
        point2 (PointType): Second point.
        point3 (PointType): Third point.

    Returns:
        tuple: Center point and radius of the circle.
    """
    ax, ay = point1[:2]
    bx, by = point2[:2]
    cx, cy = point3[:2]
    a = bx - ax
    b = by - ay
    c = cx - ax
    d = cy - ay
    e = a * (ax + bx) + b * (ay + by)
    f = c * (ax + cx) + d * (ay + cy)
    g = 2.0 * (a * (cy - by) - b * (cx - bx))
    if g == 0:
        raise ValueError("Points are collinear!")

    px = ((d * e) - (b * f)) / g
    py = ((a * f) - (c * e)) / g
    r = ((ax - px) ** 2 + (ay - py) ** 2) ** 0.5
    return ((px, py), r)

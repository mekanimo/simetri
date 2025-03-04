from math import pi, cos, sqrt, acos, atan
from dataclasses import dataclass
import cmath

import numpy as np

from .geometry import distance, homogenize, side_len_to_radius
from ..graphics.affine import rotate, scale_matrix, rotation_matrix
from ..graphics.shapes import Circle

array = np.array
dot = np.dot
linalg = np.linalg

@dataclass
class Circle_:
    """A simple circle class."""

    center: tuple
    radius: float

def circle_tangent_to_3_circles(c1, r1, c2, r2, c3, r3, s1=-1, s2=-1, s3=-1):
    """Given the centers and radii of 3 circles, return the center and radius
    of a circle that is tangent to all 3 circles."""

    # if internal:
    #     s1 = s2 = s3 = 1
    # if external:
    #     s1 = s2 = s3 = -1

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

    a = N*N + Q*Q - 1
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
    """
    Solves the Problem of Apollonius using Descartes' Theorem.

    Args:
        r1: Radius of the first circle.
        r2: Radius of the second circle.
        r3: Radius of the third circle.
        plus_minus: +1 for outer tangent circle, -1 for inner tangent circle.

    Returns:
        A tuple containing the radius and center coordinates (x, y) of the tangent circle,
        or None if no solution is found.
    """
    # Example usage:
    # r1, r2, r3 = 40, 40, 40
    # z1 = complex(69.28, 40)
    # z2 = complex(0, 80)
    # z3 = complex(69.82, 120)
    # r, center = apollonius(r1, r2, r3, z1, z2, z3, -1)

    k1, k2, k3 = 1/r1, 1/r2, 1/r3

    # Applying Descartes' Theorem
    k4_values = (k1 + k2 + k3) + plus_minus * 2 * sqrt(k1*k2 + k2*k3 + k3*k1)

    # Handle cases where no solution exists (e.g., division by zero)
    if k4_values == 0:
        return None

    r4 = 1/k4_values
    z4 = ((k1*z1 + k2*z2 + k3*z3 + plus_minus * 2 *
           cmath.sqrt(k1*k2*z1*z2 + k2*k3*z2*z3 + k3*k1*z3*z1)) / k4_values)

    return r4, z4



def circle_tangent_to_2_circles(c1, r1, c2, r2, r):
    """Given the centers and radii of 2 circles, return the center
    of a circle with radius r that is tangent to both circles."""

    # Simultaneous equations are solved with Sympy.
    # x, y, x1, y1, x2, y2, r1, r2, r = symbols('x y x1 y1 x2, y2 r1 r2 r')
    # eq1 = Eq(sqrt((x - x1)**2 + (y - y1)**2) -r -r1, 0)
    # eq2 = Eq(sqrt((x - x2)**2 + (y - y2)**2) -r -r2, 0)
    # res = solve([eq1, eq2], (x, y), dict=True)

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
                (-r12 + 2 * r1r2 - r22 + x12 - 2 * x1x2 + x22 + y12 - 2 * y1y2 + y22)
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
            (-r12 + 2 * r1r2 - r22 + x12 - 2 * x1x2 + x22 + y12 - 2 * y1y2 + y22)
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
                (-r12 + 2 * r1r2 - r22 + x12 - 2 * x1x2 + x22 + y12 - 2 * y1y2 + y22)
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
            (-r12 + 2 * r1r2 - r22 + x12 - 2 * x1x2 + x22 + y12 - 2 * y1y2 + y22)
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
    """Returns the tangent points (p1, p2, p3, p4) in world coordinates"""
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
    alpha = angle_between_lines((pos[0] + x, pos[1] + dr), pos, c2.center)
    tp1w, tp2w = rotate(points, alpha, pos)

    if x == 0:
        beta = 0
    else:
        beta = pi / 2 - atan(dr / x)
    tp3w = rotate([tp1w], -2 * beta, pos)[0]
    tp4w = rotate([tp2w], -2 * beta, c2.center)[0]

    return (tp1w, tp2w, tp3w, tp4w)


def circle_area(rad):
    """Given the radius of a circle, return the area of the circle."""

    return pi * rad**2


def circle_circumference(rad):
    """Given the radius of a circle, return the circumference of the circle."""

    return 2 * pi * rad


def flower_angle(r1, r2, r3):
    """Given the radii of 3 circles forming an interstice, return the angle between
    the lines connecting circles' centers to center of the circle with r1 radius."""

    angle = acos(
        ((r1 + r2) ** 2 + (r1 + r3) ** 2 - (r2 + r3) ** 2) / (2 * (r1 + r2) * (r1 + r3))
    )

    return angle



ratios = {8: 0.4974, 9: 0.5394, 10: 0.575, 11: 0.6056,
        12: 0.6321, 13: 0.6553, 14: 0.6757, 15: 0.6939, 16: 0.7101,
        17: 0.7248, 18: 0.738, 19: 0.75, 20: 0.7609}

def circle_flower(n, radius=25, layers=6, ratio = None):
    '''Steiner chain. Return a list of circles that form a flower-like pattern.'''
    if n<8:
        raise ValueError('n must be greater than 7')
    if ratio is None:
        if n<21:
            ratio = ratios[n]
        else:
            ratio = (-0.000000089767 * n**4 + 0.000015821834 * n**3 +
                    -0.001100867708 * n **2+ 0.038096046379 * n + 0.327363569038)

    r1 = side_len_to_radius(n, 2*radius)
    circles = Circle((r1, 0), radius).rotate(pi/(n/2), (0, 0), reps=n-1)
    xform = scale_matrix(ratio) @ rotation_matrix(pi/n)

    return circles.transform(xform_matrix=xform, reps=layers)

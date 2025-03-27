"""Functions for working with Bezier curves.
https://pomax.github.io/bezierinfo is a good resource for understanding Bezier curves.
"""

from typing import Sequence
from functools import lru_cache as memoize

import numpy as np

from ..graphics.shape import Shape
from ..graphics.all_enums import Types
from ..graphics.common import Point
from ..helpers.utilities import find_closest_value
from ..settings.settings import defaults
from .geometry import (
    distance,
    line_angle,
    line_by_point_angle_length,
    normal,
    norm,
    normalize,
)

array = np.array

cubic_poly_matrix = np.array(
    [[1, 0, 0, 0], [-3, 3, 0, 0], [3, -6, 3, 0], [-1, 3, -3, 1]]
)

quad_poly_matrix = np.array([[1, 0, 0], [-2, 2, 0], [1, -2, 1]])


class Bezier(Shape):
    """A Bezier curve defined by control points.

    For cubic Bezier curves: [V1, CP1, CP2, V2]
    For quadratic Bezier curves: [V1, CP, V2]
    Like all other geometry in simetri.graphics,
    bezier curves are represented as a sequence of points.
    Both quadratic and cubic bezier curves are supported.
    Number of control points determines the type of Bezier curve.
    A cubic Bezier curve has 4 control points, while a quadratic Bezier curve has 3.
    curve.subtype reflects this as Types.BEZIER or Types.Q_BEZIER.

    Attributes:
        control_points (Sequence[Point]): Control points of the Bezier curve.
        cubic (bool): True if the Bezier curve is cubic, False if quadratic.
        matrix (array): Polynomial matrix for the Bezier curve.
    """

    def __init__(
        self,
        control_points: Sequence[Point],
        xform_matrix: array = None,
        n_points=None,
        **kwargs,
    ) -> None:
        """Initializes a Bezier curve.

        Args:
            control_points (Sequence[Point]): Control points of the Bezier curve.
            xform_matrix (array, optional): Transformation matrix. Defaults to None.
            n_points (int, optional): Number of points on the curve. Defaults to None.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the number of control points is not 3 or 4.
        """
        if len(control_points) == 3:
            if n_points is None:
                n = defaults["n_bezier_points"]
            else:
                n = n_points
            vertices = q_bezier_points(*control_points, n)
            super().__init__(
                vertices, subtype=Types.Q_BEZIER, xform_matrix=xform_matrix, **kwargs
            )
            self.cubic = False
            self.matrix = quad_poly_matrix @ array(control_points)

        elif len(control_points) == 4:
            if n_points is None:
                n = defaults["n_bezier_points"]
            else:
                n = n_points
            vertices = bezier_points(*control_points, n)
            super().__init__(
                vertices, subtype=Types.BEZIER, xform_matrix=xform_matrix, **kwargs
            )
            self.cubic = True
            self.matrix = cubic_poly_matrix @ array(control_points)
        else:
            raise ValueError("Invalid number of control points.")
        self.control_points = control_points

    @property
    def control_points(self) -> Sequence[Point]:
        """Return the control points of the Bezier curve.

        Returns:
            Sequence[Point]: Control points of the Bezier curve.
        """
        return self.__dict__['control_points']

    @control_points.setter
    def control_points(self, new_control_points: Sequence[Point]) -> None:
        """Set new control points for the Bezier curve.

        Args:
            new_control_points (Sequence[Point]): New control points.

        Raises:
            ValueError: If the number of control points is not 3 or 4.
        """
        self.control_points = new_control_points
        if len(new_control_points) == 3:
            vertices = q_bezier_points(*new_control_points)
            self[:] = vertices
            self.subtype = Types.Q_BEZIER
        elif len(new_control_points) == 4:
            vertices = bezier_points(*new_control_points)
            self[:] = vertices
            self.subtype = Types.BEZIER
        else:
            raise ValueError("Invalid number of control points.")

    def copy(self) -> Shape:
        """Return a copy of the Bezier curve.

        Returns:
            Shape: Copy of the Bezier curve.
        """
        # to do: copy style and other attributes
        copy_ = Bezier(self.control_points, xform_matrix=self.xform_matrix, n_points=len(self.vertices))

        return copy_

    def point(self, t: float):
        """Return the point on the Bezier curve at t.

        Args:
            t (float): Parameter t, where 0 <= t <= 1.

        Returns:
            list: Point on the Bezier curve at t.
        """
        # if self.cubic:
        #     np.array([t**3, t**2, t, 1]) @ self.matrix
        # else:
        #     np.array([t**2, t, 1]) @ self.matrix

    def point2(self, t: float):
        """Return the point on the Bezier curve at t.

        Args:
            t (float): Parameter t, where 0 <= t <= 1.

        Returns:
            list: Point on the Bezier curve at t.
        """
        p0, p1, p2, p3 = self.control_points
        m = 1 - t
        m2 = m * m
        m3 = m2 * m
        t2 = t * t
        t3 = t2 * t
        if self.cubic:
            x = m3 * p0[0] + 3 * m2 * t * p1[0] + 3 * m * t2 * p2[0] + t3 * p3[0]
            y = m3 * p0[1] + 3 * m2 * t * p1[1] + 3 * m * t2 * p2[1] + t3 * p3[1]
        else:
            x = m2 * p0[0] + 2 * m * t * p1[0] + t2 * p2[0]
            y = m2 * p0[1] + 2 * m * t * p1[1] + t2 * p2[1]

        return [x, y]

    def derivative(self, t: float):
        """Return the derivative of the Bezier curve at t.

        Args:
            t (float): Parameter t, where 0 <= t <= 1.

        Returns:
            list: Derivative of the Bezier curve at t.
        """
        if self.cubic:
            return get_cubic_derivative(t, self.control_points)
        else:
            return get_quadratic_derivative(t, self.control_points)

    def normal(self, t: float):
        """Return the normal of the Bezier curve at t.

        Args:
            t (float): Parameter t, where 0 <= t <= 1.

        Returns:
            list: Normal of the Bezier curve at t.
        """
        d = self.derivative(t)
        q = np.sqrt(d[0] * d[0] + d[1] * d[1])
        return [-d[1] / q, d[0] / q]

    def tangent(self, t: float):
        """Draw a unit tangent vector at t.

        Args:
            t (float): Parameter t, where 0 <= t <= 1.

        Returns:
            list: Unit tangent vector at t.
        """
        d = self.derivative(t)
        m = np.sqrt(d[0] * d[0] + d[1] * d[1])
        d = [d[0] / m, d[1] / m]
        return d


def equidistant_points(p0: Point, p1: Point, p2: Point, p3: Point, n_points=10):
    """Return the points on a Bezier curve with equidistant spacing.

    Args:
        p0 (list): First control point.
        p1 (list): Second control point.
        p2 (list): Third control point.
        p3 (list): Fourth control point.
        n_points (int, optional): Number of points. Defaults to 10.

    Returns:
        tuple: Points on the Bezier curve, equidistant points, tangents, and normals.
    """
    controls = [p0, p1, p2, p3]
    n = 100
    points = bezier_points(p0, p1, p2, p3, n)
    tot = 0
    seg_lengths = [0]
    tangents = [norm((p1[0] - p0[0], p1[1] - p0[1]))]
    normals = [normal(p0, p1)]
    eq_points = [p0]
    for i in range(1, n):
        dist = distance(points[i - 1], points[i])
        tot += dist
        seg_lengths.append(tot)

    for i in range(1, n_points):
        _, ind = find_closest_value(seg_lengths, i * tot / n_points)
        pnt = points[ind]
        eq_points.append(pnt)
        d = get_cubic_derivative(ind / n, controls)
        d1 = normalize(d)
        p1 = pnt
        p2 = get_normal(d)
        tangents.append(d1)
        normals.append(p2)

    return points, eq_points, tangents, normals


def offset_points(controls, offset, n_points, double=False):
    """Return the points on the offset curve.

    Args:
        controls (list): Control points of the Bezier curve.
        offset (float): Offset distance.
        n_points (int): Number of points on the curve.
        double (bool, optional): If True, return double offset points. Defaults to False.

    Returns:
        list: Points on the offset curve.
    """
    n = 100
    points = bezier_points(*controls, n_points=n)
    tot = 0
    seg_lengths = [0]
    for i in range(1, n):
        dist = distance(points[i - 1], points[i])
        tot += dist
        seg_lengths.append(tot)

    x, y = controls[0][:2]
    np = normal(x, y)
    x2, y2 = (x + offset * np[0], y + offset * np[1])
    offset_pnts = [(x2, y2)]
    if double:
        p1, p2 = mirror_point((x, y), (x2, y2))
        offset_pnts2 = [p2]
    for i in range(1, n_points):
        _, ind = find_closest_value(seg_lengths, i * tot / n)
        pnt = points[ind]
        d = get_cubic_derivative(ind / n, controls)
        p1 = pnt
        p2 = get_normal(d)
        x2, y2 = p1[0] + offset * p2[0], p1[1] + offset * p2[1]
        offset_pnts.append((x2, y2))
        if double:
            _, p3 = mirror_point(pnt, (x2, y2))
            offset_pnts2.append(p3)

    if double:
        return offset_pnts, offset_pnts2
    else:
        return offset_pnts


class BezierPoints(Shape):
    """Points of a Bezier curve defined by the given control points.

    These points are spaced evenly along the curve (unlike parametric points).
    Normal and tangent unit vectors are also available at these points.

    Attributes:
        control_points (Sequence[Point]): Control points of the Bezier curve.
        param_points (list): Parametric points on the Bezier curve.
        tangents (list): Tangent vectors at the points.
        normals (list): Normal vectors at the points.
        n_points (int): Number of points on the curve.
    """

    def __init__(
        self, control_points: Sequence[Point], n_points: int = 10, **kwargs
    ) -> None:
        """Initializes Bezier points.

        Args:
            control_points (Sequence[Point]): Control points of the Bezier curve.
            n_points (int, optional): Number of points on the curve. Defaults to 10.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the number of control points is not 3 or 4.
        """
        if len(control_points) not in (4, 3):
            raise ValueError("Invalid number of control points.")

        param_points, vertices, tangents, normals = equidistant_points(
            *control_points, n_points
        )
        super().__init__(vertices, subtype=Types.BEZIER_POINTS, **kwargs)
        self.control_points = control_points
        self.param_points = param_points
        self.tangents = tangents
        self.normals = normals
        self.n_points = n_points

    def offsets(self, offset: float, double: bool=False):
        """Return the points on the offset curve.

        Args:
            offset (float): Offset distance.
            double (bool, optional): If True, return double offset points. Defaults to False.

        Returns:
            list: Points on the offset curve.
        """
        offset_points1 = []
        if double:
            offset_points2 = []
        for i, pnt in enumerate(self.vertices):
            n_p = self.normals[i]
            x2, y2 = pnt[0] + offset * n_p[0], pnt[1] + offset * n_p[1]
            offset_points1.append((x2, y2))
            if double:
                _, p3 = mirror_point((x2, y2), pnt)
                offset_points2.append(p3)

        if double:
            return offset_points1, offset_points2

        return offset_points1


M = array([[1, 0, 0, 0], [-3, 3, 0, 0], [3, -6, 3, 0], [-1, 3, -3, 1]])


def bezier_points(p0, p1: Point, p2: Point, p3: Point, n_points=10):
    """Return the points on a cubic Bezier curve.

    Args:
        p0 (list): First control point.
        p1 (list): Second control point.
        p2 (list): Third control point.
        p3 (list): Fourth control point.
        n_points (int, optional): Number of points. Defaults to 10.

    Returns:
        list: Points on the cubic Bezier curve.

    Raises:
        ValueError: If n_points is less than 5.
    """
    if n_points < 5:
        raise ValueError("n_points must be at least 5.")

    n = n_points
    f = np.ones(n)
    t = np.linspace(0, 1, n)
    t2 = t * t
    t3 = t2 * t
    T = np.column_stack((f, t, t2, t3))
    TM = T @ M
    P = array([p0, p1, p2, p3])

    return TM @ P


MQ = array([[1, 0, 0], [-2, 2, 0], [1, -2, 1]])


def q_bezier_points(p0: Point, p1: Point, p2: Point, n_points: int):
    """Return the points on a quadratic Bezier curve.

    Args:
        p0 (list): First control point.
        p1 (list): Second control point.
        p2 (list): Third control point.
        n_points (int): Number of points.

    Returns:
        list: Points on the quadratic Bezier curve.

    Raises:
        ValueError: If n_points is less than 5.
    """
    if n_points < 5:
        raise ValueError("n_points must be at least 5.")

    n = n_points
    f = np.ones(n)
    t = np.linspace(0, 1, n)
    t2 = t * t
    T = np.column_stack((f, t, t2))
    TMQ = T @ MQ
    P = array([p0, p1, p2])

    return TMQ @ P


def split_bezier(p0: Point, p1: Point, p2: Point, p3: Point, z:float, n_points=10):
    """Split a cubic Bezier curve at t=z.

    Args:
        p0 (list): First control point.
        p1 (list): Second control point.
        p2 (list): Third control point.
        p3 (list): Fourth control point.
        z (float): Parameter z, where 0 <= z <= 1.
        n_points (int, optional): Number of points. Defaults to 10.

    Returns:
        tuple: Two Bezier curves split at t=z.
    """
    p0 = array(p0)
    p1 = array(p1)
    p2 = array(p2)
    p3 = array(p3)
    bezier1 = [
        [p0],
        [z * p1 - (z - 1) * p0],
        [z**2 * p2 - 2 * z * (z - 1) * p1 + (z - 1) ** 2 * p0],
        [
            z**3 * p3
            - 3 * z**2 * (z - 1) * p2
            + 3 * z * (z - 1) ** 2 * p1
            - (z - 1) ** 3 * p0
        ],
    ]

    bezier2 = [
        [z**3 * p0],
        [3 * z**2 * (z - 1) * p1 - 3 * z * (z - 1) ** 2 * p0],
        [3 * z * (z - 1) * p2 - 3 * (z - 1) ** 2 * p1],
        [z * p3 - (z - 1) * p2],
    ]

    return Bezier(bezier1, n_points=n_points), Bezier(bezier2, n_points=n_points)


def split_q_bezier(p0: Point, p1: Point, p2: Point, z:float, n_points=10):
    """Split a quadratic Bezier curve at t=z.

    Args:
        p0 (list): First control point.
        p1 (list): Second control point.
        p2 (list): Third control point.
        z (float): Parameter z, where 0 <= z <= 1.
        n_points (int, optional): Number of points. Defaults to 10.

    Returns:
        tuple: Two Bezier curves split at t=z.
    """
    p0 = array(p0)
    p1 = array(p1)
    p2 = array(p2)
    bezier1 = [
        [p0],
        [z * p1 - (z - 1) * p0],
        [z**2 * p2 - 2 * z * (z - 1) * p1 + (z - 1) ** 2 * p0],
    ]

    bezier2 = [
        [z**2 * p0],
        [2 * z * (z - 1) * p1 - (z - 1) ** 2 * p0],
        [z * p2 - (z - 1) * p1],
    ]

    return Bezier(bezier1, n_points=n_points), Bezier(bezier2, n_points=n_points)


def mirror_point(cp: Point, vertex: Point):
    """Return the mirror of cp about vertex.

    Args:
        cp (list): Control point to be mirrored.
        vertex (list): Vertex point.

    Returns:
        list: Mirrored control point.
    """
    length = distance(cp, vertex)
    angle = line_angle(cp, vertex)
    cp2 = line_by_point_angle_length(vertex, angle, length)
    return cp2


def curve(v1: Point, c1: Point, c2: Point, v2: Point, *args, **kwargs):
    """Return a cubic Bezier curve/s.

    Args:
        v1 (list): First vertex.
        c1 (list): First control point.
        c2 (list): Second control point.
        v2 (list): Second vertex.
        *args: Additional control points and vertices.
        **kwargs: Additional keyword arguments.

    Returns:
        list: List of cubic Bezier curves.

    Raises:
        ValueError: If the number of control points is invalid.
    """
    curves = [Bezier([v1, c1, c2, v2], **kwargs)]
    last_vertex = v2
    for arg in args:
        if len(arg) == 2:
            c3 = mirror_point(c2, v2)
            v3 = arg[1]
            c4 = arg[0]
            curves.append(Bezier([last_vertex, c3, c4, v3], **kwargs))
            last_vertex = v3
        elif len(arg) == 3:
            c3 = arg[0]
            c4 = arg[1]
            v3 = arg[2]
            curves.append(Bezier([last_vertex, c3, c4, v3], **kwargs))
            last_vertex = v3
        else:
            raise ValueError("Invalid number of control points.")

    return curves


def q_curve(v1: Point, c: Point, v2: Point, *args, **kwargs):
    """Return a quadratic Bezier curve/s.

    Args:
        v1 (list): First vertex.
        c (list): Control point.
        v2 (list): Second vertex.
        *args: Additional control points and vertices.
        **kwargs: Additional keyword arguments.

    Returns:
        list: List of quadratic Bezier curves.

    Raises:
        ValueError: If the number of control points is invalid.
    """
    curves = [Bezier([v1, c, v2], **kwargs)]
    last_vertex = v2
    for arg in args:
        if len(arg) == 1:
            c3 = mirror_point(c, v2)
            v3 = arg[0]
            curves.append(Bezier([last_vertex, c3, v3], **kwargs))
            last_vertex = v3
        elif len(arg) == 2:
            c3 = arg[0]
            v3 = arg[1]
            curves.append(Bezier([last_vertex, c3, v3], **kwargs))
            last_vertex = v3
        else:
            raise ValueError("Invalid number of control points.")

    return curves


def get_quadratic_derivative(t: float, points: Sequence[Point]):
    """Return the derivative of a quadratic Bezier curve at t.

    Args:
        t (float): Parameter t, where 0 <= t <= 1.
        points (list): Control points of the Bezier curve.

    Returns:
        list: Derivative of the quadratic Bezier curve at t.
    """
    mt = 1 - t
    d = [
        2 * (points[1][0] - points[0][0]),
        2 * (points[1][1] - points[0][1]),
        2 * (points[2][0] - points[1][0]),
        2 * (points[2][1] - points[1][1]),
    ]

    return [mt * d[0] + t * d[2], mt * d[1] + t * d[3]]


def get_cubic_derivative(t: float, points: Sequence[Point]):
    """Return the derivative of a cubic Bezier curve at t.

    Args:
        t (float): Parameter t, where 0 <= t <= 1.
        points (list): Control points of the Bezier curve.

    Returns:
        list: Derivative of the cubic Bezier curve at t.
    """
    mt = 1 - t
    a = mt * mt
    b = 2 * mt * t
    c = t * t
    d = [
        3 * (points[1][0] - points[0][0]),
        3 * (points[1][1] - points[0][1]),
        3 * (points[2][0] - points[1][0]),
        3 * (points[2][1] - points[1][1]),
        3 * (points[3][0] - points[2][0]),
        3 * (points[3][1] - points[2][1]),
    ]

    return [a * d[0] + b * d[2] + c * d[4], a * d[1] + b * d[3] + c * d[5]]


def get_normal(d: Sequence[float]):
    """Return the normal of a given line.

    Args:
        d (list): Derivative of the line.

    Returns:
        list: Normal of the line.
    """
    q = np.sqrt(d[0] * d[0] + d[1] * d[1])
    return [-d[1] / q, d[0] / q]

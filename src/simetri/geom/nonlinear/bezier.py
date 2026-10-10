"""Functions for working with Bezier curves.
https://pomax.github.io/bezierinfo is a good resource for understanding Bezier curves.
Many of the functions and methods in this module is based on this resource.
"""

from __future__ import annotations

from collections.abc import Sequence

import math

import numpy as np
from numpy import array
from numpy.typing import NDArray

from ...base.all_enums import Types
from ...base.common import PointType
from ...config.settings import runtime_defaults
from ...helpers.utilities import find_closest_value
from ...shapes.shape import Shape
from ..points.point_utils import distance
from ..segments.line_utils import line_angle, line_by_point_angle_length
from ..vectors import norm, normal, normalize


class Bezier(Shape):
    """A Bezier curve defined by control points.

    For cubic Bezier curves: ``[V1, CP1, CP2, V2]``.
    For quadratic Bezier curves: ``[V1, CP, V2]``.
    Like other geometry in ``simetri.graphics``, Bezier curves are represented
    as a sequence of sampled points. The number of control points selects the
    type: 4 for cubic (``Types.BEZIER``) or 3 for quadratic (``Types.Q_BEZIER``).

    Attributes:
        control_points (Sequence[PointType]): Control points of the Bezier curve.
        cubic (bool): True if cubic, False if quadratic.
        matrix (array): Polynomial matrix for the Bezier curve.

    Examples:

        >>> import simetri.graphics as sg
        >>> curve = sg.Bezier([(0, 0), (20, 40), (60, 40), (80, 0)], n_points=5)
        >>> [[round(float(c), 6) for c in q[:2]] for q in curve.vertices]
        [[0.0, 0.0], [18.125, 22.5], [40.0, 30.0], [61.875, 22.5], [80.0, 0.0]]
    """

    def __init__(
        self,
        control_points: Sequence[PointType],
        xform_matrix: array = None,
        n_points: int | None = None,
        **kwargs: object,
    ) -> None:
        """Initializes a Bezier curve.

        Args:
            control_points (Sequence[PointType]): Control points of the Bezier curve.
            xform_matrix (array, optional): Transformation matrix. Defaults to None.
            n_points (int, optional): Number of points on the curve. Defaults to None.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the number of control points is not 3 or 4.
        """
        if len(control_points) == 3:
            if n_points is None:
                n = runtime_defaults["n_bezier_points"]
            else:
                n = n_points
            vertices = q_bezier_points(*control_points, n)
            super().__init__(
                vertices,
                subtype=Types.Q_BEZIER,
                xform_matrix=xform_matrix,
                **kwargs,
            )
            self.cubic = False
            quad_poly_matrix = array([[1, 0, 0], [-2, 2, 0], [1, -2, 1]])
            self.matrix = quad_poly_matrix @ array(control_points)

        elif len(control_points) == 4:
            if n_points is None:
                n = runtime_defaults["n_bezier_points"]
            else:
                n = n_points
            vertices = bezier_points(*control_points, n)
            super().__init__(
                vertices,
                subtype=Types.BEZIER,
                xform_matrix=xform_matrix,
                **kwargs,
            )
            self.cubic = True
            cubic_poly_matrix = array(
                [[1, 0, 0, 0], [-3, 3, 0, 0], [3, -6, 3, 0], [-1, 3, -3, 1]]
            )
            self.matrix = cubic_poly_matrix @ array(control_points)
        else:
            raise ValueError("Invalid number of control points.")
        self.__dict__["control_points"] = control_points

    def __repr__(self) -> str:
        """Return a Bezier or Q_BEZIER string from this curve's vertices.

        Subtype ``BEZIER`` prints ``Bezier``. Subtype ``Q_BEZIER`` prints
        the enum token ``Q_BEZIER``.

        Examples:
            >>> import simetri.graphics as sg
            >>> cubic = sg.Bezier(
            ...     [(0, 0), (20, 40), (60, 40), (80, 0)], n_points=5
            ... )
            >>> repr(cubic)
            'Bezier([(0.0, 0.0), ..., (80.0, 0.0)])'
            >>> quad = sg.Bezier([(0, 0), (20, 40), (80, 0)], n_points=5)
            >>> repr(quad)
            'Q_BEZIER([(0.0, 0.0), ..., (80.0, 0.0)])'
            >>> str(cubic).startswith("Shape")
            True
        """
        if self.subtype == Types.Q_BEZIER:
            name = "Q_BEZIER"
        else:
            name = "Bezier"
        if len(self.primary_points) == 0:
            return f"{name}()"
        if len(self.primary_points) < 4:
            return f"{name}({self.vertices})"
        return f"{name}([{self.vertices[0]}, ..., {self.vertices[-1]}])"

    @property
    def control_points(self) -> Sequence[PointType]:
        """Return the control points of the Bezier curve.

        Returns:
            Sequence[PointType]: Control points of the Bezier curve.
        """
        return self.__dict__["control_points"]

    @control_points.setter
    def control_points(self, new_control_points: Sequence[PointType]) -> None:
        """Set new control points for the Bezier curve.

        Args:
            new_control_points (Sequence[PointType]): New control points.

        Raises:
            ValueError: If the number of control points is not 3 or 4.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (20, 40), (60, 40), (80, 0)], n_points=5)
            >>> curve.control_points
            [(0, 0), (20, 40), (60, 40), (80, 0)]
            >>> curve.control_points = [(0, 0), (0, 40), (80, 40), (80, 0)]
            >>> curve.control_points
            [(0, 0), (0, 40), (80, 40), (80, 0)]
        """
        self.__dict__["control_points"] = new_control_points
        n_points = runtime_defaults["n_bezier_points"]
        if len(new_control_points) == 3:
            vertices = q_bezier_points(*new_control_points, n_points)
            self[:] = vertices
            self.subtype = Types.Q_BEZIER
        elif len(new_control_points) == 4:
            vertices = bezier_points(*new_control_points, n_points)
            self[:] = vertices
            self.subtype = Types.BEZIER
        else:
            raise ValueError("Invalid number of control points.")

    def copy(self, **kwargs: object) -> Shape:
        """Return a copy of the Bezier curve.

        Returns:
            Shape: Copy of the Bezier curve.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (20, 40), (60, 40), (80, 0)], n_points=5)
            >>> copy = curve.copy()
            >>> copy.control_points == curve.control_points
            True
            >>> copy.vertices == curve.vertices
            True
        """
        # to do: copy style and other attributes
        copy_ = Bezier(
            self.control_points,
            xform_matrix=self.xform_matrix,
            n_points=len(self.vertices),
        )
        for k, v in kwargs.items():
            setattr(copy_, k, v)

        return copy_

    def point(self, t: float) -> list[float]:
        """Return the point on the Bezier curve at t.

        Args:
            t (float): Parameter t, where 0 <= t <= 1.

        Returns:
            list: PointType on the Bezier curve at t.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (20, 40), (60, 40), (80, 0)], n_points=5)
            >>> [round(c, 6) for c in curve.point(0.0)]
            [0.0, 0.0]
            >>> [round(c, 6) for c in curve.point(0.5)]
            [40.0, 30.0]
            >>> [round(c, 6) for c in curve.point(1.0)]
            [80.0, 0.0]
        """
        if self.cubic:
            p0, p1, p2, p3 = self.control_points
            m = 1 - t
            m2 = m * m
            m3 = m2 * m
            t2 = t * t
            t3 = t2 * t
            x = (
                m3 * p0[0]
                + 3 * m2 * t * p1[0]
                + 3 * m * t2 * p2[0]
                + t3 * p3[0]
            )
            y = (
                m3 * p0[1]
                + 3 * m2 * t * p1[1]
                + 3 * m * t2 * p2[1]
                + t3 * p3[1]
            )
        else:
            p0, p1, p2 = self.control_points
            m = 1 - t
            m2 = m * m
            t2 = t * t
            x = m2 * p0[0] + 2 * m * t * p1[0] + t2 * p2[0]
            y = m2 * p0[1] + 2 * m * t * p1[1] + t2 * p2[1]

        return [x, y]

    def derivative(self, t: float) -> list[float]:
        """Return the derivative of the Bezier curve at t.

        Args:
            t (float): Parameter t, where 0 <= t <= 1.

        Returns:
            list: Derivative of the Bezier curve at t.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (20, 40), (60, 40), (80, 0)], n_points=5)
            >>> [round(c, 6) for c in curve.derivative(0.0)]
            [60.0, 120.0]
        """
        if self.cubic:
            return get_cubic_derivative(t, self.control_points)
        else:
            return get_quadratic_derivative(t, self.control_points)

    def normal(self, t: float) -> list[float]:
        """Return the normal of the Bezier curve at t.

        Args:
            t (float): Parameter t, where 0 <= t <= 1.

        Returns:
            list: Normal of the Bezier curve at t.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (20, 40), (60, 40), (80, 0)], n_points=5)
            >>> [round(c, 6) for c in curve.normal(0.0)]
            [-0.894427, 0.447214]
        """
        d = self.derivative(t)
        q = np.sqrt(d[0] * d[0] + d[1] * d[1])
        return [-d[1] / q, d[0] / q]

    def tangent(self, t: float) -> list[float]:
        """Draw a unit tangent vector at t.

        Args:
            t (float): Parameter t, where 0 <= t <= 1.

        Returns:
            list: Unit tangent vector at t.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (20, 40), (60, 40), (80, 0)], n_points=5)
            >>> [round(c, 6) for c in curve.tangent(0.0)]
            [0.447214, 0.894427]
        """
        d = self.derivative(t)
        m = np.sqrt(d[0] * d[0] + d[1] * d[1])
        d = [d[0] / m, d[1] / m]
        return d

    def second_derivative(self, t: float) -> list[float]:
        """Return the second derivative at ``t``.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (1, 2), (2, 0)], n_points=5)
            >>> [round(c, 6) for c in curve.second_derivative(0.5)]
            [0.0, -8.0]
        """
        return get_second_derivative(t, self.control_points)

    def curvature(self, t: float) -> float:
        """Return the signed curvature at ``t``.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (1, 2), (2, 0)], n_points=5)
            >>> round(curve.curvature(0.5), 6)
            -2.0
        """
        return get_curvature(t, self.control_points)

    def derivative_roots(self) -> tuple[list[float], list[float]]:
        """Return ``t`` values where ``x'`` or ``y'`` is zero, in ``[0, 1]``.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (0, 60), (80, 60), (80, 0)], n_points=5)
            >>> xs, ys = curve.derivative_roots()
            >>> [round(t, 6) for t in ys]
            [0.5]
        """
        return get_derivative_roots(self.control_points)

    def exact_bbox(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the exact axis-aligned box as southwest and northeast.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (0, 60), (80, 60), (80, 0)], n_points=5)
            >>> sw, ne = curve.exact_bbox()
            >>> ([round(c, 6) for c in sw], [round(c, 6) for c in ne])
            ([0.0, 0.0], [80.0, 45.0])
        """
        return get_exact_bbox(self.control_points)

    def aligned(
        self,
    ) -> tuple[Bezier, tuple[float, float], float]:
        """Return this curve moved onto the x-axis, plus origin and angle.

        The returned curve starts at the origin and ends on the positive
        x-axis. ``origin`` and ``angle`` map that curve back.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (0, 20), (0, 40)], n_points=5)
            >>> aligned, origin, angle = curve.aligned()
            >>> [round(c, 6) for c in origin]
            [0.0, 0.0]
            >>> round(angle, 6)
            1.570796
            >>> [round(c, 6) for c in aligned.control_points[-1][:2]]
            [40.0, 0.0]
        """
        controls, origin, angle = get_aligned_controls(self.control_points)
        return Bezier(controls, n_points=len(self.vertices)), origin, angle

    def tight_bbox(self) -> list[tuple[float, float]]:
        """Return the four corners of the aligned bounding box.

        Corners are southwest, southeast, northeast, northwest of the
        aligned box, mapped back to this curve.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (0, 60), (80, 60), (80, 0)], n_points=5)
            >>> [[round(c, 6) for c in p] for p in curve.tight_bbox()]
            [[0.0, 0.0], [80.0, 0.0], [80.0, 45.0], [0.0, 45.0]]
        """
        return get_tight_bbox(self.control_points)

    def inflections(self) -> list[float]:
        """Return ``t`` values where the cubic curvature changes sign.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (0, 1), (1, 0), (1, 1)], n_points=5)
            >>> [round(t, 6) for t in curve.inflections()]
            [0.5]
        """
        return get_inflections(self.control_points)

    def arc_length(self, t: float = 1.0) -> float:
        """Return the arc length from the start to ``t``.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (10, 0), (20, 0), (30, 0)], n_points=5)
            >>> curve.arc_length()
            30.0
        """
        return get_arc_length(self.control_points, t)

    def t_at_length(self, length: float) -> float:
        """Return ``t`` at ``length`` along the curve.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (10, 0), (20, 0), (30, 0)], n_points=5)
            >>> round(curve.t_at_length(10), 6)
            0.333333
        """
        return get_t_at_arc_length(self.control_points, length)

    def y_at_x(self, x: float) -> list[float]:
        """Return every ``y`` on the curve at coordinate ``x``.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (0, 60), (80, 60), (80, 0)], n_points=5)
            >>> [round(y, 6) for y in curve.y_at_x(40)]
            [45.0]
        """
        return get_y_at_x(self.control_points, x)

    def intersect_line(
        self,
        start: PointType,
        end: PointType,
        segment: bool = False,
    ) -> list[tuple[float, tuple[float, float]]]:
        """Return intersections with the line through ``start`` and ``end``.

        ``segment=True`` keeps hits that lie on the segment.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (0, 60), (80, 60), (80, 0)], n_points=5)
            >>> hits = curve.intersect_line((0, 45), (80, 45))
            >>> [(round(t, 6), [round(c, 6) for c in p]) for t, p in hits]
            [(0.5, [40.0, 45.0])]
        """
        return get_line_intersections(
            self.control_points, start, end, segment=segment
        )

    def elevate(self) -> Bezier:
        """Return this quadratic as an exact cubic.

        Examples:
            >>> import simetri.graphics as sg
            >>> raised = sg.Bezier([(0, 0), (30, 60), (60, 0)], n_points=5).elevate()
            >>> [[round(c, 6) for c in p[:2]] for p in raised.control_points]
            [[0.0, 0.0], [20.0, 40.0], [40.0, 40.0], [60.0, 0.0]]
        """
        return Bezier(
            elevate_quadratic(self.control_points),
            n_points=len(self.vertices),
        )

    def reduce(self) -> Bezier:
        """Return a quadratic least-squares fit of this cubic.

        Examples:
            >>> import simetri.graphics as sg
            >>> cubic = sg.Bezier(
            ...     [(0, 0), (20, 40), (40, 40), (60, 0)], n_points=5
            ... )
            >>> [[round(c, 6) for c in p[:2]] for p in cubic.reduce().control_points]
            [[0.0, 0.0], [30.0, 60.0], [60.0, 0.0]]
        """
        return Bezier(
            reduce_cubic(self.control_points),
            n_points=len(self.vertices),
        )

    def arcs(self, tolerance: float = 0.01) -> list[BezierArc]:
        """Return circular arcs that follow this curve within ``tolerance``.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (10, 0), (20, 0), (30, 0)], n_points=5)
            >>> curve.arcs()[0].straight
            True
        """
        return bezier_to_arcs(self.control_points, tolerance)

    def flatten(
        self, n_points: int | None = None
    ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """Return straight segments that follow this curve.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (40, 80), (80, 80), (60, 0)], n_points=5)
            >>> len(curve.flatten(n_points=5))
            4
        """
        return flatten_curve(self.control_points, n_points)

    def closest_point(
        self, point: PointType, tolerance: float = 0.01
    ) -> tuple[tuple[float, float], float]:
        """Return the closest point on the arc fit and its distance.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (10, 0), (20, 0), (30, 0)], n_points=5)
            >>> point, distance = curve.closest_point((15, 5))
            >>> [round(c, 6) for c in point]
            [15.0, 0.0]
            >>> round(distance, 6)
            5.0
        """
        return bezier_closest_point(self.control_points, point, tolerance)

    def intersect_circle(
        self,
        center: PointType,
        radius: float,
        tolerance: float = 0.01,
    ) -> list[tuple[float, float]]:
        """Return intersections with a circle, using the arc fit.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (10, 0), (20, 0), (30, 0)], n_points=5)
            >>> hits = curve.intersect_circle((15, 0), 5)
            >>> [[round(c, 6) for c in hit] for hit in hits]
            [[10.0, 0.0], [20.0, 0.0]]
        """
        return bezier_circle_intersections(
            self.control_points, center, radius, tolerance
        )

    def intersect_curve(
        self, other: Bezier, tolerance: float = 0.01
    ) -> list[tuple[float, float]]:
        """Return intersections with another Bézier, using both arc fits.

        Examples:
            >>> import simetri.graphics as sg
            >>> horizontal = sg.Bezier([(0, 0), (10, 0), (30, 0), (40, 0)], n_points=5)
            >>> vertical = sg.Bezier([(20, -20), (20, -10), (20, 10), (20, 20)], n_points=5)
            >>> hits = horizontal.intersect_curve(vertical)
            >>> [[round(c, 6) for c in hit] for hit in hits]
            [[20.0, 0.0]]
        """
        return bezier_curve_intersections(
            self.control_points, other.control_points, tolerance
        )

    def self_intersections(
        self, tolerance: float = 0.01
    ) -> list[tuple[float, float]]:
        """Return interior points where this curve crosses itself.

        Examples:
            >>> import simetri.graphics as sg
            >>> curve = sg.Bezier([(0, 0), (80, 80), (-20, 80), (60, 0)], n_points=5)
            >>> hits = curve.self_intersections()
            >>> [[round(c, 2) for c in hit] for hit in hits]
            [[30.0, 40.0]]
        """
        return bezier_self_intersections(self.control_points, tolerance)


def equidistant_points(
    p0: PointType,
    p1: PointType,
    p2: PointType,
    p3: PointType,
    n_points: int = 10,
) -> tuple[NDArray, list[PointType], list, list]:
    """Return the points on a Bezier curve with equidistant spacing.

    Args:
        p0 (list): First control point.
        p1 (list): Second control point.
        p2 (list): Third control point.
        p3 (list): Fourth control point.
        n_points (int, optional): Number of points. Defaults to 10.

    Returns:
        tuple: Points on the Bezier curve, equidistant points, tangents, and normals.

    Examples:
        >>> import simetri.graphics as sg
        >>> points, eq_points, tangents, normals = sg.equidistant_points(
        ...     (0, 0), (40, 80), (60, 80), (80, 0), n_points=4
        ... )
        >>> len(eq_points)
        4
        >>> [round(c, 6) for c in eq_points[0][:2]]
        [0, 0]
        >>> [round(float(c), 6) for c in eq_points[-1][:2]]
        [68.970699, 35.702479]
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


def offset_points(
    controls: Sequence[PointType],
    offset: float,
    n_points: int,
    double: bool = False,
) -> list[PointType] | tuple[list[PointType], list[PointType]]:
    """Return the points on the offset curve.

    Args:
        controls (list): Control points of the Bezier curve.
        offset (float): Offset distance.
        n_points (int): Number of points on the curve.
        double (bool, optional): If True, return double offset points. Defaults to False.

    Returns:
        list: Points on the offset curve.

    Examples:
        >>> import simetri.graphics as sg
        >>> pts = sg.offset_points([(0, 0), (40, 80), (60, 80), (80, 0)], 1.0, 4)
        >>> [[round(float(c), 6) for c in q[:2]] for q in pts]
        [[-0.894427, 0.447214], [0.313431, 2.850626], [0.313431, 2.850626], [1.509272, 5.20523]]
    """
    n = 100
    points = bezier_points(*controls, n_points=n)
    tot = 0
    seg_lengths = [0]
    for i in range(1, n):
        dist = distance(points[i - 1], points[i])
        tot += dist
        seg_lengths.append(tot)

    unit_normal = normal(controls[0], controls[1])
    x, y = controls[0][:2]
    x2, y2 = (x + offset * unit_normal[0], y + offset * unit_normal[1])
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
        control_points (Sequence[PointType]): Control points of the Bezier curve.
        param_points (list): Parametric points on the Bezier curve.
        tangents (list): Tangent vectors at the points.
        normals (list): Normal vectors at the points.
        n_points (int): Number of points on the curve.

    Examples:
        >>> import simetri.graphics as sg
        >>> pts = sg.BezierPoints([(0, 0), (20, 40), (60, 40), (80, 0)], n_points=4)
        >>> [[round(float(c), 6) for c in q[:2]] for q in pts.vertices]
        [[0.0, 0.0], [15.857339, 20.740741], [39.54546, 29.996939], [64.142661, 20.740741]]
    """

    def __init__(
        self,
        control_points: Sequence[PointType],
        n_points: int = 10,
        **kwargs: object,
    ) -> None:
        """Initializes Bezier points.

        Args:
            control_points (Sequence[PointType]): Control points of the Bezier curve.
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
        super().__init__(vertices, **kwargs)
        self.control_points = control_points
        self.param_points = param_points
        self.tangents = tangents
        self.normals = normals
        self.n_points = n_points

    def offsets(
        self, offset: float, double: bool = False
    ) -> list[PointType] | tuple[list[PointType], list[PointType]]:
        """Return the points on the offset curve.

        Args:
            offset (float): Offset distance.
            double (bool, optional): If True, return double offset points. Defaults to False.

        Returns:
            list: Points on the offset curve.

        Examples:
            >>> import simetri.graphics as sg
            >>> pts = sg.BezierPoints([(0, 0), (20, 40), (60, 40), (80, 0)], n_points=4)
            >>> [[round(float(c), 6) for c in q[:2]] for q in pts.offsets(5)]
            [[-4.472136, 2.236068], [12.655292, 24.58091], [39.412156, 34.995162], [67.260219, 24.649812]]
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


def bezier_points(
    p0: PointType,
    p1: PointType,
    p2: PointType,
    p3: PointType,
    n_points: int = 10,
) -> NDArray:
    """Return sampled points on a cubic Bezier curve.

    Args:
        p0: First control point (start).
        p1: Second control point.
        p2: Third control point.
        p3: Fourth control point (end).
        n_points: Number of samples (must be >= 5). Defaults to 10.

    Returns:
        ndarray: Sampled points along the cubic Bezier.

    Raises:
        ValueError: If ``n_points`` is less than 5.

    Examples:

        >>> import simetri.graphics as sg
        >>> pts = sg.bezier_points((0, 0), (40, 80), (80, 80), (60, 0), n_points=5)
        >>> [[round(float(c), 6) for c in q[:2]] for q in pts]
        [[0.0, 0.0], [29.0625, 45.0], [52.5, 60.0], [64.6875, 45.0], [60.0, 0.0]]
    """
    if n_points < 5:
        raise ValueError("n_points must be at least 5.")

    n = n_points
    f = np.ones(n)
    t = np.linspace(0, 1, n)
    t2 = t * t
    t3 = t2 * t
    M = array([[1, 0, 0, 0], [-3, 3, 0, 0], [3, -6, 3, 0], [-1, 3, -3, 1]])
    T = np.column_stack((f, t, t2, t3))
    TM = T @ M
    P = array([p0, p1, p2, p3])

    return TM @ P


def q_bezier_points(
    p0: PointType, p1: PointType, p2: PointType, n_points: int
) -> NDArray:
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

    Examples:
        >>> import simetri.graphics as sg
        >>> pts = sg.q_bezier_points((0, 0), (40, 40), (80, 0), 5)
        >>> [[round(float(c), 6) for c in q[:2]] for q in pts]
        [[0.0, 0.0], [20.0, 15.0], [40.0, 20.0], [60.0, 15.0], [80.0, 0.0]]
    """
    if n_points < 5:
        raise ValueError("n_points must be at least 5.")

    n = n_points
    f = np.ones(n)
    t = np.linspace(0, 1, n)
    t2 = t * t
    MQ = array([[1, 0, 0], [-2, 2, 0], [1, -2, 1]])
    T = np.column_stack((f, t, t2))
    TMQ = T @ MQ
    P = array([p0, p1, p2])

    return TMQ @ P


def split_bezier(
    p0: PointType,
    p1: PointType,
    p2: PointType,
    p3: PointType,
    z: float,
    n_points: int = 10,
) -> tuple[Bezier, Bezier]:
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

    Examples:
        >>> import simetri.graphics as sg
        >>> left, right = sg.split_bezier((0, 0), (40, 80), (60, 80), (80, 0), 0.5, n_points=5)
        >>> [round(float(c), 6) for c in left.point(0.0)]
        [0.0, 0.0]
        >>> [round(float(c), 6) for c in left.point(1.0)]
        [47.5, 60.0]
        >>> [round(float(c), 6) for c in right.point(0.0)]
        [47.5, 60.0]
        >>> [round(float(c), 6) for c in right.point(1.0)]
        [80.0, 0.0]
    """
    p0 = array(p0, dtype=float)
    p1 = array(p1, dtype=float)
    p2 = array(p2, dtype=float)
    p3 = array(p3, dtype=float)
    one_minus_z = 1.0 - z
    p01 = one_minus_z * p0 + z * p1
    p12 = one_minus_z * p1 + z * p2
    p23 = one_minus_z * p2 + z * p3
    p012 = one_minus_z * p01 + z * p12
    p123 = one_minus_z * p12 + z * p23
    p0123 = one_minus_z * p012 + z * p123

    def as_xy(value: NDArray) -> tuple[float, float]:
        return (float(value[0]), float(value[1]))

    bezier1 = [as_xy(p0), as_xy(p01), as_xy(p012), as_xy(p0123)]
    bezier2 = [as_xy(p0123), as_xy(p123), as_xy(p23), as_xy(p3)]
    return Bezier(bezier1, n_points=n_points), Bezier(
        bezier2, n_points=n_points
    )


def split_q_bezier(
    p0: PointType, p1: PointType, p2: PointType, z: float, n_points: int = 10
) -> tuple[Bezier, Bezier]:
    """Split a quadratic Bezier curve at t=z.

    Args:
        p0 (list): First control point.
        p1 (list): Second control point.
        p2 (list): Third control point.
        z (float): Parameter z, where 0 <= z <= 1.
        n_points (int, optional): Number of points. Defaults to 10.

    Returns:
        tuple: Two Bezier curves split at t=z.

    Examples:
        >>> import simetri.graphics as sg
        >>> left, right = sg.split_q_bezier((0, 0), (40, 40), (80, 0), 0.5, n_points=5)
        >>> [round(float(c), 6) for c in left.point(0.0)]
        [0.0, 0.0]
        >>> [round(float(c), 6) for c in left.point(1.0)]
        [40.0, 20.0]
        >>> [round(float(c), 6) for c in right.point(0.0)]
        [40.0, 20.0]
        >>> [round(float(c), 6) for c in right.point(1.0)]
        [80.0, 0.0]
    """
    p0 = array(p0, dtype=float)
    p1 = array(p1, dtype=float)
    p2 = array(p2, dtype=float)
    one_minus_z = 1.0 - z
    p01 = one_minus_z * p0 + z * p1
    p12 = one_minus_z * p1 + z * p2
    p012 = one_minus_z * p01 + z * p12

    def as_xy(value: NDArray) -> tuple[float, float]:
        return (float(value[0]), float(value[1]))

    bezier1 = [as_xy(p0), as_xy(p01), as_xy(p012)]
    bezier2 = [as_xy(p012), as_xy(p12), as_xy(p2)]
    return Bezier(bezier1, n_points=n_points), Bezier(
        bezier2, n_points=n_points
    )


def mirror_point(cp: PointType, vertex: PointType) -> PointType:
    """Return the mirror of cp about vertex.

    Args:
        cp (list): Control point to be mirrored.
        vertex (list): Vertex point.

    Returns:
        list: Mirrored control point.

    Examples:
        >>> import simetri.graphics as sg
        >>> end = sg.mirror_point((80, 0), (40, 0))[-1]
        >>> tuple(round(c, 10) for c in end[:2])
        (0.0, 0.0)
    """
    length = distance(cp, vertex)
    angle = line_angle(cp, vertex)
    cp2 = line_by_point_angle_length(vertex, angle, length)
    return cp2


def curve(
    v1: PointType,
    c1: PointType,
    c2: PointType,
    v2: PointType,
    *args: object,
    **kwargs: object,
) -> list[Bezier]:
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

    Examples:
        >>> import simetri.graphics as sg
        >>> curves = sg.curve((0, 0), (40, 80), (60, 80), (80, 0))
        >>> len(curves)
        1
        >>> [round(float(c), 6) for c in curves[0].point(0.0)]
        [0.0, 0.0]
        >>> [round(float(c), 6) for c in curves[0].point(1.0)]
        [80.0, 0.0]
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


def q_curve(
    v1: PointType,
    c: PointType,
    v2: PointType,
    *args: object,
    **kwargs: object,
) -> list[Bezier]:
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

    Examples:
        >>> import simetri.graphics as sg
        >>> curves = sg.q_curve((0, 0), (40, 40), (80, 0))
        >>> len(curves)
        1
        >>> [round(float(c), 6) for c in curves[0].point(0.0)]
        [0.0, 0.0]
        >>> [round(float(c), 6) for c in curves[0].point(1.0)]
        [80.0, 0.0]
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


def get_quadratic_derivative(
    t: float, points: Sequence[PointType]
) -> list[float]:
    """Return the derivative of a quadratic Bezier curve at t.

    Args:
        t (float): Parameter t, where 0 <= t <= 1.
        points (list): Control points of the Bezier curve.

    Returns:
        list: Derivative of the quadratic Bezier curve at t.

    Examples:
        >>> import simetri.graphics as sg
        >>> [round(c, 6) for c in sg.get_quadratic_derivative(0.0, [(0, 0), (40, 40), (80, 0)])]
        [80.0, 80.0]
    """
    mt = 1 - t
    d = [
        2 * (points[1][0] - points[0][0]),
        2 * (points[1][1] - points[0][1]),
        2 * (points[2][0] - points[1][0]),
        2 * (points[2][1] - points[1][1]),
    ]

    return [mt * d[0] + t * d[2], mt * d[1] + t * d[3]]


def get_cubic_derivative(t: float, points: Sequence[PointType]) -> list[float]:
    """Return the derivative of a cubic Bezier curve at t.

    Args:
        t (float): Parameter t, where 0 <= t <= 1.
        points (list): Control points of the Bezier curve.

    Returns:
        list: Derivative of the cubic Bezier curve at t.

    Examples:
        >>> import simetri.graphics as sg
        >>> [round(c, 6) for c in sg.get_cubic_derivative(0.0, [(0, 0), (40, 80), (60, 80), (80, 0)])]
        [120.0, 240.0]
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


def get_normal(d: Sequence[float]) -> list[float]:
    """Return the normal of a given line.

    Args:
        d (list): Derivative of the line.

    Returns:
        list: Normal of the line.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.get_normal([0, 40])
        [-1.0, 0.0]
    """
    q = np.sqrt(d[0] * d[0] + d[1] * d[1])
    return [-d[1] / q, d[0] / q]


_ROOT_EPS = 1e-10
_ARC_SPLIT_LIMIT = 10
_GAUSS_NODES = (
    0.0,
    0.4058451513773972,
    0.7415311855993945,
    0.9491079123427585,
)
_GAUSS_WEIGHTS = (
    0.4179591836734694,
    0.3818300505051189,
    0.27970539148927664,
    0.1294849661688697,
)


def _xy(point: PointType) -> tuple[float, float]:
    """Return the first two coordinates of ``point``."""
    return float(point[0]), float(point[1])


def _control_array(points: Sequence[PointType]) -> list[tuple[float, float]]:
    """Return control points as ``(x, y)`` pairs."""
    if len(points) not in (3, 4):
        raise ValueError("Invalid number of control points.")
    return [_xy(point) for point in points]


def _cbrt(value: float) -> float:
    """Return the real cube root of ``value``."""
    if value < 0:
        return -((-value) ** (1.0 / 3.0))
    return value ** (1.0 / 3.0)


def _unique_unit_roots(roots: Sequence[float]) -> list[float]:
    """Return sorted roots that fall in ``[0, 1]``, without duplicates."""
    accepted = []
    for root in sorted(roots):
        if root < -_ROOT_EPS or root > 1.0 + _ROOT_EPS:
            continue
        if root <= 0.0:
            root = 0.0
        elif root >= 1.0:
            root = 1.0
        if accepted and abs(root - accepted[-1]) <= 1e-8:
            continue
        accepted.append(root)
    return accepted


def _quadratic_roots(a: float, b: float, c: float) -> list[float]:
    """Return the real roots of ``a t^2 + b t + c``."""
    scale = max(1.0, abs(a), abs(b), abs(c))
    if abs(a) <= _ROOT_EPS * scale:
        if abs(b) <= _ROOT_EPS * scale:
            if abs(c) <= _ROOT_EPS * scale:
                raise ValueError("The polynomial is identically zero.")
            return []
        return [-c / b]
    discriminant = b * b - 4.0 * a * c
    if discriminant < -_ROOT_EPS * scale * scale:
        return []
    if discriminant <= 0.0:
        return [-b / (2.0 * a)]
    root = math.sqrt(discriminant)
    return [(-b + root) / (2.0 * a), (-b - root) / (2.0 * a)]


def _cubic_roots(a: float, b: float, c: float, d: float) -> list[float]:
    """Return the real roots of ``a t^3 + b t^2 + c t + d``."""
    scale = max(1.0, abs(a), abs(b), abs(c), abs(d))
    if abs(a) <= _ROOT_EPS * scale:
        return _quadratic_roots(b, c, d)
    b /= a
    c /= a
    d /= a
    shift = b / 3.0
    p = c - b * b / 3.0
    q = (2.0 * b * b * b - 9.0 * b * c + 27.0 * d) / 27.0
    discriminant = (q / 2.0) ** 2 + (p / 3.0) ** 3
    if discriminant > _ROOT_EPS:
        root = math.sqrt(discriminant)
        u_value = _cbrt(-q / 2.0 + root) + _cbrt(-q / 2.0 - root)
        return [u_value - shift]
    if discriminant >= -_ROOT_EPS:
        u_value = _cbrt(-q / 2.0)
        return [2.0 * u_value - shift, -u_value - shift]
    radius = math.sqrt(-p / 3.0)
    cosine = (-q / 2.0) / (radius**3)
    if cosine < -1.0:
        cosine = -1.0
    elif cosine > 1.0:
        cosine = 1.0
    angle = math.acos(cosine)
    return [
        2.0 * radius * math.cos(angle / 3.0) - shift,
        2.0 * radius * math.cos((angle + 2.0 * math.pi) / 3.0) - shift,
        2.0 * radius * math.cos((angle + 4.0 * math.pi) / 3.0) - shift,
    ]


def _real_roots(coeffs: Sequence[float]) -> list[float]:
    """Return the real roots of a polynomial, highest degree first."""
    values = [float(coeff) for coeff in coeffs]
    scale = max(1.0, max(abs(coeff) for coeff in values))
    while len(values) > 1 and abs(values[0]) <= _ROOT_EPS * scale:
        values.pop(0)
    if all(abs(coeff) <= _ROOT_EPS * scale for coeff in values):
        raise ValueError("The polynomial is identically zero.")
    if len(values) == 1:
        return []
    if len(values) == 2:
        return [-values[1] / values[0]]
    if len(values) == 3:
        return _quadratic_roots(values[0], values[1], values[2])
    return _cubic_roots(values[0], values[1], values[2], values[3])


def _power_coeffs(
    points: Sequence[tuple[float, float]], index: int
) -> list[float]:
    """Return power-basis coefficients of one coordinate, highest degree first."""
    coords = [point[index] for point in points]
    if len(coords) == 4:
        x0, x1, x2, x3 = coords
        return [
            -x0 + 3.0 * x1 - 3.0 * x2 + x3,
            3.0 * x0 - 6.0 * x1 + 3.0 * x2,
            -3.0 * x0 + 3.0 * x1,
            x0,
        ]
    x0, x1, x2 = coords
    return [x0 - 2.0 * x1 + x2, -2.0 * x0 + 2.0 * x1, x0]


def _derivative_coeffs(
    points: Sequence[tuple[float, float]], index: int
) -> list[float]:
    """Return power-basis coefficients of one derivative coordinate."""
    coords = [point[index] for point in points]
    if len(coords) == 3:
        d0 = 2.0 * (coords[1] - coords[0])
        d1 = 2.0 * (coords[2] - coords[1])
        return [d1 - d0, d0]
    d0 = 3.0 * (coords[1] - coords[0])
    d1 = 3.0 * (coords[2] - coords[1])
    d2 = 3.0 * (coords[3] - coords[2])
    return [d0 - 2.0 * d1 + d2, 2.0 * (d1 - d0), d0]


def get_second_derivative(t: float, points: Sequence[PointType]) -> list[float]:
    """Return the second derivative of a quadratic or cubic Bézier at ``t``.

    Examples:
        >>> import simetri.graphics as sg
        >>> [round(c, 6) for c in sg.get_second_derivative(0.0, [(0, 0), (20, 40), (60, 40), (80, 0)])]
        [120.0, -240.0]
    """
    controls = _control_array(points)
    if len(controls) == 3:
        p0, p1, p2 = controls
        return [
            2.0 * (p2[0] - 2.0 * p1[0] + p0[0]),
            2.0 * (p2[1] - 2.0 * p1[1] + p0[1]),
        ]
    p0, p1, p2, p3 = controls
    start_x = 6.0 * (p2[0] - 2.0 * p1[0] + p0[0])
    start_y = 6.0 * (p2[1] - 2.0 * p1[1] + p0[1])
    end_x = 6.0 * (p3[0] - 2.0 * p2[0] + p1[0])
    end_y = 6.0 * (p3[1] - 2.0 * p2[1] + p1[1])
    return [
        (1.0 - t) * start_x + t * end_x,
        (1.0 - t) * start_y + t * end_y,
    ]


def get_curvature(t: float, points: Sequence[PointType]) -> float:
    """Return the signed curvature at ``t``.

    Examples:
        >>> import simetri.graphics as sg
        >>> round(sg.get_curvature(0.5, [(0, 0), (1, 2), (2, 0)]), 6)
        -2.0
    """
    if len(points) == 4:
        first = get_cubic_derivative(t, points)
    else:
        first = get_quadratic_derivative(t, points)
    second = get_second_derivative(t, points)
    speed_sq = first[0] * first[0] + first[1] * first[1]
    if speed_sq <= _ROOT_EPS:
        raise ValueError("Curvature is undefined where the derivative is zero.")
    cross = first[0] * second[1] - first[1] * second[0]
    return cross / (speed_sq**1.5)


def get_derivative_roots(
    points: Sequence[PointType],
) -> tuple[list[float], list[float]]:
    """Return ``t`` in ``[0, 1]`` where the x or y derivative is zero.

    Examples:
        >>> import simetri.graphics as sg
        >>> xs, ys = sg.get_derivative_roots([(0, 0), (0, 60), (80, 60), (80, 0)])
        >>> [round(t, 6) for t in xs]
        [0.0, 1.0]
        >>> [round(t, 6) for t in ys]
        [0.5]
    """
    controls = _control_array(points)

    def isolated(index: int) -> list[float]:
        try:
            roots = _real_roots(_derivative_coeffs(controls, index))
        except ValueError:
            return []
        return _unique_unit_roots(roots)

    return isolated(0), isolated(1)


def _curve_point(
    points: Sequence[tuple[float, float]], t: float
) -> tuple[float, float]:
    """Evaluate a quadratic or cubic Bézier at ``t``."""
    if len(points) == 3:
        p0, p1, p2 = points
        mt = 1.0 - t
        return (
            mt * mt * p0[0] + 2.0 * mt * t * p1[0] + t * t * p2[0],
            mt * mt * p0[1] + 2.0 * mt * t * p1[1] + t * t * p2[1],
        )
    p0, p1, p2, p3 = points
    mt = 1.0 - t
    mt2 = mt * mt
    t2 = t * t
    return (
        mt2 * mt * p0[0]
        + 3.0 * mt2 * t * p1[0]
        + 3.0 * mt * t2 * p2[0]
        + t2 * t * p3[0],
        mt2 * mt * p0[1]
        + 3.0 * mt2 * t * p1[1]
        + 3.0 * mt * t2 * p2[1]
        + t2 * t * p3[1],
    )


def get_exact_bbox(
    points: Sequence[PointType],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the exact axis-aligned box as southwest and northeast.

    Examples:
        >>> import simetri.graphics as sg
        >>> sw, ne = sg.get_exact_bbox([(0, 0), (0, 60), (80, 60), (80, 0)])
        >>> ([round(c, 6) for c in sw], [round(c, 6) for c in ne])
        ([0.0, 0.0], [80.0, 45.0])
    """
    controls = _control_array(points)
    x_roots, y_roots = get_derivative_roots(controls)
    samples = [0.0, 1.0, *x_roots, *y_roots]
    xs = []
    ys = []
    for t in samples:
        x, y = _curve_point(controls, t)
        xs.append(x)
        ys.append(y)
    return (min(xs), min(ys)), (max(xs), max(ys))


def get_aligned_controls(
    points: Sequence[PointType],
) -> tuple[list[tuple[float, float]], tuple[float, float], float]:
    """Return controls with the start at the origin and the end on ``+x``.

    Also returns the original start point and the angle that maps the
    aligned curve back.

    Examples:
        >>> import simetri.graphics as sg
        >>> controls, origin, angle = sg.get_aligned_controls([(0, 0), (0, 20), (0, 40)])
        >>> [round(c, 6) for c in origin]
        [0.0, 0.0]
        >>> round(angle, 6)
        1.570796
        >>> [[round(c, 6) for c in p] for p in controls]
        [[0.0, 0.0], [20.0, 0.0], [40.0, 0.0]]
    """
    controls = _control_array(points)
    origin = controls[0]
    end_x = controls[-1][0] - origin[0]
    end_y = controls[-1][1] - origin[1]
    if end_x * end_x + end_y * end_y <= _ROOT_EPS:
        raise ValueError(
            "Bezier endpoints must be distinct to align the curve."
        )
    angle = math.atan2(end_y, end_x)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    aligned = []
    for x, y in controls:
        dx = x - origin[0]
        dy = y - origin[1]
        aligned.append((dx * cosine + dy * sine, -dx * sine + dy * cosine))
    return aligned, origin, angle


def _map_from_aligned(
    point: tuple[float, float],
    origin: tuple[float, float],
    angle: float,
) -> tuple[float, float]:
    """Map an aligned-curve point back to the original curve."""
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        origin[0] + point[0] * cosine - point[1] * sine,
        origin[1] + point[0] * sine + point[1] * cosine,
    )


def get_tight_bbox(points: Sequence[PointType]) -> list[tuple[float, float]]:
    """Return the aligned box corners mapped back to ``points``.

    The order is southwest, southeast, northeast, northwest of the aligned box.

    Examples:
        >>> import simetri.graphics as sg
        >>> corners = sg.get_tight_bbox([(0, 0), (0, 60), (80, 60), (80, 0)])
        >>> [[round(c, 6) for c in p] for p in corners]
        [[0.0, 0.0], [80.0, 0.0], [80.0, 45.0], [0.0, 45.0]]
    """
    aligned, origin, angle = get_aligned_controls(points)
    (xmin, ymin), (xmax, ymax) = get_exact_bbox(aligned)
    corners = (
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
    )
    return [_map_from_aligned(corner, origin, angle) for corner in corners]


def get_inflections(points: Sequence[PointType]) -> list[float]:
    """Return ``t`` values where a cubic curve changes curvature sign.

    A quadratic curve has no inflection, so the result is empty.

    Examples:
        >>> import simetri.graphics as sg
        >>> [round(t, 6) for t in sg.get_inflections([(0, 0), (0, 1), (1, 0), (1, 1)])]
        [0.5]
        >>> sg.get_inflections([(0, 0), (0, 60), (80, 60), (80, 0)])
        []
    """
    controls = _control_array(points)
    if len(controls) == 3:
        return []
    x_deriv = _derivative_coeffs(controls, 0)
    y_deriv = _derivative_coeffs(controls, 1)
    # B'' from the derivative of the quadratic derivative polynomial.
    # Derivative coeffs are [a, b, c] for a t^2 + b t + c.
    # Second derivative is 2 a t + b, so [2a, b].
    x_second = [2.0 * x_deriv[0], x_deriv[1]]
    y_second = [2.0 * y_deriv[0], y_deriv[1]]
    # (a t^2 + b t + c)(p t + q) cross product, as a cubic.
    ax, bx, cx = x_deriv
    ay, by, cy = y_deriv
    px, qx = x_second
    py, qy = y_second
    coeffs = [
        ax * py - ay * px,
        ax * qy + bx * py - ay * qx - by * px,
        bx * qy + cx * py - by * qx - cy * px,
        cx * qy - cy * qx,
    ]
    try:
        roots = _real_roots(coeffs)
    except ValueError:
        return []
    return _unique_unit_roots(roots)


def _speed(points: Sequence[PointType], t: float) -> float:
    """Return the parametric speed at ``t``."""
    if len(points) == 4:
        derivative = get_cubic_derivative(t, points)
    else:
        derivative = get_quadratic_derivative(t, points)
    return math.hypot(derivative[0], derivative[1])


def _gauss_length(points: Sequence[PointType], end: float) -> float:
    """Return a Gauss-Legendre estimate of length from 0 to ``end``."""
    if end == 0.0:
        return 0.0
    panels = 8
    step = end / panels
    total = 0.0
    for index in range(panels):
        half = 0.5 * step
        mid = index * step + half
        panel = _GAUSS_WEIGHTS[0] * _speed(points, mid)
        for node, weight in zip(_GAUSS_NODES[1:], _GAUSS_WEIGHTS[1:]):
            panel += weight * (
                _speed(points, mid + half * node)
                + _speed(points, mid - half * node)
            )
        total += half * panel
    return total


def _sqrt_quadratic_antiderivative(
    a_coeff: float, b_coeff: float, c_coeff: float, t: float
) -> float:
    """Return the antiderivative of ``sqrt(a t^2 + b t + c)`` at ``t``."""
    scale = max(1.0, abs(a_coeff), abs(b_coeff), abs(c_coeff))
    if abs(a_coeff) <= _ROOT_EPS * scale:
        if abs(b_coeff) <= _ROOT_EPS * scale:
            return t * math.sqrt(max(0.0, c_coeff))
        height = b_coeff * t + c_coeff
        return (2.0 / 3.0) * (max(0.0, height) ** 1.5) / b_coeff
    inside = a_coeff * t * t + b_coeff * t + c_coeff
    root = math.sqrt(max(0.0, inside))
    linear = (2.0 * a_coeff * t + b_coeff) * root / (4.0 * a_coeff)
    gap = 4.0 * a_coeff * c_coeff - b_coeff * b_coeff
    log_arg = 2.0 * math.sqrt(a_coeff) * root + 2.0 * a_coeff * t + b_coeff
    if log_arg <= _ROOT_EPS:
        return linear
    log_term = gap / (8.0 * (a_coeff**1.5)) * math.log(log_arg)
    return linear + log_term


def _quadratic_arc_length(
    points: Sequence[tuple[float, float]], end: float
) -> float:
    """Return the exact arc length of a quadratic from 0 to ``end``."""
    p0, p1, p2 = points
    v0x = 2.0 * (p1[0] - p0[0])
    v0y = 2.0 * (p1[1] - p0[1])
    v1x = 2.0 * (p2[0] - p1[0])
    v1y = 2.0 * (p2[1] - p1[1])
    ax = v1x - v0x
    ay = v1y - v0y
    a_coeff = ax * ax + ay * ay
    b_coeff = 2.0 * (ax * v0x + ay * v0y)
    c_coeff = v0x * v0x + v0y * v0y
    return _sqrt_quadratic_antiderivative(
        a_coeff, b_coeff, c_coeff, end
    ) - _sqrt_quadratic_antiderivative(a_coeff, b_coeff, c_coeff, 0.0)


def get_arc_length(points: Sequence[PointType], t: float = 1.0) -> float:
    """Return the arc length from the start to ``t``.

    Quadratic curves use the closed form. Cubic curves use Gauss-Legendre.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.get_arc_length([(0, 0), (10, 0), (20, 0), (30, 0)])
        30.0
        >>> sg.get_arc_length([(0, 0), (15, 0), (30, 0)])
        30.0
    """
    if t < 0.0 or t > 1.0:
        raise ValueError("t must satisfy 0 <= t <= 1.")
    controls = _control_array(points)
    if len(controls) == 3:
        return _quadratic_arc_length(controls, t)
    return _gauss_length(controls, t)


def get_t_at_arc_length(points: Sequence[PointType], length: float) -> float:
    """Return ``t`` at ``length`` measured from the start.

    Examples:
        >>> import simetri.graphics as sg
        >>> round(sg.get_t_at_arc_length([(0, 0), (10, 0), (20, 0), (30, 0)], 10), 6)
        0.333333
    """
    controls = _control_array(points)
    total = get_arc_length(controls, 1.0)
    tolerance = _ROOT_EPS * max(1.0, total)
    if length < -tolerance or length > total + tolerance:
        raise ValueError("length is outside the curve.")
    if length <= tolerance:
        return 0.0
    if abs(length - total) <= tolerance:
        return 1.0
    t = length / total
    for _ in range(20):
        traveled = get_arc_length(controls, t)
        speed = _speed(controls, t)
        if speed <= _ROOT_EPS:
            break
        t_next = t - (traveled - length) / speed
        if t_next < 0.0:
            t_next = 0.0
        elif t_next > 1.0:
            t_next = 1.0
        if abs(t_next - t) <= 1e-12:
            return t_next
        t = t_next
    low = 0.0
    high = 1.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if get_arc_length(controls, mid) < length:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def get_y_at_x(points: Sequence[PointType], x: float) -> list[float]:
    """Return every ``y`` on the curve where the x coordinate is ``x``.

    Examples:
        >>> import simetri.graphics as sg
        >>> [round(y, 6) for y in sg.get_y_at_x([(0, 0), (0, 60), (80, 60), (80, 0)], 40)]
        [45.0]
    """
    controls = _control_array(points)
    coeffs = _power_coeffs(controls, 0)
    coeffs[-1] -= x
    try:
        roots = _unique_unit_roots(_real_roots(coeffs))
    except ValueError as error:
        raise ValueError("The curve has constant x.") from error
    return [_curve_point(controls, t)[1] for t in roots]


def get_line_intersections(
    points: Sequence[PointType],
    start: PointType,
    end: PointType,
    segment: bool = False,
) -> list[tuple[float, tuple[float, float]]]:
    """Return ``(t, point)`` hits of the curve with a line or segment.

    Examples:
        >>> import simetri.graphics as sg
        >>> hits = sg.get_line_intersections(
        ...     [(0, 0), (0, 60), (80, 60), (80, 0)], (0, 45), (80, 45)
        ... )
        >>> [(round(t, 6), [round(c, 6) for c in p]) for t, p in hits]
        [(0.5, [40.0, 45.0])]
        >>> sg.get_line_intersections(
        ...     [(0, 0), (0, 60), (80, 60), (80, 0)],
        ...     (40, 0),
        ...     (40, 10),
        ...     segment=True,
        ... )
        []
    """
    controls = _control_array(points)
    x1, y1 = _xy(start)
    x2, y2 = _xy(end)
    dx = x2 - x1
    dy = y2 - y1
    if dx * dx + dy * dy <= _ROOT_EPS:
        raise ValueError("Line endpoints must be distinct.")
    x_coeffs = _power_coeffs(controls, 0)
    y_coeffs = _power_coeffs(controls, 1)
    coeffs = [
        dy * x_coeff - dx * y_coeff
        for x_coeff, y_coeff in zip(x_coeffs, y_coeffs)
    ]
    coeffs[-1] -= x1 * dy - y1 * dx
    try:
        roots = _unique_unit_roots(_real_roots(coeffs))
    except ValueError as error:
        raise ValueError("The curve lies on the line.") from error
    length_sq = dx * dx + dy * dy
    hits = []
    for t in roots:
        point = _curve_point(controls, t)
        if segment:
            along = ((point[0] - x1) * dx + (point[1] - y1) * dy) / length_sq
            if along < -1e-8 or along > 1.0 + 1e-8:
                continue
        hits.append((t, point))
    return hits


def elevate_quadratic(
    points: Sequence[PointType],
) -> list[tuple[float, float]]:
    """Return the cubic controls that match a quadratic exactly.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.elevate_quadratic([(0, 0), (30, 60), (60, 0)])
        [(0.0, 0.0), (20.0, 40.0), (40.0, 40.0), (60.0, 0.0)]
    """
    controls = _control_array(points)
    if len(controls) != 3:
        raise ValueError("Only a quadratic Bezier can be elevated.")
    p0, p1, p2 = controls
    return [
        p0,
        ((p0[0] + 2.0 * p1[0]) / 3.0, (p0[1] + 2.0 * p1[1]) / 3.0),
        ((2.0 * p1[0] + p2[0]) / 3.0, (2.0 * p1[1] + p2[1]) / 3.0),
        p2,
    ]


def reduce_cubic(points: Sequence[PointType]) -> list[tuple[float, float]]:
    """Return a least-squares quadratic approximation of a cubic.

    Examples:
        >>> import simetri.graphics as sg
        >>> reduced = sg.reduce_cubic([(0, 0), (20, 40), (40, 40), (60, 0)])
        >>> [[round(c, 6) for c in p] for p in reduced]
        [[0.0, 0.0], [30.0, 60.0], [60.0, 0.0]]
    """
    controls = _control_array(points)
    if len(controls) != 4:
        raise ValueError("Only a cubic Bezier can be reduced.")
    elevation = array(
        [
            [1.0, 0.0, 0.0],
            [1.0 / 3.0, 2.0 / 3.0, 0.0],
            [0.0, 2.0 / 3.0, 1.0 / 3.0],
            [0.0, 0.0, 1.0],
        ]
    )
    target = array(controls, dtype=float)
    fitted, _, _, _ = np.linalg.lstsq(elevation, target, rcond=None)
    return [(float(point[0]), float(point[1])) for point in fitted]


def rational_bezier_point(
    t: float,
    points: Sequence[PointType],
    weights: Sequence[float],
) -> list[float]:
    """Return the point at ``t`` on a rational quadratic or cubic.

    Examples:
        >>> import simetri.graphics as sg
        >>> [round(c, 6) for c in sg.rational_bezier_point(0.5, [(0, 0), (20, 40), (60, 40), (80, 0)], [1, 1, 1, 1])]
        [40.0, 30.0]
    """
    controls = _control_array(points)
    if len(weights) != len(controls):
        raise ValueError("weights must match the control points.")
    if len(controls) == 3:
        mt = 1.0 - t
        basis = [mt * mt, 2.0 * mt * t, t * t]
    else:
        mt = 1.0 - t
        mt2 = mt * mt
        t2 = t * t
        basis = [mt2 * mt, 3.0 * mt2 * t, 3.0 * mt * t2, t2 * t]
    denominator = 0.0
    x = 0.0
    y = 0.0
    for weight, basis_value, point in zip(weights, basis, controls):
        weighted = float(weight) * basis_value
        denominator += weighted
        x += weighted * point[0]
        y += weighted * point[1]
    if abs(denominator) <= _ROOT_EPS:
        raise ValueError("Rational Bezier weight basis is zero.")
    return [x / denominator, y / denominator]


def _arc_piece(
    center: tuple[float, float],
    radius: float,
    start_angle: float,
    sweep: float,
) -> Bezier:
    """Return one cubic of at most a quarter turn."""
    kappa = (4.0 / 3.0) * math.tan(sweep / 4.0)
    handle = radius * kappa
    start = (
        center[0] + radius * math.cos(start_angle),
        center[1] + radius * math.sin(start_angle),
    )
    end_angle = start_angle + sweep
    end = (
        center[0] + radius * math.cos(end_angle),
        center[1] + radius * math.sin(end_angle),
    )
    start_tangent = (-math.sin(start_angle), math.cos(start_angle))
    end_tangent = (-math.sin(end_angle), math.cos(end_angle))
    controls = [
        start,
        (
            start[0] + handle * start_tangent[0],
            start[1] + handle * start_tangent[1],
        ),
        (
            end[0] - handle * end_tangent[0],
            end[1] - handle * end_tangent[1],
        ),
        end,
    ]
    return Bezier(controls)


def circular_arc_beziers(
    center: PointType,
    radius: float,
    start_angle: float,
    sweep: float,
) -> list[Bezier]:
    """Return cubics that approximate a circular arc.

    ``sweep`` is the signed angle in radians. Pieces are at most a
    quarter turn.

    Examples:
        >>> import simetri.graphics as sg
        >>> arcs = sg.circular_arc_beziers((0, 0), 1, 0, sg.pi / 2)
        >>> len(arcs)
        1
        >>> [round(c, 6) for c in arcs[0].point(0.5)]
        [0.707107, 0.707107]
        >>> len(sg.circular_arc_beziers((0, 0), 10, 0, 2 * sg.pi))
        4
    """
    if radius <= 0:
        raise ValueError("radius must be positive.")
    if sweep == 0:
        raise ValueError("sweep must be nonzero.")
    origin = _xy(center)
    remaining = sweep
    angle = start_angle
    direction = 1.0 if sweep > 0.0 else -1.0
    curves = []
    while abs(remaining) > _ROOT_EPS:
        piece = direction * min(abs(remaining), math.pi / 2.0)
        curves.append(_arc_piece(origin, radius, angle, piece))
        angle += piece
        remaining -= piece
    return curves


class BezierArc:
    """One circular arc, or a straight segment, from a Bézier fit.

    ``straight`` is True when the piece is a line. A line has no center and
    its radius is infinite.
    """

    def __init__(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        center: tuple[float, float] | None,
        radius: float,
        start_angle: float,
        sweep: float,
        straight: bool,
    ) -> None:
        self.start = start
        self.end = end
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.sweep = sweep
        self.straight = straight

    def __repr__(self) -> str:
        """Return a concise arc representation."""
        if self.straight:
            return f"BezierArc({self.start}, {self.end})"
        return (
            f"BezierArc({self.start}, {self.end}, center={self.center}, "
            f"radius={self.radius})"
        )


def _sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _cross(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _hypot(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])


def _unit_tangent(points: Sequence[tuple[float, float]], t: float) -> tuple[float, float]:
    if len(points) == 3:
        derivative = get_quadratic_derivative(t, points)
    else:
        derivative = get_cubic_derivative(t, points)
    length = math.hypot(derivative[0], derivative[1])
    if length <= _ROOT_EPS:
        raise ValueError("Bezier derivative is zero.")
    return (derivative[0] / length, derivative[1] / length)


def _end_tangent(
    points: Sequence[tuple[float, float]], at_start: bool
) -> tuple[float, float]:
    """Unit tangent at an end, stepping inward when that end is a cusp."""
    parameter = 0.0 if at_start else 1.0
    try:
        return _unit_tangent(points, parameter)
    except ValueError:
        parameter = 0.02 if at_start else 0.98
        return _unit_tangent(points, parameter)


def _cusp_times(points: Sequence[tuple[float, float]]) -> list[float]:
    """Parameters in (0, 1) where the derivative vanishes."""
    x_roots, y_roots = get_derivative_roots(points)
    times = []
    for parameter in x_roots + y_roots:
        if 0.02 < parameter < 0.98 and _speed(points, parameter) <= 1.0e-6:
            if all(abs(parameter - found) > 1.0e-5 for found in times):
                times.append(parameter)
    return times


def _left_normal(tangent: tuple[float, float]) -> tuple[float, float]:
    return (-tangent[1], tangent[0])


def _split_controls(
    points: Sequence[tuple[float, float]], t: float
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    if len(points) == 3:
        left, right = split_q_bezier(points[0], points[1], points[2], t, n_points=5)
    else:
        left, right = split_bezier(
            points[0], points[1], points[2], points[3], t, n_points=5
        )
    return _control_array(left.control_points), _control_array(right.control_points)


def _rotation_angle(
    origin: tuple[float, float], target: tuple[float, float]
) -> float:
    """Signed short sweep from ``origin`` to ``target``, at most half a turn."""
    cross = _cross(origin, target)
    cosine = _dot(origin, target) / (_hypot(origin) * _hypot(target))
    cosine = max(-1.0, min(1.0, cosine))
    acute = math.acos(cosine)
    if cross > 0.0:
        return acute
    return -acute


def _arc_between(
    start: tuple[float, float],
    tangent: tuple[float, float],
    end: tuple[float, float],
) -> BezierArc:
    """Arc from ``start`` along ``tangent`` that ends at ``end``."""
    chord = _sub(end, start)
    normal = _left_normal(tangent)
    denom = 2.0 * _dot(normal, chord)
    if abs(denom) <= _ROOT_EPS * max(1.0, _hypot(chord)):
        return BezierArc(start, end, None, math.inf, 0.0, 0.0, True)
    scale = _dot(chord, chord) / denom
    if abs(scale) > 1.0e6:
        return BezierArc(start, end, None, math.inf, 0.0, 0.0, True)
    center = (start[0] + scale * normal[0], start[1] + scale * normal[1])
    radius = abs(scale)
    start_vector = _sub(start, center)
    end_vector = _sub(end, center)
    sweep = _rotation_angle(start_vector, end_vector)
    start_angle = math.atan2(start_vector[1], start_vector[0])
    if _dot(_travel_tangent(start_angle, sweep), tangent) < 0.0:
        sweep += -2.0 * math.pi if sweep >= 0.0 else 2.0 * math.pi
    return BezierArc(start, end, center, radius, start_angle, sweep, False)


def _travel_tangent(angle: float, sweep: float) -> tuple[float, float]:
    """Unit tangent of a circle traversal at ``angle``."""
    if sweep >= 0.0:
        return (-math.sin(angle), math.cos(angle))
    return (math.sin(angle), -math.cos(angle))


def _line_arc(start: tuple[float, float], end: tuple[float, float]) -> BezierArc:
    return BezierArc(start, end, None, math.inf, 0.0, 0.0, True)


def _biarc(
    start: tuple[float, float],
    start_tangent: tuple[float, float],
    end: tuple[float, float],
    end_tangent: tuple[float, float],
) -> list[BezierArc] | None:
    """Two tangent-matched arcs from ``start`` to ``end``.

    Returns None when the equal-distance joint is not a short biarc.
    """
    chord = _sub(end, start)
    chord_length2 = _dot(chord, chord)
    if chord_length2 <= _ROOT_EPS:
        return None
    tangent_dot = max(-1.0, min(1.0, _dot(start_tangent, end_tangent)))
    if 1.0 - tangent_dot <= 1.0e-8:
        along = _dot(chord, end_tangent)
        if abs(along) <= _ROOT_EPS * math.sqrt(chord_length2):
            joint = (
                (start[0] + end[0]) / 2.0,
                (start[1] + end[1]) / 2.0,
            )
            radius = math.sqrt(chord_length2) / 4.0
            first_center = (
                start[0] + chord[0] / 4.0,
                start[1] + chord[1] / 4.0,
            )
            second_center = (
                start[0] + 3.0 * chord[0] / 4.0,
                start[1] + 3.0 * chord[1] / 4.0,
            )
            turn = math.pi if _cross(chord, end_tangent) < 0.0 else -math.pi
            first_vector = _sub(start, first_center)
            second_vector = _sub(joint, second_center)
            return [
                BezierArc(
                    start,
                    joint,
                    first_center,
                    radius,
                    math.atan2(first_vector[1], first_vector[0]),
                    turn,
                    False,
                ),
                BezierArc(
                    joint,
                    end,
                    second_center,
                    radius,
                    math.atan2(second_vector[1], second_vector[0]),
                    -turn,
                    False,
                ),
            ]
        distance = chord_length2 / (4.0 * along)
    else:
        tangent_sum = (
            start_tangent[0] + end_tangent[0],
            start_tangent[1] + end_tangent[1],
        )
        along_sum = _dot(chord, tangent_sum)
        discriminant = along_sum * along_sum + 2.0 * (1.0 - tangent_dot) * chord_length2
        if discriminant < 0.0:
            discriminant = 0.0
        distance = (-along_sum + math.sqrt(discriminant)) / (2.0 * (1.0 - tangent_dot))
    if distance <= _ROOT_EPS:
        return None
    joint = (
        (start[0] + end[0] + distance * (start_tangent[0] - end_tangent[0])) / 2.0,
        (start[1] + end[1] + distance * (start_tangent[1] - end_tangent[1])) / 2.0,
    )
    first = _arc_between(start, start_tangent, joint)
    if first.straight:
        arrival = start_tangent
    else:
        arrival = _travel_tangent(first.start_angle + first.sweep, first.sweep)
    return [first, _arc_between(joint, arrival, end)]


def _on_sweep(start_angle: float, sweep: float, angle: float) -> bool:
    if sweep >= 0.0:
        delta = (angle - start_angle) % (2.0 * math.pi)
        if delta > 2.0 * math.pi - 1.0e-6:
            delta = 0.0
        return delta <= sweep + 1.0e-6
    clockwise = (start_angle - angle) % (2.0 * math.pi)
    if clockwise > 2.0 * math.pi - 1.0e-6:
        clockwise = 0.0
    return clockwise <= -sweep + 1.0e-6


def _distance_to_arc(arc: BezierArc, point: tuple[float, float]) -> float:
    hit, gap = _closest_on_arc(arc, point)
    return gap


def _closest_on_arc(
    arc: BezierArc, point: tuple[float, float]
) -> tuple[tuple[float, float], float]:
    if arc.straight or arc.center is None:
        return _closest_on_segment(arc.start, arc.end, point)
    offset = _sub(point, arc.center)
    center_distance = _hypot(offset)
    if center_distance <= _ROOT_EPS:
        return arc.start, arc.radius
    angle = math.atan2(offset[1], offset[0])
    if _on_sweep(arc.start_angle, arc.sweep, angle):
        hit = (
            arc.center[0] + arc.radius * math.cos(angle),
            arc.center[1] + arc.radius * math.sin(angle),
        )
        return hit, abs(center_distance - arc.radius)
    return _closer_endpoint(arc.start, arc.end, point)


def _closest_on_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    point: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    chord = _sub(end, start)
    length2 = _dot(chord, chord)
    if length2 <= _ROOT_EPS:
        return start, _hypot(_sub(point, start))
    parameter = _dot(_sub(point, start), chord) / length2
    if parameter < 0.0:
        parameter = 0.0
    elif parameter > 1.0:
        parameter = 1.0
    hit = (start[0] + parameter * chord[0], start[1] + parameter * chord[1])
    return hit, _hypot(_sub(point, hit))


def _closer_endpoint(
    start: tuple[float, float],
    end: tuple[float, float],
    point: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    start_gap = _hypot(_sub(point, start))
    end_gap = _hypot(_sub(point, end))
    if start_gap <= end_gap:
        return start, start_gap
    return end, end_gap


def _fit_gap(
    points: Sequence[tuple[float, float]], arcs: Sequence[BezierArc]
) -> float:
    worst = 0.0
    for step in range(1, 8):
        sample = _curve_point(points, step / 8.0)
        worst = max(worst, min(_distance_to_arc(arc, sample) for arc in arcs))
    return worst


def _arcs_of_curve(
    points: Sequence[tuple[float, float]], tolerance: float, depth: int
) -> list[BezierArc]:
    if depth < _ARC_SPLIT_LIMIT:
        for parameter in _cusp_times(points) + [
            inflection
            for inflection in get_inflections(points)
            if 0.02 < inflection < 0.98
        ]:
            left, right = _split_controls(points, parameter)
            return _arcs_of_curve(left, tolerance, depth + 1) + _arcs_of_curve(
                right, tolerance, depth + 1
            )
    start = points[0]
    end = points[-1]
    if _hypot(_sub(end, start)) <= _ROOT_EPS:
        if depth >= _ARC_SPLIT_LIMIT:
            return []
        left, right = _split_controls(points, 0.5)
        return _arcs_of_curve(left, tolerance, depth + 1) + _arcs_of_curve(
            right, tolerance, depth + 1
        )
    arcs = _biarc(start, _end_tangent(points, True), end, _end_tangent(points, False))
    if arcs is None or (
        depth < _ARC_SPLIT_LIMIT and _fit_gap(points, arcs) > tolerance
    ):
        if depth >= _ARC_SPLIT_LIMIT:
            if arcs is None:
                return [_line_arc(start, end)]
            return arcs
        left, right = _split_controls(points, 0.5)
        return _arcs_of_curve(left, tolerance, depth + 1) + _arcs_of_curve(
            right, tolerance, depth + 1
        )
    return arcs


def bezier_to_arcs(
    points: Sequence[PointType], tolerance: float = 0.01
) -> list[BezierArc]:
    """Approximate a quadratic or cubic Bézier with circular arcs.

    Each piece matches the curve's endpoints and end tangents. Pieces that
    miss the curve by more than ``tolerance`` are split and fitted again.
    Straight pieces have ``straight`` True.

    Examples:
        >>> import simetri.graphics as sg
        >>> line = sg.bezier_to_arcs([(0, 0), (10, 0), (20, 0), (30, 0)])
        >>> line[0].start == (0.0, 0.0) and line[-1].end == (30.0, 0.0)
        True
        >>> all(arc.straight for arc in line)
        True
        >>> quarter = sg.circular_arc_beziers((0, 0), 10, 0, sg.pi / 2)[0]
        >>> arcs = sg.bezier_to_arcs(quarter.control_points)
        >>> all(abs(c) < 1e-9 for c in arcs[0].center)
        True
        >>> round(arcs[0].radius, 5)
        10.0
        >>> [round(c, 5) for c in arcs[-1].end]
        [0.0, 10.0]
    """
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive.")
    controls = _control_array(points)
    if len(controls) not in (3, 4):
        raise ValueError("A Bezier curve needs 3 or 4 control points.")
    return _arcs_of_curve(controls, tolerance, 0)


def bezier_closest_point(
    points: Sequence[PointType],
    point: PointType,
    tolerance: float = 0.01,
) -> tuple[tuple[float, float], float]:
    """Return the closest point on the arc fit and the distance to it.

    Examples:
        >>> import simetri.graphics as sg
        >>> point, distance = sg.bezier_closest_point(
        ...     [(0, 0), (10, 0), (20, 0), (30, 0)], (15, 5)
        ... )
        >>> [round(c, 6) for c in point]
        [15.0, 0.0]
        >>> round(distance, 6)
        5.0
        >>> top, gap = sg.bezier_closest_point([(0, 0), (1, 2), (2, 0)], (1, 2))
        >>> [round(c, 6) for c in top]
        [1.0, 1.0]
        >>> round(gap, 6)
        1.0
    """
    target = _xy(point)
    best_point = _xy(points[0])
    best_gap = math.inf
    for arc in bezier_to_arcs(points, tolerance):
        hit, gap = _closest_on_arc(arc, target)
        if gap < best_gap:
            best_point = hit
            best_gap = gap
    return best_point, best_gap


def project_point_onto_curve(
    points: Sequence[PointType],
    point: PointType,
    tolerance: float = 0.01,
) -> tuple[float, float]:
    """Project ``point`` onto a quadratic or cubic Bézier.

    The projection is the closest point on the curve's circular-arc fit.

    Examples:
        >>> import simetri.graphics as sg
        >>> projected = sg.project_point_onto_curve(
        ...     [(0, 0), (10, 0), (20, 0), (30, 0)], (15, 5)
        ... )
        >>> [round(c, 6) for c in projected]
        [15.0, 0.0]
    """
    projected, _distance = bezier_closest_point(points, point, tolerance)
    return projected


def flatten_curve(
    points: Sequence[PointType], n_points: int | None = None
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Sample a Bézier and join the samples with straight segments.

    Samples are spaced evenly in ``t``. ``n_points`` is the number of
    samples, including both endpoints. The default is
    ``n_bezier_points``.

    Examples:
        >>> import simetri.graphics as sg
        >>> lines = sg.flatten_curve([(0, 0), (40, 80), (80, 80), (60, 0)], n_points=5)
        >>> len(lines)
        4
        >>> [[round(c, 6) for c in lines[0][0]], [round(c, 6) for c in lines[0][1]]]
        [[0.0, 0.0], [29.0625, 45.0]]
        >>> [round(c, 6) for c in lines[-1][1]]
        [60.0, 0.0]
    """
    controls = _control_array(points)
    if n_points is None:
        n_points = runtime_defaults["n_bezier_points"]
    if len(controls) == 3:
        samples = q_bezier_points(controls[0], controls[1], controls[2], n_points)
    else:
        samples = bezier_points(
            controls[0], controls[1], controls[2], controls[3], n_points
        )
    vertices = [(float(sample[0]), float(sample[1])) for sample in samples]
    return list(zip(vertices, vertices[1:]))


def _unique_points(
    points: Sequence[tuple[float, float]], separation: float = 1.0e-5
) -> list[tuple[float, float]]:
    unique: list[tuple[float, float]] = []
    for point in sorted(points):
        if all(_hypot(_sub(point, kept)) > separation for kept in unique):
            unique.append(point)
    return unique


def _segment_hits(first: BezierArc, second: BezierArc) -> list[tuple[float, float]]:
    direction_a = _sub(first.end, first.start)
    direction_b = _sub(second.end, second.start)
    area = _cross(direction_a, direction_b)
    scale = max(1.0, _hypot(direction_a), _hypot(direction_b))
    if abs(area) <= 1.0e-8 * scale:
        offset = _sub(second.start, first.start)
        if abs(_cross(direction_a, offset)) <= 1.0e-8 * scale:
            if _segments_overlap(first.start, first.end, second.start, second.end):
                raise ValueError("The curves overlap.")
        return []
    offset = _sub(second.start, first.start)
    parameter_a = _cross(offset, direction_b) / area
    parameter_b = _cross(offset, direction_a) / area
    if -1.0e-8 <= parameter_a <= 1.0 + 1.0e-8 and -1.0e-8 <= parameter_b <= 1.0 + 1.0e-8:
        parameter_a = min(1.0, max(0.0, parameter_a))
        return [(
            first.start[0] + parameter_a * direction_a[0],
            first.start[1] + parameter_a * direction_a[1],
        )]
    return []


def _segments_overlap(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
) -> bool:
    axis = _sub(a1, a0)
    length2 = _dot(axis, axis)
    if length2 <= _ROOT_EPS:
        return False

    def parameter(point: tuple[float, float]) -> float:
        return _dot(_sub(point, a0), axis) / length2

    lo = max(min(parameter(a0), parameter(a1)), min(parameter(b0), parameter(b1)))
    hi = min(max(parameter(a0), parameter(a1)), max(parameter(b0), parameter(b1)))
    return hi - lo > 1.0e-5


def _circle_points(
    first_center: tuple[float, float],
    first_radius: float,
    second_center: tuple[float, float],
    second_radius: float,
) -> list[tuple[float, float]]:
    between = _sub(second_center, first_center)
    separation = _hypot(between)
    if separation <= 1.0e-8 and abs(first_radius - second_radius) <= 1.0e-6:
        raise ValueError("The curves overlap.")
    if separation <= 1.0e-8:
        return []
    if separation > first_radius + second_radius + 1.0e-6:
        return []
    if separation < abs(first_radius - second_radius) - 1.0e-6:
        return []
    along = (
        first_radius * first_radius - second_radius * second_radius + separation * separation
    ) / (2.0 * separation)
    height2 = first_radius * first_radius - along * along
    if height2 < 0.0:
        height2 = 0.0
    height = math.sqrt(height2)
    base = (
        first_center[0] + along * between[0] / separation,
        first_center[1] + along * between[1] / separation,
    )
    if height <= 1.0e-8:
        return [base]
    perpendicular = (
        -between[1] / separation * height,
        between[0] / separation * height,
    )
    return [
        (base[0] + perpendicular[0], base[1] + perpendicular[1]),
        (base[0] - perpendicular[0], base[1] - perpendicular[1]),
    ]


def _sweep_ranges(start_angle: float, sweep: float) -> list[tuple[float, float]]:
    if abs(sweep) <= 1.0e-8:
        return []
    begin = start_angle % (2.0 * math.pi)
    finish = (start_angle + sweep) % (2.0 * math.pi)
    if sweep > 0.0:
        if finish >= begin:
            return [(begin, finish)]
        return [(begin, 2.0 * math.pi), (0.0, finish)]
    if begin >= finish:
        return [(finish, begin)]
    return [(finish, 2.0 * math.pi), (0.0, begin)]


def _sweeps_overlap(first: BezierArc, second: BezierArc) -> bool:
    if first.center is None or second.center is None:
        return False
    amount = 0.0
    for left in _sweep_ranges(first.start_angle, first.sweep):
        for right in _sweep_ranges(second.start_angle, second.sweep):
            amount += max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    return amount > 1.0e-4


def _point_on_arc(arc: BezierArc, point: tuple[float, float]) -> bool:
    if arc.straight or arc.center is None:
        hit, gap = _closest_on_segment(arc.start, arc.end, point)
        return gap <= 1.0e-5
    if abs(_hypot(_sub(point, arc.center)) - arc.radius) > 1.0e-4 * max(1.0, arc.radius):
        return False
    angle = math.atan2(point[1] - arc.center[1], point[0] - arc.center[0])
    return _on_sweep(arc.start_angle, arc.sweep, angle)


def _arc_hits(first: BezierArc, second: BezierArc) -> list[tuple[float, float]]:
    if first.straight and second.straight:
        return _segment_hits(first, second)
    if first.straight:
        return [
            point for point in _segment_circle_hits(first, second)
            if _point_on_arc(second, point)
        ]
    if second.straight:
        return [
            point for point in _segment_circle_hits(second, first)
            if _point_on_arc(first, point)
        ]
    if first.center is None or second.center is None:
        return []
    same_circle = (
        _hypot(_sub(first.center, second.center)) <= 1.0e-6
        and abs(first.radius - second.radius) <= 1.0e-6
    )
    if same_circle:
        if _sweeps_overlap(first, second):
            raise ValueError("The curves overlap.")
        return []
    return [
        point
        for point in _circle_points(
            first.center, first.radius, second.center, second.radius
        )
        if _point_on_arc(first, point) and _point_on_arc(second, point)
    ]


def _segment_circle_hits(
    segment: BezierArc, arc: BezierArc
) -> list[tuple[float, float]]:
    if arc.center is None:
        return []
    direction = _sub(segment.end, segment.start)
    offset = _sub(segment.start, arc.center)
    a = _dot(direction, direction)
    if a <= _ROOT_EPS:
        if abs(_hypot(offset) - arc.radius) <= 1.0e-6:
            return [segment.start]
        return []
    b = 2.0 * _dot(offset, direction)
    c = _dot(offset, offset) - arc.radius * arc.radius
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return []
    root = math.sqrt(discriminant)
    hits = []
    for sign in (-1.0, 1.0):
        parameter = (-b + sign * root) / (2.0 * a)
        if -1.0e-8 <= parameter <= 1.0 + 1.0e-8:
            parameter = min(1.0, max(0.0, parameter))
            hits.append((
                segment.start[0] + parameter * direction[0],
                segment.start[1] + parameter * direction[1],
            ))
    return hits


def bezier_circle_intersections(
    points: Sequence[PointType],
    center: PointType,
    radius: float,
    tolerance: float = 0.01,
) -> list[tuple[float, float]]:
    """Return intersections of a Bézier with a circle.

    The curve is converted to arcs first. A curve that contains an arc of
    the circle raises ``ValueError``.

    Examples:
        >>> import simetri.graphics as sg
        >>> hits = sg.bezier_circle_intersections(
        ...     [(0, 0), (10, 0), (20, 0), (30, 0)], (15, 0), 5
        ... )
        >>> [[round(c, 6) for c in hit] for hit in hits]
        [[10.0, 0.0], [20.0, 0.0]]
        >>> quarter = sg.circular_arc_beziers((0, 0), 10, 0, sg.pi / 2)[0]
        >>> sg.bezier_circle_intersections(quarter.control_points, (0, 0), 10)
        Traceback (most recent call last):
        ...
        ValueError: The curve lies on the circle.
        >>> cuts = sg.bezier_circle_intersections(quarter.control_points, (10, 10), 10)
        >>> [[round(c, 6) for c in hit] for hit in cuts]
        [[0.0, 10.0], [10.0, 0.0]]
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    origin = _xy(center)
    query = BezierArc(
        (0.0, 0.0), (0.0, 0.0), origin, radius, 0.0, 2.0 * math.pi, False
    )
    hits: list[tuple[float, float]] = []
    for arc in bezier_to_arcs(points, tolerance):
        if arc.straight or arc.center is None:
            hits.extend(
                point
                for point in _segment_circle_hits(arc, query)
                if _point_on_arc(arc, point)
            )
            continue
        same = (
            _hypot(_sub(arc.center, origin)) <= 1.0e-5
            and abs(arc.radius - radius) <= 1.0e-5
        )
        if same:
            raise ValueError("The curve lies on the circle.")
        hits.extend(
            point
            for point in _circle_points(arc.center, arc.radius, origin, radius)
            if _point_on_arc(arc, point)
        )
    return _unique_points(hits)


def bezier_curve_intersections(
    points: Sequence[PointType],
    other: Sequence[PointType],
    tolerance: float = 0.01,
) -> list[tuple[float, float]]:
    """Return intersections of two Bézier curves.

    Both curves are converted to arcs first. Overlapping pieces raise
    ``ValueError``.

    Examples:
        >>> import simetri.graphics as sg
        >>> hits = sg.bezier_curve_intersections(
        ...     [(0, 0), (10, 0), (30, 0), (40, 0)],
        ...     [(20, -20), (20, -10), (20, 10), (20, 20)],
        ... )
        >>> [[round(c, 6) for c in hit] for hit in hits]
        [[20.0, 0.0]]
    """
    hits: list[tuple[float, float]] = []
    for arc in bezier_to_arcs(points, tolerance):
        for other_arc in bezier_to_arcs(other, tolerance):
            hits.extend(_arc_hits(arc, other_arc))
    return _unique_points(hits)


def bezier_self_intersections(
    points: Sequence[PointType], tolerance: float = 0.01
) -> list[tuple[float, float]]:
    """Return interior points where a Bézier crosses itself.

    Adjacent fitted arcs share a joint, so those joints are not reported.
    A curve that starts and ends at the same point does not report that
    shared endpoint. Overlapping pieces raise ``ValueError``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.bezier_self_intersections([(0, 0), (40, 80), (40, -80), (0, 0)])
        []
        >>> sg.bezier_self_intersections([(0, 0), (30, 0), (30, 0), (0, 0)])
        Traceback (most recent call last):
        ...
        ValueError: The curve overlaps itself.
        >>> hits = sg.bezier_self_intersections([(0, 0), (80, 80), (-20, 80), (60, 0)])
        >>> [[round(c, 2) for c in hit] for hit in hits]
        [[30.0, 40.0]]
    """
    arcs = bezier_to_arcs(points, tolerance)
    hits: list[tuple[float, float]] = []
    try:
        for index, arc in enumerate(arcs):
            if index + 1 < len(arcs):
                hits.extend(
                    hit
                    for hit in _arc_hits(arc, arcs[index + 1])
                    if _hypot(_sub(hit, arc.end)) > 1.0e-4
                )
            for other in arcs[index + 2 :]:
                hits.extend(_arc_hits(arc, other))
    except ValueError:
        raise ValueError("The curve overlaps itself.") from None
    controls = _control_array(points)
    if _hypot(_sub(controls[0], controls[-1])) <= 1.0e-6:
        hits = [hit for hit in hits if _hypot(_sub(hit, controls[0])) > 1.0e-4]
    return _unique_points(hits)


def segmentize_catmull_rom(
    a: float, b: float, c: float, d: float, n: int = 100
) -> Sequence[float]:
    """a and b are the control points and c and d are
    start and end points respectively,
    n is the number of segments to generate.

    Args:
        a (float): First control point.
        b (float): Second control point.
        c (float): Start point.
        d (float): End point.
        n (int, optional): Number of segments to generate. Defaults to 100.

    Returns:
        Sequence[float]: List of points representing the segments.

    Examples:
        >>> import simetri.graphics as sg
        >>> pts = sg.segmentize_catmull_rom((0, 0), (40, 0), (80, 40), (60, 0), n=2)
        >>> [[round(float(c), 6) for c in q[:2]] for q in pts]
        [[40.0, 0.0], [63.75, 22.5], [80.0, 40.0]]
    """
    a = array(a[:2], dtype=float)
    b = array(b[:2], dtype=float)
    c = array(c[:2], dtype=float)
    d = array(d[:2], dtype=float)

    t = 0
    dt = 1.0 / n
    points = []
    term1 = 2 * b
    term2 = -a + c
    term3 = 2 * a - 5 * b + 4 * c - d
    term4 = -a + 3 * b - 3 * c + d

    for _ in range(n + 1):
        q = 0.5 * (term1 + term2 * t + term3 * t**2 + term4 * t**3)
        points.append([q[0], q[1]])
        t += dt
    return points


def _spline_segment_points(total_points: int, n_segments: int) -> list[int]:
    """Return per-segment sample counts for a composite Bezier curve.

    Each segment needs at least 5 samples to match the single-segment Bezier
    utilities. Adjacent segments share an endpoint, so a spline with
    ``n_segments`` requires at least ``4 * n_segments + 1`` total unique points.
    """
    min_points = 4 * n_segments + 1
    if total_points < min_points:
        raise ValueError(
            f"n_points must be at least {min_points} for a spline with "
            f"{n_segments} segment(s)."
        )

    per_segment = [5] * n_segments
    extra = total_points - min_points
    for i in range(extra):
        per_segment[i % n_segments] += 1

    return per_segment


def _join_spline_vertices(curves: Sequence[Bezier]) -> list[PointType]:
    """Return the sampled spline vertices without duplicated join points."""
    vertices = []
    for i, curve in enumerate(curves):
        curve_vertices = [tuple(map(float, p[:2])) for p in curve.vertices]
        if i:
            curve_vertices = curve_vertices[1:]
        vertices.extend(curve_vertices)
    return vertices


class _BezierSpline(Shape):
    """Shared implementation for composite quadratic and cubic Bezier curves."""

    _stride = 0
    _segment_size = 0
    _subtype = Types.SHAPE
    _name = "Spline"

    def __init__(
        self,
        controls: Sequence[PointType],
        xform_matrix: array = None,
        n_points: int | None = None,
        **kwargs: object,
    ) -> None:
        self._set_geometry(controls, n_points)
        super().__init__(
            self._vertices,
            subtype=self._subtype,
            xform_matrix=xform_matrix,
            **kwargs,
        )

    def __repr__(self) -> str:
        """Return a concise spline representation."""
        if len(self.primary_points) == 0:
            return f"{self._name}()"
        if len(self.primary_points) < 4:
            return f"{self._name}({self.vertices})"
        return f"{self._name}([{self.vertices[0]}, ..., {self.vertices[-1]}])"

    @property
    def control_points(self) -> Sequence[PointType]:
        """Return the control points used to define the spline."""
        return self.__dict__["control_points"]

    @control_points.setter
    def control_points(self, new_control_points: Sequence[PointType]) -> None:
        """Set new control points and rebuild the spline geometry."""
        self._set_geometry(new_control_points, self.n_points)
        self[:] = self._vertices
        self.subtype = self._subtype

    def copy(self, **kwargs: object) -> Shape:
        """Return a copy of the spline."""
        copy_ = type(self)(
            self.control_points,
            xform_matrix=self.xform_matrix,
            n_points=self.n_points,
        )
        for k, v in kwargs.items():
            setattr(copy_, k, v)
        return copy_

    def point(self, t: float) -> list[float]:
        """Return the point on the spline at parameter ``t``."""
        curve, local_t = self._curve_at(t)
        return curve.point(local_t)

    def derivative(self, t: float) -> list[float]:
        """Return the derivative of the spline at parameter ``t``."""
        curve, local_t = self._curve_at(t)
        return curve.derivative(local_t)

    def normal(self, t: float) -> list[float]:
        """Return the unit normal of the spline at parameter ``t``."""
        curve, local_t = self._curve_at(t)
        return curve.normal(local_t)

    def tangent(self, t: float) -> list[float]:
        """Return the unit tangent of the spline at parameter ``t``."""
        curve, local_t = self._curve_at(t)
        return curve.tangent(local_t)

    def _curve_at(self, t: float) -> tuple[Bezier, float]:
        if not 0 <= t <= 1:
            raise ValueError("t must satisfy 0 <= t <= 1.")

        if t == 1:
            return self.curves[-1], 1.0

        scaled = t * self.n_segments
        index = int(scaled)
        local_t = scaled - index
        return self.curves[index], local_t

    def _set_geometry(
        self, controls: Sequence[PointType], n_points: int | None
    ) -> None:
        self._validate_controls(controls)
        n_segments = (len(controls) - 1) // self._stride
        if n_points is None:
            total_points = (
                runtime_defaults["n_bezier_points"] - 1
            ) * n_segments + 1
        else:
            total_points = n_points

        points_per_segment = _spline_segment_points(total_points, n_segments)
        curves = []
        for i, seg_points in enumerate(points_per_segment):
            start = i * self._stride
            stop = start + self._segment_size
            curves.append(Bezier(controls[start:stop], n_points=seg_points))

        self.__dict__["control_points"] = controls
        self.__dict__["curves"] = curves
        self.__dict__["n_segments"] = n_segments
        self.__dict__["n_points"] = total_points
        self._vertices = _join_spline_vertices(curves)

    def _validate_controls(self, controls: Sequence[PointType]) -> None:
        if len(controls) < self._segment_size:
            raise ValueError(
                f"{self._name} requires at least {self._segment_size} control points."
            )
        if (len(controls) - 1) % self._stride != 0:
            raise ValueError(
                f"Invalid number of control points for {self._name.lower()}."
            )


class SplineQ(_BezierSpline):
    """A composite quadratic Bezier curve.

    The control points are given in walk order as
    ``[v0, c0, v1, c1, v2, ...]`` so each additional segment contributes a
    control point and an endpoint.

    Examples:
        >>> import simetri.graphics as sg
        >>> spline = sg.SplineQ([(0, 0), (20, 40), (40, 0), (60, -40), (80, 0)], n_points=9)
        >>> len(spline.curves)
        2
        >>> [round(c, 6) for c in spline.point(0.0)]
        [0.0, 0.0]
        >>> [round(c, 6) for c in spline.point(1.0)]
        [80.0, 0.0]
    """

    _stride = 2
    _segment_size = 3
    _subtype = Types.Q_BEZIER
    _name = "SplineQ"


class SplineC(_BezierSpline):
    """A composite cubic Bezier curve.

    The control points are given in walk order as
    ``[v0, c0, c1, v1, c2, c3, v2, ...]`` so each additional segment
    contributes two control points and an endpoint.

    Examples:
        >>> import simetri.graphics as sg
        >>> spline = sg.SplineC(
        ...     [(0, 0), (20, 40), (40, 40), (60, 0), (80, -40), (100, -40), (120, 0)],
        ...     n_points=9,
        ... )
        >>> len(spline.curves)
        2
        >>> [round(c, 6) for c in spline.point(0.0)]
        [0.0, 0.0]
        >>> [round(c, 6) for c in spline.point(1.0)]
        [120.0, 0.0]
    """

    _stride = 3
    _segment_size = 4
    _subtype = Types.BEZIER
    _name = "SplineC"

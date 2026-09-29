"""Ellipses, circular/elliptic arcs, and related intersection helpers."""

from __future__ import annotations

import cmath
from collections.abc import Sequence
from typing import Any
from copy import deepcopy
from math import atan2, ceil, cos, isclose, pi, sin, sqrt

import numpy as np
from numpy.typing import NDArray

from ...base.all_enums import (
    InPlace,
    TransformationType,
    Types,
    WarningType,
)
from ...base.common import PointType, alias_argument
from ...config.settings import issue_warning, runtime_defaults
from ...group.batch import Group
from ...helpers.utilities import solve_quadratic_eq
from ...render.style_map import shape_style_map
from ...shapes.points import Points
from ...shapes.shape import Shape, custom_attributes
from ..affine import rotate_point, rotation_matrix
from ..geometry import positive_angle
from ..homogenize import homogenize
from ..points.point_utils import distance
from ..segments.line_utils import line_angle


_ARC_ATTRIBS = frozenset(
    (
        "center",
        "clockwise",
        "radius_x",
        "radius_y",
        "span_angle",
        "start_angle",
    )
)


class Arc(Shape):
    """A circular or elliptic arc.

    Defined by center, ``radius_x``, optional ``radius_y``, start angle, and
    either ``span_angle`` or ``end_angle``. If ``radius_y`` is omitted, the
    arc is circular. ``clockwise=True`` draws clockwise. A negative
    ``span_angle`` also draws clockwise.

    Attributes:
        start_angle: Starting angle in radians.
        span_angle: Sweep magnitude in radians (non-negative after construction).
        clockwise: True when the arc is drawn clockwise.
        n_points: Number of sampled points.

    Examples:

        >>> import simetri.graphics as sg
        >>> arc = sg.Arc((0, 0), 40, start_angle=0, span_angle=sg.pi / 2)
        >>> [[round(float(c), 6) or 0.0 for c in q[:2]] for q in arc.vertices]
        [[40.0, 0.0], [39.39231, 6.945927], [37.587705, 13.680806], [34.641016, 20.0], [30.641778, 25.711504], [25.711504, 30.641778], [20.0, 34.641016], [13.680806, 37.587705], [6.945927, 39.39231], [0.0, 40.0]]
        >>> cw = sg.Arc((0, 0), 40, start_angle=0, span_angle=-sg.pi / 2)
        >>> cw.clockwise
        True
        >>> round(cw.span_angle, 6)
        1.570796
    """

    @alias_argument({"radius_x": "rx", "radius_y": "ry"})
    def __init__(
        self,
        center: PointType,
        radius_x: float,
        radius_y: float | None = None,
        start_angle: float = 0,
        span_angle: float | None = None,
        rot_angle: float = 0,
        n_points: int | None = None,
        xform_matrix: NDArray | None = None,
        *,
        end_angle: float | None = None,
        clockwise: bool = False,
        **kwargs: object,
    ) -> None:
        """Create a circular or elliptic arc.

        Args:
            center: Arc center ``(x, y)``.
            radius_x: Semi-axis along x (before rotation).
            radius_y: Semi-axis along y; defaults to ``radius_x`` (circle).
            start_angle: Starting angle in radians. Defaults to 0.
            span_angle: Sweep in radians. A negative value draws clockwise.
                Mutually exclusive with ``end_angle``. At least one of
                ``span_angle`` or ``end_angle`` is required.
            rot_angle: Extra rotation about the center. Defaults to 0.
            n_points: Sample count; defaults from settings scaled by span.
            xform_matrix: Optional transformation matrix.
            end_angle: Ending angle in radians. Mutually exclusive with
                ``span_angle``.
            clockwise: If True, the arc is drawn clockwise. Defaults to False.
            **kwargs: Additional keyword arguments passed to ``Shape``.
        """
        if radius_y is None:
            radius_y = radius_x
        signed_span = resolve_arc_sweep(
            start_angle,
            span_angle,
            end_angle,
            clockwise,
        )
        clockwise = signed_span < 0
        span_angle = abs(signed_span)
        if n_points is None:
            n = runtime_defaults["n_arc_points"]
            n_points = ceil(n * span_angle / (2 * pi))

        vertices = elliptic_arc_points(
            center,
            radius_x,
            radius_y,
            start_angle,
            span_angle,
            n_points=n_points,
            clockwise=clockwise,
        )
        if rot_angle:
            rot_matrix = rotation_matrix(rot_angle, center)
            if xform_matrix is not None:
                xform_matrix = np.dot(rot_matrix, xform_matrix)
            else:
                xform_matrix = rot_matrix

        super().__init__(vertices, xform_matrix=xform_matrix, **kwargs)
        self.subtype = Types.ARC
        self.n_points = n_points
        self.__dict__["start_angle"] = start_angle
        self.__dict__["span_angle"] = span_angle
        self.__dict__["clockwise"] = clockwise
        cx, cy = center[:2]
        self._c = [cx, cy, 1]
        _a = [radius_x, 0, 1]
        _b = [0, radius_y, 1]
        self._orig_triangle = [self._c[:], _a, _b]

    def __setattr__(self, name: str, value: object) -> None:
        """Set an attribute of the arc.

        Args:
            name (str): The name of the attribute.
            value (Any): The value of the attribute.
        """
        if name == "center":
            diff = np.array(value[:2]) - np.array(self.center[:2])
            self.translate(diff[0], diff[1], reps=0)
        elif name == "radius_x":
            c, a, _ = self._orig_triangle @ self.xform_matrix
            cur_radius = distance(c, a)
            ratio = value / cur_radius
            self.scale(ratio, 1, about=self.center)
        elif name == "radius_y":
            c, _, b = self._orig_triangle @ self.xform_matrix
            cur_radius = distance(c, b)
            ratio = value / cur_radius
            self.scale(1, ratio, about=self.center)
        elif name == "start_angle":
            center, a, b = self._orig_triangle @ self.xform_matrix
            a = distance(center, a)
            b = distance(center, b)
            points = elliptic_arc_points(
                center,
                a,
                b,
                value,
                self.span_angle,
                n_points=self.n_points,
                clockwise=self.clockwise,
            )
            self.primary_points = Points(points)
            self.__dict__["start_angle"] = value
        elif name == "span_angle":
            center, a, b = self._orig_triangle @ self.xform_matrix
            a = distance(center, a)
            b = distance(center, b)
            points = elliptic_arc_points(
                center,
                a,
                b,
                self.start_angle,
                value,
                n_points=self.n_points,
                clockwise=self.clockwise,
            )
            self.primary_points = Points(points)
            self.__dict__["span_angle"] = value
        elif name == "clockwise":
            center, a, b = self._orig_triangle @ self.xform_matrix
            a = distance(center, a)
            b = distance(center, b)
            points = elliptic_arc_points(
                center,
                a,
                b,
                self.start_angle,
                self.span_angle,
                n_points=self.n_points,
                clockwise=bool(value),
            )
            self.primary_points = Points(points)
            self.__dict__["clockwise"] = bool(value)
        else:
            super().__setattr__(name, value)

    @property
    def center(self) -> PointType:
        """Return the center of the arc.

        Returns:
            PointType: The center of the arc.

        Examples:
            >>> import simetri.graphics as sg
            >>> [round(float(c), 6) or 0.0 for c in sg.Arc((10, 20), 5, span_angle=sg.pi / 2).center[:2]]
            [10.0, 20.0]
        """
        return (self._c @ self.xform_matrix).tolist()[:2]

    @property
    def radius_x(self) -> float:
        """Return the x radius of the arc.

        Returns:
            float: The x radius of the arc.

        Examples:
            >>> import simetri.graphics as sg
            >>> round(sg.Arc((0, 0), 5, 3, span_angle=sg.pi / 2).radius_x, 6)
            5.0
        """
        c, a, _ = self._orig_triangle @ self.xform_matrix
        return distance(a, c)

    @property
    def radius_y(self) -> float:
        """Return the y radius of the arc.

        Returns:
            float: The y radius of the arc.

        Examples:
            >>> import simetri.graphics as sg
            >>> round(sg.Arc((0, 0), 5, 3, span_angle=sg.pi / 2).radius_y, 6)
            3.0
        """
        c, _, b = self._orig_triangle @ self.xform_matrix
        return distance(b, c)

    def copy(self, **kwargs: object) -> Arc:
        """Return a copy of the arc.

        Args:
            **kwargs: Attribute overrides applied to the copy.

        Returns:
            Arc: Copied arc.

        Examples:
            >>> import simetri.graphics as sg
            >>> arc = sg.Arc((0, 0), 10, start_angle=0, span_angle=sg.pi / 2)
            >>> copy = arc.copy()
            >>> copy.center == arc.center and copy.radius_x == arc.radius_x
            True
        """
        center = self.center
        start_angle = self.start_angle
        span_angle = self.span_angle
        radius_x = self.radius_x
        radius_y = self.radius_y

        arc = Arc(
            center,
            radius_x,
            radius_y,
            start_angle,
            span_angle,
            rot_angle=0,
            clockwise=self.clockwise,
        )
        arc.primary_points = self.primary_points.copy()
        arc.xform_matrix = self.xform_matrix.copy()
        arc._orig_triangle = deepcopy(self._orig_triangle)
        arc._c = self._c[:]
        arc.n_points = self.n_points
        arc.copy_style(self)
        # for attrib in shape_style_map:
        #     setattr(arc, attrib, getattr(self, attrib))
        arc.subtype = self.subtype
        custom_attribs = custom_attributes(self)
        arc_attribs = _ARC_ATTRIBS

        for attrib in custom_attribs:
            if attrib not in arc_attribs:
                setattr(arc, attrib, getattr(self, attrib))

        for k, v in kwargs.items():
            setattr(arc, k, v)

        return arc


class Ellipse(Shape):
    """Ellipse defined by width, height, and optional center.

    Size comes first so ``Ellipse(80, 40)`` is an 80×40 ellipse at the origin.

    Attributes:
        a: Semi-axis along width (``width / 2``).
        b: Semi-axis along height (``height / 2``).
        center: Ellipse center.
        width: Full width.
        height: Full height.
        angle: Rotation angle in radians.

    Examples:

        >>> import simetri.graphics as sg
        >>> ell = sg.Ellipse(80, 40)
        >>> ell.width, ell.height
        (80.0, 40.0)
    """

    def __init__(
        self,
        width: float | None = None,
        height: float | None = None,
        center: PointType = (0, 0),
        angle: float = 0,
        xform_matrix: NDArray | None = None,
        **kwargs: object,
    ) -> None:
        """Create an ellipse.

        Args:
            width: Full width. ``None`` uses ``runtime_defaults["ellipse_width_height"]``.
            height: Full height. ``None`` uses ``runtime_defaults["ellipse_width_height"]``.
            center: Ellipse center ``(x, y)``. Defaults to ``(0, 0)``.
            angle: Rotation angle in radians. Defaults to 0.
            xform_matrix: Optional transformation matrix.
            **kwargs: Additional keyword arguments passed to ``Shape``.
        """
        if width is None or height is None:
            default_width, default_height = runtime_defaults["ellipse_width_height"]
            if width is None:
                width = default_width
            if height is None:
                height = default_height
        n_points = runtime_defaults["n_ellipse_points"]
        vertices = [
            tuple(p)
            for p in ellipse_points(center, width / 2, height / 2, 0, n_points)
        ]
        if angle:
            rot_matrix = rotation_matrix(angle, center)
            if xform_matrix is not None:
                xform_matrix = rot_matrix @ xform_matrix
            else:
                xform_matrix = rot_matrix
        super().__init__(
            vertices, closed=True, xform_matrix=xform_matrix, **kwargs
        )
        a = width / 2
        b = height / 2
        self.a = a
        self.b = b
        self.center = center
        self.smooth = True
        self.closed = True
        self.subtype = Types.ELLIPSE

    def __setattr__(self, name: str, value: object) -> None:
        """Set an attribute of the ellipse.

        ``width`` and ``height`` scale about the center.

        Args:
            name: The name of the attribute.
            value: The value of the attribute.
        """
        if name == "width" and "a" in self.__dict__:
            current = 2 * self.__dict__["a"]
            super().__setattr__("a", value / 2)
            if current != 0:
                self.scale(value / current, 1, about=self.center, reps=0)
        elif name == "height" and "b" in self.__dict__:
            current = 2 * self.__dict__["b"]
            super().__setattr__("b", value / 2)
            if current != 0:
                self.scale(1, value / current, about=self.center, reps=0)
        else:
            super().__setattr__(name, value)

    @property
    def width(self) -> float:
        """Return the full width of the ellipse (twice the x semi-axis).

        Returns:
            float: ``2 * a``.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Ellipse(80, 40).width
            80.0
        """
        return 2 * self.a

    @property
    def height(self) -> float:
        """Return the full height of the ellipse (twice the y semi-axis).

        Returns:
            float: ``2 * b``.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Ellipse(80, 40).height
            40.0
        """
        return 2 * self.b

    @property
    def closed(self) -> bool:
        """Return True ellipse is always closed.

        Returns:
            bool: Always returns True.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Ellipse(80, 40).closed
            True
        """
        return True

    @closed.setter
    def closed(self, value: bool) -> None:
        """Ellipses are always closed; assignment is ignored.

        Args:
            value: Ignored closed flag.

        Examples:
            >>> import simetri.graphics as sg
            >>> ell = sg.Ellipse(80, 40)
            >>> ell.closed = False
            >>> ell.closed
            True
        """

    def _update(
        self,
        xform_matrix: np.array,
        reps: int = 0,
        take: slice | None = None,
        incr: float
        | tuple[float, float]
        | tuple[callable, Any]
        | tuple[InPlace, Any]
        | NDArray
        | Sequence[Sequence[float]]
        | None = None,
        dyn_ref: bool | None = None,
        merge: bool = False,
        xform_type: TransformationType = None,
    ) -> Group:
        """Used internally. Update the shape with a transformation matrix.

        Args:
            xform_matrix (array): The transformation matrix.
            reps (int, optional): The number of repetitions, defaults to 0.
            take: Not supported; must be ``None``.
            incr: Increment applied between repetitions when ``reps > 0``.
            merge: If True and ``reps > 0``, merge the copies.
            xform_type: Transform kind used with ``incr``.

        Returns:
            Group: The updated shape or a group of shapes.

        Raises:
            ValueError: If ``take`` is set, or if ``dyn_ref`` is used.
        """
        if dyn_ref:
            raise ValueError(
                "Ellipse does not support dynamic references. Only Shape and "
                "Group resolve dyn_ref."
            )
        if take is not None:
            raise ValueError(
                "Ellipse._update does not support take=; transform the whole ellipse."
            )
        if reps == 0:
            center = list(self.center[:2]) + [1]
            start = list(self.vertices[0][:2]) + [1]
            end = list(self.vertices[-1][:2]) + [1]
            points = [center, start, end]
            center2, start2, end2 = np.dot(points, xform_matrix).tolist()
            self.center = center2[:2]
            self.start_point = start2[:2]
            self.end_point = end2[:2]
            self.start_angle = line_angle(center2, start2)

        return super()._update(
            xform_matrix,
            reps=reps,
            incr=incr,
            merge=merge,
            xform_type=xform_type,
        )

    def copy(self, **kwargs: object) -> Ellipse:
        """Return a copy of the ellipse.

        Returns:
            Ellipse: A copy of the ellipse.

        Examples:
            >>> import simetri.graphics as sg
            >>> ell = sg.Ellipse(80, 40)
            >>> copy = ell.copy()
            >>> copy.width, copy.height
            (80.0, 40.0)
        """
        ellipse = super().copy()
        for key, value in kwargs.items():
            setattr(ellipse, key, value)

        return ellipse


def ellipse_tangent(
    a: float, b: float, x: float, y: float, abs_tol: float = 0.001
) -> float | bool:
    """Calculates the angle of the tangent line to an ellipse at the point (x, y).

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.
        abs_tol (float, optional): Tolerance for point on ellipse check. Defaults to .001.

    Returns:
        float: Angle of the tangent line in radians.

    Examples:
        >>> import simetri.graphics as sg
        >>> round(sg.ellipse_tangent(2.0, 1.0, 2.0, 0.0), 6)
        1.570796
        >>> sg.ellipse_tangent(2.0, 1.0, 0.0, 0.0)
        False
    """
    if abs((x**2 / a**2) + (y**2 / b**2) - 1) >= abs_tol:
        res = False
    else:
        res = atan2((b**2 * x), -(a**2 * y))

    return res


def r_central(a: float, b: float, theta: float) -> float:
    """Return the radius (distance between the center and the intersection point)
    of the ellipse at the given angle.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        theta (float): Angle in radians.

    Returns:
        float: Radius at the given angle.

    Examples:
        >>> import simetri.graphics as sg
        >>> round(sg.r_central(2.0, 1.0, 0.0), 6)
        2.0
    """
    return (a * b) / sqrt((b * cos(theta)) ** 2 + (a * sin(theta)) ** 2)


def ellipse_line_intersection(
    a: float, b: float, point: PointType
) -> list[PointType]:
    """Return the intersection points of an ellipse and a line segment
    connecting the given point to the ellipse center at (0, 0).

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        point (tuple): PointType coordinates (x, y).

    Returns:
        list: Intersection points.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.ellipse_line_intersection(2.0, 1.0, (4, 0))
        [(2.0, 0.0), (-2.0, -0.0)]
    """
    # adapted from http:# mathworld.wolfram.com/Ellipse-LineIntersection.html
    # a, b is the ellipse width/2 and height/2 and (x_0, y_0) is the point

    x_0, y_0 = point[:2]
    x = ((a * b) / (sqrt(a**2 * y_0**2 + b**2 * x_0**2))) * x_0
    y = ((a * b) / (sqrt(a**2 * y_0**2 + b**2 * x_0**2))) * y_0

    return [(x, y), (-x, -y)]


def resolve_arc_sweep(
    start_angle: float,
    span_angle: float | None,
    end_angle: float | None,
    clockwise: bool = False,
    *,
    default_span: float | None = None,
) -> float:
    """Return a signed sweep for arc point generation.

    Positive sweep is counter-clockwise. Negative sweep is clockwise.
    Pass either ``span_angle`` or ``end_angle``, not both. A negative
    ``span_angle`` is a clockwise sweep. ``clockwise=True`` also makes
    the sweep clockwise, using ``abs(span_angle)``. Combining
    ``clockwise=True`` with a negative ``span_angle`` still draws
    clockwise and issues ``WarningType.geometry.clockwise_negative_span``.

    Args:
        start_angle: Starting angle in radians.
        span_angle: Sweep in radians, or ``None``. Negative is clockwise.
        end_angle: Ending angle in radians, or ``None``.
        clockwise: If True, the arc is drawn clockwise. Defaults to False.
        default_span: Used when both ``span_angle`` and ``end_angle`` are
            ``None``. If this is also ``None``, a ``TypeError`` is raised.

    Returns:
        float: Signed sweep in radians.

    Raises:
        TypeError: If both ``span_angle`` and ``end_angle`` are given, or
            if neither is given and ``default_span`` is ``None``.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.geom.nonlinear.ellipse import resolve_arc_sweep
        >>> round(resolve_arc_sweep(0, sg.pi / 2, None), 6)
        1.570796
        >>> round(resolve_arc_sweep(0, sg.pi / 2, None, clockwise=True), 6)
        -1.570796
        >>> round(resolve_arc_sweep(0, -sg.pi / 2, None), 6)
        -1.570796
        >>> sg.pause_warnings()
        >>> round(resolve_arc_sweep(0, -sg.pi / 2, None, clockwise=True), 6)
        -1.570796
        >>> sg.resume_warnings()
        >>> round(resolve_arc_sweep(0, None, sg.pi / 2), 6)
        1.570796
        >>> resolve_arc_sweep(0, sg.pi / 2, sg.pi / 2)
        Traceback (most recent call last):
            ...
        TypeError: Received both 'span_angle' and 'end_angle'!
    """
    if span_angle is not None and end_angle is not None:
        raise TypeError("Received both 'span_angle' and 'end_angle'!")
    if span_angle is None and end_angle is None:
        if default_span is None:
            raise TypeError("Arc requires 'span_angle' or 'end_angle'.")
        span_angle = default_span
    if end_angle is not None:
        two_pi = 2 * pi
        if clockwise:
            sweep = (start_angle - end_angle) % two_pi
        else:
            sweep = (end_angle - start_angle) % two_pi
        if sweep == 0:
            sweep = two_pi
        if clockwise:
            return -sweep
        return sweep
    if clockwise:
        if span_angle < 0:
            issue_warning(
                "clockwise=True with a negative span_angle is redundant; "
                "the arc is still drawn clockwise.",
                warning_type=WarningType.geometry.clockwise_negative_span,
            )
        return -abs(span_angle)
    return span_angle


def _elliptic_arc_points_from_signed(
    center: PointType,
    radius_x: float,
    radius_y: float,
    start_angle: float,
    span_angle: float,
    n_points: int | None,
) -> NDArray:
    rx = radius_x
    ry = radius_y
    if n_points is None:
        n = runtime_defaults["n_arc_points"]
        n_points = ceil(n * abs(span_angle) / (2 * pi))
    start_angle = positive_angle(start_angle)
    clockwise = span_angle < 0
    if clockwise:
        if start_angle + span_angle < 0:
            end_angle = positive_angle(start_angle + span_angle)
            t0 = get_ellipse_t_for_angle(end_angle, rx, ry)
            t1 = get_ellipse_t_for_angle(2 * pi, rx, ry)
            t = np.linspace(t0, t1, n_points)
            x = center[0] + rx * np.cos(t)
            y = center[1] + ry * np.sin(t)
            slice1 = np.column_stack((x, y))
            t0 = get_ellipse_t_for_angle(0, rx, ry)
            t1 = get_ellipse_t_for_angle(start_angle, rx, ry)
            t = np.linspace(t0, t1, n_points)
            x = center[0] + rx * np.cos(t)
            y = center[1] + ry * np.sin(t)
            slice2 = np.column_stack((x, y))
            res = np.flip(np.concatenate((slice1, slice2)), axis=0)
        else:
            end_angle = start_angle + span_angle
            t0 = get_ellipse_t_for_angle(end_angle, rx, ry)
            t1 = get_ellipse_t_for_angle(start_angle, rx, ry)
            t = np.linspace(t0, t1, n_points)
            x = center[0] + rx * np.cos(t)
            y = center[1] + ry * np.sin(t)
            res = np.flip(np.column_stack((x, y)), axis=0)

    else:
        if start_angle + span_angle > 2 * pi:
            t0 = get_ellipse_t_for_angle(start_angle, rx, ry)
            t1 = get_ellipse_t_for_angle(2 * pi, rx, ry)
            t = np.linspace(t0, t1, n_points)
            x = center[0] + rx * np.cos(t)
            y = center[1] + ry * np.sin(t)
            slice1 = np.column_stack((x, y))

            t0 = get_ellipse_t_for_angle(0, rx, ry)
            t1 = get_ellipse_t_for_angle(
                start_angle + span_angle - 2 * pi, rx, ry
            )
            t = np.linspace(t0, t1, n_points)
            x = center[0] + rx * np.cos(t)
            y = center[1] + ry * np.sin(t)
            slice2 = np.column_stack((x, y))

            res = np.concatenate((slice1, slice2))
        else:
            end_angle = start_angle + span_angle
            t0 = get_ellipse_t_for_angle(start_angle, rx, ry)
            t1 = get_ellipse_t_for_angle(end_angle, rx, ry)
            t = np.linspace(t0, t1, n_points)
            x = center[0] + rx * np.cos(t)
            y = center[1] + ry * np.sin(t)
            res = np.column_stack((x, y))

    return res


@alias_argument({"radius_x": "rx", "radius_y": "ry"})
def elliptic_arc_points(
    center: PointType,
    radius_x: float,
    radius_y: float | None = None,
    start_angle: float = 0,
    span_angle: float | None = None,
    n_points: int | None = None,
    *,
    end_angle: float | None = None,
    clockwise: bool = False,
) -> NDArray:
    """Generate points on an elliptic arc.
    These are generated from the parametric equations of the ellipse.
    They are not evenly spaced.

    Args:
        center: Center of the ellipse ``(x, y)``.
        radius_x: Semi-axis along x.
        radius_y: Semi-axis along y; defaults to ``radius_x``.
        start_angle: Starting angle in radians. Defaults to 0.
        span_angle: Sweep in radians. A negative value draws clockwise.
            Mutually exclusive with ``end_angle``. At least one of
            ``span_angle`` or ``end_angle`` is required.
        n_points: Number of points to generate.
        end_angle: Ending angle in radians. Mutually exclusive with
            ``span_angle``.
        clockwise: If True, the arc is drawn clockwise. Defaults to False.

    Returns:
        numpy.ndarray: Array of (x, y) coordinates of the ellipse points.

    Examples:
        >>> import simetri.graphics as sg
        >>> pts = sg.elliptic_arc_points((0, 0), 2, 1, 0, sg.pi / 2, n_points=3)
        >>> [[round(float(c), 6) or 0.0 for c in q[:2]] for q in pts]
        [[2.0, 0.0], [1.414214, 0.707107], [0.0, 1.0]]
        >>> pts = sg.elliptic_arc_points(
        ...     (0, 0), 2, 1, 0, end_angle=sg.pi / 2, n_points=3
        ... )
        >>> [[round(float(c), 6) or 0.0 for c in q[:2]] for q in pts]
        [[2.0, 0.0], [1.414214, 0.707107], [0.0, 1.0]]
        >>> pts = sg.elliptic_arc_points((0, 0), 2, 1, sg.pi / 2, -sg.pi / 2, n_points=3)
        >>> [[round(float(c), 6) or 0.0 for c in q[:2]] for q in pts]
        [[0.0, 1.0], [1.414214, 0.707107], [2.0, 0.0]]
    """
    if radius_y is None:
        radius_y = radius_x
    signed_span = resolve_arc_sweep(
        start_angle, span_angle, end_angle, clockwise
    )
    return _elliptic_arc_points_from_signed(
        center, radius_x, radius_y, start_angle, signed_span, n_points
    )


def ellipse_points(
    center: PointType,
    a: float,
    b: float,
    angle: float,
    n_points: int | None = None,
) -> NDArray:
    """Generate points on an ellipse.
    These are generated from the parametric equations of the ellipse.
    They are not evenly spaced.

    Args:
        center (tuple): (x, y) coordinates of the ellipse center.
        a (float): Length of the semi-major axis.
        b (float): Length of the semi-minor axis.
        angle (float): Rotation angle of the ellipse.
        n_points (int): Number of points to generate.

    Returns:
        numpy.ndarray: Array of (x, y) coordinates of the ellipse points.

    Examples:
        >>> import simetri.graphics as sg
        >>> pts = sg.ellipse_points((0, 0), 2, 1, 0, n_points=5)
        >>> [[round(float(c), 6) or 0.0 for c in q[:2]] for q in pts]
        [[2.0, 0.0], [0.0, 1.0], [-2.0, 0.0], [0.0, -1.0], [2.0, 0.0]]
    """
    if n_points is None:
        n_points = runtime_defaults["n_ellipse_points"]

    t = np.linspace(0, 2 * pi, n_points)
    x = center[0] + a * np.cos(t)
    y = center[1] + b * np.sin(t)

    points = homogenize(np.column_stack((x, y))) @ rotation_matrix(
        angle, center
    )

    return points[:, :2].tolist()


def elliptic_arclength(t_0: float, t_1: float, a: float, b: float) -> float:
    """Return the arclength of an ellipse between the given parametric angles.
    The ellipse has semi-major axis a and semi-minor axis b.

    Args:
        t_0 (float): Starting parametric angle.
        t_1 (float): Ending parametric angle.
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.

    Returns:
        float: Arclength of the ellipse.

    Examples:
        >>> import simetri.graphics as sg
        >>> round(sg.elliptic_arclength(0, sg.pi / 2, 2.0, 1.0), 6)
        2.422112
    """
    from scipy.special import ellipeinc  # this takes too long to import

    m = 1 - (b / a) ** 2
    t1 = ellipeinc(t_1 - 0.5 * pi, m)
    t0 = ellipeinc(t_0 - 0.5 * pi, m)
    return a * (t1 - t0)


def central_to_parametric_angle(a: float, b: float, phi: float) -> float:
    """
    Converts a central angle to a parametric angle on an ellipse.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        phi (float): Angle of the line intersecting the center and the point.

    Returns:
        float: Parametric angle (in radians).

    Examples:
        >>> import simetri.graphics as sg
        >>> round(sg.central_to_parametric_angle(2.0, 1.0, 0.0), 6)
        0.0
        >>> round(sg.central_to_parametric_angle(2.0, 1.0, sg.pi / 2), 6)
        1.570796
    """
    t = atan2((a / b) * sin(phi), cos(phi))
    if t < 0:
        t += 2 * pi

    return t


def parametric_to_central_angle(a: float, b: float, t: float) -> float:
    """
    Converts a parametric angle on an ellipse to a central angle.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        t (float): Parametric angle (in radians).

    Returns:
        float: Angle of the line intersecting the center and the point.

    Examples:
        >>> import simetri.graphics as sg
        >>> round(sg.parametric_to_central_angle(2.0, 1.0, 0.0), 6)
        0.0
    """
    phi = atan2((b / a) * sin(t), cos(t))
    if phi < 0:
        phi += 2 * pi

    return phi


def ellipse_point(a: float, b: float, angle: float) -> PointType:
    """Return a point on an ellipse with the given a=width/2, b=height/2, and angle.
    angle is the central-angle and in radians.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        angle (float): Central angle in radians.

    Returns:
        tuple: Coordinates of the point on the ellipse.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.ellipse_point(2.0, 1.0, 0.0)
        (2.0, 0.0)
    """
    r = r_central(a, b, angle)

    return (r * cos(angle), r * sin(angle))


def ellipse_param_point(a: float, b: float, t: float) -> PointType:
    """Return a point on an ellipse with the given a=width/2, b=height/2, and parametric angle.
    t is the parametric angle and in radians.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        t (float): Parametric angle in radians.

    Returns:
        tuple: Coordinates of the point on the ellipse.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.ellipse_param_point(2.0, 1.0, 0.0)
        (2.0, 0.0)
    """
    return (a * cos(t), b * sin(t))


def get_ellipse_t_for_angle(angle: float, a: float, b: float) -> float:
    """
    Calculates the parameter t for a given angle on an ellipse.

    Args:
        angle (float): The angle in radians.
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.

    Returns:
        float: The parameter t.

    Examples:
        >>> import simetri.graphics as sg
        >>> round(sg.get_ellipse_t_for_angle(0.0, 2.0, 1.0), 6)
        0.0
    """
    t = atan2(a * sin(angle), b * cos(angle))
    if t < 0:
        t += 2 * pi
    return t


def ellipse_central_angle(t: float, a: float, b: float) -> float:
    """
    Calculates the central angle of an ellipse for a given parameter t.

    Args:
        t (float): The parameter value.
        a (float): The semi-major axis of the ellipse.
        b (float): The semi-minor axis of the ellipse.

    Returns:
        float: The central angle in radians.

    Examples:
        >>> import simetri.graphics as sg
        >>> round(sg.ellipse_central_angle(0.0, 2.0, 1.0), 6)
        0.0
    """
    theta = atan2(a * sin(t), b * cos(t))

    return theta


def ellipse_intersection(
    x1: float,
    y1: float,
    a: float,
    b: float,
    phi: float,
    x2: float,
    y2: float,
    c: float,
    d: float,
    phi2: float,
) -> list[PointType]:
    """Calculate the intersection points of two ellipses.
    The ellipses are defined by their center, radii, and rotation angle.

    Args:
        x1: Center x of the first ellipse.
        y1: Center y of the first ellipse.
        a: Semi-axis along x of the first ellipse (before rotation).
        b: Semi-axis along y of the first ellipse (before rotation).
        phi: Rotation of the first ellipse in radians.
        x2: Center x of the second ellipse.
        y2: Center y of the second ellipse.
        c: Semi-axis along x of the second ellipse (before rotation).
        d: Semi-axis along y of the second ellipse (before rotation).
        phi2: Rotation of the second ellipse in radians.

    Returns:
        list[PointType]: Intersection points.

    Examples:
        >>> import simetri.graphics as sg
        >>> pts = sg.ellipse_intersection(0, 0, 2, 2, 0, 3, 0, 2, 2, 0)
        >>> sorted((round(float(p[0]), 6), round(float(p[1]), 6)) for p in pts)
        [(1.5, -1.322876), (1.5, 1.322876)]
    """
    # Taken from https:# github.com/VoyakaGOD/intersection-of-two-ellipses/blob/master/geometry.js
    phi0 = phi2 - phi

    p0 = rotate_point(((x2 - x1), (y2 - y1)), -phi)

    # cos(t1) = A + Bcos(t2) + Csin(t2)
    # sin(t1) = D + Ecos(t2) + Fsin(t2)
    A = p0[0] / a
    B = c * cos(phi0) / a
    C = -d * sin(phi0) / a
    D = p0[1] / b
    E = c * sin(phi0) / b
    F = d * cos(phi0) / b

    # Gx^2 + Hx + Ixy + Jy + K = 0, x = cos(t2), y = sin(t2)
    G = B * B + E * E - C * C - F * F
    H = 2 * (A * B + D * E)
    coeff_i = 2 * (B * C + E * F)
    J = 2 * (A * C + D * F)
    K = A * A + D * D + C * C + F * F - 1

    roots = []
    L = G * G + coeff_i * coeff_i
    if isclose(L, 0, rel_tol=0, abs_tol=1e-7):
        # Gx^2 + Hx + K = 0 (linear when G is 0)
        if isclose(G, 0, rel_tol=0, abs_tol=1e-7):
            if isclose(H, 0, rel_tol=0, abs_tol=1e-7):
                roots = []
            else:
                roots = [-K / H]
        else:
            roots = solve_quadratic_eq(G, H, K)

    elif isclose(coeff_i, 0, rel_tol=0, abs_tol=1e-7):
        # Gx^2 + Hx + K = 0
        if isclose(G, 0, rel_tol=0, abs_tol=1e-7):
            if isclose(H, 0, rel_tol=0, abs_tol=1e-7):
                roots = []
            else:
                roots = [-K / H]
        else:
            roots = solve_quadratic_eq(G, H, K)

    elif isclose(G, 0, rel_tol=0, abs_tol=1e-7):
        # Hx + Jy + K = 0
        quad_a = H * H + J * J
        if isclose(quad_a, 0, rel_tol=0, abs_tol=1e-7):
            roots = []
        else:
            roots = solve_quadratic_eq(quad_a, 2 * K * H, K * K - J * J)

    else:
        # Lx^4 + Mx^3 + Nx^2 + Ox + P = 0
        M = 2 * (G * H + coeff_i * J)
        N = H * H + 2 * G * K + J * J - coeff_i * coeff_i
        coeff_o = 2 * (K * H - coeff_i * J)
        P = K * K - J * J

        iL = 1 / L
        roots = solve_quartic_equation(M * iL, N * iL, coeff_o * iL, P * iL)

    points = []
    # for(i = 0 i < roots.length i++)
    # for i in range(len(roots)):
    for i, x in enumerate(roots):
        # x = roots[i]
        if isclose(coeff_i * x + J, 0, rel_tol=0, abs_tol=1e-7):
            y = sqrt(1 - x * x)
            points.append((x, y))

            if not isclose(y, 0, rel_tol=0, abs_tol=1e-7):
                points.append((x, -y))
        else:
            y = -(G * x * x + H * x + K) / (coeff_i * x + J)
            if abs(y) < 1e-2:
                y = sqrt(1 - x * x)
                points.append((x, y))
                points.append((x, -y))
            else:
                points.append((x, y))

    for i, pnt in enumerate(points):
        x, y = pnt
        points[i] = (A + B * x + C * y) * a, (D + E * x + F * y) * b

    for i, pnt in enumerate(points):
        x, y = pnt
        x, y = rotate_point((x, y), phi)
        points[i] = (x + x1, y + y1)

    return points


# def IsCloseToZero(num):
#     return abs(num) < 1e-7


def inverse_complex_number(z: complex) -> complex:
    """
    Calculate the inverse of a complex number.

    Args:
        z (complex): The complex number.

    Returns:
        complex: The inverse of the complex number.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.inverse_complex_number(1 + 0j)
        (1-0j)
    """
    a = z.real
    b = z.imag

    if a == 0 and b == 0:
        raise ZeroDivisionError("Cannot calculate the inverse of 0")

    return complex(a / (a**2 + b**2), -b / (a**2 + b**2))


def Re(num: float) -> complex:
    """Return ``num`` as a complex number on the real axis.

    Args:
        num: Real scalar.

    Returns:
        complex: ``complex(num, 0)``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.Re(3)
        (3+0j)
    """
    return complex(num, 0)
    # return Complex(num, 0)


def Im(num: float) -> complex:
    """Return ``num`` as a complex number on the imaginary axis.

    Args:
        num: Real scalar.

    Returns:
        complex: ``complex(0, num)``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.Im(4)
        4j
    """
    return complex(0, num)
    # return Complex(0, num)


def Sqrt(complex_: complex) -> complex:
    """Return the principal square root of a complex number.

    Args:
        complex_: Complex value.

    Returns:
        complex: Principal square root of ``complex_``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.Sqrt(4 + 0j)
        (2+0j)
    """
    return cmath.sqrt(complex_)


def Qbrt(complex_: complex) -> complex:
    """Return a complex cube root used by the ellipse-intersection solver.

    Args:
        complex_: Complex value.

    Returns:
        complex: One cube root of ``complex_``.

    Examples:
        >>> import simetri.graphics as sg
        >>> round(sg.Qbrt(8 + 0j).real, 6)
        2.0
    """
    # Taken from https:# github.com/VoyakaGOD/intersection-of-two-ellipses/blob/master/quartic.js

    angle = cmath.phase(complex_) * 0.33333333333
    # angle = atan2(complex.y, complex.x)  *  0.33333333333
    magnitude = pow(
        complex_.real * complex_.real + complex_.imag * complex_.imag,
        0.16666666666,
    )
    # return Complex(magnitude * cos(angle), magnitude * sin(angle))
    return complex(magnitude * cos(angle), magnitude * sin(angle))


# x^2 + bx + c = 0
def solve_complex_quadratic_equation(
    b: complex, c: complex
) -> list[complex]:
    """Solve ``z^2 + b z + c = 0`` over the complexes.

    Args:
        b: Linear coefficient.
        c: Constant term.

    Returns:
        list[complex]: The two roots.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.solve_complex_quadratic_equation(0, -1)
        [(-1+0j), (1-0j)]
    """
    # Taken from https:# github.com/VoyakaGOD/intersection-of-two-ellipses/blob/master/quartic.js

    # sqrtD = Sqrt(b.MulComplex(b).Sub(c.Mul(4)))
    sqrtD = cmath.sqrt(b * b - (c * 4))
    return [(b + sqrtD) * (-0.5), (b - sqrtD) * (-0.5)]


# x^3 + ax^2 + bx + c = 0
def get_one_cubic_equation_root(
    a: complex, b: complex, c: complex
) -> complex:
    """Return one root of the cubic ``z^3 + a z^2 + b z + c = 0``.

    Args:
        a: ``z^2`` coefficient.
        b: ``z`` coefficient.
        c: Constant term.

    Returns:
        complex: One cubic root.

    Examples:
        >>> import simetri.graphics as sg
        >>> round(sg.get_one_cubic_equation_root(0, 0, -8).real, 6)
        2.0
    """
    # Taken from https:# github.com/VoyakaGOD/intersection-of-two-ellipses/blob/master/quartic.js

    p = b - a * a * 0.33333333333
    q = (2 / 27) * a * a * a - a * b * 0.33333333333 + c
    sqrt_Q = cmath.sqrt(Re(0.03703703703 * p * p * p + 0.25 * q * q))
    A = Qbrt(Re(-0.5 * q) - sqrt_Q)
    if A.real == 0 and A.imag == 0:
        A = Qbrt(Re(-0.5 * q) + sqrt_Q)
    B = inverse_complex_number(A) * (-p * 0.33333333333)

    return (A + B) - (Re(a * 0.33333333333))


# x^4 + ax^3 + bx^2 + cx + d = 0
def solve_quartic_equation(
    a: complex, b: complex, c: complex, d: complex
) -> list[complex]:
    """Solve ``z^4 + a z^3 + b z^2 + c z + d = 0``.

    Args:
        a: ``z^3`` coefficient.
        b: ``z^2`` coefficient.
        c: ``z`` coefficient.
        d: Constant term.

    Returns:
        list[complex]: The four roots.

    Examples:
        >>> import simetri.graphics as sg
        >>> roots = sg.solve_quartic_equation(0, -5, 0, 4)
        >>> sorted(round(r.real, 6) for r in roots if abs(r.imag) < 1e-6)
        [-2.0, -1.0, 1.0, 2.0]
    """
    # Taken from https:# github.com/VoyakaGOD/intersection-of-two-ellipses/blob/master/geometry.js

    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    p = b - (3 / 8) * a2
    q = (1 / 8) * a3 - 0.5 * a * b + c
    r = d - 0.25 * a * c + (1 / 16) * a2 * b - (3 / 256) * a4

    result = []

    if isclose(q, 0, rel_tol=0, abs_tol=1e-7):
        D = p * p - 4 * r
        if abs(D) < 1e-5:
            m = -0.5 * p
            if m >= 0:
                result.append(sqrt(m))
            if m > 0:
                result.append(-sqrt(m))
        elif D > 0:
            sqrt_D = sqrt(D)
            m1 = (-p - sqrt_D) * 0.5
            m2 = (-p + sqrt_D) * 0.5
            sqrt_m1 = sqrt(m1)
            sqrt_m2 = sqrt(m2)
            if m1 >= 0:
                result.append(sqrt_m1)
            if m1 > 0:
                result.append(-sqrt_m1)
            if m2 >= 0:
                result.append(sqrt_m2)
            if m2 > 0:
                result.append(-sqrt_m2)
    else:
        t = get_one_cubic_equation_root(2 * p, p * p - 4 * r, -q * q)
        z = cmath.sqrt(t)
        u = (Re(p) + (t)) * 0.5
        v = inverse_complex_number(z) * (q * 0.5)
        x12 = solve_complex_quadratic_equation(z, u - v)
        x34 = solve_complex_quadratic_equation(-z, u + v)

        if isclose(x12[0].imag, 0, rel_tol=0, abs_tol=1e-7):
            result.append(x12[0].real)
        if isclose(x12[1].imag, 0, rel_tol=0, abs_tol=1e-7):
            result.append(x12[1].real)
        if isclose(x34[0].imag, 0, rel_tol=0, abs_tol=1e-7):
            result.append(x34[0].real)
        if isclose(x34[1].imag, 0, rel_tol=0, abs_tol=1e-7):
            result.append(x34[1].real)

    # for(i = 0 i < result.length i++)
    for i in range(len(result)):
        result[i] -= 0.25 * a

    return result


ellipse = Ellipse

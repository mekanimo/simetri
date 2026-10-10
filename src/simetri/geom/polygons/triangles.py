"""Triangle object and related utility functions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import acos, cos, isclose, pi, sin, sqrt
from typing import Any

from ...base.all_enums import Connection, Types
from ...base.common import LineType, PointType, resolve_tol
from ...geom.geom_utils import midpoint
from ...geom.geometry import (
    double_area3,
    triangle_angles_from_sides,
    triangle_area as _triangle_area_from_sides,
    triangle_centroid3,
)
from ...geom.points.point_utils import (
    distance,
    equal_points,
    on_segment,
)
from ...geom.segments.line_utils import (
    collinear3,
    intersect,
    intersection,
    perp_bisector,
)
from ...helpers.validation import check_position
from ...shapes.geom_items import Circle
from ...shapes.shape import Shape

TriangleLike = "Triangle | Sequence[PointType]"


class TrianglePointClass(StrEnum):
    """Where a query point sits relative to a triangle."""

    INSIDE = "Inside"
    OUTSIDE = "Outside"
    ON_VERTEX = "OnVertex"
    ON_EDGE = "OnEdge"
    DEGENERATE = "Degenerate"


class TriangleIntersectionKind(StrEnum):
    """Kind of a 2D triangle intersection result."""

    EMPTY = "Empty"
    POINT = "A point"
    SEGMENT = "A segment"
    POLYGON = "A polygon"


@dataclass(frozen=True)
class RayTriangleHit:
    """Hit of a ray against a triangle.

    Attributes:
        point: Intersection point.
        distance: Distance from the ray origin to ``point``.
        barycentric_coordinates: Barycentric weights at ``point``.
        front_facing: True when the triangle is counter-clockwise.
    """

    point: PointType
    distance: float
    barycentric_coordinates: tuple[float, float, float]
    front_facing: bool


@dataclass(frozen=True)
class TriangleIntersection:
    """Intersection of a line, segment, or triangle with a triangle.

    Attributes:
        kind: Empty, a point, a segment, or a polygon.
        points: Intersection vertices in walk order.
    """

    kind: TriangleIntersectionKind
    points: tuple[PointType, ...]


def _xy(point: PointType) -> tuple[float, float]:
    x, y = point[:2]
    return (float(x) + 0.0, float(y) + 0.0)


def _triangle_vertices(
    triangle: TriangleLike,
) -> tuple[PointType, PointType, PointType]:
    if isinstance(triangle, Triangle):
        vertices = triangle.vertices
    else:
        vertices = tuple(triangle)
    if len(vertices) != 3:
        raise ValueError("A triangle must have three vertices.")
    return _xy(vertices[0]), _xy(vertices[1]), _xy(vertices[2])


def _side_lengths(
    p1: PointType, p2: PointType, p3: PointType
) -> tuple[float, float, float]:
    side_a = distance(p2, p3)
    side_b = distance(p3, p1)
    side_c = distance(p1, p2)
    return side_a, side_b, side_c


def _triangle_edges(
    p1: PointType, p2: PointType, p3: PointType
) -> tuple[tuple[PointType, PointType], ...]:
    p1, p2, p3 = _xy(p1), _xy(p2), _xy(p3)
    return ((p1, p2), (p2, p3), (p3, p1))


def _require_positive_area(
    p1: PointType, p2: PointType, p3: PointType, abs_tol: float
) -> float:
    signed = triangle_signed_area(p1, p2, p3)
    if abs(signed) <= abs_tol:
        raise ValueError("Degenerate triangle.")
    return signed


def _closest_point_on_segment(
    point: PointType, start: PointType, end: PointType
) -> PointType:
    start_x, start_y = start[:2]
    end_x, end_y = end[:2]
    point_x, point_y = point[:2]
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    length_sq = delta_x * delta_x + delta_y * delta_y
    if length_sq == 0.0:
        return (start_x, start_y)
    param = (
        (point_x - start_x) * delta_x + (point_y - start_y) * delta_y
    ) / length_sq
    param = max(0.0, min(1.0, param))
    return (start_x + param * delta_x, start_y + param * delta_y)


def _unique_points(
    points: Sequence[PointType],
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> list[PointType]:
    unique: list[PointType] = []
    for point in points:
        xy = _xy(point)
        if not any(equal_points(xy, kept, rel_tol, abs_tol) for kept in unique):
            unique.append(xy)
    return unique


def _intersection_from_points(
    points: Sequence[PointType],
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> TriangleIntersection:
    unique = _unique_points(points, rel_tol, abs_tol)
    if not unique:
        return TriangleIntersection(TriangleIntersectionKind.EMPTY, ())
    if len(unique) == 1:
        return TriangleIntersection(
            TriangleIntersectionKind.POINT, tuple(unique)
        )
    if len(unique) == 2:
        return TriangleIntersection(
            TriangleIntersectionKind.SEGMENT, tuple(unique)
        )
    return TriangleIntersection(TriangleIntersectionKind.POLYGON, tuple(unique))


def triangle_signed_area(p1: PointType, p2: PointType, p3: PointType) -> float:
    """Return the signed area of triangle ``p1 p2 p3``.

    Positive when the vertices are counter-clockwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_signed_area((0, 0), (60, 0), (0, 60))
        1800.0
        >>> sg.triangle_signed_area((0, 0), (0, 60), (60, 0))
        -1800.0
    """
    return double_area3(p1, p2, p3) / 2.0


def triangle_area(p1: PointType, p2: PointType, p3: PointType) -> float:
    """Return the unsigned area of triangle ``p1 p2 p3``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_area((0, 0), (60, 0), (0, 60))
        1800.0
        >>> sg.triangle_area((0, 0), (60, 0), (0, 80))
        2400.0
    """
    return abs(triangle_signed_area(p1, p2, p3))


def triangle_side_lengths(
    p1: PointType, p2: PointType, p3: PointType
) -> tuple[float, float, float]:
    """Return side lengths opposite ``p1``, ``p2``, and ``p3``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_side_lengths((0, 0), (60, 0), (0, 80))
        (100.0, 80.0, 60.0)
    """
    return _side_lengths(p1, p2, p3)


def triangle_longest_edge(
    p1: PointType, p2: PointType, p3: PointType
) -> tuple[PointType, PointType]:
    """Return the longest edge as a pair of vertices.

    Edges are ``(p1, p2)``, ``(p2, p3)``, ``(p3, p1)``. If two edges have
    the same length, the first of those in that order is returned.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_longest_edge((0, 0), (60, 0), (0, 80))
        ((60.0, 0.0), (0.0, 80.0))
    """
    return max(_triangle_edges(p1, p2, p3), key=lambda edge: distance(*edge))


def triangle_shortest_edge(
    p1: PointType, p2: PointType, p3: PointType
) -> tuple[PointType, PointType]:
    """Return the shortest edge as a pair of vertices.

    Edges are ``(p1, p2)``, ``(p2, p3)``, ``(p3, p1)``. If two edges have
    the same length, the first of those in that order is returned.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_shortest_edge((0, 0), (60, 0), (0, 80))
        ((0.0, 0.0), (60.0, 0.0))
    """
    return min(_triangle_edges(p1, p2, p3), key=lambda edge: distance(*edge))


def triangle_opposite_edge(
    p1: PointType,
    p2: PointType,
    p3: PointType,
    vertex_index: PointType | int,
) -> tuple[PointType, PointType]:
    """Return the edge opposite a vertex or vertex index.

    Index ``0`` is ``p1`` (opposite ``(p2, p3)``), ``1`` is ``p2``
    (opposite ``(p3, p1)``), and ``2`` is ``p3`` (opposite ``(p1, p2)``).

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_opposite_edge((0, 0), (60, 0), (0, 80), 0)
        ((60.0, 0.0), (0.0, 80.0))
        >>> sg.triangle_opposite_edge((0, 0), (60, 0), (0, 80), (60, 0))
        ((0.0, 80.0), (0.0, 0.0))
    """
    p1, p2, p3 = _xy(p1), _xy(p2), _xy(p3)
    if vertex_index in (0, p1) or (
        not isinstance(vertex_index, int) and equal_points(vertex_index, p1)
    ):
        return (p2, p3)
    if vertex_index in (1, p2) or (
        not isinstance(vertex_index, int) and equal_points(vertex_index, p2)
    ):
        return (p3, p1)
    if vertex_index in (2, p3) or (
        not isinstance(vertex_index, int) and equal_points(vertex_index, p3)
    ):
        return (p1, p2)
    if isinstance(vertex_index, int):
        raise ValueError("vertex index must be 0, 1, or 2.")
    raise ValueError("vertex is not a vertex of this triangle.")


def triangle_perimeter(p1: PointType, p2: PointType, p3: PointType) -> float:
    """Return the perimeter of triangle ``p1 p2 p3``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_perimeter((0, 0), (60, 0), (0, 80))
        240.0
    """
    side_a, side_b, side_c = _side_lengths(p1, p2, p3)
    return side_a + side_b + side_c


def triangle_semiperimeter(
    p1: PointType, p2: PointType, p3: PointType
) -> float:
    """Return the semiperimeter of triangle ``p1 p2 p3``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_semiperimeter((0, 0), (60, 0), (0, 80))
        120.0
    """
    return triangle_perimeter(p1, p2, p3) / 2.0


def triangle_altitudes(
    p1: PointType, p2: PointType, p3: PointType
) -> tuple[float, float, float]:
    """Return altitude lengths to the sides opposite ``p1``, ``p2``, ``p3``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_altitudes((0, 0), (60, 0), (0, 80))
        (48.0, 60.0, 80.0)
    """
    area = triangle_area(p1, p2, p3)
    side_a, side_b, side_c = _side_lengths(p1, p2, p3)
    if side_a == 0.0 or side_b == 0.0 or side_c == 0.0:
        raise ValueError("Degenerate triangle.")
    return (2.0 * area / side_a, 2.0 * area / side_b, 2.0 * area / side_c)


def triangle_medians(
    p1: PointType, p2: PointType, p3: PointType
) -> tuple[float, float, float]:
    """Return median lengths from ``p1``, ``p2``, and ``p3``.

    Examples:
        >>> import simetri.graphics as sg
        >>> medians = sg.triangle_medians((0, 0), (60, 0), (0, 60))
        >>> tuple(round(length, 10) for length in medians)
        (42.4264068712, 67.082039325, 67.082039325)
    """
    side_a, side_b, side_c = _side_lengths(p1, p2, p3)

    def median_length(opp: float, adj_b: float, adj_c: float) -> float:
        return 0.5 * sqrt(2.0 * adj_b * adj_b + 2.0 * adj_c * adj_c - opp * opp)

    return (
        median_length(side_a, side_b, side_c),
        median_length(side_b, side_a, side_c),
        median_length(side_c, side_a, side_b),
    )


def triangle_area_from_sides(side1: float, side2: float, side3: float) -> float:
    """Return triangle area from three side lengths (Heron's formula).

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_area_from_sides(60, 80, 100)
        2400.0
    """
    return _triangle_area_from_sides(side1, side2, side3)


def triangle_angle_at(p1: PointType, p2: PointType, p3: PointType) -> float:
    """Return the interior angle at ``p1`` in radians.

    Examples:
        >>> import simetri.graphics as sg
        >>> round(sg.triangle_angle_at((0, 0), (60, 0), (0, 60)), 10)
        1.5707963268
    """
    side_a, side_b, side_c = _side_lengths(p1, p2, p3)
    return law_of_cosines_angle(side_a, side_b, side_c)


def triangle_angles(
    p1: PointType, p2: PointType, p3: PointType
) -> tuple[float, float, float]:
    """Return interior angles at ``p1``, ``p2``, and ``p3`` in radians.

    Examples:
        >>> import simetri.graphics as sg
        >>> angles = sg.triangle_angles((0, 0), (60, 0), (0, 60))
        >>> tuple(round(angle, 10) for angle in angles)
        (1.5707963268, 0.7853981634, 0.7853981634)
    """
    return (
        triangle_angle_at(p1, p2, p3),
        triangle_angle_at(p2, p3, p1),
        triangle_angle_at(p3, p1, p2),
    )


def triangle_side_from_sas(side1: float, angle: float, side2: float) -> float:
    """Return the side opposite the included ``angle`` (SAS).

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_side_from_sas(60, sg.pi / 2, 80)
        100.0
    """
    return law_of_cosines_side(side1, side2, angle)


def triangle_side_from_aas(
    angle_a: float, angle_b: float, side_a: float
) -> float:
    """Return the side opposite ``angle_b`` given AAS data.

    Examples:
        >>> import simetri.graphics as sg
        >>> round(sg.triangle_side_from_aas(sg.pi / 2, sg.pi / 4, 60), 10)
        42.4264068712
    """
    return law_of_sines_side(side_a, angle_a, angle_b)


def triangle_2sides_angle(side1: float, angle: float, side2: float) -> Triangle:
    """Return a triangle from two sides and the included angle (SAS).

    The included-angle vertex is at the origin. ``side1`` lies along the
    positive x-axis. ``side2`` is placed at ``angle`` from that axis, so a
    positive ``angle`` yields a counterclockwise triangle.

    Args:
        side1: Length of the side from the origin along the positive x-axis.
        angle: Included angle at the origin, in radians.
        side2: Length of the side from the origin at ``angle``.

    Returns:
        Triangle: Vertices at the origin, along the x-axis, and at ``angle``.

    Examples:
        >>> import simetri.graphics as sg
        >>> tri = sg.triangle_2sides_angle(60, sg.pi / 2, 80)
        >>> tri.vertices
        ((0.0, 0.0), (60.0, 0.0), (0.0, 80.0))
        >>> tri.side_lengths()
        (100.0, 80.0, 60.0)
    """
    if side1 <= 0.0 or side2 <= 0.0:
        raise ValueError("Degenerate triangle.")
    if not (0.0 < angle < pi):
        raise ValueError("Degenerate triangle.")
    _, abs_tol = resolve_tol()
    p1 = (0.0, 0.0)
    p2 = (float(side1) + 0.0, 0.0)
    p3_x = side2 * cos(angle) + 0.0
    p3_y = side2 * sin(angle) + 0.0
    if abs(p3_x) <= abs_tol:
        p3_x = 0.0
    if abs(p3_y) <= abs_tol:
        p3_y = 0.0
    p3 = (p3_x, p3_y)
    _require_positive_area(p1, p2, p3, abs_tol)
    return Triangle(p1, p2, p3)


def triangle_2angles_side(
    angle_a: float, angle_b: float, side_a: float
) -> Triangle:
    """Return a triangle from two angles and the included side (ASA).

    The vertex of ``angle_a`` is at the origin. The included side (between
    that vertex and the vertex of ``angle_b``) lies along the positive
    x-axis and has length ``side_a``. A positive construction yields a
    counterclockwise triangle.

    Args:
        angle_a: Interior angle at the origin, in radians.
        angle_b: Interior angle at the positive-x vertex, in radians.
        side_a: Length of the included side, along the positive x-axis.

    Returns:
        Triangle: Vertices at the origin, along the x-axis, and at ``angle_a``.

    Examples:
        >>> import simetri.graphics as sg
        >>> tri = sg.triangle_2angles_side(sg.pi / 2, sg.pi / 4, 60)
        >>> tri.vertices
        ((0.0, 0.0), (60.0, 0.0), (0.0, 60.0))
        >>> tuple(round(angle, 10) for angle in tri.angles())
        (1.5707963268, 0.7853981634, 0.7853981634)
    """
    if side_a <= 0.0:
        raise ValueError("Degenerate triangle.")
    if not (0.0 < angle_a < pi) or not (0.0 < angle_b < pi):
        raise ValueError("Degenerate triangle.")
    angle_c = pi - angle_a - angle_b
    if not (0.0 < angle_c < pi):
        raise ValueError("Degenerate triangle.")
    side_b = law_of_sines_side(side_a, angle_c, angle_b)
    return triangle_2sides_angle(side_a, angle_a, side_b)


def law_of_cosines_side(side_b: float, side_c: float, angle_a: float) -> float:
    """Return the side opposite ``angle_a`` from two sides and the included angle.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.law_of_cosines_side(60, 80, sg.pi / 2)
        100.0
    """
    return sqrt(
        side_b * side_b + side_c * side_c - 2.0 * side_b * side_c * cos(angle_a)
    )


def law_of_cosines_angle(side_a: float, side_b: float, side_c: float) -> float:
    """Return the angle opposite ``side_a`` from three side lengths.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.law_of_cosines_angle(100, 80, 60) == sg.pi / 2
        True
    """
    denom = 2.0 * side_b * side_c
    if denom == 0.0:
        raise ValueError("Degenerate triangle.")
    cosine = (side_b * side_b + side_c * side_c - side_a * side_a) / denom
    cosine = max(-1.0, min(1.0, cosine))
    return acos(cosine)


def law_of_sines_side(side_a: float, angle_a: float, angle_b: float) -> float:
    """Return the side opposite ``angle_b`` from one side and two angles.

    Examples:
        >>> import simetri.graphics as sg
        >>> round(sg.law_of_sines_side(60, sg.pi / 2, sg.pi / 6), 10)
        30.0
    """
    if isclose(sin(angle_a), 0.0, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Degenerate triangle.")
    return side_a * sin(angle_b) / sin(angle_a)


def is_collinear(
    p1: PointType,
    p2: PointType,
    p3: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if ``p1``, ``p2``, and ``p3`` are collinear.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.is_collinear((0, 0), (40, 40), (80, 80))
        True
        >>> sg.is_collinear((0, 0), (60, 0), (0, 60))
        False
    """
    return collinear3(p1, p2, p3, rel_tol=rel_tol, abs_tol=abs_tol)


def is_degenerate_triangle(
    p1: PointType,
    p2: PointType,
    p3: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if the three points do not form a triangle of positive area.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.is_degenerate_triangle((0, 0), (40, 0), (80, 0))
        True
        >>> sg.is_degenerate_triangle((0, 0), (60, 0), (0, 60))
        False
    """
    _, abs_tol = resolve_tol(rel_tol, abs_tol)
    return abs(triangle_signed_area(p1, p2, p3)) <= abs_tol


def is_valid_triangle(
    p1: PointType,
    p2: PointType,
    p3: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if ``p1 p2 p3`` has positive area.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.is_valid_triangle((0, 0), (60, 0), (0, 60))
        True
        >>> sg.is_valid_triangle((0, 0), (40, 0), (80, 0))
        False
    """
    return not is_degenerate_triangle(p1, p2, p3, rel_tol, abs_tol)


def is_equilateral_triangle(
    p1: PointType,
    p2: PointType,
    p3: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if all three sides are equal within tolerance.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.is_equilateral_triangle((0, 0), (60, 0), (30, 51.9615242268))
        True
        >>> sg.is_equilateral_triangle((0, 0), (60, 0), (0, 60))
        False
    """
    rel_tol, abs_tol = resolve_tol(rel_tol, abs_tol)
    side_a, side_b, side_c = _side_lengths(p1, p2, p3)
    return isclose(
        side_a, side_b, rel_tol=rel_tol, abs_tol=abs_tol
    ) and isclose(side_b, side_c, rel_tol=rel_tol, abs_tol=abs_tol)


def is_isosceles_triangle(
    p1: PointType,
    p2: PointType,
    p3: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if at least two sides are equal within tolerance.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.is_isosceles_triangle((0, 0), (60, 0), (0, 60))
        True
        >>> sg.is_isosceles_triangle((0, 0), (60, 0), (0, 80))
        False
    """
    rel_tol, abs_tol = resolve_tol(rel_tol, abs_tol)
    side_a, side_b, side_c = _side_lengths(p1, p2, p3)
    return (
        isclose(side_a, side_b, rel_tol=rel_tol, abs_tol=abs_tol)
        or isclose(side_b, side_c, rel_tol=rel_tol, abs_tol=abs_tol)
        or isclose(side_c, side_a, rel_tol=rel_tol, abs_tol=abs_tol)
    )


def is_scalene_triangle(
    p1: PointType,
    p2: PointType,
    p3: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if no two sides are equal within tolerance.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.is_scalene_triangle((0, 0), (60, 0), (0, 80))
        True
        >>> sg.is_scalene_triangle((0, 0), (60, 0), (0, 60))
        False
    """
    return is_valid_triangle(p1, p2, p3, rel_tol, abs_tol) and not (
        is_isosceles_triangle(p1, p2, p3, rel_tol, abs_tol)
    )


def is_right_triangle(
    p1: PointType,
    p2: PointType,
    p3: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if one angle is a right angle within tolerance.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.is_right_triangle((0, 0), (60, 0), (0, 80))
        True
        >>> sg.is_right_triangle((0, 0), (60, 0), (30, 60))
        False
    """
    rel_tol, abs_tol = resolve_tol(rel_tol, abs_tol)
    sides = sorted(_side_lengths(p1, p2, p3))
    return isclose(
        sides[0] * sides[0] + sides[1] * sides[1],
        sides[2] * sides[2],
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def is_acute_triangle(
    p1: PointType,
    p2: PointType,
    p3: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if all angles are acute.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.is_acute_triangle((0, 0), (60, 0), (30, 60))
        True
        >>> sg.is_acute_triangle((0, 0), (60, 0), (0, 80))
        False
    """
    rel_tol, abs_tol = resolve_tol(rel_tol, abs_tol)
    if is_degenerate_triangle(p1, p2, p3, rel_tol, abs_tol):
        return False
    sides = sorted(_side_lengths(p1, p2, p3))
    return (
        sides[0] * sides[0] + sides[1] * sides[1]
        > sides[2] * sides[2] + abs_tol
    )


def is_obtuse_triangle(
    p1: PointType,
    p2: PointType,
    p3: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if one angle is obtuse.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.is_obtuse_triangle((0, 0), (20, 0), (60, 2))
        True
        >>> sg.is_obtuse_triangle((0, 0), (60, 0), (0, 80))
        False
    """
    rel_tol, abs_tol = resolve_tol(rel_tol, abs_tol)
    if is_degenerate_triangle(p1, p2, p3, rel_tol, abs_tol):
        return False
    sides = sorted(_side_lengths(p1, p2, p3))
    return (
        sides[0] * sides[0] + sides[1] * sides[1]
        < sides[2] * sides[2] - abs_tol
    )


def triangle_orientation(
    p1: PointType,
    p2: PointType,
    p3: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> int:
    """Return ``1`` if CCW, ``-1`` if CW, ``0`` if collinear.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_orientation((0, 0), (60, 0), (0, 60))
        1
        >>> sg.triangle_orientation((0, 0), (0, 60), (60, 0))
        -1
        >>> sg.triangle_orientation((0, 0), (40, 0), (80, 0))
        0
    """
    _, abs_tol = resolve_tol(rel_tol, abs_tol)
    signed = triangle_signed_area(p1, p2, p3)
    if abs(signed) <= abs_tol:
        return 0
    if signed > 0:
        return 1
    return -1


def is_counterclockwise(
    p1: PointType,
    p2: PointType,
    p3: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if ``p1 p2 p3`` is counter-clockwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.is_counterclockwise((0, 0), (60, 0), (0, 60))
        True
    """
    return triangle_orientation(p1, p2, p3, rel_tol, abs_tol) == 1


def is_clockwise(
    p1: PointType,
    p2: PointType,
    p3: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if ``p1 p2 p3`` is clockwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.is_clockwise((0, 0), (0, 60), (60, 0))
        True
    """
    return triangle_orientation(p1, p2, p3, rel_tol, abs_tol) == -1


def reverse_winding(triangle: TriangleLike) -> Triangle:
    """Return a new ``Triangle`` with reversed vertex order.

    Examples:
        >>> import simetri.graphics as sg
        >>> tri = sg.reverse_winding(((0, 0), (60, 0), (0, 60)))
        >>> tri.vertices
        ((0.0, 60.0), (60.0, 0.0), (0.0, 0.0))
    """
    p1, p2, p3 = _triangle_vertices(triangle)
    return Triangle(p3, p2, p1)


def triangle_centroid(p1: PointType, p2: PointType, p3: PointType) -> PointType:
    """Return the centroid of triangle ``p1 p2 p3``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_centroid((0, 0), (60, 0), (0, 60))
        (20.0, 20.0)
    """
    return triangle_centroid3(p1, p2, p3)


def triangle_incenter(p1: PointType, p2: PointType, p3: PointType) -> PointType:
    """Return the incenter of triangle ``p1 p2 p3``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_incenter((0, 0), (80, 0), (0, 60))
        (20.0, 20.0)
    """
    side_a, side_b, side_c = _side_lengths(p1, p2, p3)
    perimeter = side_a + side_b + side_c
    if perimeter == 0.0:
        raise ValueError("Degenerate triangle.")
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    x3, y3 = p3[:2]
    center_x = (side_a * x1 + side_b * x2 + side_c * x3) / perimeter
    center_y = (side_a * y1 + side_b * y2 + side_c * y3) / perimeter
    return (center_x, center_y)


def triangle_circumcenter(
    p1: PointType, p2: PointType, p3: PointType
) -> PointType:
    """Return the circumcenter of triangle ``p1 p2 p3``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_circumcenter((0, 0), (60, 0), (0, 60))
        (30.0, 30.0)
    """
    bisector_ab = perp_bisector((p1, p2))
    bisector_ac = perp_bisector((p1, p3))
    center = intersect(bisector_ab, bisector_ac)
    if center is None:
        raise ValueError("Degenerate triangle.")
    return _xy(center)


def triangle_orthocenter(
    p1: PointType, p2: PointType, p3: PointType
) -> PointType:
    """Return the orthocenter of triangle ``p1 p2 p3``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_orthocenter((0, 0), (60, 0), (0, 60))
        (0.0, 0.0)
    """
    centroid_x, centroid_y = triangle_centroid(p1, p2, p3)
    circum_x, circum_y = triangle_circumcenter(p1, p2, p3)
    return (
        3.0 * centroid_x - 2.0 * circum_x,
        3.0 * centroid_y - 2.0 * circum_y,
    )


def triangle_nine_point_center(
    p1: PointType, p2: PointType, p3: PointType
) -> PointType:
    """Return the nine-point center of triangle ``p1 p2 p3``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_nine_point_center((0, 0), (60, 0), (0, 60))
        (15.0, 15.0)
    """
    circum = triangle_circumcenter(p1, p2, p3)
    ortho = triangle_orthocenter(p1, p2, p3)
    return midpoint(circum, ortho)


def triangle_excenters(
    p1: PointType, p2: PointType, p3: PointType
) -> tuple[PointType, PointType, PointType]:
    """Return the three excenters opposite ``p1``, ``p2``, and ``p3``.

    Examples:
        >>> import simetri.graphics as sg
        >>> centers = sg.triangle_excenters((0, 0), (80, 0), (0, 60))
        >>> tuple(tuple(round(coord, 10) for coord in center) for center in centers)
        ((120.0, 120.0), (-40.0, 40.0), (60.0, -60.0))
    """
    side_a, side_b, side_c = _side_lengths(p1, p2, p3)
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    x3, y3 = p3[:2]

    def excenter(
        opp: float,
        adj_b: float,
        adj_c: float,
        px: float,
        py: float,
        qx: float,
        qy: float,
        rx: float,
        ry: float,
    ) -> PointType:
        denom = -opp + adj_b + adj_c
        if denom == 0.0:
            raise ValueError("Degenerate triangle.")
        center_x = (-opp * px + adj_b * qx + adj_c * rx) / denom
        center_y = (-opp * py + adj_b * qy + adj_c * ry) / denom
        return (center_x, center_y)

    return (
        excenter(side_a, side_b, side_c, x1, y1, x2, y2, x3, y3),
        excenter(side_b, side_a, side_c, x2, y2, x1, y1, x3, y3),
        excenter(side_c, side_a, side_b, x3, y3, x1, y1, x2, y2),
    )


def triangle_inradius(p1: PointType, p2: PointType, p3: PointType) -> float:
    """Return the inradius of triangle ``p1 p2 p3``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_inradius((0, 0), (80, 0), (0, 60))
        20.0
    """
    semiperimeter = triangle_semiperimeter(p1, p2, p3)
    if semiperimeter == 0.0:
        raise ValueError("Degenerate triangle.")
    return triangle_area(p1, p2, p3) / semiperimeter


def triangle_circumradius(p1: PointType, p2: PointType, p3: PointType) -> float:
    """Return the circumradius of triangle ``p1 p2 p3``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_circumradius((0, 0), (60, 0), (0, 60))
        42.42640687119285
    """
    return distance(triangle_circumcenter(p1, p2, p3), p1)


def triangle_incircle(triangle: TriangleLike) -> Circle:
    """Return the incircle of ``triangle``.

    Examples:
        >>> import simetri.graphics as sg
        >>> circle = sg.triangle_incircle(((0, 0), (80, 0), (0, 60)))
        >>> tuple(round(float(coord), 10) for coord in circle.center[:2])
        (20.0, 20.0)
        >>> circle.radius
        20.0
    """
    p1, p2, p3 = _triangle_vertices(triangle)
    return Circle(
        radius=triangle_inradius(p1, p2, p3),
        center=triangle_incenter(p1, p2, p3),
    )


def triangle_circumcircle(triangle: TriangleLike) -> Circle:
    """Return the circumcircle of ``triangle``.

    Examples:
        >>> import simetri.graphics as sg
        >>> circle = sg.triangle_circumcircle(((0, 0), (60, 0), (0, 60)))
        >>> tuple(circle.center[:2])
        (30.0, 30.0)
    """
    p1, p2, p3 = _triangle_vertices(triangle)
    return Circle(
        radius=triangle_circumradius(p1, p2, p3),
        center=triangle_circumcenter(p1, p2, p3),
    )


def triangle_nine_point_circle(triangle: TriangleLike) -> Circle:
    """Return the nine-point circle of ``triangle``.

    Examples:
        >>> import simetri.graphics as sg
        >>> circle = sg.triangle_nine_point_circle(((0, 0), (60, 0), (0, 60)))
        >>> tuple(circle.center[:2])
        (15.0, 15.0)
    """
    p1, p2, p3 = _triangle_vertices(triangle)
    return Circle(
        radius=triangle_circumradius(p1, p2, p3) / 2.0,
        center=triangle_nine_point_center(p1, p2, p3),
    )


def barycentric_coordinates(
    point: PointType, triangle: TriangleLike
) -> tuple[float, float, float]:
    """Return barycentric weights of ``point`` with respect to ``triangle``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.barycentric_coordinates((0, 0), ((0, 0), (60, 0), (0, 60)))
        (1.0, 0.0, 0.0)
        >>> sg.barycentric_coordinates((30, 30), ((0, 0), (60, 0), (0, 60)))
        (0.0, 0.5, 0.5)
    """
    p1, p2, p3 = _triangle_vertices(triangle)
    area = triangle_signed_area(p1, p2, p3)
    if area == 0.0:
        raise ValueError("Degenerate triangle.")
    weight1 = triangle_signed_area(point, p2, p3) / area
    weight2 = triangle_signed_area(p1, point, p3) / area
    weight3 = triangle_signed_area(p1, p2, point) / area
    return (weight1, weight2, weight3)


def cartesian_from_barycentric(
    weights: Sequence[float], triangle: TriangleLike
) -> PointType:
    """Return the Cartesian point for barycentric ``weights``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.cartesian_from_barycentric((1, 0, 0), ((0, 0), (60, 0), (0, 60)))
        (0.0, 0.0)
        >>> sg.cartesian_from_barycentric((0, 0.5, 0.5), ((0, 0), (60, 0), (0, 60)))
        (30.0, 30.0)
    """
    p1, p2, p3 = _triangle_vertices(triangle)
    w1, w2, w3 = weights[:3]
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    x3, y3 = p3[:2]
    return (w1 * x1 + w2 * x2 + w3 * x3, w1 * y1 + w2 * y2 + w3 * y3)


def is_valid_barycentric(
    weights: Sequence[float],
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if ``weights`` sum to 1 within tolerance.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.is_valid_barycentric((0.2, 0.3, 0.5))
        True
        >>> sg.is_valid_barycentric((1, 1, 1))
        False
    """
    rel_tol, abs_tol = resolve_tol(rel_tol, abs_tol)
    return isclose(sum(weights[:3]), 1.0, rel_tol=rel_tol, abs_tol=abs_tol)


def normalize_barycentric(
    weights: Sequence[float],
) -> tuple[float, float, float]:
    """Return ``weights`` scaled so they sum to 1.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.normalize_barycentric((1, 1, 2))
        (0.25, 0.25, 0.5)
    """
    w1, w2, w3 = weights[:3]
    total = w1 + w2 + w3
    if total == 0.0:
        raise ValueError("Cannot normalize barycentric weights that sum to 0.")
    return (w1 / total, w2 / total, w3 / total)


def interpolate_triangle(
    values: Sequence[Any], barycentric_weights: Sequence[float]
) -> Any:
    """Interpolate three vertex values with barycentric weights.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.interpolate_triangle((0, 40, 80), (0.5, 0.5, 0))
        20.0
    """
    w1, w2, w3 = barycentric_weights[:3]
    v1, v2, v3 = values[:3]
    return w1 * v1 + w2 * v2 + w3 * v3


def interpolate_triangle_at_point(
    point: PointType, triangle: TriangleLike, values: Sequence[Any]
) -> Any:
    """Interpolate ``values`` at ``point`` on ``triangle``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.interpolate_triangle_at_point(
        ...     (0, 0), ((0, 0), (60, 0), (0, 60)), (30, 40, 50)
        ... )
        30.0
    """
    return interpolate_triangle(
        values, barycentric_coordinates(point, triangle)
    )


def triangle_contains_point_inclusive(
    triangle: TriangleLike,
    point: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if ``point`` is inside ``triangle`` or on its boundary.

    Examples:
        >>> import simetri.graphics as sg
        >>> tri = ((0, 0), (60, 0), (0, 60))
        >>> sg.triangle_contains_point_inclusive(tri, (20, 20))
        True
        >>> sg.triangle_contains_point_inclusive(tri, (0, 0))
        True
        >>> sg.triangle_contains_point_inclusive(tri, (80, 80))
        False
    """
    rel_tol, abs_tol = resolve_tol(rel_tol, abs_tol)
    p1, p2, p3 = _triangle_vertices(triangle)
    if is_degenerate_triangle(p1, p2, p3, rel_tol, abs_tol):
        return False
    w1, w2, w3 = barycentric_coordinates(point, (p1, p2, p3))
    return (
        w1 >= -abs_tol
        and w2 >= -abs_tol
        and w3 >= -abs_tol
        and isclose(w1 + w2 + w3, 1.0, rel_tol=rel_tol, abs_tol=abs_tol)
    )


def triangle_contains_point_strictly(
    triangle: TriangleLike,
    point: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if ``point`` is strictly inside ``triangle``.

    Examples:
        >>> import simetri.graphics as sg
        >>> tri = ((0, 0), (60, 0), (0, 60))
        >>> sg.triangle_contains_point_strictly(tri, (20, 20))
        True
        >>> sg.triangle_contains_point_strictly(tri, (0, 0))
        False
    """
    rel_tol, abs_tol = resolve_tol(rel_tol, abs_tol)
    p1, p2, p3 = _triangle_vertices(triangle)
    if is_degenerate_triangle(p1, p2, p3, rel_tol, abs_tol):
        return False
    w1, w2, w3 = barycentric_coordinates(point, (p1, p2, p3))
    return w1 > abs_tol and w2 > abs_tol and w3 > abs_tol


def triangle_contains_point(
    triangle: TriangleLike,
    point: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if ``point`` is inside ``triangle`` or on its boundary.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.triangle_contains_point(((0, 0), (60, 0), (0, 60)), (20, 20))
        True
    """
    return triangle_contains_point_inclusive(triangle, point, rel_tol, abs_tol)


def classify_point_in_triangle(
    triangle: TriangleLike,
    point: PointType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> TrianglePointClass:
    """Classify ``point`` relative to ``triangle``.

    Examples:
        >>> import simetri.graphics as sg
        >>> tri = ((0, 0), (60, 0), (0, 60))
        >>> sg.classify_point_in_triangle(tri, (20, 20))
        <TrianglePointClass.INSIDE: 'Inside'>
        >>> sg.classify_point_in_triangle(tri, (0, 0))
        <TrianglePointClass.ON_VERTEX: 'OnVertex'>
        >>> sg.classify_point_in_triangle(tri, (30, 0))
        <TrianglePointClass.ON_EDGE: 'OnEdge'>
        >>> sg.classify_point_in_triangle(tri, (80, 80))
        <TrianglePointClass.OUTSIDE: 'Outside'>
    """
    rel_tol, abs_tol = resolve_tol(rel_tol, abs_tol)
    p1, p2, p3 = _triangle_vertices(triangle)
    if is_degenerate_triangle(p1, p2, p3, rel_tol, abs_tol):
        return TrianglePointClass.DEGENERATE
    if (
        equal_points(point, p1, rel_tol, abs_tol)
        or equal_points(point, p2, rel_tol, abs_tol)
        or equal_points(point, p3, rel_tol, abs_tol)
    ):
        return TrianglePointClass.ON_VERTEX
    if (
        on_segment(p1, p2, point, eps=abs_tol)
        or on_segment(p2, p3, point, eps=abs_tol)
        or on_segment(p3, p1, point, eps=abs_tol)
    ):
        return TrianglePointClass.ON_EDGE
    if triangle_contains_point_strictly(triangle, point, rel_tol, abs_tol):
        return TrianglePointClass.INSIDE
    return TrianglePointClass.OUTSIDE


def closest_point_on_triangle(
    point: PointType, triangle: TriangleLike
) -> PointType:
    """Return the closest point of ``triangle`` to ``point``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.closest_point_on_triangle((20, 20), ((0, 0), (60, 0), (0, 60)))
        (20.0, 20.0)
        >>> sg.closest_point_on_triangle((80, 0), ((0, 0), (60, 0), (0, 60)))
        (60.0, 0.0)
    """
    p1, p2, p3 = _triangle_vertices(triangle)
    if triangle_contains_point_inclusive((p1, p2, p3), point):
        return _xy(point)
    candidates = (
        _closest_point_on_segment(point, p1, p2),
        _closest_point_on_segment(point, p2, p3),
        _closest_point_on_segment(point, p3, p1),
    )
    return min(candidates, key=lambda candidate: distance(point, candidate))


def distance_to_triangle(point: PointType, triangle: TriangleLike) -> float:
    """Return the distance from ``point`` to ``triangle``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.distance_to_triangle((20, 20), ((0, 0), (60, 0), (0, 60)))
        0.0
        >>> sg.distance_to_triangle((80, 0), ((0, 0), (60, 0), (0, 60)))
        20.0
    """
    return distance(point, closest_point_on_triangle(point, triangle))


def _ray_param(
    origin: PointType, direction: PointType, point: PointType
) -> float:
    origin_x, origin_y = origin[:2]
    dir_x, dir_y = direction[:2]
    point_x, point_y = point[:2]
    denom = dir_x * dir_x + dir_y * dir_y
    if denom == 0.0:
        return 0.0
    return ((point_x - origin_x) * dir_x + (point_y - origin_y) * dir_y) / denom


def _edge_hits_with_line(
    line: LineType, triangle: TriangleLike
) -> list[PointType]:
    p1, p2, p3 = _triangle_vertices(triangle)
    hits: list[PointType] = []
    for edge in ((p1, p2), (p2, p3), (p3, p1)):
        hit = intersect(line, edge)
        if hit is None:
            continue
        if on_segment(edge[0], edge[1], hit):
            hits.append(_xy(hit))
    return hits


def intersect_line_triangle(
    line: LineType, triangle: TriangleLike
) -> TriangleIntersection:
    """Intersect an infinite line with ``triangle``.

    Examples:
        >>> import simetri.graphics as sg
        >>> hit = sg.intersect_line_triangle([(0, 15), (60, 15)], ((0, 0), (60, 0), (0, 60)))
        >>> hit.kind
        <TriangleIntersectionKind.SEGMENT: 'A segment'>
        >>> hit.points
        ((45.0, 15.0), (0.0, 15.0))
    """
    return _intersection_from_points(_edge_hits_with_line(line, triangle))


def intersect_segment_triangle(
    segment: LineType, triangle: TriangleLike
) -> TriangleIntersection:
    """Intersect a line segment with ``triangle``.

    Examples:
        >>> import simetri.graphics as sg
        >>> hit = sg.intersect_segment_triangle(
        ...     [(-60, 15), (60, 15)], ((0, 0), (60, 0), (0, 60))
        ... )
        >>> hit.kind
        <TriangleIntersectionKind.SEGMENT: 'A segment'>
        >>> hit.points
        ((45.0, 15.0), (0.0, 15.0))
    """
    p1, p2, p3 = _triangle_vertices(triangle)
    start, end = _xy(segment[0]), _xy(segment[1])
    points: list[PointType] = []
    if triangle_contains_point_inclusive((p1, p2, p3), start):
        points.append(start)
    if triangle_contains_point_inclusive((p1, p2, p3), end):
        points.append(end)
    for edge in ((p1, p2), (p2, p3), (p3, p1)):
        kind, hit = intersection((start, end), edge)
        if kind == Connection.INTERSECT and hit is not None:
            points.append(_xy(hit))
    return _intersection_from_points(points)


def intersect_ray_triangle(
    ray: LineType, triangle: TriangleLike
) -> RayTriangleHit | None:
    """Intersect a ray with ``triangle``.

    ``ray`` is ``(origin, point_on_ray)``. The first hit along the ray is
    returned, or ``None`` if the ray misses.

    Examples:
        >>> import simetri.graphics as sg
        >>> hit = sg.intersect_ray_triangle(
        ...     [(-60, 15), (60, 15)], ((0, 0), (60, 0), (0, 60))
        ... )
        >>> hit.point
        (0.0, 15.0)
        >>> hit.distance
        60.0
        >>> hit.front_facing
        True
    """
    origin = _xy(ray[0])
    through = _xy(ray[1])
    direction = (through[0] - origin[0], through[1] - origin[1])
    p1, p2, p3 = _triangle_vertices(triangle)
    _, abs_tol = resolve_tol()
    candidates: list[PointType] = []
    if triangle_contains_point_inclusive((p1, p2, p3), origin):
        candidates.append(origin)
    for point in _edge_hits_with_line((origin, through), (p1, p2, p3)):
        param = _ray_param(origin, direction, point)
        if param >= -abs_tol:
            candidates.append(point)
    unique = _unique_points(candidates)
    if not unique:
        return None
    hit_point = min(
        unique, key=lambda point: _ray_param(origin, direction, point)
    )
    return RayTriangleHit(
        point=hit_point,
        distance=distance(origin, hit_point),
        barycentric_coordinates=barycentric_coordinates(
            hit_point, (p1, p2, p3)
        ),
        front_facing=is_counterclockwise(p1, p2, p3),
    )


def _is_left_or_on(
    origin: PointType, dest: PointType, point: PointType, abs_tol: float
) -> bool:
    signed = double_area3(origin, dest, point)
    return signed >= -abs_tol * 2.0


def _clip_polygon_by_edge(
    polygon: Sequence[PointType],
    origin: PointType,
    dest: PointType,
    abs_tol: float,
) -> list[PointType]:
    if not polygon:
        return []
    output: list[PointType] = []
    prev = polygon[-1]
    prev_inside = _is_left_or_on(origin, dest, prev, abs_tol)
    for current in polygon:
        current_inside = _is_left_or_on(origin, dest, current, abs_tol)
        if current_inside:
            if not prev_inside:
                hit = intersect((prev, current), (origin, dest))
                if hit is not None:
                    output.append(_xy(hit))
            output.append(_xy(current))
        elif prev_inside:
            hit = intersect((prev, current), (origin, dest))
            if hit is not None:
                output.append(_xy(hit))
        prev = current
        prev_inside = current_inside
    return output


def intersect_triangle_triangle(
    triangle1: TriangleLike, triangle2: TriangleLike
) -> TriangleIntersection:
    """Intersect two triangles.

    Examples:
        >>> import simetri.graphics as sg
        >>> hit = sg.intersect_triangle_triangle(
        ...     ((0, 0), (60, 0), (0, 60)), ((30, -30), (30, 60), (90, 0))
        ... )
        >>> hit.kind
        <TriangleIntersectionKind.POLYGON: 'A polygon'>
        >>> hit.points
        ((30.0, 30.0), (30.0, 0.0), (60.0, 0.0))
    """
    _, abs_tol = resolve_tol()
    p1, p2, p3 = _triangle_vertices(triangle1)
    q1, q2, q3 = _triangle_vertices(triangle2)
    clip = [p1, p2, p3]
    if triangle_orientation(p1, p2, p3) < 0:
        clip = [p1, p3, p2]
    subject = [q1, q2, q3]
    if triangle_orientation(q1, q2, q3) < 0:
        subject = [q1, q3, q2]
    for i, origin in enumerate(clip):
        dest = clip[(i + 1) % 3]
        subject = _clip_polygon_by_edge(subject, origin, dest, abs_tol)
        if not subject:
            break
    return _intersection_from_points(subject)


def subdivide_triangle_by_midpoints(
    triangle: TriangleLike,
) -> tuple[Triangle, Triangle, Triangle, Triangle]:
    """Subdivide ``triangle`` into four triangles at the side midpoints.

    Examples:
        >>> import simetri.graphics as sg
        >>> parts = sg.subdivide_triangle_by_midpoints(((0, 0), (60, 0), (0, 60)))
        >>> len(parts)
        4
        >>> parts[0].vertices
        ((0.0, 0.0), (30.0, 0.0), (0.0, 30.0))
    """
    p1, p2, p3 = _triangle_vertices(triangle)
    m12 = midpoint(p1, p2)
    m23 = midpoint(p2, p3)
    m31 = midpoint(p3, p1)
    return (
        Triangle(p1, m12, m31),
        Triangle(p2, m23, m12),
        Triangle(p3, m31, m23),
        Triangle(m12, m23, m31),
    )


def subdivide_triangle_at_centroid(
    triangle: TriangleLike,
) -> tuple[Triangle, Triangle, Triangle]:
    """Subdivide ``triangle`` into three triangles at the centroid.

    Examples:
        >>> import simetri.graphics as sg
        >>> parts = sg.subdivide_triangle_at_centroid(((0, 0), (60, 0), (0, 60)))
        >>> len(parts)
        3
        >>> parts[0].vertices
        ((0.0, 0.0), (60.0, 0.0), (20.0, 20.0))
    """
    p1, p2, p3 = _triangle_vertices(triangle)
    center = triangle_centroid(p1, p2, p3)
    return (
        Triangle(p1, p2, center),
        Triangle(p2, p3, center),
        Triangle(p3, p1, center),
    )


def subdivide_triangle_at_point(
    triangle: TriangleLike, point: PointType
) -> tuple[Triangle, Triangle, Triangle]:
    """Subdivide ``triangle`` into three triangles at ``point``.

    Examples:
        >>> import simetri.graphics as sg
        >>> parts = sg.subdivide_triangle_at_point(((0, 0), (60, 0), (0, 60)), (15, 15))
        >>> len(parts)
        3
        >>> parts[0].vertices
        ((0.0, 0.0), (60.0, 0.0), (15.0, 15.0))
    """
    p1, p2, p3 = _triangle_vertices(triangle)
    if not triangle_contains_point_inclusive((p1, p2, p3), point):
        raise ValueError("Point is outside the triangle.")
    interior = _xy(point)
    return (
        Triangle(p1, p2, interior),
        Triangle(p2, p3, interior),
        Triangle(p3, p1, interior),
    )


def similar_triangles(
    triangle1: TriangleLike,
    triangle2: TriangleLike,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if the two triangles have proportional side lengths.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.similar_triangles(
        ...     ((0, 0), (30, 0), (0, 40)), ((0, 0), (60, 0), (0, 80))
        ... )
        True
        >>> sg.similar_triangles(
        ...     ((0, 0), (60, 0), (0, 80)), ((0, 0), (60, 0), (0, 60))
        ... )
        False
    """
    rel_tol, abs_tol = resolve_tol(rel_tol, abs_tol)
    p1, p2, p3 = _triangle_vertices(triangle1)
    q1, q2, q3 = _triangle_vertices(triangle2)
    if is_degenerate_triangle(p1, p2, p3) or is_degenerate_triangle(q1, q2, q3):
        return False
    sides1 = sorted(_side_lengths(p1, p2, p3))
    sides2 = sorted(_side_lengths(q1, q2, q3))
    if sides1[0] == 0.0 or sides2[0] == 0.0:
        return False
    ratio = sides1[0] / sides2[0]
    return all(
        isclose(side1, ratio * side2, rel_tol=rel_tol, abs_tol=abs_tol)
        for side1, side2 in zip(sides1, sides2)
    )


class Triangle(Shape):
    """A three-vertex closed shape with triangle geometry helpers.

    Methods wrap the module-level ``triangle_*`` functions, the same pattern
    as ``Vector`` wrapping ``v_*``.

    Examples:
        >>> import simetri.graphics as sg
        >>> tri = sg.Triangle((0, 0), (60, 0), (0, 80))
        >>> tri.area()
        2400.0
        >>> tri.side_lengths()
        (100.0, 80.0, 60.0)
        >>> tri.is_right()
        True
    """

    def __init__(
        self,
        p1: PointType | Sequence[PointType],
        p2: PointType | None = None,
        p3: PointType | None = None,
        **kwargs: object,
    ) -> None:
        """Create a closed triangle.

        Args:
            p1: First vertex, or a sequence of three vertices.
            p2: Second vertex when ``p1`` is a single point.
            p3: Third vertex when ``p1`` is a single point.
            **kwargs: Passed to ``Shape``.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Triangle((0, 0), (60, 0), (0, 60)).vertices
            ((0.0, 0.0), (60.0, 0.0), (0.0, 60.0))
            >>> sg.Triangle([(0, 0), (60, 0), (0, 60)]).vertices
            ((0.0, 0.0), (60.0, 0.0), (0.0, 60.0))
        """
        if p2 is None and p3 is None:
            points = list(p1)
        elif p2 is None or p3 is None:
            raise ValueError("Triangle requires three vertices.")
        else:
            if not (
                check_position(p1) and check_position(p2) and check_position(p3)
            ):
                raise ValueError("Triangle vertices must be points.")
            points = [p1, p2, p3]
        if len(points) != 3:
            raise ValueError("Triangle requires three vertices.")
        kwargs = dict(kwargs)
        kwargs.setdefault("closed", True)
        kwargs.setdefault("subtype", Types.TRIANGLE)
        super().__init__(points, **kwargs)
        self.subtype = Types.TRIANGLE

    def __repr__(self) -> str:
        """Return a Triangle string from this triangle's vertices.

        Examples:
            >>> import simetri.graphics as sg
            >>> repr(sg.Triangle((0, 0), (60, 0), (0, 60)))
            'Triangle(((0.0, 0.0), (60.0, 0.0), (0.0, 60.0)))'
            >>> str(sg.Triangle((0, 0), (60, 0), (0, 60))).startswith("Shape")
            True
        """
        if len(self.primary_points) == 0:
            return "Triangle()"
        if len(self.primary_points) < 4:
            return f"Triangle({self.vertices})"
        return f"Triangle([{self.vertices[0]}, ..., {self.vertices[-1]}])"

    @property
    def p1(self) -> PointType:
        """First vertex."""
        return self.vertices[0]

    @property
    def p2(self) -> PointType:
        """Second vertex."""
        return self.vertices[1]

    @property
    def p3(self) -> PointType:
        """Third vertex."""
        return self.vertices[2]

    def _pts(self) -> tuple[PointType, PointType, PointType]:
        return _triangle_vertices(self)

    def signed_area(self) -> float:
        """Return the signed area."""
        return triangle_signed_area(*self._pts())

    def area(self) -> float:
        """Return the unsigned area."""
        return triangle_area(*self._pts())

    def perimeter(self) -> float:
        """Return the perimeter."""
        return triangle_perimeter(*self._pts())

    def semiperimeter(self) -> float:
        """Return the semiperimeter."""
        return triangle_semiperimeter(*self._pts())

    def side_lengths(self) -> tuple[float, float, float]:
        """Return side lengths opposite ``p1``, ``p2``, and ``p3``."""
        return triangle_side_lengths(*self._pts())

    def longest_edge(self) -> tuple[PointType, PointType]:
        """Return the longest edge as a pair of vertices.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Triangle((0, 0), (60, 0), (0, 80)).longest_edge()
            ((60.0, 0.0), (0.0, 80.0))
        """
        return triangle_longest_edge(*self._pts())

    def shortest_edge(self) -> tuple[PointType, PointType]:
        """Return the shortest edge as a pair of vertices.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.Triangle((0, 0), (60, 0), (0, 80)).shortest_edge()
            ((0.0, 0.0), (60.0, 0.0))
        """
        return triangle_shortest_edge(*self._pts())

    def opposite_edge(
        self, vertex_index: PointType | int
    ) -> tuple[PointType, PointType]:
        """Return the edge opposite a vertex or vertex index.

        Examples:
            >>> import simetri.graphics as sg
            >>> tri = sg.Triangle((0, 0), (60, 0), (0, 80))
            >>> tri.opposite_edge(0)
            ((60.0, 0.0), (0.0, 80.0))
            >>> tri.opposite_edge(tri.p2)
            ((0.0, 80.0), (0.0, 0.0))
            >>> tri.opposite_edge(2)
            ((0.0, 0.0), (60.0, 0.0))
        """
        return triangle_opposite_edge(*self._pts(), vertex_index)

    def altitudes(self) -> tuple[float, float, float]:
        """Return altitude lengths opposite ``p1``, ``p2``, and ``p3``."""
        return triangle_altitudes(*self._pts())

    def medians(self) -> tuple[float, float, float]:
        """Return median lengths from ``p1``, ``p2``, and ``p3``."""
        return triangle_medians(*self._pts())

    def angles(self) -> tuple[float, float, float]:
        """Return interior angles at ``p1``, ``p2``, and ``p3``."""
        return triangle_angles(*self._pts())

    def angle_at(self, vertex: PointType | int = 0) -> float:
        """Return the interior angle at a vertex or vertex index."""
        p1, p2, p3 = self._pts()
        if vertex in (0, p1) or (
            not isinstance(vertex, int) and equal_points(vertex, p1)
        ):
            return triangle_angle_at(p1, p2, p3)
        if vertex in (1, p2) or (
            not isinstance(vertex, int) and equal_points(vertex, p2)
        ):
            return triangle_angle_at(p2, p3, p1)
        if vertex in (2, p3) or (
            not isinstance(vertex, int) and equal_points(vertex, p3)
        ):
            return triangle_angle_at(p3, p1, p2)
        if isinstance(vertex, int):
            raise ValueError("vertex index must be 0, 1, or 2.")
        raise ValueError("vertex is not a vertex of this triangle.")

    @staticmethod
    def area_from_sides(side1: float, side2: float, side3: float) -> float:
        """Return area from three side lengths."""
        return triangle_area_from_sides(side1, side2, side3)

    @staticmethod
    def angles_from_sides(
        side1: float, side2: float, side3: float
    ) -> tuple[float, float, float]:
        """Return interior angles from three side lengths."""
        return triangle_angles_from_sides(side1, side2, side3)

    @staticmethod
    def side_from_sas(side1: float, angle: float, side2: float) -> float:
        """Return the side opposite the included angle (SAS)."""
        return triangle_side_from_sas(side1, angle, side2)

    @staticmethod
    def side_from_aas(angle_a: float, angle_b: float, side_a: float) -> float:
        """Return the side opposite ``angle_b`` (AAS)."""
        return triangle_side_from_aas(angle_a, angle_b, side_a)

    def is_valid(
        self, rel_tol: float | None = None, abs_tol: float | None = None
    ) -> bool:
        """Return True if this triangle has positive area."""
        return is_valid_triangle(*self._pts(), rel_tol, abs_tol)

    def is_degenerate(
        self, rel_tol: float | None = None, abs_tol: float | None = None
    ) -> bool:
        """Return True if this triangle has zero area."""
        return is_degenerate_triangle(*self._pts(), rel_tol, abs_tol)

    def is_collinear(
        self, rel_tol: float | None = None, abs_tol: float | None = None
    ) -> bool:
        """Return True if the vertices are collinear."""
        return is_collinear(*self._pts(), rel_tol, abs_tol)

    def is_equilateral(
        self, rel_tol: float | None = None, abs_tol: float | None = None
    ) -> bool:
        """Return True if all sides are equal."""
        return is_equilateral_triangle(*self._pts(), rel_tol, abs_tol)

    def is_isosceles(
        self, rel_tol: float | None = None, abs_tol: float | None = None
    ) -> bool:
        """Return True if at least two sides are equal."""
        return is_isosceles_triangle(*self._pts(), rel_tol, abs_tol)

    def is_scalene(
        self, rel_tol: float | None = None, abs_tol: float | None = None
    ) -> bool:
        """Return True if no two sides are equal."""
        return is_scalene_triangle(*self._pts(), rel_tol, abs_tol)

    def is_right(
        self, rel_tol: float | None = None, abs_tol: float | None = None
    ) -> bool:
        """Return True if one angle is a right angle."""
        return is_right_triangle(*self._pts(), rel_tol, abs_tol)

    def is_acute(
        self, rel_tol: float | None = None, abs_tol: float | None = None
    ) -> bool:
        """Return True if all angles are acute."""
        return is_acute_triangle(*self._pts(), rel_tol, abs_tol)

    def is_obtuse(
        self, rel_tol: float | None = None, abs_tol: float | None = None
    ) -> bool:
        """Return True if one angle is obtuse."""
        return is_obtuse_triangle(*self._pts(), rel_tol, abs_tol)

    def orientation(
        self, rel_tol: float | None = None, abs_tol: float | None = None
    ) -> int:
        """Return ``1`` if CCW, ``-1`` if CW, ``0`` if collinear."""
        return triangle_orientation(*self._pts(), rel_tol, abs_tol)

    def clockwise(
        self, rel_tol: float | None = None, abs_tol: float | None = None
    ) -> bool:
        """Return True if the vertices are clockwise."""
        return is_clockwise(*self._pts(), rel_tol, abs_tol)

    def counterclockwise(
        self, rel_tol: float | None = None, abs_tol: float | None = None
    ) -> bool:
        """Return True if the vertices are counter-clockwise."""
        return is_counterclockwise(*self._pts(), rel_tol, abs_tol)

    def reverse_winding(self) -> Triangle:
        """Return a new triangle with reversed vertex order."""
        return reverse_winding(self)

    def centroid(self) -> PointType:
        """Return the centroid."""
        return triangle_centroid(*self._pts())

    def incenter(self) -> PointType:
        """Return the incenter."""
        return triangle_incenter(*self._pts())

    def circumcenter(self) -> PointType:
        """Return the circumcenter."""
        return triangle_circumcenter(*self._pts())

    def orthocenter(self) -> PointType:
        """Return the orthocenter."""
        return triangle_orthocenter(*self._pts())

    def nine_point_center(self) -> PointType:
        """Return the nine-point center."""
        return triangle_nine_point_center(*self._pts())

    def excenters(self) -> tuple[PointType, PointType, PointType]:
        """Return the three excenters."""
        return triangle_excenters(*self._pts())

    def inradius(self) -> float:
        """Return the inradius."""
        return triangle_inradius(*self._pts())

    def circumradius(self) -> float:
        """Return the circumradius."""
        return triangle_circumradius(*self._pts())

    def incircle(self) -> Circle:
        """Return the incircle."""
        return triangle_incircle(self)

    def circumcircle(self) -> Circle:
        """Return the circumcircle."""
        return triangle_circumcircle(self)

    def nine_point_circle(self) -> Circle:
        """Return the nine-point circle."""
        return triangle_nine_point_circle(self)

    def barycentric_coordinates(
        self, point: PointType
    ) -> tuple[float, float, float]:
        """Return barycentric weights of ``point``."""
        return barycentric_coordinates(point, self)

    def cartesian_from_barycentric(self, weights: Sequence[float]) -> PointType:
        """Return the Cartesian point for barycentric ``weights``."""
        return cartesian_from_barycentric(weights, self)

    def interpolate(self, values: Sequence[Any], point: PointType) -> Any:
        """Interpolate ``values`` at ``point``."""
        return interpolate_triangle_at_point(point, self, values)

    def contains_point(self, point: PointType) -> bool:
        """Return True if ``point`` is inside or on the boundary."""
        return triangle_contains_point(self, point)

    def contains_point_strictly(self, point: PointType) -> bool:
        """Return True if ``point`` is strictly inside."""
        return triangle_contains_point_strictly(self, point)

    def contains_point_inclusive(self, point: PointType) -> bool:
        """Return True if ``point`` is inside or on the boundary."""
        return triangle_contains_point_inclusive(self, point)

    def classify_point(self, point: PointType) -> TrianglePointClass:
        """Classify ``point`` relative to this triangle."""
        return classify_point_in_triangle(self, point)

    def closest_point(self, point: PointType) -> PointType:
        """Return the closest point of this triangle to ``point``."""
        return closest_point_on_triangle(point, self)

    def distance_to_point(self, point: PointType) -> float:
        """Return the distance from ``point`` to this triangle."""
        return distance_to_triangle(point, self)

    def intersect_ray(self, ray: LineType) -> RayTriangleHit | None:
        """Intersect a ray with this triangle."""
        return intersect_ray_triangle(ray, self)

    def intersect_line(self, line: LineType) -> TriangleIntersection:
        """Intersect an infinite line with this triangle."""
        return intersect_line_triangle(line, self)

    def intersect_segment(self, segment: LineType) -> TriangleIntersection:
        """Intersect a segment with this triangle."""
        return intersect_segment_triangle(segment, self)

    def intersect_triangle(self, other: TriangleLike) -> TriangleIntersection:
        """Intersect another triangle with this triangle."""
        return intersect_triangle_triangle(self, other)

    def subdivide_by_midpoints(
        self,
    ) -> tuple[Triangle, Triangle, Triangle, Triangle]:
        """Subdivide at the side midpoints."""
        return subdivide_triangle_by_midpoints(self)

    def subdivide_at_centroid(self) -> tuple[Triangle, Triangle, Triangle]:
        """Subdivide at the centroid."""
        return subdivide_triangle_at_centroid(self)

    def subdivide_at_point(
        self, point: PointType
    ) -> tuple[Triangle, Triangle, Triangle]:
        """Subdivide at ``point``."""
        return subdivide_triangle_at_point(self, point)

    def similar(
        self,
        other: TriangleLike,
        rel_tol: float | None = None,
        abs_tol: float | None = None,
    ) -> bool:
        """Return True if ``other`` has proportional side lengths."""
        return similar_triangles(self, other, rel_tol, abs_tol)

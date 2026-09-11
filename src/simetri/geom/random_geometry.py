"""Random geometric entities."""

from __future__ import annotations

import random
from collections.abc import Sequence
from itertools import product
from math import atan2, cos, isclose, pi, sin, tau

import numpy as np

from ..config.settings import defaults
from ..group.batch import Group
from ..shapes.shape import Shape
from .geometry import double_area3, normalize_angle
from .polygons.polygon import polygon_internal_angles
from .polygons.polygon_utils import is_ccw, is_simple
from .vectors import distance

MIN_LENGTH = 20
MAX_LENGTH = 80
MIN_EDGE_LENGTH = 20
MAX_EDGE_LENGTH = 60
MIN_ANGLE = 0
MAX_ANGLE = tau
MIN_GEOM_ANGLE = pi / 8
MAX_GEOM_ANGLE = 9 * pi / 8
MAX_TRIANGLE_ANGLE = 5 * pi / 6
MIN_X = 0
MIN_Y = 0
MAX_X = 200
MAX_Y = 150
N = 5
N_MIN_EDGES = 3
N_MAX_EDGES = 6


def _resolve_rng(
    seed: int | None = None,
    rng: random.Random | None = None,
) -> random.Random:
    """Return ``rng`` if given, otherwise a new ``Random(seed)``.

    A local generator never calls ``random.seed`` on the process-global RNG.
    ``seed=None`` draws from OS entropy.
    """
    if rng is not None:
        return rng
    return random.Random(seed)


def random_angle(
    min_angle: float = MIN_ANGLE,
    max_angle: float = MAX_ANGLE,
    incr: float | None = None,
    normalized: bool = True,
    seed: int | None = None,
    *,
    rng: random.Random | None = None,
) -> float:
    """Return a random angle in radians within the requested range.

    If ``incr`` is None, the angle is sampled uniformly from the continuous
    interval. Otherwise, it is chosen from ``min_angle + k * incr`` without
    exceeding ``max_angle``. If ``normalized`` is True, the selected angle
    is converted to ``(-pi, pi]``.

    Args:
        min_angle (float, optional): Minimum angle. Defaults to ``MIN_ANGLE``.
        max_angle (float, optional): Maximum angle. Defaults to ``MAX_ANGLE``.
        incr (float, optional): Increment between discrete choices.
            Defaults to None.
        normalized (bool, optional): Normalize the result to ``(-pi, pi]``.
            Defaults to True.
        seed (int, optional): Seed for a local RNG. Defaults to None.
        rng (random.Random, optional): Existing generator to use. Defaults to
            None. When set, ``seed`` is ignored.

    Returns:
        float: The selected angle.

    Raises:
        ValueError: If ``min_angle`` exceeds ``max_angle`` or ``incr`` is
            not positive.

    Examples:
        >>> import simetri.graphics as sg
        >>> angle = sg.random_angle()
        >>> -sg.pi < angle <= sg.pi
        True
        >>> stepped = sg.random_angle(
        ...     0, sg.pi, incr=sg.pi / 2, normalized=False
        ... )
        >>> any(
        ...     sg.isclose(stepped, option)
        ...     for option in (0, sg.pi / 2, sg.pi)
        ... )
        True
        >>> deg_incr_rand_angle = sg.random_angle(incr=sg.radians(1))
        >>> sg.degrees(deg_incr_rand_angle).is_integer()
        True
        >>> sg.random_angle(seed=1) == sg.random_angle(seed=1)
        True
    """
    rng = _resolve_rng(seed, rng)
    if min_angle > max_angle:
        raise ValueError(f"min_angle ({min_angle}) > max_angle ({max_angle})")
    if incr is None:
        res = rng.uniform(min_angle, max_angle)
    else:
        if incr <= 0:
            raise ValueError(f"incr ({incr}) must be positive")
        options = []
        index = 0
        option = min_angle
        while option < max_angle or isclose(option, max_angle):
            options.append(min(option, max_angle))
            index += 1
            option = min_angle + index * incr
        res = rng.choice(options)

    if normalized:
        res = normalize_angle(res)

    return float(res)


def random_point(
    min_x: float = MIN_X,
    min_y: float = MIN_Y,
    max_x: float = MAX_X,
    max_y: float = MAX_Y,
    seed: int | None = None,
    *,
    rng: random.Random | None = None,
) -> tuple:
    """Return a random point within the given axis-aligned limits.

    Args:
        min_x (float, optional): Minimum x. Defaults to ``MIN_X``.
        min_y (float, optional): Minimum y. Defaults to ``MIN_Y``.
        max_x (float, optional): Maximum x. Defaults to ``MAX_X``.
        max_y (float, optional): Maximum y. Defaults to ``MAX_Y``.
        seed (int, optional): Seed for a local RNG. Defaults to None.
        rng (random.Random, optional): Existing generator to use. Defaults to
            None. When set, ``seed`` is ignored.

    Returns:
        tuple: A point ``(x, y)`` with each coordinate in its range.

    Examples:
        >>> import simetri.graphics as sg
        >>> x, y = sg.random_point(0, 0, 10, 20)
        >>> 0 <= x <= 10 and 0 <= y <= 20
        True
        >>> sg.random_point(seed=7) == sg.random_point(seed=7)
        True
    """
    rng = _resolve_rng(seed, rng)
    return (rng.uniform(min_x, max_x), rng.uniform(min_y, max_y))


def random_points(
    n: int = N,
    min_x: float = MIN_X,
    min_y: float = MIN_Y,
    max_x: float = MAX_X,
    max_y: float = MAX_Y,
    seed: int | None = None,
    *,
    rng: random.Random | None = None,
) -> list[tuple]:
    """Return ``n`` random points within the given axis-aligned limits.

    Args:
        n (int, optional): Number of points. Defaults to ``N``.
        min_x (float, optional): Minimum x. Defaults to ``MIN_X``.
        min_y (float, optional): Minimum y. Defaults to ``MIN_Y``.
        max_x (float, optional): Maximum x. Defaults to ``MAX_X``.
        max_y (float, optional): Maximum y. Defaults to ``MAX_Y``.
        seed (int, optional): Seed for a local RNG. Defaults to None.
        rng (random.Random, optional): Existing generator to use. Defaults to
            None. When set, ``seed`` is ignored.

    Returns:
        list[tuple]: ``n`` points ``(x, y)``.

    Examples:
        >>> import simetri.graphics as sg
        >>> points = sg.random_points(5, 0, 0, 1, 1)
        >>> len(points)
        5
        >>> all(0 <= x <= 1 and 0 <= y <= 1 for x, y in points)
        True
        >>> sg.random_points(seed=3) == sg.random_points(seed=3)
        True
    """
    rng = _resolve_rng(seed, rng)
    return [
        random_point(min_x, min_y, max_x, max_y, rng=rng) for _ in range(n)
    ]


def random_segment(
    min_length: float = MIN_LENGTH,
    max_length: float = MAX_LENGTH,
    min_x: float = MIN_X,
    min_y: float = MIN_Y,
    max_x: float = MAX_X,
    max_y: float = MAX_Y,
    min_angle: float = MIN_GEOM_ANGLE,
    max_angle: float = MAX_GEOM_ANGLE,
    seed: int | None = None,
    *,
    rng: random.Random | None = None,
) -> Shape:
    """Return a randomly sized and positioned segment.

    Segment length is in ``[min_length, max_length]``. Inclination angle
    is in ``[min_angle, max_angle]``. The segment is created locally, then
    its center is moved to a random position within the coordinate limits.

    Args:
        min_length (float, optional): Minimum segment length.
            Defaults to ``MIN_LENGTH``.
        max_length (float, optional): Maximum segment length.
            Defaults to ``MAX_LENGTH``.
        min_x (float, optional): Minimum center x-position.
            Defaults to ``MIN_X``.
        min_y (float, optional): Minimum center y-position.
            Defaults to ``MIN_Y``.
        max_x (float, optional): Maximum center x-position.
            Defaults to ``MAX_X``.
        max_y (float, optional): Maximum center y-position.
            Defaults to ``MAX_Y``.
        min_angle (float, optional): Minimum inclination angle in radians.
            Defaults to ``MIN_GEOM_ANGLE``.
        max_angle (float, optional): Maximum inclination angle in radians.
            Defaults to ``MAX_GEOM_ANGLE``.
        seed (int, optional): Seed for a local RNG. Defaults to None.
        rng (random.Random, optional): Existing generator to use. Defaults to
            None. When set, ``seed`` is ignored.

    Returns:
        Shape: An open two-point shape.

    Raises:
        ValueError: If ``min_length`` exceeds ``max_length``, or if the
            inclination-angle range does not intersect ``[0, pi]``.

    Examples:
        >>> import simetri.graphics as sg
        >>> segment = sg.random_segment(1, 5, 0, 0, 10, 10)
        >>> len(segment.vertices)
        2
        >>> 1 <= sg.distance(segment.vertices[0], segment.vertices[1]) <= 5
        True
    """
    rng = _resolve_rng(seed, rng)
    if min_length > max_length:
        raise ValueError(
            f"min_length ({min_length}) > max_length ({max_length})"
        )
    if min_angle > max_angle:
        raise ValueError(f"min_angle ({min_angle}) > max_angle ({max_angle})")
    effective_min_angle = max(min_angle, 0)
    effective_max_angle = min(max_angle, pi)
    if effective_min_angle > effective_max_angle:
        raise ValueError(
            f"inclination-angle range [{min_angle}, {max_angle}] does not "
            "intersect [0, pi]"
        )
    length = rng.uniform(min_length, max_length)
    angle = rng.uniform(effective_min_angle, effective_max_angle)
    segment = Shape([(0, 0), (length * cos(angle), length * sin(angle))])
    segment.move_to(random_point(min_x, min_y, max_x, max_y, rng=rng))
    return segment


def random_segments(
    n: int = N,
    min_length: float = MIN_LENGTH,
    max_length: float = MAX_LENGTH,
    min_x: float = MIN_X,
    min_y: float = MIN_Y,
    max_x: float = MAX_X,
    max_y: float = MAX_Y,
    min_angle: float = MIN_GEOM_ANGLE,
    max_angle: float = MAX_GEOM_ANGLE,
    angles: Sequence[float] | None = None,
    seed: int | None = None,
    *,
    rng: random.Random | None = None,
) -> Group:
    """Return ``n`` random segments in a ``Group``.

    Inclination angles are constrained by ``min_angle`` and ``max_angle``,
    unless ``angles`` is given. Then each segment picks one inclination from
    ``angles`` with ``rng.choice`` and uses that exact value.

    Args:
        n (int, optional): Number of segments. Defaults to ``N``.
        min_length (float, optional): Minimum segment length.
            Defaults to ``MIN_LENGTH``.
        max_length (float, optional): Maximum segment length.
            Defaults to ``MAX_LENGTH``.
        min_x (float, optional): Minimum x. Defaults to ``MIN_X``.
        min_y (float, optional): Minimum y. Defaults to ``MIN_Y``.
        max_x (float, optional): Maximum x. Defaults to ``MAX_X``.
        max_y (float, optional): Maximum y. Defaults to ``MAX_Y``.
        min_angle (float, optional): Minimum inclination angle in radians.
            Used when ``angles`` is None. Defaults to ``MIN_GEOM_ANGLE``.
        max_angle (float, optional): Maximum inclination angle in radians.
            Used when ``angles`` is None. Defaults to ``MAX_GEOM_ANGLE``.
        angles (Sequence[float], optional): Exact inclination angles to
            choose from. Defaults to None.
        seed (int, optional): Seed for a local RNG. Defaults to None.
        rng (random.Random, optional): Existing generator to use. Defaults to
            None. When set, ``seed`` is ignored.

    Returns:
        Group: ``n`` segment shapes.

    Examples:
        >>> import simetri.graphics as sg
        >>> segments = sg.random_segments(3, 1, 5, 0, 0, 10, 10)
        >>> len(segments)
        3
        >>> fixed = sg.random_segments(5, 1, 8, 0, 0, 20, 20, angles=[sg.pi / 4])
        >>> len(fixed)
        5
        >>> all(
        ...     sg.isclose(
        ...         sg.atan2(
        ...             segment.vertices[1][1] - segment.vertices[0][1],
        ...             segment.vertices[1][0] - segment.vertices[0][0],
        ...         )
        ...         % sg.pi,
        ...         sg.pi / 4,
        ...         abs_tol=1e-9,
        ...     )
        ...     for segment in fixed
        ... )
        True
    """
    rng = _resolve_rng(seed, rng)
    segments = []
    for _ in range(n):
        if angles is not None:
            angle = rng.choice(angles)
            segment_min_angle = angle
            segment_max_angle = angle
        else:
            segment_min_angle = min_angle
            segment_max_angle = max_angle
        segments.append(
            random_segment(
                min_length,
                max_length,
                min_x,
                min_y,
                max_x,
                max_y,
                segment_min_angle,
                segment_max_angle,
                rng=rng,
            )
        )
    return Group(segments)


def random_rectangle(
    min_edge_length: float = MIN_EDGE_LENGTH,
    max_edge_length: float = MAX_EDGE_LENGTH,
    min_x: float = MIN_X,
    min_y: float = MIN_Y,
    max_x: float = MAX_X,
    max_y: float = MAX_Y,
    axis_aligned: bool = True,
    seed: int | None = None,
    *,
    rng: random.Random | None = None,
) -> Shape:
    """Return a randomly sized and positioned rectangle.

    Edge lengths are sampled from ``[min_edge_length, max_edge_length]``.
    If ``axis_aligned`` is False, the rectangle is rotated by a random
    angle. Its center is then moved to a random position within the
    coordinate limits.

    Args:
        min_edge_length (float, optional): Minimum side length.
            Defaults to ``MIN_EDGE_LENGTH``.
        max_edge_length (float, optional): Maximum side length.
            Defaults to ``MAX_EDGE_LENGTH``.
        min_x (float, optional): Minimum center x-position.
            Defaults to ``MIN_X``.
        min_y (float, optional): Minimum center y-position.
            Defaults to ``MIN_Y``.
        max_x (float, optional): Maximum center x-position.
            Defaults to ``MAX_X``.
        max_y (float, optional): Maximum center y-position.
            Defaults to ``MAX_Y``.
        axis_aligned (bool, optional): Keep sides parallel to the axes.
            Defaults to True.
        seed (int, optional): Seed for a local RNG. Defaults to None.
        rng (random.Random, optional): Existing generator to use. Defaults to
            None. When set, ``seed`` is ignored.

    Returns:
        Shape: A closed four-vertex rectangle.

    Raises:
        ValueError: If ``min_edge_length`` exceeds ``max_edge_length``.

    Examples:
        >>> import simetri.graphics as sg
        >>> rectangle = sg.random_rectangle(2, 8, 0, 0, 20, 20)
        >>> len(rectangle.vertices)
        4
        >>> rectangle.closed
        True
    """
    rng = _resolve_rng(seed, rng)
    if min_edge_length > max_edge_length:
        raise ValueError(
            f"min_edge_length ({min_edge_length}) > "
            f"max_edge_length ({max_edge_length})"
        )
    width = rng.uniform(min_edge_length, max_edge_length)
    height = rng.uniform(min_edge_length, max_edge_length)
    half_width = width / 2
    half_height = height / 2
    vertices = [
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
    ]
    rectangle = Shape(vertices, closed=True)
    if not axis_aligned:
        rectangle.rotate(rng.uniform(0, tau))
    rectangle.move_to(random_point(min_x, min_y, max_x, max_y, rng=rng))
    return rectangle


def random_rectangles(
    n: int = N,
    min_edge_length: float = MIN_EDGE_LENGTH,
    max_edge_length: float = MAX_EDGE_LENGTH,
    min_x: float = MIN_X,
    min_y: float = MIN_Y,
    max_x: float = MAX_X,
    max_y: float = MAX_Y,
    axis_aligned: bool = True,
    seed: int | None = None,
    *,
    rng: random.Random | None = None,
) -> Group:
    """Return ``n`` random rectangles in a ``Group``.

    Args:
        n (int, optional): Number of rectangles. Defaults to ``N``.
        min_edge_length (float, optional): Minimum side length.
            Defaults to ``MIN_EDGE_LENGTH``.
        max_edge_length (float, optional): Maximum side length.
            Defaults to ``MAX_EDGE_LENGTH``.
        min_x (float, optional): Minimum x. Defaults to ``MIN_X``.
        min_y (float, optional): Minimum y. Defaults to ``MIN_Y``.
        max_x (float, optional): Maximum x. Defaults to ``MAX_X``.
        max_y (float, optional): Maximum y. Defaults to ``MAX_Y``.
        axis_aligned (bool, optional): Keep sides parallel to the axes.
            Defaults to True.
        seed (int, optional): Seed for a local RNG. Defaults to None.
        rng (random.Random, optional): Existing generator to use. Defaults to
            None. When set, ``seed`` is ignored.

    Returns:
        Group: ``n`` closed rectangle shapes.

    Examples:
        >>> import simetri.graphics as sg
        >>> rectangles = sg.random_rectangles(4, 2, 8, 0, 0, 30, 30)
        >>> len(rectangles)
        4
    """
    rng = _resolve_rng(seed, rng)
    return Group(
        [
            random_rectangle(
                min_edge_length,
                max_edge_length,
                min_x,
                min_y,
                max_x,
                max_y,
                axis_aligned,
                rng=rng,
            )
            for _ in range(n)
        ]
    )


def random_triangle(
    min_edge_length: float = MIN_EDGE_LENGTH,
    max_edge_length: float = MAX_EDGE_LENGTH,
    min_x: float = MIN_X,
    min_y: float = MIN_Y,
    max_x: float = MAX_X,
    max_y: float = MAX_Y,
    min_angle: float = MIN_GEOM_ANGLE,
    max_angle: float = MAX_TRIANGLE_ANGLE,
    angles: Sequence[float] | None = None,
    seed: int | None = None,
    *,
    rng: random.Random | None = None,
) -> Shape:
    """Return a randomly sized and positioned triangle.

    All three edge lengths are in ``[min_edge_length, max_edge_length]``.
    Each interior angle is in ``[min_angle, max_angle]``. Degenerate
    (near-zero area) triangles are rejected. If ``angles`` is given, each
    interior angle is chosen from that sequence and the three angles must
    be able to sum to ``pi``. The triangle is created locally, then its
    center is moved to a random position within the coordinate limits.

    Args:
        min_edge_length (float, optional): Minimum edge length.
            Defaults to ``MIN_EDGE_LENGTH``.
        max_edge_length (float, optional): Maximum edge length.
            Defaults to ``MAX_EDGE_LENGTH``.
        min_x (float, optional): Minimum center x-position.
            Defaults to ``MIN_X``.
        min_y (float, optional): Minimum center y-position.
            Defaults to ``MIN_Y``.
        max_x (float, optional): Maximum center x-position.
            Defaults to ``MAX_X``.
        max_y (float, optional): Maximum center y-position.
            Defaults to ``MAX_Y``.
        min_angle (float, optional): Minimum interior angle in radians.
            Defaults to ``MIN_GEOM_ANGLE``.
        max_angle (float, optional): Maximum interior angle in radians.
            Defaults to ``MAX_TRIANGLE_ANGLE``.
        angles (Sequence[float], optional): Discrete interior angles to
            choose from. Defaults to None.
        seed (int, optional): Seed for a local RNG. Defaults to None.
        rng (random.Random, optional): Existing generator to use. Defaults to
            None. When set, ``seed`` is ignored.

    Returns:
        Shape: A closed three-vertex triangle.

    Raises:
        ValueError: If ``min_edge_length`` exceeds ``max_edge_length``,
            if ``min_angle`` exceeds ``max_angle``, if ``angles`` is empty,
            or if no three allowed angles can form a triangle satisfying
            the edge-length constraints.

    Examples:
        >>> import simetri.graphics as sg
        >>> triangle = sg.random_triangle(1, 8, 0, 0, 20, 20)
        >>> len(triangle.vertices)
        3
        >>> triangle.closed
        True
        >>> angles = sg.polygon_internal_angles(triangle.vertices)
        >>> all(sg.pi / 8 <= angle <= 5 * sg.pi / 6 for angle in angles)
        True
        >>> equilateral = sg.random_triangle(
        ...     1, 8, 0, 0, 20, 20, angles=[sg.pi / 3]
        ... )
        >>> all(
        ...     sg.isclose(angle, sg.pi / 3, abs_tol=1e-6)
        ...     for angle in sg.polygon_internal_angles(equilateral.vertices)
        ... )
        True
    """
    rng = _resolve_rng(seed, rng)
    if min_edge_length > max_edge_length:
        raise ValueError(
            f"min_edge_length ({min_edge_length}) > "
            f"max_edge_length ({max_edge_length})"
        )
    if min_angle > max_angle:
        raise ValueError(f"min_angle ({min_angle}) > max_angle ({max_angle})")
    angle_tolerance = defaults["angle_tol"]
    if (
        3 * min_angle > pi + angle_tolerance
        or 3 * max_angle < pi - angle_tolerance
    ):
        raise ValueError(
            f"three interior angles in [{min_angle}, {max_angle}] "
            "cannot sum to pi"
        )
    if angles is not None:
        if len(angles) == 0:
            raise ValueError("angles must be non-empty")
        candidates = [
            angle for angle in angles if min_angle <= angle <= max_angle
        ]
        if not candidates:
            raise ValueError(
                f"no angles in {list(angles)!r} lie in "
                f"[{min_angle}, {max_angle}]"
            )

        constructions = [
            interior_angles
            for interior_angles in product(candidates, repeat=3)
            if isclose(sum(interior_angles), pi, abs_tol=angle_tolerance)
        ]
        if not constructions:
            raise ValueError(f"no three angles from {list(angles)!r} sum to pi")

        rng.shuffle(constructions)
        for interior_angles in constructions:
            side_proportions = [sin(angle) for angle in interior_angles]
            if any(proportion <= 0 for proportion in side_proportions):
                continue
            minimum_scale = min_edge_length / min(side_proportions)
            maximum_scale = max_edge_length / max(side_proportions)
            if minimum_scale > maximum_scale and not isclose(
                minimum_scale, maximum_scale
            ):
                continue
            maximum_scale = max(maximum_scale, minimum_scale)

            side_ab = side_proportions[2]
            side_ac = side_proportions[1]
            angle_a = interior_angles[0]
            scale = rng.uniform(minimum_scale, maximum_scale)
            vertices = [
                (0.0, 0.0),
                (side_ab * scale, 0.0),
                (
                    side_ac * scale * cos(angle_a),
                    side_ac * scale * sin(angle_a),
                ),
            ]
            triangle = Shape(vertices, closed=True)
            triangle.rotate(rng.uniform(0, tau))
            triangle.move_to(random_point(min_x, min_y, max_x, max_y, rng=rng))
            return triangle

        raise ValueError(
            "the selected interior angles cannot satisfy the edge-length "
            "constraints"
        )

    area_tol = defaults["area_tol"]
    while True:
        side_ab = rng.uniform(min_edge_length, max_edge_length)
        side_ac = rng.uniform(min_edge_length, max_edge_length)
        angle_a = rng.uniform(min_angle, max_angle)
        vertex_a = (0.0, 0.0)
        vertex_b = (side_ab, 0.0)
        vertex_c = (
            side_ac * cos(angle_a),
            side_ac * sin(angle_a),
        )
        side_bc = distance(vertex_b, vertex_c)
        if not (min_edge_length <= side_bc <= max_edge_length):
            continue
        if abs(double_area3(vertex_a, vertex_b, vertex_c)) <= area_tol:
            continue
        vertices = [vertex_a, vertex_b, vertex_c]
        if not is_ccw(vertices):
            vertices = list(reversed(vertices))
        interior_angles = polygon_internal_angles(vertices)
        if any(
            angle < min_angle or angle > max_angle for angle in interior_angles
        ):
            continue
        triangle = Shape(vertices, closed=True)
        triangle.rotate(rng.uniform(0, tau))
        triangle.move_to(random_point(min_x, min_y, max_x, max_y, rng=rng))
        return triangle


def random_triangles(
    n: int = N,
    min_edge_length: float = MIN_EDGE_LENGTH,
    max_edge_length: float = MAX_EDGE_LENGTH,
    min_x: float = MIN_X,
    min_y: float = MIN_Y,
    max_x: float = MAX_X,
    max_y: float = MAX_Y,
    min_angle: float = MIN_GEOM_ANGLE,
    max_angle: float = MAX_TRIANGLE_ANGLE,
    angles: Sequence[float] | None = None,
    seed: int | None = None,
    *,
    rng: random.Random | None = None,
) -> Group:
    """Return ``n`` random triangles in a ``Group``.

    When ``angles`` is given, each triangle is built by choosing interior
    angles from that sequence (see ``random_triangle``).

    Args:
        n (int, optional): Number of triangles. Defaults to ``N``.
        min_edge_length (float, optional): Minimum edge length.
            Defaults to ``MIN_EDGE_LENGTH``.
        max_edge_length (float, optional): Maximum edge length.
            Defaults to ``MAX_EDGE_LENGTH``.
        min_x (float, optional): Minimum x. Defaults to ``MIN_X``.
        min_y (float, optional): Minimum y. Defaults to ``MIN_Y``.
        max_x (float, optional): Maximum x. Defaults to ``MAX_X``.
        max_y (float, optional): Maximum y. Defaults to ``MAX_Y``.
        min_angle (float, optional): Minimum interior angle in radians.
            Defaults to ``MIN_GEOM_ANGLE``.
        max_angle (float, optional): Maximum interior angle in radians.
            Defaults to ``MAX_TRIANGLE_ANGLE``.
        angles (Sequence[float], optional): Discrete interior angles to
            choose from. Defaults to None.
        seed (int, optional): Seed for a local RNG. Defaults to None.
        rng (random.Random, optional): Existing generator to use. Defaults to
            None. When set, ``seed`` is ignored.

    Returns:
        Group: ``n`` closed triangle shapes.

    Examples:
        >>> import simetri.graphics as sg
        >>> triangles = sg.random_triangles(3, 1, 8, 0, 0, 20, 20)
        >>> len(triangles)
        3
        >>> equilateral = sg.random_triangles(
        ...     2, 1, 8, 0, 0, 20, 20, angles=[sg.pi / 3]
        ... )
        >>> len(equilateral)
        2
    """
    rng = _resolve_rng(seed, rng)
    return Group(
        [
            random_triangle(
                min_edge_length,
                max_edge_length,
                min_x,
                min_y,
                max_x,
                max_y,
                min_angle,
                max_angle,
                angles,
                rng=rng,
            )
            for _ in range(n)
        ]
    )


def random_polygon(
    n_min_edges: int = N_MIN_EDGES,
    n_max_edges: int = N_MAX_EDGES,
    min_edge_length: float = MIN_EDGE_LENGTH,
    max_edge_length: float = MAX_EDGE_LENGTH,
    min_x: float = MIN_X,
    min_y: float = MIN_Y,
    max_x: float = MAX_X,
    max_y: float = MAX_Y,
    closed: bool = True,
    simple: bool = True,
    min_angle: float = MIN_GEOM_ANGLE,
    max_angle: float = MAX_GEOM_ANGLE,
    angles: Sequence[float] | None = None,
    seed: int | None = None,
    *,
    rng: random.Random | None = None,
) -> Shape:
    """Return a random polygon within the given axis-aligned limits.
    All angles are in radians.


    n_min_edges <= n_edges <= n_max_edges (n_edges picked randomly)
    interior angles should be chosen randomly and within the limits.
    if angles (a list of angle values) is given then for each interior
    angle random.choice(angles) should be used. If it is not possible to
    create a closed polygon with the given constraints then return
    the closest you found and issue a warning.
    Edge lengths determine the polygon size. The polygon is created locally,
    then its center is moved to a random position within the coordinate limits.

    Args:
        n_min_edges (int, optional): Minimum number of edges (and vertices).
            Defaults to ``N_MIN_EDGES``.
        n_max_edges (int, optional): Maximum number of edges (and vertices).
            Defaults to ``N_MAX_EDGES``.
        min_edge_length (float, optional): Minimum polygon edge length.
            Defaults to ``MIN_EDGE_LENGTH``.
        max_edge_length (float, optional): Maximum polygon edge length.
            Defaults to ``MAX_EDGE_LENGTH``.
        min_x (float, optional): Minimum center x-position.
            Defaults to ``MIN_X``.
        min_y (float, optional): Minimum center y-position.
            Defaults to ``MIN_Y``.
        max_x (float, optional): Maximum center x-position.
            Defaults to ``MAX_X``.
        max_y (float, optional): Maximum center y-position.
            Defaults to ``MAX_Y``.
        closed (bool, optional): Whether the shape is closed. Defaults to True.
        simple (bool, optional): Require a non-self-intersecting polygon.
            Defaults to True.
        min_angle (float, optional): Minimum interior angle in radians.
            Defaults to ``MIN_GEOM_ANGLE``.
        max_angle (float, optional): Maximum interior angle in radians.
            Defaults to ``MAX_GEOM_ANGLE``.
        angles (Sequence[float], optional): Discrete interior angles to
            choose from when building the polygon. Defaults to None.
        seed (int, optional): Seed for a local RNG. Defaults to None.
        rng (random.Random, optional): Existing generator to use. Defaults to
            None. When set, ``seed`` is ignored.

    Returns:
        Shape: A polygon with a random vertex count in the given range.

    Raises:
        ValueError: If ``n_min_edges`` is less than 3, greater than
            ``n_max_edges``, if ``min_angle`` exceeds ``max_angle``, if
            ``angles`` is empty, if no entry of ``angles`` lies in
            ``[min_angle, max_angle]``, if ``angles`` is given with
            ``closed=False``, or if no edge count in the requested range
            can close with interior angles from ``angles``.

    Examples:
        >>> import simetri.graphics as sg
        >>> polygon = sg.random_polygon(
        ...     3, 6, 1, 8, 0, 0, 20, 20
        ... )
        >>> 3 <= len(polygon.vertices) <= 6
        True
        >>> polygon.closed
        True
        >>> interior = sg.polygon_internal_angles(polygon.vertices)
        >>> all(sg.pi / 8 <= angle <= 9 * sg.pi / 8 for angle in interior)
        True
        >>> angled = sg.random_polygon(
        ...     4, 4, 1, 8, 0, 0, 50, 50, angles=[sg.pi / 2]
        ... )
        >>> len(angled.vertices)
        4
        >>> angled.closed
        True
        >>> try:
        ...     sg.random_polygon(
        ...         5, 5, 1, 8, 0, 0, 50, 50, angles=[sg.pi / 3]
        ...     )
        ... except ValueError as exc:
        ...     "can close" in str(exc)
        True
    """
    rng = _resolve_rng(seed, rng)
    if n_min_edges < 3:
        raise ValueError(f"n_min_edges ({n_min_edges}) must be >= 3")
    if n_min_edges > n_max_edges:
        raise ValueError(
            f"n_min_edges ({n_min_edges}) > n_max_edges ({n_max_edges})"
        )
    if min_edge_length > max_edge_length:
        raise ValueError(
            f"min_edge_length ({min_edge_length}) > "
            f"max_edge_length ({max_edge_length})"
        )
    if min_angle > max_angle:
        raise ValueError(f"min_angle ({min_angle}) > max_angle ({max_angle})")

    angle_tolerance = defaults["angle_tol"]
    distance_tolerance = defaults["dist_tol"]

    if angles is not None:
        if not closed:
            raise ValueError("angles requires closed=True")
        if len(angles) == 0:
            raise ValueError("angles must be non-empty")

        candidates = [
            angle for angle in angles if min_angle <= angle <= max_angle
        ]
        if not candidates:
            raise ValueError(
                f"no angles in {list(angles)!r} lie in "
                f"[{min_angle}, {max_angle}]"
            )

        constructions = []
        for edge_count in range(n_min_edges, n_max_edges + 1):
            required_sum = (edge_count - 2) * pi
            for interior_angles in product(candidates, repeat=edge_count):
                if isclose(
                    sum(interior_angles),
                    required_sum,
                    abs_tol=angle_tolerance,
                ):
                    constructions.append((edge_count, interior_angles))

        if not constructions:
            raise ValueError(
                f"no edge count in [{n_min_edges}, {n_max_edges}] can close "
                f"with interior angles from {list(angles)!r}"
            )

        rng.shuffle(constructions)
        for edge_count, interior_angles in constructions:
            first_heading = rng.uniform(0, tau)
            headings = [first_heading]
            for vertex_index in range(1, edge_count):
                headings.append(
                    headings[-1] + pi - interior_angles[vertex_index]
                )

            directions = np.array(
                [(cos(heading), sin(heading)) for heading in headings]
            )
            closure_matrix = np.vstack((directions.T, np.ones(edge_count)))
            edge_lengths = np.linalg.lstsq(
                closure_matrix,
                np.array([0.0, 0.0, 1.0]),
                rcond=None,
            )[0]
            if np.any(edge_lengths <= distance_tolerance):
                continue
            minimum_scale = min_edge_length / np.min(edge_lengths)
            maximum_scale = max_edge_length / np.max(edge_lengths)
            if minimum_scale > maximum_scale and not isclose(
                minimum_scale, maximum_scale
            ):
                continue
            maximum_scale = max(maximum_scale, minimum_scale)
            edge_lengths *= rng.uniform(minimum_scale, maximum_scale)

            vertices = [(0.0, 0.0)]
            for edge_index in range(edge_count - 1):
                previous_x, previous_y = vertices[-1][:2]
                direction_x, direction_y = directions[edge_index]
                vertices.append(
                    (
                        previous_x + edge_lengths[edge_index] * direction_x,
                        previous_y + edge_lengths[edge_index] * direction_y,
                    )
                )

            break
        else:
            raise ValueError(
                "the selected interior angles cannot form a polygon with "
                "positive edge lengths"
            )
    else:
        feasible_edge_counts = [
            edge_count
            for edge_count in range(n_min_edges, n_max_edges + 1)
            if edge_count * min_angle
            <= (edge_count - 2) * pi
            <= edge_count * max_angle
        ]
        if not feasible_edge_counts:
            raise ValueError(
                f"no edge count in [{n_min_edges}, {n_max_edges}] can close "
                f"with interior angles in [{min_angle}, {max_angle}]"
            )

        while True:
            edge_count = rng.choice(feasible_edge_counts)
            center_x = rng.uniform(-max_edge_length, max_edge_length)
            center_y = rng.uniform(-max_edge_length, max_edge_length)
            points = [
                (
                    rng.uniform(-max_edge_length, max_edge_length),
                    rng.uniform(-max_edge_length, max_edge_length),
                )
                for _ in range(edge_count)
            ]
            vertices = sorted(
                points,
                key=lambda point: atan2(
                    point[1] - center_y, point[0] - center_x
                ),
            )
            if simple and not is_simple(vertices):
                continue
            edge_lengths = [
                distance(point, vertices[(index + 1) % edge_count])
                for index, point in enumerate(vertices)
            ]
            if any(
                edge_length < min_edge_length or edge_length > max_edge_length
                for edge_length in edge_lengths
            ):
                continue
            interior_angles = polygon_internal_angles(vertices)
            if all(
                min_angle <= angle <= max_angle for angle in interior_angles
            ):
                break

    polygon = Shape(vertices, closed=closed)
    polygon.rotate(rng.uniform(0, tau))
    polygon.move_to(random_point(min_x, min_y, max_x, max_y, rng=rng))

    if angles is not None:
        actual_angles = polygon_internal_angles(polygon.vertices)
        if any(
            not any(
                isclose(actual, candidate, abs_tol=angle_tolerance)
                for candidate in candidates
            )
            for actual in actual_angles
        ):
            raise ValueError("constructed polygon does not preserve its angles")

    return polygon


def random_polygons(
    n: int = N,
    n_min_edges: int = N_MIN_EDGES,
    n_max_edges: int = N_MAX_EDGES,
    min_edge_length: float = MIN_EDGE_LENGTH,
    max_edge_length: float = MAX_EDGE_LENGTH,
    min_x: float = MIN_X,
    min_y: float = MIN_Y,
    max_x: float = MAX_X,
    max_y: float = MAX_Y,
    closed: bool = True,
    simple: bool = True,
    min_angle: float = MIN_GEOM_ANGLE,
    max_angle: float = MAX_GEOM_ANGLE,
    angles: Sequence[float] | None = None,
    seed: int | None = None,
    *,
    rng: random.Random | None = None,
) -> Group:
    """Return ``n`` random polygons in a ``Group``.

    When ``angles`` is given, each polygon is built by choosing interior
    angles from that sequence (see ``random_polygon``).

    Args:
        n (int, optional): Number of polygons. Defaults to ``N``.
        n_min_edges (int, optional): Minimum number of edges per polygon.
            Defaults to ``N_MIN_EDGES``.
        n_max_edges (int, optional): Maximum number of edges per polygon.
            Defaults to ``N_MAX_EDGES``.
        min_edge_length (float, optional): Minimum polygon edge length.
            Defaults to ``MIN_EDGE_LENGTH``.
        max_edge_length (float, optional): Maximum polygon edge length.
            Defaults to ``MAX_EDGE_LENGTH``.
        min_x (float, optional): Minimum center x-position.
            Defaults to ``MIN_X``.
        min_y (float, optional): Minimum center y-position.
            Defaults to ``MIN_Y``.
        max_x (float, optional): Maximum center x-position.
            Defaults to ``MAX_X``.
        max_y (float, optional): Maximum center y-position.
            Defaults to ``MAX_Y``.
        closed (bool, optional): Whether each shape is closed. Defaults to True.
        simple (bool, optional): Require non-self-intersecting polygons.
            Defaults to True.
        min_angle (float, optional): Minimum interior angle in radians.
            Defaults to ``MIN_GEOM_ANGLE``.
        max_angle (float, optional): Maximum interior angle in radians.
            Defaults to ``MAX_GEOM_ANGLE``.
        angles (Sequence[float], optional): Discrete interior angles to
            choose from when building each polygon. Defaults to None.
        seed (int, optional): Seed for a local RNG. Defaults to None.
        rng (random.Random, optional): Existing generator to use. Defaults to
            None. When set, ``seed`` is ignored.

    Returns:
        Group: ``n`` polygon shapes.

    Examples:
        >>> import simetri.graphics as sg
        >>> polygons = sg.random_polygons(
        ...     3, 3, 5, 1, 8, 0, 0, 30, 30
        ... )
        >>> len(polygons)
        3
        >>> angled = sg.random_polygons(
        ...     2, 4, 4, 1, 8, 0, 0, 40, 40, angles=[sg.pi / 2]
        ... )
        >>> len(angled)
        2
    """
    rng = _resolve_rng(seed, rng)

    return Group(
        [
            random_polygon(
                n_min_edges,
                n_max_edges,
                min_edge_length,
                max_edge_length,
                min_x,
                min_y,
                max_x,
                max_y,
                closed,
                simple,
                min_angle,
                max_angle,
                angles,
                rng=rng,
            )
            for _ in range(n)
        ]
    )

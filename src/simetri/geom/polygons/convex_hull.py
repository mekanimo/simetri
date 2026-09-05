"""Gift-wrapping (Jarvis march) convex hull.

Translated from TopCoder geometry notes:
https://www.topcoder.com/thrive/articles/Geometry%20Concepts%20part%202:%20%20Line%20Intersection%20and%20its%20Applications
"""

from collections.abc import Sequence

from ...config.settings import defaults

Point = tuple[float, float]


def _as_xy(point) -> Point:
    """Return ``(x, y)`` floats from a point-like value."""
    return float(point[0]), float(point[1])


def _sub(a: Point, b: Point) -> Point:
    """Return vector ``a - b``."""
    return a[0] - b[0], a[1] - b[1]


def _cross(a: Point, b: Point) -> float:
    """Return 2D cross product ``a_x * b_y - a_y * b_x``."""
    return a[0] * b[1] - a[1] * b[0]


def _dot(a: Point, b: Point) -> float:
    """Return 2D dot product."""
    return a[0] * b[0] + a[1] * b[1]


def convex_hull(points: Sequence, on_edge: bool = False) -> list[Point]:
    """Return the convex hull of 2D points in counter-clockwise order.

    Args:
        points: Input points as ``(x, y)`` pairs (extra coordinates ignored).
        on_edge: If True, include collinear boundary points and prefer the
            nearest candidate on an edge. If False, keep only extreme vertices
            and prefer the farthest collinear point.

    Returns:
        Hull vertices starting at the leftmost point, without repeating the
        start point at the end.

    Examples:
        ::

            from simetri.geom.convex_hull import convex_hull

            hull = convex_hull([(0, 0), (1, 0), (0.5, 0.5), (0, 1)])
            # [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    """
    if not points:
        return []

    x = [_as_xy(p) for p in points]
    n = len(x)
    if n == 1:
        return x

    inf = float(defaults.get("INF", float("inf")))

    # Leftmost point (lexicographic).
    p = min(range(n), key=lambda i: (x[i][0], x[i][1]))
    start = p
    used = [False] * n
    hull: list[Point] = []

    while True:
        hull.append(x[p])
        n_idx = -1
        dist = inf if on_edge else 0.0

        for i in range(n):
            if i == p or used[i]:
                continue
            if n_idx == -1:
                n_idx = i

            v_i = _sub(x[i], x[p])
            v_n = _sub(x[n_idx], x[p])
            cross = _cross(v_i, v_n)
            d = _dot(v_i, v_i)

            if cross < 0:
                n_idx = i
                dist = d
            elif cross == 0:
                if on_edge and d < dist:
                    dist = d
                    n_idx = i
                elif not on_edge and d > dist:
                    dist = d
                    n_idx = i

        p = n_idx
        used[p] = True
        if p == start:
            break

    return hull

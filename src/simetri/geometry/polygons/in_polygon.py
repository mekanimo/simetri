"""Point-in-polygon test using the winding-number algorithm."""

from simetri.geometry.points.point_utils import point_on_line_segment
from simetri.geometry.points.point_utils import left
from simetri.geometry.vectors import PointType, Sequence
from simetri.core.common import PointType


def in_polygon(
    point: PointType,
    polygon_vertices: Sequence[PointType],
    exclude_border: bool = False,
) -> bool:
    """Return whether a point lies inside a polygon (winding number).

    Args:
        point: Point ``(x, y)`` to test.
        polygon_vertices: Ordered polygon vertices (clockwise or
            counter-clockwise). The last vertex need not repeat the first.
        exclude_border: If True, points on an edge return False. If False
            (default), border points are treated as inside.

    Returns:
        True if the point is inside the polygon (subject to
        ``exclude_border``), otherwise False.

    Examples:
        ::

            from simetri.geometry.in_polygon import in_polygon

            square = [(0, 0), (1, 0), (1, 1), (0, 1)]
            in_polygon((0.5, 0.5), square)  # True
            in_polygon((2, 2), square)  # False
            in_polygon((0, 0), square, exclude_border=True)  # False
    """
    _, y = point[:2]
    n_winding = 0  # Initialize the winding number

    n = len(polygon_vertices)
    for i_ in range(n):
        p1 = polygon_vertices[i_]
        p2 = polygon_vertices[(i_ + 1) % n]  # Connect last vertex to first
        _, y1 = p1
        _, y2 = p2
        if point_on_line_segment(point, [p1, p2]):
            return not exclude_border
        if y1 <= y:  # Start y <= P.y
            if y2 > y and left(p1, p2, point):  # An upward crossing
                n_winding += 1  # P left of edge
        elif y2 <= y and not left(p1, p2, point):  # A downward crossing
            n_winding -= 1  # P right of edge

    return n_winding != 0

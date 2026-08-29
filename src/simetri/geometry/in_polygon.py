from simetri.geometry.geometry import left, point_on_line_segment
from simetri.geometry.vectors import PointType, Sequence
from simetri.graphics.common import PointType


def in_polygon(
    point: PointType,
    polygon_vertices: Sequence[PointType],
    exclude_border: bool = False,
) -> bool:
    """
    Checks if a point is inside a polygon using the winding number algorithm.

    Args:
        point (tuple): A tuple (x, y) representing the point to test.
        polygon_vertices (list): A list of tuples, where each tuple (x, y)
                                represents a vertex of the polygon. The vertices
                                should be ordered (e.g., clockwise or counter-clockwise).

    Returns:
        bool: True if the point is inside the polygon, False otherwise.
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

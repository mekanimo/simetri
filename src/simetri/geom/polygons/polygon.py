"""Polygon topology and boolean/partition helpers.

Objects here (``Node``, ``Edge``, ``Polygon``, ``Partition``, …) are not meant
to be transformed directly. Each exposes a ``shape`` (or ``group``) property
that returns a drawable ``simetri.graphics`` object.

Examples:
    ::

        import simetri.graphics as sg

        area = sg.polygon_area([(0, 0), (1, 0), (1, 1), (0, 1)])
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from math import atan2, ceil, isclose, log10, pi, sqrt
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from ...base.common import (
    LineType,
    PointType,
    PolygonLike,
    d_id_obj,
    get_defaults,
    get_unique_id,
)
from ...config.settings import defaults, issue_warning
from ..points.point_utils import (
    close_points_square,
    distance,
    left3,
    midpoint,
    on_segment,
    round_point,
)
from ..segments.line_utils import (
    all_intersections,
    angle_between_lines3,
    equal_edges,
    offset_line,
    round_segment,
    stitch,
)
from ..vectors import v_from_points

if TYPE_CHECKING:
    from simetri.group.batch import Group
    from simetri.shapes.shape import Shape

from .polygon_utils import (
    right_handed,
)
from ..points.point_utils import (
    point_on_line_segment,
)


def _shape(*args: Any, **kwargs: Any) -> Shape:
    """Build a ``Shape`` without importing it at module import time.

    Args:
        *args: Positional arguments for ``Shape``.
        **kwargs: Keyword arguments for ``Shape``.

    Returns:
        Shape: The constructed shape.

    Examples:
        >>> from simetri.geom.polygons.polygon import _shape
        >>> _shape([(0, 0), (1, 0)]).vertices
        [(0, 0), (1, 0)]
    """
    from simetri.shapes.shape import Shape

    return Shape(*args, **kwargs)


def _group(*args: Any, **kwargs: Any) -> Group:
    """Build a ``Group`` without importing it at module import time.

    Args:
        *args: Positional arguments for ``Group``.
        **kwargs: Keyword arguments for ``Group``.

    Returns:
        Group: The constructed group.

    Examples:
        >>> from simetri.geom.polygons.polygon import _group
        >>> len(_group([]))
        0
    """
    from simetri.group.batch import Group

    return Group(*args, **kwargs)


@dataclass
class Node:
    """A polygon vertex with a unique id.

    Attributes:
        pos: Point coordinates ``(x, y)``.
    """

    pos: PointType
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.id: int = get_unique_id(self)

    @property
    def shape(self) -> Shape:
        """Return a single-point ``Shape`` at this node.

        Returns:
            Shape: Drawable point shape.
        """
        return _shape([self.pos])

    @property
    def closed(self) -> bool:
        """Whether this geometry is closed (always False for a node).

        Returns:
            bool: Always False.
        """
        return self._closed


@dataclass
class Edge:
    """A polygon edge between two ``Node`` endpoints.

    Attributes:
        nodes: Pair of endpoint nodes.
    """

    nodes: tuple[Node, Node]
    _closed: bool = field(default=False, init=False, repr=False)
    _nodes: tuple[Node, Node] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.id: int = get_unique_id(self)

    @property
    def shape(self) -> Shape:
        """Return a two-point ``Shape`` for this edge.

        Returns:
            Shape: Drawable segment.
        """
        return _shape([n.pos for n in self.nodes])

    @property
    def closed(self) -> bool:
        """Whether this geometry is closed (always False for an edge).

        Returns:
            bool: Always False.
        """
        return self._closed

    @property
    def nodes(self) -> tuple[Node, Node]:
        """Endpoint nodes of the edge.

        Returns:
            tuple[Node, Node]: Start and end nodes.
        """
        return self._nodes

    @nodes.setter
    def nodes(self, value: tuple[Node, Node]) -> None:
        """Set endpoint nodes (invalidates cached length).

        Args:
            value: New ``(start, end)`` node pair.
        """
        if (
            "_nodes" in self.__dict__
            and self._nodes != value
            and "_length" in self.__dict__
        ):
            del self._length
        self._nodes = value

    @property
    def length(self) -> float:
        """Euclidean length of the edge (cached).

        Returns:
            float: Edge length.
        """
        # Used cached value if it exists
        if "_length" not in self.__dict__:
            a, b = self.nodes
            res = distance(a.pos, b.pos)
            self._length = res
        else:
            res = self._length

        return res


@dataclass
class Polyline:
    """Connected line segments that can be open or closed (a ring).

    Attributes:
        nodes: Ordered vertices.
        edges: Ordered edges between consecutive nodes.
        closed: If True, the polyline is a closed ring.
    """

    nodes: Sequence[Node]
    edges: Sequence[Edge]
    closed: bool = False  # If closed then it becomes a ring

    def __post_init__(self) -> None:
        self.id: int = get_unique_id(self)

    @property
    def shape(self) -> Shape:
        """Return a drawable ``Shape`` for this polyline.

        Returns:
            Shape: Polyline shape (closed if ``self.closed``).
        """
        return _shape([n.pos for n in self.nodes], closed=self.closed)

    @property
    def vertices(self) -> Sequence[PointType]:
        """Vertex positions as a sequence of points.

        Returns:
            Sequence[PointType]: Node positions.
        """
        return tuple(n.pos for n in self.nodes)

    @property
    def length(self) -> float:
        """Total length of all edges.

        Returns:
            float: Sum of edge lengths.
        """
        return sum([e.length for e in self.edges])


def polygon_area(
    polygon: Sequence[PointType], dist_tol: float | None = None
) -> float:
    """Return the signed area of a polygon.

    A counter-clockwise walk is positive. A clockwise walk is negative.
    An unclosed ring is closed for the calculation only.

    Args:
        polygon (Sequence[PointType]): Vertices in walk order.
        dist_tol (float | None): Distance used to decide whether the ring
            is already closed. Defaults to ``defaults["dist_tol"]``.

    Returns:
        float: Signed area.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.polygon_area([(0, 0), (1, 0), (1, 1), (0, 1)])
        1.0
        >>> sg.polygon_area([(0, 0), (0, 1), (1, 1), (1, 0)])
        -1.0
        >>> sg.polygon_area([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        1.0
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    dist_tol2 = dist_tol * dist_tol
    if not close_points_square(polygon[0], polygon[-1], dist2=dist_tol2):
        polygon = list(polygon[:])
        polygon.append(polygon[0])
    area_ = 0
    for i, point in enumerate(polygon[:-1]):
        x1, y1 = point[:2]
        x2, y2 = polygon[i + 1][:2]
        area_ += x1 * y2 - x2 * y1

    return area_ / 2


def ccw_positive_vertices(
    vertices: Sequence[PointType],
) -> list[tuple[float, float]]:
    """Return a counter-clockwise copy of ``vertices``.

    A clockwise walk is reversed so the signed area is positive.

    Args:
        vertices (Sequence[PointType]): Polygon vertices in order.

    Returns:
        list[tuple[float, float]]: Counter-clockwise vertex copy.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.ccw_positive_vertices([(0, 0), (1, 0), (1, 1)])
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        >>> sg.ccw_positive_vertices([(0, 0), (0, 1), (1, 0)])
        [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
    """
    verts = [(float(x), float(y)) for x, y in vertices]
    if polygon_area(verts) < 0:
        verts = list(reversed(verts))

    return verts


@dataclass
class Polygon:
    """Closed polygon with optional holes.

    Attributes:
        nodes: Boundary vertices.
        edges: Boundary edges.
        holes: Interior hole polylines.
    """

    nodes: Sequence[Node]
    edges: Sequence[Edge]
    holes: Sequence[Polyline]
    _closed: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        self.id: int = get_unique_id(self)

    @property
    def closed(self) -> bool:
        """Whether this polygon is closed (always True).

        Returns:
            bool: Always True.
        """
        return self._closed

    @property
    def shape(self) -> Shape:
        """Return a closed drawable ``Shape`` for the outer boundary.

        Returns:
            Shape: Closed polygon shape.
        """
        return _shape([n.pos for n in self.nodes], closed=True)

    @property
    def vertices(self) -> Sequence[PointType]:
        """Outer-boundary vertex positions.

        Returns:
            Sequence[PointType]: Node positions.
        """
        return tuple(n.pos for n in self.nodes)

    @property
    def area(self) -> float:
        """Signed area of the outer boundary (cached).

        Returns:
            float: Polygon area.
        """
        if "_area" not in self.__dict__:
            self._area = polygon_area(self.vertices)
        return self._area

    @property
    def perimeter(self) -> float:
        """Sum of outer-boundary edge lengths.

        Returns:
            float: Perimeter length.
        """
        return sum([e.length for e in self.edges])


@dataclass
class Side:
    """A partition side (like an edge, but for partitions).

    Attributes:
        nodes: Endpoint nodes.
    """

    nodes: tuple[Node, Node]
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.id: int = get_unique_id(self)

    @property
    def shape(self) -> Shape:
        """Return a two-point ``Shape`` for this side.

        Returns:
            Shape: Drawable segment.
        """
        return _shape([n.pos for n in self.nodes])

    @property
    def closed(self) -> bool:
        """Whether this geometry is closed (always False for a side).

        Returns:
            bool: Always False.
        """
        return self._closed


@dataclass
class Partition:
    """A closed polygonal region defined by nodes and sides.

    Attributes:
        nodes: Ordered vertices of the partition.
        sides: Boundary sides.
    """

    nodes: Sequence[Node]
    sides: Sequence[Side]
    _closed: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        self.id: int = get_unique_id(self)

    @property
    def shape(self) -> Shape:
        """Return a drawable ``Shape`` for this partition.

        Returns:
            Shape: Partition outline.
        """
        return _shape([n.pos for n in self.nodes])

    @property
    def closed(self) -> bool:
        """Whether this partition is closed (always True).

        Returns:
            bool: Always True.
        """
        return self._closed

    @property
    def vertices(self) -> Sequence[PointType]:
        """Vertex positions of the partition.

        Returns:
            Sequence[PointType]: Node positions.
        """
        return tuple(n.pos for n in self.nodes)

    @property
    def area(self) -> float:
        """Signed area of the partition (cached).

        Returns:
            float: Partition area.
        """
        if "_area" not in self.__dict__:
            self._area = polygon_area(self.vertices)
        return self._area

    @property
    def perimeter(self) -> float:
        """Sum of side lengths.

        Returns:
            float: Perimeter length.
        """
        return sum([e.length for e in self.sides])


@dataclass
class Polyset:
    """Collection of polygons/polylines with topological relations.

    Provides relationship dictionaries and (stub) boolean operations.
    Not for cosmetic styling — use graphics collections for that.

    Attributes:
        polys: Member polygons and/or polylines.
    """

    polys: Sequence[Polygon | Polyline]

    def __post_init__(self) -> None:
        self.id: int = get_unique_id(self)

    @property
    def group(self) -> Group:
        """Return a ``Group`` of shapes for each member.

        Returns:
            Group: Drawable group of member shapes.
        """
        return _group([poly.shape for poly in self.polys])

    @property
    def union(self) -> Any:
        """Boolean union of member polygons (stub).

        Returns:
            Any: Not yet implemented.
        """
        pass

    @property
    def intersection(self) -> Any:
        """Boolean intersection of member polygons (stub).

        Returns:
            Any: Not yet implemented.
        """
        pass

    @property
    def symmetric_difference(self) -> Any:
        """Boolean symmetric difference of member polygons (stub).

        Returns:
            Any: Not yet implemented.
        """
        pass

    @property
    def partitions(self) -> Any:
        """Partitions derived from member polygons (stub).

        Returns:
            Any: Not yet implemented.
        """
        pass

    @property
    def d_node_poly(self) -> Any:
        """Node-to-polygon relation dictionary (stub)."""
        pass

    @property
    def d_node_edge(self) -> Any:
        """Node-to-edge relation dictionary (stub)."""
        pass

    @property
    def d_node_side(self) -> Any:
        """Node-to-side relation dictionary (stub)."""
        pass

    @property
    def d_node_part(self) -> Any:
        """Node-to-partition relation dictionary (stub)."""
        pass

    @property
    def d_edge_poly(self) -> Any:
        """Edge-to-polygon relation dictionary (stub)."""
        pass

    @property
    def d_edge_part(self) -> Any:
        """Edge-to-partition relation dictionary (stub)."""
        pass

    @property
    def d_edge_side(self) -> Any:
        """Edge-to-side relation dictionary (stub)."""
        pass

    @property
    def d_edge_node(self) -> Any:
        """Edge-to-node relation dictionary (stub)."""
        pass

    @property
    def d_part_poly(self) -> Any:
        """Partition-to-polygon relation dictionary (stub)."""
        pass

    @property
    def d_part_edge(self) -> Any:
        """Partition-to-edge relation dictionary (stub)."""
        pass

    @property
    def d_side_edge(self) -> Any:
        """Side-to-edge relation dictionary (stub)."""
        pass

    @property
    def d_side_part(self) -> Any:
        """Side-to-partition relation dictionary (stub)."""
        pass

    @property
    def d_side_poly(self) -> Any:
        """Side-to-polygon relation dictionary (stub)."""
        pass


def _segment_containment_counts(
    midpoints: Sequence[PointType],
    shape_vertices: Sequence[Sequence[PointType]],
) -> NDArray[np.int16]:
    """Count how many polygons contain each segment midpoint.

    Args:
        midpoints (Sequence[PointType]): Points to test.
        shape_vertices (Sequence[Sequence[PointType]]): Polygon vertex rings.

    Returns:
        NDArray[np.int16]: One count per midpoint.

    Examples:
        >>> from simetri.geom.polygons.polygon import _segment_containment_counts
        >>> square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        >>> list(_segment_containment_counts([(0.5, 0.5), (2, 2)], [square]))
        [1, 0]
    """
    counts = np.zeros(len(midpoints), dtype=np.int16)
    for vertices in shape_vertices:
        for index, midpoint in enumerate(midpoints):
            if in_polygon(midpoint, vertices):
                counts[index] += 1
    return counts


def point_inside_polygon(
    p: PointType, poly: Sequence[PointType], eps: float = 1e-5
) -> bool:
    """Return True only if ``p`` is strictly inside ``poly``.

    Points on the boundary return False.

    Args:
        p (PointType): Query point ``(x, y)``.
        poly (Sequence[PointType]): Ordered polygon vertices.
        eps (float): Tolerance for on-segment tests. Defaults to ``1e-5``.

    Returns:
        bool: True if strictly inside; False on the boundary or outside.

    Examples:
        >>> import simetri.graphics as sg
        >>> square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        >>> sg.point_inside_polygon((0.5, 0.5), square)
        True
        >>> sg.point_inside_polygon((0.5, 0), square)
        False
        >>> sg.point_inside_polygon((2, 2), square)
        False
    """
    x, y = p
    n = len(poly)

    # boundary check
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        if on_segment(a, b, p, eps=eps):
            return False

    inside = False
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ax, ay = a
        bx, by = b

        # edge straddles horizontal ray?
        if (ay > y) != (by > y):
            # x intersection
            x_int = (bx - ax) * (y - ay) / (by - ay) + ax
            if x_int > x + eps:
                inside = not inside
    return inside


def polygons_union(
    shapes: Sequence[Shape],
    all_segments: Sequence[LineType],
    all_midpoints: Sequence[PointType],
    min_seg_len: float = 0.001,
) -> tuple[Shape, Group]:
    """Return the union boundary of polygon shapes from a segment arrangement.

    A single shape is returned unchanged, with an empty hole group.
    Shared interior edges are dropped. The midpoint of a kept segment
    lies in exactly one input polygon.

    Args:
        shapes (Sequence[Shape]): Input polygon shapes.
        all_segments (Sequence[LineType]): Arrangement segments.
        all_midpoints (Sequence[PointType]): Midpoints parallel to
            ``all_segments``.
        min_seg_len (float): Drop segments shorter than this. Defaults
            to ``0.001``.

    Returns:
        tuple[Shape, Group]: Outer union boundary and a group of holes.

    Raises:
        ValueError: If ``shapes`` is empty.

    Examples:
        >>> import simetri.graphics as sg
        >>> square = sg.Shape([(0, 0), (1, 0), (1, 1), (0, 1)], closed=True)
        >>> outer, holes = sg.polygons_union([square], [], [])
        >>> len(outer.vertices)
        4
        >>> len(holes)
        0
        >>> sg.polygons_union([], [], [])
        Traceback (most recent call last):
            ...
        ValueError: polygons_union requires at least one polygon
    """
    if len(shapes) == 0:
        raise ValueError("polygons_union requires at least one polygon")
    if len(shapes) == 1:
        return shapes[0], _group([])

    shape_vertices = [shape.vertices for shape in shapes]
    counts = _segment_containment_counts(all_midpoints, shape_vertices)

    # Keep arrangement segments whose midpoint lies in exactly one input polygon.
    # Shared interior edges (count >= 2) and exterior void edges (count == 0)
    # are dropped; count == 1 is the union-boundary (XOR) rule.
    union_segments = []
    for segment, midpoint, count in zip(all_segments, all_midpoints, counts):
        if distance(*segment) < min_seg_len:
            continue
        if count == 1:
            union_segments.append(segment)

    merged = _group(
        [_shape(segment) for segment in union_segments]
    ).merge_shapes()
    outer = merged[0]
    holes = [
        part
        for part in merged[1:]
        if point_inside_polygon(part[-1], outer.vertices)
    ]
    return outer, _group(holes)


def all_close_points(
    points: Sequence[Sequence[float]],
    dist_tol: float | None = None,
    with_dist: bool = False,
) -> tuple[
    dict[int, Sequence[int]],
    Sequence[tuple[int, int] | tuple[int, int, float]],
]:
    """Return close point ids, and the pairs that produced them.

    Each input row is ``[x, y, id]``. Nearby points are linked in both
    directions.

    Args:
        points (Sequence[Sequence[float]]): Rows ``[x, y, id]``.
        dist_tol (float | None): Distance tolerance. Defaults to
            ``defaults["dist_tol"]``.
        with_dist (bool): If True, each pair is ``(id1, id2, distance)``.
            Defaults to False.

    Returns:
        tuple: ``({id: [nearby ids], ...}, pairs)``. Ids with no neighbor
        are omitted from the dictionary.

    Examples:
        >>> import simetri.graphics as sg
        >>> rows = [[0, 0, 1], [0.01, 0, 2], [5, 5, 3]]
        >>> sg.all_close_points(rows, dist_tol=0.05)
        ({1: [2], 2: [1]}, [(1, 2)])
        >>> links, pairs = sg.all_close_points(rows, dist_tol=0.05, with_dist=True)
        >>> links
        {1: [2], 2: [1]}
        >>> round(pairs[0][2], 2)
        0.01
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    point_arr = np.array(
        points, dtype=np.float32
    )  # points array [[x1, y1, id1], ...]]
    n_rows = len(points)
    point_arr = point_arr[point_arr[:, 0].argsort()]  # sort by x values in the
    # first column
    xmin = point_arr[:, 0] - dist_tol * 2
    xmin = xmin.reshape(n_rows, 1)
    xmax = point_arr[:, 0] + dist_tol * 2
    xmax = xmax.reshape(n_rows, 1)
    point_arr = np.concatenate(
        (point_arr, xmin, xmax), 1
    )  # [x, y, id, xmin, xmax]

    i_id, i_xmin, i_xmax = 2, 3, 4  # column indices
    d_connections = {}
    for i in range(n_rows):
        d_connections[int(point_arr[i, 2])] = []
    pairs = []
    dist_tol2 = dist_tol * dist_tol
    for i in range(n_rows):
        x, y, id1, sl_xmin, sl_xmax = point_arr[i, :]
        id1 = int(id1)
        point = (x, y)
        start = i + 1
        candidates = point_arr[start:, :][
            (
                (point_arr[start:, i_xmax] >= sl_xmin)
                & (point_arr[start:, i_xmin] <= sl_xmax)
            )
        ]
        for cand in candidates:
            id2 = int(cand[i_id])
            point2 = cand[:2]
            if close_points_square(point, point2, dist2=dist_tol2):
                d_connections[id1].append(id2)
                d_connections[id2].append(id1)
                if with_dist:
                    pairs.append((id1, id2, distance(point, point2)))
                else:
                    pairs.append((id1, id2))
    res = {}
    for k, v in d_connections.items():
        if v:
            res[k] = v
    return res, pairs


def node_dictionaries(
    coords: Sequence[PointType],
    dist_tol: float,
    debug: bool = False,
) -> tuple[
    dict[int, tuple[float, ...]],
    dict[tuple[float, ...], int],
    dict[tuple[float, ...], PointType],
]:
    """Return node maps for coordinates that fall within ``dist_tol``.

    Nearby coordinates share one node id. The third map keeps the original
    coordinate for each rounded key.

    Args:
        coords (Sequence[PointType]): Vertices to index.
        dist_tol (float): Distance used to merge nearby coordinates.
        debug (bool): If True, print the closest unmerged pair.
            Defaults to False.

    Returns:
        tuple: ``(node_to_coord, coord_to_node, rounded_to_original)``.

    Examples:
        >>> import simetri.graphics as sg
        >>> nodes, coord_node, rounded = sg.node_dictionaries(
        ...     [(0, 0), (0.01, 0), (5, 5)], 0.05
        ... )
        >>> nodes
        {0: (0, 0), 1: (5, 5)}
        >>> coord_node[(0, 0)] == coord_node[(0.01, 0)]
        True
        >>> rounded[(5, 5)]
        (5, 5)
    """
    n_round = max(0, ceil(log10(sqrt(2) / dist_tol)))
    d_rounded_coord = {}
    rounded = []
    for coord in coords:
        val = tuple(round_point(coord, n_round))
        rounded.append(val)
        d_rounded_coord[val] = coord

    rounded_coords = list(set(rounded))
    rounded_coords.sort()
    rounded_coords.sort(key=lambda point: point[1])

    indexed_coordinates = [
        (*coordinate[:2], index)
        for index, coordinate in enumerate(rounded_coords)
    ]
    _, close_pairs = all_close_points(indexed_coordinates, dist_tol=dist_tol)
    parent = list(range(len(rounded_coords)))
    for first_index, second_index in close_pairs:
        first_root = first_index
        while parent[first_root] != first_root:
            first_root = parent[first_root]
        second_root = second_index
        while parent[second_root] != second_root:
            second_root = parent[second_root]
        if first_root != second_root:
            parent[second_root] = first_root

    d_node_coord = {}
    d_coord_node = {}

    root_node = {}
    for coordinate_index, coordinate in enumerate(rounded_coords):
        root = coordinate_index
        while parent[root] != root:
            root = parent[root]
        if root not in root_node:
            node = len(root_node)
            root_node[root] = node
            d_node_coord[node] = rounded_coords[root]
        d_coord_node[coordinate] = root_node[root]

    if debug:
        closest_distance = None
        closest_points = None
        for first_point, second_point in combinations(coords, 2):
            first_coordinate = tuple(round_point(first_point, n_round))
            second_coordinate = tuple(round_point(second_point, n_round))
            if (
                d_coord_node[first_coordinate]
                == d_coord_node[second_coordinate]
            ):
                continue
            point_distance = distance(first_point, second_point)
            if closest_distance is None or point_distance < closest_distance:
                closest_distance = point_distance
                closest_points = (first_point[:2], second_point[:2])
        print(
            "Node diagnostics: "
            f"dist_tol={dist_tol}; automatic n_round={n_round}; "
            f"nodes={len(d_node_coord)}"
        )
        print(
            "  Closest unmerged point distance: "
            f"{closest_distance}; points={closest_points}"
        )

    return (d_node_coord, d_coord_node, d_rounded_coord)


def segment_cycles(
    segments, length_bound: int = 10, cycle_basis=False, dist_tol=None
):
    """Return closed walks formed by line segments.

    Args:
        segments (Sequence[LineType]): Segments ``[(p1, p2), ...]``.
        length_bound (int): Maximum cycle length. Defaults to 10.
        cycle_basis (bool): If True, use a cycle basis. Defaults to False.
        dist_tol (float | None): Distance for merging nearby endpoints.
            Defaults to ``defaults["dist_tol"]``.

    Returns:
        tuple: ``(coordinate_cycles, node_id_cycles)``.

    Examples:
        >>> import simetri.graphics as sg
        >>> square = [
        ...     ((0, 0), (1, 0)),
        ...     ((1, 0), (1, 1)),
        ...     ((1, 1), (0, 1)),
        ...     ((0, 1), (0, 0)),
        ... ]
        >>> coords, nodes = sg.segment_cycles(square)
        >>> coords
        [[(0, 0), (1, 0), (1, 1), (0, 1)]]
        >>> len(nodes[0])
        4
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    coordinates = []
    for seg in segments:
        coordinates.extend(seg)

    d_node_coord, d_coord_node, _ = node_dictionaries(coordinates, dist_tol)
    n_round = max(0, ceil(log10(sqrt(2) / dist_tol)))
    g_segments = [
        [d_coord_node[tuple(round_point(coord, n_round))] for coord in seg]
        for seg in segments
    ]

    nx_graph = nx.Graph()
    nx_graph.update(g_segments)
    if cycle_basis:
        cycles = [
            cycle
            for cycle in nx.cycle_basis(nx_graph)
            if len(cycle) <= length_bound
        ]
    else:
        cycles = list(nx.simple_cycles(nx_graph, length_bound=length_bound))
    res = []
    for cycle in cycles:
        res.append([d_node_coord[node] for node in cycle])

    return res, cycles


def segments_from_points(
    points: Sequence[PointType],
) -> tuple[PointType, PointType] | None:
    """Return consecutive segments after sorting collinear points.

    Args:
        points (Sequence[PointType]): Collinear points in any order.

    Returns:
        list[tuple[PointType, PointType]] | None: Sorted segments, or
        ``None`` if fewer than two points are given.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.segments_from_points([(2, 0), (0, 0), (1, 0)])
        [((0, 0), (1, 0)), ((1, 0), (2, 0))]
        >>> sg.segments_from_points([(0, 0), (1, 0)])
        [((0, 0), (1, 0))]
        >>> sg.segments_from_points([(0, 0)]) is None
        True
    """
    n = len(points)
    if n < 2:
        res = None
    elif n == 2:
        res = [tuple(points)]
    else:
        sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
        segments = list(zip(sorted_points, sorted_points[1:]))
        res = segments

    return res


def set_fills(
    partitions: Sequence[Shape], d_edge_part: dict[frozenset, set[int]]
) -> None:
    """Set each partition's fill from the symmetric-difference rule.

    An outer partition, one that owns an edge alone, is filled. Across a
    shared edge the neighboring fill is the opposite value.

    Args:
        partitions (Sequence[Shape]): Closed partition shapes (mutated).
        d_edge_part (dict[frozenset, set[int]]): Edge to partition-id map.

    Returns:
        None: Fills are written onto the partition objects.

    Examples:
        >>> import simetri.graphics as sg
        >>> outer = sg.Shape([(0, 0), (2, 0), (2, 2), (0, 2)], closed=True)
        >>> edges = {frozenset(edge): {outer.id} for edge in outer.edges}
        >>> sg.set_fills([outer], edges)
        >>> outer.fill
        True
    """

    # To start, find an edge with a single partition.
    # and the partition to the queue.
    # This is one of the outermost partitions.
    # Since outer partitions are always filled,
    # set this partition's fill property True.
    for edge, part in d_edge_part.items():
        if len(part) == 1:
            cur_part = d_id_obj[next(iter(part))]
            queue = {cur_part.id}
            cur_part.fill = True
            break
    # If an edge is between two partitions,
    # only one partition can be filled.
    # Alternate through all partitions.
    processed = set()
    count = 0
    qcount = 0
    while queue:
        count += 1
        cur_part = d_id_obj[queue.pop()]
        count += 1
        if cur_part.id not in processed:
            qcount += 1
            edges = [frozenset(e) for e in cur_part.edges]
            for edge in edges:
                partitions = d_edge_part[edge]
                if len(partitions) == 2:
                    part1, part2 = [d_id_obj[p] for p in partitions]
                    fill = not cur_part.fill
                    if part1 == cur_part:
                        part2.fill = fill
                    else:
                        part1.fill = fill

                queue.update(set(partitions))
            processed.add(cur_part.id)
            queue.difference_update(processed)


def any_point_inside_polygon(
    points: Sequence[PointType],
    polygon: Shape | NDArray,
    eps: float = 1e-12,
) -> bool:
    """Return True if any point is strictly inside the polygon.

    Boundary points are treated as outside.

    Args:
        points (Sequence[PointType]): Query points ``(x, y)``.
        polygon (Shape | NDArray): Closed vertex ring, or a closed ``Shape``.
        eps (float): Tolerance for on-edge tests. Defaults to ``1e-12``.

    Returns:
        bool: True if at least one point is inside and not on the boundary.

    Examples:
        >>> import simetri.graphics as sg
        >>> square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        >>> sg.any_point_inside_polygon([(2, 2), (0.2, 0.2)], square)
        True
        >>> sg.any_point_inside_polygon([(2, 2), (3, 3)], square)
        False
        >>> sg.any_point_inside_polygon([(0.5, 0)], square)
        False
    """

    pts = np.asarray(points, dtype=float)
    poly = np.asarray(polygon, dtype=float)

    px = pts[:, 0][:, None]  # (M,1)
    py = pts[:, 1][:, None]

    x1 = poly[:, 0]
    y1 = poly[:, 1]
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)

    # -----------------------------
    # Boundary detection (on edge)
    # -----------------------------
    cross = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)

    on_seg = (
        (np.abs(cross) < eps)
        & (np.minimum(x1, x2) <= px)
        & (px <= np.maximum(x1, x2))
        & (np.minimum(y1, y2) <= py)
        & (py <= np.maximum(y1, y2))
    )

    # Any point on boundary is NOT considered inside
    on_boundary = np.any(on_seg, axis=1)

    # -----------------------------
    # Ray casting (interior test)
    # -----------------------------
    cond = (y1 > py) != (y2 > py)
    x_intersect = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-15) + x1

    inside = np.sum(cond & (px < x_intersect), axis=1) % 2 == 1

    # Exclude boundary points from interior
    inside_strict = inside & (~on_boundary)

    return np.any(inside_strict)


def get_partitions(
    shapes: Group, length_bound: int = 10
) -> tuple[Sequence[Shape], defaultdict[frozenset, set[int]], Shape]:
    """Partition overlapping shapes into face regions from their arrangement.

    Prints cycle and hole counts while it runs.

    Args:
        shapes (Group): Group whose segments define the arrangement.
        length_bound (int): Maximum cycle length when enumerating faces.
            Defaults to 10.

    Returns:
        tuple: ``(partition_shapes, membership_map, merged_outline)``.

    Examples:
        >>> import simetri.graphics as sg
        >>> square = sg.Shape([(0, 0), (1, 0), (1, 1), (0, 1)], closed=True)
        >>> group = sg.Group([square])
        >>> len(group.all_segments)
        4
    """
    n_edges = len(shapes.all_segments)
    intersections = all_intersections(
        shapes.all_segments, return_points_list=True
    )
    points_by_edge = defaultdict(list)
    for edge_id, edge in enumerate(shapes.all_segments):
        start_point, end_point = edge
        points_by_edge[edge_id].extend(
            (round_point(start_point), round_point(end_point))
        )
    for point, (first_edge_id, second_edge_id) in intersections:
        rounded_point = round_point(point)
        points_by_edge[first_edge_id].append(rounded_point)
        points_by_edge[second_edge_id].append(rounded_point)

    all_segments = []
    all_midpoints = []
    for i in range(n_edges):
        points = points_by_edge[i]
        if len(points) >= 2:
            segments = segments_from_points(points)
            rounded_segments = [round_segment(seg, 2) for seg in segments]
            all_segments.extend(rounded_segments)
            all_midpoints.extend(
                midpoint(*segment) for segment in rounded_segments
            )

    midpoint_array = np.unique(np.asarray(all_midpoints, dtype=float), axis=0)
    midpoint_order = midpoint_array[:, 0].argsort()
    midpoint_array = midpoint_array[midpoint_order]

    cycles, cycles_nodes = segment_cycles(
        all_segments, length_bound=length_bound
    )

    print(
        f"Number of total cycles with less than {length_bound} nodes: {len(cycles)}"
    )
    sorted_cycles = sorted(
        zip(cycles, cycles_nodes), key=lambda cycle_pair: len(cycle_pair[0])
    )
    cycles = [cycle_pair[0] for cycle_pair in sorted_cycles]
    cycles_nodes = [cycle_pair[1] for cycle_pair in sorted_cycles]
    union, holes = polygons_union(shapes, all_segments, all_midpoints)
    print(f"Number of holes: {len(holes)}")
    holes_area = sum([polygon_area(hole.vertices) for hole in holes])
    union_area = polygon_area(union.vertices) - holes_area
    hole_index, sorted_hole_arrays, hole_processed = _build_hole_index(holes)
    dist_tol = defaults["dist_tol"]

    count = 0
    area = 0
    partitions = []
    d_edge_partition = defaultdict(set)
    done = False
    for poly in cycles:
        if done:
            break
        polygon_array = np.asarray(poly, dtype=float)
        polygon_min_x = polygon_array[:, 0].min()
        polygon_max_x = polygon_array[:, 0].max()
        polygon_min_y = polygon_array[:, 1].min()
        polygon_max_y = polygon_array[:, 1].max()
        midpoint_start = np.searchsorted(
            midpoint_array[:, 0], polygon_min_x, side="left"
        )
        midpoint_end = np.searchsorted(
            midpoint_array[:, 0], polygon_max_x, side="right"
        )
        candidate_midpoints = midpoint_array[midpoint_start:midpoint_end]
        candidate_midpoints = candidate_midpoints[
            (candidate_midpoints[:, 1] >= polygon_min_y)
            & (candidate_midpoints[:, 1] <= polygon_max_y)
        ]
        if not any_point_inside_polygon(candidate_midpoints, polygon_array):
            count += 1
            sorted_partition = sorted_polygon_xy_array(poly)
            candidate_hole_ids = _candidate_hole_ids(
                hole_index,
                hole_processed,
                polygon_min_x,
                polygon_min_y,
                polygon_max_x,
                polygon_max_y,
                dist_tol,
            )
            is_hole = False
            for hole_id in candidate_hole_ids:
                if equal_sorted_arrays(
                    sorted_partition,
                    sorted_hole_arrays[hole_id],
                    dist_tol,
                ):
                    is_hole = True
                    hole_processed[hole_id] = True
                    break
            if is_hole:
                continue
            partition = _shape(poly, closed=True)
            partitions.append(partition)
            for edge in partition.edges:
                d_edge_partition[frozenset(edge)].add(partition.id)
            poly_area = abs(
                polygon_area(poly)
            )  # Here, we are not strict about polygons' orientation
            area += poly_area

            if isclose(area, union_area, rel_tol=0.001):
                print(
                    f"Total partition-area: {area:.2f}, Union-area: {union_area:.2f}"
                )
                done = True

    print(f"{len(partitions)} partitions.")
    n = max((len(p) for p in partitions), default=0)
    print(f"Largest* partition has {n} edges.")
    print(f"Used {count} cycles.")

    return partitions, d_edge_partition, union


def equal_sorted_arrays(
    array1: NDArray[np.float64],
    array2: NDArray[np.float64],
    dist_tol: float,
) -> bool:
    """Return True if two same-shaped point arrays match within ``dist_tol``.

    Args:
        array1 (NDArray[np.float64]): First ``(n, 2)`` point array.
        array2 (NDArray[np.float64]): Second ``(n, 2)`` point array.
        dist_tol (float): Maximum allowed per-point distance.

    Returns:
        bool: True if every corresponding pair is within tolerance.

    Examples:
        >>> import simetri.graphics as sg
        >>> square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        >>> same = [(1, 1), (0, 1), (0, 0), (1, 0)]
        >>> sg.equal_sorted_arrays(
        ...     sg.sorted_polygon_xy_array(square),
        ...     sg.sorted_polygon_xy_array(same),
        ...     0.05,
        ... )
        True
        >>> sg.equal_sorted_arrays(
        ...     sg.sorted_polygon_xy_array(square),
        ...     sg.sorted_polygon_xy_array([(0, 0), (2, 0), (0, 2)]),
        ...     0.05,
        ... )
        False
    """
    if array1.shape != array2.shape:
        return False
    n_vertices = array1.shape[0]
    if n_vertices == 0:
        return True
    dist_tol2 = dist_tol * dist_tol
    if n_vertices <= 12:
        for index in range(n_vertices):
            dx = array1[index, 0] - array2[index, 0]
            dy = array1[index, 1] - array2[index, 1]
            if dx * dx + dy * dy > dist_tol2:
                return False
        return True
    delta = array1 - array2

    return bool(np.all((delta * delta).sum(axis=1) <= dist_tol2))


def _build_hole_index(
    holes: Sequence[Shape],
) -> tuple[NDArray[Any], Sequence[NDArray[np.float64]], NDArray[np.bool_]]:
    """Build a bbox and sorted-vertex index for hole lookup.

    Args:
        holes (Sequence[Shape]): Hole shapes to index.

    Returns:
        tuple: ``(hole_index, sorted_hole_arrays, unused_mask)``.

    Examples:
        >>> from simetri.geom.polygons.polygon import _build_hole_index
        >>> index, arrays, unused = _build_hole_index([])
        >>> len(index), arrays, list(unused)
        (0, [], [])
    """
    n_holes = len(holes)
    hole_dtype = [
        ("xmin", np.float64),
        ("ymin", np.float64),
        ("xmax", np.float64),
        ("ymax", np.float64),
        ("x", object),
        ("y", object),
        ("hole_id", np.int64),
    ]
    if n_holes == 0:
        return np.empty(0, dtype=hole_dtype), [], np.zeros(0, dtype=bool)

    sorted_hole_arrays = [sorted_polygon_xy_array(hole) for hole in holes]
    hole_index = np.empty(n_holes, dtype=hole_dtype)
    for hole_id, sorted_xy in enumerate(sorted_hole_arrays):
        hole_index[hole_id] = (
            sorted_xy[:, 0].min(),
            sorted_xy[:, 1].min(),
            sorted_xy[:, 0].max(),
            sorted_xy[:, 1].max(),
            sorted_xy[:, 0].copy(),
            sorted_xy[:, 1].copy(),
            hole_id,
        )

    return hole_index, sorted_hole_arrays, np.zeros(n_holes, dtype=bool)


def _candidate_hole_ids(
    hole_index: NDArray[Any],
    hole_processed: NDArray[np.bool_],
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    dist_tol: float,
) -> NDArray[np.int_]:
    """Return hole ids whose bbox matches the given bounds within dist_tol.

    Args:
        hole_index (NDArray[Any]): Index from :func:`_build_hole_index`.
        hole_processed (NDArray[np.bool_]): Holes already claimed.
        xmin (float): Candidate minimum x.
        ymin (float): Candidate minimum y.
        xmax (float): Candidate maximum x.
        ymax (float): Candidate maximum y.
        dist_tol (float): Absolute tolerance for bbox matching.

    Returns:
        NDArray[np.int_]: Matching hole ids.

    Examples:
        >>> from simetri.geom.polygons.polygon import (
        ...     _build_hole_index,
        ...     _candidate_hole_ids,
        ... )
        >>> index, _, unused = _build_hole_index([])
        >>> list(_candidate_hole_ids(index, unused, 0, 0, 1, 1, 0.05))
        []
    """
    if len(hole_index) == 0:
        return np.array([], dtype=int)
    mask = (
        ~hole_processed
        & np.isclose(hole_index["xmin"], xmin, atol=dist_tol)
        & np.isclose(hole_index["ymin"], ymin, atol=dist_tol)
        & np.isclose(hole_index["xmax"], xmax, atol=dist_tol)
        & np.isclose(hole_index["ymax"], ymax, atol=dist_tol)
    )

    return hole_index["hole_id"][mask].astype(int)


def polygon_xy_array(
    polygon: Shape | Sequence[PointType] | NDArray[Any],
) -> NDArray[np.float64]:
    """Return an ``(n, 2)`` float array of polygon xy coordinates.

    Args:
        polygon (Shape | Sequence[PointType] | NDArray[Any]): A ``Shape``,
            point sequence, or array-like.

    Returns:
        NDArray[np.float64]: XY coordinates. A ``Shape`` uses
        ``final_coords``.

    Examples:
        >>> import simetri.graphics as sg
        >>> [tuple(row) for row in sg.polygon_xy_array([(0, 0), (1, 0), (1, 1)])]
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
        >>> sg.polygon_xy_array([(3, 4)]).shape
        (1, 2)
    """
    from simetri.shapes.shape import Shape

    if isinstance(polygon, Shape):
        # final_coords is the cached primary_points @ xform_matrix result
        return polygon.final_coords[:, :2]
    array = np.asarray(polygon, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)

    return array[:, :2]


def sorted_polygon_xy_array(
    polygon: Shape | Sequence[PointType] | NDArray[Any],
) -> NDArray[np.float64]:
    """Return polygon xy coordinates sorted by ``x`` then ``y``.

    Args:
        polygon (Shape | Sequence[PointType] | NDArray[Any]): A ``Shape``,
            point sequence, or array-like.

    Returns:
        NDArray[np.float64]: Lexicographically sorted XY coordinates.

    Examples:
        >>> import simetri.graphics as sg
        >>> [tuple(row) for row in sg.sorted_polygon_xy_array([(1, 1), (0, 0), (0, 1)])]
        [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    """
    array = polygon_xy_array(polygon)
    order = np.lexsort((array[:, 1], array[:, 0]))

    return array[order]


def polygon_vertices(polygon: PolygonLike) -> Sequence[PointType]:
    """Return vertices from a ``Shape``, or the point sequence itself.

    Args:
        polygon (PolygonLike): A ``Shape`` or a sequence of points. Groups
            are not accepted.

    Returns:
        Sequence[PointType]: Vertex coordinates.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.polygon_vertices([(0, 0), (1, 0), (1, 1), (0, 1)])
        [(0, 0), (1, 0), (1, 1), (0, 1)]
        >>> shape = sg.Shape([(0, 0), (2, 0), (0, 2)], closed=True)
        >>> len(sg.polygon_vertices(shape))
        3
    """
    from ...shapes.shape import Shape

    if isinstance(polygon, Shape):
        res = polygon.vertices
    else:
        res = polygon

    return res


def polygon_turns(vertices: Sequence[PointType]) -> list[float]:
    """Return the signed turn sequence of a polygon.

    For each vertex, records the outgoing edge length and the signed turn
    angle at the next vertex. Angles are rounded to
    ``defaults["turn_angle_digits"]``.

    Args:
        vertices (Sequence[PointType]): Polygon vertices in walk order.
            The first vertex is not repeated.

    Returns:
        list[float]: Alternating side lengths and turn angles.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.polygon_turns([(0, 0), (1, 0), (1, 1), (0, 1)])
        [1.0, -1.57, 1.0, -1.57, 1.0, -1.57, 1.0, -1.57]
        >>> len(sg.polygon_turns([(0, 0), (2, 0), (0, 1)]))
        6
    """
    n = len(vertices)
    res = []
    TURN_ANGLE_DIGITS = defaults["turn_angle_digits"]
    for i in range(n):
        vert = vertices[i]
        next_vert = vertices[(i + 1) % n]
        next_seg = (next_vert, vertices[(i + 2) % n])
        seg = (vert, next_vert)
        angle = angle_between_lines3(vert, *next_seg)
        res.append(distance(*seg))
        res.append(round(angle, TURN_ANGLE_DIGITS))

    return res


def rotate_turns_to_min_edge(turns: Sequence[float]) -> list[float]:
    """Rotate a flat ``[length, angle, ...]`` cycle to start at a shortest edge.

    Args:
        turns (Sequence[float]): Alternating edge lengths and turn angles.

    Returns:
        list[float]: Cycle starting at the shortest edge.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.rotate_turns_to_min_edge([3.0, 1.57, 1.0, 1.57, 2.0, 1.57])
        [1.0, 1.57, 2.0, 1.57, 3.0, 1.57]
        >>> sg.rotate_turns_to_min_edge([1.0])
        [1.0]
    """
    if len(turns) < 2:
        return list(turns)
    n_edges = len(turns) // 2
    min_i = min(range(n_edges), key=lambda i: turns[2 * i])
    start = 2 * min_i
    return list(turns[start:]) + list(turns[:start])


def congruent_polygons(
    polygon1: PolygonLike,
    polygon2: PolygonLike,
    mirror: bool = False,
) -> bool:
    """Return True if ``polygon1`` and ``polygon2`` are congruent.

    Congruence ignores translation and rotation. Signed turn sequences are
    compared cyclically so convex and reflex corners stay distinct.

    Args:
        polygon1 (PolygonLike): First polygon as a ``Shape`` or points.
        polygon2 (PolygonLike): Second polygon as a ``Shape`` or points.
        mirror (bool): If True, a reflection also counts as congruent.

    Returns:
        bool: True if the polygons are congruent.

    Examples:
        >>> import simetri.graphics as sg
        >>> square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        >>> sg.congruent_polygons(square, [(1, 1), (2, 1), (2, 2), (1, 2)])
        True
        >>> sg.congruent_polygons(square, [(0, 0), (2, 0), (0, 2)])
        False
    """
    from ...helpers.utilities import equal_cycles

    verts1 = polygon_vertices(polygon1)
    verts2 = polygon_vertices(polygon2)
    if len(verts1) != len(verts2):
        res = False
    else:
        poly1_turns = polygon_turns(verts1)
        poly2_turns = polygon_turns(verts2)

        if mirror:
            if equal_cycles(poly1_turns, poly2_turns):
                res = True
            else:
                poly2_turns.reverse()
                res = equal_cycles(poly1_turns, poly2_turns)
        else:
            res = equal_cycles(poly1_turns, poly2_turns)

    return res


def equal_polygons(
    polygon1: PolygonLike,
    polygon2: PolygonLike,
    mirror: bool = False,
) -> bool:
    """Return True if two polygons are congruent.

    This is an alias of :func:`congruent_polygons`.

    Args:
        polygon1 (PolygonLike): First polygon.
        polygon2 (PolygonLike): Second polygon.
        mirror (bool): If True, a reflection also counts. Defaults to False.

    Returns:
        bool: True when the polygons match under congruence.

    Examples:
        >>> import simetri.graphics as sg
        >>> square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        >>> sg.equal_polygons(square, [(1, 0), (1, 1), (0, 1), (0, 0)])
        True
        >>> sg.equal_polygons(square, [(0, 0), (2, 0), (0, 2)])
        False
    """
    return congruent_polygons(polygon1, polygon2, mirror)


def congruent_shapes(
    shape1: Shape,
    shape2: Shape,
    mirror: bool = False,
) -> bool:
    """Return True if ``shape1`` and ``shape2`` are congruent.

    Args:
        shape1 (Shape): First shape.
        shape2 (Shape): Second shape.
        mirror (bool): If True, a reflection also counts as congruent.

    Returns:
        bool: True if the shapes are congruent.

    Examples:
        >>> import simetri.graphics as sg
        >>> left = sg.Shape([(0, 0), (1, 0), (1, 1), (0, 1)], closed=True)
        >>> right = sg.Shape([(10, 0), (11, 0), (11, 1), (10, 1)], closed=True)
        >>> sg.congruent_shapes(left, right)
        True
        >>> other = sg.Shape([(0, 0), (2, 0), (0, 2)], closed=True)
        >>> sg.congruent_shapes(left, other)
        False
    """
    return congruent_polygons(shape1, shape2, mirror=mirror)


def remove_duplicate_edges(
    edges: Sequence[LineType],
    keep_one: bool = False,
) -> list[LineType]:
    """Return edges with congruent duplicates handled.

    Candidates are filtered by axis-aligned bounding-box overlap before
    ``equal_edges``.

    Args:
        edges (Sequence[LineType]): Segments ``(point1, point2)``.
        keep_one (bool): If True, keep the first copy of each congruent
            edge. If False, drop every edge that has a congruent duplicate.

    Returns:
        list[LineType]: Selected edges.

    Examples:
        >>> import simetri.graphics as sg
        >>> edges = [((0, 0), (1, 0)), ((1, 0), (0, 0)), ((0, 1), (1, 1))]
        >>> sg.remove_duplicate_edges(edges)
        [((0, 1), (1, 1))]
        >>> sg.remove_duplicate_edges(edges, keep_one=True)
        [((0, 0), (1, 0)), ((0, 1), (1, 1))]
    """
    dist_tol = defaults["dist_tol"]

    if not keep_one:
        n = len(edges)
        if n == 0:
            res = []
        else:
            bboxes = np.empty((n, 4), dtype=float)
            for i, edge in enumerate(edges):
                x1, y1 = edge[0][:2]
                x2, y2 = edge[1][:2]
                bboxes[i] = [
                    min(x1, x2),
                    min(y1, y2),
                    max(x1, x2),
                    max(y1, y2),
                ]

            duplicate_mask = np.zeros(n, dtype=bool)
            for i in range(n):
                min_x, min_y, max_x, max_y = bboxes[i]
                overlap_mask = (
                    (bboxes[:, 2] >= min_x)
                    & (bboxes[:, 0] <= max_x)
                    & (bboxes[:, 3] >= min_y)
                    & (bboxes[:, 1] <= max_y)
                )
                for j in np.nonzero(overlap_mask)[0]:
                    if j <= i:
                        continue
                    if equal_edges(
                        edges[i],
                        edges[j],
                        dist_tol=dist_tol,
                    ):
                        duplicate_mask[i] = True
                        duplicate_mask[j] = True

            res = [
                edge for i, edge in enumerate(edges) if not duplicate_mask[i]
            ]
    else:
        unique_edges = []
        bbox_array = np.empty((0, 4), dtype=float)

        for edge in edges:
            x1, y1 = edge[0][:2]
            x2, y2 = edge[1][:2]
            min_x = min(x1, x2)
            min_y = min(y1, y2)
            max_x = max(x1, x2)
            max_y = max(y1, y2)

            duplicate = False
            if bbox_array.size:
                overlap_mask = (
                    (bbox_array[:, 2] >= min_x)
                    & (bbox_array[:, 0] <= max_x)
                    & (bbox_array[:, 3] >= min_y)
                    & (bbox_array[:, 1] <= max_y)
                )
                candidate_indices = np.nonzero(overlap_mask)[0]
                for candidate_index in candidate_indices:
                    if equal_edges(
                        edge,
                        unique_edges[candidate_index],
                        dist_tol=dist_tol,
                    ):
                        duplicate = True
                        break

            if not duplicate:
                unique_edges.append(edge)
                bbox_row = np.array([[min_x, min_y, max_x, max_y]], dtype=float)
                if bbox_array.size == 0:
                    bbox_array = bbox_row
                else:
                    bbox_array = np.vstack((bbox_array, bbox_row))

        res = unique_edges

    return res


def polygon_verts_and_bbox(
    poly: PolygonLike,
) -> tuple[Sequence[PointType], float, float, float, float]:
    """Return comparison vertices and an axis-aligned bounding box.

    For a ``Group``, shared edges are removed and the largest closed
    outline is used. If no closed outline is found, ``all_vertices`` is
    used instead.

    Args:
        poly (PolygonLike): A ``Shape``, a ``Group``, or vertex sequence.

    Returns:
        tuple: ``(vertices, min_x, min_y, max_x, max_y)``.

    Examples:
        >>> import simetri.graphics as sg
        >>> verts, min_x, min_y, max_x, max_y = sg.polygon_verts_and_bbox(
        ...     [(0, 0), (1, 0), (1, 1), (0, 1)]
        ... )
        >>> verts
        [(0, 0), (1, 0), (1, 1), (0, 1)]
        >>> (min_x, min_y, max_x, max_y)
        (0, 0, 1, 1)
    """
    from ...group.batch import Group
    from ...shapes.shape import Shape

    if isinstance(poly, Group):
        sw = poly.b_box.southwest
        ne = poly.b_box.northeast
        min_x, min_y = sw[0], sw[1]
        max_x, max_y = ne[0], ne[1]
        boundary_edges = remove_duplicate_edges(poly.all_edges, keep_one=False)
        merged = Group([Shape(edge) for edge in boundary_edges]).merge_shapes()
        verts = []
        largest_area = -1
        for shape in merged:
            if (
                isinstance(shape, Shape)
                and shape.closed
                and shape.area > largest_area
            ):
                largest_area = shape.area
                verts = list(shape.vertices)
        if not verts:
            verts = poly.all_vertices
    elif isinstance(poly, Shape):
        verts = list(poly.vertices)
        sw = poly.b_box.southwest
        ne = poly.b_box.northeast
        min_x, min_y = sw[0], sw[1]
        max_x, max_y = ne[0], ne[1]
    else:
        verts = list(poly)
        xs = [p[0] for p in verts]
        ys = [p[1] for p in verts]
        min_x, min_y = min(xs), min(ys)
        max_x, max_y = max(xs), max(ys)

    return verts, min_x, min_y, max_x, max_y


def remove_duplicate_polygons(
    polygons: Sequence[PolygonLike],
    mirror: bool = True,
    keep_one: bool = True,
) -> list[PolygonLike]:
    """Return polygons with congruent duplicates removed.

    Polygons are grouped by vertex count. Only polygons whose shortest
    edge lengths match are compared with :func:`congruent_polygons`.

    Args:
        polygons (Sequence[PolygonLike]): ``Shape`` objects or vertex
            sequences. Groups are not accepted.
        mirror (bool): If True, treat reflections as duplicates.
        keep_one (bool): If True, keep the first of each congruent class.
            If False, drop every polygon that has a congruent partner.

    Returns:
        list[PolygonLike]: Selected polygon objects from ``polygons``.

    Examples:
        >>> import simetri.graphics as sg
        >>> square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        >>> same = [(1, 0), (1, 1), (0, 1), (0, 0)]
        >>> tri = [(0, 0), (2, 0), (0, 2)]
        >>> sg.remove_duplicate_polygons([square, same, tri])
        [[(0, 0), (1, 0), (1, 1), (0, 1)], [(0, 0), (2, 0), (0, 2)]]
        >>> sg.remove_duplicate_polygons([square, same], keep_one=False)
        []
    """
    dist_tol = defaults["dist_tol"]
    entries = []
    by_n: dict[int, list[int]] = defaultdict(list)

    for i, poly in enumerate(polygons):
        verts = polygon_vertices(poly)
        turns = rotate_turns_to_min_edge(polygon_turns(verts))
        start_len = turns[0] if turns else 0.0
        entries.append(
            {
                "poly": poly,
                "turns": turns,
                "start_len": start_len,
            }
        )
        by_n[len(verts)].append(i)

    def _same_start_len(i: int, j: int) -> bool:
        return isclose(
            entries[i]["start_len"],
            entries[j]["start_len"],
            rel_tol=0.0,
            abs_tol=dist_tol,
        )

    def _are_congruent(i: int, j: int) -> bool:
        if not _same_start_len(i, j):
            return False
        return congruent_polygons(
            entries[i]["poly"],
            entries[j]["poly"],
            mirror=mirror,
        )

    n = len(polygons)
    if n == 0:
        return []

    if keep_one:
        keep = [True] * n
        for indices in by_n.values():
            kept_in_group: list[int] = []
            for i in indices:
                if any(_are_congruent(i, j) for j in kept_in_group):
                    keep[i] = False
                else:
                    kept_in_group.append(i)
        res = [polygons[i] for i in range(n) if keep[i]]
    else:
        has_duplicate = [False] * n
        for indices in by_n.values():
            for a, i in enumerate(indices):
                for j in indices[a + 1 :]:
                    if _are_congruent(i, j):
                        has_duplicate[i] = True
                        has_duplicate[j] = True
        res = [polygons[i] for i in range(n) if not has_duplicate[i]]

    return res


def symmetric_difference(
    shapes: Group, length_bound: int = 10
) -> tuple[Sequence[Shape], Shape]:
    """Return the filled faces of the symmetric difference of overlapping shapes.

    This calls :func:`get_partitions` and :func:`set_fills`.

    Args:
        shapes (Group): Overlapping polygon shapes.
        length_bound (int): Maximum cycle length for face enumeration.
            Defaults to 10.

    Returns:
        tuple[Sequence[Shape], Shape]: Filled partitions and their union
        outline.

    Examples:
        >>> import simetri.graphics as sg
        >>> square = sg.Shape([(0, 0), (1, 0), (1, 1), (0, 1)], closed=True)
        >>> group = sg.Group([square])
        >>> group.closed
        True
    """
    partitions, d_edge_part, union = get_partitions(shapes, length_bound)
    set_fills(partitions, d_edge_part)
    print(
        f"* Largest partition that can be computed with length_bound = {length_bound}."
    )
    return partitions, union


def in_polygon(
    point: PointType,
    polygon_vertices: Sequence[PointType],
    exclude_border: bool = False,
) -> bool:
    """Return whether a point lies inside a polygon.

    Uses the winding number. A point on an edge is inside unless
    ``exclude_border`` is True.

    Args:
        point (PointType): Point ``(x, y)`` to test.
        polygon_vertices (Sequence[PointType]): Ordered polygon vertices.
        exclude_border (bool): If True, points on an edge return False.
            Defaults to False.

    Returns:
        bool: True if inside, else False.

    Examples:
        >>> import simetri.graphics as sg
        >>> square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        >>> sg.in_polygon((0.5, 0.5), square)
        True
        >>> sg.in_polygon((0.5, 0), square)
        True
        >>> sg.in_polygon((0.5, 0), square, exclude_border=True)
        False
        >>> sg.in_polygon((2, 2), square)
        False
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
            if y2 > y and left3(p1, p2, point):  # An upward crossing
                n_winding += 1  # P left of edge
        elif y2 <= y and not left3(p1, p2, point):  # A downward crossing
            n_winding -= 1  # P right of edge

    return n_winding != 0


def double_offset_lines(
    line: LineType, offset: float = 1
) -> tuple[LineType, LineType]:
    """Return the two lines offset on either side of a segment.

    Args:
        line (LineType): Input segment.
        offset (float): Offset distance. Defaults to 1.

    Returns:
        tuple[LineType, LineType]: The positive-offset line and the
        negative-offset line.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.double_offset_lines(((0, 0), (2, 0)), 1)
        ([[0.0, 1.0], [2.0, 1.0]], [[0.0, -1.0], [2.0, -1.0]])
    """
    line1 = offset_line(line, offset)
    line2 = offset_line(line, -offset)

    return line1, line2


def double_offset_polylines(
    lines: Sequence[PointType],
    offset: float = 1,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> Sequence[Sequence[PointType]]:
    """Return both offset polylines of an open chain of points.

    Args:
        lines (Sequence[PointType]): Polyline vertices.
        offset (float): Offset distance. Defaults to 1.
        rel_tol (float | None): Relative stitch tolerance. Defaults to
            ``defaults["rel_tol"]``.
        abs_tol (float | None): Absolute stitch tolerance. Defaults to
            ``defaults["abs_tol"]``.

    Returns:
        list: ``[positive_offset, negative_offset]``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.double_offset_polylines([(0, 0), (2, 0), (2, 2)], 1)
        [[[0.0, 1.0], (1.0, 1.0), [1.0, 2.0]], [[0.0, -1.0], (3.0, -1.0), [3.0, 2.0]]]
    """
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    lines1 = []
    lines2 = []
    for i, point in enumerate(lines[:-1]):
        line = [point, lines[i + 1]]
        line1, line2 = double_offset_lines(line, offset)
        lines1.append(line1)
        lines2.append(line2)
    lines1 = stitch(lines1, closed=False)
    lines2 = stitch(lines2, closed=False)
    return [lines1, lines2]


def polygon_cg(points: Sequence[PointType]) -> PointType | None:
    """Return the area center of a polygon.

    This is the center of gravity of the filled polygon, not the average
    of the vertices. A zero-area ring returns ``None``.

    Args:
        points (Sequence[PointType]): Vertices in walk order.

    Returns:
        PointType | None: Area center, or ``None`` if the area is zero.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.polygon_cg([(0, 0), (1, 0), (1, 1), (0, 1)])
        [0.5, 0.5]
        >>> sg.polygon_cg([(0, 0), (2, 0), (0, 2)])
        [0.6666666666666666, 0.6666666666666666]
        >>> sg.polygon_cg([(0, 0), (1, 0), (2, 0)]) is None
        True
    """
    cx = cy = 0
    n_points = len(points)
    for i in range(n_points):
        x = points[i][0]
        y = points[i][1]
        xnext = points[(i + 1) % n_points][0]
        ynext = points[(i + 1) % n_points][1]

        temp = x * ynext - xnext * y
        cx += (x + xnext) * temp
        cy += (y + ynext) * temp
    area_ = polygon_area(points)
    denom = area_ * 6
    if denom:
        res = [cx / denom, cy / denom]
    else:
        res = None
    return res


def offset_polygon(
    polygon: Sequence[PointType],
    offset: float = -1,
    dist_tol: float | None = None,
) -> Sequence[PointType]:
    """Return a closed polygon offset from ``polygon``.

    The result is a stitched closed ring. A positive ``offset`` is inward
    after the walk is made right-handed; the default ``-1`` therefore
    offsets outward.

    Args:
        polygon (Sequence[PointType]): Polygon vertices.
        offset (float): Offset distance. Defaults to -1.
        dist_tol (float | None): Distance used to decide whether the ring
            is already closed. Defaults to ``defaults["dist_tol"]``.

    Returns:
        Sequence[PointType]: Closed offset ring.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.offset_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], 0.5)
        [(-0.5, -0.5), (1.5, -0.5), (1.5, 1.5), (-0.5, 1.5), (-0.5, -0.5)]
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    polygon = list(polygon[:])
    dist_tol2 = dist_tol * dist_tol
    if not right_handed(polygon):
        polygon.reverse()
    if not close_points_square(polygon[0], polygon[-1], dist2=dist_tol2):
        polygon.append(polygon[0])
    poly = []
    for i, point in enumerate(polygon[:-1]):
        line = [point, polygon[i + 1]]
        offset_edge = offset_line(line, -offset)
        poly.append(offset_edge)

    poly = stitch(poly, closed=True)
    return poly


def double_offset_polygons(
    polygon: Sequence[PointType],
    offset: float = 1,
    dist_tol: float | None = None,
    **kwargs: Any,
) -> Sequence[Sequence[PointType]]:
    """Return both offset polygons of a vertex ring.

    Args:
        polygon (Sequence[PointType]): Polygon vertices (mutated). An
            unclosed list is closed, and a clockwise list is reversed.
        offset (float): Offset distance. Defaults to 1.
        dist_tol (float | None): Distance used to decide whether the ring
            is already closed. Defaults to ``defaults["dist_tol"]``.
        **kwargs (Any): If ``canvas`` is a canvas, both offsets are drawn.

    Returns:
        list: ``[positive_offset, negative_offset]``.

    Examples:
        >>> import simetri.graphics as sg
        >>> raw = [(0, 0), (2, 0), (2, 1)]
        >>> offsets = sg.double_offset_polygons(raw, 0.5)
        >>> raw
        [(0, 0), (2, 0), (2, 1), (0, 0)]
        >>> offsets[0][0]
        (2.118033988749895, 0.5)
        >>> offsets[1][1]
        (2.5, -0.5)
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    dist_tol2 = dist_tol * dist_tol

    # helper to ensure polygon is closed
    if not close_points_square(polygon[0], polygon[-1], dist2=dist_tol2):
        polygon.append(polygon[0])

    if not right_handed(polygon):
        polygon.reverse()
    poly1 = []
    poly2 = []
    for i, point in enumerate(polygon[:-1]):
        line = [point, polygon[i + 1]]
        line1, line2 = double_offset_lines(line, offset)
        poly1.append(line1)
        poly2.append(line2)
    poly1 = stitch(poly1)
    poly2 = stitch(poly2)
    if "canvas" in kwargs:
        canvas = kwargs["canvas"]
        if canvas:
            canvas.new_page()
            closed = close_points_square(poly1[0], poly1[-1])
            canvas.draw(_shape(poly1, closed=closed), fill=False)
            closed = close_points_square(poly2[0], poly2[-1])
            canvas.draw(_shape(poly2, closed=closed), fill=False)
    return [poly1, poly2]


def offset_polygon_points(
    polygon: Sequence[PointType],
    offset: float = 1,
    dist_tol: float | None = None,
) -> Sequence[PointType]:
    """Return a stitched offset polygon.

    Args:
        polygon (Sequence[PointType]): Polygon vertices.
        offset (float): Offset distance. Defaults to 1.
        dist_tol (float | None): Distance used to decide whether the ring
            is already closed. Defaults to ``defaults["dist_tol"]``.

    Returns:
        Sequence[PointType]: Offset ring. A clockwise result is reversed.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.offset_polygon_points([(0, 0), (1, 0), (1, 1), (0, 1)], 0.5)
        [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    dist_tol2 = dist_tol * dist_tol
    polygon = list(polygon)
    if not close_points_square(polygon[0], polygon[-1], dist2=dist_tol2):
        polygon.append(polygon[0])
    poly = []
    for i, point in enumerate(polygon[:-1]):
        line = [point, polygon[i + 1]]
        offset_edge = offset_line(line, offset)
        poly.append(offset_edge)

    poly = stitch(poly)
    if not right_handed(poly):
        poly.reverse()
    return poly


def polyline_length(
    polygon: Sequence[PointType],
    closed: bool = False,
    dist_tol: float | None = None,
) -> float:
    """Return the length of a polyline, or the perimeter if it is closed.

    Args:
        polygon (Sequence[PointType]): Vertices in order.
        closed (bool): If True, include the closing edge when it is missing.
            Defaults to False.
        dist_tol (float | None): Distance used to decide whether the ring
            is already closed. Defaults to ``defaults["dist_tol"]``.

    Returns:
        float: Path length.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.polyline_length([(0, 0), (3, 0), (3, 4)])
        7.0
        >>> sg.polyline_length([(0, 0), (1, 0), (1, 1), (0, 1)], closed=True)
        4.0
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    dist_tol2 = dist_tol * dist_tol
    if closed and not close_points_square(polygon[0], polygon[-1], dist2=dist_tol2):
        polygon = polygon[:]
        polygon.append(polygon[0])
    perimeter = 0
    for i, point in enumerate(polygon[:-1]):
        perimeter += distance(point, polygon[i + 1])
    return perimeter


def polygon_internal_angles(vertices: Sequence[PointType]) -> Sequence[float]:
    """Return the interior angle at each vertex.

    Vertices are expected in counter-clockwise order. A clockwise ring is
    reversed first, and a warning is issued. Each value is in radians.

    Args:
        vertices (Sequence[PointType]): Polygon vertices.

    Returns:
        Sequence[float]: Interior angles, or an empty list if fewer than
        three vertices are given.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.polygon_internal_angles([(0, 0), (1, 0), (1, 1), (0, 1)])
        [1.5707963267948966, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966]
        >>> sg.polygon_internal_angles([(0, 0), (1, 0)])
        []
    """
    n = len(vertices)
    if n < 3:
        return []

    # 1. Determine Winding Order (Signed Area)
    # Positive = CCW, Negative = CW
    area = polygon_area(vertices)
    is_ccw_ = area > 0
    if not is_ccw_:
        issue_warning("""Vertices are not in counterclockwise positive order!
                         Result is for the reversed sequence of the given vertices.""")
        vertices = list(vertices)[:]
        vertices.reverse()
    angles = []
    for i in range(n):
        # Define three consecutive points
        p_prev = vertices[(i - 1) % n]
        p_curr = vertices[i]
        p_next = vertices[(i + 1) % n]

        # Vector 1: Incoming (from previous to current)
        v1 = v_from_points(p_prev, p_curr)
        # Vector 2: Outgoing (from current to next)
        v2 = v_from_points(p_curr, p_next)

        cross_prod = v1.cross(v2)
        dot_prod = v1.dot(v2)

        turning_angle = atan2(cross_prod, dot_prod)
        # Convert Turning Angle to Internal Angle
        internal_angle = pi - turning_angle
        angles.append(internal_angle)

    return angles


# with cProfile.Profile() as pr:
#     draw_symm_diff(squares, 13)


# # # Format and print the results
# stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME).print_stats(20)

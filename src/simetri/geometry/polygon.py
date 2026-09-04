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

import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from math import atan2, ceil, isclose, log10, pi, sqrt
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from ..graphics.common import (
    LineType,
    PointType,
    PolygonLike,
    d_id_obj,
    get_defaults,
    get_unique_id,
)
from ..settings.settings import defaults, issue_warning
from .vectors import v_from_points

if TYPE_CHECKING:
    from simetri.graphics.batch import Group
    from simetri.graphics.shape import Shape

from .geom_utils import (
    close_points2,
    distance,
    equal_edges,
    midpoint,
    offset_line,
    right_handed,
    round_point,
    round_segment,
)
from .geometry import (
    all_intersections,
    angle_between_lines2,
    left,
    on_segment,
    point_on_line_segment,
    stitch,
)


def _shape(*args: Any, **kwargs: Any) -> Shape:
    from simetri.graphics.shape import Shape

    return Shape(*args, **kwargs)


def _group(*args: Any, **kwargs: Any) -> Group:
    from simetri.graphics.batch import Group

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
    """Calculate the area of a polygon.

    Args:
        polygon (Sequence[PointType]): Sequence of points representing the polygon.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        float: Area of the polygon.
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    dist_tol2 = dist_tol * dist_tol
    if not close_points2(polygon[0], polygon[-1], dist2=dist_tol2):
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
    """Return a CCW+ vertex copy without mutating the input.

    Vertices are copied. If the signed area is negative (clockwise walk),
    the copy is reversed so the walk is counter-clockwise with positive
    signed area.

    Args:
        vertices: Polygon vertices in order.

    Returns:
        list[tuple[float, float]]: CCW-ordered vertex copy.
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
    """Count how many input polygons contain each segment midpoint."""
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
        p: Query point ``(x, y)``.
        poly: Ordered polygon vertices.
        eps: Tolerance for on-segment tests. Defaults to ``1e-5``.

    Returns:
        bool: True if strictly inside; False on boundary or outside.
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
    """Compute polygon union from a pre-built segment arrangement.

    Uses an XOR boundary rule on the full segment arrangement instead of
    repeated pairwise ``polygon_union`` calls.

    Args:
        shapes: Input polygon shapes.
        all_segments: Arrangement segments covering the union problem.
        all_midpoints: Midpoints parallel to ``all_segments``.
        min_seg_len: Drop segments shorter than this. Defaults to 0.001.

    Returns:
        tuple[Shape, Group]: Outer union boundary and a group of holes.

    Raises:
        ValueError: If ``shapes`` is empty.
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


def get_time(start: int, end: int) -> str:
    """Format an elapsed ``perf_counter_ns`` interval as text.

    Args:
        start: Start time from ``time.perf_counter_ns()``.
        end: End time from ``time.perf_counter_ns()``.

    Returns:
        str: Human-readable duration, e.g. ``\"12.34 milliseconds\"``.
    """

    if ms < 900:
        elapsed = ms
        units = "milliseconds"
    else:
        elapsed = ms / 1000
        units = "seconds"

    return f"{elapsed:.2f} {units}"


def all_close_points(
    points: Sequence[Sequence[float]],
    dist_tol: float | None = None,
    with_dist: bool = False,
) -> tuple[
    dict[int, Sequence[int]],
    Sequence[tuple[int, int] | tuple[int, int, float]],
]:
    """
    Find all close points in a sequence of points along with their ids.

    Args:
        points (Sequence[Sequence]): Sequence of points with ids [[x1, y1, id1], [x2, y2, id2], ...].
        dist_tol (float, optional): Distance tolerance. Defaults to None.
        with_dist (bool, optional): Whether to include distances in the result. Defaults to False.

    Returns:
        dict: Dictionary of the form {id1: [id2, id3, ...], ...}.
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
            if close_points2(point, point2, dist2=dist_tol2):
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
    """Set dictionaries for nodes and coordinates.
    d_node_coord: Dictionary of node id to coordinates.
    d_coord_node: Dictionary of coordinates to node id.

    Args:
        coords (Sequence[PointType]): Sequence of vertices.
        dist_tol (float): Distance tolerance for grouping coordinates.
        debug (bool, optional): Print node proximity diagnostics.
            Defaults to False.
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
    """Return cycles (closed walks) formed by a set of line segments.

    Args:
        segments: Sequence of line segments ``[(p1, p2), ...]``.
        length_bound: Maximum cycle length considered. Defaults to 10.
        cycle_basis: If True, use a cycle-basis extraction. Defaults to False.
        dist_tol: Distance tolerance for merging nearby endpoints.
            Defaults to ``defaults[\"dist_tol\"]``.

    Returns:
        list: Cycles found among the segments.
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
    """Given a sequence of collinear points (in any order), returns the connected segments."""
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
    """
        Set the partitions' fill property according to their symmetric difference.
        Used for creating a symmetric-difference of multiple polygons.
        This function mutates the input partitions!
        Modifies the partitions's fill properties.

    Args:
        partitions (Sequence[Shape]): A list or array of polygons (closed Shape objects).
        d_edge_part (dict[Sequence[LineType, Shape]]): A dictionary with LineType keys
                                                       and closed Shape values.

    Returns:
        Sequence[Shape]: The modified input polygons.
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
    """
    Returns True if ANY point is strictly inside the polygon.
    Boundary points are treated as outside.

    Args:
        points (Sequence[PointType]): A list or numpy array of (x, y) coordinates.
        polygon (Shape): A closed Shape object.
        eps (float, optional): _description_. Defaults to 1e-12.

    Returns:
        bool: If any of the input points is in the polygon returns True,
              False otherwise.
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

    Args:
        shapes: Group whose segments define the arrangement.
        length_bound: Maximum cycle length when enumerating faces.
            Defaults to 10.

    Returns:
        tuple: ``(partition_shapes, membership_map, merged_outline)``.
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

    c_start = time.perf_counter_ns()
    cycles, cycles_nodes = segment_cycles(
        all_segments, length_bound=length_bound
    )
    c_end = time.perf_counter_ns()

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
    start = time.perf_counter_ns()
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
    end = time.perf_counter_ns()
    print(
        f"Computing cycles with length_bounde={length_bound} took {get_time(c_start, c_end)}."
    )
    print(f"{len(partitions)} partitions.")
    n = max((len(p) for p in partitions), default=0)
    print(f"Largest* partition has {n} edges.")
    print(f"Used {count} cycles.")
    print(f"Computing partitions took {get_time(start, end)}.")

    return partitions, d_edge_partition, union


def equal_sorted_arrays(
    array1: NDArray[np.float64],
    array2: NDArray[np.float64],
    dist_tol: float,
) -> bool:
    """Return True if two same-shaped point arrays match within ``dist_tol``.

    Args:
        array1: First ``(n, 2)`` point array.
        array2: Second ``(n, 2)`` point array (same order).
        dist_tol: Maximum allowed per-point distance.

    Returns:
        bool: True if every corresponding pair is within tolerance.
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
    """Build a bbox / sorted-vertex index for hole lookup.

    Args:
        holes: Hole shapes to index.

    Returns:
        tuple: ``(hole_index, sorted_hole_arrays, unused_mask)``.
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
    """Return hole ids whose bbox matches the given bounds within dist_tol."""
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
        polygon: A ``Shape``, point sequence, or array-like.

    Returns:
        NDArray[np.float64]: XY coordinates (uses ``final_coords`` for shapes).
    """
    from simetri.graphics.shape import Shape

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
        polygon: A ``Shape``, point sequence, or array-like.

    Returns:
        NDArray[np.float64]: Lexicographically sorted XY coordinates.
    """
    array = polygon_xy_array(polygon)
    order = np.lexsort((array[:, 1], array[:, 0]))

    return array[order]


def polygon_vertices(polygon: PolygonLike) -> Sequence[PointType]:
    """Extract vertices from a Shape or return a vertex sequence as-is.

    Args:
        polygon: A ``Shape`` or a sequence of points. Groups are not accepted;
            the caller must unwrap them first.

    Returns:
        Vertex coordinates of the polygon.
    """
    from ..graphics.shape import Shape

    if isinstance(polygon, Shape):
        res = polygon.vertices
    else:
        res = polygon

    return res


def polygon_turns(vertices: Sequence[PointType]) -> list[float]:
    """Return the signed turn sequence of a polygon.

    For each vertex ``i``, records the length of the edge from ``i`` to
    ``i + 1`` and the signed turn angle at ``i + 1`` between that edge and
    the next (via ``angle_between_lines2``). Angles are rounded to
    ``defaults["turn_angle_digits"]``. Convex and reflex corners keep opposite signs.

    Args:
        vertices: Polygon vertices in walk order (closed; first vertex is
            not repeated).

    Returns:
        A list with alternating side-length and angle values.
    """
    n = len(vertices)
    res = []
    TURN_ANGLE_DIGITS = defaults["turn_angle_digits"]
    for i in range(n):
        vert = vertices[i]
        next_vert = vertices[(i + 1) % n]
        next_seg = (next_vert, vertices[(i + 2) % n])
        seg = (vert, next_vert)
        angle = angle_between_lines2(vert, *next_seg)
        res.append(distance(*seg))
        res.append(round(angle, TURN_ANGLE_DIGITS))

    return res


def rotate_turns_to_min_edge(turns: Sequence[float]) -> list[float]:
    """Rotate a flat ``[length, angle, ...]`` cycle to start at a min edge.

    Args:
        turns: Alternating edge lengths and turn angles.

    Returns:
        list[float]: Rotated cycle starting at the shortest edge.
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
    compared cyclically so convex and reflex corners remain distinct.

    Args:
        polygon1: First polygon as a ``Shape`` or a sequence of points.
        polygon2: Second polygon as a ``Shape`` or a sequence of points.
        mirror: If True, also treat reflection (mirror image) as congruent.

    Returns:
        True if the polygons are congruent; False otherwise.
    """
    from ..helpers.utilities import equal_cycles

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
    """Return True if two polygons are congruent (alias of ``congruent_polygons``).

    Args:
        polygon1: First polygon.
        polygon2: Second polygon.
        mirror: If True, allow mirror congruence. Defaults to False.

    Returns:
        bool: True when the polygons match under congruence.
    """
    return congruent_polygons(polygon1, polygon2, mirror)


def congruent_shapes(
    shape1: Shape,
    shape2: Shape,
    mirror: bool = False,
) -> bool:
    """Return True if ``shape1`` and ``shape2`` are congruent.

    Args:
        shape1: First shape as a ``Shape`` object.
        shape2: Second shape as a ``Shape`` object.
        mirror: If True, also treat reflection (mirror image) as congruent.

    Returns:
        True if the shapes are congruent; False otherwise.
    """
    return congruent_polygons(shape1, shape2, mirror=mirror)


def remove_duplicate_edges(
    edges: Sequence[LineType],
    keep_one: bool = False,
) -> list[LineType]:
    """Return edges with congruent duplicates handled.

    Candidates are filtered with axis-aligned bounding-box overlap before
    calling ``equal_edges``.

    Args:
        edges: List of line segments ``(point1, point2)``.
        keep_one: If True, keep the first occurrence of each congruent edge.
            If False, drop every edge that has a congruent duplicate (shared
            internal edges of a polyomino are removed).

    Returns:
        A new list of edges. The input list is not modified.
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
    """Extract comparison vertices and axis-aligned bbox for a polygon.

    For a ``Group`` of squares, shared edges are removed, the boundary is
    merged, and the largest closed outline is used. If no closed outline is
    found, falls back to ``all_vertices``.

    Args:
        poly: A ``Shape``, a ``Group``, or a sequence of polygon vertices.

    Returns:
        ``(vertices, min_x, min_y, max_x, max_y)`` for congruence filtering
        and comparison. Does not modify ``poly``.
    """
    from ..graphics.batch import Group
    from ..graphics.shape import Shape

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

    Algorithm:
        1. Group polygons by vertex count.
        2. Build ``polygon_turns`` for each polygon.
        3. Rotate each turn cycle so it starts at a smallest edge length.
        4. Within each vertex-count group, only compare polygons whose
           starting edge lengths match; use ``congruent_polygons`` there.

    Args:
        polygons: Sequence of ``Shape`` objects or vertex sequences.
            Groups are not accepted; unwrap them first.
        mirror: If True, treat reflections as duplicates.
        keep_one: If True, keep the first of each congruent class.
            If False, drop every polygon that has a congruent partner.

    Returns:
        A new list of polygon references (same objects as in ``polygons``).
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
    """Partition overlapping shapes and return the XOR (symmetric difference) faces.

    Args:
        shapes: Group of overlapping polygon shapes.
        length_bound: Maximum cycle length for face enumeration. Defaults to 10.

    Returns:
        tuple[Sequence[Shape], Shape]: Filled partitions and their union outline.
    """
    start = time.perf_counter_ns()
    partitions, d_edge_part, union = get_partitions(shapes, length_bound)
    set_fills(partitions, d_edge_part)
    end = time.perf_counter_ns()
    print(f"Total elapsed time was {get_time(start, end)}.")
    print(
        f"* Largest partition that can be computed with length_bound = {length_bound}."
    )
    return partitions, union


def in_polygon(
    point: PointType,
    polygon_vertices: Sequence[PointType],
    exclude_border: bool = False,
) -> bool:
    """Return whether a point lies inside a polygon (winding number).

    Args:
        point: Point ``(x, y)`` to test.
        polygon_vertices: Ordered polygon vertices (clockwise or
            counter-clockwise).
        exclude_border: If True, points on an edge return False. If False
            (default), border points are treated as inside.

    Returns:
        bool: True if inside (subject to ``exclude_border``), else False.

    Examples:
        ::

            import simetri.graphics as sg

            square = [(0, 0), (1, 0), (1, 1), (0, 1)]
            sg.in_polygon((0.5, 0.5), square)  # True
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


def double_offset_lines(
    line: LineType, offset: float = 1
) -> tuple[LineType, LineType]:
    """
    Return two offset lines to a given line segment with the given offset amount.

    Args:
        line (LineType): Input line segment.
        offset (float, optional): Offset distance. Defaults to 1.

    Returns:
        tuple[LineType, LineType]: Two offset lines.
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
    """
    Return a sequence of double offset lines from a sequence of lines.

    Args:
        lines (Sequence[PointType]): Sequence of points representing the lines.
        offset (float, optional): Offset distance. Defaults to 1.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        Sequence[PointType]: Sequence of double offset lines.
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
    """
    Given a sequence of points that define a polygon, return the center point.

    Args:
        points (Sequence[PointType]): Sequence of points representing the polygon.

    Returns:
        PointType: Center point of the polygon.
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


def polygon_center2(polygon_points: Sequence[PointType]) -> PointType:
    """
    Given a sequence of points that define a polygon, return the center point.

    Args:
        polygon_points (Sequence[PointType]): Sequence of points representing the polygon.

    Returns:
        PointType: Center point of the polygon.
    """
    n = len(polygon_points)
    x = 0
    y = 0
    for point in polygon_points:
        x += point[0]
        y += point[1]
    x = x / n
    y = y / n
    return [x, y]


def polygon_center(polygon_points: Sequence[PointType]) -> PointType:
    """
    Given a sequence of points that define a polygon, return the center point.

    Args:
        polygon_points (Sequence[PointType]): Sequence of points representing the polygon.

    Returns:
        PointType: Center point of the polygon.
    """
    x = 0
    y = 0
    for i, point in enumerate(polygon_points[:-1]):
        x += point[0] * (polygon_points[i - 1][1] - polygon_points[i + 1][1])
        y += point[1] * (polygon_points[i - 1][0] - polygon_points[i + 1][0])
    area_ = polygon_area(polygon_points)
    return (x / (6 * area_), y / (6 * area_))


def offset_polygon(
    polygon: Sequence[PointType],
    offset: float = -1,
    dist_tol: float | None = None,
) -> Sequence[PointType]:
    """
    Return a sequence of offset lines from a sequence of lines.

    Args:
        polygon (Sequence[PointType]): Sequence of points representing the polygon.
        offset (float, optional): Offset distance. Defaults to -1.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        Sequence[PointType]: Sequence of offset lines.
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    polygon = list(polygon[:])
    dist_tol2 = dist_tol * dist_tol
    if not right_handed(polygon):
        polygon.reverse()
    if not close_points2(polygon[0], polygon[-1], dist2=dist_tol2):
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
    """
    Return a sequence of double offset lines from a sequence of lines.

    Args:
        polygon (Sequence[PointType]): Sequence of points representing the polygon.
        offset (float, optional): Offset distance. Defaults to 1.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        Sequence[PointType]: Sequence of double offset lines.
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    dist_tol2 = dist_tol * dist_tol

    # helper to ensure polygon is closed
    if not close_points2(polygon[0], polygon[-1], dist2=dist_tol2):
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
            closed = close_points2(poly1[0], poly1[-1])
            canvas.draw(_shape(poly1, closed=closed), fill=False)
            closed = close_points2(poly2[0], poly2[-1])
            canvas.draw(_shape(poly2, closed=closed), fill=False)
    return [poly1, poly2]


def offset_polygon_points(
    polygon: Sequence[PointType],
    offset: float = 1,
    dist_tol: float | None = None,
) -> Sequence[PointType]:
    """
    Return a sequence of double offset lines from a sequence of lines.

    Args:
        polygon (Sequence[PointType]): Sequence of points representing the polygon.
        offset (float, optional): Offset distance. Defaults to 1.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        Sequence[PointType]: Sequence of double offset lines.
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    dist_tol2 = dist_tol * dist_tol
    polygon = list(polygon)
    if not close_points2(polygon[0], polygon[-1], dist2=dist_tol2):
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
    """Calculate the perimeter of a polygon.

    Args:
        polygon (Sequence[PointType]): Sequence of points representing the polygon.
        closed (bool, optional): Whether the polygon is closed. Defaults to False.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        float: Perimeter of the polygon.
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    dist_tol2 = dist_tol * dist_tol
    if closed and not close_points2(polygon[0], polygon[-1], dist2=dist_tol2):
        polygon = polygon[:]
        polygon.append(polygon[0])
    perimeter = 0
    for i, point in enumerate(polygon[:-1]):
        perimeter += distance(point, polygon[i + 1])
    return perimeter


def polygon_internal_angles(vertices: Sequence[PointType]) -> Sequence[float]:
    """
    Computes internal angles for a polygon given as a sequence of (x, y) tuples.
    Works for both convex and concave polygons.

    Vertices are expected to be in counterclockwise positive order. If not
    they are reversed and the result is for the reversed order.

    Args:
        vertices (Sequence[PointType]): Sequence of points representing the polygon.

    Returns:
        Sequence[float]: Sequence of internal angles of the polygon.
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

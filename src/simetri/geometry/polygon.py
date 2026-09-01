"""Used for polygon operations. These objects are not meant to be
transformed. They all have a 'shape' property that returns an
equivalent Shape object that can be transformed."""

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
    TurnPair,
    TurnSequence,
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
    pos: PointType
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.id: int = get_unique_id(self)

    @property
    def shape(self) -> Shape:
        return _shape([self.pos])

    @property
    def closed(self) -> bool:
        """Immutable property, it is always False."""
        return self._closed


@dataclass
class Edge:
    """Polygon edges."""

    nodes: tuple[Node, Node]
    _closed: bool = field(default=False, init=False, repr=False)
    _nodes: tuple[Node, Node] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.id: int = get_unique_id(self)

    @property
    def shape(self) -> Shape:
        return _shape([n.pos for n in self.nodes])

    @property
    def closed(self) -> bool:
        """Immutable property, it is always False."""
        return self._closed

    @property
    def nodes(self) -> tuple[Node, Node]:
        return self._nodes

    @nodes.setter
    def nodes(self, value: tuple[Node, Node]) -> None:
        if (
            "_nodes" in self.__dict__
            and self._nodes != value
            and "_length" in self.__dict__
        ):
            del self._length
        self._nodes = value

    @property
    def length(self) -> float:
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
    """Connected line segments that can be closed or open."""

    nodes: Sequence[Node]
    edges: Sequence[Edge]
    closed: bool = False  # If closed then it becomes a ring

    def __post_init__(self) -> None:
        self.id: int = get_unique_id(self)

    @property
    def shape(self) -> Shape:
        return _shape([n.pos for n in self.nodes], closed=self.closed)

    @property
    def vertices(self) -> Sequence[PointType]:
        return tuple(n.pos for n in self.nodes)

    @property
    def length(self) -> float:
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
    """
    verts = [(float(x), float(y)) for x, y in vertices]
    if polygon_area(verts) < 0:
        verts = list(reversed(verts))

    return verts


@dataclass
class Polygon:
    """Polygon geometry."""

    nodes: Sequence[Node]
    edges: Sequence[Edge]
    holes: Sequence[Polyline]
    _closed: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        self.id: int = get_unique_id(self)

    @property
    def closed(self) -> bool:
        """Immutable property, it is always True."""
        return self._closed

    @property
    def shape(self) -> Shape:
        return _shape([n.pos for n in self.nodes], closed=True)

    @property
    def vertices(self) -> Sequence[PointType]:
        return tuple(n.pos for n in self.nodes)

    @property
    def area(self) -> float:
        if "_area" not in self.__dict__:
            self._area = polygon_area(self.vertices)
        return self._area

    @property
    def perimeter(self) -> float:
        return sum([e.length for e in self.edges])


@dataclass
class Side:
    """Partitions have sides instead of edges."""

    nodes: tuple[Node, Node]
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.id: int = get_unique_id(self)

    @property
    def shape(self) -> Shape:
        return _shape([n.pos for n in self.nodes])

    @property
    def closed(self) -> bool:
        """Immutable property, it is always False."""
        return self._closed


@dataclass
class Partition:
    nodes: Sequence[Node]
    sides: Sequence[Side]
    _closed: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        self.id: int = get_unique_id(self)

    @property
    def shape(self) -> Shape:
        return _shape([n.pos for n in self.nodes])

    @property
    def closed(self) -> bool:
        """Immutable property, it is always True."""
        return self._closed

    @property
    def vertices(self) -> Sequence[PointType]:
        return tuple(n.pos for n in self.nodes)

    @property
    def area(self) -> float:
        if "_area" not in self.__dict__:
            self._area = polygon_area(self.vertices)
        return self._area

    @property
    def perimeter(self) -> float:
        return sum([e.length for e in self.sides])


@dataclass
class Polyset:
    """Used for multiple polygons/polylines.

    Provides relationships between polygons, polylines, partitions, etc.
    Boolean operations.
    Not for cosmetic properties.
    Use collections for cosmetic properties.
    """

    polys: Sequence[Polygon | Polyline]

    def __post_init__(self) -> None:
        self.id: int = get_unique_id(self)

    @property
    def group(self) -> Group:
        return _group([poly.shape for poly in self.polys])

    @property
    def union(self) -> Any:
        pass

    @property
    def intersection(self) -> Any:
        pass

    @property
    def symmetric_difference(self) -> Any:
        pass

    @property
    def partitions(self) -> Any:
        pass

    @property
    def d_node_poly(self) -> Any:
        pass

    @property
    def d_node_edge(self) -> Any:
        pass

    @property
    def d_node_side(self) -> Any:
        pass

    @property
    def d_node_part(self) -> Any:
        pass

    @property
    def d_edge_poly(self) -> Any:
        pass

    @property
    def d_edge_part(self) -> Any:
        pass

    @property
    def d_edge_side(self) -> Any:
        pass

    @property
    def d_edge_node(self) -> Any:
        pass

    @property
    def d_part_poly(self) -> Any:
        pass

    @property
    def d_part_edge(self) -> Any:
        pass

    @property
    def d_side_edge(self) -> Any:
        pass

    @property
    def d_side_part(self) -> Any:
        pass

    @property
    def d_side_poly(self) -> Any:
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
    """
    Strictly inside only.
    Boundary returns False (consistent with "does not contain any vertices inside").
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
    """Compute polygon union from a pre-built arrangement.

    Uses XOR boundary rule on the full segment arrangement instead of
    repeated pairwise ``polygon_union`` calls.
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
    ms = (end - start) / 1e6

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
    """Given a sequence of line segments, returns all cycles."""
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
    (
        """_summary_

    Parameters
    ----------
    holes : Sequence[Shape]
        _description_

    Returns
    -------
    _type_
        _description_
    """
        """"""
    )

    # """Build bbox/sorted-vertex index for hole lookup."""
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
    array = polygon_xy_array(polygon)
    order = np.lexsort((array[:, 1], array[:, 0]))

    return array[order]


def cyclic_mirror_turns_equal(
    turns1: TurnSequence, turns2: TurnSequence, dist_tol: float
) -> bool:
    """Return True if ``turns2`` matches a reflection of ``turns1`` cyclically.

    For some starting-vertex offset, each side length agrees within
    ``dist_tol`` and each turn angle of ``turns2`` is the negation of the
    corresponding angle in ``turns1`` (via ``mirror_turn_sequences_equal``).
    Empty sequences of equal length are treated as equal.

    Args:
        turns1: First sequence of ``(side_length, turn_angle)`` pairs.
        turns2: Second sequence of ``(side_length, turn_angle)`` pairs.
        dist_tol: Maximum allowed difference in side lengths.

    Returns:
        True if some cyclic rotation of ``turns2`` is a mirror match of
        ``turns1``; False otherwise.
    """
    res = False
    n = len(turns1)
    if n == len(turns2):
        if n == 0:
            res = True
        else:
            for offset in range(n):
                rotated = turns2[offset:] + turns2[:offset]
                if mirror_turn_sequences_equal(turns1, rotated, dist_tol):
                    res = True
                    break

    return res


def negated_turns(turns: TurnSequence) -> list[TurnPair]:
    """Return a turn sequence with each signed angle negated.

    Used when comparing opposite winding of the same chirality: reverse
    the walk order and negate angles so left turns become right turns.

    Args:
        turns: Sequence of ``(side_length, turn_angle)`` pairs.

    Returns:
        A new list with the same side lengths and each angle replaced by
        ``-angle``, rounded to ``defaults["turn_angle_digits"]``.
    """
    TURN_ANGLE_DIGITS = defaults["turn_angle_digits"]

    return [
        (length, round(-angle, TURN_ANGLE_DIGITS)) for length, angle in turns
    ]


def equal_turns(
    turns1: TurnSequence,
    turns2: TurnSequence,
    check_mirror: bool = False,
) -> bool:
    """Return True if two signed turn sequences represent congruent walks.

    Sequences are ``(side_length, turn_angle)`` pairs. Comparison is cyclic
    (starting vertex may differ). Signed angles distinguish convex and
    reflex corners. Callers should pass turns from CCW+ vertex walks.

    Args:
        turns1: First turn sequence.
        turns2: Second turn sequence.
        check_mirror: If True, also match reflections (angles negate under
            cyclic alignment), including the reverse-order listing of a
            mirrored walk.

    Returns:
        True if the sequences are equivalent under the requested checks.
    """
    dist_tol = defaults["dist_tol"]
    res = cyclic_turns_equal(turns1, turns2, dist_tol)
    if not res and check_mirror:
        res = cyclic_mirror_turns_equal(turns1, turns2, dist_tol)
        if not res:
            reversed_turns = list(reversed(turns2))
            res = cyclic_mirror_turns_equal(turns1, reversed_turns, dist_tol)

    return res


def polygon_turns(vertices: Sequence[PointType]) -> list[TurnPair]:
    """Return the signed turn sequence of a polygon.

    For each vertex ``i``, records the length of the edge from ``i`` to
    ``i + 1`` and the signed turn angle at ``i + 1`` between that edge and
    the next (via ``angle_between_lines2``). Angles are rounded to
    ``defaults["turn_angle_digits"]``. Convex and reflex corners keep opposite signs.

    Args:
        vertices: Polygon vertices in walk order (closed; first vertex is
            not repeated).

    Returns:
        A list of ``(side_length, turn_angle)`` pairs, one per vertex.
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
        res.append((distance(*seg), round(angle, TURN_ANGLE_DIGITS)))

    return res


def equal_polygon_turns(
    polygon1: Sequence[PointType],
    polygon2: Sequence[PointType],
    check_mirror: bool = False,
) -> bool:
    """Return True if two polygons have congruent signed turn sequences.

    Each polygon is copied to CCW+ order before turn extraction. Builds
    ``polygon_turns`` for each copy and delegates to ``equal_turns``. When
    ``check_mirror`` is True and turn matching fails, also tries
    ``mirror_equivalent_polygons`` (axis reflections after normalizing the
    first vertex to the origin).

    Args:
        polygon1: First polygon as a sequence of points.
        polygon2: Second polygon as a sequence of points.
        check_mirror: If True, also match mirror images.

    Returns:
        True if the polygons are congruent under the requested checks.
    """
    verts1 = ccw_positive_vertices(polygon1)
    verts2 = ccw_positive_vertices(polygon2)
    turns1 = polygon_turns(verts1)
    turns2 = polygon_turns(verts2)
    res = equal_turns(turns1, turns2, check_mirror=check_mirror)
    if not res and check_mirror:
        res = mirror_equivalent_polygons(verts1, verts2)

    return res


def congruent_shapes(
    shape1: Shape,
    shape2: Shape,
    mirror: bool = False,
) -> bool:
    """Return True if ``shape1`` and ``shape2`` are congruent.

    Congruence ignores translation and rotation. Vertices are normalized to
    CCW+ before signed turn sequences are compared so convex and reflex
    corners remain distinct.

    Args:
        shape1: First shape as a ``Shape`` object.
        shape2: Second shape as a ``Shape`` object.
        mirror: If True, also treat reflection (mirror image) as congruent.

    Returns:
        True if the shapes are congruent; False otherwise.
    """
    verts1 = shape1.vertices
    verts2 = shape2.vertices

    if len(verts1) != len(verts2):
        res = False
    else:
        res = equal_polygon_turns(verts1, verts2, check_mirror=mirror)

    return res


def _polygon_vertices(polygon: PolygonLike) -> Sequence[PointType]:
    """Extract vertices from a polygon-like value.

    Args:
        polygon: A ``Shape``, a ``Group`` of shapes, or a sequence of points.

    Returns:
        A list of vertex coordinates. For a ``Group``, uses the largest
        closed outline (same logic as ``polygon_verts_and_bbox``).
    """
    from ..graphics.batch import Group
    from ..graphics.shape import Shape

    if isinstance(polygon, Shape):
        res = list(polygon.vertices)
    elif isinstance(polygon, Group):
        verts, _, _, _, _ = polygon_verts_and_bbox(polygon)
        res = list(verts)
    else:
        res = list(polygon)

    return res


def congruent_polygons(
    polygon1: PolygonLike,
    polygon2: PolygonLike,
    mirror: bool = False,
) -> bool:
    """Return True if ``polygon1`` and ``polygon2`` are congruent.

    Congruence ignores translation and rotation. Vertices are normalized to
    CCW+ before signed turn sequences are compared so convex and reflex
    corners remain distinct.

    Args:
        polygon1: First polygon as a ``Shape`` or a sequence of points.
        polygon2: Second polygon as a ``Shape`` or a sequence of points.
        mirror: If True, also treat reflection (mirror image) as congruent.

    Returns:
        True if the polygons are congruent; False otherwise.
    """
    verts1 = _polygon_vertices(polygon1)
    verts2 = _polygon_vertices(polygon2)
    if len(verts1) != len(verts2):
        res = False
    else:
        res = equal_polygon_turns(verts1, verts2, check_mirror=mirror)

    return res


# alias for congruent_polygons
def equal_polygons(
    polygon1: PolygonLike,
    polygon2: PolygonLike,
    mirror: bool = False,
) -> bool:
    """Return True if ``polygon1`` and ``polygon2`` are congruent.

    Congruence ignores translation and rotation. Vertices are normalized to
    CCW+ before signed turn sequences are compared so convex and reflex
    corners remain distinct.

    Args:
        polygon1: First polygon as a ``Shape`` or a sequence of points.
        polygon2: Second polygon as a ``Shape`` or a sequence of points.
        mirror: If True, also treat reflection (mirror image) as congruent.

    Returns:
        True if the polygons are congruent; False otherwise.
    """

    return congruent_polygons(polygon1, polygon2, mirror)


def normalize_at_origin(
    vertices: Sequence[PointType],
) -> list[tuple[float, float]]:
    """Translate vertices so the first point is at the origin.

    Args:
        vertices: Polygon vertices in walk order.

    Returns:
        A new list of ``(x, y)`` points with ``vertices[0]`` mapped to
        ``(0, 0)``; relative positions are unchanged.
    """
    origin_x, origin_y = vertices[0][:2]

    return [(x - origin_x, y - origin_y) for x, y in vertices]


def mirror_equivalent_polygons(
    polygon1: Sequence[PointType],
    polygon2: Sequence[PointType],
    dist_tol: float | None = None,
) -> bool:
    """Return True if ``polygon2`` matches a reflection of ``polygon1``.

    Both polygons are translated so their first vertex is at the origin and
    normalized to CCW+, then axis reflections of ``polygon1`` are compared
    with ``equal_polygon_turns`` (without re-entering mirror checks).
    """
    normalized1 = ccw_positive_vertices(normalize_at_origin(polygon1))
    normalized2 = ccw_positive_vertices(normalize_at_origin(polygon2))
    res = False
    for flip_x, flip_y in ((1, -1), (-1, 1), (-1, -1)):
        mirrored = ccw_positive_vertices(
            [(flip_x * x, flip_y * y) for x, y in normalized1]
        )
        if equal_polygon_turns(mirrored, normalized2, check_mirror=False):
            res = True
            break

    return res


def turn_angle_equal(angle1: float, angle2: float, ang_tol: float) -> bool:
    """Return True if two signed turn angles agree within ``ang_tol``.

    The absolute difference is reduced modulo ``2π`` so collinear turns
    ``π`` and ``-π`` are treated as equal.

    Args:
        angle1: First signed turn angle in radians.
        angle2: Second signed turn angle in radians.
        ang_tol: Maximum allowed angular difference.

    Returns:
        True if the angles match within tolerance; False otherwise.
    """
    tau = 2 * pi
    delta = abs(angle1 - angle2)
    delta = min(delta, abs(delta - tau), abs(delta + tau))

    return delta <= ang_tol


def _turn_pair_equal(pair1: TurnPair, pair2: TurnPair, dist_tol: float) -> bool:
    """Return True if two ``(length, angle)`` pairs match within tolerance.

    Side lengths must agree within ``dist_tol``. Signed turn angles must
    agree within ``10 ** -defaults["turn_angle_digits"]`` (modulo ``2π``).

    Args:
        pair1: First ``(side_length, turn_angle)`` pair.
        pair2: Second ``(side_length, turn_angle)`` pair.
        dist_tol: Maximum allowed difference in side lengths.

    Returns:
        True if both length and angle match; False otherwise.
    """
    TURN_ANGLE_DIGITS = defaults["turn_angle_digits"]
    length1, angle1 = pair1
    length2, angle2 = pair2
    turn_angle_tol = 10**-TURN_ANGLE_DIGITS
    res = abs(length1 - length2) <= dist_tol and turn_angle_equal(
        angle1, angle2, turn_angle_tol
    )

    return res


def turn_sequences_equal(
    turns1: TurnSequence, turns2: TurnSequence, dist_tol: float
) -> bool:
    """Return True if two turn sequences match entry-wise in order.

    Sequences must have the same length. Each corresponding pair is
    compared with ``_turn_pair_equal`` (no cyclic shift).

    Args:
        turns1: First sequence of ``(side_length, turn_angle)`` pairs.
        turns2: Second sequence of ``(side_length, turn_angle)`` pairs.
        dist_tol: Maximum allowed difference in side lengths.

    Returns:
        True if every entry matches; False if lengths differ or any
        pair fails.
    """
    res = False
    if len(turns1) == len(turns2):
        res = all(
            _turn_pair_equal(pair1, pair2, dist_tol)
            for pair1, pair2 in zip(turns1, turns2)
        )

    return res


def cyclic_turns_equal(
    turns1: TurnSequence, turns2: TurnSequence, dist_tol: float
) -> bool:
    """Return True if ``turns2`` matches ``turns1`` under cyclic shift.

    Used for rotation invariance: the starting vertex may differ.
    Empty sequences of equal length are treated as equal.

    Args:
        turns1: First sequence of ``(side_length, turn_angle)`` pairs.
        turns2: Second sequence of ``(side_length, turn_angle)`` pairs.
        dist_tol: Maximum allowed difference in side lengths.

    Returns:
        True if some cyclic rotation of ``turns2`` matches ``turns1``
        entry-wise; False otherwise.
    """
    res = False
    n = len(turns1)
    if n == len(turns2):
        if n == 0:
            res = True
        else:
            for offset in range(n):
                rotated = turns2[offset:] + turns2[:offset]
                if turn_sequences_equal(turns1, rotated, dist_tol):
                    res = True
                    break

    return res


def mirror_turn_angle_equal(
    angle1: float, angle2: float, ang_tol: float
) -> bool:
    """Return True if signed turn angles are negatives of each other.

    Used for mirror-image walks (``angle1 ≈ -angle2``). The absolute
    sum is reduced modulo ``2π`` so wrap-around cases still match.

    Args:
        angle1: First signed turn angle in radians.
        angle2: Second signed turn angle in radians.
        ang_tol: Maximum allowed deviation from exact negation.

    Returns:
        True if the angles are mirrors within tolerance; False otherwise.
    """
    tau = 2 * pi
    delta = abs(angle1 + angle2)
    delta = min(delta, abs(delta - tau), abs(delta + tau))

    return delta <= ang_tol


def mirror_turn_pair_equal(
    pair1: TurnPair, pair2: TurnPair, dist_tol: float
) -> bool:
    """Return True if lengths match and turn angles are mirrored.

    Side lengths must agree within ``dist_tol``. Signed turn angles must
    satisfy ``angle1 ≈ -angle2`` within ``10 ** -defaults["turn_angle_digits"]``.

    Args:
        pair1: First ``(side_length, turn_angle)`` pair.
        pair2: Second ``(side_length, turn_angle)`` pair.
        dist_tol: Maximum allowed difference in side lengths.

    Returns:
        True if length and mirrored angle both match; False otherwise.
    """
    TURN_ANGLE_DIGITS = defaults["turn_angle_digits"]
    length1, angle1 = pair1
    length2, angle2 = pair2
    turn_angle_tol = 10**-TURN_ANGLE_DIGITS
    res = abs(length1 - length2) <= dist_tol and mirror_turn_angle_equal(
        angle1, angle2, turn_angle_tol
    )
    return res


def mirror_turn_sequences_equal(
    turns1: TurnSequence, turns2: TurnSequence, dist_tol: float
) -> bool:
    """Return True if sequences match entry-wise with mirrored angles.

    Sequences must have the same length. Each corresponding pair is
    compared with ``mirror_turn_pair_equal`` (no cyclic shift).

    Args:
        turns1: First sequence of ``(side_length, turn_angle)`` pairs.
        turns2: Second sequence of ``(side_length, turn_angle)`` pairs.
        dist_tol: Maximum allowed difference in side lengths.

    Returns:
        True if every entry is a mirror match; False if lengths differ
        or any pair fails.
    """
    res = False
    if len(turns1) == len(turns2):
        res = all(
            mirror_turn_pair_equal(pair1, pair2, dist_tol)
            for pair1, pair2 in zip(turns1, turns2)
        )

    return res


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
    dist_tol: float = 0.001,
    check_mirrors: bool = True,
    keep_one: bool = True,
) -> list[PolygonLike]:
    """Return polygons with congruent duplicates removed.

    Congruence uses signed turn sequences (``polygon_turns`` / ``equal_turns``)
    so reflex and convex corners remain distinct. Bounding-box overlap and
    dimension filters narrow candidates before turn comparison.

    Args:
        polygons: Polygons as ``Shape``, ``Group``, or vertex sequences.
        dist_tol: Tolerance for bbox dimension matching and side lengths.
        check_mirrors: If True, mirror-image equivalents are treated as
            duplicates (includes ``mirror_equivalent_polygons`` fallback).
        keep_one: If True, keep the first occurrence of each congruent
            polygon. If False, drop every polygon that has a congruent
            duplicate.

    Returns:
        A new list of polygon references. The input sequence is not modified;
        returned items are the same objects as in ``polygons``.
    """

    def _entry(poly: PolygonLike) -> dict:
        verts, min_x, min_y, max_x, max_y = polygon_verts_and_bbox(poly)
        verts = ccw_positive_vertices(verts)
        turns = polygon_turns(verts)
        return {
            "verts": verts,
            "turns": turns,
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "width": max_x - min_x,
            "height": max_y - min_y,
            "n_vertices": len(verts),
        }

    def _congruent(entry_a: dict, entry_b: dict) -> bool:
        if entry_a["n_vertices"] != entry_b["n_vertices"]:
            return False

        overlap = (
            entry_b["max_x"] >= entry_a["min_x"]
            and entry_a["max_x"] >= entry_b["min_x"]
            and entry_b["max_y"] >= entry_a["min_y"]
            and entry_a["max_y"] >= entry_b["min_y"]
        )
        aligned_dims = (
            np.abs(entry_b["width"] - entry_a["width"]) <= dist_tol
            and np.abs(entry_b["height"] - entry_a["height"]) <= dist_tol
        )
        rotated_dims = (
            np.abs(entry_b["width"] - entry_a["height"]) <= dist_tol
            and np.abs(entry_b["height"] - entry_a["width"]) <= dist_tol
        )
        if not (overlap or aligned_dims or rotated_dims):
            return False

        duplicate_match = equal_turns(
            entry_a["turns"],
            entry_b["turns"],
            check_mirror=check_mirrors,
        )
        if not duplicate_match and check_mirrors:
            duplicate_match = mirror_equivalent_polygons(
                entry_a["verts"],
                entry_b["verts"],
            )
        return duplicate_match

    if not keep_one:
        n = len(polygons)
        if n == 0:
            res = []
        else:
            entries = [_entry(poly) for poly in polygons]
            bboxes = np.array(
                [
                    [e["min_x"], e["min_y"], e["max_x"], e["max_y"]]
                    for e in entries
                ],
                dtype=float,
            )
            duplicate_mask = np.zeros(n, dtype=bool)
            for i in range(n):
                min_x = entries[i]["min_x"]
                min_y = entries[i]["min_y"]
                max_x = entries[i]["max_x"]
                max_y = entries[i]["max_y"]
                overlap_mask = (
                    (bboxes[:, 2] >= min_x)
                    & (bboxes[:, 0] <= max_x)
                    & (bboxes[:, 3] >= min_y)
                    & (bboxes[:, 1] <= max_y)
                )
                for j in np.nonzero(overlap_mask)[0]:
                    if j <= i:
                        continue
                    if _congruent(entries[i], entries[j]):
                        duplicate_mask[i] = True
                        duplicate_mask[j] = True

            res = [
                poly for i, poly in enumerate(polygons) if not duplicate_mask[i]
            ]
    else:
        unique_polygons = []
        unique_entries = []

        for poly in polygons:
            entry = _entry(poly)
            duplicate = False
            for unique_entry in unique_entries:
                if _congruent(entry, unique_entry):
                    duplicate = True
                    break

            if not duplicate:
                unique_polygons.append(poly)
                unique_entries.append(entry)

        res = unique_polygons

    return res


def symmetric_difference(
    shapes: Group, length_bound: int = 10
) -> tuple[Sequence[Shape], Shape]:
    """_summary_

    Parameters
    ----------
    shapes : _type_
        _description_
    length_bound : int, optional
        _description_, by default 10

    Returns
    -------
    _type_
        _description_
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
    """
    Checks if a point is inside a polygon using the winding number algorithm.

    Args:
        point (tuple): A tuple (x, y) representing the point to test.
        polygon_vertices (list): A sequence of tuples, where each tuple (x, y)
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

"""Used for polygon operations. These objects are not meant to be
transformed. They all have a 'shape' property that returns an
equivalent Shape object that can be transformed."""

import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations
from math import atan2, ceil, isclose, log10, pi, sqrt
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from simetri.geometry.vectors import (
    PointType,
    Sequence,
    distance,
    v_from_points,
)
from simetri.graphics.common import LineType, PointType, d_id_obj, get_unique_id
from simetri.settings.settings import defaults

if TYPE_CHECKING:
    from simetri.graphics.batch import Group
    from simetri.graphics.shape import Shape

from .geometry import (
    all_intersections,
    close_points2,
    distance,
    left,
    midpoint,
    offset_line,
    on_segment,
    point_on_line_segment,
    right_handed,
    round_point,
    round_segment,
    stitch,
)


def _shape(*args, **kwargs) -> "Shape":
    from simetri.graphics.shape import Shape

    return Shape(*args, **kwargs)


def _group(*args, **kwargs) -> "Group":
    from simetri.graphics.batch import Group

    return Group(*args, **kwargs)


@dataclass
class Node:
    pos: PointType
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        self.id: int = get_unique_id(self)

    @property
    def shape(self) -> "Shape":
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

    def __post_init__(self):
        self.id: int = get_unique_id(self)

    @property
    def shape(self) -> "Shape":
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
    def length(self):
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

    def __post_init__(self):
        self.id: int = get_unique_id(self)

    @property
    def shape(self) -> "Shape":
        return _shape([n.pos for n in self.nodes], closed=self.closed)

    @property
    def vertices(self) -> Sequence[PointType]:
        return tuple(n.pos for n in self.nodes)

    @property
    def length(self) -> float:
        return sum([e.length for e in self.edges])


def polygon_area(polygon: Sequence[PointType], dist_tol=None) -> float:
    """Calculate the area of a polygon.

    Args:
        polygon (Sequence[PointType]): List of points representing the polygon.
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


@dataclass
class Polygon:
    """Polygon geometry."""

    nodes: Sequence[Node]
    edges: Sequence[Edge]
    holes: Sequence[Polyline]
    _closed: bool = field(default=True, init=False, repr=False)

    def __post_init__(self):
        self.id: int = get_unique_id(self)

    @property
    def closed(self) -> bool:
        """Immutable property, it is always True."""
        return self._closed

    @property
    def shape(self) -> "Shape":
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

    def __post_init__(self):
        self.id: int = get_unique_id(self)

    @property
    def shape(self) -> "Shape":
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

    def __post_init__(self):
        self.id: int = get_unique_id(self)

    @property
    def shape(self) -> "Shape":
        return _shape([n.pos for n in self.nodes])

    @property
    def closed(self) -> bool:
        """Immutable property, it is always True."""
        return self._closed

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

    def __post_init__(self):
        self.id: int = get_unique_id(self)

    @property
    def group(self):
        return _group([poly.shape for poly in self.polys])

    @property
    def union(self):
        pass

    @property
    def intersection(self):
        pass

    @property
    def symmetric_difference(self):
        pass

    @property
    def partitions(self):
        pass

    @property
    def d_node_poly(self):
        pass

    @property
    def d_node_edge(self):
        pass

    @property
    def d_node_side(self):
        pass

    @property
    def d_node_part(self):
        pass

    @property
    def d_edge_poly(self):
        pass

    @property
    def d_edge_part(self):
        pass

    @property
    def d_edge_side(self):
        pass

    @property
    def d_edge_node(self):
        pass

    @property
    def d_part_poly(self):
        pass

    @property
    def d_part_edge(self):
        pass

    @property
    def d_side_edge(self):
        pass

    @property
    def d_side_part(self):
        pass

    @property
    def d_side_poly(self):
        pass


def _segment_containment_counts(midpoints, shape_vertices):
    """Count how many input polygons contain each segment midpoint."""
    counts = np.zeros(len(midpoints), dtype=np.int16)
    for vertices in shape_vertices:
        for index, midpoint in enumerate(midpoints):
            if in_polygon(midpoint, vertices):
                counts[index] += 1
    return counts


def point_inside_polygon(p, poly, eps=1e-5):
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


def polygons_union(shapes, all_segments, all_midpoints, min_seg_len=0.001):
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


def get_time(start, end):
    ms = (end - start) / 1e6

    if ms < 900:
        elapsed = ms
        units = "milliseconds"
    else:
        elapsed = ms / 1000
        units = "seconds"

    return f"{elapsed:.2f} {units}"


def all_close_points(
    points: Sequence[Sequence],
    dist_tol: float | None = None,
    with_dist: bool = False,
) -> dict[int, list[tuple[PointType, int]]]:
    """
    Find all close points in a list of points along with their ids.

    Args:
        points (Sequence[Sequence]): List of points with ids [[x1, y1, id1], [x2, y2, id2], ...].
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
    coords: list, dist_tol: float, debug: bool = False
) -> tuple[dict, dict, dict]:
    """Set dictionaries for nodes and coordinates.
    d_node_coord: Dictionary of node id to coordinates.
    d_coord_node: Dictionary of coordinates to node id.

    Args:
        coords (list[PointType]): List of vertices.
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


def segment_cycles(segments, length_bound=10):
    """Given a list of line segments, returns all cycles."""
    coordinates = []
    for seg in segments:
        coordinates.extend(seg)

    d_node_coord, d_coord_node, _ = node_dictionaries(coordinates, dist_tol=0.5)
    g_segments = [[d_coord_node[coord] for coord in seg] for seg in segments]

    nx_graph = nx.Graph()
    nx_graph.update(g_segments)
    cycles = list(nx.simple_cycles(nx_graph, length_bound=length_bound))
    res = []
    for cycle in cycles:
        res.append([d_node_coord[node] for node in cycle])

    return res, cycles


def segments_from_points(points):
    """Given a list of collinear points (in any order), returns the connected segments."""
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


def set_fills(partitions, d_edge_part):
    """
    Set the partitions' fill property according to their
    symmetric difference.
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


def any_point_inside_polygon(points, polygon, eps=1e-12):
    """
    Returns True if ANY point is strictly inside the polygon.
    Boundary points are treated as outside.

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


def get_partitions(shapes, length_bound=10):
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
    array1: np.ndarray,
    array2: np.ndarray,
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


def _build_hole_index(holes):
    """Build bbox/sorted-vertex index for hole lookup."""
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
    hole_index,
    hole_processed,
    xmin,
    ymin,
    xmax,
    ymax,
    dist_tol,
):
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


def polygon_xy_array(polygon) -> np.ndarray:
    from simetri.graphics.shape import Shape

    if isinstance(polygon, Shape):
        # final_coords is the cached primary_points @ xform_matrix result
        return polygon.final_coords[:, :2]
    array = np.asarray(polygon, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array[:, :2]


def sorted_polygon_xy_array(polygon) -> np.ndarray:
    array = polygon_xy_array(polygon)
    order = np.lexsort((array[:, 1], array[:, 0]))
    return array[order]


def equal_sorted_arrays(
    array1: np.ndarray,
    array2: np.ndarray,
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


def equal_polygons(
    poly1,
    poly2,
    dist_tol: float | None = None,
    *,
    _sorted_poly1: np.ndarray | None = None,
    _sorted_poly2: np.ndarray | None = None,
) -> bool:
    """Return True if two polygons match within ``dist_tol``."""
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    sorted_poly1 = (
        _sorted_poly1
        if _sorted_poly1 is not None
        else sorted_polygon_xy_array(poly1)
    )
    sorted_poly2 = (
        _sorted_poly2
        if _sorted_poly2 is not None
        else sorted_polygon_xy_array(poly2)
    )
    return equal_sorted_arrays(sorted_poly1, sorted_poly2, dist_tol)


def symmetric_difference(shapes, length_bound=10):
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
    lines: list[PointType],
    offset: float = 1,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> list[PointType]:
    """
    Return a list of double offset lines from a list of lines.

    Args:
        lines (list[PointType]): List of points representing the lines.
        offset (float, optional): Offset distance. Defaults to 1.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        list[PointType]: List of double offset lines.
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


def polygon_cg(points: list[PointType]) -> PointType:
    """
    Given a list of points that define a polygon, return the center point.

    Args:
        points (list[PointType]): List of points representing the polygon.

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


def polygon_center2(polygon_points: list[PointType]) -> PointType:
    """
    Given a list of points that define a polygon, return the center point.

    Args:
        polygon_points (list[PointType]): List of points representing the polygon.

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


def polygon_center(polygon_points: list[PointType]) -> PointType:
    """
    Given a list of points that define a polygon, return the center point.

    Args:
        polygon_points (list[PointType]): List of points representing the polygon.

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
    polygon: list[PointType], offset: float = -1, dist_tol: float | None = None
) -> list[PointType]:
    """
    Return a list of offset lines from a list of lines.

    Args:
        polygon (list[PointType]): List of points representing the polygon.
        offset (float, optional): Offset distance. Defaults to -1.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        list[PointType]: List of offset lines.
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
    polygon: list[PointType],
    offset: float = 1,
    dist_tol: float | None = None,
    **kwargs,
) -> list[PointType]:
    """
    Return a list of double offset lines from a list of lines.

    Args:
        polygon (list[PointType]): List of points representing the polygon.
        offset (float, optional): Offset distance. Defaults to 1.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        list[PointType]: List of double offset lines.
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
    polygon: list[PointType], offset: float = 1, dist_tol: float | None = None
) -> list[PointType]:
    """
    Return a list of double offset lines from a list of lines.

    Args:
        polygon (list[PointType]): List of points representing the polygon.
        offset (float, optional): Offset distance. Defaults to 1.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        list[PointType]: List of double offset lines.
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
    polygon: Sequence[PointType], closed=False, dist_tol=None
) -> float:
    """Calculate the perimeter of a polygon.

    Args:
        polygon (Sequence[PointType]): List of points representing the polygon.
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


def polygon_internal_angles(vertices: list[PointType]) -> list[float]:
    """
    Computes internal angles for a polygon given as a list of (x, y) tuples.
    Works for both convex and concave polygons.

    Vertices are expected to be in counterclockwise positive order. If not
    they are reversed and the result is for the reversed order.

    Args:
        vertices (list[PointType]): List of points representing the polygon.

    Returns:
        list[float]: List of internal angles of the polygon.
    """
    n = len(vertices)
    if n < 3:
        return []

    # 1. Determine Winding Order (Signed Area)
    # Positive = CCW, Negative = CW
    area = polygon_area(vertices)
    is_ccw_ = area > 0
    if not is_ccw_:
        raise ValueError("""Vertices are not in counterclockwise positive order!
                         Result is for the reversed list of the given vertices.""")
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

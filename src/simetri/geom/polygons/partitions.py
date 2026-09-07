"""Polygon partition helpers via edge intersections and cycle extraction."""

import time
from collections import defaultdict

import networkx as nx
import numpy as np
import simetri.geom.points.point_utils
import simetri.graphics as sg


def to_numpy_array(ip_edge_list):
    """Convert intersection/edge tuples to an object NumPy array.

    Args:
        ip_edge_list: Sequence of ``((x, y), (i1, i2))`` records.

    Returns:
        NumPy object array with columns point, ``i1``, ``i2``.
    """
    rows = []
    for (x, y), (i1, i2) in ip_edge_list:
        rows.append([(x, y), i1, i2])
    return np.array(rows, dtype=object)


def filter_by_edge(arr, edge_id):
    """Return rows of ``arr`` that touch ``edge_id`` in either edge column.

    Args:
        arr: Object array from ``to_numpy_array``.
        edge_id: Edge index to match against columns 1 and 2.

    Returns:
        Filtered NumPy array.
    """
    # i1 is column 1, i2 is column 2
    i1 = arr[:, 1]
    i2 = arr[:, 2]

    mask = (i1 == edge_id) | (i2 == edge_id)

    return arr[mask]


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


def node_dictionaries(coords: list, n_round: int = 2) -> list[dict]:
    """Set dictionaries for nodes and coordinates.
    d_node_coord: Dictionary of node id to coordinates.
    d_coord_node: Dictionary of coordinates to node id.

    Args:
        nodes (list[PointType]): List of vertices.
        n_round (int, optional): Number of rounding digits. Defaults to 2.
    """

    rounded = []
    for coord in coords:
        val = tuple(simetri.geom.points.point_utils.round_point(coord, n_round))
        rounded.append(val)

    coords = list(set(rounded))  # remove duplicates
    coords.sort()  # sort by x coordinates
    coords.sort(key=lambda x: x[1])  # sort by y coordinates

    d_node_coord = {}
    d_coord_node = {}

    for i, coord in enumerate(coords):
        d_node_coord[i] = coord
        d_coord_node[coord] = i

    return (d_node_coord, d_coord_node)


def all_polygon_intersections(polygons):
    """
    Return all proper intersections between the polygons' edges.
    Polygons' edges are not checked for self intersections.
    This if for non-intersecting polygons.

    Bounding-box candidates are collected into one NumPy array. Their
    line-segment intersections are then computed as NumPy arrays.
    """
    edge_coordinates = []
    polygon_ids = []
    for poly in polygons:
        for edge in poly.edges:
            start_point, end_point = edge
            start_x, start_y = start_point[:2]
            end_x, end_y = end_point[:2]
            edge_coordinates.append([start_x, start_y, end_x, end_y])
            polygon_ids.append(poly.id)

    edge_array = np.asarray(edge_coordinates, dtype=float)
    polygon_ids = np.asarray(polygon_ids)
    edge_count = edge_array.shape[0]
    edge_min_x = np.minimum(edge_array[:, 0], edge_array[:, 2])
    edge_min_y = np.minimum(edge_array[:, 1], edge_array[:, 3])
    edge_max_x = np.maximum(edge_array[:, 0], edge_array[:, 2])
    edge_max_y = np.maximum(edge_array[:, 1], edge_array[:, 3])
    edge_ids = np.arange(edge_count)
    sort_order = edge_min_x.argsort()
    edge_array = edge_array[sort_order]
    edge_min_x = edge_min_x[sort_order]
    edge_min_y = edge_min_y[sort_order]
    edge_max_x = edge_max_x[sort_order]
    edge_max_y = edge_max_y[sort_order]
    edge_ids = edge_ids[sort_order]
    polygon_ids = polygon_ids[sort_order]

    # if a candidate's parent polygon same as the other edge's parent
    # then this candidates should be excluded.

    candidate_starts = np.arange(edge_count) + 1
    candidate_ends = np.searchsorted(edge_min_x, edge_max_x, side="right")
    candidate_counts = candidate_ends - candidate_starts
    candidate_count = candidate_counts.sum()
    if not candidate_count:
        return []

    candidate_offsets = np.cumsum(candidate_counts) - candidate_counts
    candidate_rows = np.arange(candidate_count)
    first_indices = np.repeat(np.arange(edge_count), candidate_counts)
    candidate_offsets = np.repeat(candidate_offsets, candidate_counts)
    candidate_starts = np.repeat(candidate_starts, candidate_counts)
    second_indices = candidate_rows - candidate_offsets + candidate_starts
    y_overlap_mask = (
        edge_min_y[first_indices] <= edge_max_y[second_indices]
    ) & (edge_max_y[first_indices] >= edge_min_y[second_indices])
    different_polygon_mask = (
        polygon_ids[first_indices] != polygon_ids[second_indices]
    )
    candidate_mask = y_overlap_mask & different_polygon_mask
    first_indices = first_indices[candidate_mask]
    second_indices = second_indices[candidate_mask]
    candidate_array = np.hstack(
        (
            edge_array[first_indices],
            edge_array[second_indices],
            edge_ids[first_indices, None],
            edge_ids[second_indices, None],
        )
    )
    first_edges = candidate_array[:, :4]
    second_edges = candidate_array[:, 4:8]
    first_delta_x = first_edges[:, 2] - first_edges[:, 0]
    first_delta_y = first_edges[:, 3] - first_edges[:, 1]
    second_delta_x = second_edges[:, 2] - second_edges[:, 0]
    second_delta_y = second_edges[:, 3] - second_edges[:, 1]
    denominator = (
        second_delta_y * first_delta_x - second_delta_x * first_delta_y
    )

    relative_tolerance, absolute_tolerance = sg.get_defaults(
        ["rel_tol", "abs_tol"], [None, None]
    )
    parallel_mask = np.abs(denominator) <= np.maximum(
        absolute_tolerance, relative_tolerance * np.abs(denominator)
    )
    candidate_array = candidate_array[~parallel_mask]
    first_edges = first_edges[~parallel_mask]
    second_edges = second_edges[~parallel_mask]
    first_delta_x = first_delta_x[~parallel_mask]
    first_delta_y = first_delta_y[~parallel_mask]
    second_delta_x = second_delta_x[~parallel_mask]
    second_delta_y = second_delta_y[~parallel_mask]
    denominator = denominator[~parallel_mask]
    first_to_second_x = first_edges[:, 0] - second_edges[:, 0]
    first_to_second_y = first_edges[:, 1] - second_edges[:, 1]
    first_parameter = (
        second_delta_x * first_to_second_y - second_delta_y * first_to_second_x
    ) / denominator
    second_parameter = (
        first_delta_x * first_to_second_y - first_delta_y * first_to_second_x
    ) / denominator
    intersecting_mask = (
        (first_parameter >= 0)
        & (first_parameter <= 1)
        & (second_parameter >= 0)
        & (second_parameter <= 1)
    )
    intersection_x = first_edges[:, 0] + first_parameter * first_delta_x
    intersection_y = first_edges[:, 1] + first_parameter * first_delta_y
    intersection_rows = candidate_array[intersecting_mask]

    return [
        ((x_coordinate, y_coordinate), (int(first_id), int(second_id)))
        for x_coordinate, y_coordinate, first_id, second_id in zip(
            intersection_x[intersecting_mask],
            intersection_y[intersecting_mask],
            intersection_rows[:, 8],
            intersection_rows[:, 9],
        )
    ]


def all_edge_intersections(edges):
    """
    Return all proper intersections between the edges of ``edges``.

    Bounding-box candidates are collected into one NumPy array. Their
    line-segment intersections are then computed as NumPy arrays.
    """
    edge_coordinates = []
    for edge in edges:
        start_point, end_point = edge
        start_x, start_y = start_point[:2]
        end_x, end_y = end_point[:2]
        edge_coordinates.append([start_x, start_y, end_x, end_y])

    edge_array = np.asarray(edge_coordinates, dtype=float)
    edge_count = edge_array.shape[0]
    edge_min_x = np.minimum(edge_array[:, 0], edge_array[:, 2])
    edge_min_y = np.minimum(edge_array[:, 1], edge_array[:, 3])
    edge_max_x = np.maximum(edge_array[:, 0], edge_array[:, 2])
    edge_max_y = np.maximum(edge_array[:, 1], edge_array[:, 3])
    edge_ids = np.arange(edge_count)
    sort_order = edge_min_x.argsort()
    edge_array = edge_array[sort_order]
    edge_min_x = edge_min_x[sort_order]
    edge_min_y = edge_min_y[sort_order]
    edge_max_x = edge_max_x[sort_order]
    edge_max_y = edge_max_y[sort_order]
    edge_ids = edge_ids[sort_order]

    candidate_starts = np.arange(edge_count) + 1
    candidate_ends = np.searchsorted(edge_min_x, edge_max_x, side="right")
    candidate_counts = candidate_ends - candidate_starts
    candidate_count = candidate_counts.sum()
    if not candidate_count:
        return []

    candidate_offsets = np.cumsum(candidate_counts) - candidate_counts
    candidate_rows = np.arange(candidate_count)
    first_indices = np.repeat(np.arange(edge_count), candidate_counts)
    candidate_offsets = np.repeat(candidate_offsets, candidate_counts)
    candidate_starts = np.repeat(candidate_starts, candidate_counts)
    second_indices = candidate_rows - candidate_offsets + candidate_starts
    y_overlap_mask = (
        edge_min_y[first_indices] <= edge_max_y[second_indices]
    ) & (edge_max_y[first_indices] >= edge_min_y[second_indices])
    first_indices = first_indices[y_overlap_mask]
    second_indices = second_indices[y_overlap_mask]
    candidate_array = np.hstack(
        (
            edge_array[first_indices],
            edge_array[second_indices],
            edge_ids[first_indices, None],
            edge_ids[second_indices, None],
        )
    )
    first_edges = candidate_array[:, :4]
    second_edges = candidate_array[:, 4:8]
    first_delta_x = first_edges[:, 2] - first_edges[:, 0]
    first_delta_y = first_edges[:, 3] - first_edges[:, 1]
    second_delta_x = second_edges[:, 2] - second_edges[:, 0]
    second_delta_y = second_edges[:, 3] - second_edges[:, 1]
    denominator = (
        second_delta_y * first_delta_x - second_delta_x * first_delta_y
    )

    relative_tolerance, absolute_tolerance = sg.get_defaults(
        ["rel_tol", "abs_tol"], [None, None]
    )
    parallel_mask = np.abs(denominator) <= np.maximum(
        absolute_tolerance, relative_tolerance * np.abs(denominator)
    )
    candidate_array = candidate_array[~parallel_mask]
    first_edges = first_edges[~parallel_mask]
    second_edges = second_edges[~parallel_mask]
    first_delta_x = first_delta_x[~parallel_mask]
    first_delta_y = first_delta_y[~parallel_mask]
    second_delta_x = second_delta_x[~parallel_mask]
    second_delta_y = second_delta_y[~parallel_mask]
    denominator = denominator[~parallel_mask]
    first_to_second_x = first_edges[:, 0] - second_edges[:, 0]
    first_to_second_y = first_edges[:, 1] - second_edges[:, 1]
    first_parameter = (
        second_delta_x * first_to_second_y - second_delta_y * first_to_second_x
    ) / denominator
    second_parameter = (
        first_delta_x * first_to_second_y - first_delta_y * first_to_second_x
    ) / denominator
    intersecting_mask = (
        (first_parameter >= 0)
        & (first_parameter <= 1)
        & (second_parameter >= 0)
        & (second_parameter <= 1)
    )
    intersection_x = first_edges[:, 0] + first_parameter * first_delta_x
    intersection_y = first_edges[:, 1] + first_parameter * first_delta_y
    intersection_rows = candidate_array[intersecting_mask]

    return [
        ((x_coordinate, y_coordinate), (int(first_id), int(second_id)))
        for x_coordinate, y_coordinate, first_id, second_id in zip(
            intersection_x[intersecting_mask],
            intersection_y[intersecting_mask],
            intersection_rows[:, 8],
            intersection_rows[:, 9],
        )
    ]


def segment_cycles(segments, length_bound=10):
    """Given a list of line segments, returns all cycles."""
    coordinates = []
    for seg in segments:
        coordinates.extend(seg)

    d_node_coord, d_coord_node = node_dictionaries(coordinates)
    g_segments = [[d_coord_node[coord] for coord in seg] for seg in segments]

    nx_graph = nx.Graph()
    nx_graph.update(g_segments)
    # cycles = get_cycles(g_segments)
    cycles = list(nx.simple_cycles(nx_graph, length_bound=length_bound))
    res = []
    for cycle in cycles:
        res.append([d_node_coord[node] for node in cycle])

    return res, cycles


def get_partitions(shapes, length_bound=10):
    """Partition overlapping shapes into cycles from edge intersections.

    Args:
        shapes: Sequence of closed shapes whose edges may intersect.
        length_bound: Maximum cycle length retained by ``segment_cycles``.

    Returns:
        Result of the partition pipeline (cycles / sorted cycles as implemented).
    """
    n_rows = sum([len(shp.vertices) for shp in shapes])
    intersections = all_polygon_intersections(shapes)
    start = time.perf_counter_ns()
    points_by_edge = defaultdict(list)
    for point, (first_edge_id, second_edge_id) in intersections:
        rounded_point = sg.round_point(point)
        points_by_edge[first_edge_id].append(rounded_point)
        points_by_edge[second_edge_id].append(rounded_point)

    all_segments = []
    all_midpoints = []
    for i in range(n_rows):
        points = points_by_edge[i]
        if points:
            segments = segments_from_points(points)
            rounded_segments = [sg.round_segment(seg, 2) for seg in segments]
            all_segments.extend(rounded_segments)
            all_midpoints.extend(
                sg.midpoint(*segment) for segment in rounded_segments
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
    shp1 = shapes[0]
    shp2 = shapes[1]
    rest = shapes[2:]
    union = sg.union(shp1, shp2).merge_shapes()[0]
    for shape in rest:
        union = sg.union(union, shape).merge_shapes()[0]

    union_area = sg.polygon_area(union.vertices)

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
        if not sg.any_point_inside_polygon(candidate_midpoints, polygon_array):
            count += 1
            color = sg.black
            partition = sg.Shape(
                poly, closed=True, fill=True, color=color, alpha=0.8
            )
            partitions.append(partition)
            for edge in partition.edges:
                d_edge_partition[frozenset(edge)].add(partition.id)
            poly_area = abs(
                sg.polygon_area(poly)
            )  # Here, we are not strict about polygons' orientation
            area += poly_area

            if sg.isclose(area, union_area, rel_tol=0.001):
                print(
                    f"Total partition-area: {area:.2f}, Union-area: {union_area:.2f}"
                )
                done = True
    end = time.perf_counter_ns()
    print(
        f"Computing cycles with length_bounde={length_bound} took {get_time(c_start, c_end)}."
    )
    print(f"{len(partitions)} partitions.")
    n = max(len(p) for p in partitions)
    print(f"Largest* partition has {n} edges.")
    print(f"Used {count} cycles.")
    print(f"Computing partitions took {get_time(start, end)}.")

    return partitions, d_edge_partition, union

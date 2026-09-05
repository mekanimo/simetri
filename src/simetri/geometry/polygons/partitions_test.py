from collections import defaultdict
import time

import simetri.geometry.points.point_utils
import simetri.geometry.polygons.polygon
import simetri.geometry.segments.line_utils
import simetri.graphics as sg


import time

import networkx as nx
import numpy as np


DEBUG = False

new_time = []
old_time = []


def get_time(start, end, with_units=False):
    ms = (end - start) / 1e6
    units = ""
    if ms < 900:
        elapsed = ms
        if with_units:
            units = "milliseconds"
    else:
        elapsed = ms / 1000

        if with_units:
            units = "seconds"

    return f"{elapsed:.2f} {units}"


def to_numpy_array(ip_edge_list):
    rows = []
    for (x, y), (i1, i2) in ip_edge_list:
        rows.append([(x, y), i1, i2])
    return np.array(rows, dtype=object)


def filter_by_edge(arr, edge_id):
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


def node_dictionaries(coords: List, n_round: int = 2) -> list[dict]:
    """Set dictionaries for nodes and coordinates.
    d_node_coord: Dictionary of node id to coordinates.
    d_coord_node: Dictionary of coordinates to node id.

    Args:
        nodes (list[PointType]): List of vertices.
        n_round (int, optional): Number of rounding digits. Defaults to 2.
    """

    rounded = []
    for coord in coords:
        val = tuple(
            simetri.geometry.points.point_utils.round_point(coord, n_round)
        )
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


def point_in_polygon_strict(p, poly, eps=1e-5):
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


def cross(ax, ay, bx, by):
    return ax * by - ay * bx


def orient(a, b, c):
    # cross((b-a),(c-a))
    return cross(b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1])


def on_segment(a, b, p, eps=1e-12):
    # check collinear + within bbox
    if abs(orient(a, b, p)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def any_point_in_polygon(polygon, points):
    """Checks if any of the given points is "in" the given polygon.
    Points on edges or coincident with the vertices are not considered
    as in the polygon.
    """
    for p in points:
        if p in polygon:
            continue
        if point_in_polygon_strict(p, polygon):
            return True

    return False


def all_polygon_intersections(polygons):
    """
    Return all proper intersections between the polygons' edges.
    Polygons' edges are not checked for self intersections.
    This is for non-intersecting polygons only.

    Bounding-box candidates are collected into one NumPy array. Their
    line-segment intersections are then computed as NumPy arrays.
    """
    edge_coordinates = []
    polygon_ids = []
    for polygon_index, poly in enumerate(polygons):
        for edge in poly.edges:
            start_point, end_point = edge
            start_x, start_y = start_point[:2]
            end_x, end_y = end_point[:2]
            edge_coordinates.append([start_x, start_y, end_x, end_y])
            polygon_ids.append(polygon_index)

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

    # For sorted edge i, candidate_starts[i] skips the edge itself and all
    # earlier edges. candidate_ends[i] stops after the last later edge whose
    # minimum x-coordinate is no greater than edge i's maximum x-coordinate.
    # Thus candidate_starts[i]:candidate_ends[i] contains the possible x-overlaps.
    candidate_starts = np.arange(edge_count) + 1
    candidate_ends = np.searchsorted(edge_min_x, edge_max_x, side="right")
    candidate_counts = candidate_ends - candidate_starts
    candidate_count = candidate_counts.sum()
    if not candidate_count:
        return []

    # The flattened candidate list is grouped by first edge; each offset is
    # that edge's starting position in the flattened list.
    candidate_offsets = np.cumsum(candidate_counts) - candidate_counts
    candidate_rows = np.arange(candidate_count)
    # Each pair is represented by one index into the sorted edge array for
    # the first edge and one index for a later edge that may overlap it.
    first_indices = np.repeat(np.arange(edge_count), candidate_counts)
    candidate_offsets = np.repeat(candidate_offsets, candidate_counts)
    candidate_starts = np.repeat(candidate_starts, candidate_counts)
    second_indices = candidate_rows - candidate_offsets + candidate_starts
    # first_indices[i] and second_indices[i] identify the two edges in row i.
    y_overlap_mask = (
        edge_min_y[first_indices] <= edge_max_y[second_indices]
    ) & (edge_max_y[first_indices] >= edge_min_y[second_indices])
    different_polygon_mask = (
        polygon_ids[first_indices] != polygon_ids[second_indices]
    )
    candidate_mask = y_overlap_mask & different_polygon_mask
    # Retain only the paired edge indices that pass the bounding-box and
    # different-polygon tests.
    first_indices = first_indices[candidate_mask]
    second_indices = second_indices[candidate_mask]
    # Keep only edge pairs whose bounding boxes overlap in both dimensions
    # and whose edges belong to different polygons. The box test only finds
    # possible intersections; the segment calculation below confirms them.
    # Columns: e1 x1, e1 y1, e1 x2, e1 y2, e2 x1, e2 y1, e2 x2, e2 y2,
    #          e1 id, e2 id.
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
    # Parallel candidates have a zero cross-product denominator, so their
    # lines do not meet at one unique point and must not be divided by it.
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
    Return all proper intersections between the edges.

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


def all_intersections(edges):
    """
    Return all proper intersections between the edges of ``edges``.

    Bounding-box candidates are collected into one NumPy array. Their
    line-segment intersections are then computed as NumPy arrays.
    """
    start = time.perf_counter_ns()
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
    end = time.perf_counter_ns()
    if DEBUG:
        print(f"Computing all intersections took {get_time(start, end)}.")

    return [
        ((x_coordinate, y_coordinate), (int(first_id), int(second_id)))
        for x_coordinate, y_coordinate, first_id, second_id in zip(
            intersection_x[intersecting_mask],
            intersection_y[intersecting_mask],
            intersection_rows[:, 8],
            intersection_rows[:, 9],
        )
    ]


def get_partitions_old(shapes, length_bound=10):
    n_rows = sum([len(shp.vertices) for shp in shapes])
    intersections = all_intersections(
        shapes.all_segments
    )  # ((x, y), (id1, id2))
    start = time.perf_counter_ns()
    res_array = to_numpy_array(intersections)  # ((x, y), edge1_id, edge2_id)
    all_segments = []
    all_midpoints = []
    for i in range(n_rows):
        points = filter_by_edge(res_array, i)[:, 0]
        points = [
            simetri.geometry.points.point_utils.round_point(p) for p in points
        ]
        if points:
            segments = segments_from_points(points)
            all_segments.extend(
                [
                    simetri.geometry.segments.line_utils.round_segment(seg, 2)
                    for seg in segments
                ]
            )
            all_midpoints.extend(
                [
                    simetri.geometry.points.point_utils.midpoint(*seg)
                    for seg in all_segments
                ]
            )

    all_midpoints = set(all_midpoints)
    d_ip_polygons = {}
    d_node_polygons = {}
    segment_coordinates = []
    for seg in all_segments:
        segment_coordinates.extend(seg)

    d_node_coord, d_coord_node = node_dictionaries(segment_coordinates)

    # for x in intersections:
    #     ip, edges = x
    #     ip = sg.round_point(ip, 2) # is this necessary?
    c_start = time.perf_counter_ns()
    cycles, cycles_nodes = segment_cycles(
        all_segments, length_bound=length_bound
    )
    c_end = time.perf_counter_ns()
    if DEBUG:
        print(
            f"Number of total cycles with less than {length_bound} nodes: {len(cycles)}"
        )
    cycles, cycles_nodes = zip(
        *sorted(zip(cycles, cycles_nodes), key=len)
    )  # sort cycles and cycles_nodes in sync
    shp1 = shapes[0]
    shp2 = shapes[1]
    rest = shapes[2:]
    union = sg.union(shp1, shp2).merge_shapes()[0]
    for shape in rest:
        union = sg.union(union, shape).merge_shapes()[0]

    union_area = simetri.geometry.polygons.polygon.polygon_area(union.vertices)

    count = 0
    area = 0
    partitions = []
    d_edge_partition = defaultdict(set)
    done = False
    for poly in cycles:
        if done:
            break
        if not any_point_in_polygon(poly, all_midpoints):
            count += 1
            color = sg.black
            partition = sg.Shape(
                poly, closed=True, fill=True, color=color, alpha=0.8
            )
            partitions.append(partition)
            for edge in partition.edges:
                d_edge_partition[frozenset(edge)].add(partition.id)
            poly_area = abs(
                simetri.geometry.polygons.polygon.polygon_area(poly)
            )  # Here, we are not strict about polygons' orientation
            area += poly_area

            if sg.isclose(area, union_area, rel_tol=0.001):
                if DEBUG:
                    print(
                        f"Total partition-area: {area:.2f}, Union-area: {union_area:.2f}"
                    )
                done = True
    end = time.perf_counter_ns()
    if DEBUG:
        print(
            f"Computing cycles with length_bounde={length_bound} took {get_time(c_start, c_end)}."
        )
        print(f"{len(partitions)} partitions.")
        n = max(len(p) for p in partitions)
        print(f"Largest* partition has {n} edges.")
        print(f"Used {count} cycles.")
        print(f"Computing partitions took {get_time(start, end)}.")

    return partitions, d_edge_partition, union


def get_partitions(shapes, length_bound=10):
    n_rows = sum([len(shp.vertices) for shp in shapes])
    intersections = all_polygon_intersections(shapes)
    start = time.perf_counter_ns()
    if not intersections:
        partitions = list(shapes)
        d_edge_partition = defaultdict(set)
        for partition in partitions:
            for edge in partition.edges:
                d_edge_partition[frozenset(edge)].add(partition.id)
        union = sg.union(partitions[0], partitions[1]).merge_shapes()[0]
        for shape in partitions[2:]:
            union = sg.union(union, shape).merge_shapes()[0]
        return partitions, d_edge_partition, union

    points_by_edge = defaultdict(list)
    for edge_id, edge in enumerate(shapes.all_segments):
        start_point, end_point = edge
        points_by_edge[edge_id].extend(
            (
                simetri.geometry.points.point_utils.round_point(start_point),
                simetri.geometry.points.point_utils.round_point(end_point),
            )
        )
    for point, (first_edge_id, second_edge_id) in intersections:
        rounded_point = simetri.geometry.points.point_utils.round_point(point)
        points_by_edge[first_edge_id].append(rounded_point)
        points_by_edge[second_edge_id].append(rounded_point)

    all_segments = []
    all_midpoints = []
    for i in range(n_rows):
        points = points_by_edge[i]
        if len(points) >= 2:
            segments = segments_from_points(points)
            rounded_segments = [
                simetri.geometry.segments.line_utils.round_segment(seg, 2)
                for seg in segments
            ]
            all_segments.extend(rounded_segments)
            all_midpoints.extend(
                simetri.geometry.points.point_utils.midpoint(*segment)
                for segment in rounded_segments
            )

    midpoint_array = np.unique(np.asarray(all_midpoints, dtype=float), axis=0)
    midpoint_order = midpoint_array[:, 0].argsort()
    midpoint_array = midpoint_array[midpoint_order]

    c_start = time.perf_counter_ns()
    cycles, cycles_nodes = segment_cycles(
        all_segments, length_bound=length_bound
    )
    c_end = time.perf_counter_ns()
    if DEBUG:
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

    union_area = simetri.geometry.polygons.polygon.polygon_area(union.vertices)

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
        if not simetri.geometry.polygons.polygon.any_point_inside_polygon(
            candidate_midpoints, polygon_array
        ):
            count += 1
            color = sg.black
            partition = sg.Shape(
                poly, closed=True, fill=True, color=color, alpha=0.8
            )
            partitions.append(partition)
            for edge in partition.edges:
                d_edge_partition[frozenset(edge)].add(partition.id)
            poly_area = abs(
                simetri.geometry.polygons.polygon.polygon_area(poly)
            )  # Here, we are not strict about polygons' orientation
            area += poly_area

            if sg.isclose(area, union_area, rel_tol=0.001):
                if DEBUG:
                    print(
                        f"Total partition-area: {area:.2f}, Union-area: {union_area:.2f}"
                    )
                done = True
    end = time.perf_counter_ns()
    if DEBUG:
        print(
            f"Computing cycles with length_bounds = {length_bound} took {get_time(c_start, c_end)}."
        )
        print(f"{len(partitions)} partitions.")
        n = max((len(p) for p in partitions), default=0)
        print(f"Largest* partition has {n} edges.")
        print(f"Used {count} cycles.")
        print(f"Computing partitions took {get_time(start, end)}.")

    return partitions, d_edge_partition, union


def symmetric_difference(shapes, length_bound=10, new=True):
    start = time.perf_counter_ns()
    if new:
        partitions, d_edge_part, union = get_partitions(shapes, length_bound)
    else:
        partitions, d_edge_part, union = get_partitions_old(
            shapes, length_bound
        )

    set_fills(partitions, d_edge_part)
    end = time.perf_counter_ns()
    #########################################################
    # print(f"Total elapsed time was {get_time(start, end)}.")
    elapsed = get_time(start, end, False)
    if new:
        new_time.append(elapsed)
    else:
        old_time.append(elapsed)
    print(f"Total time: {elapsed}")
    #########################################################
    if DEBUG:
        print(f"Total elapsed time was {get_time(start, end)}.")
        print(
            f"* Largest partition that can be computed with length_bound = {length_bound}."
        )
    return partitions, union


def draw_symm_diff(shapes, length_bounds=10, new=True):
    canvas = sg.Canvas()
    partitions, union = symmetric_difference(shapes, length_bounds, new)
    if DEBUG:
        dy = sg.Group(partitions).height * 1.15
        canvas.draw(shapes, fill=False)
        canvas.translate(0, -dy)
        canvas.draw(union, fill=False)
        canvas.translate(0, -dy)
        for part in partitions:
            if not part.fill:
                color = sg.yellow
            else:
                color = sg.navy
            canvas.draw(part, fill=True, color=color)
        canvas.save("c:/tmp/partitions_test.svg", overwrite=True)


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
    start = time.perf_counter_ns()
    queue = set()
    for edge, part in d_edge_part.items():
        if len(part) == 1:
            cur_part = sg.d_id_obj[list(part)[0]]
            queue = set([cur_part.id])
            cur_part.fill = True
            break
    # If an edge is between two partitions,
    # only one partition can be filled.
    # Alternate through all partitions.
    processed = set()
    n_partitions = len(partitions)
    count = 0
    qcount = 0
    while queue:
        count += 1
        cur_part = sg.d_id_obj[queue.pop()]
        count += 1
        if cur_part.id not in processed:
            qcount += 1
            edges = [frozenset(e) for e in cur_part.edges]
            for edge in edges:
                partitions = d_edge_part[edge]
                if len(partitions) == 2:
                    part1, part2 = [sg.d_id_obj[p] for p in partitions]
                    fill = not cur_part.fill
                    if part1 == cur_part:
                        part2.fill = fill
                    else:
                        part1.fill = fill

                queue.update(set(partitions))
            processed.add(cur_part.id)
            queue.difference_update(processed)
    end = time.perf_counter_ns()
    if DEBUG:
        print(
            f"Visited {count} partitions, processed {qcount} for symmetric difference coloring."
        )

        print(
            f"Coloring shapes for symmetric difference took {get_time(start, end)}."
        )


def square_test():
    points = [(0, 0), (40, 0), (40, 40), (0, 40)]
    square = sg.Shape(points, closed=True)
    segments = []
    for i in range(2, 8):
        squares = square.translate(20, 30, reps=1).translate(50, 0, reps=i * 3)
        squares.translate(-10, 50, reps=2)
        segments.append(len(squares.all_segments))
        print(
            f"{len(squares.all_segments)}",
        )
        # print(f"Total number of edges: {len(squares.all_segments)}")
        # print("New Partition Algorithm")
        draw_symm_diff(squares, 12, new=True)
        # print("Old Partition Algorithm")
        draw_symm_diff(squares, 12, new=False)

    print(segments)
    print(old_time)
    print(new_time)
    print(list(zip(segments, [float(t) for t in new_time])))
    print(list(zip(segments, [float(t) for t in old_time])))


# poly1 = sg.reg_poly_shape(16, 50)
# poly2 = sg.reg_poly_shape(16, 70)
# circles = sg.Group([poly1, poly2])
# circles2 = circles.copy().translate(90).rotate(sg.pi / 3, reps=5)
# all_circles = circles + circles2

# draw_symm_diff(all_circles, 19, new=True)
# draw_symm_diff(all_circles, 19, new=False)


def fit_k_p(pairs):
    """
    Given (n, time) pairs, fit T(n) = k * n^p
    and return (k, p).
    """
    pairs = np.array(pairs, dtype=float)
    n = pairs[:, 0]
    t = pairs[:, 1]

    # Filter out non-positive times
    mask = t > 0
    n = n[mask]
    t = t[mask]

    logn = np.log(n)
    logt = np.log(t)

    # Linear regression: logt = a + p * logn
    p, a = np.polyfit(logn, logt, 1)

    k = np.exp(a)
    return k, p


def best_fit_exponent(pairs):
    """
    Given a list of (n, time) pairs, returns the best-fit exponent p
    in the model T(n) = k * n^p using log-log linear regression.
    """
    pairs = np.array(pairs, dtype=float)
    n = pairs[:, 0]
    t = pairs[:, 1]

    # Filter out zero or negative times to avoid log issues
    mask = t > 0
    n = n[mask]
    t = t[mask]

    # Linear regression on log-log scale
    logn = np.log(n)
    logt = np.log(t)

    # p = slope of log(t) vs log(n)
    p, _ = np.polyfit(logn, logt, 1)

    return p


new_data = [
    (72, 0.221),
    (96, 0.393),
    (120, 0.609),
    (144, 0.97),
    (168, 1.38),
    (192, 1.94),
]
old_data = [
    (72, 0.303),
    (96, 0.585),
    (120, 0.97),
    (144, 1.44),
    (168, 2.13),
    (192, 2.87),
]

print(best_fit_exponent(new_data))
print(best_fit_exponent(old_data))

print(fit_k_p(new_data))
print(fit_k_p(old_data))

k_new, p_new = (1.620484526420594e-05, 2.215201133327269)
k_old, p_old = (1.6801226817690227e-05, 2.290521421294983)
# T(n) = k x n^p
# T_new(n) = 1.62e-5 x n^2.21
# T_old(n) = 1.68e-5 x n^2.29

percent_differences = []
for i in range(len(old_data)):
    percent_differences.append(new_data[i][1] / old_data[i][1])
print(f"Percent differences: {percent_differences}")

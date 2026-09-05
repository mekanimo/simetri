from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from itertools import chain, combinations
from itertools import cycle as cycle_
from math import ceil, log10, sqrt

import networkx as nx
import numpy as np

import simetri.geometry.points.point_utils
import simetri.geometry.polygons.polygon
import simetri.geometry.segments.line_utils
import simetri.graphics as sg

flatten = chain.from_iterable
rel_tol, abs_tol = sg.get_defaults(["rel_tol", "abs_tol"], [None, None])


def rp(min_val=-200, max_val=200):
    return (
        float(random.randint(min_val, max_val)),
        float(random.randint(min_val, max_val)),
    )


def edge_polygon_dict(polygons):
    """
    polygons: list of polygon shapes
              each polygon is a list of vertices (any format)

    Returns a flat list of polygon IDs, one per edge.
    """
    d_edge_polygon = {}
    count = 0
    for i, poly in enumerate(polygons):
        edge_count = len(poly)
        for edge in range(edge_count):
            d_edge_polygon[count] = i
            count += 1

    return d_edge_polygon


def polygon_bounds_ids(polygons):
    """
    polygons: list of polygon shapes

    Polygon ids are the indices of the polygons list.
    xmin, ymin, xmax, ymax computed using boundary boxes
    of the polygons.
    Returns an array [xmin, ymin, xmax, ymax, p_id].
    """
    bounds_ids = []
    for i, poly in enumerate(polygons):
        edge_count = len(poly)
        xmin, ymin = poly.southwest
        xmax, ymax = poly.northeast
        for j in range(edge_count):
            bounds_ids.append([xmin, ymin, xmax, ymax, i])

    return bounds_ids


def get_edge_array(polygons):
    """
    polygons: list of polygon shapes
    """
    edges = [list(x.edges) for x in polygons]
    edges = list(flatten(edges))
    edge_coordinates = []

    for edge in edges:
        edge_coordinates.append(list(flatten(edge)))
    edge_arr = np.array(edge_coordinates)

    return edge_arr


def point_in_polygon_np(point, polygon):
    """
    point: (x, y) tuple or array-like
    polygon: Nx2 NumPy array of vertices (ordered CW or CCW)

    Returns True if point is inside or on the boundary.
    """

    px, py = point
    poly = np.asarray(polygon)

    # Extract vertices
    x1 = poly[:, 0]
    y1 = poly[:, 1]
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)

    # ---- Boundary check (vectorized collinearity + bounding box) ----
    # Compute cross product for collinearity
    cross = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)

    on_segment = (
        (np.abs(cross) < 1e-12)
        & (np.minimum(x1, x2) <= px)
        & (px <= np.maximum(x1, x2))
        & (np.minimum(y1, y2) <= py)
        & (py <= np.maximum(y1, y2))
    )

    if np.any(on_segment):
        return True

    # ---- Ray casting: count intersections ----
    # Check if edge crosses horizontal ray to the right
    cond = (y1 > py) != (y2 > py)

    # x coordinate of intersection with horizontal ray
    x_intersect = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-15) + x1

    # Count intersections where intersection is to the right of point
    inside = np.sum(cond & (px < x_intersect)) % 2 == 1

    return inside


def cross(ax, ay, bx, by):
    return ax * by - ay * bx


def orient(a, b, c):
    # cross((b-a),(c-a))
    return cross(b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1])


def dot(ax, ay, bx, by):
    return ax * bx + ay * by


def on_segment(a, b, p, eps=1e-12):
    # check collinear + within bbox
    if abs(orient(a, b, p)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def get_edge_candidates(i, i_xmin, i_ymin, i_xmax, i_ymax, edge_array):
    start = i + 1

    (
        x1,
        y1,
        x2,
        y2,
        edge_xmin,
        edge_ymin,
        edge_xmax,
        edge_ymax,
        id1,
    ) = edge_array[i, :]
    id1 = int(id1)
    segment = [x1, y1, x2, y2]
    # if the boundary boxes overlap, there may be an intersection
    edge_candidates = edge_array[start:, :][
        (
            (
                (edge_array[start:, i_xmax] >= edge_xmin)
                & (edge_array[start:, i_xmin] <= edge_xmax)
            )
            & (
                (edge_array[start:, i_ymax] >= edge_ymin)
                & (edge_array[start:, i_ymin] <= edge_ymax)
            )
        )
    ]
    return segment, id1, edge_candidates


def get_poly_candidates(
    start, x, y, i_pxmin, i_pymin, i_pxmax, i_pymax, edge_array
):
    poly_candidates = edge_array[start:, :][
        (
            (
                (edge_array[start:, i_pxmax] >= x)
                & (edge_array[start:, i_pxmin] <= x)
            )
            & (
                (edge_array[start:, i_pymax] >= y)
                & (edge_array[start:, i_pymin] <= y)
            )
        )
    ]
    return poly_candidates


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


def segment_cycles(segments, dist_tol, debug=False):
    """Given a list of line segments, returns all cycles."""
    coordinates = []
    for seg in segments:
        coordinates.extend(seg)

    d_node_coord, d_coord_node, d_rounded_coord = node_dictionaries(
        coordinates, dist_tol, debug=debug
    )
    g_segments = [[d_coord_node[coord] for coord in seg] for seg in segments]

    nx_graph = nx.Graph()
    nx_graph.update(g_segments)

    cycles = list(nx.simple_cycles(nx_graph))
    res = []
    for cycle in cycles:
        res.append([d_node_coord[node] for node in cycle])

    return res, cycles


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
        val = tuple(
            simetri.geometry.points.point_utils.round_point(coord, n_round)
        )
        rounded.append(val)
        d_rounded_coord[val] = coord

    rounded_coords = list(set(rounded))
    rounded_coords.sort()
    rounded_coords.sort(key=lambda point: point[1])

    indexed_coordinates = [
        (*coordinate[:2], index)
        for index, coordinate in enumerate(rounded_coords)
    ]
    _, close_pairs = simetri.geometry.polygons.polygon.all_close_points(
        indexed_coordinates, dist_tol=dist_tol
    )
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
            first_coordinate = tuple(
                simetri.geometry.points.point_utils.round_point(
                    first_point, n_round
                )
            )
            second_coordinate = tuple(
                simetri.geometry.points.point_utils.round_point(
                    second_point, n_round
                )
            )
            if (
                d_coord_node[first_coordinate]
                == d_coord_node[second_coordinate]
            ):
                continue
            point_distance = simetri.geometry.points.point_utils.distance(
                first_point, second_point
            )
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


def any_point_in_polygon_strict(points, polygon, eps=1e-12):
    """
    Returns True if ANY point is strictly inside the polygon.
    Boundary points are treated as outside.

    points:  (M,2) NumPy array
    polygon: (N,2) NumPy array
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


def to_numpy_array(ip_edge_poly_list):
    rows = []
    for (x, y), (i1, i2), plist in ip_edge_poly_list:
        rows.append([(x, y), i1, i2, tuple(plist)])
    return np.array(rows, dtype=object)


def all_intersections(edges):
    """Return all proper intersections between the edges of ``shapes``."""
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

    relative_tolerance, absolute_tolerance = sg.get_defaults(
        ["rel_tol", "abs_tol"], [None, None]
    )
    intersections = []
    candidate_counts = []
    for edge_index in range(edge_count):
        candidate_start = edge_index + 1
        candidate_x_min = edge_min_x[candidate_start:]
        candidate_y_min = edge_min_y[candidate_start:]
        candidate_x_max = edge_max_x[candidate_start:]
        candidate_y_max = edge_max_y[candidate_start:]
        candidate_mask = (
            (candidate_x_min <= edge_max_x[edge_index])
            & (candidate_x_max >= edge_min_x[edge_index])
            & (candidate_y_min <= edge_max_y[edge_index])
            & (candidate_y_max >= edge_min_y[edge_index])
        )
        candidate_indices = np.flatnonzero(candidate_mask) + candidate_start
        candidate_counts.append(len(candidate_indices))
        if candidate_indices.size == 0:
            continue

        first_edge = edge_array[edge_index]
        second_edges = edge_array[candidate_indices]
        first_delta_x = first_edge[2] - first_edge[0]
        first_delta_y = first_edge[3] - first_edge[1]
        second_delta_x = second_edges[:, 2] - second_edges[:, 0]
        second_delta_y = second_edges[:, 3] - second_edges[:, 1]
        denominator = (
            second_delta_y * first_delta_x - second_delta_x * first_delta_y
        )
        parallel_mask = np.abs(denominator) <= np.maximum(
            absolute_tolerance, relative_tolerance * np.abs(denominator)
        )
        nonparallel_indices = np.flatnonzero(~parallel_mask)
        if nonparallel_indices.size == 0:
            continue

        valid_denominator = denominator[nonparallel_indices]
        valid_edges = second_edges[nonparallel_indices]
        first_to_second_x = first_edge[0] - valid_edges[:, 0]
        first_to_second_y = first_edge[1] - valid_edges[:, 1]
        first_parameter = (
            valid_edges[:, 2] - valid_edges[:, 0]
        ) * first_to_second_y - (
            valid_edges[:, 3] - valid_edges[:, 1]
        ) * first_to_second_x
        first_parameter /= valid_denominator
        second_parameter = (
            first_delta_x * first_to_second_y
            - first_delta_y * first_to_second_x
        ) / valid_denominator
        intersecting_mask = (
            (first_parameter >= 0)
            & (first_parameter <= 1)
            & (second_parameter >= 0)
            & (second_parameter <= 1)
        )

        for valid_index in np.flatnonzero(intersecting_mask):
            intersection_x = (
                first_edge[0] + first_parameter[valid_index] * first_delta_x
            )
            intersection_y = (
                first_edge[1] + first_parameter[valid_index] * first_delta_y
            )
            second_index = nonparallel_indices[valid_index]
            intersections.append(
                (
                    (intersection_x, intersection_y),
                    (
                        int(edge_ids[edge_index]),
                        int(edge_ids[candidate_indices[second_index]]),
                    ),
                )
            )
    # print(candidate_counts)
    return intersections


def all_intersections_vectorized(edges):
    """
    Return all proper intersections between the edges of ``edges``.

    Bounding-box candidates are collected into one NumPy array. Their
    line-segment intersections are then computed as NumPy arrays. The
    returned format matches :func:`all_intersections`.
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


def get_candidates(edges):
    """Return the edge pairs whose bounding boxes overlap."""
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

    active_indices = []
    candidate_rows = []
    candidate_first_indices = []
    for edge_index in range(edge_count):
        current_min_x = edge_min_x[edge_index]
        active_indices = [
            active_index
            for active_index in active_indices
            if edge_max_x[active_index] >= current_min_x
        ]
        if active_indices:
            active_array = np.asarray(active_indices, dtype=int)
            y_overlap_mask = (
                edge_min_y[active_array] <= edge_max_y[edge_index]
            ) & (edge_max_y[active_array] >= edge_min_y[edge_index])
            candidate_indices = active_array[y_overlap_mask]
            if candidate_indices.size:
                second_edge_rows = np.broadcast_to(
                    edge_array[edge_index],
                    (candidate_indices.size, edge_array.shape[1]),
                )
                candidate_rows.append(
                    np.hstack(
                        (
                            edge_array[candidate_indices],
                            second_edge_rows,
                            edge_ids[candidate_indices, None],
                            np.full(
                                (candidate_indices.size, 1),
                                edge_ids[edge_index],
                            ),
                        )
                    )
                )
                candidate_first_indices.append(candidate_indices)
        active_indices.append(edge_index)

    if not candidate_rows:
        res = np.empty((0, 10), dtype=float)
    else:
        candidate_array = np.concatenate(candidate_rows, axis=0)
        candidate_first_indices = np.concatenate(candidate_first_indices)
        candidate_order = np.argsort(candidate_first_indices, kind="stable")

        res = candidate_array[candidate_order]

    # print(len(res))
    return res


def all_intersections_bbox(edges):
    """Return intersections using an active bounding-box sweep."""
    candidate_array = get_candidates(edges)
    if not candidate_array.size:
        return []

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


def set_fills(partitions, d_edge_part):
    # find an edge with a single partition
    # add the partition to the queue
    # this means this is one of the outermost partitions
    # outer partitions are always filled
    for edge, part in d_edge_part.items():
        if len(part) == 1:
            cur_part = sg.d_id_obj[list(part)[0]]
            queue = set([cur_part.id])
            cur_part.fill = True
            break

    # only one of the neighbors can be filled
    # alternate through all partitions
    processed = set()
    qcount = 0
    while queue:
        cur_part = sg.d_id_obj[queue.pop()]
        # qcount += 1
        if cur_part.id not in processed:
            # print(queue)
            edges = [frozenset(e) for e in cur_part.edges]
            for edge in edges:
                partitions = d_edge_part[edge]
                if len(partitions) == 2:
                    part1, part2 = [sg.d_id_obj[p] for p in partitions]
                    fill = not cur_part.fill
                    if part1 == cur_part and part2 not in processed:
                        part2.fill = fill
                    elif part1 not in processed:
                        part1.fill = fill

                queue.update(set(partitions))
                queue.difference_update(set([cur_part.id]))
            processed.add(cur_part.id)
    # print(f"{qcount} times in queue.")


def symmetric_difference(shapes, dist_tol, debug=False):
    n_rows = sum([len(shp.vertices) for shp in shapes])
    ips_edges = all_ips_edges(shapes)

    res_array = to_numpy_array2(ips_edges)

    all_segments = []
    all_midpoints = []
    for i in range(n_rows):
        points = filter_by_edge(res_array, i)[:, 0]
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

    d_node_coord, d_coord_node, d_rounded_coord = node_dictionaries(
        segment_coordinates, dist_tol, debug=debug
    )

    for x in ips_edges:
        ip, edges = x
        ip = simetri.geometry.points.point_utils.round_point(
            ip, 2
        )  # is this necessary?

    cycles, cycles_nodes = segment_cycles(all_segments, dist_tol, debug=debug)
    print("Number of total cycles:", len(cycles))
    cycles.sort(key=len)  # partitions are usually the smallest cycles
    cycles_nodes.sort(
        key=len
    )  # let's pray cycles and cycles_nodes are sorted the same way
    # cycles, cycles_nodes = zip(*sorted(zip(cycles, cycles_nodes), key=len)) # sort cycles and cycles_nodes in sync
    count = 0
    area = 0
    shp1 = shapes[0]
    shp2 = shapes[1]
    rest = shapes[2:]
    union = sg.union(shp1, shp2).merge_shapes()[0]
    for shape in rest:
        union = sg.union(union, shape).merge_shapes()[0]

    union_area = simetri.geometry.polygons.polygon.polygon_area(union.vertices)

    count = 0
    colors = [
        sg.green,
        sg.red,
        sg.blue,
        sg.yellow,
        sg.gold,
        sg.black,
        sg.brown,
        sg.teal,
        sg.aqua,
        sg.navy,
        sg.light_gray,
    ]
    partitions = []
    d_edge_partition = defaultdict(set)
    done = False
    for poly in cycles:
        count += 1
        if done:
            break
        # print('np.array(list(all_midpoints)', np.array(list(all_midpoints)[:3]))
        # print('midpoints', list(all_midpoints)[:3])
        # print('poly', poly)
        # print('any_point_in_polygon(poly, all_midpoints)', any_point_in_polygon(poly, all_midpoints))
        # print('any_point_in_polygon_strict(np.array(list(all_midpoints)), poly)', any_point_in_polygon_strict(np.array(list(all_midpoints)), poly))
        if not any_point_in_polygon(poly, all_midpoints):
            # if not any_point_in_polygon_strict(np.array(list(all_midpoints)), poly):
            color = sg.black
            partition = sg.Shape(
                poly, closed=True, fill=True, color=color, alpha=0.8
            )
            partitions.append(partition)
            for edge in partition.edges:
                d_edge_partition[frozenset(edge)].add(partition.id)
            # canvas.text(f'{count-1}', sg.polygon_cg(poly))
            poly_area = abs(
                simetri.geometry.polygons.polygon.polygon_area(poly)
            )  # Here, we are not strict about polygons' orientation
            area += poly_area

            if sg.isclose(area, union_area, rel_tol=0.001):
                print(f"area: {area}, union_area: {union_area}")
                done = True
    print(f"{len(partitions)} partitions.")
    print(f"Used {count} cycles.")
    return partitions, d_edge_partition, union


def symm_diff():
    dist_tol = sg.get_defaults(["dist_tol"], [None])[0]
    sqr1 = sg.reg_poly_shape(4, 100)
    sqr2 = sg.reg_poly_shape(4, 100, (40, 0))
    rect = sg.Rectangle((40, 0), 300, 50)

    shapes = [sqr1, sqr2, rect]
    partitions, d_edge_part, union = symmetric_difference(shapes, dist_tol)
    set_fills(partitions, d_edge_part)
    canvas = sg.Canvas()
    for part in partitions:
        if not part.fill:
            color = sg.yellow
        else:
            color = sg.navy
        canvas.draw(part, fill=True, color=color)
    canvas.translate(0, -250)
    canvas.draw(shapes, fill=False)
    canvas.translate(0, -250)
    canvas.draw(union, fill=False)

    canvas.translate(0, -300)

    ###########################################################

    n = 6
    r = 50
    hex = sg.reg_poly_shape(n, r)
    hex2 = sg.reg_poly_shape(n, r)
    hex3 = sg.reg_poly_shape(n, r)
    hex4 = sg.reg_poly_shape(n, r)
    d = 30
    hex2.translate(d, d)
    hex3.translate(d, -d)
    hex4.translate(-d, -d / 2)

    shapes = sg.Group([hex, hex2, hex3, hex4])
    partitions, d_edge_part, union = symmetric_difference(shapes, dist_tol)
    set_fills(partitions, d_edge_part)
    # canvas = sg.Canvas()
    for part in partitions:
        if not part.fill:
            color = sg.yellow
        else:
            color = sg.navy
        canvas.draw(part, fill=True, color=color)
    canvas.translate(0, -200)
    canvas.draw(shapes, fill=False)
    canvas.translate(0, -200)
    canvas.draw(union, fill=False)

    canvas.translate(0, -300)
    ###########################################################

    n = 4
    r = 50
    shp = sg.reg_poly_shape(n, r)
    shp2 = sg.reg_poly_shape(n, r)
    shp3 = sg.reg_poly_shape(n, r)
    shp4 = sg.reg_poly_shape(n, r)
    shp5 = sg.reg_poly_shape(n, r * 1.5)
    d = 30
    shp2.translate(1.25 * d, 0.75 * d)
    shp3.translate(0.65 * d, -1.2 * d)
    shp4.translate(-0.45 * d, -1.75 * d)
    shp5.translate(-0.75 * d, 0.45 * d)
    shapes = sg.Group([shp, shp2, shp3, shp4, shp5])
    partitions, d_edge_part, union = symmetric_difference(shapes, dist_tol)
    set_fills(partitions, d_edge_part)
    # canvas = sg.Canvas()
    for part in partitions:
        if not part.fill:
            color = sg.yellow
        else:
            color = sg.navy
        canvas.draw(part, fill=True, color=color)
    canvas.translate(0, -200)
    canvas.draw(shapes, fill=False)
    canvas.translate(0, -200)
    canvas.draw(union, fill=False)
    canvas.translate(0, -300)

    canvas.save("c:/tmp/all_intersections2.svg")


#####################
# Time Test
#####################
import time
import random


def time_test():
    for n in [2, 10, 20, 50, 100, 200, 300, 400]:
        shapes = []
        for i in range(n):
            verts = []
            for j in range(10):
                verts.append(rp())
            shapes.append(sg.Shape(verts, closed=True))
        start = time.perf_counter_ns()
        res = all_ips_edges(shapes)

        end = time.perf_counter_ns()
        print(
            f"{n} polygons took: {end - start:.5f} seconds. Found {len(res)} intersections."
        )

    # Old Results:

    # 2 polygons took: 0.00048 seconds. Found 52 intersections.
    # 10 polygons took: 0.00544 seconds. Found 1179 intersections.
    # 20 polygons took: 0.01819 seconds. Found 4716 intersections.
    # 50 polygons took: 0.12718 seconds. Found 27507 intersections.
    # 100 polygons took: 0.42716 seconds. Found 113123 intersections.
    # 200 polygons took: 1.79236 seconds. Found 456118 intersections.
    # 300 polygons took: 4.28258 seconds. Found 1048683 intersections.
    # 400 polygons took: 7.85771 seconds. Found 1865323 intersections.


# New Results:

# 2 polygons took: 0.00065 seconds. Found 59 intersections.
# 10 polygons took: 0.00340 seconds. Found 1112 intersections.
# 20 polygons took: 0.00814 seconds. Found 4490 intersections.
# 50 polygons took: 0.05054 seconds. Found 30803 intersections.
# 100 polygons took: 0.10811 seconds. Found 116456 intersections.
# 200 polygons took: 0.36611 seconds. Found 451960 intersections.
# 300 polygons took: 0.84888 seconds. Found 1068701 intersections.
# 400 polygons took: 1.45236 seconds. Found 1832816 intersections.


@dataclass
class Poly:
    edges: tuple


# def run_tests():
# # n = 2000
# # for i in range(n):
# #     verts = []
# #     for j in range(10):
# #         verts.append(rp())
# #     shapes.append(sg.Shape(verts, closed=True))

# for n in [100, 200, 500, 1000, 2000, 3000]:
#     shapes = []
#     for i in range(n):
#         shapes.append((rp(), rp()))

#     # print(shapes)
#     # shapes = [
#     #     ((200.0, 135.0), (18.0, 94.0)),
#     #     ((77.0, 62.0), (177.0, 185.0)),
#     #     ((44.0, 22.0), (184.0, 155.0)),
#     #     ((154.0, 122.0), (114.0, 50.0)),
#     #     ((118.0, 193.0), (96.0, 141.0)),
#     #     ((23.0, 102.0), (163.0, 180.0)),
#     #     ((156.0, 51.0), (118.0, 153.0)),
#     #     ((126.0, 86.0), (134.0, 38.0)),
#     #     ((147.0, 85.0), (144.0, 189.0)),
#     #     ((161.0, 169.0), (147.0, 93.0)),
#     # ]
#     start = time.perf_counter_ns()
#     res = all_intersections_bbox(shapes)
#     end = time.perf_counter_ns()
#     print(
#         f"all_interscetions_bbox {n} segments took: {end - start:.5f} seconds. Found {len(res)} intersections."
#     )

#     # poly = Poly(shapes)

#     start = time.perf_counter_ns()
#     res = all_intersections_vectorized(shapes)

#     end = time.perf_counter_ns()
#     print(
#         f"vectorized {n} segments took: {end - start:.5f} seconds. Found {len(res)} intersections."
#     )

#     start = time.perf_counter_ns()
#     res = all_intersections(shapes)

#     end = time.perf_counter_ns()
#     print(
#         f"all_intersections {n} segments took: {end - start:.5f} seconds. Found {len(res)} intersections."
#     )


# canvas = sg.Canvas()

# for shape in shapes:
#     canvas.draw(sg.Shape(shape), fill=False)

# for p, ids in res:
#     canvas.circle(2, p)

# canvas.save("c:/tmp/intersection_check.svg", overwrite=True)


# def segments_intersections_to_list(arr, eps=1e-12):
#     """
#     arr shape (N,10):
#       [x1,y1,x2,y2, x3,y3,x4,y4, id1,id2]
#     Returns:
#       out: list of ((x,y), (id1,id2)) for each intersecting row pair.
#     Notes:
#       - Intersection includes endpoints (touching counts).
#       - Collinear overlap: returns a representative point (one point per pair).
#     """
#     N = arr.shape[0]
#     a = arr[:, 0:4].astype(float)
#     b = arr[:, 4:8].astype(float)

#     x1, y1, x2, y2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
#     x3, y3, x4, y4 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

#     # IDs
#     id1 = arr[:, 8].astype(np.int64)
#     id2 = arr[:, 9].astype(np.int64)

#     r_x = x2 - x1
#     r_y = y2 - y1
#     s_x = x4 - x3
#     s_y = y4 - y3

#     def cross(ax, ay, bx, by):
#         return ax * by - ay * bx

#     rxs = cross(r_x, r_y, s_x, s_y)
#     q_p_x = x3 - x1
#     q_p_y = y3 - y1
#     qpxr = cross(q_p_x, q_p_y, r_x, r_y)

#     intersects = np.zeros(N, dtype=bool)
#     kind = np.zeros(N, dtype=np.int8)  # 0 none, 1 point, 2 overlap

#     # Helper: point on segment (inclusive)
#     def on_segment(px, py, ax, ay, bx, by):
#         return (
#             (px >= np.minimum(ax, bx) - eps)
#             & (px <= np.maximum(ax, bx) + eps)
#             & (py >= np.minimum(ay, by) - eps)
#             & (py <= np.maximum(ay, by) + eps)
#         )

#     collinear = (np.abs(rxs) <= eps) & (np.abs(qpxr) <= eps)
#     parallel = (np.abs(rxs) <= eps) & (~collinear)
#     general = ~parallel & ~collinear

#     # General intersection (single point)
#     t = np.full(N, np.nan, dtype=float)
#     u = np.full(N, np.nan, dtype=float)
#     t[general] = (
#         cross(q_p_x[general], q_p_y[general], s_x[general], s_y[general])
#         / rxs[general]
#     )
#     u[general] = qpxr[general] / rxs[general]

#     gen_hit = (
#         general & (t >= -eps) & (t <= 1 + eps) & (u >= -eps) & (u <= 1 + eps)
#     )

#     intersects[gen_hit] = True
#     kind[gen_hit] = 1

#     inter_x = np.full(N, np.nan, dtype=float)
#     inter_y = np.full(N, np.nan, dtype=float)
#     inter_x[gen_hit] = x1[gen_hit] + t[gen_hit] * r_x[gen_hit]
#     inter_y[gen_hit] = y1[gen_hit] + t[gen_hit] * r_y[gen_hit]

#     # Collinear overlap detection + representative point
#     overlap = np.zeros(N, dtype=bool)
#     if N:
#         use_x = np.abs(r_x) >= np.abs(r_y)  # per-row choice
#         # Overlap in x-projection
#         lo1 = np.minimum(x1, x2)
#         hi1 = np.maximum(x1, x2)
#         lo2 = np.minimum(x3, x4)
#         hi2 = np.maximum(x3, x4)
#         overlap_x = (hi1 >= lo2 - eps) & (hi2 >= lo1 - eps)

#         # Overlap in y-projection
#         lo1y = np.minimum(y1, y2)
#         hi1y = np.maximum(y1, y2)
#         lo2y = np.minimum(y3, y4)
#         hi2y = np.maximum(y3, y4)
#         overlap_y = (hi1y >= lo2y - eps) & (hi2y >= lo1y - eps)

#         overlap = collinear & np.where(use_x, overlap_x, overlap_y)

#     intersects[overlap] = True
#     kind[overlap] = 2

#     # For overlap: pick a representative point = first endpoint found on the other segment
#     rep_x = inter_x.copy()
#     rep_y = inter_y.copy()

#     end1_x, end1_y = x1, y1
#     end2_x, end2_y = x2, y2
#     end3_x, end3_y = x3, y3
#     end4_x, end4_y = x4, y4

#     on3 = on_segment(end1_x, end1_y, x3, y3, x4, y4)
#     on4 = on_segment(end2_x, end2_y, x3, y3, x4, y4)
#     on1 = on_segment(end3_x, end3_y, x1, y1, x2, y2)
#     on2 = on_segment(end4_x, end4_y, x1, y1, x2, y2)

#     use_end1 = overlap & on3
#     use_end2 = overlap & (~use_end1) & on4

#     use_end3 = overlap & (~use_end1) & (~use_end2) & on1
#     use_end4 = overlap & (~use_end1) & (~use_end2) & (~use_end3) & on2

#     rep_x[use_end1] = end1_x[use_end1]
#     rep_y[use_end1] = end1_y[use_end1]
#     rep_x[use_end2] = end2_x[use_end2]
#     rep_y[use_end2] = end2_y[use_end2]
#     rep_x[use_end3] = end3_x[use_end3]
#     rep_y[use_end3] = end3_y[use_end3]
#     rep_x[use_end4] = end4_x[use_end4]
#     rep_y[use_end4] = end4_y[use_end4]

#     # Final list
#     hit_idx = np.nonzero(intersects)[0]
#     out = [
#         ((float(rep_x[i]), float(rep_y[i])), (int(id1[i]), int(id2[i])))
#         for i in hit_idx
#     ]
#     return out


def segments_intersections_to_list(arr, eps=1e-12):
    """
    arr shape (N,10):
      [x1,y1,x2,y2, x3,y3,x4,y4, id1,id2]
    Returns:
      out: list of ((x,y), (id1,id2)) for each intersecting row pair.
    Notes:
      - Intersection includes endpoints (touching counts).
      - Collinear overlap: returns a representative point (one point per pair).
    """
    arr = np.asarray(arr)
    N = arr.shape[0]
    if N == 0:
        return []

    a = arr[:, 0:4].astype(float)
    b = arr[:, 4:8].astype(float)

    x1, y1, x2, y2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x3, y3, x4, y4 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    id1 = arr[:, 8].astype(np.int64)
    id2 = arr[:, 9].astype(np.int64)

    # Direction vectors
    r_x = x2 - x1
    r_y = y2 - y1
    s_x = x4 - x3
    s_y = y4 - y3

    def cross(ax, ay, bx, by):
        return ax * by - ay * bx

    rxs = cross(r_x, r_y, s_x, s_y)  # (r x s)
    q_p_x = x3 - x1
    q_p_y = y3 - y1
    qpxr = cross(q_p_x, q_p_y, r_x, r_y)  # (q - p) x r

    # Classification
    collinear = (np.abs(rxs) <= eps) & (np.abs(qpxr) <= eps)
    parallel = (np.abs(rxs) <= eps) & (~collinear)
    general = ~(parallel | collinear)

    # Outputs
    intersects = np.zeros(N, dtype=bool)
    kind = np.zeros(N, dtype=np.int8)  # 0 none, 1 point, 2 overlap

    inter_x = np.full(N, np.nan, dtype=float)
    inter_y = np.full(N, np.nan, dtype=float)

    # --- General (non-parallel & non-collinear) single-point intersection ---
    t = np.full(N, np.nan, dtype=float)
    u = np.full(N, np.nan, dtype=float)

    idx = general
    if np.any(idx):
        denom = rxs[idx]
        t[idx] = cross(q_p_x[idx], q_p_y[idx], s_x[idx], s_y[idx]) / denom
        u[idx] = qpxr[idx] / denom

        gen_hit = (
            (t[idx] >= -eps)
            & (t[idx] <= 1 + eps)
            & (u[idx] >= -eps)
            & (u[idx] <= 1 + eps)
        )

        hit_rows = np.nonzero(idx)[0][gen_hit]
        if hit_rows.size:
            intersects[hit_rows] = True
            kind[hit_rows] = 1
            inter_x[hit_rows] = x1[hit_rows] + t[hit_rows] * r_x[hit_rows]
            inter_y[hit_rows] = y1[hit_rows] + t[hit_rows] * r_y[hit_rows]

    # --- Collinear overlap detection + representative point ---
    # Overlap in x-projection or y-projection, depending on which axis is "better"
    use_x = np.abs(r_x) >= np.abs(r_y)

    lo1 = np.minimum(x1, x2)
    hi1 = np.maximum(x1, x2)
    lo2 = np.minimum(x3, x4)
    hi2 = np.maximum(x3, x4)
    overlap_x = (hi1 >= lo2 - eps) & (hi2 >= lo1 - eps)

    lo1y = np.minimum(y1, y2)
    hi1y = np.maximum(y1, y2)
    lo2y = np.minimum(y3, y4)
    hi2y = np.maximum(y3, y4)
    overlap_y = (hi1y >= lo2y - eps) & (hi2y >= lo1y - eps)

    overlap = collinear & np.where(use_x, overlap_x, overlap_y)
    if np.any(overlap):
        intersects[overlap] = True
        kind[overlap] = 2

    # Representative point: first endpoint found on the other segment
    # Helper: point on segment (inclusive, eps)
    def on_segment(px, py, ax, ay, bx, by):
        return (
            (px >= np.minimum(ax, bx) - eps)
            & (px <= np.maximum(ax, bx) + eps)
            & (py >= np.minimum(ay, by) - eps)
            & (py <= np.maximum(ay, by) + eps)
        )

    rep_x = inter_x.copy()
    rep_y = inter_y.copy()

    on3 = on_segment(x1, y1, x3, y3, x4, y4)
    on4 = on_segment(x2, y2, x3, y3, x4, y4)
    on1 = on_segment(x3, y3, x1, y1, x2, y2)
    on2 = on_segment(x4, y4, x1, y1, x2, y2)

    use_end1 = overlap & on3
    use_end2 = overlap & (~use_end1) & on4
    use_end3 = overlap & (~use_end1) & (~use_end2) & on1
    use_end4 = overlap & (~use_end1) & (~use_end2) & (~use_end3) & on2

    rep_x[use_end1], rep_y[use_end1] = x1[use_end1], y1[use_end1]
    rep_x[use_end2], rep_y[use_end2] = x2[use_end2], y2[use_end2]
    rep_x[use_end3], rep_y[use_end3] = x3[use_end3], y3[use_end3]
    rep_x[use_end4], rep_y[use_end4] = x4[use_end4], y4[use_end4]

    hit_idx = np.nonzero(intersects)[0]
    return [
        ((float(rep_x[i]), float(rep_y[i])), (int(id1[i]), int(id2[i])))
        for i in hit_idx
    ]


def tests():
    # for n in [100, 200, 500, 1000, 2000, 3000, 4000]:
    for n in [10, 20, 40, 80, 160, 320, 640, 1280, 2560]:
        edges = []
        count = 0
        while count < n:
            p1 = rp()
            p2 = rp()
            edge = (p1, p2)
            if simetri.geometry.points.point_utils.distance(p1, p2) < 40:
                edges.append(edge)
                count += 1

        start = time.perf_counter_ns()
        candidates = get_candidates(edges)
        res = segments_intersections_to_list(candidates)

        end = time.perf_counter_ns()
        # print(
        #     f"segments_intersections_to_list {n} segments took: {end - start:.5f} seconds. Found {len(res)} intersections."
        # )
        start = time.perf_counter_ns()

        res = all_intersections(edges)

        end = time.perf_counter_ns()
        # print(
        #     f"all_intersections {n} segments took: {end - start:.5f} seconds. Found {len(res)} intersections."
        # )

        start = time.perf_counter_ns()

        res = all_intersections_vectorized(edges)

        end = time.perf_counter_ns()
        # print(
        #     f"all_intersections_vectorized {n} segments took: {end - start:.5f} seconds. Found {len(res)} intersections."
        # )
        print(n, f"{(end - start) / 1e6:.5f} milliseconds")

    # all_interscetions_bbox 100 segments took: 0.00192 seconds. Found 1266 intersections.
    # vectorized 100 segments took: 0.00092 seconds. Found 1266 intersections.
    # all_intersections 100 segments took: 0.00284 seconds. Found 1266 intersections.
    # all_interscetions_bbox 200 segments took: 0.00468 seconds. Found 4209 intersections.
    # vectorized 200 segments took: 0.02303 seconds. Found 4209 intersections.
    # all_intersections 200 segments took: 0.00650 seconds. Found 4209 intersections.
    # all_interscetions_bbox 500 segments took: 0.02634 seconds. Found 28250 intersections.
    # vectorized 500 segments took: 0.01525 seconds. Found 28250 intersections.
    # all_intersections 500 segments took: 0.02798 seconds. Found 28250 intersections.
    # all_interscetions_bbox 1000 segments took: 0.10495 seconds. Found 117860 intersections.
    # vectorized 1000 segments took: 0.06965 seconds. Found 117860 intersections.
    # all_intersections 1000 segments took: 0.09462 seconds. Found 117860 intersections.
    # all_interscetions_bbox 2000 segments took: 0.42505 seconds. Found 460590 intersections.
    # vectorized 2000 segments took: 0.29882 seconds. Found 460590 intersections.
    # all_intersections 2000 segments took: 0.37435 seconds. Found 460590 intersections.
    # all_interscetions_bbox 3000 segments took: 0.97453 seconds. Found 1014286 intersections.
    # vectorized 3000 segments took: 0.72007 seconds. Found 1014286 intersections.
    # all_intersections 3000 segments took: 0.82455 seconds. Found 1014286 intersections.


import matplotlib.pyplot as plt
from SweepIntersectorLib.SweepIntersector import SweepIntersector


# segList = []

# # create some random segments
# for i in range(50):
#     vs = (random.uniform(-1,1),random.uniform(-1,1))
#     ve = (random.uniform(-1,1),random.uniform(-1,1))
#     segList.append( (vs,ve) )

# # add some vertical segments
# for i in range(5):
#     vs = (random.uniform(-1,1),random.uniform(-1,1))
#     ve = (vs[0],random.uniform(-1,1))
#     segList.append( (vs,ve) )

# # compute intersections
# isector = SweepIntersector()
# isecDic = isector.findIntersections(segList)

# # plot original segments
# for seg in segList:
#     vs,ve = seg
#     plt.plot([vs[0],ve[0]],[vs[1],ve[1]],'k:')

# # plot intersection points
# for seg,isects in isecDic.items():
#     for p in isects[1:-1]:
#         plt.plot(p[0],p[1],'r.')


# plt.gca().axis('equal')
# plt.show()
def test2():
    n = 1000
    edges = []
    for i in range(n):
        edges.append((rp(), rp()))

    start = time.perf_counter_ns()
    # compute intersections
    isector = SweepIntersector()
    isecDic = isector.findIntersections(edges)
    end = time.perf_counter_ns()
    print(
        f"SweepIntersector {n} segments took: {end - start:.5f} seconds. Found {len(isecDic)} intersections."
    )

    start = time.perf_counter_ns()

    res = all_intersections_vectorized(edges)

    end = time.perf_counter_ns()
    print(
        f"all_intersections_vectorized {n} segments took: {end - start:.5f} seconds. Found {len(res)} intersections."
    )

    from linesegmentintersections import bentley_ottman

    start = time.perf_counter_ns()
    try:
        intersections = bentley_ottman(edges)
        end = time.perf_counter_ns()
        print(
            f"line_segment_intersections {n} segments took: {end - start:.5f} seconds. Found {len(isecDic)} intersections."
        )
    except:
        print("Nope")


import bisect


class Segment:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2


def intersect(s1, s2):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(
            p[1], r[1]
        ) <= q[1] <= max(p[1], r[1])

    p1, q1 = (s1.x1, s1.y1), (s1.x2, s1.y2)
    p2, q2 = (s2.x1, s2.y1), (s2.x2, s2.y2)

    o1, o2, o3, o4 = (
        orientation(p1, q1, p2),
        orientation(p1, q1, q2),
        orientation(p2, q2, p1),
        orientation(p2, q2, q1),
    )

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    return False


def line_sweep(segments):
    events = []
    for s in segments:
        events.append((min(s.x1, s.x2), "L", s))
        events.append((max(s.x1, s.x2), "R", s))

    events.sort(key=lambda e: e[0])

    active = []
    intersections = []
    points = []

    for x, typ, seg in events:
        y_avg = (seg.y1 + seg.y2) / 2
        seg_id = id(seg)

        if typ == "L":
            bisect.insort(active, (y_avg, seg_id, seg))
            idx = active.index((y_avg, seg_id, seg))

            if idx > 0 and intersect(seg, active[idx - 1][2]):
                intersections.append((seg, active[idx - 1][2]))
            if idx < len(active) - 1 and intersect(seg, active[idx + 1][2]):
                intersections.append((seg, active[idx + 1][2]))
        else:
            idx = active.index((y_avg, seg_id, seg))
            above = active[idx - 1][2] if idx > 0 else None
            below = active[idx + 1][2] if idx < len(active) - 1 else None
            active.pop(idx)

            if above and below and intersect(above, below):
                intersections.append((above, below))
    # FB
    for s1, s2 in intersections:
        ip = simetri.geometry.segments.line_utils.intersect2(
            s1.x1, s1.y1, s1.x2, s1.y2, s2.x1, s2.y1, s2.x2, s2.y2
        )
        points.append(ip)
    # FB
    return points, intersections


def bharadwaj():

    n = 100
    segments = []
    edges = []
    for i in range(n):
        p1 = rp()
        p2 = rp()
        segments.append(Segment(*p1, *p2))
        edges.append((p1, p2))

    start = time.perf_counter_ns()
    # segments = [Segment(1, 1, 4, 4), Segment(1, 4, 4, 1), Segment(5, 2, 7, 2)]

    result = line_sweep(segments)
    end = time.perf_counter_ns()
    bharadwaj_time = end - start
    print(
        f"line_sweep {n} segments took: {end - start:.5f} seconds. Found {len(result)} intersections."
    )

    start = time.perf_counter_ns()

    res = all_intersections_vectorized(edges)

    end = time.perf_counter_ns()
    vector_time = end - start
    print(
        f"all_intersections_vectorized {n} segments took: {end - start:.5f} seconds. Found {len(res)} intersections."
    )

    print(f"vector/bharadwaj: {vector_time / bharadwaj_time}")

    # plot original segments
    for edge in edges:
        vs, ve = edge
        plt.plot([vs[0], ve[0]], [vs[1], ve[1]], "k:")

    # plot intersection points
    for p, ids in res:
        plt.plot(p[0], p[1], "r.")

    print(f"bharadwaj found {len(result)} intersections")
    # for s1, s2 in result:
    #     print(
    #         f"Segment ({s1.x1},{s1.y1})-({s1.x2},{s1.y2}) intersects with ({s2.x1},{s2.y1})-({s2.x2},{s2.y2})"
    #     )

    plt.gca().axis("equal")
    plt.show()


# bharadwaj()


def oparin():
    # from https://every-algorithm.github.io/2024/10/23/sweep_line_algorithm.html
    # byEugen Sławomir Oparin
    # Sweep Line Algorithm for detecting any intersection among line segments
    # The algorithm sweeps a vertical line from left to right, maintaining an
    # active set of segments ordered by their y-coordinate at the sweep line.

    import bisect

    class Segment:
        def __init__(self, p1, p2):
            # Ensure p1 is the left endpoint
            if p1[0] > p2[0] or (p1[0] == p2[0] and p1[1] > p2[1]):
                p1, p2 = p2, p1
            self.p1 = p1  # left endpoint
            self.p2 = p2  # right endpoint
            self.index = None  # will be set later

        def y_at(self, x):
            # Compute y coordinate of the segment at given x
            if self.p1[0] == self.p2[0]:
                # Vertical segment
                return self.p1[1]
            slope = (self.p2[1] - self.p1[1]) / (self.p2[0] - self.p1[0])
            return self.p1[1] + slope * (x - self.p1[0])

    def segments_intersect(a, b):
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        p1, p2 = a.p1, a.p2
        p3, p4 = b.p1, b.p2
        if max(p1[0], p2[0]) < min(p3[0], p4[0]) or max(p3[0], p4[0]) < min(
            p1[0], p2[0]
        ):
            return False
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
            (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
        ):
            return True
        if d1 == 0 and on_segment(p3, p4, p1):
            return True
        if d2 == 0 and on_segment(p3, p4, p2):
            return True
        if d3 == 0 and on_segment(p1, p2, p3):
            return True
        if d4 == 0 and on_segment(p1, p2, p4):
            return True
        return False

    def on_segment(a, b, p):
        return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(
            a[1], b[1]
        ) <= p[1] <= max(a[1], b[1])

    def sweep_line_intersections(segments):
        for i, seg in enumerate(segments):
            seg.index = i

        events = []
        for seg in segments:
            events.append((seg.p1[0], 0, seg))  # left endpoint
            events.append((seg.p2[0], 1, seg))  # right endpoint
        events.sort(key=lambda e: (e[0], e[1]))

        active = []  # list of (y, segment) tuples sorted by y

        def active_key(seg, x):
            return seg.y_at(x)

        for x, typ, seg in events:
            if typ == 0:  # left endpoint, insert
                y = seg.y_at(x)
                pos = bisect.bisect_left(active, (y, seg))
                if pos > 0 and segments_intersect(active[pos - 1][1], seg):
                    return True
                if pos < len(active) and segments_intersect(
                    active[pos][1], seg
                ):
                    return True
                active.insert(pos, (y, seg))
            else:  # right endpoint, remove
                y = seg.y_at(x)
                pos = bisect.bisect_left(active, (y, seg))
                if pos < len(active) and active[pos][1] == seg:
                    # Check neighbors after removal
                    prev_seg = active[pos - 1][1] if pos - 1 >= 0 else None
                    next_seg = (
                        active[pos + 1][1] if pos + 1 < len(active) else None
                    )
                    if (
                        prev_seg
                        and next_seg
                        and segments_intersect(prev_seg, next_seg)
                    ):
                        return True
                    active.pop(pos)
        return False

    # Example usage:
    # segs = [Segment((0,0),(3,3)), Segment((1,0),(1,4)), Segment((2,2),(5,2))]
    n = 100
    segments = []
    edges = []
    for i in range(n):
        p1 = rp()
        p2 = rp()
        segments.append(Segment(p1, p2))
        edges.append((p1, p2))

    start = time.perf_counter_ns()
    # segments = [Segment(1, 1, 4, 4), Segment(1, 4, 4, 1), Segment(5, 2, 7, 2)]

    result = sweep_line_intersections(segments)
    end = time.perf_counter_ns()
    bharadwaj_time = end - start
    print(
        f"oparin {n} segments took: {end - start:.5f} seconds. Found {len(result)} intersections."
    )


# oparin()
tests()


import numpy as np


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

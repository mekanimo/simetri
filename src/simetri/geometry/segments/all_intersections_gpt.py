"""AI version of all_ips_edges function in all_intersections.py"""

import time
import random
import numpy as np
import simetri.graphics as sg


def all_ips_edges(shapes):
    """Return all proper intersections between the edges of ``shapes``."""
    edge_coordinates = []
    for shape in shapes:
        for edge in shape.edges:
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

    return intersections


#####################
# Time Test
#####################


def rp():
    return (float(random.randint(1, 100)), float(random.randint(1, 100)))


for n in [100, 200, 500, 1000, 2000, 3000, 4000, 5000]:
    shapes = []
    for i in range(n):
        verts = []
        for j in range(10):
            verts.append(rp())
        shapes.append(sg.Shape(verts, closed=True))
    start = time.perf_counter()
    res = all_ips_edges(shapes)

    end = time.perf_counter()
    print(
        f"New algorithm. {n} segments took: {end - start:.5f} seconds. Found {len(res)} intersections."
    )


# Results:

# 2 polygons took: 0.00065 seconds. Found 59 intersections.
# 10 polygons took: 0.00340 seconds. Found 1112 intersections.
# 20 polygons took: 0.00814 seconds. Found 4490 intersections.
# 50 polygons took: 0.05054 seconds. Found 30803 intersections.
# 100 polygons took: 0.10811 seconds. Found 116456 intersections.
# 200 polygons took: 0.36611 seconds. Found 451960 intersections.
# 300 polygons took: 0.84888 seconds. Found 1068701 intersections.
# 400 polygons took: 1.45236 seconds. Found 1832816 intersections.



How can we make this shorter?

fill = not cur_part.fill
if part1 == cur_part:
    part2.fill = fill
else:
    part1.fill = fill
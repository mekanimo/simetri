"""AI version of all_ips_edges function in all_intersections.py"""

import time
import random
import numpy as np
import simetri.graphics as sg
from shapely.geometry import LineString
from shapely.strtree import STRtree


def rp(min_val=10, max_val=200):
    return (
        float(random.randint(min_val, max_val)),
        float(random.randint(min_val, max_val)),
    )


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
        candidate_indices = np.arange(candidate_start, edge_count)
        candidate_mask = (
            (edge_min_x[candidate_start:] <= edge_max_x[edge_index])
            & (edge_max_x[candidate_start:] >= edge_min_x[edge_index])
            & (edge_min_y[candidate_start:] <= edge_max_y[edge_index])
            & (edge_max_y[candidate_start:] >= edge_min_y[edge_index])
        )
        candidate_indices = candidate_indices[candidate_mask]
        candidate_edges = edge_array[candidate_indices]
        first_edge = edge_array[edge_index]
        first_delta_x = first_edge[2] - first_edge[0]
        first_delta_y = first_edge[3] - first_edge[1]
        second_delta_x = candidate_edges[:, 2] - candidate_edges[:, 0]
        second_delta_y = candidate_edges[:, 3] - candidate_edges[:, 1]
        denominator = (
            second_delta_y * first_delta_x - second_delta_x * first_delta_y
        )
        parallel_mask = np.abs(denominator) <= np.maximum(
            absolute_tolerance, relative_tolerance * np.abs(denominator)
        )
        valid_indices = np.flatnonzero(~parallel_mask)
        valid_denominator = denominator[valid_indices]
        valid_edges = candidate_edges[valid_indices]
        first_to_second_x = first_edge[0] - valid_edges[:, 0]
        first_to_second_y = first_edge[1] - valid_edges[:, 1]
        first_parameter = (
            (valid_edges[:, 2] - valid_edges[:, 0]) * first_to_second_y
            - (valid_edges[:, 3] - valid_edges[:, 1]) * first_to_second_x
        ) / valid_denominator
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
        intersecting_indices = valid_indices[intersecting_mask]
        intersection_points = np.column_stack(
            (
                first_edge[0]
                + first_parameter[intersecting_mask] * first_delta_x,
                first_edge[1]
                + first_parameter[intersecting_mask] * first_delta_y,
            )
        )
        intersections.extend(
            [
                (
                    tuple(point),
                    (
                        int(edge_ids[edge_index]),
                        int(edge_ids[candidate_indices[candidate_index]]),
                    ),
                )
                for point, candidate_index in zip(
                    intersection_points, intersecting_indices
                )
            ]
        )

    return intersections


#####################
# Time Test
#####################


def time_test():
    def rp():
        return (float(random.randint(1, 100)), float(random.randint(1, 100)))

    for n in [2, 10, 20, 50, 100, 200, 300, 400]:
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
            f"{n} polygons took: {end - start:.5f} seconds. Found {len(res)} intersections."
        )


# Results:

# 2 polygons took: 0.00072 seconds. Found 56 intersections.
# 10 polygons took: 0.00360 seconds. Found 1267 intersections.
# 20 polygons took: 0.00770 seconds. Found 3977 intersections.
# 50 polygons took: 0.05076 seconds. Found 27146 intersections.
# 100 polygons took: 0.10725 seconds. Found 124809 intersections.
# 200 polygons took: 0.39382 seconds. Found 463655 intersections.
# 300 polygons took: 0.87212 seconds. Found 1060450 intersections.
# 400 polygons took: 1.53546 seconds. Found 1895913 intersections.


# Shapely implementation


def shapely_ips_edges(shapes):
    edge_geometries = []
    for shape in shapes:
        for edge in shape.edges:
            start_point, end_point = edge
            edge_geometries.append(LineString([start_point[:2], end_point[:2]]))

    edge_tree = STRtree(edge_geometries)
    intersections = []
    for first_index, first_edge in enumerate(edge_geometries):
        for second_index in edge_tree.query(first_edge, predicate="intersects"):
            second_index = int(second_index)
            if second_index <= first_index:
                continue
            intersection = first_edge.intersection(
                edge_geometries[second_index]
            )
            if intersection.geom_type == "Point":
                intersections.append(
                    (
                        (intersection.x, intersection.y),
                        (first_index, second_index),
                    )
                )

    return intersections


def shapely_test():

    print("\nShapely STRtree comparison")
    random.seed(41)
    for polygon_count in [100, 200, 400, 500, 1000, 2000]:
        shapes = []
        for _ in range(polygon_count):
            vertices = []
            for _ in range(10):
                vertices.append(rp())
            shapes.append(sg.Shape(vertices, closed=True))

        start = time.perf_counter()
        numpy_result = all_ips_edges(shapes)
        numpy_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        shapely_result = shapely_ips_edges(shapes)
        shapely_elapsed = time.perf_counter() - start

        print(
            f"{polygon_count} polygons: "
            f"GPT2 {numpy_elapsed:.5f}s/{len(numpy_result)}; "
            f"Shapely {shapely_elapsed:.5f}s/{len(shapely_result)}"
        )


shapely_test()
# Results

# Shapely STRtree comparison
# 2 polygons: GPT2 0.00101s/59; Shapely 0.00138s/59
# 10 polygons: GPT2 0.00380s/1212; Shapely 0.01792s/1212
# 20 polygons: GPT2 0.00751s/4236; Shapely 0.06100s/4236
# 50 polygons: GPT2 0.03009s/29279; Shapely 0.42837s/29280
# 100 polygons: GPT2 0.15457s/113268; Shapely 1.59728s/113268
# 200 polygons: GPT2 0.42762s/461764; Shapely 6.71362s/461764
# 300 polygons: GPT2 0.93458s/1058654; Shapely 15.15038s/1058654
# 400 polygons: GPT2 1.74904s/1916238; Shapely 27.64100s/1916242


import heapq
from sortedcontainers import SortedList

# Define event types
START = 0
END = 1


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __lt__(self, other):
        if self.x != other.x:
            return self.x < other.x
        return self.y < other.y

    def __repr__(self):
        return f"({self.x}, {self.y})"


class Segment:
    def __init__(self, id, p1, p2):
        self.id = id
        # Ensure p1 is always the left endpoint
        if p1.x < p2.x or (p1.x == p2.x and p1.y < p2.y):
            self.p1 = p1
            self.p2 = p2
        else:
            self.p1 = p2
            self.p2 = p1

    def get_y(self, x):
        """Calculates the exact y-coordinate of the segment at a given x position."""
        if self.p1.x == self.p2.x:
            return self.p1.y
        # Linear interpolation: y = y1 + (x - x1) * slope
        slope = (self.p2.y - self.p1.y) / (self.p2.x - self.p1.x)
        return self.p1.y + (x - self.p1.x) * slope

    def __repr__(self):
        return f"Seg_{self.id}[{self.p1} -> {self.p2}]"


class Event:
    def __init__(self, x, type, segment):
        self.x = x
        self.type = type
        self.segment = segment

    def __lt__(self, other):
        # Sort primarily by X coordinate
        if self.x != other.x:
            return self.x < other.x
        # If X is identical, process START events before END events
        return self.type < other.type


def calculate_intersection(seg1, seg2):
    """Computes the intersection point between two line segments, if it exists."""
    # Line 1: A1x + B1y = C1
    a1 = seg1.p2.y - seg1.p1.y
    b1 = seg1.p1.x - seg1.p2.x
    c1 = a1 * seg1.p1.x + b1 * seg1.p1.y

    # Line 2: A2x + B2y = C2
    a2 = seg2.p2.y - seg2.p1.y
    b2 = seg2.p1.x - seg2.p2.x
    c2 = a2 * seg2.p1.x + b2 * seg2.p1.y

    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        return None  # Parallel segments

    # Calculate exact point of intersection
    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant

    # Verify if the intersection falls within the bounding box of both segments
    if (
        min(seg1.p1.x, seg1.p2.x) <= x <= max(seg1.p1.x, seg1.p2.x)
        and min(seg1.p1.y, seg1.p2.y) <= y <= max(seg1.p1.y, seg1.p2.y)
        and min(seg2.p1.x, seg2.p2.x) <= x <= max(seg2.p1.x, seg2.p2.x)
        and min(seg2.p1.y, seg2.p2.y) <= y <= max(seg2.p1.y, seg2.p2.y)
    ):
        return Point(x, y)

    return None


def sweep_line_intersections(segments):
    event_queue = []
    intersections = []

    # 1. Populate the event queue with start and end points
    for seg in segments:
        heapq.heappush(event_queue, Event(seg.p1.x, START, seg))
        heapq.heappush(event_queue, Event(seg.p2.x, END, seg))

    # Dynamic key class wrapper to evaluate Y ordering relative to current Sweep X
    current_x = 0.0

    class SweepKey:
        def __init__(self, segment):
            self.segment = segment

        def __lt__(self, other):
            return self.segment.get_y(current_x) < other.segment.get_y(
                current_x
            )

    # The active structure containing segments cutting through the current vertical sweep line
    active_segments = SortedList(key=SweepKey)

    def check_and_add_intersection(s1, s2):
        if s1 and s2:
            pt = calculate_intersection(s1, s2)
            if (
                pt and pt.x >= current_x
            ):  # Only capture future or current intersections
                intersections.append((s1.id, s2.id, pt))

    # 2. Process all timeline events sequentially from left to right
    while event_queue:
        event = heapq.heappop(event_queue)
        current_x = event.x
        seg = event.segment
        active_segments = SortedList(active_segments, key=SweepKey)

        if event.type == START:
            active_segments.add(seg)
            for idx, active_segment in enumerate(active_segments):
                if active_segment is seg:
                    break
            else:
                raise ValueError(
                    f"Segment {seg!r} was not added to the active set"
                )

            # Check for intersections with immediate neighbors above and below
            above = (
                active_segments[idx + 1]
                if idx + 1 < len(active_segments)
                else None
            )
            below = active_segments[idx - 1] if idx - 1 >= 0 else None

            check_and_add_intersection(seg, above)
            check_and_add_intersection(seg, below)

        elif event.type == END:
            for idx, active_segment in enumerate(active_segments):
                if active_segment is seg:
                    break
            else:
                raise ValueError(f"Segment {seg!r} is not in the active set")

            if active_segment is seg:
                above = (
                    active_segments[idx + 1]
                    if idx + 1 < len(active_segments)
                    else None
                )
                below = active_segments[idx - 1] if idx - 1 >= 0 else None

                del active_segments[idx]

                # Check if the closing segment's neighbors now cross paths
                check_and_add_intersection(above, below)

    return intersections


# # --- Example Execution ---
# if __name__ == "__main__":
#     # Formulate a criss-crossing set of segments
#     raw_segments = [
#         Segment(1, Point(1, 2), Point(4, 4)),
#         Segment(2, Point(2, 5), Point(5, 1)),
#         Segment(3, Point(3, 1), Point(6, 5)),
#     ]

#     found_intersections = sweep_line_intersections(raw_segments)

#     print(f"Discovered {len(found_intersections)} intersection(s):")
#     for seg1_id, seg2_id, pt in found_intersections:
#         print(
#             f"Segment {seg1_id} intersects Segment {seg2_id} at coordinate {pt}"
#         )


def sweep_line_test():
    def rp(min_val=10, max_val=200):
        return (
            float(random.randint(min_val, max_val)),
            float(random.randint(min_val, max_val)),
        )

    for n in [100, 200, 500, 1000, 2000, 3000]:
        segments = []
        for i in range(n):
            segment = Segment(i, Point(*rp()), Point(*rp()))
            segments.append(segment)
            # verts = []
            # for j in range(10):
            #     verts.append(rp())
            # shapes.append(sg.Shape(verts, closed=True))
        start = time.perf_counter()
        res = sweep_line_intersections(segments)
        end = time.perf_counter()
        print(
            f"{n} polygons took: {end - start:.5f} seconds. Found {len(res)} intersections."
        )


# sweep_line_test()

# Results

# 2 segments took: 0.00054 seconds. Found 11 intersections.
# 10 segments took: 0.00705 seconds. Found 122 intersections.
# 20 segments took: 0.02183 seconds. Found 234 intersections.
# 50 segments took: 0.10463 seconds. Found 675 intersections.
# 100 segments took: 0.50193 seconds. Found 1538 intersections.
# 200 segments took: 2.25676 seconds. Found 3398 intersections.
# 300 segments took: 6.08330 seconds. Found 5291 intersections.
# 400 segments took: 12.29438 seconds. Found 7344 intersections.

# 100 segments took: 0.00821 seconds. Found 112 intersections.
# 200 segments took: 0.02706 seconds. Found 235 intersections.
# 500 segments took: 0.12024 seconds. Found 682 intersections.
# 1000 segments took: 0.52598 seconds. Found 1397 intersections.
# 2000 segments took: 2.39159 seconds. Found 2998 intersections.
# 3000 segments took: 5.81061 seconds. Found 4712 intersections.
# 4000 segments took: 12.07660 seconds. Found 6421 intersections.
# 5000 segments took: 19.82649 seconds. Found 8188 intersections.

import cProfile
import pstats
import time
from collections import defaultdict
from itertools import pairwise

import networkx as nx
import numpy as np

import simetri.geom.points.point_utils
import simetri.geom.polygons.polygon
import simetri.geom.segments.line_utils
import simetri.graphics as sg


def cross(ax, ay, bx, by):
    return ax * by - ay * bx


def orient3(a, b, c):
    # cross((b-a),(c-a))
    return cross(b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1])


def on_segment(a, b, p, eps=1e-12):
    # check collinear + within bbox
    if abs(orient3(a, b, p)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def get_time(start, end):
    ms = (end - start) / 1e6

    if ms < 900:
        elapsed = ms
        units = "milliseconds"
    else:
        elapsed = ms / 1000
        units = "seconds"

    return f"{elapsed:.2f} {units}"


def segment_cycles(segments, length_bound=10):
    """Given a list of line segments, returns all cycles."""
    coordinates = []
    for seg in segments:
        coordinates.extend(seg)

    d_node_coord, d_coord_node, _d_rounded_coord = (
        simetri.geom.polygons.polygon.node_dictionaries(
            coordinates, dist_tol=0.5
        )
    )
    g_segments = [[d_coord_node[coord] for coord in seg] for seg in segments]

    nx_graph = nx.Graph()
    nx_graph.update(g_segments)
    # cycles = get_cycles(g_segments)
    cycles = list(nx.simple_cycles(nx_graph, length_bound=length_bound))
    res = []
    for cycle in cycles:
        res.append([d_node_coord[node] for node in cycle])

    return res, cycles


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


def filter_by_edge(arr, edge_id):
    # i1 is column 1, i2 is column 2
    i1 = arr[:, 1]
    i2 = arr[:, 2]

    mask = (i1 == edge_id) | (i2 == edge_id)

    return arr[mask]


def to_numpy_array(ip_edge_list):
    rows = []
    for (x, y), (i1, i2) in ip_edge_list:
        rows.append([(x, y), i1, i2])
    return np.array(rows, dtype=object)


def segments_from_points(points):
    """Given a list of collinear points (in any order), returns the connected segments."""
    n = len(points)
    if n < 2:
        res = None
    elif n == 2:
        res = [tuple(points)]
    else:
        sorted_points = sorted(points, key=lambda p: (p[0], p[1]))
        segments = list(pairwise(sorted_points))
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
    start = time.perf_counter_ns()
    for edge, part in d_edge_part.items():
        if len(part) == 1:
            cur_part = sg.d_id_obj[next(iter(part))]
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

    print(
        f"Visited {count} partitions, processed {qcount} for symmetric difference coloring."
    )

    print(
        f"Coloring shapes for symmetric difference took {get_time(start, end)}."
    )


def get_partitions(shapes, length_bound=10):
    n_rows = sum([len(shp.vertices) for shp in shapes])
    intersections = simetri.geom.segments.line_utils.all_intersections(
        shapes.all_segments, return_points_list=True
    )  # ((x, y), (id1, id2))
    start = time.perf_counter_ns()
    res_array = to_numpy_array(intersections)  # ((x, y), edge1_id, edge2_id)
    # print('res_array', res_array)
    all_segments = []
    all_midpoints = []
    for i in range(n_rows):
        points = filter_by_edge(res_array, i)[:, 0]
        points = [
            simetri.geom.points.point_utils.round_point(p) for p in points
        ]
        if points:
            segments = segments_from_points(points)
            all_segments.extend(
                [
                    simetri.geom.segments.line_utils.round_segment(seg, 2)
                    for seg in segments
                ]
            )
            all_midpoints.extend(
                [
                    simetri.geom.points.point_utils.midpoint(*seg)
                    for seg in all_segments
                ]
            )

    all_midpoints = set(all_midpoints)

    segment_coordinates = []
    for seg in all_segments:
        segment_coordinates.extend(seg)

    c_start = time.perf_counter_ns()
    cycles, cycles_nodes = segment_cycles(
        all_segments, length_bound=length_bound
    )
    c_end = time.perf_counter_ns()

    print(
        f"Number of total cycles with less than {length_bound} nodes: {len(cycles)}"
    )
    cycles, cycles_nodes = zip(
        *sorted(zip(cycles, cycles_nodes), key=len)
    )  # sort cycles and cycles_nodes in sync
    # shp1 = shapes[0]
    # shp2 = shapes[1]
    # rest = shapes[2:]
    # union = sg.polygon_union(shp1, shp2).merge_shapes()[0]
    # for shape in rest:
    #     union = sg.polygon_union(union, shape).merge_shapes()[0]

    union, holes = sg.polygons_union(shapes)
    print(f"Number of holes: {len(holes)}")
    holes_area = sum(
        [
            simetri.geom.polygons.polygon.polygon_area(hole.vertices)
            for hole in holes
        ]
    )
    union_area = (
        simetri.geom.polygons.polygon.polygon_area(union.vertices)
        - holes_area
    )

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
            # check if partition is a hole
            is_hole = False
            for hole in holes:
                if getattr(hole, "processed", False):
                    continue
                if sg.equal_polygons(hole, partition):
                    is_hole = True
                    hole.processed = True
                    break
            if is_hole:
                continue
            partitions.append(partition)
            for edge in partition.edges:
                d_edge_partition[frozenset(edge)].add(partition.id)
            poly_area = abs(
                simetri.geom.polygons.polygon.polygon_area(poly)
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


def draw_symm_diff(shapes, length_bounds=10):
    canvas = sg.Canvas()
    partitions, union = symmetric_difference(shapes, length_bounds)
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
    canvas.save("c:/tmp/partition_test.svg", overwrite=True)


points = [(0, 0), (40, 0), (40, 40), (0, 40)]
square = sg.Shape(points, closed=True)
squares = square.translate(20, 30, reps=1).translate(50, 0, reps=4)
squares.translate(-10, 50, reps=4)


with cProfile.Profile() as pr:
    draw_symm_diff(squares, 13)


# # Format and print the results
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME).print_stats(20)

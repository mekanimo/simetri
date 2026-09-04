"""Core 2D geometry operations used across simetri.

Includes line clipping, intersections, polygon simplicity tests, trimming,
fillets, and related utilities. Many helpers are also re-exported via
``import simetri.graphics as sg``.
"""

# To do: Clean up this module.

from __future__ import annotations

import re
from collections.abc import Callable
from functools import cmp_to_key
from math import (
    acos,
    atan2,
    cos,
    exp,
    floor,
    hypot,
    isclose,
    pi,
    sin,
    sqrt,
    tan,
)
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import around, array, ndarray
from numpy.typing import NDArray

from ..geometry.geom_utils import (
    area,
    close_points2,
    collinear,
    connected_pairs,
    intersect,
    intersect2,
    intersection2,
    intersection3,
    line_angle,
    midpoint,
    positive_angle,
)
from ..graphics.affine import rotate_point
from ..graphics.all_enums import Connection, Types
from ..graphics.common import (
    LineType,
    PointType,
    VecType,
    get_defaults,
    i_vec,
    j_vec,
)
from ..helpers.utilities import (
    lerp,
    reg_poly_points,
)
from ..settings.settings import defaults
from .vectors import *

if TYPE_CHECKING:
    from ..graphics.shape import Shape

tau = 2 * pi  # 360 degrees


def clip_line_to_rect(point, direction, lower_left, upper_right):
    """Clip an infinite line against an axis-aligned rectangle.

    The line is defined by a point and a direction vector.

    Args:
        point: A point on the line ``(x, y)``.
        direction: Direction vector ``(dx, dy)``.
        lower_left: Rectangle lower-left corner.
        upper_right: Rectangle upper-right corner.

    Returns:
        tuple | None: Clipped segment ``((x1, y1), (x2, y2))``, or None if
        the line misses the rectangle.
    """
    x_min, y_min = lower_left[:2]
    x_max, y_max = upper_right[:2]
    rectangle = (x_min, y_min, x_max, y_max)
    t_min, t_max = -float("inf"), float("inf")

    # Iterate over x and y dimensions
    for i in range(2):
        if direction[i] == 0:
            if point[i] < rectangle[i] or point[i] > rectangle[i + 2]:
                return None
        else:
            t1 = (rectangle[i] - point[i]) / direction[i]
            t2 = (rectangle[i + 2] - point[i]) / direction[i]
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))

    if t_min <= t_max:
        start = (
            point[0] + t_min * direction[0],
            point[1] + t_min * direction[1],
        )
        end = (point[0] + t_max * direction[0], point[1] + t_max * direction[1])
        return (start, end)
    return None


def sine_wave(
    amplitude: float,
    frequency: float,
    duration: float,
    sample_rate: float,
    phase: float = 0,
) -> NDArray:
    """
    Generate a sine wave.

    Args:
        amplitude (float): Amplitude of the wave.
        frequency (float): Frequency of the wave.
        duration (float): Duration of the wave.
        sample_rate (float): Sample rate.
        phase (float, optional): Phase angle of the wave. Defaults to 0.

    Returns:
        np.ndarray: Time and signal arrays representing the sine wave.
    """
    time = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * time + phase)
    # plt.plot(time, signal)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Discretized Sine Wave')
    # plt.grid(True)
    # plt.show()
    return time, signal


def damping_function(amplitude, duration, sample_rate):
    """
    Generates a damping function based on the given amplitude, duration, and sample rate.

    Args:
        amplitude (float): The initial amplitude of the damping function.
        duration (float): The duration over which the damping occurs, in seconds.
        sample_rate (float): The number of samples per second.

    Returns:
        list: A list of float values representing the damping function over time.
    """
    damping = []
    for i in range(int(duration * sample_rate)):
        damping.append(amplitude * exp(-i / (duration * sample_rate)))
    return damping


def sine_points(
    period: float = 40,
    amplitude: float = 20,
    duration: float = 40,
    n_points: int = 100,
    phase_angle: float = 0,
    damping: float = 0,
) -> NDArray:
    """
    Generate sine wave points.

    Args:
        amplitude (float): Amplitude of the wave.
        frequency (float): Frequency of the wave.
        duration (float): Duration of the wave.
        sample_rate (float): Sample rate.
        phase (float, optional): Phase angle of the wave. Defaults to 0.
        damping (float, optional): Damping coefficient. Defaults to 0.
    Returns:
        np.ndarray: Array of points representing the sine wave.
    """
    phase = phase_angle
    freq = 1 / period
    n_cycles = duration / period
    x = np.linspace(0, duration, int(n_points * n_cycles))
    y = amplitude * np.sin(2 * np.pi * freq * x + phase)
    if damping:
        y *= np.exp(-damping * x)
    vertices = np.column_stack((x, y)).tolist()

    return vertices


def check_consecutive_duplicates(points, rel_tol=0, abs_tol=None) -> bool:
    """Check for consecutive duplicate points in a list of points.

    Args:
        points (list): List of points to check.
        rel_tol (float, optional): Relative tolerance. Defaults to 0.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        bool: True if consecutive duplicate points are found, False otherwise.
    """
    if abs_tol is None:
        abs_tol = defaults["abs_tol"]
    if isinstance(points, np.ndarray):
        points = points.tolist()
    if points and len(points) > 1:
        for i, pnt in enumerate(points[:-1]):
            next_pnt = points[i + 1]
            val1 = pnt[0] + pnt[1]
            val2 = next_pnt[0] + next_pnt[1]
            if isclose(val1, val2, rel_tol=0, abs_tol=abs_tol) and np.allclose(
                pnt, next_pnt, rtol=0, atol=abs_tol
            ):
                return True

    return False


def circle_inversion(point, center, radius):
    """
    Inverts a point with respect to a circle.

    Args:
        point (tuple): The point to invert, represented as a tuple (x, y).
        center (tuple): The center of the circle, represented as a tuple (x, y).
        radius (float): The radius of the circle.

    Returns:
        tuple: The inverted point, represented as a tuple (x, y).
    """
    x, y = point[:2]
    cx, cy = center[:2]
    # Calculate the distance from the point to the center of the circle
    dist = sqrt((x - cx) ** 2 + (y - cy) ** 2)
    # If the point is at the center of the circle, return the point at infinity
    if dist == 0:
        return float("inf"), float("inf")
    # Calculate the distance from the inverted point to the center of the circle
    inv_dist = radius**2 / dist
    # Calculate the inverted point
    inv_x = cx + inv_dist * (x - cx) / dist
    inv_y = cy + inv_dist * (y - cy) / dist
    return inv_x, inv_y


def sorted_edges(polygon):
    """Return polygon edges sorted for sweep-line processing.

    Edges are oriented left-to-right, then sorted by increasing start ``x``
    and ``y``. Used by simplicity tests.

    Args:
        polygon: Sequence of polygon vertices.

    Returns:
        list: Oriented and sorted edges ``[(p1, p2), ...]``.
    """
    # order the edges:increasing x coordinates then increasing y coordinates for the start points
    # this is used for line sweep algorithm to check if the polygon is simple
    def get_edges(polygon):
        edges = []
        for i, p in enumerate(polygon[:-1]):
            np = polygon[i + 1]  # next point
            edges.append((p, np))
        p = polygon[-1]
        np = polygon[0]
        edges.append((p, np))
        return edges

    def compare_edges(edge1, edge2):
        x1 = edge1[0][0]
        x2 = edge2[0][0]
        y1 = edge1[0][1]
        y2 = edge2[0][1]
        if x1 < x2:
            return -1
        elif x1 > x2:
            return 1
        else:
            if y1 < y2:
                return -1
            elif y1 > y2:
                return 1
            else:
                return 0

    edges = get_edges(polygon)
    oriented_edges = []
    for edge in edges:
        if edge[0][0] > edge[1][0]:
            oriented_edges.append((edge[1], edge[0]))
        else:
            oriented_edges.append(edge)
    oriented_edges.sort(key=cmp_to_key(compare_edges))

    return oriented_edges


def equal_angles(
    angle1: float,
    angle2: float,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if two angles are equal within tolerance.

    Negative angles are converted to positive values before comparison.

    Args:
        angle1: First angle in radians.
        angle2: Second angle in radians.
        rel_tol: Relative tolerance. Defaults to ``defaults[\"rel_tol\"]``.
        abs_tol: Absolute tolerance. Defaults to ``defaults[\"abs_tol\"]``.

    Returns:
        bool: True if the angles match within tolerance.
    """
    if rel_tol is None:
        rel_tol = defaults["rel_tol"]

    if abs_tol is None:
        abs_tol = defaults["abs_tol"]

    angle1 = positive_angle(angle1)
    angle2 = positive_angle(angle2)

    return isclose(angle1, angle2, rel_tol=rel_tol, abs_tol=abs_tol)


def is_simple(polygon):
    """Return True if ``polygon`` is simple (non-self-intersecting).

    Uses a sweep-line style test. Polygons with more than 100 vertices
    are delegated to ``is_simple2``.

    Args:
        polygon: Ordered polygon vertices.

    Returns:
        bool: True if the polygon does not self-intersect.
    """
    if len(polygon) > 100:
        return is_simple2(polygon)

    queue = []
    points = []
    edges = sorted_edges(polygon)

    points = [tuple(p) for edge in edges for p in edge]
    points = list(set(points))
    points.sort()

    for p in points:
        for edge in edges:
            if p == tuple(edge[0]):
                for e in queue:
                    if e[0] in edge or e[1] in edge:
                        continue
                    res = intersection(e, edge)

                    if res == Connection.INTERSECT:
                        return False
                queue.append(edge)
            elif p == tuple(edge[1]):
                queue.remove(edge)

    return True


def is_simple2(
    polygon,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """
    Return True if the polygon is simple.

    Args:
        polygon (list): List of points representing the polygon.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        bool: True if the polygon is simple, False otherwise.
    """
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])

    if not close_points2(polygon[0], polygon[-1]):
        polygon.append(polygon[0])
    segments = [[polygon[i], polygon[i + 1]] for i in range(len(polygon) - 1)]

    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    segment_coords = []
    for segment in segments:
        segment_coords.append(
            [segment[0][0], segment[0][1], segment[1][0], segment[1][1]]
        )
    seg_arr = np.array(segment_coords)  # segments array
    n_rows = seg_arr.shape[0]
    xmin = np.minimum(seg_arr[:, 0], seg_arr[:, 2]).reshape(n_rows, 1)
    xmax = np.maximum(seg_arr[:, 0], seg_arr[:, 2]).reshape(n_rows, 1)
    ymin = np.minimum(seg_arr[:, 1], seg_arr[:, 3]).reshape(n_rows, 1)
    ymax = np.maximum(seg_arr[:, 1], seg_arr[:, 3]).reshape(n_rows, 1)
    id_ = np.arange(n_rows).reshape(n_rows, 1)
    seg_arr = np.concatenate((seg_arr, xmin, ymin, xmax, ymax, id_), 1)
    seg_arr = seg_arr[seg_arr[:, 4].argsort()]
    i_xmin, i_ymin, i_xmax, i_ymax, i_id = range(4, 9)  # column indices

    s_processed = set()  # set of processed segment pairs
    for i in range(n_rows):
        x1, y1, x2, y2, sl_xmin, sl_ymin, sl_xmax, sl_ymax, id1 = seg_arr[i, :]
        id1 = int(id1)
        segment = [x1, y1, x2, y2]
        start = i + 1  # keep pushing the sweep line forward
        candidates = seg_arr[start:, :][
            (
                (
                    (seg_arr[start:, i_xmax] >= sl_xmin)
                    & (seg_arr[start:, i_xmin] <= sl_xmax)
                )
                & (
                    (seg_arr[start:, i_ymax] >= sl_ymin)
                    & (seg_arr[start:, i_ymin] <= sl_ymax)
                )
            )
        ]
        for cand in candidates:
            id2 = int(cand[i_id])
            pair = frozenset((id1, id2))
            if pair in s_processed:
                continue
            s_processed.add(pair)
            seg2 = cand[:4]
            x1, y1, x2, y2 = segment
            x3, y3, x4, y4 = seg2
            res = intersection3(x1, y1, x2, y2, x3, y3, x4, y4)
            if res[0] == Connection.COLL_CHAIN:
                length1 = distance((x1, y1), (x2, y2))
                length2 = distance((x3, y3), (x4, y4))
                p1, p2 = res[1][0], res[1][2]
                chain_length = distance(p1, p2)
                if not isclose(
                    length1 + length2,
                    chain_length,
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                ):
                    return False
                else:
                    continue
            if res[0] in (Connection.CHAIN, Connection.PARALLEL):
                continue
            if res[0] != Connection.DISJOINT:
                return False

    return True


def all_segments_sorted(
    edges: list[tuple[PointType, PointType]],
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> list[tuple[PointType]]:
    """Split edges at intersection points, ordered along each edge.

    Intersection points on an edge are sorted by distance from the edge's
    first endpoint. The result is a flat list of consecutive sub-segments.

    Args:
        edges: Input line segments.
        rel_tol: Relative tolerance for intersections.
        abs_tol: Absolute tolerance for intersections.

    Returns:
        list: Sub-segments covering each input edge in order.
    """
    intersection_map, _ = all_intersections(edges, rel_tol, abs_tol)

    intersections_by_edge = {edge_id: [] for edge_id in range(len(edges))}
    for edge_id, intersections in intersection_map.items():
        intersections_by_edge[edge_id] = intersections

    sorted_segments = []
    for edge_id, edge in enumerate(edges):
        start_point, end_point = edge
        intersections = sorted(
            intersections_by_edge[edge_id],
            key=lambda item: distance(start_point, item[0]),
        )
        points_on_edge = []
        point_coordinates = {tuple(start_point[:2]), tuple(end_point[:2])}
        for point, _ in intersections:
            coordinates = tuple(point[:2])
            if coordinates not in point_coordinates:
                points_on_edge.append(point)
                point_coordinates.add(coordinates)
        edge_points = [start_point] + points_on_edge + [end_point]
        sorted_segments.extend(connected_pairs(edge_points))
        # sorted_segments.append(
        #     tuple([start_point] + points_on_edge + [end_point])

    return sorted_segments


def all_intersections(
    edges: list[tuple[PointType, PointType]],
    rel_tol: float | None = None,
    abs_tol: float | None = None,
    return_points_list: bool = False,
) -> tuple[dict, list[tuple]]:
    """Return all proper intersections between the given edges.

    Bounding-box candidates are collected into one NumPy array, then
    line-segment intersections are computed in bulk.

    Args:
        edges: Line segments to test.
        rel_tol: Relative tolerance. Defaults to settings default.
        abs_tol: Absolute tolerance. Defaults to settings default.
        return_points_list: If True, include a flat list of intersection
            points in the return value.

    Returns:
        tuple: ``(intersection_map, points)`` when ``return_points_list``
        is True; otherwise an intersection map keyed by edge id.
    """

    relative_tolerance, absolute_tolerance = get_defaults(
        ["rel_tol", "abs_tol"], [rel_tol, abs_tol]
    )
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
        if return_points_list:
            return []
        return {}, []

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

    if return_points_list:
        res = [
            ((x_coordinate, y_coordinate), (int(first_id), int(second_id)))
            for x_coordinate, y_coordinate, first_id, second_id in zip(
                intersection_x[intersecting_mask],
                intersection_y[intersecting_mask],
                intersection_rows[:, 8],
                intersection_rows[:, 9],
            )
        ]
    else:
        d_results = {}
        points = []
        for x_coordinate, y_coordinate, first_id, second_id in zip(
            intersection_x[intersecting_mask],
            intersection_y[intersecting_mask],
            intersection_rows[:, 8],
            intersection_rows[:, 9],
        ):
            point = (x_coordinate, y_coordinate)
            first_id = int(first_id)
            second_id = int(second_id)
            if first_id not in d_results:
                d_results[first_id] = []
            if second_id not in d_results:
                d_results[second_id] = []
            d_results[first_id].append((point, second_id))
            d_results[second_id].append((point, first_id))
            points.append(point)
        res = (d_results, points)

    return res


def triangle_centroid(p1, p2, p3):
    """Return the centroid of a triangle given its three vertices.

    Args:
        p1: First vertex ``(x, y)``.
        p2: Second vertex ``(x, y)``.
        p3: Third vertex ``(x, y)``.

    Returns:
        tuple: Centroid ``(cx, cy)``.
    """

    cx = (p1[0] + p2[0] + p3[0]) / 3
    cy = (p1[1] + p2[1] + p3[1]) / 3

    return (cx, cy)


def triangle_angles_from_sides(
    a: float, b: float, c: float
) -> tuple[float, float, float]:
    """
    Calculates the inner angles of a triangle given its side lengths.

    Args:
        a float: Length of side a.
        b float: Length of side b.
        c float: Length of side c.

    Returns:
        tuple[float, float, float]: A tuple containing the angles (A, B, C) in degrees.
    """
    a2 = a * a
    b2 = b * b
    c2 = c * c
    A = acos((b2 + c2 - a2) / (2 * b * c))
    B = acos((a2 + c2 - b2) / (2 * a * c))
    C = acos((a2 + b2 - c2) / (2 * a * b))

    return A, B, C


def angle_between_lines2(
    point1: PointType, point2: PointType, point3: PointType
) -> float:
    """
    Given line1 as point1 and point2, and line2 as point2 and point3
    return the angle between two lines
    (point2 is the corner point)

    Args:
        point1 (PointType): First point of the first line.
        point2 (PointType): Second point of the first line and first point of the second line.
        point3 (PointType): Second point of the second line.

    Returns:
        float: Signed angle between the two lines in radians (-π to π).
    """
    vec1 = v_from_points(point2, point1)
    vec2 = v_from_points(point2, point3)
    cross = v_cross(vec1, vec2)
    dot = v_mul(vec1, vec2)
    return atan2(cross, dot)


def close_angles(angle1: float, angle2: float, angtol=None) -> bool:
    """
    Return True if two angles are close to each other.

    Args:
        angle1 (float): First angle in radians.
        angle2 (float): Second angle in radians.
        angtol (float, optional): Angle tolerance. Defaults to None.

    Returns:
        bool: True if the angles are close to each other, False otherwise.
    """
    if angtol is None:
        angtol = defaults["angtol"]

    return (abs(angle1 - angle2) % (2 * pi)) < angtol


def connect2(
    poly_point1: list[PointType],
    poly_point2: list[PointType],
    dist_tol: float | None = None,
    rel_tol: float | None = None,
) -> list[PointType]:
    """
    Connect two polypoints together.

    Args:
        poly_point1 (list[PointType]): First list of points.
        poly_point2 (list[PointType]): Second list of points.
        dist_tol (float, optional): Distance tolerance. Defaults to None.
        rel_tol (float, optional): Relative tolerance. Defaults to None.

    Returns:
        list[PointType]: Connected list of points.
    """
    rel_tol, dist_tol = get_defaults(
        ["rel_tol", "dist_tol"], [rel_tol, dist_tol]
    )
    dist_tol2 = dist_tol * dist_tol
    start1, end1 = poly_point1[0], poly_point1[-1]
    start2, end2 = poly_point2[0], poly_point2[-1]
    pp1 = poly_point1[:]
    pp2 = poly_point2[:]
    points = []
    if close_points2(end1, start2, dist2=dist_tol2):
        points.extend(pp1)
        points.extend(pp2[1:])
    elif close_points2(end1, end2, dist2=dist_tol2):
        points.extend(pp1)
        pp2.reverse()
        points.extend(pp2[1:])
    elif close_points2(start1, start2, dist2=dist_tol2):
        pp1.reverse()
        points.extend(pp1)
        points.extend(pp2[1:])
    elif close_points2(start1, end2, dist2=dist_tol2):
        pp1.reverse()
        points.extend(pp1)
        pp2.reverse()
        points.extend(pp2[1:])

    return points


def trim_right(line, x_value):
    """
    Trim a line to the right at a given x-coordinate.

    Args:
        line (LineType): The line to trim.
        x_value (float): The x-coordinate to trim the line at.

    Returns:
        LineType: The trimmed line.
    """
    reverse = False
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        reverse = True

    if x1 >= x_value:
        res = None
    elif x2 >= x_value:
        intersection_ = intersect(line, [(x_value, 0), (x_value, 1)])
        res = [(x1, y1), intersection_]
    else:
        res = line
        reverse = False
    if res and reverse:
        res = [res[1], res[0]]  # Reverse the order if we swapped x1 and x2

    return res


def trim_left(line, x_value):
    """
    Trim a line to the left at a given x-coordinate.

    Args:
        line (LineType): The line to trim.
        x_value (float): The x-coordinate to trim the line at.

    Returns:
        LineType: The trimmed line.
    """
    reverse = False
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        reverse = True

    if x1 >= x_value:
        res = line
        reverse = False
    elif x2 >= x_value:
        intersection_ = intersect(line, [(x_value, 0), (x_value, 1)])
        res = [intersection_, (x2, y2)]
    else:
        res = None

    if res and reverse:
        res = [res[1], res[0]]  # Reverse the order if we swapped x1 and x2

    return res


def trim_top(line, y_value):
    """
    Trim a line to the top at a given y-coordinate.

    Args:
        line (LineType): The line to trim.
        y_value (float): The y-coordinate to trim the line at.

    Returns:
        LineType: The trimmed line.
    """
    reverse = False
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    if y1 > y2:
        y1, y2 = y2, y1
        x1, x2 = x2, x1
        reverse = True

    if y1 >= y_value:
        res = None
    elif y2 >= y_value:
        intersection_ = intersect(line, [(0, y_value), (1, y_value)])
        res = [(x1, y1), intersection_]
    else:
        res = line
        reverse = False

    if res and reverse:
        res = [res[1], res[0]]

    return res


def trim_bottom(line, y_value):
    """
    Trim a line to the bottom at a given y-coordinate.

    Args:
        line (LineType): The line to trim.
        y_value (float): The y-coordinate to trim the line at.

    Returns:
        LineType: The trimmed line.
    """
    reverse = False
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    if y1 > y2:
        y1, y2 = y2, y1
        x1, x2 = x2, x1
        reverse = True

    if y1 >= y_value:
        res = line
        reverse = False
    elif y2 >= y_value:
        intersection_ = intersect(line, [(0, y_value), (1, y_value)])
        res = [intersection_, (x2, y2)]
    else:
        res = None

    if res and reverse:
        res = [res[1], res[0]]

    return res


def trim_shape(shape: Shape, trim_func: Callable, value: float):
    """
    Trim a shape using a specified trim function and value.

    Args:
        shape (Shape): The shape to trim.
        trim_func (function): The trimming function to use. trim_righ, trim_left,
                                                        trim_top, or trim_bottom.
        value (float): The value at which to trim the shape.

    Returns:
        Shape: The trimmed shape.
    """
    new_shape = shape.copy()
    edges = []
    for edge in shape.edges:
        trimmed = trim_func(edge, value)
        if trimmed:
            edges.append(trimmed)

    points = []
    for edge in edges:
        if edge:
            p1, p2 = edge
            if points:
                if points[-1] != p1:
                    points.append(p1)
            else:
                points.append(p1)
            points.append(p2)
    points = remove_duplicate_points(points)
    new_shape[:] = points
    return new_shape


def left(a: PointType, b: PointType, c: PointType) -> bool:
    """
    Check if point c is left of line ab.
    Args:
        a (PointType): The first point defining the line.
        b (PointType): The second point defining the line.
        c (PointType): The point to test.
    Returns:
        bool: True if point c is left of line ab, False otherwise.
    """

    return area(a, b, c) > 0


def stitch(
    lines: list[LineType],
    closed: bool = True,
    return_points: bool = True,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> list[PointType]:
    """
    Stitches a list of lines together.

    Args:
        lines (list[LineType]): List of lines to stitch.
        closed (bool, optional): Whether the lines form a closed shape. Defaults to True.
        return_points (bool, optional): Whether to return points or lines. Defaults to True.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        list[PointType]: Stitched list of points or lines.
    """
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    if closed:
        points = []
    else:
        points = [lines[0][0]]
    for i, line in enumerate(lines[:-1]):
        x1, y1 = line[0][:2]
        x2, y2 = line[1][:2]
        x3, y3 = lines[i + 1][0][:2]
        x4, y4 = lines[i + 1][1][:2]
        x_point = intersect2(x1, y1, x2, y2, x3, y3, x4, y4)
        if x_point:
            points.append(x_point)
    if closed:
        x1, y1 = lines[-1][0][:2]
        x2, y2 = lines[-1][1][:2]
        x3, y3 = lines[0][0][:2]
        x4, y4 = lines[0][1][:2]
        final_x = intersect2(x1, y1, x2, y2, x3, y3, x4, y4)
        if final_x:
            points.insert(0, final_x)
            points.append(final_x)
    else:
        points.append(lines[-1][1])
    if return_points:
        res = points
    else:
        res = connected_pairs(points)

    return res


def equal_lines(
    line1: LineType, line2: LineType, dist_tol: float | None = None
) -> bool:
    """
    Return True if two lines are close enough.

    Args:
        line1 (LineType): First line.
        line2 (LineType): Second line.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        bool: True if the lines are close enough, False otherwise.
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    dist_tol2 = dist_tol * dist_tol
    p1, p2 = line1
    p3, p4 = line2
    return (
        close_points2(p1, p3, dist2=dist_tol2)
        and close_points2(p2, p4, dist2=dist_tol2)
    ) or (
        close_points2(p1, p4, dist2=dist_tol2)
        and close_points2(p2, p3, dist2=dist_tol2)
    )


def extended_line(dist: float, line: LineType, extend_both=False) -> LineType:
    """
    Given a line ((x1, y1), (x2, y2)) and a distance,
    the given line is extended by distance units.
    Return a new line ((x1, y1), (x2', y2')).

    Args:
        dist (float): Distance to extend the line.
        line (LineType): Input line.
        extend_both (bool, optional): Whether to extend both ends of the line. Defaults to False.

    Returns:
        LineType: Extended line.
    """

    def extend(dist, line):
        # p = (1-t)*p1 + t*p2 : parametric equation of a line segment (p1, p2)
        line_length = length(line)
        t = (line_length + dist) / line_length
        p1, p2 = line
        x1, y1 = p1[:2][:2]
        x2, y2 = p2[:2][:2]
        c = 1 - t

        return [(x1, y1), (c * x1 + t * x2, c * y1 + t * y2)]

    if extend_both:
        p1, p2 = extend(dist, line)
        p1, p2 = extend(dist, [p2, p1])
        res = [p2, p1]
    else:
        res = extend(dist, line)

    return res


def line_through_point_angle(
    point: PointType, angle: float, length_: float, both_sides=False
) -> LineType:
    """
    Return a line that passes through the given point
    with the given angle and length.
    If both_side is True, the line is extended on both sides by the given
    length.

    Args:
        point (PointType): PointType through which the line passes.
        angle (float): Angle of the line in radians.
        length_ (float): Length of the line.
        both_sides (bool, optional): Whether to extend the line on both sides. Defaults to False.

    Returns:
        LineType: Line passing through the given point with the given angle and length.
    """
    x, y = point[:2]
    line = [(x, y), (x + length_ * cos(angle), y + length_ * sin(angle))]
    if both_sides:
        p1, p2 = line
        line = extended_line(length_, [p2, p1])

    return line


def split_segment(segment: LineType, point: PointType):
    """Split a segment into two pieces at ``point``.

    Args:
        segment: Line segment ``(p1, p2)``.
        point: Split point (must lie on the segment, not at an endpoint).

    Returns:
        tuple | None: ``((p1, point), (point, p2))``, or None if ``point``
        is an endpoint or not on the segment.
    """
    p1, p2 = segment
    if close_points2(point, p1) or close_points2(point, p2):
        return None
    if not point_on_line_segment(point, segment):
        return None

    return [(p1, point), (point, p2)]


def multi_split_segment(segment: LineType, points: Sequence, dist_tol=0.1):
    """Split a segment into multiple pieces at the given points.

    Split points are ordered by distance from the segment start.

    Args:
        segment: Line segment ``(p1, p2)``.
        points: Points that lie on the segment.
        dist_tol: Unused legacy tolerance parameter. Defaults to 0.1.

    Returns:
        list: Consecutive sub-segments from start to end.
    """
    p1, p2 = segment
    distances = []
    for i, pnt in enumerate(points):
        dist = distance(p1, pnt)
        distances.append((dist, i))
    distances.sort()
    points = [points[ind] for (_, ind) in distances]

    if len(points) == 2:
        close_p1 = close_points2(points[0], p1)
        close_p2 = close_points2(points[1], p2)
        if close_p1 and close_p2:
            return [segment]

    for i, pnt in enumerate(points):
        if close_points2(p1, pnt):
            continue
        if not point_on_line_segment(pnt, segment):
            print("point not on line")
            return None

    segments = []
    start = p1
    for point in points:
        if distance(start, point) < dist_tol:
            continue
        segments.append((start, point))
        start = point

    return segments


def remove_duplicate_points(
    points: list[PointType], dist_tol=None
) -> list[PointType]:
    """
    Return a list of points with duplicate points removed.

    Args:
        points (list[PointType]): List of points.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        list[PointType]: List of points with duplicate points removed.
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    new_points = []
    for i, point in enumerate(points):
        if i == 0:
            new_points.append(point)
        else:
            dist_tol2 = dist_tol * dist_tol
            if not close_points2(point, new_points[-1], dist2=dist_tol2):
                new_points.append(point)
    return new_points


def remove_collinear_points(
    points: list[PointType],
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> list[PointType]:
    """
    Return a list of points with collinear points removed.

    Args:
        points (list[PointType]): List of points.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        list[PointType]: List of points with collinear points removed.
    """
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    new_points = []
    for i, point in enumerate(points):
        if i == 0:
            new_points.append(point)
        else:
            if not collinear(
                new_points[-1],
                point,
                points[(i + 1) % len(points)],
                rel_tol,
                abs_tol,
            ):
                new_points.append(point)
    return new_points


def clockwise(p: PointType, q: PointType, r: PointType) -> bool:
    """Return 1 if the points p, q, and r are in clockwise order,
    return -1 if the points are in counter-clockwise order,
    return 0 if the points are collinear

    Args:
        p (PointType): First point.
        q (PointType): Second point.
        r (PointType): Third point.

    Returns:
        int: 1 if the points are in clockwise order, -1 if counter-clockwise, 0 if collinear.
    """
    area_ = area(p, q, r)
    if area_ > 0:
        res = 1
    elif area_ < 0:
        res = -1
    else:
        res = 0

    return res


def intersects(seg1, seg2):
    """Checks if the line segments intersect.
    If they are chained together, they are considered as intersecting.
    Returns True if the segments intersect, False otherwise.

    Args:
        seg1 (LineType): First line segment.
        seg2 (LineType): Second line segment.

    Returns:
        bool: True if the segments intersect, False otherwise.
    """
    p1, q1 = seg1
    p2, q2 = seg2
    o1 = clockwise(p1, q1, p2)
    o2 = clockwise(p1, q1, q2)
    o3 = clockwise(p2, q2, p1)
    o4 = clockwise(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and between(p1, p2, q1):
        return True
    if o2 == 0 and between(p1, q2, q1):
        return True
    if o3 == 0 and between(p2, p1, q2):
        return True
    return bool(o4 == 0 and between(p2, q1, q2))


def is_chained(seg1, seg2):
    """Checks if the line segments are chained together.

    Args:
        seg1 (LineType): First line segment.
        seg2 (LineType): Second line segment.

    Returns:
        bool: True if the segments are chained together, False otherwise.
    """
    p1, q1 = seg1
    p2, q2 = seg2
    return bool(
        close_points2(p1, p2)
        or close_points2(p1, q2)
        or close_points2(q1, p2)
        or close_points2(q1, q2)
    )


def global_to_local(
    x: float, y: float, xi: float, yi: float, theta: float = 0
) -> PointType:
    """Given a point(x, y) in global coordinates
    and local CS position and orientation,
    return a point(ksi, eta) in local coordinates

    Args:
        x (float): Global x-coordinate.
        y (float): Global y-coordinate.
        xi (float): Local x-coordinate.
        yi (float): Local y-coordinate.
        theta (float, optional): Angle in radians. Defaults to 0.

    Returns:
        PointType: Local coordinates (ksi, eta).
    """
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    ksi = (x - xi) * cos_theta + (y - yi) * sin_theta
    eta = (y - yi) * cos_theta - (x - xi) * sin_theta
    return (ksi, eta)


def stitch_lines(line1: LineType, line2: LineType) -> Sequence[LineType]:
    """if the lines intersect, trim the lines
    if the lines don't intersect, extend the lines

    Args:
        line1 (LineType): First line.
        line2 (LineType): Second line.

    Returns:
        Sequence[LineType]: Trimmed or extended lines.
    """
    intersection_ = intersect(line1, line2)
    res = None
    if intersection_:
        p1, _ = line1
        _, p2 = line2
        line1 = [p1, intersection_]
        line2 = [intersection_, p2]

        res = (line1, line2)

    return res


def get_quadrant(x: float, y: float) -> int:
    """quadrants:
    +x, +y = 1st
    +x, -y = 2nd
    -x, -y = 3rd
    +x, -y = 4th

    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.

    Returns:
        int: Quadrant number.
    """
    return int(floor((atan2(y, x) % (tau)) / (pi / 2)) + 1)


def get_quadrant_from_deg_angle(deg_angle: float) -> int:
    """quadrants:
    (0, 90) = 1st
    (90, 180) = 2nd
    (180, 270) = 3rd
    (270, 360) = 4th

    Args:
        deg_angle (float): Angle in degrees.

    Returns:
        int: Quadrant number.
    """
    return int(floor(deg_angle / 90.0) % 4 + 1)


def _homogenize(coordinates: Sequence[float]) -> NDArray:
    """Internal use only. API provides a homogenize function.
    Given a sequence of coordinates(x1, y1, x2, y2, ... xn, yn),
    return a numpy array of points array(((x1, y1, 1.),
    (x2, y2, 1.), ... (xn, yn, 1.))).

    Args:
        coordinates (Sequence[float]): Sequence of coordinates.

    Returns:
        np.ndarray: Homogeneous coordinates.
    """
    xy_array = np.array(
        list(zip(coordinates[0::2], coordinates[1::2])), dtype=float
    )
    n_rows = xy_array.shape[0]
    ones = np.ones((n_rows, 1), dtype=float)
    homogeneous_array = np.append(xy_array, ones, axis=1)

    return homogeneous_array


def on_segment(a, b, p, eps=1e-12):
    """Return True if point ``p`` lies on segment ``ab`` within ``eps``.

    Args:
        a: Segment start point.
        b: Segment end point.
        p: Query point.
        eps: Numeric tolerance. Defaults to ``1e-12``.

    Returns:
        bool: True if ``p`` is collinear with ``ab`` and inside its bbox.
    """
    # check collinear + within bbox
    def cross(ax, ay, bx, by):
        return ax * by - ay * bx

    def orient(a, b, c):
        # cross((b-a),(c-a))
        return cross(b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1])

    if abs(orient(a, b, p)) > eps:
        return False
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def intersection(
    line1: LineType, line2: LineType, rel_tol: float | None = None
) -> int:
    """return the intersection point of two line segments.
    segment1: ((x1, y1), (x2, y2))
    segment2: ((x3, y3), (x4, y4))
    To find the intersection point of two lines use the "intersect" function

    Args:
        line1 (LineType): First line segment.
        line2 (LineType): Second line segment.
        rel_tol (float, optional): Relative tolerance. Defaults to None.

    Returns:
        int: Intersection type.
    """
    if rel_tol is None:
        rel_tol = defaults["rel_tol"]
    x1, y1 = line1[0][:2]
    x2, y2 = line1[1][:2]
    x3, y3 = line2[0][:2]
    x4, y4 = line2[1][:2]
    return intersection2(x1, y1, x2, y2, x3, y3, x4, y4)


def merge_segments(
    seg1: Sequence[PointType], seg2: Sequence[PointType]
) -> Sequence[PointType]:
    """Merge two segments into one if they overlap or chain.

    Order of endpoints does not matter. Unconnected segments are not merged.

    Args:
        seg1: First segment ``(p1, p2)``.
        seg2: Second segment ``(p3, p4)``.

    Returns:
        Sequence[PointType]: Merged segment endpoints, or the originals if
        they cannot be merged.
    """

    # """Merge two segments into one segment if they are connected.
    # They need to be overlapping or simply connected to each other,
    # otherwise they will not be merged. Order doesn't matter.

    # Args:
    #     seg1 (Sequence[PointType]): First segment.
    #     seg2 (Sequence[PointType]): Second segment.

    # Returns:
    #     Sequence[PointType]: Merged segment.
    # """
    Conn = Connection
    p1, p2 = seg1
    p3, p4 = seg2

    res = all_intersections([(p1, p2), (p3, p4)], use_intersection3=True)
    if res:
        conn_type = next(iter(res.values()))[0][0]
        verts = next(iter(res.values()))[0][1]
        if conn_type in (Conn.OVERLAPS, Conn.CONGRUENT, Conn.CHAIN):
            res = verts
        elif conn_type == Conn.COLL_CHAIN:
            res = (verts[0], verts[1])
        else:
            res = None
    else:
        res = None  # need this to avoid returning an empty dict

    return res


def invert(p, center, radius):
    """Inverts p about a circle at the given center and radius

    Args:
        p (PointType): PointType to invert.
        center (PointType): Center of the circle.
        radius (float): Radius of the circle.

    Returns:
        PointType: Inverted point.
    """
    dist = distance(p, center)
    if dist == 0:
        return p
    p = np.array(p)
    center = np.array(center)
    return center + (radius**2 / dist**2) * (p - center)
    # return radius**2 * (p - center) / dist


def is_horizontal(line: LineType, eps: float = 0.0001) -> bool:
    """Return True if the line is horizontal.

    Args:
        line (LineType): Input line.
        eps (float, optional): Tolerance. Defaults to 0.0001.

    Returns:
        bool: True if the line is horizontal, False otherwise.
    """
    return abs(j_vec.dot(line_vector(line))) <= eps


def is_vertical(line: LineType, eps: float = 0.0001) -> bool:
    """Return True if the line is vertical.

    Args:
        line (LineType): Input line.
        eps (float, optional): Tolerance. Defaults to 0.0001.

    Returns:
        bool: True if the line is vertical, False otherwise.
    """
    return abs(i_vec.dot(line_vector(line))) <= eps


def length(line: LineType) -> float:
    """Return the length of a line.

    Args:
        line (LineType): Input line.

    Returns:
        float: Length of the line.
    """
    p1, p2 = line
    return distance(p1, p2)


def lerp_point(p1: PointType, p2: PointType, t: float) -> PointType:
    """Linear interpolation of two points.

    Args:
        p1 (PointType): First point.
        p2 (PointType): Second point.
        t (float): Interpolation parameter. t = 0 => p1, t = 1 => p2.

    Returns:
        PointType: Interpolated point.
    """
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    return (lerp(x1, x2, t), lerp(y1, y2, t))


def slope(
    start_point: PointType, end_point: PointType, rel_tol=None, abs_tol=None
) -> float:
    """Return the slope of a line given by two points.
    Order makes a difference.

    Args:
        start_point (PointType): Start point of the line.
        end_point (PointType): End point of the line.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        float: Slope of the line.
    """
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    x1, y1 = start_point[:2]
    x2, y2 = end_point[:2]
    if isclose(x1, x2, rel_tol=rel_tol, abs_tol=abs_tol):
        res = defaults["INF"]
    else:
        res = (y2 - y1) / (x2 - x1)

    return res


def segmentize_line(line: LineType, segment_length: float) -> list[LineType]:
    """Return a list of points that would form segments with the given length.

    Args:
        line (LineType): Input line.
        segment_length (float): Length of each segment.

    Returns:
        list[LineType]: List of segments.
    """
    length_ = distance(line[0], line[1])
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    increments = int(length_ / segment_length)
    x_segments = np.linspace(x1, x2, increments)
    y_segments = np.linspace(y1, y2, increments)

    return list(zip(x_segments, y_segments))


def angle(point: PointType) -> float:
    """Return the angle of a line drawn from the given point to the origin in radians.

    Args:
        point (PointType): Input point.

    Returns:
        float: Angle of the point in radians.
    """
    return atan2(point[1], point[0])


def line_through_point_and_angle(
    point: PointType, angle: float, length_: float = 100
) -> LineType:
    """Return a line through the given point with the given angle and length

    Args:
        point (PointType): PointType through which the line passes.
        angle (float): Angle of the line in radians.
        length_ (float, optional): Length of the line. Defaults to 100.

    Returns:
        LineType: Line passing through the given point with the given angle and length.
    """
    x, y = point[:2]
    dx = length_ * cos(angle)
    dy = length_ * sin(angle)
    return [[x, y], [x + dx, y + dy]]


def ndarray_to_xy_list(arr: NDArray) -> Sequence[PointType]:
    """Convert a numpy array to a list of points.

    Args:
        arr (np.ndarray): Input numpy array.

    Returns:
        Sequence[PointType]: List of points.
    """
    return arr[:, :2].tolist()


def tfl_by_sides(
    point1: PointType, point2: PointType, side1: float, side2: float
):
    """Triangle from line segment and two sides.
    Returns the points of the triangle given by the two points and the two sides.

        Args:
            point1 (PointType): First point of the line segment.
            point2 (PointType): Second point of the line segment.
            side1 (float): Length of the first side.
            side2 (float): Length of the second side.

        Returns:
            list[PointType]: List of points of the triangle.

    """
    c = sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    if c == 0:
        raise ValueError("Error! Points are coincident.")

    if side1 + side2 < c:
        raise ValueError(
            "Error! The sum of the sides is less than the distance between the points."
        )

    if side1 + c < side2:
        raise ValueError(
            "Error! The sum of the first side and the distance between the points is less than the second side."
        )

    if side2 + c < side1:
        raise ValueError(
            "Error! The sum of the second side and the distance between the points is less than the first side."
        )

    if side1 == 0 or side2 == 0:
        raise ValueError("Error! One of the sides is zero.")

    if side1 < 0 or side2 < 0:
        raise ValueError("Error! One of the sides is negative.")

    intersections = circle_circle_intersections(point1, side1, point2, side2)

    return intersections


def point_on_line(
    point: PointType,
    line: LineType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if the given point is on the given line

    Args:
        point (PointType): Input point.
        line (LineType): Input line.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        bool: True if the point is on the line, False otherwise.
    """
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    p1, p2 = line
    return isclose(
        slope(p1, point), slope(point, p2), rel_tol=rel_tol, abs_tol=abs_tol
    )


def point_on_line_segment(
    point: PointType,
    line: LineType,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Return True if the given point is on the given line segment

    Args:
        point (PointType): Input point.
        line (LineType): Input line segment.
        rel_tol (float, optional): Relative tolerance. Defaults to None.
        abs_tol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        bool: True if the point is on the line segment, False otherwise.
    """
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    p1, p2 = line
    return isclose(
        (distance(p1, point) + distance(p2, point)),
        distance(p1, p2),
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def point_to_line_distance(point: PointType, line: LineType) -> float:
    """Return the distance between a line and a point.

    Args:
        point (PointType): Input point.
        line (LineType): Input line.

    Returns:
        float: Distance from the point to the line.
    """
    x0, y0 = point
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    dx = x2 - x1
    dy = y2 - y1
    return abs(dx * (y1 - y0) - (x1 - x0) * dy) / sqrt(dx**2 + dy**2)


def point_to_line_seg_distance(p, lp1, lp2):
    """Given a point p and a line segment defined by boundary points
    lp1 and lp2, returns the distance between the line segment and the point.
    If the point is not located in the perpendicular area between the
    boundary points, returns False.

    Args:
        p (PointType): Input point.
        lp1 (PointType): First boundary point of the line segment.
        lp2 (PointType): Second boundary point of the line segment.

    Returns:
        float: Distance between the point and the line segment, or False if the point is not in the perpendicular area.
    """
    if lp1[:2] == lp2[:2]:
        msg = "Error! Line is ill defined. Start and end points are coincident."
        raise ValueError(msg)
    x3, y3 = p[:2]
    x1, y1 = lp1[:2]
    x2, y2 = lp2[:2]

    u = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / distance(
        lp1, lp2
    ) ** 2
    if 0 <= u <= 1:
        x = x1 + u * (x2 - x1)
        y = y1 + u * (y2 - y1)
        res = distance((x, y), p)
    else:
        res = False  # p is not between lp1 and lp2

    return res


def radius_to_side_len(n: int, radius: float) -> float:
    """Given a radius and the number of sides, return the side length
    of an n-sided regular polygon with the given radius

    Args:
        n (int): Number of sides.
        radius (float): Radius of the polygon.

    Returns:
        float: Side length of the polygon.
    """
    return 2 * radius * sin(pi / n)


def tokenize_svg_path(path: str) -> list[str]:
    """Tokenize an SVG path string.

    Args:
        path (str): SVG path string.

    Returns:
        list[str]: List of tokens.
    """
    return re.findall(r"[a-zA-Z]|[-+]?\d*\.\d+|\d+", path)


def law_of_cosines(a: float, b: float, c: float) -> float:
    """Return the angle of a triangle given the three sides.
    Returns the angle of A in radians. A is the angle between
    sides b and c.
    cos(A) = (b^2 + c^2 - a^2) / (2 * b * c)

    Args:
        a (float): Length of side a.
        b (float): Length of side b.
        c (float): Length of side c.

    Returns:
        float: Angle of A in radians.
    """
    return acos((b**2 + c**2 - a**2) / (2 * b * c))


def segmentize_catmull_rom(
    a: float, b: float, c: float, d: float, n: int = 100
) -> Sequence[float]:
    """a and b are the control points and c and d are
    start and end points respectively,
    n is the number of segments to generate.

    Args:
        a (float): First control point.
        b (float): Second control point.
        c (float): Start point.
        d (float): End point.
        n (int, optional): Number of segments to generate. Defaults to 100.

    Returns:
        Sequence[float]: List of points representing the segments.
    """
    a = array(a[:2], dtype=float)
    b = array(b[:2], dtype=float)
    c = array(c[:2], dtype=float)
    d = array(d[:2], dtype=float)

    t = 0
    dt = 1.0 / n
    points = []
    term1 = 2 * b
    term2 = -a + c
    term3 = 2 * a - 5 * b + 4 * c - d
    term4 = -a + 3 * b - 3 * c + d

    for _ in range(n + 1):
        q = 0.5 * (term1 + term2 * t + term3 * t**2 + term4 * t**3)
        points.append([q[0], q[1]])
        t += dt
    return points


def side_len_to_radius(n: int, side_len: float) -> float:
    """Given a side length and the number of sides, return the radius
    of an n-sided regular polygon with the given side_len length

    Args:
        n (int): Number of sides.
        side_len (float): Side length of the polygon.

    Returns:
        float: Radius of the polygon.
    """
    return side_len / (2 * sin(pi / n))


def translate_line(dx: float, dy: float, line: LineType) -> LineType:
    """Return a translated line by dx and dy

    Args:
        dx (float): Translation distance in x-direction.
        dy (float): Translation distance in y-direction.
        line (LineType): Input line.

    Returns:
        LineType: Translated line.
    """
    x1, y1 = line[0][:2]
    x2, y2 = line[1][:2]
    return [[x1 + dx, y1 + dy], [x2 + dx, y2 + dy]]


def trim_line(line1: LineType, line2: LineType) -> LineType:
    """Trim line1 to the intersection of line1 and line2.
    Extend it if necessary.

    Args:
        line1 (LineType): First line.
        line2 (LineType): Second line.

    Returns:
        LineType: Trimmed line.
    """
    intersection_ = intersection(line1, line2)
    return [line1[0], intersection_]


def tri_to_cart(points):
    """
    Convert a list of points from triangular to cartesian coordinates.

    Args:
        points (list[PointType]): List of points in triangular coordinates.

    Returns:
        np.ndarray: List of points in cartesian coordinates.
    """
    u = [1, 0]
    v = cos(pi / 3), sin(pi / 3)
    convert = array([u, v])

    return array(points) @ convert


def cart_to_tri(points):
    """
    Convert a list of points from cartesian to triangular coordinates.

    Args:
        points (list[PointType]): List of points in cartesian coordinates.

    Returns:
        np.ndarray: List of points in triangular coordinates.
    """
    u = [1, 0]
    v = cos(pi / 3), sin(pi / 3)
    convert = np.linalg.inv(array([u, v]))

    return array(points) @ convert


def flat_points(connected_segments):
    """Return a list of points from a list of connected pairs of points.

    Args:
        connected_segments (list[tuple]): List of connected pairs of points.

    Returns:
        list[PointType]: List of points.
    """
    points = [line[0] for line in connected_segments]
    points.append(connected_segments[-1][1])
    return points


def point_in_quad(point: PointType, quad: list[PointType]) -> bool:
    """Return True if the point is inside the quad.

    Args:
        point (PointType): Input point.
        quad (list[PointType]): List of points representing the quad.

    Returns:
        bool: True if the point is inside the quad, False otherwise.
    """
    x, y = point[:2]
    x1, y1 = quad[0][:2]
    x2, y2 = quad[1][:2]
    x3, y3 = quad[2][:2]
    x4, y4 = quad[3][:2]
    xs = [x1, x2, x3, x4]
    ys = [y1, y2, y3, y4]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    return min_x <= x <= max_x and min_y <= y <= max_y


def offset_point_from_start(p1, p2, offset):
    """p1, p2: points on a line
    offset: distance from p1
    return the point on the line at the given offset

    Args:
        p1 (PointType): First point on the line.
        p2 (PointType): Second point on the line.
        offset (float): Distance from p1.

    Returns:
        PointType: PointType on the line at the given offset.
    """
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    dx, dy = x2 - x1, y2 - y1
    d = (dx**2 + dy**2) ** 0.5
    if d == 0:
        res = p1
    else:
        res = (x1 + offset * dx / d, y1 + offset * dy / d)

    return res


def angle_between_two_lines(line1, line2):
    """Return the angle between two lines in radians.

    Args:
        line1 (LineType): First line.
        line2 (LineType): Second line.

    Returns:
        float: Angle between the two lines in radians.
    """
    alpha1 = line_angle(*line1)
    alpha2 = line_angle(*line2)
    return abs(alpha1 - alpha2)


def circle_tangent_to2lines(line1, line2, intersection_, radius):
    """Given two lines, their intersection point and a radius,
    return the center of the circle tangent to both lines and
    with the given radius.

    Args:
        line1 (LineType): First line.
        line2 (LineType): Second line.
        intersection_ (PointType): Intersection point of the lines.
        radius (float): Radius of the circle.

    Returns:
        tuple: Center of the circle, start and end points of the tangent lines.
    """
    alpha = angle_between_two_lines(line1, line2)
    dist = radius / sin(alpha / 2)
    start = offset_point_from_start(intersection_, line1.p1, dist)
    center = rotate_point(start, intersection_, alpha / 2)
    end = offset_point_from_start(intersection_, line2.p1, dist)

    return center, start, end


def triangle_area(a: float, b: float, c: float) -> float:
    """
    Given side lengths a, b and c, return the area of the triangle.

    Args:
        a (float): Length of side a.
        b (float): Length of side b.
        c (float): Length of side c.

    Returns:
        float: Area of the triangle.
    """
    a_b = a - b
    return sqrt((a + (b + c)) * (c - (a_b)) * (c + (a_b)) * (a + (b - c))) / 4


def get_polygon_grid_point(n, line1, line2, circumradius=100):
    """See chapter ??? for explanation of this function.

    Args:
        n (int): Number of sides.
        line1 (LineType): First line.
        line2 (LineType): Second line.
        circumradius (float, optional): Circumradius of the polygon. Defaults to 100.

    Returns:
        PointType: Grid point of the polygon.
    """
    s = circumradius * 2 * sin(pi / n)  # side length
    points = reg_poly_points(0, 0, n, s)[:-1]
    p1 = points[line1[0]]
    p2 = points[line1[1]]
    p3 = points[line2[0]]
    p4 = points[line2[1]]

    return intersection((p1, p2), (p3, p4))[1]


def is_ccw(vertices, *, eps=0.0):
    """Return True if polygon vertices are in counter-clockwise order.

    Args:
        vertices: List of ``(x, y)`` without repeating the first vertex at
            the end.
        eps: Tolerance; if ``abs(signed_area) <= eps``, returns False
            (degenerate).

    Returns:
        bool: True for CCW orientation.

    Raises:
        ValueError: If fewer than 3 vertices are provided.
    """
    n = len(vertices)
    if n < 3:
        raise ValueError("Need at least 3 vertices")

    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1  # 2 * signed area

    return area > 0


def bisector_line(a: PointType, b: PointType, c: PointType) -> LineType:
    """
    Given three points that form two lines [a, b] and [b, c]
    return the bisector line between them.

    Args:
        a (PointType): First point.
        b (PointType): Second point.
        c (PointType): Third point.

    Returns:
        LineType: Bisector line.
    """
    d = midpoint(a, c)

    return [d, b]


def fillet(
    a: PointType, b: PointType, c: PointType, radius: float
) -> tuple[LineType, LineType, PointType]:
    """
    Given three points that form two lines [a, b] and [b, c]
    return the clipped lines [a, d], [e, c], center point
    of the radius circle (tangent to both lines), and the arc
    angle of the formed fillet.

    Args:
        a (PointType): First point.
        b (PointType): Second point.
        c (PointType): Third point.
        radius (float): Radius of the fillet.

    Returns:
        tuple: Clipped lines [a, d], [e, c], center point of the radius circle, and the arc angle.
    """
    alpha2 = angle_between_lines2(a, b, c) / 2
    sin_alpha2 = sin(alpha2)
    cos_alpha2 = cos(alpha2)
    clip_length = radius * cos_alpha2 / sin_alpha2
    d = offset_point_from_start(b, a, clip_length)
    e = offset_point_from_start(b, c, clip_length)
    mp = midpoint(a, c)  # [b, mp] is the bisector line
    center = offset_point_from_start(b, mp, radius / sin_alpha2)
    arc_angle = angle_between_lines2(e, center, d)

    return [a, d], [e, c], center, arc_angle


def fillet_points(
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
    radius: float,
    n: int,
    *,
    clamp_radius: bool = False,
    eps: float = 1e-12,
) -> list[PointType]:
    """Sample points along a circular fillet at vertex ``p2``.

    The fillet is between segments ``p1→p2`` and ``p2→p3`` (2D). When
    ``n >= 2``, samples include tangency endpoints. When ``n == 1``, returns
    the arc midpoint. When ``n <= 0``, returns an empty list.

    Args:
        p1: First point ``(x, y)``.
        p2: Corner vertex ``(x, y)``.
        p3: Third point ``(x, y)``.
        radius: Desired fillet radius (must be > 0).
        n: Number of sample points along the arc.
        clamp_radius: If True, reduce ``radius`` to the maximal feasible
            value when it is too large for the segments.
        eps: Numeric tolerance.

    Returns:
        list[PointType]: Sampled points along the fillet arc.

    Raises:
        ValueError: If geometry is degenerate or radius is infeasible
            (unless ``clamp_radius`` is True).
    """
    if radius <= 0:
        raise ValueError("radius must be > 0")

    p1 = p1[:2]
    p2 = p2[:2]
    p3 = p3[:2]

    # Convert to Vector objects
    v1 = Vector(float(p1[0]), float(p1[1]))
    v2 = Vector(float(p2[0]), float(p2[1]))
    v3 = Vector(float(p3[0]), float(p3[1]))

    # Direction vectors along the polyline
    v_in = (v2 - v1).normalize()  # direction into p2
    v_out = (v3 - v2).normalize()  # direction out of p2

    # Rays from the corner along each segment
    u1 = -v_in  # from p2 toward p1
    u2 = v_out  # from p2 toward p3

    # Interior angle between u1 and u2
    c = max(-1.0, min(1.0, u1.dot(u2)))
    theta = acos(c)
    if theta < eps or abs(theta - pi) < eps:
        raise ValueError(
            "Points are collinear or angle too close to 0/180 degrees."
        )

    # Distances and feasibility of the radius
    L1 = (v2 - v1).mag()
    L2 = (v3 - v2).mag()
    # Tangency distance along each leg
    t = radius / tan(theta / 2.0)
    # Max radius allowed by each leg: r_max_i = Li * tan(theta/2)
    r_max = min(L1, L2) * tan(theta / 2.0)

    if t > L1 + eps or t > L2 + eps:
        if clamp_radius:
            # Clamp radius to feasible value (slightly inside to avoid degeneracy)
            radius = max(0.0, min(radius, r_max * (1.0 - 1e-9)))
            t = radius / tan(theta / 2.0)
        else:
            raise ValueError(
                f"Radius too large for given segments. Max feasible ~ {r_max:.6f}"
            )

    # Tangency points on each leg
    A = v2 + u1 * t
    B = v2 + u2 * t

    # Bisector direction (inside the angle)
    bis = u1 + u2
    if bis.mag() < eps:
        raise ValueError("Angle too close to 180°, bisector undefined.")
    w_hat = bis.normalize()

    # Center of the fillet circle
    center_dist = radius / sin(theta / 2.0)
    C = v2 + w_hat * center_dist

    # Angles of tangency points around center
    a1 = atan2(A.y - C.y, A.x - C.x)
    a2 = atan2(B.y - C.y, B.x - C.x)

    # Determine sweep direction based on the turn (left/CCW or right/CW)
    turn = v_in.cross(v_out)  # >0 => left turn (CCW), <0 => right turn (CW)
    delta = a2 - a1
    # Normalize delta to follow the turn direction along the minor arc
    if turn > 0:  # CCW
        if delta < 0:
            delta += 2.0 * pi
    else:  # CW
        if delta > 0:
            delta -= 2.0 * pi

    # Generate points on the arc
    if n <= 0:
        return []
    if n == 1:
        ang = a1 + 0.5 * delta
        pt = C + Vector(radius * cos(ang), radius * sin(ang))
        return [(pt.x, pt.y)]
    # n >= 2: include both tangency endpoints
    pts: list[PointType] = []
    for i in range(n):
        t_frac = i / (n - 1)
        ang = a1 + t_frac * delta
        pt = C + Vector(radius * cos(ang), radius * sin(ang))
        pts.append((pt.x, pt.y))
    return pts


def fillet_corners(
    vertices: Sequence[PointType], d_vert_radius: dict[int, float], n: int = 12
) -> Sequence[PointType]:
    """Return a new vertex list with selected corners filleted.

    Args:
        vertices: Original polygon/polyline vertices.
        d_vert_radius: Map from vertex index to fillet radius.
        n: Number of sample points per fillet. Defaults to 12.

    Returns:
        Sequence[PointType]: Vertices including fillet arc samples.
    """
    count = len(vertices)

    # Build set of indices to fillet for quick lookup
    indices = d_vert_radius.keys()
    fillet_set = set(indices)

    # Build new vertex list
    new_vertices = []
    fillet_count = 0
    for i in range(count):
        if i not in fillet_set:
            # Keep original vertex
            new_vertices.append(vertices[i][:2])
        else:
            # Replace with fillet arc
            prev_idx = (i - 1) % count
            next_idx = (i + 1) % count

            p1 = vertices[prev_idx][:2]
            p2 = vertices[i][:2]
            p3 = vertices[next_idx][:2]

            # Generate fillet points
            radius = d_vert_radius[i]
            arc_points = fillet_points(p1, p2, p3, radius, n, clamp_radius=True)
            new_vertices.extend(arc_points)
            fillet_count += 1

    # Create new shape with same properties
    return new_vertices


def line_by_point_angle_length(point, angle, length_):
    """
    Given a point, an angle, and a length, return the line
    that starts at the point and has the given angle and length.

    Args:
        point (PointType): Start point of the line.
        angle (float): Angle of the line in radians.
        length_ (float): Length of the line.

    Returns:
        LineType: Line with the given angle and length.
    """
    x, y = point[:2]
    dx = length_ * cos(angle)
    dy = length_ * sin(angle)

    return [(x, y), (x + dx, y + dy)]


def calc_area(points):
    """Calculate the area of a simple polygon (given by a list of its vertices).

    Args:
        points (list[PointType]): List of points representing the polygon.

    Returns:
        tuple: Area of the polygon and whether it is clockwise.
    """
    area_ = 0
    n_points = len(points)
    for i in range(n_points):
        v = points[i]
        vnext = points[(i + 1) % n_points]
        area_ += v[0] * vnext[1] - vnext[0] * v[1]
    clockwise = area_ > 0

    return (abs(area_ / 2.0), clockwise)


def remove_bad_points(points):
    """Remove redundant and collinear points from a list of points.

    Args:
        points (list[PointType]): List of points.

    Returns:
        list[PointType]: List of points with redundant and collinear points removed.
    """
    EPSILON = 1e-16
    n_points = len(points)
    # check for redundant points
    for i, p in enumerate(points[:]):
        for j in range(i + 1, n_points - 1):
            if p == points[j]:  # then remove the redundant point
                # maybe we should display a warning message here indicating
                # that redundant point is removed!!!
                points.remove(p)

    n_points = len(points)
    # check for three consecutive points on a line
    lin_points = []
    for i in range(2, n_points - 1):
        if (
            EPSILON
            > calc_area([points[i - 2], points[i - 1], points[i]])[0]
            > -EPSILON
        ):
            lin_points.append(points[i - 1])

    if EPSILON > calc_area([points[-2], points[-1], points[0]])[0] > -EPSILON:
        lin_points.append(points[-1])

    for p in lin_points:
        # maybe we should display a warning message here indicating that linear
        # point is removed!!!
        points.remove(p)

    return points


def is_convex(points):
    """Return True if the polygon is convex.

    Args:
        points (list[PointType]): List of points representing the polygon.

    Returns:
        bool: True if the polygon is convex, False otherwise.
    """
    points = remove_bad_points(points)
    n_checks = len(points)
    points = points + [points[0]]
    senses = []
    for i in range(n_checks):
        if i == (n_checks - 1):
            senses.append(cross_product_sense(points[i], points[0], points[1]))
        else:
            senses.append(
                cross_product_sense(points[i], points[i + 1], points[i + 2])
            )
    s = set(senses)
    return len(s) == 1


def set_vertices(points):
    """Set the next and previous vertices of a list of vertices.

    Args:
        points (list[Vertex]): List of vertices.
    """
    if not isinstance(points[0], Vertex):
        points = [Vertex(*p[:]) for p in points]
    n_points = len(points)
    for i, p in enumerate(points):
        if i == 0:
            p.prev = points[-1]
            p.next = points[i + 1]
        elif i == (n_points - 1):
            p.prev = points[i - 1]
            p.next = points[0]
        else:
            p.prev = points[i - 1]
            p.next = points[i + 1]
        p.angle = cross_product_sense(p.prev, p, p.next)


def circle_circle_intersections(point1, radius1, point2, radius2):
    """Return the intersection points of two circles.

    Args:
        point1 (PointType): Center of the first circle.
        radius1 (float): Radius of the first circle.
        point2 (PointType): Center of the second circle.
        radius2 (float): Radius of the second circle.

    Returns:
        tuple: Intersection points of the two circles.
    """
    # taken from https://stackoverflow.com/questions/55816902/finding-the-
    # intersection-of-two-circles
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    x0, y0 = point1[:2]
    x1, y1 = point2[:2]
    r0 = radius1
    r1 = radius2

    d = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # non intersecting
    if d > r0 + r1 or d < abs(r0 - r1) or d == 0 and r0 == r1:
        res = None
    else:
        a = (r0**2 - r1**2 + d**2) / (2 * d)
        h = sqrt(r0**2 - a**2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (x1 - x0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d
        x4 = x2 - h * (y1 - y0) / d
        y4 = y2 + h * (x1 - x0) / d

        res = ((x3, y3), (x4, y4))

    return res


def circle_segment_intersection(circle, p1, p2):
    """Return True if the circle and the line segment intersect.

    Args:
        circle (Circle): Input circle.
        p1 (PointType): First point of the line segment.
        p2 (PointType): Second point of the line segment.

    Returns:
        bool: True if the circle and the line segment intersect, False otherwise.
    """
    # if line seg and circle intersects returns true, false otherwise
    # c: circle
    # p1 and p2 are the endpoints of the line segment

    x3, y3 = circle.pos[:2]
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    if (
        distance(p1, circle.pos) < circle.radius
        or distance(p2, circle.pos) < circle.radius
    ):
        return True
    u = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / (
        (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
    )
    res = False
    if 0 <= u <= 1:
        x = x1 + u * (x2 - x1)
        y = y1 + u * (y2 - y1)
        if distance((x, y), circle.pos) < circle.radius:
            res = True

    return res  # p is not between lp1 and lp2


def r_polar(a, b, theta):
    """Return the radius (distance between the center and the intersection point)
    of the ellipse at the given angle.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        theta (float): Angle in radians.

    Returns:
        float: Radius of the ellipse at the given angle.
    """
    return (a * b) / sqrt((b * cos(theta)) ** 2 + (a * sin(theta)) ** 2)


def ellipse_line_intersection(a, b, point):
    """Return the intersection points of an ellipse and a line segment
    connecting the given point to the ellipse center at (0, 0).

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        point (PointType): PointType on the line segment.

    Returns:
        list[PointType]: Intersection points of the ellipse and the line segment.
    """
    # adapted from http://mathworld.wolfram.com/Ellipse-LineIntersection.html
    # a, b is the ellipse width/2 and height/2 and (x_0, y_0) is the point

    x_0, y_0 = point[:2]
    x = ((a * b) / (sqrt(a**2 * y_0**2 + b**2 * x_0**2))) * x_0
    y = ((a * b) / (sqrt(a**2 * y_0**2 + b**2 * x_0**2))) * y_0

    return [(x, y), (-x, -y)]


def ellipse_tangent(a, b, x, y, tol=0.001):
    """Calculates the slope of the tangent line to an ellipse at the point (x, y).
    If point is not on the ellipse, return False.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        x (float): x-coordinate of the point.
        y (float): y-coordinate of the point.
        tol (float, optional): Tolerance. Defaults to 0.001.

    Returns:
        float: Slope of the tangent line, or False if the point is not on the ellipse.
    """
    if abs((x**2 / a**2) + (y**2 / b**2) - 1) > tol:
        res = False
    else:
        res = -(b**2 * x) / (a**2 * y)

    return res


def elliptic_arclength(t_0, t_1, a, b):
    """Return the arclength of an ellipse between the given parametric angles.
    The ellipse has semi-major axis a and semi-minor axis b.

    Args:
        t_0 (float): Start parametric angle in radians.
        t_1 (float): End parametric angle in radians.
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.

    Returns:
        float: Arclength of the ellipse between the given parametric angles.
    """
    # from: https://www.johndcook.com/blog/2022/11/02/elliptic-arc-length/
    from scipy.special import ellipeinc  # this takes too long to import!!!

    m = 1 - (b / a) ** 2
    t1 = ellipeinc(t_1 - 0.5 * pi, m)
    t0 = ellipeinc(t_0 - 0.5 * pi, m)
    return a * (t1 - t0)


def central_to_parametric_angle(a, b, phi):
    """
    Converts a central angle to a parametric angle on an ellipse.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        phi (float): Central angle in radians.

    Returns:
        float: Parametric angle in radians.
    """
    t = atan2((a / b) * sin(phi), cos(phi))
    if t < 0:
        t += 2 * pi

    return t


def parametric_to_central_angle(a, b, t):
    """
    Converts a parametric angle on an ellipse to a central angle.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        t (float): Parametric angle in radians.

    Returns:
        float: Central angle in radians.
    """
    phi = atan2((b / a) * sin(t), cos(t))
    if phi < 0:
        phi += 2 * pi

    return phi


def ellipse_points(center, a, b, n_points):
    """Generate points on an ellipse.

    Args:
        center (tuple): (x, y) coordinates of the ellipse center.
        a (float): Length of the semi-major axis.
        b (float): Length of the semi-minor axis.
        n_points (int): Number of points to generate.

    Returns:
        np.ndarray: Array of (x, y) coordinates of the ellipse points.
    """
    t = np.linspace(0, 2 * np.pi, n_points)
    x = center[0] + a * np.cos(t)
    y = center[1] + b * np.sin(t)

    return np.column_stack((x, y))


def ellipse_point(a, b, angle):
    """Return a point on an ellipse with the given a=width/2, b=height/2, and angle.

    Args:
        a (float): Semi-major axis of the ellipse.
        b (float): Semi-minor axis of the ellipse.
        angle (float): Angle in radians.

    Returns:
        PointType: PointType on the ellipse.
    """
    r = r_polar(a, b, angle)

    return (r * cos(angle), r * sin(angle))


def circle_line_intersection(c, p1, p2):
    """Return the intersection points of a circle and a line segment.

    Args:
        c (Circle): Input circle.
        p1 (PointType): First point of the line segment.
        p2 (PointType): Second point of the line segment.

    Returns:
        tuple: Intersection points of the circle and the line segment.
    """

    # adapted from http://mathworld.wolfram.com/Circle-LineIntersection.html
    # c is the circle and p1 and p2 are the line points
    def sgn(num):
        if num < 0:
            res = -1
        else:
            res = 1
        return res

    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    r = c.radius
    x, y = c.pos[:2]

    x1 -= x
    x2 -= x
    y1 -= y
    y2 -= y

    dx = x2 - x1
    dy = y2 - y1
    dr = sqrt(dx**2 + dy**2)
    d = x1 * y2 - x2 * y1
    d2 = d**2
    r2 = r**2
    dr2 = dr**2

    discriminant = r2 * dr2 - d2

    if discriminant > 0:
        ddy = d * dy
        ddx = d * dx
        sqrterm = sqrt(r2 * dr2 - d2)
        temp = sgn(dy) * dx * sqrterm

        a = (ddy + temp) / dr2
        b = (-ddx + abs(dy) * sqrterm) / dr2
        if discriminant == 0:
            res = (a + x, b + y)
        else:
            c = (ddy - temp) / dr2
            d = (-ddx - abs(dy) * sqrterm) / dr2
            res = ((a + x, b + y), (c + x, d + y))

    else:
        res = False

    return res


def circle_poly_intersection(circle, polygon):
    """Return True if the circle and the polygon intersect.

    Args:
        circle (Circle): Input circle.
        polygon (Polygon): Input polygon.

    Returns:
        bool: True if the circle and the polygon intersect, False otherwise.
    """
    points = polygon.vertices
    n = len(points)
    res = False
    for i in range(n):
        x = points[i][0]
        y = points[i][1]
        x1 = points[(i + 1) % n][0]
        y1 = points[(i + 1) % n][1]
        if circle_segment_intersection(circle, (x, y), (x1, y1)):
            res = True
            break
    return res


def point_to_circle_distance(point, center, radius):
    """Given a point, center point, and radius, returns distance
    between the given point and the circle

    Args:
        point (PointType): Input point.
        center (PointType): Center of the circle.
        radius (float): Radius of the circle.

    Returns:
        float: Distance between the point and the circle.
    """
    return abs(distance(center, point) - radius)


def get_interior_points(start, end, n_points):
    """Given start and end points and number of interior points
    returns the positions of the interior points

    Args:
        start (PointType): Start point.
        end (PointType): End point.
        n_points (int): Number of interior points.

    Returns:
        list[PointType]: List of interior points.
    """
    rot_angle = line_angle(start, end)
    length_ = distance(start, end)
    seg_length = length_ / (n_points + 1.0)
    points = []
    for i in range(n_points):
        points.append(
            rotate_point(
                [start[0] + seg_length * (i + 1), start[1]], start, rot_angle
            )
        )
    return points


def circle_3point(point1, point2, point3):
    """Given three points, returns the center point and radius

    Args:
        point1 (PointType): First point.
        point2 (PointType): Second point.
        point3 (PointType): Third point.

    Returns:
        tuple: Center point and radius of the circle.
    """
    ax, ay = point1[:2]
    bx, by = point2[:2]
    cx, cy = point3[:2]
    a = bx - ax
    b = by - ay
    c = cx - ax
    d = cy - ay
    e = a * (ax + bx) + b * (ay + by)
    f = c * (ax + cx) + d * (ay + cy)
    g = 2.0 * (a * (cy - by) - b * (cx - bx))
    if g == 0:
        raise ValueError("Points are collinear!")

    px = ((d * e) - (b * f)) / g
    py = ((a * f) - (c * e)) / g
    r = ((ax - px) ** 2 + (ay - py) ** 2) ** 0.5
    return ((px, py), r)


def project_point_on_line(point: Vertex, line: Edge):
    """Given a point and a line, returns the projection of the point on the line

    Args:
        point (Vertex): Input point.
        line (Edge): Input line.

    Returns:
        Vertex: Projection of the point on the line.
    """
    v = point
    a, b = line

    av = v - a
    ab = b - a
    t = (av * ab) / (ab * ab)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return a + ab * t


class Vertex(list):
    """A 3D vertex."""

    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.type = Types.VERTEX

    def __repr__(self):
        return f"Vertex({self.x}, {self.y}, {self.z})"

    def __eq__(self, other):
        return (
            self[0] == other[0] and self[1] == other[1] and self[2] == other[2]
        )

    def copy(self):
        """Return a new ``Vertex`` with the same coordinates.

        Returns:
            Vertex: Copy of this vertex.
        """
        return Vertex(self.x, self.y, self.z)

    def __add__(self, other):
        return Vertex(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vertex(self.x - other.x, self.y - other.y, self.z - other.z)

    @property
    def coords(self):
        """Return the coordinates as a tuple."""
        return (self.x, self.y, self.z)

    @property
    def array(self):
        """Homogeneous coordinates as a numpy array."""
        return array([self.x, self.y, 1])

    def v_tuple(self):
        """Return the vertex as a tuple."""
        return (self.x, self.y, self.z)

    def below(self, other):
        """This is for 2D points only

        Args:
            other (Vertex): Other vertex.

        Returns:
            bool: True if this vertex is below the other vertex, False otherwise.
        """
        res = False
        if self.y < other.y or self.y == other.y and self.x > other.x:
            res = True
        return res

    def above(self, other):
        """This is for 2D points only

        Args:
            other (Vertex): Other vertex.

        Returns:
            bool: True if this vertex is above the other vertex, False otherwise.
        """
        if self.y > other.y or self.y == other.y and self.x < other.x:
            res = True
        else:
            res = False

        return res


class Edge:
    """A 2D edge."""

    def __init__(
        self,
        start_point: PointType | Vertex,
        end_point: PointType | Vertex,
    ):
        if isinstance(start_point, PointType):
            start = Vertex(*start_point)
        elif isinstance(end_point, Vertex):
            start = start_point
        else:
            raise TypeError(
                "Start point should be a PointType or Vertex instance."
            )

        if isinstance(end_point, PointType):
            end = Vertex(*end_point)
        elif isinstance(end_point, Vertex):
            end = end_point
        else:
            raise TypeError(
                "End point should be a PointType or Vertex instance."
            )

        self.start = start
        self.end = end
        self.type = Types.EDGE

    def __repr__(self):
        return str(f"Edge({self.start}, {self.end})")

    def __str__(self):
        return str(f"Edge({self.start.point}, {self.end.point})")

    def __eq__(self, other):
        start = other.start.point
        end = other.end.point

        return (
            isclose(
                self.start.point,
                start,
                rel_tol=defaults["rel_tol"],
                abs_tol=defaults["abs_tol"],
            )
            and isclose(
                self.end.point,
                end,
                rel_tol=defaults["rel_tol"],
                abs_tol=defaults["abs_tol"],
            )
        ) or (
            isclose(
                self.start.point,
                end,
                rel_tol=defaults["rel_tol"],
                abs_tol=defaults["abs_tol"],
            )
            and isclose(
                self.end.point,
                start,
                rel_tol=defaults["rel_tol"],
                abs_tol=defaults["abs_tol"],
            )
        )

    def __getitem__(self, subscript):
        vertices = self.vertices
        if isinstance(subscript, slice):
            res = vertices[subscript.start : subscript.stop : subscript.step]
        elif isinstance(subscript, int):
            res = vertices[subscript]
        else:
            raise TypeError("Invalid subscript.")
        return res

    def __setitem__(self, subscript, value):
        vertices = self.vertices
        if isinstance(subscript, slice):
            vertices[subscript.start : subscript.stop : subscript.step] = value
        else:
            isinstance(subscript, int)
            vertices[subscript] = value

    @property
    def slope(self):
        """Line slope. The slope of the line passing through the start and end points."""
        return (self.y2 - self.y1) / (self.x2 - self.x1)

    @property
    def angle(self):
        """Line angle. Angle between the line and the x-axis."""
        return atan2(self.y2 - self.y1, self.x2 - self.x1)

    @property
    def inclination(self):
        """Inclination angle. Angle between the line and the x-axis converted to
        a value between zero and pi."""
        return self.angle % pi

    @property
    def length(self):
        """Length of the line segment."""
        return distance(self.start.point, self.end.point)

    @property
    def x1(self):
        """x-coordinate of the start point."""
        return self.start.x

    @property
    def y1(self):
        """y-coordinate of the start point."""
        return self.start.y

    @property
    def x2(self):
        """x-coordinate of the end point."""
        return self.end.x

    @property
    def y2(self):
        """y-coordinate of the end point."""
        return self.end.y

    @property
    def points(self):
        """Start and end"""
        return [self.start.point, self.end.point]

    @property
    def vertices(self):
        """Start and end vertices."""
        return [self.start, self.end]

    @property
    def array(self):
        """Homogeneous coordinates as a numpy array."""
        return array([self.start.array, self.end.array])

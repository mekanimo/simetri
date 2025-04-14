"""Geometry related utilities.
These functions are used to perform geometric operations.
They are not documented. Some of the are one off functions that
are not used in the main codebase or tested."""

# To do: Clean up this module and add documentation.

from __future__ import annotations

from math import hypot, atan2, floor, pi, sin, cos, sqrt, exp, sqrt, acos
from itertools import cycle
from typing import Any, Union, Sequence
import re


import numpy as np
from numpy import isclose, array, around

from simetri.helpers.utilities import (
    flatten,
    lerp,
    sanitize_graph_edges,
    equal_cycles,
    reg_poly_points,
)

from ..helpers.vector import Vector2D
from ..graphics.common import (
    get_defaults,
    common_properties,
    Point,
    Line,
    Sequence,
    i_vec,
    j_vec,
    VecType,
    axis_x,
    axis_y,
)
from ..graphics.all_enums import Connection, Types
from ..settings.settings import defaults

array = np.array
around = np.around

TWO_PI = 2 * pi  # 360 degrees


def is_number(x: Any) -> bool:
    """
    Return True if x is a number.

    Args:
        x (Any): The input value to check.

    Returns:
        bool: True if x is a number, False otherwise.
    """
    return isinstance(x, (int, float, complex)) and not isinstance(x, bool)


def bbox_overlap(
    min_x1: float,
    min_y1: float,
    max_x2: float,
    max_y2: float,
    min_x3: float,
    min_y3: float,
    max_x4: float,
    max_y4: float,
) -> bool:
    """
    Given two bounding boxes, return True if they overlap.

    Args:
        min_x1 (float): Minimum x-coordinate of the first bounding box.
        min_y1 (float): Minimum y-coordinate of the first bounding box.
        max_x2 (float): Maximum x-coordinate of the first bounding box.
        max_y2 (float): Maximum y-coordinate of the first bounding box.
        min_x3 (float): Minimum x-coordinate of the second bounding box.
        min_y3 (float): Minimum y-coordinate of the second bounding box.
        max_x4 (float): Maximum x-coordinate of the second bounding box.
        max_y4 (float): Maximum y-coordinate of the second bounding box.

    Returns:
        bool: True if the bounding boxes overlap, False otherwise.
    """
    return not (
        max_x2 < min_x3 or max_x4 < min_x1 or max_y2 < min_y3 or max_y4 < min_y1
    )


def sine_wave(
    amplitude: float,
    frequency: float,
    duration: float,
    sample_rate: float,
    phase: float = 0,
) -> 'ndarray':
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
) -> 'ndarray':
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


def check_consecutive_duplicates(points, rtol=0, atol=None) -> bool:
    """Check for consecutive duplicate points in a list of points.

        Args:
            points (list): List of points to check.
            rtol (float, optional): Relative tolerance. Defaults to 0.
            atol (float, optional): Absolute tolerance. Defaults to None.

        Returns:
            bool: True if consecutive duplicate points are found, False otherwise.
    """
    if atol is None:
        atol = defaults["atol"]

    for i, pnt in enumerate(points[:-1]):
        next = points[i+1]
        val1 = pnt[0] + pnt[1]
        val2 = next[0] + next[1]
        if isclose(val1, val2, rtol=0, atol=atol):
            if np.allclose(pnt, next, rtol=0, atol=atol):
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


def line_segment_bbox(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float, float]:
    """
    Return the bounding box of a line segment.

    Args:
        x1 (float): Segment start point x-coordinate.
        y1 (float): Segment start point y-coordinate.
        x2 (float): Segment end point x-coordinate.
        y2 (float): Segment end point y-coordinate.

    Returns:
        tuple: Bounding box as (min_x, min_y, max_x, max_y).
    """
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def line_segment_bbox_check(seg1: Line, seg2: Line) -> bool:
    """
    Given two line segments, return True if their bounding boxes overlap.

    Args:
        seg1 (Line): First line segment.
        seg2 (Line): Second line segment.

    Returns:
        bool: True if the bounding boxes overlap, False otherwise.
    """
    x1, y1 = seg1[0]
    x2, y2 = seg1[1]
    x3, y3 = seg2[0]
    x4, y4 = seg2[1]
    return bbox_overlap(
        *line_segment_bbox(x1, y1, x2, y2), *line_segment_bbox(x3, y3, x4, y4)
    )


def all_close_points(
    points: Sequence[Sequence], dist_tol: float = None, with_dist: bool = False
) -> dict[int, list[tuple[Point, int]]]:
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
    point_arr = np.array(points, dtype=np.float32)  # points array [[x1, y1, id1], ...]]
    n_rows = len(points)
    point_arr = point_arr[point_arr[:, 0].argsort()]  # sort by x values in the
    # first column
    xmin = point_arr[:, 0] - dist_tol * 2
    xmin = xmin.reshape(n_rows, 1)
    xmax = point_arr[:, 0] + dist_tol * 2
    xmax = xmax.reshape(n_rows, 1)
    point_arr = np.concatenate((point_arr, xmin, xmax), 1)  # [x, y, id, xmin, xmax]

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


def is_simple(
    polygon,
    rtol: float = None,
    atol: float = None,
) -> bool:
    """
    Return True if the polygon is simple.

    Args:
        polygon (list): List of points representing the polygon.
        rtol (float, optional): Relative tolerance. Defaults to None.
        atol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        bool: True if the polygon is simple, False otherwise.
    """
    rtol, atol = get_defaults(["rtol", "atol"], [rtol, atol])

    if not close_points2(polygon[0], polygon[-1]):
        polygon.append(polygon[0])
    segments = [[polygon[i], polygon[i + 1]] for i in range(len(polygon) - 1)]

    rtol, atol = get_defaults(["rtol", "atol"], [rtol, atol])
    segment_coords = []
    for segment in segments:
        segment_coords.append(
            [segment[0][0], segment[0][1], segment[1][0], segment[1][1]]
        )
    seg_arr = np.array(segment_coords)  # segments array
    n_rows = seg_arr.shape[0]
    xmin = np.minimum(seg_arr[:, 0], seg_arr[:, 2]).reshape(n_rows, 1)
    xmax = np.maximum(seg_arr[:, 0], seg_arr[:, 2]).reshape(n_rows, 1)
    ymin = np.minimum(seg_arr[:, 1], np.maximum(seg_arr[:, 3])).reshape(n_rows, 1)
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
                if not isclose(length1 + length2, chain_length, rtol=rtol, atol=atol):
                    return False
                else:
                    continue
            if res[0] in (Connection.CHAIN, Connection.PARALLEL):
                continue
            if res[0] != Connection.DISJOINT:
                return False

    return True


def all_intersections(
    segments: Sequence[Line],
    rtol: float = None,
    atol: float = None,
    use_intersection3: bool = False,
) -> dict[int, list[tuple[Point, int]]]:
    """
    Find all intersection points of the given list of segments
    (sweep line algorithm variant)

    Args:
        segments (Sequence[Line]): List of line segments [[[x1, y1], [x2, y2]], [[x1, y1], [x2, y2]], ...].
        rtol (float, optional): Relative tolerance. Defaults to None.
        atol (float, optional): Absolute tolerance. Defaults to None.
        use_intersection3 (bool, optional): Whether to use intersection3 function. Defaults to False.

    Returns:
        dict: Dictionary of the form {segment_id: [[id1, (x1, y1)], [id2, (x2, y2)]], ...}.
    """
    rtol, atol = get_defaults(["rtol", "atol"], [rtol, atol])
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
    # ind1, ind2 are indexes of segments in the list of segments
    d_ind1_x_point_ind2 = {}  # {id1: [((x, y), id2), ...], ...}
    d_ind1_conn_type_x_res_ind2 = {}  # {id1: [(conn_type, x_res, id2), ...], ...}
    for i in range(n_rows):
        if use_intersection3:
            d_ind1_conn_type_x_res_ind2[i] = []
        else:
            d_ind1_x_point_ind2[i] = []
    x_points = []  # intersection points
    s_processed = set()  # set of processed segment pairs
    for i in range(n_rows):
        x1, y1, x2, y2, sl_xmin, sl_ymin, sl_xmax, sl_ymax, id1 = seg_arr[i, :]
        id1 = int(id1)
        segment = [x1, y1, x2, y2]
        start = i + 1  # keep pushing the sweep line forward
        # filter by overlap of the bounding boxes of the segments with the
        # sweep line's active segment. If the bounding boxes do not overlap,
        # the segments cannot intersect. If the bounding boxes overlap,
        # the segments may intersect.
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
            if use_intersection3:
                # connection type, point/segment
                res = intersection3(*segment, *seg2, rtol, atol)
                conn_type, x_res = res  # x_res can be a segment or a point
            else:
                # connection type, point
                res = intersection2(*segment, *seg2, rtol, atol)
                conn_type, x_point = res
            if use_intersection3:
                if conn_type not in [Connection.DISJOINT, Connection.PARALLEL]:
                    d_ind1_conn_type_x_res_ind2[id1].append((conn_type, x_res, id2))
                    d_ind1_conn_type_x_res_ind2[id2].append((conn_type, x_res, id1))
            else:
                if conn_type == Connection.INTERSECT:
                    d_ind1_x_point_ind2[id1].append((x_point, id2))
                    d_ind1_x_point_ind2[id2].append((x_point, id1))
                    x_points.append(res[1])

    d_results = {}
    if use_intersection3:
        for k, v in d_ind1_conn_type_x_res_ind2.items():
            if v:
                d_results[k] = v
        res = d_results
    else:
        for k, v in d_ind1_x_point_ind2.items():
            if v:
                d_results[k] = v
        res = d_results, x_points

    return res


def dot_product2(a: Point, b: Point, c: Point) -> float:
    """Dot product of two vectors. AB and BC
    Args:
        a (Point): First point, creating vector BA
        b (Point): Second point, common point for both vectors
        c (Point): Third point, creating vector BC

    Returns:
        float: The dot product of vectors BA and BC
    Note:
        The function calculates (a-b)·(c-b) which is the dot product of vectors BA and BC.
        This is useful for finding angles between segments that share a common point.
    """
    a_x, a_y = a[:2]
    b_x, b_y = b[:2]
    c_x, c_y = c[:2]
    b_a_x = a_x - b_x
    b_a_y = a_y - b_y
    b_c_x = c_x - b_x
    b_c_y = c_y - b_y
    return b_a_x * b_c_x + b_a_y * b_c_y


def cross_product2(a: Point, b: Point, c: Point) -> float:
    """
    Return the cross product of two vectors: BA and BC.

    Args:
        a (Point): First point, creating vector BA
        b (Point): Second point, common point for both vectors
        c (Point): Third point, creating vector BC

    Returns:
        float: The z-component of cross product between vectors BA and BC

    Note:
        This gives the signed area of the parallelogram formed by the vectors BA and BC.
        The sign indicates the orientation (positive for counter-clockwise, negative for clockwise).
        It is useful for determining the orientation of three points and calculating angles.

    vec1 = b - a
    vec2 = c - b
    """
    a_x, a_y = a[:2]
    b_x, b_y = b[:2]
    c_x, c_y = c[:2]
    b_a_x = a_x - b_x
    b_a_y = a_y - b_y
    b_c_x = c_x - b_x
    b_c_y = c_y - b_y
    return b_a_x * b_c_y - b_a_y * b_c_x


def angle_between_lines2(point1: Point, point2: Point, point3: Point) -> float:
    """
    Given line1 as point1 and point2, and line2 as point2 and point3
    return the angle between two lines
    (point2 is the corner point)

    Args:
        point1 (Point): First point of the first line.
        point2 (Point): Second point of the first line and first point of the second line.
        point3 (Point): Second point of the second line.

    Returns:
        float: Angle between the two lines in radians.
    """
    return atan2(
        cross_product2(point1, point2, point3), dot_product2(point1, point2, point3)
    )


def angled_line(line: Line, theta: float) -> Line:
    """
    Given a line find another line with theta radians between them.

    Args:
        line (Line): Input line.
        theta (float): Angle in radians.

    Returns:
        Line: New line with the given angle.
    """
    # find the angle of the line
    x1, y1 = line[0]
    x2, y2 = line[1]
    theta1 = atan2(y2 - y1, x2 - x1)
    theta2 = theta1 + theta
    # find the length of the line
    dx = x2 - x1
    dy = y2 - y1
    length_ = (dx**2 + dy**2) ** 0.5
    # find the new line
    x3 = x1 + length_ * cos(theta2)
    y3 = y1 + length_ * sin(theta2)

    return [(x1, y1), (x3, y3)]


def angled_vector(angle: float) -> Sequence[float]:
    """
    Return a vector with the given angle

    Args:
        angle (float): Angle in radians.

    Returns:
        Sequence[float]: Vector with the given angle.
    """
    return [cos(angle), sin(angle)]


def close_points2(p1: Point, p2: Point, dist2: float = 0.01) -> bool:
    """
    Return True if two points are close to each other.

    Args:
        p1 (Point): First point.
        p2 (Point): Second point.
        dist2 (float, optional): Square of the threshold distance. Defaults to 0.01.

    Returns:
        bool: True if the points are close to each other, False otherwise.
    """
    return distance2(p1, p2) <= dist2


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


def distance(p1: Point, p2: Point) -> float:
    """
    Return the distance between two points.

    Args:
        p1 (Point): First point.
        p2 (Point): Second point.

    Returns:
        float: Distance between the two points.
    """
    return hypot(p2[0] - p1[0], p2[1] - p1[1])


def distance2(p1: Point, p2: Point) -> float:
    """
    Return the squared distance between two points.
    Useful for comparing distances without the need to
    compute the square root.

    Args:
        p1 (Point): First point.
        p2 (Point): Second point.

    Returns:
        float: Squared distance between the two points.
    """
    return (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2


def connect2(
    poly_point1: list[Point],
    poly_point2: list[Point],
    dist_tol: float = None,
    rtol: float = None,
) -> list[Point]:
    """
    Connect two polypoints together.

    Args:
        poly_point1 (list[Point]): First list of points.
        poly_point2 (list[Point]): Second list of points.
        dist_tol (float, optional): Distance tolerance. Defaults to None.
        rtol (float, optional): Relative tolerance. Defaults to None.

    Returns:
        list[Point]: Connected list of points.
    """
    rtol, dist_tol = get_defaults(["rtol", "dist_tol"], [rtol, dist_tol])
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


def stitch(
    lines: list[Line],
    closed: bool = True,
    return_points: bool = True,
    rtol: float = None,
    atol: float = None,
) -> list[Point]:
    """
    Stitches a list of lines together.

    Args:
        lines (list[Line]): List of lines to stitch.
        closed (bool, optional): Whether the lines form a closed shape. Defaults to True.
        return_points (bool, optional): Whether to return points or lines. Defaults to True.
        rtol (float, optional): Relative tolerance. Defaults to None.
        atol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        list[Point]: Stitched list of points or lines.
    """
    rtol, atol = get_defaults(["rtol", "atol"], [rtol, atol])
    if closed:
        points = []
    else:
        points = [lines[0][0]]
    for i, line in enumerate(lines[:-1]):
        x1, y1 = line[0]
        x2, y2 = line[1]
        x3, y3 = lines[i + 1][0]
        x4, y4 = lines[i + 1][1]
        x_point = intersect2(x1, y1, x2, y2, x3, y3, x4, y4)
        if x_point:
            points.append(x_point)
    if closed:
        x1, y1 = lines[-1][0]
        x2, y2 = lines[-1][1]
        x3, y3 = lines[0][0]
        x4, y4 = lines[0][1]
        final_x = intersect2(
            x1,
            y1,
            x2,
            y2,
            x3,
            y3,
            x4,
            y4,
        )
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


def double_offset_polylines(
    lines: list[Point], offset: float = 1, rtol: float = None, atol: float = None
) -> list[Point]:
    """
    Return a list of double offset lines from a list of lines.

    Args:
        lines (list[Point]): List of points representing the lines.
        offset (float, optional): Offset distance. Defaults to 1.
        rtol (float, optional): Relative tolerance. Defaults to None.
        atol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        list[Point]: List of double offset lines.
    """
    rtol, atol = get_defaults(["rtol", "atol"], [rtol, atol])
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


def polygon_cg(points: list[Point]) -> Point:
    """
    Given a list of points that define a polygon, return the center point.

    Args:
        points (list[Point]): List of points representing the polygon.

    Returns:
        Point: Center point of the polygon.
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


def polygon_center2(polygon_points: list[Point]) -> Point:
    """
    Given a list of points that define a polygon, return the center point.

    Args:
        polygon_points (list[Point]): List of points representing the polygon.

    Returns:
        Point: Center point of the polygon.
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


def polygon_center(polygon_points: list[Point]) -> Point:
    """
    Given a list of points that define a polygon, return the center point.

    Args:
        polygon_points (list[Point]): List of points representing the polygon.

    Returns:
        Point: Center point of the polygon.
    """
    x = 0
    y = 0
    for i, point in enumerate(polygon_points[:-1]):
        x += point[0] * (polygon_points[i - 1][1] - polygon_points[i + 1][1])
        y += point[1] * (polygon_points[i - 1][0] - polygon_points[i + 1][0])
    area_ = polygon_area(polygon_points)
    return (x / (6 * area_), y / (6 * area_))


def offset_polygon(
    polygon: list[Point], offset: float = -1, dist_tol: float = None
) -> list[Point]:
    """
    Return a list of offset lines from a list of lines.

    Args:
        polygon (list[Point]): List of points representing the polygon.
        offset (float, optional): Offset distance. Defaults to -1.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        list[Point]: List of offset lines.
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
    polygon: list[Point], offset: float = 1, dist_tol: float = None, **kwargs
) -> list[Point]:
    """
    Return a list of double offset lines from a list of lines.

    Args:
        polygon (list[Point]): List of points representing the polygon.
        offset (float, optional): Offset distance. Defaults to 1.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        list[Point]: List of double offset lines.
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
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
            from ..graphics.shape import Shape

            closed = close_points2(poly1[0], poly1[-1])
            canvas.draw(Shape(poly1, closed=closed), fill=False)
            closed = close_points2(poly2[0], poly2[-1])
            canvas.draw(Shape(poly2, closed=closed), fill=False)
    return [poly1, poly2]


def offset_polygon_points(
    polygon: list[Point], offset: float = 1, dist_tol: float = None
) -> list[Point]:
    """
    Return a list of double offset lines from a list of lines.

    Args:
        polygon (list[Point]): List of points representing the polygon.
        offset (float, optional): Offset distance. Defaults to 1.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        list[Point]: List of double offset lines.
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


def double_offset_lines(line: Line, offset: float = 1) -> tuple[Line, Line]:
    """
    Return two offset lines to a given line segment with the given offset amount.

    Args:
        line (Line): Input line segment.
        offset (float, optional): Offset distance. Defaults to 1.

    Returns:
        tuple[Line, Line]: Two offset lines.
    """
    line1 = offset_line(line, offset)
    line2 = offset_line(line, -offset)

    return line1, line2


def equal_lines(line1: Line, line2: Line, dist_tol: float = None) -> bool:
    """
    Return True if two lines are close enough.

    Args:
        line1 (Line): First line.
        line2 (Line): Second line.
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


def equal_polygons(
    poly1: Sequence[Point], poly2: Sequence[Point], dist_tol: float = None
) -> bool:
    """
    Return True if two polygons are close enough.

    Args:
        poly1 (Sequence[Point]): First polygon.
        poly2 (Sequence[Point]): Second polygon.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        bool: True if the polygons are close enough, False otherwise.
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    if len(poly1) != len(poly2):
        return False
    dist_tol2 = dist_tol * dist_tol
    for i, pnt in enumerate(poly1):
        if not close_points2(pnt, poly2[i], dist2=dist_tol2):
            return False
    return True


def extended_line(dist: float, line: Line, extend_both=False) -> Line:
    """
    Given a line ((x1, y1), (x2, y2)) and a distance,
    the given line is extended by distance units.
    Return a new line ((x1, y1), (x2', y2')).

    Args:
        dist (float): Distance to extend the line.
        line (Line): Input line.
        extend_both (bool, optional): Whether to extend both ends of the line. Defaults to False.

    Returns:
        Line: Extended line.
    """

    def extend(dist, line):
        # p = (1-t)*p1 + t*p2 : parametric equation of a line segment (p1, p2)
        line_length = length(line)
        t = (line_length + dist) / line_length
        p1, p2 = line
        x1, y1 = p1[:2]
        x2, y2 = p2[:2]
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
    point: Point, angle: float, length_: float, both_sides=False
) -> Line:
    """
    Return a line that passes through the given point
    with the given angle and length.
    If both_side is True, the line is extended on both sides by the given
    length.

    Args:
        point (Point): Point through which the line passes.
        angle (float): Angle of the line in radians.
        length_ (float): Length of the line.
        both_sides (bool, optional): Whether to extend the line on both sides. Defaults to False.

    Returns:
        Line: Line passing through the given point with the given angle and length.
    """
    x, y = point[:2]
    line = [(x, y), (x + length_ * cos(angle), y + length_ * sin(angle))]
    if both_sides:
        p1, p2 = line
        line = extended_line(length_, [p2, p1])

    return line


def remove_duplicate_points(points: list[Point], dist_tol=None) -> list[Point]:
    """
    Return a list of points with duplicate points removed.

    Args:
        points (list[Point]): List of points.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        list[Point]: List of points with duplicate points removed.
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
    points: list[Point], rtol: float = None, atol: float = None
) -> list[Point]:
    """
    Return a list of points with collinear points removed.

    Args:
        points (list[Point]): List of points.
        rtol (float, optional): Relative tolerance. Defaults to None.
        atol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        list[Point]: List of points with collinear points removed.
    """
    rtol, atol = get_defaults(["rtol", "atol"], [rtol, atol])
    new_points = []
    for i, point in enumerate(points):
        if i == 0:
            new_points.append(point)
        else:
            if not collinear(
                new_points[-1], point, points[(i + 1) % len(points)], rtol, atol
            ):
                new_points.append(point)
    return new_points


def fix_degen_points(
    points: list[Point],
    loop=False,
    closed=False,
    dist_tol: float = None,
    area_rtol: float = None,
    area_atol: float = None,
    check_collinear=True,
) -> list[Point]:
    """
    Return a list of points with duplicate points removed.
    Remove the middle point from the collinear points.

    Args:
        points (list[Point]): List of points.
        loop (bool, optional): Whether to loop the points. Defaults to False.
        closed (bool, optional): Whether the points form a closed shape. Defaults to False.
        dist_tol (float, optional): Distance tolerance. Defaults to None.
        area_rtol (float, optional): Relative tolerance for area. Defaults to None.
        area_atol (float, optional): Absolute tolerance for area. Defaults to None.
        check_collinear (bool, optional): Whether to check for collinear points. Defaults to True.

    Returns:
        list[Point]: List of points with duplicate and collinear points removed.
    """
    dist_tol, area_rtol, area_atol = get_defaults(
        ["dist_tol", "area_rtol", "area_atol"], [dist_tol, area_rtol, area_atol]
    )
    dist_tol2 = dist_tol * dist_tol
    new_points = []
    for i, point in enumerate(points):
        if i == 0:
            new_points.append(point)
        else:
            if not close_points2(point, new_points[-1], dist2=dist_tol2):
                new_points.append(point)
    if loop:
        if close_points2(new_points[0], new_points[-1], dist2=dist_tol2):
            new_points.pop(-1)

    if check_collinear:
        # Check for collinear points and remove the middle one.
        new_points = merge_consecutive_collinear_edges(
            new_points, closed, area_rtol, area_atol
        )

    return new_points


def clockwise(p: Point, q: Point, r: Point) -> bool:
    """Return 1 if the points p, q, and r are in clockwise order,
    return -1 if the points are in counter-clockwise order,
    return 0 if the points are collinear

    Args:
        p (Point): First point.
        q (Point): Second point.
        r (Point): Third point.

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
        seg1 (Line): First line segment.
        seg2 (Line): Second line segment.

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
    if o4 == 0 and between(p2, q1, q2):
        return True

    return False


def is_chained(seg1, seg2):
    """Checks if the line segments are chained together.

    Args:
        seg1 (Line): First line segment.
        seg2 (Line): Second line segment.

    Returns:
        bool: True if the segments are chained together, False otherwise.
    """
    p1, q1 = seg1
    p2, q2 = seg2
    if (
        close_points2(p1, p2)
        or close_points2(p1, q2)
        or close_points2(q1, p2)
        or close_points2(q1, q2)
    ):
        return True

    return False


def direction(p, q, r):
    """
    Checks the orientation of three points (p, q, r).

    Args:
        p (Point): First point.
        q (Point): Second point.
        r (Point): Third point.

    Returns:
        int: 0 if collinear, >0 if counter-clockwise, <0 if clockwise.
    """
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])


def collinear_segments(segment1, segment2, tol=None, atol=None):
    """
    Checks if two line segments (a1, b1) and (a2, b2) are collinear.

    Args:
        segment1 (Line): First line segment.
        segment2 (Line): Second line segment.
        tol (float, optional): Relative tolerance. Defaults to None.
        atol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        bool: True if the segments are collinear, False otherwise.
    """
    tol, atol = get_defaults(["tol", "atol"], [tol, atol])
    a1, b1 = segment1
    a2, b2 = segment2

    return isclose(direction(a1, b1, a2), 0, tol, atol) and isclose(
        direction(a1, b1, b2), 0, tol, atol
    )


def global_to_local(
    x: float, y: float, xi: float, yi: float, theta: float = 0
) -> Point:
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
        Point: Local coordinates (ksi, eta).
    """
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    ksi = (x - xi) * cos_theta + (y - yi) * sin_theta
    eta = (y - yi) * cos_theta - (x - xi) * sin_theta
    return (ksi, eta)


def stitch_lines(line1: Line, line2: Line) -> Sequence[Line]:
    """if the lines intersect, trim the lines
    if the lines don't intersect, extend the lines

    Args:
        line1 (Line): First line.
        line2 (Line): Second line.

    Returns:
        Sequence[Line]: Trimmed or extended lines.
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
    return int(floor((atan2(y, x) % (TWO_PI)) / (pi / 2)) + 1)


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


def homogenize(points: Sequence[Point]) -> 'ndarray':
    """
    Convert a list of points to homogeneous coordinates.

    Args:
        points (Sequence[Point]): List of points.

    Returns:
        np.ndarray: Homogeneous coordinates.
    """
    try:
        xy_array = np.array(points, dtype=float)
    except ValueError:
        xy_array = np.array([p[:2] for p in points], dtype=float)
    n_rows, n_cols = xy_array.shape
    if n_cols > 2:
        xy_array = xy_array[:, :2]
    ones = np.ones((n_rows, 1), dtype=float)
    homogeneous_array = np.append(xy_array, ones, axis=1)

    return homogeneous_array



def _homogenize(coordinates: Sequence[float]) -> 'ndarray':
    """Internal use only. API provides a homogenize function.
    Given a sequence of coordinates(x1, y1, x2, y2, ... xn, yn),
    return a numpy array of points array(((x1, y1, 1.),
    (x2, y2, 1.), ... (xn, yn, 1.))).

    Args:
        coordinates (Sequence[float]): Sequence of coordinates.

    Returns:
        np.ndarray: Homogeneous coordinates.
    """
    xy_array = np.array(list(zip(coordinates[0::2], coordinates[1::2])), dtype=float)
    n_rows = xy_array.shape[0]
    ones = np.ones((n_rows, 1), dtype=float)
    homogeneous_array = np.append(xy_array, ones, axis=1)

    return homogeneous_array


def intersect2(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
    rtol: float = None,
    atol: float = None,
) -> Point:
    """Return the intersection point of two lines.
    line1: (x1, y1), (x2, y2)
    line2: (x3, y3), (x4, y4)
    To find the intersection point of two line segments use the
    "intersection" function

    Args:
        x1 (float): x-coordinate of the first point of the first line.
        y1 (float): y-coordinate of the first point of the first line.
        x2 (float): x-coordinate of the second point of the first line.
        y2 (float): y-coordinate of the second point of the first line.
        x3 (float): x-coordinate of the first point of the second line.
        y3 (float): y-coordinate of the first point of the second line.
        x4 (float): x-coordinate of the second point of the second line.
        y4 (float): y-coordinate of the second point of the second line.
        rtol (float, optional): Relative tolerance. Defaults to None.
        atol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        Point: Intersection point of the two lines.
    """
    rtol, atol = get_defaults(["rtol", "atol"], [rtol, atol])
    x1_x2 = x1 - x2
    y1_y2 = y1 - y2
    x3_x4 = x3 - x4
    y3_y4 = y3 - y4

    denom = (x1_x2) * (y3_y4) - (y1_y2) * (x3_x4)
    if isclose(denom, 0, rtol=rtol, atol=atol):
        res = None  # parallel lines
    else:
        x = ((x1 * y2 - y1 * x2) * (x3_x4) - (x1_x2) * (x3 * y4 - y3 * x4)) / denom
        y = ((x1 * y2 - y1 * x2) * (y3_y4) - (y1_y2) * (x3 * y4 - y3 * x4)) / denom
        res = (x, y)

    return res


def intersect(line1: Line, line2: Line) -> Point:
    """Return the intersection point of two lines.
    line1: [(x1, y1), (x2, y2)]
    line2: [(x3, y3), (x4, y4)]
    To find the intersection point of two line segments use the
    "intersection" function

    Args:
        line1 (Line): First line.
        line2 (Line): Second line.

    Returns:
        Point: Intersection point of the two lines.
    """
    x1, y1 = line1[0][:2]
    x2, y2 = line1[1][:2]
    x3, y3 = line2[0][:2]
    x4, y4 = line2[1][:2]
    return intersect2(x1, y1, x2, y2, x3, y3, x4, y4)


def intersection2(x1, y1, x2, y2, x3, y3, x4, y4, rtol=None, atol=None):
    """Check the intersection of two line segments. See the documentation

    Args:
        x1 (float): x-coordinate of the first point of the first line segment.
        y1 (float): y-coordinate of the first point of the first line segment.
        x2 (float): x-coordinate of the second point of the first line segment.
        y2 (float): y-coordinate of the second point of the first line segment.
        x3 (float): x-coordinate of the first point of the second line segment.
        y3 (float): y-coordinate of the first point of the second line segment.
        x4 (float): x-coordinate of the second point of the second line segment.
        y4 (float): y-coordinate of the second point of the second line segment.
        rtol (float, optional): Relative tolerance. Defaults to None.
        atol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        tuple: Connection type and intersection point.
    """
    rtol, atol = get_defaults(["rtol", "atol"], [rtol, atol])
    x2_x1 = x2 - x1
    y2_y1 = y2 - y1
    x4_x3 = x4 - x3
    y4_y3 = y4 - y3
    denom = (y4_y3) * (x2_x1) - (x4_x3) * (y2_y1)
    if isclose(denom, 0, rtol=rtol, atol=atol):  # parallel
        return Connection.PARALLEL, None
    x1_x3 = x1 - x3
    y1_y3 = y1 - y3
    ua = ((x4_x3) * (y1_y3) - (y4_y3) * (x1_x3)) / denom
    if ua < 0 or ua > 1:
        return Connection.DISJOINT, None
    ub = ((x2_x1) * (y1_y3) - (y2_y1) * (x1_x3)) / denom
    if ub < 0 or ub > 1:
        return Connection.DISJOINT, None
    x = x1 + ua * (x2_x1)
    y = y1 + ua * (y2_y1)
    return Connection.INTERSECT, (x, y)


def intersection3(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
    rtol: float = None,
    atol: float = None,
    dist_tol: float = None,
    area_atol: float = None,
) -> tuple[Connection, list]:
    """Check the intersection of two line segments. See the documentation
    for more details.

    Args:
        x1 (float): x-coordinate of the first point of the first line segment.
        y1 (float): y-coordinate of the first point of the first line segment.
        x2 (float): x-coordinate of the second point of the first line segment.
        y2 (float): y-coordinate of the second point of the first line segment.
        x3 (float): x-coordinate of the first point of the second line segment.
        y3 (float): y-coordinate of the first point of the second line segment.
        x4 (float): x-coordinate of the second point of the second line segment.
        y4 (float): y-coordinate of the second point of the second line segment.
        rtol (float, optional): Relative tolerance. Defaults to None.
        atol (float, optional): Absolute tolerance. Defaults to None.
        dist_tol (float, optional): Distance tolerance. Defaults to None.
        area_atol (float, optional): Absolute tolerance for area. Defaults to None.

    Returns:
        tuple: Connection type and intersection result.
    """
    # collinear check uses area_atol

    # s1: start1 = (x1, y1)
    # e1: end1 = (x2, y2)
    # s2: start2 = (x3, y3)
    # e2: end2 = (x4, y4)
    # s1s2: start1 and start2 is connected
    # s1e2: start1 and end2 is connected
    # e1s2: end1 and start2 is connected
    # e1e2: end1 and end2 is connected
    rtol, atol, dist_tol, area_atol = get_defaults(
        ["rtol", "atol", "dist_tol", "area_atol"], [rtol, atol, dist_tol, area_atol]
    )

    s1 = (x1, y1)
    e1 = (x2, y2)
    s2 = (x3, y3)
    e2 = (x4, y4)
    segment1 = [(x1, y1), (x2, y2)]
    segment2 = [(x3, y3), (x4, y4)]

    # check if the segments' bounding boxes overlap
    if not line_segment_bbox_check(segment1, segment2):
        return (Connection.DISJOINT, None)

    # Check if the segments are parallel
    x2_x1 = x2 - x1
    y2_y1 = y2 - y1
    x4_x3 = x4 - x3
    y4_y3 = y4 - y3
    denom = (y4_y3) * (x2_x1) - (x4_x3) * (y2_y1)
    parallel = isclose(denom, 0, rtol=rtol, atol=atol)
    # angle1 = atan2(y2 - y1, x2 - x1) % pi
    # angle2 = atan2(y4 - y3, x4 - x3) % pi
    # parallel = close_angles(angle1, angle2, angtol=defaults['angtol'])

    # Coincident end points
    dist_tol2 = dist_tol * dist_tol
    s1s2 = close_points2(s1, s2, dist2=dist_tol2)
    s1e2 = close_points2(s1, e2, dist2=dist_tol2)
    e1s2 = close_points2(e1, s2, dist2=dist_tol2)
    e1e2 = close_points2(e1, e2, dist2=dist_tol2)
    connected = s1s2 or s1e2 or e1s2 or e1e2
    if parallel:
        length1 = distance((x1, y1), (x2, y2))
        length2 = distance((x3, y3), (x4, y4))
        min_x = min(x1, x2, x3, x4)
        max_x = max(x1, x2, x3, x4)
        min_y = min(y1, y2, y3, y4)
        max_y = max(y1, y2, y3, y4)
        total_length = distance((min_x, min_y), (max_x, max_y))
        l1_eq_l2 = isclose(length1, length2, rtol=rtol, atol=atol)
        l1_eq_total = isclose(length1, total_length, rtol=rtol, atol=atol)
        l2_eq_total = isclose(length2, total_length, rtol=rtol, atol=atol)
        if connected:
            if l1_eq_l2 and l1_eq_total:
                return Connection.CONGRUENT, segment1

            if l1_eq_total:
                return Connection.CONTAINS, segment1
            if l2_eq_total:
                return Connection.WITHIN, segment2
            if isclose(length1 + length2, total_length, rtol, atol):
                # chained and collienar
                if s1s2:
                    return Connection.COLL_CHAIN, (e1, s1, e2)
                if s1e2:
                    return Connection.COLL_CHAIN, (e1, s1, s2)
                if e1s2:
                    return Connection.COLL_CHAIN, (s1, s2, e2)
                if e1e2:
                    return Connection.COLL_CHAIN, (s1, e1, s2)
        else:
            if total_length < length1 + length2 and collinear_segments(
                segment1, segment2, atol
            ):
                p1 = (min_x, min_y)
                p2 = (max_x, max_y)
                seg = [p1, p2]
                return Connection.OVERLAPS, seg

            return intersection2(x1, y1, x2, y2, x3, y3, x4, y4, rtol, atol)
    else:
        if connected:
            if s1s2:
                return Connection.CHAIN, (e1, s1, e2)
            if s1e2:
                return Connection.CHAIN, (e1, s1, s2)
            if e1s2:
                return Connection.CHAIN, (s1, s2, e2)
            if e1e2:
                return Connection.CHAIN, (s1, e1, s2)
        else:
            if between(s1, e1, e2):
                return Connection.YJOINT, e1
            if between(s1, e1, s2):
                return Connection.YJOINT, s1
            if between(s2, e2, e1):
                return Connection.YJOINT, e2
            if between(s2, e2, s1):
                return Connection.YJOINT, s2

            return intersection2(x1, y1, x2, y2, x3, y3, x4, y4, rtol, atol)
    return (Connection.DISJOINT, None)


def merge_consecutive_collinear_edges(
    points, closed=False, area_rtol=None, area_atol=None
):
    """Remove the middle points from collinear edges.

    Args:
        points (list[Point]): List of points.
        closed (bool, optional): Whether the points form a closed shape. Defaults to False.
        area_rtol (float, optional): Relative tolerance for area. Defaults to None.
        area_atol (float, optional): Absolute tolerance for area. Defaults to None.

    Returns:
        list[Point]: List of points with collinear points removed.
    """
    area_rtol, area_atol = get_defaults(
        ["area_rtol", "area_atol"], [area_rtol, area_atol]
    )
    points = points[:]

    while True:
        cyc = cycle(points)
        a = next(cyc)
        b = next(cyc)
        c = next(cyc)
        looping = False
        n = len(points) - 1
        if closed:
            n += 1
        discarded = []
        for _ in range(n - 1):
            if collinear(a, b, c, area_rtol=area_rtol, area_atol=area_atol):
                discarded.append(b)
                looping = True
                break
            a = b
            b = c
            c = next(cyc)
        for point in discarded:
            points.remove(point)
        if not looping or len(points) < 3:
            break

    return points


def intersection(line1: Line, line2: Line, rtol: float = None) -> int:
    """return the intersection point of two line segments.
    segment1: ((x1, y1), (x2, y2))
    segment2: ((x3, y3), (x4, y4))
    if line segments do not intersect return -1
    if line segments are parallel return 0
    if line segments are connected (share a point) return 1
    To find the intersection point of two lines use the "intersect" function

    Args:
        line1 (Line): First line segment.
        line2 (Line): Second line segment.
        rtol (float, optional): Relative tolerance. Defaults to None.

    Returns:
        int: Intersection type.
    """
    if rtol is None:
        rtol = defaults["rtol"]
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    return intersection2(x1, y1, x2, y2, x3, y3, x4, y4)


def merge_segments(seg1: Sequence[Point], seg2: Sequence[Point]) -> Sequence[Point]:
    """Merge two segments into one segment if they are connected.
    They need to be overlapping or simply connected to each other,
    otherwise they will not be merged. Order doesn't matter.

    Args:
        seg1 (Sequence[Point]): First segment.
        seg2 (Sequence[Point]): Second segment.

    Returns:
        Sequence[Point]: Merged segment.
    """
    Conn = Connection
    p1, p2 = seg1
    p3, p4 = seg2

    res = all_intersections([(p1, p2), (p3, p4)], use_intersection3=True)
    if res:
        conn_type = list(res.values())[0][0][0]
        verts = list(res.values())[0][0][1]
        if conn_type in [Conn.OVERLAPS, Conn.CONGRUENT, Conn.CHAIN]:
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
        p (Point): Point to invert.
        center (Point): Center of the circle.
        radius (float): Radius of the circle.

    Returns:
        Point: Inverted point.
    """
    dist = distance(p, center)
    if dist == 0:
        return p
    p = np.array(p)
    center = np.array(center)
    return center + (radius**2 / dist**2) * (p - center)
    # return radius**2 * (p - center) / dist


def is_horizontal(line: Line, eps: float = 0.0001) -> bool:
    """Return True if the line is horizontal.

    Args:
        line (Line): Input line.
        eps (float, optional): Tolerance. Defaults to 0.0001.

    Returns:
        bool: True if the line is horizontal, False otherwise.
    """
    return abs(j_vec.dot(line_vector(line))) <= eps


def is_line(line_: Any) -> bool:
    """Return True if the input is a line.

    Args:
        line_ (Any): Input value.

    Returns:
        bool: True if the input is a line, False otherwise.
    """
    try:
        p1, p2 = line_
        return is_point(p1) and is_point(p2)
    except:
        return False


def is_point(pnt: Any) -> bool:
    """Return True if the input is a point.

    Args:
        pnt (Any): Input value.

    Returns:
        bool: True if the input is a point, False otherwise.
    """
    try:
        x, y = pnt[:2]
        return is_number(x) and is_number(y)
    except:
        return False


def is_vertical(line: Line, eps: float = 0.0001) -> bool:
    """Return True if the line is vertical.

    Args:
        line (Line): Input line.
        eps (float, optional): Tolerance. Defaults to 0.0001.

    Returns:
        bool: True if the line is vertical, False otherwise.
    """
    return abs(i_vec.dot(line_vector(line))) <= eps


def length(line: Line) -> float:
    """Return the length of a line.

    Args:
        line (Line): Input line.

    Returns:
        float: Length of the line.
    """
    p1, p2 = line
    return distance(p1, p2)


def lerp_point(p1: Point, p2: Point, t: float) -> Point:
    """Linear interpolation of two points.

    Args:
        p1 (Point): First point.
        p2 (Point): Second point.
        t (float): Interpolation parameter. t = 0 => p1, t = 1 => p2.

    Returns:
        Point: Interpolated point.
    """
    x1, y1 = p1
    x2, y2 = p2
    return (lerp(x1, x2, t), lerp(y1, y2, t))


def slope(start_point: Point, end_point: Point, rtol=None, atol=None) -> float:
    """Return the slope of a line given by two points.
    Order makes a difference.

    Args:
        start_point (Point): Start point of the line.
        end_point (Point): End point of the line.
        rtol (float, optional): Relative tolerance. Defaults to None.
        atol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        float: Slope of the line.
    """
    rtol, atol = get_defaults(["rtol", "atol"], [rtol, atol])
    x1, y1 = start_point[:2]
    x2, y2 = end_point[:2]
    if isclose(x1, x2, rtol=rtol, atol=atol):
        res = defaults["INF"]
    else:
        res = (y2 - y1) / (x2 - x1)

    return res


def segmentize_line(line: Line, segment_length: float) -> list[Line]:
    """Return a list of points that would form segments with the given length.

    Args:
        line (Line): Input line.
        segment_length (float): Length of each segment.

    Returns:
        list[Line]: List of segments.
    """
    length_ = distance(line[0], line[1])
    x1, y1 = line[0]
    x2, y2 = line[1]
    increments = int(length_ / segment_length)
    x_segments = np.linspace(x1, x2, increments)
    y_segments = np.linspace(y1, y2, increments)

    return list(zip(x_segments, y_segments))


def line_angle(start_point: Point, end_point: Point) -> float:
    """Return the orientation angle (in radians) of a line given by start and end points.
    Order makes a difference.

    Args:
        start_point (Point): Start point of the line.
        end_point (Point): End point of the line.

    Returns:
        float: Orientation angle of the line in radians.
    """
    return atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])


def inclination_angle(start_point: Point, end_point: Point) -> float:
    """Return the inclination angle (in radians) of a line given by start and end points.
    Inclination angle is always between zero and pi.
    Order makes no difference.

    Args:
        start_point (Point): Start point of the line.
        end_point (Point): End point of the line.

    Returns:
        float: Inclination angle of the line in radians.
    """
    return line_angle(start_point, end_point) % pi


def line2vector(line: Line) -> VecType:
    """Return the vector representation of a line

    Args:
        line (Line): Input line.

    Returns:
        VecType: Vector representation of the line.
    """
    x1, y1 = line[0]
    x2, y2 = line[1]
    dx = x2 - x1
    dy = y2 - y1
    return [dx, dy]


def line_through_point_and_angle(
    point: Point, angle: float, length_: float = 100
) -> Line:
    """Return a line through the given point with the given angle and length

    Args:
        point (Point): Point through which the line passes.
        angle (float): Angle of the line in radians.
        length_ (float, optional): Length of the line. Defaults to 100.

    Returns:
        Line: Line passing through the given point with the given angle and length.
    """
    x, y = point[:2]
    dx = length_ * cos(angle)
    dy = length_ * sin(angle)
    return [[x, y], [x + dx, y + dy]]


def line_vector(line: Line) -> VecType:
    """Return the vector representation of a line.

    Args:
        line (Line): Input line.

    Returns:
        VecType: Vector representation of the line.
    """
    x1, y1 = line[0]
    x2, y2 = line[1]
    return Vector2D(x2 - x1, y2 - y1)


def mid_point(p1: Point, p2: Point) -> Point:
    """Return the mid point of two points.

    Args:
        p1 (Point): First point.
        p2 (Point): Second point.

    Returns:
        Point: Mid point of the two points.
    """
    x = (p2[0] + p1[0]) / 2
    y = (p2[1] + p1[1]) / 2
    return (x, y)


def norm(vec: VecType) -> float:
    """Return the norm (vector length) of a vector.

    Args:
        vec (VecType): Input vector.

    Returns:
        float: Norm of the vector.
    """
    return hypot(vec[0], vec[1])


def ndarray_to_xy_list(arr: 'ndarray') -> Sequence[Point]:
    """Convert a numpy array to a list of points.

    Args:
        arr (np.ndarray): Input numpy array.

    Returns:
        Sequence[Point]: List of points.
    """
    return arr[:, :2].tolist()


def offset_line(line: Sequence[Point], offset: float) -> Sequence[Point]:
    """Return an offset line from a given line.

    Args:
        line (Sequence[Point]): Input line.
        offset (float): Offset distance.

    Returns:
        Sequence[Point]: Offset line.
    """
    unit_vec = perp_unit_vector(line)
    dx = unit_vec[0] * offset
    dy = unit_vec[1] * offset
    x1, y1 = line[0]
    x2, y2 = line[1]
    return [[x1 + dx, y1 + dy], [x2 + dx, y2 + dy]]


def offset_lines(polylines: Sequence[Line], offset: float = 1) -> list[Line]:
    """Return a list of offset lines from a list of lines.

    Args:
        polylines (Sequence[Line]): List of input lines.
        offset (float, optional): Offset distance. Defaults to 1.

    Returns:
        list[Line]: List of offset lines.
    """

    def stitch_(polyline):
        res = []
        line1 = polyline[0]
        for i, _ in enumerate(polyline):
            if i == len(polyline) - 1:
                break
            line2 = polyline[i + 1]
            line1, line2 = stitch_lines(line1, line2)
            res.extend(line1)
            line1 = line2
        res.append(line2[-1])
        return res

    poly = []
    for line in polylines:
        poly.append(offset_line(line, offset))
    poly = stitch_(poly)
    return poly


def normalize(vec: VecType) -> VecType:
    """Return the normalized vector.

    Args:
        vec (VecType): Input vector.

    Returns:
        VecType: Normalized vector.
    """
    norm_ = norm(vec)
    return [vec[0] / norm_, vec[1] / norm_]


def offset_point_on_line(point: Point, line: Line, offset: float) -> Point:
    """Return a point on a line that is offset from the given point.

    Args:
        point (Point): Input point.
        line (Line): Input line.
        offset (float): Offset distance.

    Returns:
        Point: Offset point on the line.
    """
    x, y = point[:2]
    x1, y1 = line[0]
    x2, y2 = line[1]
    dx = x2 - x1
    dy = y2 - y1
    # normalize the vector
    mag = (dx * dx + dy * dy) ** 0.5
    dx = dx / mag
    dy = dy / mag
    return x + dx * offset, y + dy * offset


def offset_point(point: Point, dx: float = 0, dy: float = 0) -> Point:
    """Return an offset point from a given point.

    Args:
        point (Point): Input point.
        dx (float, optional): Offset distance in x-direction. Defaults to 0.
        dy (float, optional): Offset distance in y-direction. Defaults to 0.

    Returns:
        Point: Offset point.
    """
    x, y = point[:2]
    return x + dx, y + dy


def parallel_line(line: Line, point: Point) -> Line:
    """Return a parallel line to the given line that goes through the given point

    Args:
        line (Line): Input line.
        point (Point): Point through which the parallel line passes.

    Returns:
        Line: Parallel line.
    """
    x1, y1 = line[0]
    x2, y2 = line[1]
    x3, y3 = point
    dx = x2 - x1
    dy = y2 - y1
    return [[x3, y3], [x3 + dx, y3 + dy]]


def perp_offset_point(point: Point, line: Line, offset: float) -> Point:
    """Return a point that is offset from the given point in the perpendicular direction to the given line.

    Args:
        point (Point): Input point.
        line (Line): Input line.
        offset (float): Offset distance.

    Returns:
        Point: Perpendicular offset point.
    """
    unit_vec = perp_unit_vector(line)
    dx = unit_vec[0] * offset
    dy = unit_vec[1] * offset
    x, y = point[:2]
    return [x + dx, y + dy]


def perp_unit_vector(line: Line) -> VecType:
    """Return the perpendicular unit vector to a line

    Args:
        line (Line): Input line.

    Returns:
        VecType: Perpendicular unit vector.
    """
    x1, y1 = line[0]
    x2, y2 = line[1]
    dx = x2 - x1
    dy = y2 - y1
    norm_ = sqrt(dx**2 + dy**2)
    return [-dy / norm_, dx / norm_]


def point_on_line(
    point: Point, line: Line, rtol: float = None, atol: float = None
) -> bool:
    """Return True if the given point is on the given line

    Args:
        point (Point): Input point.
        line (Line): Input line.
        rtol (float, optional): Relative tolerance. Defaults to None.
        atol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        bool: True if the point is on the line, False otherwise.
    """
    rtol, atol = get_defaults(["rtol", "atol"], [rtol, atol])
    p1, p2 = line
    return isclose(slope(p1, point), slope(point, p2), rtol=rtol, atol=atol)


def point_on_line_segment(
    point: Point, line: Line, rtol: float = None, atol: float = None
) -> bool:
    """Return True if the given point is on the given line segment

    Args:
        point (Point): Input point.
        line (Line): Input line segment.
        rtol (float, optional): Relative tolerance. Defaults to None.
        atol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        bool: True if the point is on the line segment, False otherwise.
    """
    rtol, atol = get_defaults(["rtol", "atol"], [rtol, atol])
    p1, p2 = line
    return isclose(
        (distance(p1, point) + distance(p2, point)),
        distance(p1, p2),
        rtol=rtol,
        atol=atol,
    )


def point_to_line_distance(point: Point, line: Line) -> float:
    """Return the vector from a point to a line

    Args:
        point (Point): Input point.
        line (Line): Input line.

    Returns:
        float: Distance from the point to the line.
    """
    x0, y0 = point
    x1, y1 = line[0]
    x2, y2 = line[1]
    dx = x2 - x1
    dy = y2 - y1
    return abs((dx * (y1 - y0) - (x1 - x0) * dy)) / sqrt(dx**2 + dy**2)


def point_to_line_seg_distance(p, lp1, lp2):
    """Given a point p and a line segment defined by boundary points
    lp1 and lp2, returns the distance between the line segment and the point.
    If the point is not located in the perpendicular area between the
    boundary points, returns False.

    Args:
        p (Point): Input point.
        lp1 (Point): First boundary point of the line segment.
        lp2 (Point): Second boundary point of the line segment.

    Returns:
        float: Distance between the point and the line segment, or False if the point is not in the perpendicular area.
    """
    if lp1[:2] == lp2[:2]:
        msg = "Error! Line is ill defined. Start and end points are coincident."
        raise ValueError(msg)
    x3, y3 = p[:2]
    x1, y1 = lp1[:2]
    x2, y2 = lp2[:2]

    u = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / distance(lp1, lp2) ** 2
    if 0 <= u <= 1:
        x = x1 + u * (x2 - x1)
        y = y1 + u * (y2 - y1)
        res = distance((x, y), p)
    else:
        res = False  # p is not between lp1 and lp2

    return res

def rotate_point(point: Point, angle: float, center: Point = (0, 0)) -> Point:
    x, y = point[:2]
    cx, cy = center[:2]
    x -= cx
    y -= cy
    cos_angle = cos(angle)
    sin_angle = sin(angle)
    x, y = x*cos_angle - y*sin_angle, x*sin_angle + y*cos_angle
    x += cx
    y += cy

    return (x, y)


def point_to_line_vec(point: Point, line: Line, unit: bool = False) -> VecType:
    """Return the perpendicular vector from a point to a line

    Args:
        point (Point): Input point.
        line (Line): Input line.
        unit (bool, optional): Whether to return a unit vector. Defaults to False.

    Returns:
        VecType: Perpendicular vector from the point to the line.
    """
    x0, y0 = point
    x1, y1 = line[0]
    x2, y2 = line[1]
    dx = x2 - x1
    dy = y2 - y1
    norm_ = sqrt(dx**2 + dy**2)
    unit_vec = [-dy / norm_, dx / norm_]
    dist = (dx * (y1 - y0) - (x1 - x0) * dy) / sqrt(dx**2 + dy**2)
    if unit:
        if dist > 0:
            res = [unit_vec[0], unit_vec[1]]
        else:
            res = [-unit_vec[0], -unit_vec[1]]
    else:
        res = [unit_vec[0] * dist, unit_vec[1] * dist]

    return res


def polygon_area(polygon: Sequence[Point], dist_tol=None) -> float:
    """Calculate the area of a polygon.

    Args:
        polygon (Sequence[Point]): List of points representing the polygon.
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
        x1, y1 = point
        x2, y2 = polygon[i + 1]
        area_ += x1 * y2 - x2 * y1
    return area_ / 2


def polyline_length(polygon: Sequence[Point], closed=False, dist_tol=None) -> float:
    """Calculate the perimeter of a polygon.

    Args:
        polygon (Sequence[Point]): List of points representing the polygon.
        closed (bool, optional): Whether the polygon is closed. Defaults to False.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        float: Perimeter of the polygon.
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    dist_tol2 = dist_tol * dist_tol
    if closed:
        if not close_points2(polygon[0], polygon[-1], dist2=dist_tol2):
            polygon = polygon[:]
            polygon.append(polygon[0])
    perimeter = 0
    for i, point in enumerate(polygon[:-1]):
        perimeter += distance(point, polygon[i + 1])
    return perimeter


def right_handed(polygon: Sequence[Point], dist_tol=None) -> float:
    """If polygon is counter-clockwise, return True

    Args:
        polygon (Sequence[Point]): List of points representing the polygon.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        bool: True if the polygon is counter-clockwise, False otherwise.
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    dist_tol2 = dist_tol * dist_tol
    added_point = False
    if not close_points2(polygon[0], polygon[-1], dist2=dist_tol2):
        polygon.append(polygon[0])
        added_point = True
    area_ = 0
    for i, point in enumerate(polygon[:-1]):
        x1, y1 = point
        x2, y2 = polygon[i + 1]
        area_ += x1 * y2 - x2 * y1
    if added_point:
        polygon.pop()
    return area_ > 0


def radius2side_len(n: int, radius: float) -> float:
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


def translate_line(dx: float, dy: float, line: Line) -> Line:
    """Return a translated line by dx and dy

    Args:
        dx (float): Translation distance in x-direction.
        dy (float): Translation distance in y-direction.
        line (Line): Input line.

    Returns:
        Line: Translated line.
    """
    x1, y1 = line[0]
    x2, y2 = line[1]
    return [[x1 + dx, y1 + dy], [x2 + dx, y2 + dy]]


def trim_line(line1: Line, line2: Line) -> Line:
    """Trim line1 to the intersection of line1 and line2.
    Extend it if necessary.

    Args:
        line1 (Line): First line.
        line2 (Line): Second line.

    Returns:
        Line: Trimmed line.
    """
    intersection_ = intersection(line1, line2)
    return [line1[0], intersection_]


def unit_vector(line: Line) -> VecType:
    """Return the unit vector of a line

    Args:
        line (Line): Input line.

    Returns:
        VecType: Unit vector of the line.
    """
    norm_ = length(line)
    p1, p2 = line
    x1, y1 = p1
    x2, y2 = p2
    return [(x2 - x1) / norm_, (y2 - y1) / norm_]


def unit_vector_(line: Line) -> Sequence[VecType]:
    """Return the cartesian unit vector of a line
    with the given line's start and end points

    Args:
        line (Line): Input line.

    Returns:
        Sequence[VecType]: Cartesian unit vector of the line.
    """
    x1, y1 = line[0]
    x2, y2 = line[1]
    dx = x2 - x1
    dy = y2 - y1
    norm_ = sqrt(dx**2 + dy**2)
    return [dx / norm_, dy / norm_]


def vec_along_line(line: Line, magnitude: float) -> VecType:
    """Return a vector along a line with the given magnitude.

    Args:
        line (Line): Input line.
        magnitude (float): Magnitude of the vector.

    Returns:
        VecType: Vector along the line with the given magnitude.
    """
    if line == axis_x:
        dx, dy = magnitude, 0
    elif line == axis_y:
        dx, dy = 0, magnitude
    else:
        # line is (p1, p2)
        theta = line_angle(*line)
        dx = magnitude * cos(theta)
        dy = magnitude * sin(theta)
    return dx, dy


def vec_dir_angle(vec: Sequence[float]) -> float:
    """Return the direction angle of a vector

    Args:
        vec (Sequence[float]): Input vector.

    Returns:
        float: Direction angle of the vector.
    """
    return atan2(vec[1], vec[0])


def cross_product_sense(a: Point, b: Point, c: Point) -> int:
    """Return the cross product sense of vectors a and b.

    Args:
        a (Point): First point.
        b (Point): Second point.
        c (Point): Third point.

    Returns:
        int: Cross product sense.
    """
    length_ = cross_product2(a, b, c)
    if length_ == 0:
        res = 1
    else:
        res = length_ / abs(length)

    return res


#      A
#      /
#     /
#   B/
#    \
#     \
#      \
#       C


def right_turn(p1, p2, p3):
    """Return True if p1, p2, p3 make a right turn.

    Args:
        p1 (Point): First point.
        p2 (Point): Second point.
        p3 (Point): Third point.

    Returns:
        bool: True if the points make a right turn, False otherwise.
    """
    return cross(p1, p2, p3) < 0


def left_turn(p1, p2, p3):
    """Return True if p1, p2, p3 make a left turn.

    Args:
        p1 (Point): First point.
        p2 (Point): Second point.
        p3 (Point): Third point.

    Returns:
        bool: True if the points make a left turn, False otherwise.
    """
    return cross(p1, p2, p3) > 0


def cross(p1, p2, p3):
    """Return the cross product of vectors p1p2 and p1p3.

    Args:
        p1 (Point): First point.
        p2 (Point): Second point.
        p3 (Point): Third point.

    Returns:
        float: Cross product of the vectors.
    """
    x1, y1 = p2[0] - p1[0], p2[1] - p1[1]
    x2, y2 = p3[0] - p1[0], p3[1] - p1[1]
    return x1 * y2 - x2 * y1


def tri_to_cart(points):
    """
    Convert a list of points from triangular to cartesian coordinates.

    Args:
        points (list[Point]): List of points in triangular coordinates.

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
        points (list[Point]): List of points in cartesian coordinates.

    Returns:
        np.ndarray: List of points in triangular coordinates.
    """
    u = [1, 0]
    v = cos(pi / 3), sin(pi / 3)
    convert = np.linalg.inv(array([u, v]))

    return array(points) @ convert


def convex_hull(points):
    """Return the convex hull of a set of 2D points.

    Args:
        points (list[Point]): List of 2D points.

    Returns:
        list[Point]: Convex hull of the points.
    """
    # From http://en.wikibooks.org/wiki/Algorithm__implementation/Geometry/
    # Convex_hull/Monotone_chain
    # Sort points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))
    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross
    # product.
    # Return a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross_(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross_(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning
    # of the other list.
    return lower[:-1] + upper[:-1]


def connected_pairs(items):
    """Return a list of connected pair tuples corresponding to the items.
    [a, b, c] -> [(a, b), (b, c)]

    Args:
        items (list): List of items.

    Returns:
        list[tuple]: List of connected pair tuples.
    """
    return list(zip(items, items[1:]))


def flat_points(connected_segments):
    """Return a list of points from a list of connected pairs of points.

    Args:
        connected_segments (list[tuple]): List of connected pairs of points.

    Returns:
        list[Point]: List of points.
    """
    points = [line[0] for line in connected_segments]
    points.append(connected_segments[-1][1])
    return points


def point_in_quad(point: Point, quad: list[Point]) -> bool:
    """Return True if the point is inside the quad.

    Args:
        point (Point): Input point.
        quad (list[Point]): List of points representing the quad.

    Returns:
        bool: True if the point is inside the quad, False otherwise.
    """
    x, y = point[:2]
    x1, y1 = quad[0]
    x2, y2 = quad[1]
    x3, y3 = quad[2]
    x4, y4 = quad[3]
    xs = [x1, x2, x3, x4]
    ys = [y1, y2, y3, y4]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    return min_x <= x <= max_x and min_y <= y <= max_y


def get_polygons(
    nested_points: Sequence[Point], n_round_digits: int = 2, dist_tol: float = None
) -> list:
    """Convert points to clean polygons. Points are vertices of polygons.

    Args:
        nested_points (Sequence[Point]): List of nested points.
        n_round_digits (int, optional): Number of decimal places to round to. Defaults to 2.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        list: List of clean polygons.
    """
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    from ..graph import get_cycles

    nested_rounded_points = []
    for points in nested_points:
        rounded_points = []
        for point in points:
            rounded_point = (around(point, n_round_digits)).tolist()
            rounded_points.append(tuple(rounded_point))
        nested_rounded_points.append(rounded_points)

    s_points = set()
    d_id__point = {}
    d_point__id = {}
    for points in nested_rounded_points:
        for point in points:
            s_points.add(point)

    for i, fs_point in enumerate(s_points):
        d_id__point[i] = fs_point  # we need a bidirectional dictionary
        d_point__id[fs_point] = i

    nested_point_ids = []
    for points in nested_rounded_points:
        point_ids = []
        for point in points:
            point_ids.append(d_point__id[point])
        nested_point_ids.append(point_ids)

    graph_edges = []
    for point_ids in nested_point_ids:
        graph_edges.extend(connected_pairs(point_ids))
    polygons = []
    graph_edges = sanitize_graph_edges(graph_edges)
    cycles = get_cycles(graph_edges)
    if cycles is None:
        return []
    for cycle_ in cycles:
        nodes = cycle_
        points = [d_id__point[i] for i in nodes]
        points = fix_degen_points(points, closed=True, dist_tol=dist_tol)
        polygons.append(points)

    return polygons


def offset_point_from_start(p1, p2, offset):
    """p1, p2: points on a line
    offset: distance from p1
    return the point on the line at the given offset

    Args:
        p1 (Point): First point on the line.
        p2 (Point): Second point on the line.
        offset (float): Distance from p1.

    Returns:
        Point: Point on the line at the given offset.
    """
    x1, y1 = p1
    x2, y2 = p2
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
        line1 (Line): First line.
        line2 (Line): Second line.

    Returns:
        float: Angle between the two lines in radians.
    """
    alpha1 = line_angle(*line1)
    alpha2 = line_angle(*line2)
    return abs(alpha1 - alpha2)


def rotate_point(point:Point, angle:float, center:Point=(0, 0)):
    """Rotate a point around a center by an angle in radians.

    Args:
        point (Point): Point to rotate.
        angle (float): Angle of rotation in radians.
        center (Point): Center of rotation.

    Returns:
        Point: Rotated point.
    """
    x, y = point[:2]
    cx, cy = center[:2]
    dx = x - cx
    dy = y - cy
    x = cx + dx * cos(angle) - dy * sin(angle)
    y = cy + dx * sin(angle) + dy * cos(angle)
    return (x, y)


def circle_tangent_to2lines(line1, line2, intersection_, radius):
    """Given two lines, their intersection point and a radius,
    return the center of the circle tangent to both lines and
    with the given radius.

    Args:
        line1 (Line): First line.
        line2 (Line): Second line.
        intersection_ (Point): Intersection point of the lines.
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


def round_point(point: list[float], n_digits: int = 2) -> list[float]:
    """
    Round a point (x, y) to a given precision.

    Args:
        point (list[float]): Input point.
        n_digits (int, optional): Number of decimal places to round to. Defaults to 2.

    Returns:
        list[float]: Rounded point.
    """
    x, y = point[:2]
    x = round(x, n_digits)
    y = round(y, n_digits)
    return (x, y)


def round_segment(segment: Sequence[Point], n_digits: int = 2):
    """Round a segment to a given precision.

    Args:
        segment (Sequence[Point]): Input segment.
        n_digits (int, optional): Number of decimal places to round to. Defaults to 2.

    Returns:
        Sequence[Point]: Rounded segment.
    """
    p1 = round_point(segment[0], n_digits)
    p2 = round_point(segment[1], n_digits)

    return [p1, p2]


def get_polygon_grid_point(n, line1, line2, circumradius=100):
    """See chapter ??? for explanation of this function.

    Args:
        n (int): Number of sides.
        line1 (Line): First line.
        line2 (Line): Second line.
        circumradius (float, optional): Circumradius of the polygon. Defaults to 100.

    Returns:
        Point: Grid point of the polygon.
    """
    s = circumradius * 2 * sin(pi / n)  # side length
    points = reg_poly_points(0, 0, n, s)[:-1]
    p1 = points[line1[0]]
    p2 = points[line1[1]]
    p3 = points[line2[0]]
    p4 = points[line2[1]]

    return intersection((p1, p2), (p3, p4))[1]


def congruent_polygons(
    polygon1: list[Point],
    polygon2: list[Point],
    dist_tol: float = None,
    area_tol: float = None,
    side_length_tol: float = None,
    angle_tol: float = None,
) -> bool:
    """
    Return True if two polygons are congruent.
    They can be translated, rotated and/or reflected.

    Args:
        polygon1 (list[Point]): First polygon.
        polygon2 (list[Point]): Second polygon.
        dist_tol (float, optional): Distance tolerance. Defaults to None.
        area_tol (float, optional): Area tolerance. Defaults to None.
        side_length_tol (float, optional): Side length tolerance. Defaults to None.
        angle_tol (float, optional): Angle tolerance. Defaults to None.

    Returns:
        bool: True if the polygons are congruent, False otherwise.
    """
    dist_tol, area_tol, angle_tol = get_defaults(
        ["dist_tol", "area_rtol", "angle_rtol"], [dist_tol, area_tol, angle_tol]
    )
    if side_length_tol is None:
        side_length_tol = defaults["rtol"]
    dist_tol2 = dist_tol * dist_tol
    poly1 = polygon1
    poly2 = polygon2
    if close_points2(poly1[0], poly1[-1], dist2=dist_tol2):
        poly1 = poly1[:-1]
    if close_points2(poly2[0], poly2[-1], dist2=dist_tol2):
        poly2 = poly2[:-1]
    len_poly1 = len(poly1)
    len_poly2 = len(poly2)
    if len_poly1 != len_poly2:
        return False
    if not isclose(
        abs(polygon_area(poly1)), abs(polygon_area(poly2)), rtol=area_tol, atol=area_tol
    ):
        return False

    side_lengths1 = [distance(poly1[i], poly1[i - 1]) for i in range(len_poly1)]
    side_lengths2 = [distance(poly2[i], poly2[i - 1]) for i in range(len_poly2)]
    check1 = equal_cycles(side_lengths1, side_lengths2, rtol=side_length_tol)
    if not check1:
        check_reverse = equal_cycles(
            side_lengths1, side_lengths2[::-1], rtol=side_length_tol
        )
        if not (check1 or check_reverse):
            return False

    angles1 = polygon_internal_angles(poly1)
    angles2 = polygon_internal_angles(poly2)
    check1 = equal_cycles(angles1, angles2, angle_tol)
    if not check1:
        poly2 = poly2[::-1]
        angles2 = polygon_internal_angles(poly2)
        check_reverse = equal_cycles(angles1, angles2, angle_tol)
        if not (check1 or check_reverse):
            return False

    return True


def positive_angle(angle, radians=True, tol=None, atol=None):
    """Return the positive angle in radians or degrees.

    Args:
        angle (float): Input angle.
        radians (bool, optional): Whether the angle is in radians. Defaults to True.
        tol (float, optional): Relative tolerance. Defaults to None.
        atol (float, optional): Absolute tolerance. Defaults to None.

    Returns:
        float: Positive angle.
    """
    tol, atol = get_defaults(["tol", "rtol"], [tol, atol])
    if radians:
        if angle < 0:
            angle += 2 * pi
        # if isclose(angle, TWO_PI, rtol=tol, atol=atol):
        #     angle = 0
    else:
        if angle < 0:
            angle += 360
        # if isclose(angle, 360, rtol=tol, atol=atol):
        #     angle = 0
    return angle


def polygon_internal_angles(polygon):
    """Return the internal angles of a polygon.

    Args:
        polygon (list[Point]): List of points representing the polygon.

    Returns:
        list[float]: List of internal angles of the polygon.
    """
    angles = []
    len_polygon = len(polygon)
    for i, pnt in enumerate(polygon):
        p1 = polygon[i - 1]
        p2 = pnt
        p3 = polygon[(i + 1) % len_polygon]
        angles.append(angle_between_lines2(p1, p2, p3))

    return angles


def bisector_line(a: Point, b: Point, c: Point) -> Line:
    """
    Given three points that form two lines [a, b] and [b, c]
    return the bisector line between them.

    Args:
        a (Point): First point.
        b (Point): Second point.
        c (Point): Third point.

    Returns:
        Line: Bisector line.
    """
    d = mid_point(a, c)

    return [d, b]


def between(a, b, c):
    """Return True if c is between a and b.

    Args:
        a (Point): First point.
        b (Point): Second point.
        c (Point): Third point.

    Returns:
        bool: True if c is between a and b, False otherwise.
    """
    if not collinear(a, b, c):
        res = False
    elif a[0] != b[0]:
        res = ((a[0] <= c[0]) and (c[0] <= b[0])) or ((a[0] >= c[0]) and (c[0] >= b[0]))
    else:
        res = ((a[1] <= c[1]) and (c[1] <= b[1])) or ((a[1] >= c[1]) and (c[1] >= b[1]))
    return res


def collinear(a, b, c, area_rtol=None, area_atol=None):
    """Return True if a, b, and c are collinear.

    Args:
        a (Point): First point.
        b (Point): Second point.
        c (Point): Third point.
        area_rtol (float, optional): Relative tolerance for area. Defaults to None.
        area_atol (float, optional): Absolute tolerance for area. Defaults to None.

    Returns:
        bool: True if the points are collinear, False otherwise.
    """
    area_rtol, area_atol = get_defaults(
        ["area_rtol", "area_atol"], [area_rtol, area_atol]
    )
    return isclose(area(a, b, c), 0, rtol=area_rtol, atol=area_atol)


def polar_to_cartesian(r, theta):
    """Convert polar coordinates to cartesian coordinates.

    Args:
        r (float): Radius.
        theta (float): Angle in radians.

    Returns:
        Point: Cartesian coordinates.
    """
    return (r * cos(theta), r * sin(theta))


def cartesian_to_polar(x, y):
    """Convert cartesian coordinates to polar coordinates.

    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.

    Returns:
        tuple: Polar coordinates (r, theta).
    """
    r = hypot(x, y)
    theta = positive_angle(atan2(y, x))
    return r, theta


def fillet(a: Point, b: Point, c: Point, radius: float) -> tuple[Line, Line, Point]:
    """
    Given three points that form two lines [a, b] and [b, c]
    return the clipped lines [a, d], [e, c], center point
    of the radius circle (tangent to both lines), and the arc
    angle of the formed fillet.

    Args:
        a (Point): First point.
        b (Point): Second point.
        c (Point): Third point.
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
    mp = mid_point(a, c)  # [b, mp] is the bisector line
    center = offset_point_from_start(b, mp, radius / sin_alpha2)
    arc_angle = angle_between_lines2(e, center, d)

    return [a, d], [e, c], center, arc_angle


def line_by_point_angle_length(point, angle, length_):
    """
    Given a point, an angle, and a length, return the line
    that starts at the point and has the given angle and length.

    Args:
        point (Point): Start point of the line.
        angle (float): Angle of the line in radians.
        length_ (float): Length of the line.

    Returns:
        Line: Line with the given angle and length.
    """
    x, y = point[:2]
    dx = length_ * cos(angle)
    dy = length_ * sin(angle)

    return [(x, y), (x + dx, y + dy)]


def surface_normal(p1: Point, p2: Point, p3: Point) -> VecType:
    """
    Calculates the surface normal of a triangle given its vertices.

    Args:
        p1 (Point): First vertex.
        p2 (Point): Second vertex.
        p3 (Point): Third vertex.

    Returns:
        VecType: Surface normal vector.
    """
    v1 = np.array(p1)
    v2 = np.array(p2)
    v3 = np.array(p3)
    # Create two vectors from the vertices
    u = v2 - v1
    v = v3 - v1

    # Calculate the cross product of the two vectors
    normal = np.cross(u, v)

    # Normalize the vector to get a unit normal vector
    normal = normal / np.linalg.norm(normal)

    return normal


def normal(point1, point2):
    """Return the normal vector of a line.

    Args:
        point1 (Point): First point of the line.
        point2 (Point): Second point of the line.

    Returns:
        VecType: Normal vector of the line.
    """
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    norm = sqrt(dx**2 + dy**2)
    return [-dy / norm, dx / norm]


def area(a, b, c):
    """Return the area of a triangle given its vertices.

    Args:
        a (Point): First vertex.
        b (Point): Second vertex.
        c (Point): Third vertex.

    Returns:
        float: Area of the triangle.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])


def calc_area(points):
    """Calculate the area of a simple polygon (given by a list of its vertices).

    Args:
        points (list[Point]): List of points representing the polygon.

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
        points (list[Point]): List of points.

    Returns:
        list[Point]: List of points with redundant and collinear points removed.
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
        if EPSILON > calc_area([points[i - 2], points[i - 1], points[i]])[0] > -EPSILON:
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
        points (list[Point]): List of points representing the polygon.

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
            senses.append(cross_product_sense(points[i], points[i + 1], points[i + 2]))
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


def circle_circle_intersections(x0, y0, r0, x1, y1, r1):
    """Return the intersection points of two circles.

    Args:
        x0 (float): x-coordinate of the center of the first circle.
        y0 (float): y-coordinate of the center of the first circle.
        r0 (float): Radius of the first circle.
        x1 (float): x-coordinate of the center of the second circle.
        y1 (float): y-coordinate of the center of the second circle.
        r1 (float): Radius of the second circle.

    Returns:
        tuple: Intersection points of the two circles.
    """
    # taken from https://stackoverflow.com/questions/55816902/finding-the-
    # intersection-of-two-circles
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    # non intersecting
    if d > r0 + r1:
        res = None
    # One circle within other
    elif d < abs(r0 - r1):
        res = None
    # coincident circles
    elif d == 0 and r0 == r1:
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

        res = (x3, y3, x4, y4)

    return res


def circle_segment_intersection(circle, p1, p2):
    """Return True if the circle and the line segment intersect.

    Args:
        circle (Circle): Input circle.
        p1 (Point): First point of the line segment.
        p2 (Point): Second point of the line segment.

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
        point (Point): Point on the line segment.

    Returns:
        list[Point]: Intersection points of the ellipse and the line segment.
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
    from scipy.special import ellipeinc # this takes too long to import!!!
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
        Point: Point on the ellipse.
    """
    r = r_polar(a, b, angle)

    return (r * cos(angle), r * sin(angle))


def circle_line_intersection(c, p1, p2):
    """Return the intersection points of a circle and a line segment.

    Args:
        c (Circle): Input circle.
        p1 (Point): First point of the line segment.
        p2 (Point): Second point of the line segment.

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
        point (Point): Input point.
        center (Point): Center of the circle.
        radius (float): Radius of the circle.

    Returns:
        float: Distance between the point and the circle.
    """
    return abs(distance(center, point) - radius)


def get_interior_points(start, end, n_points):
    """Given start and end points and number of interior points
    returns the positions of the interior points

    Args:
        start (Point): Start point.
        end (Point): End point.
        n_points (int): Number of interior points.

    Returns:
        list[Point]: List of interior points.
    """
    rot_angle = line_angle(start, end)
    length_ = distance(start, end)
    seg_length = length_ / (n_points + 1.0)
    points = []
    for i in range(n_points):
        points.append(
            rotate_point([start[0] + seg_length * (i + 1), start[1]], start, rot_angle)
        )
    return points


def circle_3point(point1, point2, point3):
    """Given three points, returns the center point and radius

    Args:
        point1 (Point): First point.
        point2 (Point): Second point.
        point3 (Point): Third point.

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
        common_properties(self, graphics_object=False)

    def __repr__(self):
        return f"Vertex({self.x}, {self.y}, {self.z})"

    def __eq__(self, other):
        return self[0] == other[0] and self[1] == other[1] and self[2] == other[2]

    def copy(self):
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
        if self.y < other.y:
            res = True
        elif self.y == other.y:
            if self.x > other.x:
                res = True
        return res

    def above(self, other):
        """This is for 2D points only

        Args:
            other (Vertex): Other vertex.

        Returns:
            bool: True if this vertex is above the other vertex, False otherwise.
        """
        if self.y > other.y:
            res = True
        elif self.y == other.y and self.x < other.x:
            res = True
        else:
            res = False

        return res


class Edge:
    """A 2D edge."""

    def __init__(
        self, start_point: Union[Point, Vertex], end_point: Union[Point, Vertex]
    ):
        if isinstance(start_point, Point):
            start = Vertex(*start_point)
        elif isinstance(end_point, Vertex):
            start = start_point
        else:
            raise ValueError("Start point should be a Point or Vertex instance.")

        if isinstance(end_point, Point):
            end = Vertex(*end_point)
        elif isinstance(end_point, Vertex):
            end = end_point
        else:
            raise ValueError("End point should be a Point or Vertex instance.")

        self.start = start
        self.end = end
        self.type = Types.EDGE
        common_properties(self, graphics_object=False)

    def __repr__(self):
        return str(f"Edge({self.start}, {self.end})")

    def __str__(self):
        return str(f"Edge({self.start.point}, {self.end.point})")

    def __eq__(self, other):
        start = other.start.point
        end = other.end.point

        return (
            isclose(
                self.start.point, start, rtol=defaults["rtol"], atol=defaults["atol"]
            )
            and isclose(
                self.end.point, end, rtol=defaults["rtol"], atol=defaults["atol"]
            )
        ) or (
            isclose(self.start.point, end, rtol=defaults["rtol"], atol=defaults["atol"])
            and isclose(
                self.end.point, start, rtol=defaults["rtol"], atol=defaults["atol"]
            )
        )

    def __getitem__(self, subscript):
        vertices = self.vertices
        if isinstance(subscript, slice):
            res = vertices[subscript.start : subscript.stop : subscript.step]
        elif isinstance(subscript, int):
            res = vertices[subscript]
        else:
            raise ValueError("Invalid subscript.")
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


def rotate_point_3D(point: Point, line: Line, angle: float) -> Point:
    """Rotate a 2d point (out of paper) about a 2d line by the given angle.
    This is used for animating mirror reflections.
     Args:
         point (Point): Point to rotate.
         line (Line): Line to rotate about.
         angle (float): Angle of rotation in radians.

     Returns:
         Point: Rotated point.
    """
    from ..graphics.affine import rotation_matrix, translation_matrix

    p1, p2 = line
    line_angle_ = line_angle(p1, p2)
    translation = translation_matrix(-p1[0], -p1[1])
    rotation = rotation_matrix(-line_angle_, (0, 0))
    xform = translation @ rotation
    x, y = point
    x, y, _ = [x, y, 1] @ xform

    y *= cos(angle)

    inv_translation = translation_matrix(p1[0], p1[1])
    inv_rotation = rotation_matrix(line_angle_, (0, 0))
    inv_xform = inv_rotation @ inv_translation
    x, y, _ = [x, y, 1] @ inv_xform

    return (x, y)


def rotate_line_3D(line: Line, about: Line, angle: float) -> Line:
    """Rotate a 3d line about a 3d line by the given angle

    Args:
        line (Line): Line to rotate.
        about (Line): Line to rotate about.
        angle (float): Angle of rotation in radians.

    Returns:
        Line: Rotated line.
    """
    p1 = rotate_point_3D(line[0], about, angle)
    p2 = rotate_point_3D(line[1], about, angle)

    return [p1, p2]

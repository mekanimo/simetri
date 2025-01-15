"""Geometry related utilities.
These functions are used to perform geometric operations.
They are not documented. Some of the are one off functions that
are not used in the main codebase or tested."""

# To do: Clean up this module and add documentation.

from __future__ import annotations

from math import hypot, atan2, floor, pi, sin, cos, sqrt, exp
from itertools import cycle
from typing import Any, Union, Sequence

import numpy as np
from numpy import isclose, array, around
from .utilities import (
    flatten,
    lerp,
    sanitize_graph_edges,
    equal_cycles,
    reg_poly_points,
)

from .vector import Vector2D
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
from ..settings.settings import logging, defaults

array = np.array
around = np.around

TWO_PI = 2 * pi  # 360 degrees


def is_number(x: Any) -> bool:
    """
    Return True if x is a number.
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
    bbox1: minx, miny, maxx, maxy
    bbox2: minx, miny, maxx, maxy
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
    damping: float = 1,
) -> np.ndarray:
    """
    Generate a sine wave.
    amplitude: amplitude of the wave
    frequency: frequency of the wave
    duration: duration of the wave
    sample_rate: sample rate
    phase: phase angle of the wave
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

    Parameters:
    amplitude (float): The initial amplitude of the damping function.
    duration (float): The duration over which the damping occurs, in seconds.
    sample_rate (float): The number of samples per second.

    Returns:
    list: A list of float values representing the damping function over time.
    """
    """"""
    damping = []
    for i in range(int(duration * sample_rate)):
        damping.append(amplitude * exp(-i / (duration * sample_rate)))
    return damping


def line_segment_bbox(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float, float]:
    """
    Return the bounding box of a line segment.
    x1, y1 : segment start point
    x2, y2 : segment end point
    """
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def line_segment_bbox_check(seg1: Line, seg2: Line) -> bool:
    """
    Given two line segments, return True if their bounding boxes
    overlap.
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
    find all close points in a list of points
    along with their ids: [[x1, y1, id1], [x2, y2, id2], ...]
    Return a dictionary of the form {id1: [id2, id3, ...], ...}
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


def all_intersections(
    segments: Sequence[Line],
    rtol: float = None,
    atol: float = None,
    use_intersection3: bool = False,
) -> dict[int, list[tuple[Point, int]]]:
    """
    Find all intersection points of the given list of segments
    (sweep line algorithm variant)
    segments: [[[x1, y1], [x2, y2]], [[x1, y1], [x2, y2]], ...
    Return a dictionary of the form {segment_id: [[id1, (x1, y1)],
                                                   [id2, (x2, y2)]], ...}
    for each segment (defined by its index) in the list, the list of
    intersections with the index of the other segment and the coordinates of
    the intersection point are returned.
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
    """Dot product of two vectors. AB and BC"""
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
    Return the cross product of two vectors.
    vec1 = B - A
    vec2 = C - B
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
    Given line1 as point point1 and point2, and line2 as point point3 and point2
    return the angle between two lines
    (point2 is the corner point)
    """
    return atan2(
        cross_product2(point1, point2, point3), dot_product2(point1, point2, point3)
    )


def angled_line(line: Line, theta: float) -> Line:
    """
    Given a line find another line with theta radians between them.
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
    """

    return [cos(angle), sin(angle)]


def close_points2(p1: Point, p2: Point, dist2: float = 0.01) -> bool:
    """
    Return True if two points are close to each other.
    dist2 is the square of the threshold distance.
    """
    return distance2(p1, p2) <= dist2


def close_angles(angle1: float, angle2: float, angtol=None) -> bool:
    """
    Return True if two angles are close to each other.
    """
    if angtol is None:
        angtol = defaults["angtol"]

    return (abs(angle1 - angle2) % (2 * pi)) < angtol


def distance(p1: Point, p2: Point) -> float:
    """
    Return the distance between two points.
    """
    return hypot(p2[0] - p1[0], p2[1] - p1[1])


def distance2(p1: Point, p2: Point) -> float:
    """
    Return the squared distance between two points.
    Useful for comparing distances without the need to
    compute the square root.
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
    poly_point1: [(x1, y1), (x2, y2), ...]
    poly_point2: [(x1, y1), (x2, y2), ...]
    return [(x1, y1), (x2, y2), ...]
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
    Return a list of points if return_points is True or lines otherwise.
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
        else:
            msg = "No intersection found in stitch."
            logging.info(msg)
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
            msg = "No intersection found in stitch. final_x"
            logging.info(msg)
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
    Lines: [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)], ...]
    return: [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)], ...]
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
    Given a list of points that define a polygon, return the center
    point.
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
    Given a list of points that define a polygon, return the center
    point.
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
    Given a list of points that define a polygon, return the center
    point.
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
    Polyline: [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)], ...]
    return: [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)], ...]
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
    Polyline: [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)], ...]
    return: [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)], ...]
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
    Polyline: [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)], ...]
    return: [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)], ...]
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
    Return two offset lines to a given line segment
       with the given offset amount.
    """
    line1 = offset_line(line, offset)
    line2 = offset_line(line, -offset)

    return line1, line2


def equal_lines(line1: Line, line2: Line, dist_tol: float = None) -> bool:
    """
    Return True if two lines are close enough.
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
    """

    def extend(dist, line):
        # p = (1-t)*p1 + t*p2 : parametric equation of a line segment (p1, p2)
        line_length = length(line)
        t = (line_length + dist) / line_length
        p1, p2 = line
        x1, y1 = p1
        x2, y2 = p2
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


def direction(p, q, r):
    """
    Checks the orientation of three points (p, q, r).
    Returns:
        0 if collinear
        >0 if counter-clockwise
        <0 if clockwise
    """
    return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])


def collinear_segments(segment1, segment2, tol=None, atol=None):
    """
    Checks if two line segments (a1, b1) and (a2, b2) are collinear.
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
    return a point(ksi, eta) in local coordinates"""
    sin_theta = sin(theta)
    cos_theta = cos(theta)
    ksi = (x - xi) * cos_theta + (y - yi) * sin_theta
    eta = (y - yi) * cos_theta - (x - xi) * sin_theta
    return (ksi, eta)


def stitch_lines(line1: Line, line2: Line) -> Sequence[Line]:
    """if the lines intersect, trim the lines
    if the lines don't intersect, extend the lines"""
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
    +x, -y = 4th"""
    return int(floor((atan2(y, x) % (TWO_PI)) / (pi / 2)) + 1)


def get_quadrant_from_deg_angle(deg_angle: float) -> int:
    """quadrants:
    (0, 90) = 1st
    (90, 180) = 2nd
    (180, 270) = 3rd
    (270, 360) = 4th"""
    return int(floor(deg_angle / 90.0) % 4 + 1)


def homogenize(points: Sequence[Point]) -> np.ndarray:
    """Points can be ((x1, y1), (x2, y2), ... (xn, yn))
    or numpy array of (x, y) or (x, y, 1) vectors.
    Return a numpy array of points array(((x1, y1, 1.),
    (x2, y2, 1.), ... (xn, yn, 1.)))."""

    return _homogenize(flatten(points))


def _homogenize(coordinates: Sequence[float]) -> np.ndarray:
    """Internal use only. API provides a homogenize function.
    Given a sequence of coordinates(x1, y1, x2, y2, ... xn, yn),
    return a numpy array of points array(((x1, y1, 1.),
    (x2, y2, 1.), ... (xn, yn, 1.)))."""
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
    """
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    return intersect2(x1, y1, x2, y2, x3, y3, x4, y4)


def intersection2(x1, y1, x2, y2, x3, y3, x4, y4, rtol=None, atol=None):
    """Check the intersection of two line segments. See the documentation"""
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
    for more details."""
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
    """Remove the middle points from collinear edges."""
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


def is_horizontal(line: Line, eps: float = 0.0001) -> bool:
    """Return True if the line is horizontal."""
    return abs(j_vec.dot(line_vector(line))) <= eps


def is_line(line_: Any) -> bool:
    """Return True if the input is a line."""
    try:
        p1, p2 = line_
        return is_point(p1) and is_point(p2)
    except:
        return False


def is_point(pnt: Any) -> bool:
    """Return True if the input is a point."""
    try:
        x, y = pnt[:2]
        return is_number(x) and is_number(y)
    except:
        return False


def is_vertical(line: Line, eps: float = 0.0001) -> bool:
    """Return True if the line is vertical."""
    return abs(i_vec.dot(line_vector(line))) <= eps


def length(line: Line) -> float:
    """Return the length of a line."""
    p1, p2 = line
    return distance(p1, p2)


def lerp_points(p1: Point, p2: Point, t: float) -> Point:
    """Linear interpolation of two points.
    p1: (x1, y1)
    p2: (x2, y2)
    0 <= t <=1."""
    x1, y1 = p1
    x2, y2 = p2
    return (lerp(x1, x2, t), lerp(y1, y2, t))


def slope(start_point: Point, end_point: Point, rtol=None, atol=None) -> float:
    """Return the slope of a line given by two points.
    Order makes a difference."""
    rtol, atol = get_defaults(["rtol", "atol"], [rtol, atol])
    x1, y1 = start_point[:2]
    x2, y2 = end_point[:2]
    if isclose(x1, x2, rtol=rtol, atol=atol):
        res = defaults["INF"]
    else:
        res = (y2 - y1) / (x2 - x1)

    return res


def line_angle(start_point: Point, end_point: Point) -> float:
    """Return the orientation angle (in radians) of a line given by start and end
    points.
    Order makes a difference."""
    return atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])


def inclination_angle(start_point: Point, end_point: Point) -> float:
    """Return the inclination angle (in radians) of a line given by start and end
    points.
    Inclination angle is always between zero and pi.
    Order makes no difference."""
    return line_angle(start_point, end_point) % pi


def line2vector(line: Line) -> VecType:
    """Return the vector representation of a line"""
    x1, y1 = line[0]
    x2, y2 = line[1]
    dx = x2 - x1
    dy = y2 - y1
    return [dx, dy]


def line_through_point_and_angle(
    point: Point, angle: float, length_: float = 100
) -> Line:
    """Return a line through the given point with the given angle
    and length"""
    x, y = point[:2]
    dx = length_ * cos(angle)
    dy = length_ * sin(angle)
    return [[x, y], [x + dx, y + dy]]


def line_vector(line: Line) -> VecType:
    """Return the vector representation of a line."""
    x1, y1 = line[0]
    x2, y2 = line[1]
    return Vector2D(x2 - x1, y2 - y1)


def mid_point(p1: Point, p2: Point) -> Point:
    """Return the mid point of two points."""
    x = (p2[0] + p1[0]) / 2
    y = (p2[1] + p1[1]) / 2
    return (x, y)


def norm(vec: VecType) -> float:
    """Return the norm (vector length) of a vector."""
    return hypot(vec[0], vec[1])


def ndarray_to_xy_list(arr: np.ndarray) -> Sequence[Point]:
    """Convert a numpy array to a list of points."""
    return arr[:, :2].tolist()


def offset_line(line: Sequence[Point], offset: float) -> Sequence[Point]:
    """Return an offset line from a given line."""
    unit_vec = perp_unit_vector(line)
    dx = unit_vec[0] * offset
    dy = unit_vec[1] * offset
    x1, y1 = line[0]
    x2, y2 = line[1]
    return [[x1 + dx, y1 + dy], [x2 + dx, y2 + dy]]


def offset_lines(polylines: Sequence[Line], offset: float = 1) -> list[Line]:
    """Return a list of offset lines from a list of lines."""

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


def offset_point_on_line(point: Point, line: Line, offset: float) -> Point:
    """Return a point on a line that is offset from the given point."""
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
    """Return an offset point from a given point."""
    x, y = point[:2]
    return x + dx, y + dy


def parallel_line(line: Line, point: Point) -> Line:
    """Return a parallel line to the given line that goes through the
    given point"""
    x1, y1 = line[0]
    x2, y2 = line[1]
    x3, y3 = point
    dx = x2 - x1
    dy = y2 - y1
    return [[x3, y3], [x3 + dx, y3 + dy]]


def perp_offset_point(point: Point, line: Line, offset: float) -> Point:
    """Return a point that is offset from the given point in the
    perpendicular direction to the given line."""
    unit_vec = perp_unit_vector(line)
    dx = unit_vec[0] * offset
    dy = unit_vec[1] * offset
    x, y = point[:2]
    return [x + dx, y + dy]


def perp_unit_vector(line: Line) -> VecType:
    """Return the perpendicular unit vector to a line"""
    x1, y1 = line[0]
    x2, y2 = line[1]
    dx = x2 - x1
    dy = y2 - y1
    norm_ = sqrt(dx**2 + dy**2)
    return [-dy / norm_, dx / norm_]


def point_on_line(
    point: Point, line: Line, rtol: float = None, atol: float = None
) -> bool:
    """Return True if the given point is on the given line"""
    rtol, atol = get_defaults(["rtol", "atol"], [rtol, atol])
    p1, p2 = line
    return isclose(slope(p1, point), slope(point, p2), rtol=rtol, atol=atol)


def point_on_line_segment(
    point: Point, line: Line, rtol: float = None, atol: float = None
) -> bool:
    """Return True if the given point is on the given line segment"""
    rtol, atol = get_defaults(["rtol", "atol"], [rtol, atol])
    p1, p2 = line
    return isclose(
        (distance(p1, point) + distance(p2, point)),
        distance(p1, p2),
        rtol=rtol,
        atol=atol,
    )


def point_to_line_distance(point: Point, line: Line) -> float:
    """Return the vector from a point to a line"""
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
    boundary points, returns False."""

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


def point_to_line_vec(point: Point, line: Line, unit: bool = False) -> VecType:
    """Return the perpendicular vector from a point to a line"""
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
    polygon: list of points [[x1, y1], [x2, y2], ...]
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
    """Calculate the perimeter of a polygon."""
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
    polygon: list of points [[x1, y1], [x2, y2], ...]
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
    """
    return 2 * radius * sin(pi / n)


def segmentize_catmull_rom(
    a: float, b: float, c: float, d: float, n: int = 100
) -> Sequence[float]:
    """a and b are the control points and c and d are
    start and end points respectively,
    n is the number of segments to generate."""
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
    of an n-sided regular polygon with the given side_len length"""
    return side_len / (2 * sin(pi / n))


def translate_line(dx: float, dy: float, line: Line) -> Line:
    """Return a translated line by dx and dy"""
    x1, y1 = line[0]
    x2, y2 = line[1]
    return [[x1 + dx, y1 + dy], [x2 + dx, y2 + dy]]


def trim_line(line1: Line, line2: Line) -> Line:
    """Trim line1 to the intersection of line1 and line2.
    Extend it if necessary.
    """
    intersection_ = intersection(line1, line2)
    return [line1[0], intersection_]


def unit_vector(line: Line) -> VecType:
    """Return the unit vector of a line"""
    norm_ = length(line)
    p1, p2 = line
    x1, y1 = p1
    x2, y2 = p2
    return [(x2 - x1) / norm_, (y2 - y1) / norm_]


def unit_vector_(line: Line) -> Sequence[VecType]:
    """Return the cartesian unit vector of a line
    with the given line's start and end points"""
    x1, y1 = line[0]
    x2, y2 = line[1]
    dx = x2 - x1
    dy = y2 - y1
    norm_ = sqrt(dx**2 + dy**2)
    return [dx / norm_, dy / norm_]


def vec_along_line(line: Line, magnitude: float) -> VecType:
    """Return a vector along a line with the given magnitude."""
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
    """Return the direction angle of a vector"""

    return atan2(vec[1], vec[0])


def cross_product_sense(a: Point, b: Point, c: Point) -> int:
    """Return the cross product sense of vectors a and b."""
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
    """Return True if p1, p2, p3 make a right turn."""
    return cross(p1, p2, p3) < 0


def left_turn(p1, p2, p3):
    """Return True if p1, p2, p3 make a left turn."""
    return cross(p1, p2, p3) > 0


def cross(p1, p2, p3):
    """Return the cross product of vectors p1p2 and p1p3."""
    x1, y1 = p2[0] - p1[0], p2[1] - p1[1]
    x2, y2 = p3[0] - p1[0], p3[1] - p1[1]
    return x1 * y2 - x2 * y1


def tri_to_cart(points):
    """
    Convert a list of points from triangular to cartesian coordinates.
    points: list of points in triangular coordinates
    """
    u = [1, 0]
    v = cos(pi / 3), sin(pi / 3)
    convert = array([u, v])

    return array(points) @ convert


def cart_to_tri(points):
    """
    Convert a list of points from cartesian to triangular coordinates.
    points: list of points in cartesian coordinates
    """
    u = [1, 0]
    v = cos(pi / 3), sin(pi / 3)
    convert = np.linalg.inv(array([u, v]))

    return array(points) @ convert


def convex_hull(points):
    """Return the convex hull of a set of 2D points."""
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
    """
    return list(zip(items, items[1:]))


def flat_points(connected_segments):
    """Return a list of points from a list of connected pairs of points."""
    points = [line[0] for line in connected_segments]
    points.append(connected_segments[-1][1])
    return points


def point_in_quad(point: Point, quad: list[Point]) -> bool:
    """Return True if the point is inside the quad."""
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
    """Convert points to clean polygons. Points are vertices of
    polygons."""
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
    return the point on the line at the given offset"""
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
    """Return the angle between two lines in radians."""
    alpha1 = line_angle(*line1)
    alpha2 = line_angle(*line2)
    return abs(alpha1 - alpha2)


def rotate_point(point, center, angle):
    """Rotate a point around a center by an angle in radians."""
    x, y = point
    cx, cy = center
    dx = x - cx
    dy = y - cy
    x = cx + dx * cos(angle) - dy * sin(angle)
    y = cy + dx * sin(angle) + dy * cos(angle)
    return (x, y)


def circle_tangent_to2lines(line1, line2, intersection_, radius):
    """Given two lines, their intersection point and a radius,
    return the center of the circle tangent to both lines and
    with the given radius."""

    alpha = angle_between_two_lines(line1, line2)
    dist = radius / sin(alpha / 2)
    start = offset_point_from_start(intersection_, line1.p1, dist)
    center = rotate_point(start, intersection_, alpha / 2)
    end = offset_point_from_start(intersection_, line2.p1, dist)

    return center, start, end


def triangle_area(a: float, b: float, c: float) -> float:
    """
    Given side lengths a, b and c, return the area of the triangle.
    """
    a_b = a - b
    return sqrt((a + (b + c)) * (c - (a_b)) * (c + (a_b)) * (a + (b - c))) / 4


def round_point(point: list[float], n_digits: int = 2) -> list[float]:
    """
    Round a point (x, y) to a given precision.
    """
    x, y = point[:2]
    x = round(x, n_digits)
    y = round(y, n_digits)
    return (x, y)


def get_polygon_grid_point(n, line1, line2, circumradius=100):
    """See chapter ??? for explanation of this function."""
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
    """Return the positive angle in radians or degrees."""
    tol, atol = get_defaults(["tol", "rtol"], [tol, atol])
    if radians:
        if angle < 0:
            angle += 2 * pi
        if isclose(angle, TWO_PI, rtol=tol, atol=atol):
            angle = 0
    else:
        if angle < 0:
            angle += 360
        if isclose(angle, 360, rtol=tol, atol=atol):
            angle = 0
    return angle


def polygon_internal_angles(polygon):
    """Return the internal angles of a polygon."""
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
    """
    d = mid_point(a, c)

    return [d, b]


def between(a, b, c):
    """Return True if c is between a and b."""
    if not collinear(a, b, c):
        res = False
    elif a[0] != b[0]:
        res = ((a[0] <= c[0]) and (c[0] <= b[0])) or ((a[0] >= c[0]) and (c[0] >= b[0]))
    else:
        res = ((a[1] <= c[1]) and (c[1] <= b[1])) or ((a[1] >= c[1]) and (c[1] >= b[1]))
    return res


def collinear(a, b, c, area_rtol=None, area_atol=None):
    """Return True if a, b, and c are collinear."""
    area_rtol, area_atol = get_defaults(
        ["area_rtol", "area_atol"], [area_rtol, area_atol]
    )
    return isclose(area(a, b, c), 0, rtol=area_rtol, atol=area_atol)


def polar_to_cartesian(r, theta):
    """Convert polar coordinates to cartesian coordinates."""
    return (r * cos(theta), r * sin(theta))


def cartesian_to_polar(x, y):
    """Convert cartesian coordinates to polar coordinates."""
    r = hypot(x, y)
    theta = atan2(y, x)
    return r, theta


def fillet(a: Point, b: Point, c: Point, radius: float) -> tuple[Line, Line, Point]:
    """
    Given three points that form two lines [a, b] and [b, c]
    return the clipped lines [a, d], [e, c], center point
    of the radius circle (tangent to both lines), and the arc
    angle of the formed fillet.
    Example:
    a = (0, 0)
    b = (30, 100)
    c = (60, 0)

    p1, p2, c, angle = fillet(a, b, c, radius=10)
    print(f"{p1}\n, {p2}\n, {c}\n, {radians * angle}")
    ->
    [(0, 0), (20.421737147788487, 68.07245715929496)]
    [(39.57826285221151, 68.07245715929496), (60, 0)]
    (30.0, 65.19897830363149)
    146.60151153201264
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
    """
    x, y = point[:2]
    dx = length_ * cos(angle)
    dy = length_ * sin(angle)

    return [(x, y), (x + dx, y + dy)]


def surface_normal(p1: Point, p2: Point, p3: Point) -> VecType:
    """
    Calculates the surface normal of a triangle given its vertices.

    Args:
        p1 (x1, y1, z1): The coordinates of the first vertex.
        p2 (x2, y2, z2): The coordinates of the second vertex.
        p3 (x3, y3, z3): The coordinates of the third vertex.

    Returns:
        numpy.ndarray: The surface normal vector.
    Example:
    p1 = [1, 0, 0]
    p2 = [0, 1, 0]
    p3 = [0, 0, 1]

    normal = surface_normal(p1, p2, p3)
    print(normal)  # Output: [0.57735027 0.57735027 0.57735027]
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


def area(a, b, c):
    """Return the area of a triangle given its vertices."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])


def calc_area(points):
    """Calculate the area of a simple polygon (given by a list of its vertices).

    Args:
      points: list of points

    Returns:
      (area, clockwise)
       area is always positive
       clockwise is either true or false
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
    """Remove redundant and collinear points from a list of points."""
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
    """Return True if the polygon is convex."""
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
    """Set the next and previous vertices of a list of vertices."""
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
    """Return the intersection points of two circles."""
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
        y2 = y0 + a * (y1 - y0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d
        x4 = x2 - h * (y1 - y0) / d
        y4 = y2 + h * (x1 - x0) / d

        res = (x3, y3, x4, y4)

    return res


def circle_segment_intersection(circle, p1, p2):
    """Return True if the circle and the line segment intersect."""
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


def circle_line_intersection(c, p1, p2):
    """Return the intersection points of a circle and a line segment."""

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
    """Return True if the circle and the polygon intersect."""
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
    between the given point and the circle"""
    return abs(distance(center, point) - radius)


def get_interior_points(start, end, n_points):
    """Given start and end points and number of interior points
    returns the positions of the interior points"""
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
    """Given three points, returns the center point and radius"""
    ax, ay = point1
    bx, by = point2
    cx, cy = point3
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
    """Given a point and a line, returns the projection of the point on the line"""
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
        """This is for 2D points only"""
        res = False
        if self.y < other.y:
            res = True
        elif self.y == other.y:
            if self.x > other.x:
                res = True
        return res

    def above(self, other):
        """This is for 2D points only"""
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

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from random import choice

import simetri.geom.points.point_utils
import simetri.geom.segments.line_utils
import simetri.graphics as sg

PointType = Sequence[float]
LineType = Sequence[PointType]
PolygonLike = sg.Shape | sg.Group | Sequence[PointType]
TurnPair = tuple[float, float]
TurnSequence = Sequence[TurnPair]

TURN_ANGLE_DIGITS = 2
GRID_ROWS = 50
GRID_COLUMNS = 8
GRID_GAP = 1.2


def get_grid_pos(index, n_columns, cell_size=25):
    row = index // n_columns
    col = index % n_columns
    return col * cell_size, row * cell_size


def _as_shape(item):
    res = item
    if not isinstance(res, sg.Shape):
        res = sg.Shape(res, closed=True)
    return res


def _drawable_height(items):
    if hasattr(items, "height"):
        res = items.height
    else:
        res = sg.Group(items).height
    return res


def show(*items, canvas, **kwargs):
    if len(items) == 1:
        items_ = items[0]
    else:
        items_ = items

    height = _drawable_height(items_)
    canvas.translate(0, -height * GRID_GAP)
    canvas.draw(items_, **kwargs)
    # canvas.display()


def show_grid(items, n_rows, n_columns, canvas, gap=GRID_GAP, **kwargs):
    """Draw items in a grid with n_rows rows and n_columns columns."""
    limit = n_rows * n_columns
    shapes = [_as_shape(item) for item in items[:limit]]
    if not shapes:
        return

    max_w = max(shape.width for shape in shapes)
    max_h = max(shape.height for shape in shapes)
    cell_w = max_w * gap
    cell_h = max_h * gap
    placed = []

    for index, shape in enumerate(shapes):
        row = index // n_columns
        col = index % n_columns
        cell_x = col * cell_w
        cell_y = -row * cell_h
        inset_x = (cell_w - shape.width) / 2
        inset_y = (cell_h - shape.height) / 2
        sw = shape.southwest
        dx = cell_x + inset_x - sw[0]
        dy = cell_y + inset_y - sw[1]
        placed.append(shape.translate(dx, dy))

    canvas.draw(placed, **kwargs)
    canvas.translate(0, -n_rows * cell_h)


# PORTED
def _turn_angle_equal(angle1: float, angle2: float, ang_tol: float) -> bool:
    """Return True if two signed turn angles agree within ``ang_tol``.

    The absolute difference is reduced modulo ``2π`` so collinear turns
    ``π`` and ``-π`` are treated as equal.

    Args:
        angle1: First signed turn angle in radians.
        angle2: Second signed turn angle in radians.
        ang_tol: Maximum allowed angular difference.

    Returns:
        True if the angles match within tolerance; False otherwise.
    """
    tau = 2 * sg.pi
    delta = abs(angle1 - angle2)
    delta = min(delta, abs(delta - tau), abs(delta + tau))
    return delta <= ang_tol


# PORTED
def polygon_turns(vertices: Sequence[PointType]) -> list[TurnPair]:
    """Return the signed turn sequence of a polygon.

    For each vertex ``i``, records the length of the edge from ``i`` to
    ``i + 1`` and the signed turn angle at ``i + 1`` between that edge and
    the next (via ``angle_between_lines3``). Angles are rounded to
    ``TURN_ANGLE_DIGITS``. Convex and reflex corners keep opposite signs.

    Args:
        vertices: Polygon vertices in walk order (closed; first vertex is
            not repeated).

    Returns:
        A list of ``(side_length, turn_angle)`` pairs, one per vertex.
    """
    n = len(vertices)
    res = []
    for i in range(n):
        vert = vertices[i]
        next_vert = vertices[(i + 1) % n]
        next_seg = (next_vert, vertices[(i + 2) % n])
        seg = (vert, next_vert)
        angle = simetri.geom.segments.line_utils.angle_between_lines3(
            vert, *next_seg
        )
        res.append(
            (
                simetri.geom.points.point_utils.distance(*seg),
                round(angle, TURN_ANGLE_DIGITS),
            )
        )

    return res


def _turn_pair_equal(pair1: TurnPair, pair2: TurnPair, dist_tol: float) -> bool:
    """Return True if two ``(length, angle)`` pairs match within tolerance.

    Side lengths must agree within ``dist_tol``. Signed turn angles must
    agree within ``10 ** -TURN_ANGLE_DIGITS`` (modulo ``2π``).

    Args:
        pair1: First ``(side_length, turn_angle)`` pair.
        pair2: Second ``(side_length, turn_angle)`` pair.
        dist_tol: Maximum allowed difference in side lengths.

    Returns:
        True if both length and angle match; False otherwise.
    """
    length1, angle1 = pair1
    length2, angle2 = pair2
    turn_angle_tol = 10**-TURN_ANGLE_DIGITS
    res = abs(length1 - length2) <= dist_tol and _turn_angle_equal(
        angle1, angle2, turn_angle_tol
    )
    return res


def _turn_sequences_equal(
    turns1: TurnSequence, turns2: TurnSequence, dist_tol: float
) -> bool:
    """Return True if two turn sequences match entry-wise in order.

    Sequences must have the same length. Each corresponding pair is
    compared with ``_turn_pair_equal`` (no cyclic shift).

    Args:
        turns1: First sequence of ``(side_length, turn_angle)`` pairs.
        turns2: Second sequence of ``(side_length, turn_angle)`` pairs.
        dist_tol: Maximum allowed difference in side lengths.

    Returns:
        True if every entry matches; False if lengths differ or any
        pair fails.
    """
    res = False
    if len(turns1) == len(turns2):
        res = all(
            _turn_pair_equal(pair1, pair2, dist_tol)
            for pair1, pair2 in zip(turns1, turns2)
        )

    return res


def _cyclic_turns_equal(
    turns1: TurnSequence, turns2: TurnSequence, dist_tol: float
) -> bool:
    """Return True if ``turns2`` matches ``turns1`` under cyclic shift.

    Used for rotation invariance: the starting vertex may differ.
    Empty sequences of equal length are treated as equal.

    Args:
        turns1: First sequence of ``(side_length, turn_angle)`` pairs.
        turns2: Second sequence of ``(side_length, turn_angle)`` pairs.
        dist_tol: Maximum allowed difference in side lengths.

    Returns:
        True if some cyclic rotation of ``turns2`` matches ``turns1``
        entry-wise; False otherwise.
    """
    res = False
    n = len(turns1)
    if n == len(turns2):
        if n == 0:
            res = True
        else:
            for offset in range(n):
                rotated = turns2[offset:] + turns2[:offset]
                if _turn_sequences_equal(turns1, rotated, dist_tol):
                    res = True
                    break

    return res


def _mirror_turn_angle_equal(
    angle1: float, angle2: float, ang_tol: float
) -> bool:
    """Return True if signed turn angles are negatives of each other.

    Used for mirror-image walks (``angle1 ≈ -angle2``). The absolute
    sum is reduced modulo ``2π`` so wrap-around cases still match.

    Args:
        angle1: First signed turn angle in radians.
        angle2: Second signed turn angle in radians.
        ang_tol: Maximum allowed deviation from exact negation.

    Returns:
        True if the angles are mirrors within tolerance; False otherwise.
    """
    tau = 2 * sg.pi
    delta = abs(angle1 + angle2)
    delta = min(delta, abs(delta - tau), abs(delta + tau))
    return delta <= ang_tol


def _mirror_turn_pair_equal(
    pair1: TurnPair, pair2: TurnPair, dist_tol: float
) -> bool:
    """Return True if lengths match and turn angles are mirrored.

    Side lengths must agree within ``dist_tol``. Signed turn angles must
    satisfy ``angle1 ≈ -angle2`` within ``10 ** -TURN_ANGLE_DIGITS``.

    Args:
        pair1: First ``(side_length, turn_angle)`` pair.
        pair2: Second ``(side_length, turn_angle)`` pair.
        dist_tol: Maximum allowed difference in side lengths.

    Returns:
        True if length and mirrored angle both match; False otherwise.
    """
    length1, angle1 = pair1
    length2, angle2 = pair2
    turn_angle_tol = 10**-TURN_ANGLE_DIGITS
    res = abs(length1 - length2) <= dist_tol and _mirror_turn_angle_equal(
        angle1, angle2, turn_angle_tol
    )
    return res


def _mirror_turn_sequences_equal(
    turns1: TurnSequence, turns2: TurnSequence, dist_tol: float
) -> bool:
    """Return True if sequences match entry-wise with mirrored angles.

    Sequences must have the same length. Each corresponding pair is
    compared with ``_mirror_turn_pair_equal`` (no cyclic shift).

    Args:
        turns1: First sequence of ``(side_length, turn_angle)`` pairs.
        turns2: Second sequence of ``(side_length, turn_angle)`` pairs.
        dist_tol: Maximum allowed difference in side lengths.

    Returns:
        True if every entry is a mirror match; False if lengths differ
        or any pair fails.
    """
    res = False
    if len(turns1) == len(turns2):
        res = all(
            _mirror_turn_pair_equal(pair1, pair2, dist_tol)
            for pair1, pair2 in zip(turns1, turns2)
        )

    return res


def _cyclic_mirror_turns_equal(
    turns1: TurnSequence, turns2: TurnSequence, dist_tol: float
) -> bool:
    """Return True if ``turns2`` matches a reflection of ``turns1`` cyclically.

    For some starting-vertex offset, each side length agrees within
    ``dist_tol`` and each turn angle of ``turns2`` is the negation of the
    corresponding angle in ``turns1`` (via ``_mirror_turn_sequences_equal``).
    Empty sequences of equal length are treated as equal.

    Args:
        turns1: First sequence of ``(side_length, turn_angle)`` pairs.
        turns2: Second sequence of ``(side_length, turn_angle)`` pairs.
        dist_tol: Maximum allowed difference in side lengths.

    Returns:
        True if some cyclic rotation of ``turns2`` is a mirror match of
        ``turns1``; False otherwise.
    """
    res = False
    n = len(turns1)
    if n == len(turns2):
        if n == 0:
            res = True
        else:
            for offset in range(n):
                rotated = turns2[offset:] + turns2[:offset]
                if _mirror_turn_sequences_equal(turns1, rotated, dist_tol):
                    res = True
                    break

    return res


def _negated_turns(turns: TurnSequence) -> list[TurnPair]:
    """Return a turn sequence with each signed angle negated.

    Used when comparing opposite winding of the same chirality: reverse
    the walk order and negate angles so left turns become right turns.

    Args:
        turns: Sequence of ``(side_length, turn_angle)`` pairs.

    Returns:
        A new list with the same side lengths and each angle replaced by
        ``-angle``, rounded to ``TURN_ANGLE_DIGITS``.
    """
    return [
        (length, round(-angle, TURN_ANGLE_DIGITS)) for length, angle in turns
    ]


def _normalize_at_origin(
    vertices: Sequence[PointType],
) -> list[tuple[float, float]]:
    """Translate vertices so the first point is at the origin.

    Args:
        vertices: Polygon vertices in walk order.

    Returns:
        A new list of ``(x, y)`` points with ``vertices[0]`` mapped to
        ``(0, 0)``; relative positions are unchanged.
    """
    origin_x, origin_y = vertices[0][:2]
    return [(x - origin_x, y - origin_y) for x, y in vertices]


def mirror_equivalent_polygons(
    poly1: Sequence[PointType],
    poly2: Sequence[PointType],
    dist_tol: float | None = None,
) -> bool:
    """Return True if ``poly2`` matches a reflection of ``poly1``.

    Both polygons are translated so their first vertex is at the origin and
    normalized to CCW+, then axis reflections of ``poly1`` are compared
    with ``equal_polygon_turns`` (without re-entering mirror checks).
    """
    normalized1 = sg.ccw_positive_vertices(_normalize_at_origin(poly1))
    normalized2 = sg.ccw_positive_vertices(_normalize_at_origin(poly2))
    res = False
    for flip_x, flip_y in ((1, -1), (-1, 1), (-1, -1)):
        mirrored = sg.ccw_positive_vertices(
            [(flip_x * x, flip_y * y) for x, y in normalized1]
        )
        if equal_polygon_turns(mirrored, normalized2, check_mirror=False):
            res = True
            break
    return res


def equal_turns(
    turns1: TurnSequence,
    turns2: TurnSequence,
    check_mirror: bool = False,
) -> bool:
    """Return True if two signed turn sequences represent congruent walks.

    Sequences are ``(side_length, turn_angle)`` pairs. Comparison is cyclic
    (starting vertex may differ). Signed angles distinguish convex and
    reflex corners. Callers should pass turns from CCW+ vertex walks.

    Args:
        turns1: First turn sequence.
        turns2: Second turn sequence.
        check_mirror: If True, also match reflections (angles negate under
            cyclic alignment), including the reverse-order listing of a
            mirrored walk.

    Returns:
        True if the sequences are equivalent under the requested checks.
    """
    dist_tol = sg.defaults["dist_tol"]
    res = _cyclic_turns_equal(turns1, turns2, dist_tol)
    if not res and check_mirror:
        res = _cyclic_mirror_turns_equal(turns1, turns2, dist_tol)
        if not res:
            reversed_turns = list(reversed(turns2))
            res = _cyclic_mirror_turns_equal(turns1, reversed_turns, dist_tol)

    return res


def equal_polygon_turns(
    polygon1: Sequence[PointType],
    polygon2: Sequence[PointType],
    check_mirror: bool = False,
) -> bool:
    """Return True if two polygons have congruent signed turn sequences.

    Each polygon is copied to CCW+ order before turn extraction. Builds
    ``polygon_turns`` for each copy and delegates to ``equal_turns``. When
    ``check_mirror`` is True and turn matching fails, also tries
    ``mirror_equivalent_polygons`` (axis reflections after normalizing the
    first vertex to the origin).

    Args:
        polygon1: First polygon as a sequence of points.
        polygon2: Second polygon as a sequence of points.
        check_mirror: If True, also match mirror images.

    Returns:
        True if the polygons are congruent under the requested checks.
    """
    verts1 = sg.ccw_positive_vertices(polygon1)
    verts2 = sg.ccw_positive_vertices(polygon2)
    turns1 = polygon_turns(verts1)
    turns2 = polygon_turns(verts2)
    res = equal_turns(turns1, turns2, check_mirror=check_mirror)
    if not res and check_mirror:
        res = mirror_equivalent_polygons(verts1, verts2)

    return res


def _polygon_vertices(polygon: PolygonLike) -> Sequence[PointType]:
    """Extract vertices from a polygon-like value.

    Args:
        polygon: A ``Shape``, a ``Group`` of shapes, or a sequence of points.

    Returns:
        A list of vertex coordinates. For a ``Group``, uses the largest
        closed outline from ``sg.polygon_verts_and_bbox``.
    """
    if isinstance(polygon, sg.Shape):
        res = list(polygon.vertices)
    elif isinstance(polygon, sg.Group):
        verts, _, _, _, _ = sg.polygon_verts_and_bbox(polygon)
        res = list(verts)
    else:
        res = list(polygon)
    return res


def congruent_polygons(
    polygon1: PolygonLike,
    polygon2: PolygonLike,
    mirror: bool = False,
) -> bool:
    """Return True if ``polygon1`` and ``polygon2`` are congruent.

    Congruence ignores translation and rotation. Vertices are normalized to
    CCW+ before signed turn sequences are compared so convex and reflex
    corners remain distinct.

    Args:
        polygon1: First polygon as a ``Shape`` or a sequence of points.
        polygon2: Second polygon as a ``Shape`` or a sequence of points.
        mirror: If True, also treat reflection (mirror image) as congruent.

    Returns:
        True if the polygons are congruent; False otherwise.
    """
    verts1 = _polygon_vertices(polygon1)
    verts2 = _polygon_vertices(polygon2)
    if len(verts1) != len(verts2):
        res = False
    else:
        res = equal_polygon_turns(verts1, verts2, check_mirror=mirror)

    return res


#######################################################################
#######################################################################
import numpy as np


def get_side_angle_sequence(vertices):
    """Generates an alternating sequence of side lengths and internal angles."""
    n = len(vertices)
    seq = []

    for i in range(n):
        # Current, previous, and next vertices
        p_prev = np.array(vertices[i - 1])
        p_curr = np.array(vertices[i])
        p_next = np.array(vertices[(i + 1) % n])

        # Calculate side length to the next vertex
        side = np.linalg.norm(p_next - p_curr)

        # Calculate internal angle at p_curr using vectors
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            seq.append(side)
            seq.append(1e6)
        else:
            cosine_angle = np.dot(v1, v2) / denom
            # Clip to handle floating point errors outside [-1, 1]
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

            seq.append(side)
            seq.append(angle)

    return seq


def is_cyclic_match(seq_a, seq_b, tol=1e-5):
    """Checks if seq_b is a cyclic shift of seq_a within a tolerance."""
    n = len(seq_b)
    # Duplicate seq_a to handle cyclic shifting
    target = seq_a + seq_a

    # Linear scan matching with floating-point tolerance
    for i in range(n):
        match = True
        for j in range(n):
            if abs(target[i + j] - seq_b[j]) > tol:
                match = False
                break
        if match:
            return True
    return False


def equivalent_polygons(
    poly_a: Sequence[PointType], poly_b: Sequence[PointType]
) -> bool:
    """Return True if the polygons are congruent (no mirror check).

    Ignores translation and rotation. Uses signed turn sequences via
    ``equal_polygon_turns``. Prefer ``congruent_polygons`` for the public API.
    """
    if len(poly_a) != len(poly_b):
        res = False
    else:
        res = equal_polygon_turns(poly_a, poly_b, check_mirror=False)

    return res


def mirrored_polygons(
    poly_a: Sequence[PointType], poly_b: Sequence[PointType]
) -> bool:
    """Return True if ``poly_a`` and ``poly_b`` are mirror images.

    Compares signed turn sequences with mirrored angles under cyclic
    alignment, including reverse order. Prefer ``congruent_polygons(...,
    mirror=True)`` for the public API.
    """
    if len(poly_a) != len(poly_b):
        res = False
    else:
        dist_tol = sg.defaults["dist_tol"]
        turns_a = polygon_turns(poly_a)
        turns_b = polygon_turns(poly_b)
        res = _cyclic_mirror_turns_equal(turns_a, turns_b, dist_tol)
        if not res:
            reversed_turns = list(reversed(turns_b))
            res = _cyclic_mirror_turns_equal(turns_a, reversed_turns, dist_tol)

    return res


# Example Usage:
# Triangle A
# poly_A = [(0, 0), (2, 0), (0, 3)]
# # Mirrored, rotated, and shifted Triangle B
# poly_B = [(5, 5), (5, 7), (2, 5)]

# print(are_polygons_mirrored(poly_A, poly_B))  # Outputs: True

#######################################################################
#######################################################################


def test():
    canvas = sg.Canvas()
    seg = sg.Shape([(0, 0), (25, 0)])
    edge = seg.translate(25, 0, reps=2)
    edges = edge.translate(0, 25, reps=3)
    edges.rotate(sg.pi / 2, about=edges.midpoint, reps=1)
    # show(edges)
    # print(len(edges.all_vertices))
    cycles = sg.segment_cycles(edges, length_bound=20)
    result = []
    for cycle in cycles[0]:
        # print(cycle, type(cycle[0][0]))
        # part = sg.Shape([*cycle], closed=True)
        seen = False
        part = [*cycle]
        for part_ in result:
            # print(part_, part)
            # if sg.equal_polygons(part_, part):
            if equal_polygon_turns(part_, part, check_mirror=True):
                # if equivalent_polygons(part_, part) or mirrored_polygons(part_, part):
                # if equal_polygon_turns(part_, part, check_mirror=True):
                seen = True
                break
        if seen:
            continue
        result.append(part)

    # print("***", len(result))

    # for part in res[:GRID_ROWS * GRID_COLUMNS]:
    #     print(part)
    #     print(polygon_turns(part))

    show_grid(
        result,
        GRID_ROWS,
        GRID_COLUMNS,
        canvas=canvas,
    )

    canvas.save("c:/tmp/hedy_hempe.svg", overwrite=True)



from math import ceil, log10, sqrt

import networkx as nx


def node_dictionaries(
    coords,
    dist_tol: float,
    debug: bool = False,
):
    """Set dictionaries for nodes and coordinates.
    d_node_coord: Dictionary of node id to coordinates.
    d_coord_node: Dictionary of coordinates to node id.

    Args:
        coords (Sequence[PointType]): Sequence of vertices.
        dist_tol (float): Distance tolerance for grouping coordinates.
        debug (bool, optional): Print node proximity diagnostics.
            Defaults to False.
    """
    n_round = max(0, ceil(log10(sqrt(2) / dist_tol)))
    d_rounded_coord = {}
    rounded = []
    for coord in coords:
        val = tuple(
            simetri.geom.points.point_utils.round_point(coord, n_round)
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
    _, close_pairs = sg.all_close_points(indexed_coordinates, dist_tol=dist_tol)
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
                simetri.geom.points.point_utils.round_point(
                    first_point, n_round
                )
            )
            second_coordinate = tuple(
                simetri.geom.points.point_utils.round_point(
                    second_point, n_round
                )
            )
            if (
                d_coord_node[first_coordinate]
                == d_coord_node[second_coordinate]
            ):
                continue
            point_distance = simetri.geom.points.point_utils.distance(
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


def segment_cycles(
    segments, length_bound: int = 10, cycle_basis=False, dist_tol=None
):
    """Given a sequence of line segments, returns all cycles."""
    if dist_tol is None:
        dist_tol = sg.defaults["dist_tol"]
    coordinates = []
    for seg in segments:
        coordinates.extend(seg)

    d_node_coord, d_coord_node, _ = node_dictionaries(coordinates, dist_tol)
    n_round = max(0, ceil(log10(sqrt(2) / dist_tol)))
    g_segments = [
        [
            d_coord_node[
                tuple(
                    simetri.geom.points.point_utils.round_point(
                        coord, n_round
                    )
                )
            ]
            for coord in seg
        ]
        for seg in segments
    ]

    nx_graph = nx.Graph()
    nx_graph.update(g_segments)
    if cycle_basis:
        cycles = [
            cycle
            for cycle in nx.cycle_basis(nx_graph)
            if len(cycle) <= length_bound
        ]
    else:
        cycles = list(nx.simple_cycles(nx_graph, length_bound=length_bound))
    res = []
    for cycle in cycles:
        res.append([d_node_coord[node] for node in cycle])

    return res, cycles


def overlapping_edges(edges):
    conn = sg.Connection
    for i, edge1 in enumerate(edges):
        for j, edge2 in enumerate(edges):
            if i == j:
                continue
            p1, p2 = edge1
            p3, p4 = edge2
            res = simetri.geom.segments.line_utils.check_intersection(
                *p1, *p2, *p3, *p4
            )
            if res[0] in [
                conn.OVERLAPS,
                conn.COINCIDENT,
                conn.CONGRUENT,
                conn.COVERS,
                conn.WITHIN,
            ]:
                return True
    return False


def test2():
    canvas = sg.Canvas()
    seg = sg.Shape([(0, 0), (25, 0)])
    edge = seg.translate(25, 0, reps=1)
    edges = edge.translate(0, 25, reps=2)
    edges.rotate(sg.pi / 2, about=edges.midpoint, reps=1)

    edges.append(sg.Shape([(0, 0), (0, 50)]))
    edges.append(sg.Shape([(25, 0), (25, 50)]))
    edges.append(sg.Shape([(50, 0), (50, 50)]))

    edges.append(sg.Shape([(0, 0), (50, 0)]))
    edges.append(sg.Shape([(0, 25), (50, 25)]))
    edges.append(sg.Shape([(0, 50), (50, 50)]))

    canvas.draw(edges)
    canvas.translate(0, -75)
    # show(edges)
    # print(len(edges.all_vertices))
    cycles = segment_cycles(edges, length_bound=10, cycle_basis=False)
    # cycles = sg.segment_cycles(edges, length_bound=10, dist_tol=.5)
    result = []
    for cycle in cycles[0]:
        # print(cycle, type(cycle[0][0]))
        # part = sg.Shape([*cycle], closed=True)
        seen = False
        part = [*cycle]
        for part_ in result:
            # print(part_, part)
            # if sg.equal_polygons(part_, part):
            if equal_polygon_turns(part_, part, check_mirror=True):
                # if equivalent_polygons(part_, part) or mirrored_polygons(part_, part):
                # if equal_polygon_turns(part_, part, check_mirror=True):
                seen = True
                break
        if seen:
            continue
        result.append(part)

    # print("***", len(result))

    # for part in res[:GRID_ROWS * GRID_COLUMNS]:
    #     print(part)
    #     print(polygon_turns(part))

    show_grid(
        result,
        GRID_ROWS,
        GRID_COLUMNS,
        canvas=canvas,
    )

    canvas.save("c:/tmp/hedy_hempe2.svg", overwrite=True)


def test3():
    canvas = sg.Canvas()
    seg = sg.Shape([(0, 0), (25, 0)])
    edge = seg.translate(25, 0, reps=2)
    edges = edge.translate(0, 25, reps=3)
    edges.rotate(sg.pi / 2, about=edges.midpoint, reps=1)

    edges.append(sg.Shape([(0, 0), (0, 75)]))
    edges.append(sg.Shape([(25, 0), (25, 75)]))
    edges.append(sg.Shape([(50, 0), (50, 75)]))
    edges.append(sg.Shape([(75, 0), (75, 75)]))

    edges.append(sg.Shape([(0, 0), (0, 50)]))
    edges.append(sg.Shape([(25, 0), (25, 50)]))
    edges.append(sg.Shape([(50, 0), (50, 50)]))
    edges.append(sg.Shape([(75, 0), (75, 50)]))

    edges.append(sg.Shape([(0, 25), (0, 75)]))
    edges.append(sg.Shape([(25, 25), (25, 75)]))
    edges.append(sg.Shape([(50, 25), (50, 75)]))
    edges.append(sg.Shape([(75, 25), (75, 75)]))

    edges.append(sg.Shape([(0, 0), (75, 0)]))
    edges.append(sg.Shape([(0, 25), (75, 25)]))
    edges.append(sg.Shape([(0, 50), (75, 50)]))
    edges.append(sg.Shape([(0, 75), (75, 75)]))

    edges.append(sg.Shape([(0, 0), (50, 0)]))
    edges.append(sg.Shape([(0, 25), (50, 25)]))
    edges.append(sg.Shape([(0, 50), (50, 50)]))
    edges.append(sg.Shape([(0, 75), (50, 75)]))

    edges.append(sg.Shape([(25, 0), (75, 0)]))
    edges.append(sg.Shape([(25, 25), (75, 25)]))
    edges.append(sg.Shape([(25, 50), (75, 50)]))
    edges.append(sg.Shape([(25, 75), (75, 75)]))

    # show(edges)
    # print(len(edges.all_vertices))
    cycles = sg.segment_cycles(edges, length_bound=10)
    # print('n_cycles', len(list(cycles[1])))
    # return
    result = []
    for cycle in cycles[0]:
        # print(cycle, type(cycle[0][0]))
        # part = sg.Shape([*cycle], closed=True)
        seen = False
        part = [*cycle]
        for part_ in result:
            # print(part_, part)
            # if sg.equal_polygons(part_, part):
            if equal_polygon_turns(part_, part, check_mirror=True):
                # if (equivalent_polygons(part_, part) or
                #     mirrored_polygons(part_, part)
                #     # or
                #     # overlapping_edges(sg.connected_pairs(part_, closed=True))
                #     ):
                # if equal_polygon_turns(part_, part, check_mirror=True):
                seen = True
                break
        if seen:
            continue
        # print('cycle:', cycle)

        angles = sg.polygon_internal_angles(part)
        if (
            0 in angles
            or 3.14 in [round(angle, 2) for angle in angles]
            or
            # 4.71 in [round(angle, 2) for angle in angles] or
            6.28 in [round(angle, 2) for angle in angles]
        ):
            continue
        # print(angles)
        result.append(part)

    # print("***", len(result))
    # for p in result:
    #     print(p)
    # print(p.edges)

    # for part in res[:GRID_ROWS * GRID_COLUMNS]:
    #     print(part)
    #     print(polygon_turns(part))

    show_grid(
        result,
        GRID_ROWS,
        GRID_COLUMNS,
        canvas=canvas,
    )

    canvas.save("c:/tmp/hedy_hempe3.pdf", overwrite=True)


def connected(
    index1: int,
    index2: int,
    n_rows: int = 3,
    n_cols: int = 3,
    diagonal_neighbors: bool = True,
) -> bool:
    row1, col1 = divmod(index1, n_cols)
    row2, col2 = divmod(index2, n_cols)
    if row2 < 0 or row2 >= n_rows or col2 < 0 or col2 >= n_cols:
        return False
    dr = abs(row1 - row2)
    dc = abs(col1 - col2)
    if dr > 1 or dc > 1 or (dr == 0 and dc == 0):
        return False
    if diagonal_neighbors:
        return True
    return dr + dc == 1


def diagonal(index1, index2, n_rows=3, n_cols=3):
    x1 = index1 // n_cols
    y1 = index1 % n_rows
    x2 = index2 // n_cols
    y2 = index2 % n_rows

    return (1, 1) == (abs(x1 - x2), abs(y1 - y2))


def any_diagonal(indices):
    for i in indices:
        for j in indices:
            if i == j:
                continue
            if diagonal(i, j):
                return True

    return False


def all_cells_connected(
    indices,
    n_rows: int = 3,
    n_cols: int = 3,
    diagonal_neighbors: bool = True,
) -> bool:
    """Return True if every cell in ``indices`` belongs to one connected group.

    Args:
        indices: Cell indices on the grid (e.g. ``range(9)`` for 3×3).
        n_rows: Number of grid rows.
        n_cols: Number of grid columns.
        diagonal_neighbors: If True, cells sharing a corner are adjacent. If
            False, only edge-adjacent cells are adjacent.

    Returns:
        True when all given cells form a single connected component (including
        when the set is empty or has one cell); False when the cells form two
        or more separate groups.
    """
    cells = set(indices)
    if len(cells) <= 1:
        res = True
    else:
        visited = set()
        stack = [next(iter(cells))]
        while stack:
            i = stack.pop()
            if i in visited:
                continue
            visited.add(i)
            for j in cells:
                if j not in visited and connected(
                    i, j, n_rows, n_cols, diagonal_neighbors
                ):
                    stack.append(j)

        res = len(visited) == len(cells)
    return res


def test4():
    squares = list(range(9))
    coords = []
    for i in range(3):
        for j in range(3):
            coords.append((j * 25, i * 25))

    # print(coords)

    square = sg.Shape([(0, 0), (25, 0), (25, 25), (0, 25)], closed=True)
    combs = combinations(coords, 4)

    canvas = sg.Canvas(back_color=sg.light_gray)
    shapes = sg.Group()

    seen = set()
    for n in [3, 4, 5, 6, 7, 8]:
        combs = combinations(coords, n)
        combs2 = list(combinations(range(9), n))

        for i, comb in enumerate(combs):
            comb_2 = combs2[i]
            if comb_2 in seen:
                continue
            else:
                seen.add(comb_2)

            if not all_cells_connected(comb_2) or not any_diagonal:
                continue
            squares = sg.Group()
            for pos in comb:
                square.move(pos)
                squares.append(square.copy())

            shapes.append(squares.copy())

    # show_grid(shapes, n_rows=20, n_columns=10, canvas=canvas, gap=40)
    print("before duplicate removal", len(shapes))
    merged = []
    for shape in shapes:
        edges = shape.all_edges
        edges = sg.Group(
            [sg.Shape(edge) for edge in remove_duplicate_edges(edges)]
        )
        merged.append(
            sg.Group([sg.Shape(edge) for edge in edges]).merge_shapes()
        )
    merged = remove_duplicate_polygons(merged)
    print("after duplicate removal", len(merged))
    #################################################################
    # print('polygon at merged[192]', [sg.distance(p1, p2) for (p1, p2) in merged[192].all_edges])
    # print('polygon at merged[144]', [sg.distance(p1, p2) for (p1, p2) in merged[144].all_edges])
    # sas_1 = get_side_angle_sequence(merged[192][0])
    # sas_2 = get_side_angle_sequence(merged[144][0])
    # print('side-angle seq', sas_1)
    # print('side-angle seq', sas_2)
    # print('is_cyclic_match(sas_1, sas_2)', is_cyclic_match(sas_1, sas_2))

    # print('polygon at shapes[128]', [sg.distance(p1, p2) for (p1, p2) in shapes[128].all_edges])
    # for j, g in enumerate(shapes):
    for index, (j, g) in enumerate(enumerate(merged)):
        x, y = get_grid_pos(index, 10, 100)
        for shape in g:
            shape.translate(x, y)
            # canvas.draw(shape)
        # edges = g.all_edges
        # edges = sg.Group([sg.Shape(edge) for edge in remove_duplicate_edges(edges)])
        # print(len(edges.merge_shapes()))
        # for i, x in enumerate(edges.merge_shapes()):
        # for i, x in enumerate(g):
        #     canvas.draw(x, fill_color=colors[i%2], alpha=.5)
        # canvas.draw(g)
        # print(len(g.all_edges))
        # print(len(remove_duplicate_edges(g.all_edges)))
        # edges = remove_duplicate_edges(g.all_edges)
        # merged = sg.Group([sg.Shape(edge) for edge in edges]).merge_shapes()
        if len(g) == 1:
            g.translate(-3.5, -3.5, reps=3)
            for i, shp in enumerate(g):
                canvas.draw(
                    shp,
                    fill=False,
                    line_alpha=0.3 + i * 0.15,
                    line_width=1.25 + i * 0.25,
                    line_dash_array=[5, 2],
                    line_dash_phase=choice([0.2, 0.3, 0.4]),
                    line_color=sg.navy,
                )
            canvas.text(str(j), shp.midpoint, font_size=18)
        else:
            canvas.draw(g, fill_color=sg.light_gray, alpha=0.5)
        # break

    canvas.save("c:/tmp/hedy_hempe4_6_.svg", overwrite=True)


def test5():
    squares = list(range(9))
    coords = []
    for i in range(3):
        for j in range(3):
            coords.append((j * 25, i * 25))

    # print(coords)

    square = sg.Shape([(0, 0), (25, 0), (25, 25), (0, 25)], closed=True)
    combs = combinations(coords, 4)

    shapes = sg.Group()

    seen = set()
    for n in [3, 4, 5, 6, 7, 8]:
        combs = combinations(coords, n)
        combs2 = list(combinations(range(9), n))

        for i, comb in enumerate(combs):
            comb_2 = combs2[i]
            if comb_2 in seen:
                continue
            else:
                seen.add(comb_2)

            if not all_connected(comb_2) or not any_diagonal:
                continue
            squares = sg.Group()
            for pos in comb:
                square.move(pos)
                squares.append(square.copy())

            shapes.append(squares.copy())
    sqr_22 = sg.Shape([(0, 0), (50, 0), (50, 50), (0, 50)], closed=True)
    sqr_22 = sqr_22.translate(25, 25, reps=1)
    sqr_22.translate(-12.5, -12.5)

    rect_12_1 = sg.Shape([(0, 0), (50, 0), (50, 75), (0, 75)], closed=True)
    rect_12_2 = sg.Shape(
        [(-25, 25), (25, 25), (25, 50), (-25, 50)], closed=True
    )
    rect_12 = sg.Group([rect_12_1, rect_12_2])
    rect_12.translate(12.5, -12.5)

    merged = []
    for shape in shapes:
        edges = shape.all_edges
        edges = sg.Group(
            [sg.Shape(edge) for edge in remove_duplicate_edges(edges)]
        )
        merged.append(
            sg.Group([sg.Shape(edge) for edge in edges]).merge_shapes()
        )
    # merged = remove_duplicate_polygons(merged)
    merged.append(rect_12)
    merged.append(sqr_22)

    hempe = [
        27,
        45,
        (35, ("rot", sg.pi)),
        (56, ("rot", sg.pi)),
        (21, ("mir", "right")),
        (31, ("mir", "bot")),
        (37, ("rot", sg.pi)),
        18,
        (70, ("rot", -sg.pi / 2)),
        (42, ("rot", sg.pi)),
        (5, ("rot", -sg.pi / 2)),
        (34, ("mir", "right")),
        -2,
        (44, ("rot", -sg.pi / 2), ("mir", "right")),
        (66, ("rot", sg.pi)),
        (35, ("rot", sg.pi)),
        (59, ("rot", sg.pi)),
        (72, ("mir", "right")),
        (33, ("rot", -sg.pi / 2)),
        17,
        (31, ("rot", sg.pi), ("mir", "right")),
        (22, ("rot", sg.pi)),
        -1,
        (23, ("mir", "right")),
    ]
    # print('polygon at shapes[128]', [sg.distance(p1, p2) for (p1, p2) in shapes[128].all_edges])
    canvas = sg.Canvas(back_color=sg.light_gray)
    # for j, g in enumerate(shapes):
    for index, (j, item) in enumerate(enumerate(hempe)):
        x, y = get_grid_pos(index, 4, 100)
        if isinstance(item, int):
            n_item = 1
        else:
            n_item = len(item)
        if n_item == 1:
            shape = merged[item].copy()
            shape.translate(x, y)
            shape.translate(-3.5, -3.5, reps=3)
            print(j, len(shape))
            for i, shp in enumerate(shape):
                canvas.draw(
                    shp,
                    fill=False,
                    line_alpha=0.3 + (i % 4) * 0.15,
                    line_width=1.25 + (i % 4) * 0.25,
                    line_dash_array=[5, 2],
                    line_dash_phase=choice([0.2, 0.3, 0.4]),
                    line_color=sg.navy,
                )
        else:
            shape = merged[item[0]].copy()
            shape.translate(x, y)
            for xform in item[1:]:
                if xform[0] == "rot":
                    shape.rotate(xform[1], about=shape.midpoint)
                elif xform[0] == "mir":
                    if xform[1] == "right":
                        shape.mirror(about=shape.vert_centerline)
                    elif xform[1] == "bot":
                        shape.mirror(about=shape.horiz_centerline)
            if item[0] == 35:
                print(shape, len(shape))
            shape.translate(-3.5, -3.5, reps=3)
            for i, shp in enumerate(shape):
                if j in [15, 20]:
                    line_color = sg.red
                else:
                    line_color = sg.navy
                canvas.draw(
                    shp,
                    fill=False,
                    line_alpha=0.3 + i * 0.15,
                    line_width=1.25 + i * 0.25,
                    line_dash_array=[5, 2],
                    line_dash_phase=choice([0.2, 0.3, 0.4]),
                    line_color=line_color,
                )

    canvas.save("c:/tmp/hedy_hempe_all.svg", overwrite=True)


def get_cell_position(
    index: int,
    n_columns: int,
    cell_width: float,
    cell_height: float,
    gap: float | None = None,
    horiz_gap: float | None = None,
    vert_gap: float | None = None,
    margin: float | None = None,
    left_margin: float | None = None,
    bot_margin: float | None = None,
    right_margin: float | None = None,
    top_margin: float | None = None,
    from_bottom_left: bool = True,
) -> tuple[float, float]:
    """Return the center of a grid cell by linear index.

    Cells are laid out in row-major order: index increases along columns first,
    then along rows. With ``from_bottom_left=True`` (default), row 0 is the
    bottom row and later rows stack upward. With ``from_bottom_left=False``,
    row 0 is the top row and later rows stack downward.

    Args:
        index: Linear cell index (0-based).
        n_columns: Number of columns in the grid.
        cell_width: Width of each cell.
        cell_height: Height of each cell.
        gap: Uniform spacing between cells. Defaults to 0. Do not pass together
            with ``horiz_gap`` or ``vert_gap``.
        horiz_gap: Horizontal spacing between adjacent cells. Defaults to
            ``gap`` when not given.
        vert_gap: Vertical spacing between adjacent cells. Defaults to ``gap``
            when not given.
        margin: Uniform offset for unset sides. Defaults to 0. May be combined
            with individual side margins; any side not passed explicitly uses
            ``margin``. Do not pass together with all four per-side margins.
        left_margin: Offset of the grid from the left. Defaults to ``margin``
            when not given.
        bot_margin: Offset of the grid from the bottom when
            ``from_bottom_left=True``. Defaults to ``margin`` when not given.
        right_margin: Right inset. With the other side margins, rejects
            ``margin`` together with all four per-side margins.
        top_margin: Offset of the grid from the top when
            ``from_bottom_left=False``. Defaults to ``margin`` when not given.
        from_bottom_left: If True, index 0 is the bottom-left cell and rows
            increase upward. If False, index 0 is the top-left cell and rows
            increase downward.

    Returns:
        ``(x, y)`` coordinates of the cell center.

    Raises:
        ValueError: If ``gap`` is passed together with ``horiz_gap`` or
            ``vert_gap``, or if ``margin`` is passed together with all four
            per-side margins.
    """
    row = index // n_columns
    col = index % n_columns

    if gap is not None and (horiz_gap is not None or vert_gap is not None):
        raise ValueError(
            "Cannot set both gap and horiz_gap/vert_gap; pass one or the other."
        )

    all_margins = (
        margin is not None
        and left_margin is not None
        and bot_margin is not None
        and right_margin is not None
        and top_margin is not None
    )
    if all_margins:
        raise ValueError(
            "Cannot set margin together with all per-side margins."
        )

    if gap is None:
        gap = 0
    if horiz_gap is None:
        horiz_gap = gap
    if vert_gap is None:
        vert_gap = gap

    if margin is None:
        margin = 0
    if left_margin is None:
        left_margin = margin
    if bot_margin is None:
        bot_margin = margin
    if top_margin is None:
        top_margin = margin

    pitch_x = cell_width + horiz_gap
    pitch_y = cell_height + vert_gap
    x = left_margin + col * pitch_x + cell_width / 2

    if from_bottom_left:
        y = bot_margin + row * pitch_y + cell_height / 2
    else:
        y = -(top_margin + row * pitch_y + cell_height / 2)

    res = (x, y)
    return res


def test6():
    squares = list(range(9))
    coords = []
    for i in range(3):
        for j in range(3):
            coords.append((j * 25, i * 25))

    square = sg.Shape([(0, 0), (25, 0), (25, 25), (0, 25)], closed=True)
    combs = combinations(coords, 4)

    canvas = sg.Canvas(back_color=sg.light_gray)
    n_columns = 10
    shapes = sg.Group()

    seen = set()
    for n in [3, 4, 5, 6, 7, 8]:
        combs = combinations(coords, n)
        combs2 = list(combinations(range(9), n))

        for i, comb in enumerate(combs):
            comb_2 = combs2[i]
            if comb_2 in seen:
                continue
            else:
                seen.add(comb_2)

            if not sg.all_cells_connected(comb_2):
                continue
            squares = sg.Group()
            for pos in comb:
                square.move(pos)
                squares.append(square.copy())

            shapes.append(squares.copy())

    # show_grid(shapes, n_rows=20, n_columns=10, canvas=canvas, gap=40)
    print("before duplicate removal", len(shapes))
    merged = []
    for shape in shapes:
        edges = shape.all_edges
        edges = sg.Group(
            [sg.Shape(edge) for edge in remove_duplicate_edges(edges)]
        )
        merged.append(
            sg.Group([sg.Shape(edge) for edge in edges]).merge_shapes()
        )

    print(
        "Check if 173 and 330 equal:",
        sg.equal_polygons(merged[173][0], merged[330][0]),
    )

    unique_shapes = sg.remove_duplicate_polygons(
        [g[0] for g in merged if len(g) == 1]
    )
    unique = [sg.Group([shape]) for shape in unique_shapes]
    # print("after duplicate removal", len(unique))
    for index, (j, g) in enumerate(enumerate(unique)):
        # x, y = get_grid_pos(index, n_columns, 100)
        x, y = sg.get_cell_position(index, n_columns, 75, 75, gap=25)
        for shape in g:
            shape.translate(x, y)
            # canvas.draw(shape)
        if len(g) == 1:
            g.translate(-3.5, -3.5, reps=3)
            for i, shp in enumerate(g):
                canvas.draw(
                    shp,
                    fill=False,
                    line_alpha=0.3 + i * 0.15,
                    line_width=1.25 + i * 0.25,
                    line_dash_array=[5, 2],
                    line_dash_phase=choice([0.2, 0.3, 0.4]),
                    line_color=sg.navy,
                )
            canvas.text(str(j), shp.midpoint, font_size=18)
        else:
            canvas.draw(g, fill_color=sg.light_gray, alpha=0.5)
        # break

    canvas.save("c:/tmp/hedy_hempe6_unique.svg", overwrite=True)


def remove_duplicate_edges(
    edges: Sequence[LineType],
    both: bool = True,
) -> list[LineType]:
    """Return edges with congruent duplicates handled.

    Candidates are filtered with axis-aligned bounding-box overlap before
    calling ``equal_edges``.

    Args:
        edges: List of line segments ``(point1, point2)``.
        both: If False, keep the first occurrence of each congruent edge.
            If True, drop every edge that has a congruent duplicate (shared
            internal edges of a polyomino are removed).

    Returns:
        A new list of edges. The input list is not modified.
    """
    dist_tol = sg.defaults["dist_tol"]

    if both:
        n = len(edges)
        if n == 0:
            res = []
        else:
            bboxes = np.empty((n, 4), dtype=float)
            for i, edge in enumerate(edges):
                x1, y1 = edge[0][:2]
                x2, y2 = edge[1][:2]
                bboxes[i] = [
                    min(x1, x2),
                    min(y1, y2),
                    max(x1, x2),
                    max(y1, y2),
                ]

            duplicate_mask = np.zeros(n, dtype=bool)
            for i in range(n):
                min_x, min_y, max_x, max_y = bboxes[i]
                overlap_mask = (
                    (bboxes[:, 2] >= min_x)
                    & (bboxes[:, 0] <= max_x)
                    & (bboxes[:, 3] >= min_y)
                    & (bboxes[:, 1] <= max_y)
                )
                for j in np.nonzero(overlap_mask)[0]:
                    if j <= i:
                        continue
                    if simetri.geom.segments.line_utils.equal_edges(
                        edges[i],
                        edges[j],
                        dist_tol=dist_tol,
                    ):
                        duplicate_mask[i] = True
                        duplicate_mask[j] = True

            res = [
                edge for i, edge in enumerate(edges) if not duplicate_mask[i]
            ]
    else:
        unique_edges = []
        bbox_array = np.empty((0, 4), dtype=float)

        for edge in edges:
            x1, y1 = edge[0][:2]
            x2, y2 = edge[1][:2]
            min_x = min(x1, x2)
            min_y = min(y1, y2)
            max_x = max(x1, x2)
            max_y = max(y1, y2)

            duplicate = False
            if bbox_array.size:
                overlap_mask = (
                    (bbox_array[:, 2] >= min_x)
                    & (bbox_array[:, 0] <= max_x)
                    & (bbox_array[:, 3] >= min_y)
                    & (bbox_array[:, 1] <= max_y)
                )
                candidate_indices = np.nonzero(overlap_mask)[0]
                for candidate_index in candidate_indices:
                    if simetri.geom.segments.line_utils.equal_edges(
                        edge,
                        unique_edges[candidate_index],
                        dist_tol=dist_tol,
                    ):
                        duplicate = True
                        break

            if not duplicate:
                unique_edges.append(edge)
                bbox_row = np.array([[min_x, min_y, max_x, max_y]], dtype=float)
                if bbox_array.size == 0:
                    bbox_array = bbox_row
                else:
                    bbox_array = np.vstack((bbox_array, bbox_row))

        res = unique_edges

    return res


def _polygon_verts_and_bbox(
    poly: PolygonLike,
) -> tuple[Sequence[PointType], float, float, float, float]:
    """Extract comparison vertices and axis-aligned bbox for a polygon.

    For a ``Group`` of squares, shared edges are removed, the boundary is
    merged, and the largest closed outline is used. If no closed outline is
    found, falls back to ``all_vertices``.

    Args:
        poly: A ``Shape``, a ``Group``, or a sequence of polygon vertices.

    Returns:
        ``(vertices, min_x, min_y, max_x, max_y)`` for congruence filtering
        and comparison. Does not modify ``poly``.
    """
    if isinstance(poly, sg.Group):
        sw = poly.b_box.southwest
        ne = poly.b_box.northeast
        min_x, min_y = sw[0], sw[1]
        max_x, max_y = ne[0], ne[1]
        boundary_edges = remove_duplicate_edges(poly.all_edges, both=True)
        merged = sg.Group(
            [sg.Shape(edge) for edge in boundary_edges]
        ).merge_shapes()
        verts = []
        largest_area = -1
        for shape in merged:
            if (
                isinstance(shape, sg.Shape)
                and shape.closed
                and shape.area > largest_area
            ):
                largest_area = shape.area
                verts = list(shape.vertices)
        if not verts:
            verts = poly.all_vertices
    elif isinstance(poly, sg.Shape):
        verts = list(poly.vertices)
        sw = poly.b_box.southwest
        ne = poly.b_box.northeast
        min_x, min_y = sw[0], sw[1]
        max_x, max_y = ne[0], ne[1]
    else:
        verts = list(poly)
        xs = [p[0] for p in verts]
        ys = [p[1] for p in verts]
        min_x, min_y = min(xs), min(ys)
        max_x, max_y = max(xs), max(ys)

    return verts, min_x, min_y, max_x, max_y


def get_island_cells(
    starting_cell_index: int,
    all_cells: Sequence[Sequence],
    empty=None,
    diagonal_neighbors: bool = True,
    from_bottom_left: bool = True,
) -> tuple[tuple, tuple]:
    """Return values and indices of non-empty cells connected to the start cell.

    ``all_cells`` is a rectangular grid stored row by row. With
    ``from_bottom_left=True``, ``all_cells[0]`` is the bottom row and index 0
    is the bottom-left cell. With ``from_bottom_left=False``, ``all_cells[0]``
    is the top row and index 0 is the top-left cell. Indexing is row-major in
    both cases.

    Args:
        starting_cell_index: Linear index of the cell to grow the island from.
        all_cells: Grid of cell values (e.g. ``((2, 3, None), ...)``).
        empty: Value(s) treated as empty. Default ``None`` means only ``None``
            is empty. Pass a collection to mark several values as empty.
        diagonal_neighbors: If True, cells sharing a corner are neighbors. If
            False, only edge-adjacent cells are neighbors.
        from_bottom_left: If True, row 0 is the bottom row; if False, row 0
            is the top row (see above).

    Returns:
        ``(values, indices)`` — parallel tuples of connected non-empty cell
        values and their indices. If the starting cell is out of range or
        empty, both tuples are empty.

    Examples:
        Grid ``((2, 3, None), (5, None, 6), (7, None, 4))`` with bottom row
        ``(2, 3, None)`` and edge-only neighbors::

            get_island_cells(0, grid, diagonal_neighbors=False)
            # ((2, 3, 5, 7), (0, 1, 3, 6))
            get_island_cells(8, grid, diagonal_neighbors=False)
            # ((4, 6), (8, 5))
            get_island_cells(7, grid)  # ((), ())  — start cell is empty

        With ``diagonal_neighbors=True``, corner-adjacent cells are included
        and this example grid becomes one connected island.
    """
    n_rows = len(all_cells)
    if n_rows == 0:
        res = ((), ())
        return res
    n_cols = len(all_cells[0])
    n_cells = n_rows * n_cols

    def cell_empty(value) -> bool:
        if isinstance(empty, (tuple, list, set, frozenset)):
            return value in empty
        if empty is None:
            return value is None
        return value == empty

    def cells_neighbors(index1: int, index2: int) -> bool:
        row1, col1 = divmod(index1, n_cols)
        row2, col2 = divmod(index2, n_cols)
        if row2 < 0 or row2 >= n_rows or col2 < 0 or col2 >= n_cols:
            return False
        dr = abs(row1 - row2)
        dc = abs(col1 - col2)
        if dr > 1 or dc > 1 or (dr == 0 and dc == 0):
            return False
        if diagonal_neighbors:
            return True
        return dr + dc == 1

    if starting_cell_index < 0 or starting_cell_index >= n_cells:
        res = ((), ())
        return res

    start_row, start_col = divmod(starting_cell_index, n_cols)
    start_value = all_cells[start_row][start_col]
    if cell_empty(start_value):
        res = ((), ())
        return res

    visited: set[int] = set()
    stack = [starting_cell_index]
    indices_list: list[int] = []
    values_list: list = []

    while stack:
        i = stack.pop()
        if i in visited:
            continue
        row, col = divmod(i, n_cols)
        value = all_cells[row][col]
        if cell_empty(value):
            continue
        visited.add(i)
        indices_list.append(i)
        values_list.append(value)
        for j in range(n_cells):
            if j not in visited and cells_neighbors(i, j):
                stack.append(j)

    res = (tuple(values_list), tuple(indices_list))
    return res


def remove_duplicate_polygons(
    polygons: Sequence[PolygonLike],
    dist_tol: float = 0.001,
    check_mirrors: bool = True,
    keep_one: bool = True,
) -> list[PolygonLike]:
    """Return polygons with congruent duplicates removed.

    Congruence uses signed turn sequences (``polygon_turns`` / ``equal_turns``)
    so reflex and convex corners remain distinct. Bounding-box overlap and
    dimension filters narrow candidates before turn comparison.

    Args:
        polygons: Polygons as ``Shape``, ``Group``, or vertex sequences.
        dist_tol: Tolerance for bbox dimension matching and side lengths.
        check_mirrors: If True, mirror-image equivalents are treated as
            duplicates (includes ``mirror_equivalent_polygons`` fallback).
        keep_one: If True, keep the first occurrence of each congruent
            polygon. If False, drop every polygon that has a congruent
            duplicate.

    Returns:
        A new list of polygon references. The input sequence is not modified;
        returned items are the same objects as in ``polygons``.
    """

    def _entry(poly: PolygonLike) -> dict:
        verts, min_x, min_y, max_x, max_y = _polygon_verts_and_bbox(poly)
        verts = sg.ccw_positive_vertices(verts)
        turns = polygon_turns(verts)
        return {
            "verts": verts,
            "turns": turns,
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "width": max_x - min_x,
            "height": max_y - min_y,
            "n_vertices": len(verts),
        }

    def _congruent(entry_a: dict, entry_b: dict) -> bool:
        if entry_a["n_vertices"] != entry_b["n_vertices"]:
            return False

        overlap = (
            entry_b["max_x"] >= entry_a["min_x"]
            and entry_a["max_x"] >= entry_b["min_x"]
            and entry_b["max_y"] >= entry_a["min_y"]
            and entry_a["max_y"] >= entry_b["min_y"]
        )
        aligned_dims = (
            np.abs(entry_b["width"] - entry_a["width"]) <= dist_tol
            and np.abs(entry_b["height"] - entry_a["height"]) <= dist_tol
        )
        rotated_dims = (
            np.abs(entry_b["width"] - entry_a["height"]) <= dist_tol
            and np.abs(entry_b["height"] - entry_a["width"]) <= dist_tol
        )
        if not (overlap or aligned_dims or rotated_dims):
            return False

        duplicate_match = equal_turns(
            entry_a["turns"],
            entry_b["turns"],
            check_mirror=check_mirrors,
        )
        if not duplicate_match and check_mirrors:
            duplicate_match = mirror_equivalent_polygons(
                entry_a["verts"],
                entry_b["verts"],
            )
        return duplicate_match

    if not keep_one:
        n = len(polygons)
        if n == 0:
            res = []
        else:
            entries = [_entry(poly) for poly in polygons]
            bboxes = np.array(
                [
                    [e["min_x"], e["min_y"], e["max_x"], e["max_y"]]
                    for e in entries
                ],
                dtype=float,
            )
            duplicate_mask = np.zeros(n, dtype=bool)
            for i in range(n):
                min_x = entries[i]["min_x"]
                min_y = entries[i]["min_y"]
                max_x = entries[i]["max_x"]
                max_y = entries[i]["max_y"]
                overlap_mask = (
                    (bboxes[:, 2] >= min_x)
                    & (bboxes[:, 0] <= max_x)
                    & (bboxes[:, 3] >= min_y)
                    & (bboxes[:, 1] <= max_y)
                )
                for j in np.nonzero(overlap_mask)[0]:
                    if j <= i:
                        continue
                    if _congruent(entries[i], entries[j]):
                        duplicate_mask[i] = True
                        duplicate_mask[j] = True

            res = [
                poly for i, poly in enumerate(polygons) if not duplicate_mask[i]
            ]
    else:
        unique_polygons = []
        unique_entries = []

        for poly in polygons:
            entry = _entry(poly)
            duplicate = False
            for unique_entry in unique_entries:
                if _congruent(entry, unique_entry):
                    duplicate = True
                    break

            if not duplicate:
                unique_polygons.append(poly)
                unique_entries.append(entry)

        res = unique_polygons

    return res


# test()
# test2()
# test3()
# test4()
# test5()
test6()


# print(all_cells_connected((0, 1, 2, 6, 7, 8)))

# canvas = sg.Canvas()
# verts = [(0.0, 0.0), (25.0, 0.0), (25.0, 25.0), (0.0, 25.0), (0.0, 50.0),
#         (25.0, 50.0), (25.0, 75.0), (0.0, 75.0)]
# shp = sg.Shape(verts, closed=True)
# edges = shp.edges

# for i, edge1 in enumerate(edges):
#     for j, edge2 in enumerate(edges):
#         if i == j:
#             continue
#         p1, p2 = edge1
#         p3, p4 = edge2
#         res = sg.check_intersection(*p1, *p2, *p3, *p4)
#         print(i, j+1, res[0])


# canvas.draw(shp, indices=True)
# canvas.save("c:/tmp/hedy_hempe4.svg", overwrite=True)
# seg = sg.Shape([(0, 0), (25, 0)])
# edge = seg.translate(25, 0, reps=1)
# edges = edge.translate(0, 25, reps=2)
# edges.rotate(sg.pi / 2, about=edges.midpoint, reps=1)

# canvas = sg.Canvas()
# for edge in edges:
#     canvas.draw(edge, indices=True)
#     canvas.translate(0, -40)
#     print(edge.vertices)
# canvas.save("c:/tmp/hedy_hempe2.svg", overwrite=True)


# coordinates = []
# for seg in edges:
#     coordinates.extend(seg)

# d_node_coord, d_coord_node, _ = node_dictionaries(coordinates, dist_tol=.05)

# print(d_coord_node)
# d = 40

# poly1 = [(0, 0), (3*d, 0), (3*d, d), (d, d), (d, 2*d), (0, 2*d)]
# poly2 = [(0, 0), (0, -2*d), (d, -2*d), (d, -d), (3*d, -d), (3*d, 0)]
# turns1 = polygon_turns(poly1)
# turns2 = polygon_turns(poly2)
# print(turns1)
# print(turns2)
# # print(list(reversed(turns2)))
# # print(equal_polygon_turns(poly1, poly2))
# print(equal_polygon_turns(poly1, poly2, check_reverse=True))
# print(equal_polygon_turns(poly1, poly2, check_mirror=True))
# print(mirror_equivalent_polygons(poly1, poly2))

# poly3 = [(0, 0), (3*d, 0), (3*d, d), (2*d, d), (2*d, 2*d), (0, 2*d)]
# poly4 = [(0, 0), (3*d, 0), (3*d, 2*d), (d, 2*d), (d, d), (0, d)]
# shp3 = sg.Shape(poly3, closed=True)
# shp4 = sg.Shape(poly4, closed=True)
# shp3.translate(4*d, 0)
# canvas.draw([shp3, shp4], fill=False)


# turns3 = polygon_turns(poly3)
# turns4 = polygon_turns(poly4)
# print(turns3)
# print(turns4)
# print(equal_polygon_turns(poly3, poly4))
# print(equal_polygon_turns(poly3, poly4, check_reverse=True))
# print(equal_polygon_turns(poly3, poly4, check_mirror=True))
# print(are_polygons_mirrored(poly3, poly4))

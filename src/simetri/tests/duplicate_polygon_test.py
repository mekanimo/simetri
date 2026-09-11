from collections import defaultdict
from itertools import combinations
from math import isclose

import simetri.geom.points.point_utils
import simetri.geom.segments.line_utils
import simetri.graphics as sg

squares = list(range(9))
coords = []
for i in range(3):
    for j in range(3):
        coords.append((j * 25, i * 25))

square = sg.Shape([(0, 0), (25, 0), (25, 25), (0, 25)], closed=True)
# combs = combinations(coords, 4)


def polygon_vertices(polygon):
    if isinstance(polygon, sg.Shape):
        res = polygon.vertices
    else:
        res = polygon

    return res


def congruent_polygons(
    polygon1: sg.PolygonLike,
    polygon2: sg.PolygonLike,
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
    verts1 = polygon_vertices(polygon1)
    verts2 = polygon_vertices(polygon2)
    if len(verts1) != len(verts2):
        res = False
    else:
        poly1_turns = polygon_turns(verts1)
        poly2_turns = polygon_turns(verts2)

        if mirror:
            if sg.equal_cycles(poly1_turns, poly2_turns):
                res = True
            else:
                poly2_turns.reverse()
                res = sg.equal_cycles(poly1_turns, poly2_turns)
        else:
            res = sg.equal_cycles(poly1_turns, poly2_turns)

    return res


def polygon_turns(vertices: sg.Sequence[sg.PointType]) -> list[float]:
    """Return the signed turn sequence of a polygon.

    For each vertex ``i``, records the length of the edge from ``i`` to
    ``i + 1`` and the signed turn angle at ``i + 1`` between that edge and
    the next (via ``angle_between_lines3``). Angles are rounded to
    ``defaults["turn_angle_digits"]``. Convex and reflex corners keep opposite signs.

    Args:
        vertices: Polygon vertices in walk order (closed; first vertex is
            not repeated).

    Returns:
        A list with alternating side_lengthe and angle values.
    """
    n = len(vertices)
    res = []
    TURN_ANGLE_DIGITS = sg.defaults["turn_angle_digits"]
    for i in range(n):
        vert = vertices[i]
        next_vert = vertices[(i + 1) % n]
        next_seg = (next_vert, vertices[(i + 2) % n])
        seg = (vert, next_vert)
        angle = simetri.geom.segments.line_utils.angle_between_lines3(
            vert, *next_seg
        )
        res.append(simetri.geom.points.point_utils.distance(*seg))
        res.append(round(angle, TURN_ANGLE_DIGITS))

    return res


def rotate_turns_to_min_edge(turns: list[float]) -> list[float]:
    """Rotate a flat ``[length, angle, ...]`` cycle to start at a min edge."""
    if len(turns) < 2:
        return list(turns)
    n_edges = len(turns) // 2
    min_i = min(range(n_edges), key=lambda i: turns[2 * i])
    start = 2 * min_i
    return turns[start:] + turns[:start]


def remove_duplicate_polygons(polygons, mirror=True, keep_one=True):
    """Return polygons with congruent duplicates removed.

    Algorithm:
        1. Group polygons by vertex count.
        2. Build ``polygon_turns`` for each polygon.
        3. Rotate each turn cycle so it starts at a smallest edge length.
        4. Within each vertex-count group, only compare polygons whose
           starting edge lengths match; use ``congruent_polygons`` there.

    Args:
        polygons: Sequence of ``Shape`` objects or vertex sequences.
        mirror: If True, treat reflections as duplicates.
        keep_one: If True, keep the first of each congruent class.
            If False, drop every polygon that has a congruent partner.

    Returns:
        A new list of polygon references (same objects as in ``polygons``).
    """
    dist_tol = sg.defaults["dist_tol"]
    entries = []
    by_n = defaultdict(list)

    for i, poly in enumerate(polygons):
        verts = polygon_vertices(poly)
        turns = rotate_turns_to_min_edge(polygon_turns(verts))
        start_len = turns[0] if turns else 0.0
        entries.append(
            {
                "poly": poly,
                "turns": turns,
                "start_len": start_len,
            }
        )
        by_n[len(verts)].append(i)

    def _same_start_len(i, j):
        return isclose(
            entries[i]["start_len"],
            entries[j]["start_len"],
            rel_tol=0.0,
            abs_tol=dist_tol,
        )

    def _are_congruent(i, j):
        if not _same_start_len(i, j):
            return False
        return congruent_polygons(
            entries[i]["poly"],
            entries[j]["poly"],
            mirror=mirror,
        )

    n = len(polygons)
    if n == 0:
        return []

    if keep_one:
        keep = [True] * n
        for indices in by_n.values():
            kept_in_group = []
            for i in indices:
                if any(_are_congruent(i, j) for j in kept_in_group):
                    keep[i] = False
                else:
                    kept_in_group.append(i)
        res = [polygons[i] for i in range(n) if keep[i]]
    else:
        has_duplicate = [False] * n
        for indices in by_n.values():
            for a, i in enumerate(indices):
                for j in indices[a + 1 :]:
                    if _are_congruent(i, j):
                        has_duplicate[i] = True
                        has_duplicate[j] = True
        res = [polygons[i] for i in range(n) if not has_duplicate[i]]

    return res


canvas = sg.Canvas()
polygons = []
index = 0

for n in (3, 4, 5, 6, 7, 8):
    combs = combinations(coords, n)
    for comb in combs:
        shapes = sg.Group()
        for pos in comb:
            sqr_ = square.copy()
            sqr_.move_to(pos)
            shapes.append(sqr_)
        edges = shapes.all_edges
        # print(edges[:2])
        polygon = sg.Group(
            [sg.Shape(edge) for edge in sg.remove_duplicate_edges(edges)]
        )
        polygon = polygon.merge_shapes()
        if len(polygon) != 1:
            continue
        polygons.append(polygon[0])
        pos = sg.get_cell_position(
            index=index,
            n_columns=10,
            cell_width=75,
            cell_height=75,
            gap=20,
            margin=20,
        )
        # canvas.draw(polygon.move_to(pos))
        index += 1


def equal(ind1, ind2, mirror=True):
    return congruent_polygons(polygons[ind1], polygons[ind2], mirror=mirror)


print(f"polygons: {len(polygons)}")
unique = sg.remove_duplicate_polygons(polygons, mirror=True, keep_one=True)
print(f"unique (mirror=True): {len(unique)}")
unique_no_mirror = sg.remove_duplicate_polygons(
    polygons, mirror=False, keep_one=True
)
print(f"unique (mirror=False): {len(unique_no_mirror)}")

# test_poly = polygons[4]
# for i, poly in enumerate(unique):
#     print(i, congruent_polygons(poly, test_poly))
# draw = True

# canvas = sg.Canvas()
# for i, poly in enumerate(polygons):
#     pos = sg.get_cell_position(
#         index=i,
#         n_columns=10,
#         cell_width=75,
#         cell_height=75,
#         gap=20,
#         margin=20,
#     )
#     canvas.draw(poly.move_to(pos))
# if draw:
#     canvas.save("c:/tmp/duplicate_polygon_test_all.svg", overwrite=True)

canvas = sg.Canvas()
for i, poly in enumerate(unique):
    pos = sg.get_cell_position(
        index=i,
        n_columns=8,
        cell_width=75,
        cell_height=75,
        gap=20,
        margin=20,
    )
    canvas.draw(poly.move_to(pos), fill=False, line_width=2)
draw = False
if draw:
    canvas.save("c:/tmp/duplicate_polygon_test.svg", overwrite=True)
################################################################################
# canvas = sg.Canvas()
# combs = combinations(coords, 8)
# res = sg.Group()
# for comb in combs:
#     shapes = sg.Group()
#     for pos in comb:
#         sqr_ = square.copy()
#         sqr_.move(pos)
#         shapes.append(sqr_)
#     res.append(shapes)


# canvas = sg.Canvas()
# for i, poly in enumerate(res):
#     cell_pos = sg.get_cell_position(
#         index=i,
#         n_columns=10,
#         cell_width=75,
#         cell_height=75,
#         gap=20,
#         margin=20,
#     )
#     canvas.draw(poly.move(cell_pos))
# draw = True
# if draw:
#     canvas.save("c:/tmp/duplicate_polygon_test_8.svg", overwrite=True)

# print(coords)
print(sg.help(sg.Rectangle))

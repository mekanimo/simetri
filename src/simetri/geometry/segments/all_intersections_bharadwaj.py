# The Line Sweep Algorithm is a geometric algorithm that detects intersections among line segments
# by conceptually sweeping a vertical line across the plane and
# maintaining an active set of segments it intersects.
# It efficiently finds intersections in O((n + k) log n) time,
# where n is the number of segments and k is the number of intersections.

# Explanation (Step-by-Step)

# Events — We create events for all segment endpoints (left & right).

# Sweep Line — Imagine a vertical line moving left to right across the plane.

# Active Set — As we move, we maintain all segments that currently intersect the sweep line.

# Intersection Check — When a new segment is added, we only need to check it with its
# immediate upper and lower neighbors in the active set.

# Result — If any intersection occurs, we record or print it.

# This reduces redundant checks and is far more efficient than comparing all pairs (O(n²)).


# --------------------------------------------------------
#   ( The Authentic PY CodeBuff )
#  ___ _                      _              _
#  | _ ) |_  __ _ _ _ __ _ __| |_ __ ____ _ (_)
#  | _ \ ' \/ _` | '_/ _` / _` \ V  V / _` || |
#  |___/_||_\__,_|_| \__,_\__,_|\_/\_/\__,_|/ |
#                                         |__/
# --------------------------------------------------------
#    Youtube: https://youtube.com/@code-with-Bharadwaj
#    Github :  https://github.com/Manu577228
# --------------------------------------------------------

import bisect


class Segment:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2


def intersect_(s1, s2):
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
    # FB
    points = []
    # FB

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

    return intersections


import time
import random


def rp(min_val=10, max_val=200):
    return (
        float(random.randint(min_val, max_val)),
        float(random.randint(min_val, max_val)),
    )


n = 1000
segments = []
for i in range(n):
    segments.append(Segment(*rp(), *rp()))

start = time.perf_counter()
# segments = [Segment(1, 1, 4, 4), Segment(1, 4, 4, 1), Segment(5, 2, 7, 2)]

result = line_sweep(segments)
end = time.perf_counter()
print(
    f"line_sweep {n} segments took: {end - start:.5f} seconds. Found {len(result)} intersections."
)

print("Intersections found:")
# for s1, s2 in result:
#     print(
#         f"Segment ({s1.x1},{s1.y1})-({s1.x2},{s1.y2}) intersects with ({s2.x1},{s2.y1})-({s2.x2},{s2.y2})"
#     )

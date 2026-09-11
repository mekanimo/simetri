"""Separate overlapping axis-aligned boxes with greedy or LP solvers.

``Box`` centers are ``(x, y)`` with ``width`` / ``height``. Public helpers
return updated center positions that reduce overlaps while preferring small
total displacement.
"""

import time
from collections import namedtuple
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pulp import (
    PULP_CBC_CMD,
    LpBinary,
    LpMinimize,
    LpProblem,
    LpVariable,
    lpSum,
    value,
)

Box = namedtuple("Box", ["x", "y", "width", "height"])


def _axis_overlaps(
    center_a: tuple[float, float],
    size_a: tuple[float, float],
    center_b: tuple[float, float],
    size_b: tuple[float, float],
) -> tuple[float, float]:
    ax, ay = center_a
    bx, by = center_b
    aw, ah = size_a
    bw, bh = size_b
    overlap_h = min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2)
    overlap_v = min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2)
    return max(0.0, overlap_h), max(0.0, overlap_v)


def _boxes_overlap(a: Box, b: Box) -> bool:
    oh, ov = _axis_overlaps(
        (a.x, a.y), (a.width, a.height), (b.x, b.y), (b.width, b.height)
    )
    return oh > 0 and ov > 0


def _separate_centers(
    centers: NDArray,
    i: int,
    j: int,
    overlap_h: float,
    overlap_v: float,
    gap: float,
) -> None:
    """Separate two center rows in ``centers`` in place.

    Args:
        centers: Center coordinate array (mutated).
        i: Row index of the first box center.
        j: Row index of the second box center.
        overlap_h: Horizontal overlap amount.
        overlap_v: Vertical overlap amount.
        gap: Extra separation to add.
    """
    if overlap_h > 0:
        amount = overlap_h / 2 + gap / 2
        if centers[i, 0] <= centers[j, 0]:
            centers[i, 0] -= amount
            centers[j, 0] += amount
        else:
            centers[i, 0] += amount
            centers[j, 0] -= amount
    if overlap_v > 0:
        amount = overlap_v / 2 + gap / 2
        if centers[i, 1] <= centers[j, 1]:
            centers[i, 1] -= amount
            centers[j, 1] += amount
        else:
            centers[i, 1] += amount
            centers[j, 1] -= amount


def push_boxes_apart_greedy(
    boxes: Sequence[Box],
    min_sep: float = 0.0,
    max_iterations: int = 48,
    debug: bool = False,
) -> list[tuple[float, float]]:
    """Separate overlapping boxes with fast axis pushes (approximate).

    Args:
        boxes: Sequence of ``Box`` namedtuples (center ``x``, ``y``, size).
        min_sep: Extra gap to leave between boxes after separation.
        max_iterations: Maximum full pairwise sweep passes.
        debug: If True, print timing diagnostics.

    Returns:
        List of updated ``(x, y)`` centers in the same order as ``boxes``.
    """
    n = len(boxes)
    if n == 0:
        return []
    if n == 1:
        return [(boxes[0].x, boxes[0].y)]

    centers = np.array([[b.x, b.y] for b in boxes], dtype=float)
    sizes = np.array([[b.width, b.height] for b in boxes], dtype=float)

    for _ in range(max_iterations):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                overlap_h, overlap_v = _axis_overlaps(
                    tuple(centers[i]),
                    tuple(sizes[i]),
                    tuple(centers[j]),
                    tuple(sizes[j]),
                )
                if overlap_h > 0 and overlap_v > 0:
                    _separate_centers(
                        centers, i, j, overlap_h, overlap_v, min_sep
                    )
                    moved = True
        if not moved:
            break

    return [(float(c[0]), float(c[1])) for c in centers]


def _overlap_components(boxes: Sequence[Box]) -> list[list[int]]:
    """Return index groups of boxes that overlap transitively."""
    n = len(boxes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if _boxes_overlap(boxes[i], boxes[j]):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def push_boxes_apart(
    boxes: Sequence[Box],
    min_sep: float = 0.0,
    bigM: int = 1e4,
    debug: bool = False,
    lp_max_boxes: int = 8,
) -> Sequence[tuple[float, float]]:
    """Separate boxes with LP on small groups, greedy on larger ones.

    Overlap-connected components of size ``<= lp_max_boxes`` use a pulp LP;
    larger components fall back to ``push_boxes_apart_greedy``.

    Args:
        boxes: Sequence of ``Box`` namedtuples (center ``x``, ``y``, size).
        min_sep: Extra gap required between non-overlapping boxes.
        bigM: Big-M constant for LP disjunctive constraints.
        debug: If True, print group and timing diagnostics.
        lp_max_boxes: Maximum group size solved with LP.

    Returns:
        Sequence of updated ``(x, y)`` centers in the same order as ``boxes``.

    Raises:
        RuntimeError: If an LP subproblem is infeasible or fails to solve.
    """
    t0 = time.perf_counter()
    n = len(boxes)
    if n <= 1:
        return [(boxes[0].x, boxes[0].y)] if n == 1 else []

    groups = _overlap_components(boxes)
    nontrivial = [g for g in groups if len(g) > 1]
    if debug:
        print(
            f"push_boxes_apart: {n} boxes, {len(nontrivial)} overlap group(s), "
            f"lp_max={lp_max_boxes}"
        )

    result = [(b.x, b.y) for b in boxes]
    for group in groups:
        if len(group) == 1:
            continue
        sub_boxes = [boxes[i] for i in group]
        t_group = time.perf_counter()
        if len(group) <= lp_max_boxes:
            method = "lp"
            solved = _push_boxes_apart_lp(
                sub_boxes, min_sep=min_sep, bigM=bigM, debug=debug
            )
        else:
            method = "greedy"
            solved = push_boxes_apart_greedy(
                sub_boxes, min_sep=min_sep, debug=debug
            )
        if debug:
            elapsed_ms = (time.perf_counter() - t_group) * 1000
            print(
                f"  push_boxes_apart {method}: group_size={len(group)} "
                f"{elapsed_ms:.2f} ms"
            )
        for idx, center in zip(group, solved):
            result[idx] = center

    if debug and nontrivial:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"push_boxes_apart total: {elapsed_ms:.2f} ms")

    return result


def _push_boxes_apart_lp(
    boxes: Sequence[Box],
    min_sep: float = 0.0,
    bigM: int = 1e4,
    debug: bool = False,
) -> list[tuple[float, float]]:
    """LP separation for a modest number of boxes."""
    t0 = time.perf_counter()
    n = len(boxes)
    W = [r.width for r in boxes]
    H = [r.height for r in boxes]
    x0 = [r.x for r in boxes]
    y0 = [r.y for r in boxes]

    prob = LpProblem("RectLayout", LpMinimize)

    x = [LpVariable(f"x_{i}") for i in range(n)]
    y = [LpVariable(f"y_{i}") for i in range(n)]

    dxp = [LpVariable(f"dxp_{i}", lowBound=0) for i in range(n)]
    dxn = [LpVariable(f"dxn_{i}", lowBound=0) for i in range(n)]
    dyp = [LpVariable(f"dyp_{i}", lowBound=0) for i in range(n)]
    dyn = [LpVariable(f"dyn_{i}", lowBound=0) for i in range(n)]

    for i in range(n):
        prob += x[i] - x0[i] == dxp[i] - dxn[i]
        prob += y[i] - y0[i] == dyp[i] - dyn[i]

    prob += lpSum(dxp[i] + dxn[i] + dyp[i] + dyn[i] for i in range(n))

    for i in range(n):
        for j in range(i + 1, n):
            left = LpVariable(f"L_{i}_{j}", cat=LpBinary)
            right = LpVariable(f"R_{i}_{j}", cat=LpBinary)
            below = LpVariable(f"B_{i}_{j}", cat=LpBinary)
            above = LpVariable(f"A_{i}_{j}", cat=LpBinary)

            prob += left + right + below + above >= 1

            Wi2 = W[i] / 2.0
            Wj2 = W[j] / 2.0
            Hi2 = H[i] / 2.0
            Hj2 = H[j] / 2.0

            prob += x[i] + Wi2 + min_sep <= x[j] - Wj2 + bigM * (1 - left)
            prob += x[j] + Wj2 + min_sep <= x[i] - Wi2 + bigM * (1 - right)
            prob += y[i] + Hi2 + min_sep <= y[j] - Hj2 + bigM * (1 - below)
            prob += y[j] + Hj2 + min_sep <= y[i] - Hi2 + bigM * (1 - above)

    prob.solve(PULP_CBC_CMD(msg=False))

    if prob.status != 1:
        raise RuntimeError(f"Solver failed or infeasible. Status={prob.status}")

    if debug:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"    lp solve: n={n} {elapsed_ms:.2f} ms")

    return [(value(x[i]), value(y[i])) for i in range(n)]

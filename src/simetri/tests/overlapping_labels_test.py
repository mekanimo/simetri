from collections import namedtuple

from pulp import (
    PULP_CBC_CMD,
    LpBinary,
    LpMinimize,
    LpProblem,
    LpVariable,
    lpSum,
    value,
)

import simetri.graphics as sg
from simetri.helpers.illustration import (
    estimate_vertex_coord_label_bbox,
    format_vertex_coord,
    vert_label_layout,
)

Box = namedtuple("Box", ["x", "y", "width", "height"])


def solve_nonoverlapping_boxes(boxes, min_sep=0.0, bigM=1e4, debug=False):
    n = len(boxes)
    W = [r.width for r in boxes]
    H = [r.height for r in boxes]
    x0 = [r.x for r in boxes]
    y0 = [r.y for r in boxes]

    prob = LpProblem("RectLayout", LpMinimize)

    # Decision variables: final centers
    x = [LpVariable(f"x_{i}") for i in range(n)]
    y = [LpVariable(f"y_{i}") for i in range(n)]

    # L1 movement decomposition for x and y
    dxp = [LpVariable(f"dxp_{i}", lowBound=0) for i in range(n)]
    dxn = [LpVariable(f"dxn_{i}", lowBound=0) for i in range(n)]
    dyp = [LpVariable(f"dyp_{i}", lowBound=0) for i in range(n)]
    dyn = [LpVariable(f"dyn_{i}", lowBound=0) for i in range(n)]

    for i in range(n):
        prob += x[i] - x0[i] == dxp[i] - dxn[i]
        prob += y[i] - y0[i] == dyp[i] - dyn[i]

    # Objective: minimize sum of L1 displacements
    prob += lpSum(dxp[i] + dxn[i] + dyp[i] + dyn[i] for i in range(n))

    # Non-overlap constraints (disjunctive: left/right/below/above)
    for i in range(n):
        for j in range(i + 1, n):
            left = LpVariable(f"L_{i}_{j}", cat=LpBinary)
            right = LpVariable(f"R_{i}_{j}", cat=LpBinary)
            below = LpVariable(f"B_{i}_{j}", cat=LpBinary)
            above = LpVariable(f"A_{i}_{j}", cat=LpBinary)

            # At least one separation relation must hold
            prob += left + right + below + above >= 1

            Wi2 = W[i] / 2.0
            Wj2 = W[j] / 2.0
            Hi2 = H[i] / 2.0
            Hj2 = H[j] / 2.0

            # If left==1 => i is completely left of j (+ min_sep)
            prob += x[i] + Wi2 + min_sep <= x[j] - Wj2 + bigM * (1 - left)

            # If right==1 => i is completely right of j
            prob += x[j] + Wj2 + min_sep <= x[i] - Wi2 + bigM * (1 - right)

            # If below==1 => i is completely below j
            prob += y[i] + Hi2 + min_sep <= y[j] - Hj2 + bigM * (1 - below)

            # If above==1 => i is completely above j
            prob += y[j] + Hj2 + min_sep <= y[i] - Hi2 + bigM * (1 - above)

    if debug:
        prob.solve()
    else:
        prob.solve(PULP_CBC_CMD(msg=False))

    if prob.status != 1:
        raise RuntimeError(f"Solver failed or infeasible. Status={prob.status}")

    return [(value(x[i]), value(y[i])) for i in range(n)]


# Example

# def get_vertex_label_bboxes(
#     shape, vertex_offset=10, vertex_font_size=6
# ) -> list[Box]:
#     """Return raw vertex coordinate label bboxes at ``vertex_offset``.

#     Returns a list of ``Box`` objects (one per vertex). Each box uses center
#     ``(x, y)`` plus ``width`` and ``height``. No overlap resolution.
#     """
#     layout = vert_label_layout(shape, vertex_offset)
#     vertices = shape.vertices
#     return [
#         Box(
#             *item["position"],
#             *estimate_vertex_coord_label_bbox(
#                 format_vertex_coord(*vertices[i]), vertex_font_size
#             ),
#         )
#         for i, item in enumerate(layout)
#     ]

# F = sg.letter_F()
# boxes = get_vertex_label_bboxes(F, vertex_offset=4, vertex_font_size=6)

# print("Initial centers:")
# for i, box in enumerate(boxes):
#     print(f"  {i}: (x={box.x}, y={box.y}), w={box.width}, h={box.height}")


# # final_centers = solve_nonoverlapping_boxes(boxes, min_sep=0.0, bigM=1000)
# final_centers = solve_nonoverlapping_boxes(
#     boxes,
#     min_sep=3,  # required gap between rectangles
#     bigM=1000,
# )

# canvas = sg.Canvas()

# for box in boxes:
#     canvas.rectangle(
#         center=(box.x, box.y), width=box.width, height=box.height, fill=False
#     )

# canvas.translate(0, -300)


# print("\nFinal centers (non-overlapping, minimal L1 push):")
# for i, (xf, yf) in enumerate(final_centers):
#     dx = xf - boxes[i].x
#     dy = yf - boxes[i].y
#     print(f"  {i}: (x={xf:.3f}, y={yf:.3f}), moved (dx={dx:.3f}, dy={dy:.3f})")
#     box = boxes[i]
#     canvas.rectangle(
#         center=(xf, yf), width=box.width, height=box.height, fill=False
#     )

# canvas.save("c:/tmp/overlapping_labels_test.svg", overwrite=True)
# # if __name__ == "__main__":
# #     boxes = get_vertex_label_bboxes(F, vertex_offset=4, vertex_font_size=6)
# #     for i, box in enumerate(boxes):
# #         print(f"v{i}: center=({box.x:.2f}, {box.y:.2f}) w={box.width:.2f} h={box.height:.2f}")

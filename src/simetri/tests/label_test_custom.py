import simetri.graphics as sg
from simetri.geom.polygons.polygon import symmetric_difference
from simetri.helpers.illustration import (
    estimate_vertex_coord_label_bbox,
    format_vertex_coord,
    vert_label_layout,
)

Box = sg.Box


class Rectangle:
    def __init__(self, id, x, y, width, height):
        self.id = id
        self.x = x  # Center X
        self.y = y  # Center Y
        self.width = width
        self.height = height


import math
from collections import defaultdict


def rects_overlap(a, b):
    # a,b: (x_center, y_center, w, h)
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (abs(ax - bx) < (aw + bw) / 2.0) and (abs(ay - by) < (ah + bh) / 2.0)


def penetration_amount(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    # overlap in x and y (positive means overlapping)
    px = (aw + bw) / 2.0 - abs(ax - bx)
    py = (ah + bh) / 2.0 - abs(ay - by)

    return px, py


def resolve_overlaps(rects, max_iterations=50):
    for _ in range(max_iterations):
        moved = False

        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                r1 = rects[i]
                r2 = rects[j]

                # Calculate distance between centers
                dx = r2.x - r1.x
                dy = r2.y - r1.y

                # Calculate minimum distance required to avoid overlap
                min_dist_x = (r1.width + r2.width) / 2
                min_dist_y = (r1.height + r2.height) / 2

                # Calculate overlap on both axes
                overlap_x = min_dist_x - abs(dx)
                overlap_y = min_dist_y - abs(dy)

                # If both overlaps are positive, the rectangles intersect
                if overlap_x > 0 and overlap_y > 0:
                    moved = True

                    # Push along the axis of shallowest penetration
                    if overlap_x < overlap_y:
                        # Determine direction
                        push_x = overlap_x if dx < 0 else -overlap_x
                        # Split the displacement equally between both rectangles
                        r1.x += push_x * 0.5
                        r2.x -= push_x * 0.5
                    else:
                        push_y = overlap_y if dy < 0 else -overlap_y
                        r1.y += push_y * 0.5
                        r2.y -= push_y * 0.5

        # Optimization: stop early if no collisions were found in this pass
        if not moved:
            break
    return rects


# # --- Example Usage ---
# if __name__ == "__main__":
#     # Define rectangles by (id, center_x, center_y, width, height)
#     rectangles = [
#         Rectangle("A", 100, 100, 50, 50),
#         Rectangle("B", 110, 110, 50, 50),  # Heavy overlap with A
#         Rectangle("C", 120, 100, 40, 40)   # Overlaps with B
#     ]

#     resolved = resolve_overlaps(rectangles)

#     for r in resolved:
#         print(f"Rectangle {r.id}: Center({r.x:.1f}, {r.y:.1f})")


def resolve_rectangles_min_push(
    rects, max_iters=2000, step=1.0, tol=1e-9, cell_scale=1.0
):
    """
    rects: list of dicts with keys {'x','y','w','h'} (x,y are centers).
    Mutates and returns the rects in-place.
    Heuristic: iteratively pushes overlapping rectangles apart.
    """
    n = len(rects)
    # Extract arrays for faster access
    xs = [r.x for r in rects]
    ys = [r.y for r in rects]
    ws = [r.width for r in rects]
    hs = [r.height for r in rects]

    # Choose spatial hash cell size based on average size
    avg_w = sum(ws) / n
    avg_h = sum(hs) / n
    cell = cell_scale * max(avg_w, avg_h, 1e-6)

    def build_grid():
        grid = defaultdict(list)
        for i in range(n):
            gx = math.floor(xs[i] / cell)
            gy = math.floor(ys[i] / cell)
            grid[(gx, gy)].append(i)
        return grid

    for it in range(max_iters):
        grid = build_grid()
        moved_any = False

        # Check local neighbors only (3x3 cells)
        for (gx, gy), ids in grid.items():
            neighbor_ids = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor_ids.extend(grid.get((gx + dx, gy + dy), []))

            # Compare pairs (i in ids, j in neighbor_ids)
            for i in ids:
                ai = (xs[i], ys[i], ws[i], hs[i])
                for j in neighbor_ids:
                    if j <= i:
                        continue
                    aj = (xs[j], ys[j], ws[j], hs[j])
                    if not rects_overlap(ai, aj):
                        continue

                    px, py = penetration_amount(ai, aj)
                    if px <= 0 or py <= 0:
                        continue

                    # direction signs
                    sx = 1.0 if xs[i] > xs[j] else -1.0
                    sy = 1.0 if ys[i] > ys[j] else -1.0

                    # Push apart along the axis with smaller penetration
                    # (often corresponds to "minimum shove" behavior)
                    if px < py:
                        # split displacement so both move equally
                        delta = 0.5 * px * step
                        xs[i] += sx * delta
                        xs[j] -= sx * delta
                    else:
                        delta = 0.5 * py * step
                        ys[i] += sy * delta
                        ys[j] -= sy * delta

                    moved_any = True

        if not moved_any:
            break

    # Write back
    for i, r in enumerate(rects):
        r["x"] = xs[i]
        r["y"] = ys[i]
    return rects


grid = sg.CircularGrid(n=20, radius=100)
p1 = grid.intersect((0, 10), (2, 14))
p2 = grid.intersect((0, 6), (2, 14))
p3 = grid.points[0]
n = 10
kernel = sg.Shape([p1, p2, p3])
petal = kernel.mirror(sg.axis_x, reps=1)
star = petal.rotate(2 * sg.pi / n, reps=n - 1)

partitions, union = symmetric_difference(star, length_bound=10)


def get_vertex_label_bboxes(
    shape, vertex_offset=10, vertex_font_size=6
) -> list[Box]:
    """Return raw vertex coordinate label bboxes at ``vertex_offset``.

    Returns a list of ``Box`` objects (one per vertex). Each box uses center
    ``(x, y)`` plus ``width`` and ``height``. No overlap resolution.
    """
    layout = vert_label_layout(shape, vertex_offset)
    vertices = shape.vertices
    return [
        Box(
            *item["position"],
            *estimate_vertex_coord_label_bbox(
                format_vertex_coord(*vertices[i]), vertex_font_size
            ),
        )
        for i, item in enumerate(layout)
    ]


# F = sg.letter_F()

input_boxes = []
for part in partitions:
    # print("part", part)
    input_boxes.extend(
        get_vertex_label_bboxes(part, vertex_offset=4, vertex_font_size=6)
    )

for box in input_boxes:
    print(box)
"""
# boxes = get_vertex_label_bboxes(
#     input_boxes, vertex_offset=4, vertex_font_size=6
# )

rectangles = []
for i, box in enumerate(input_boxes):
    rectangles.append(Rectangle(i, box.x, box.y, box.width, box.height))


resolved = resolve_rectangles_min_push(rectangles, max_iters=2000, step=1.0)

# for r in resolved:
#     print(f"Rectangle {r.id}: Center({r.x:.1f}, {r.y:.1f})")

# print("Initial centers:")
# for i, box in enumerate(boxes):
#     print(f"  {i}: (x={box.x}, y={box.y}), w={box.width}, h={box.height}")


# # final_centers = push_boxes_apart(boxes, min_sep=0.0, bigM=1000)
# final_centers = sg.push_boxes_apart(
#     boxes,
#     min_sep=3,  # required gap between rectangles
#     bigM=1000,
# )

canvas = sg.Canvas()

for box in resolved:
    print(f"({box.x}, {box.y}), {box.width}, {box.height}")
    canvas.rectangle(
        center=(box.x, box.y), width=box.width, height=box.height, fill=False
    )

canvas.translate(0, -200)


print("\nFinal centers (non-overlapping, minimal L1 push):")
for i, box in enumerate(resolved):
    # dx = xf - box.x
    # dy = yf - boxes[i].y
    # print(f"  {i}: (x={xf:.3f}, y={yf:.3f}), moved (dx={dx:.3f}, dy={dy:.3f})")
    # box = boxes[i]
    print(f"box-{i}.center:", ({box.x}, {box.y}))
    canvas.rectangle(
        center=(box.x, box.y), width=box.width, height=box.height, fill=False
    )

canvas.save("c:/tmp/label_test_custom.pdf", overwrite=True)
# if __name__ == "__main__":
#     boxes = get_vertex_label_bboxes(F, vertex_offset=4, vertex_font_size=6)
#     for i, box in enumerate(boxes):
#         print(f"v{i}: center=({box.x:.2f}, {box.y:.2f}) w={box.width:.2f} h={box.height:.2f}")
"""

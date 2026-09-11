from time import perf_counter_ns

import simetri.graphics as sg
from simetri.helpers.illustration import (
    estimate_vertex_coord_label_bbox,
    format_vertex_coord,
    vert_label_layout,
)


class Box:
    """Centered label bbox; attributes are mutable for overlap experiments."""

    def __init__(
        self,
        id: int,
        x: float,
        y: float,
        width: float,
        height: float,
    ):
        self.id = id
        self.x = x
        self.y = y
        half_width = width / 2
        half_height = height / 2
        self.min_x = self.x - half_width
        self.min_y = self.y - half_height
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return (
            f"Box(id={self.id}, x={self.x}, y={self.y}, "
            f"width={self.width}, height={self.height})"
        )

    def as_solver_box(self):
        """Convert to ``box_solver.Box`` namedtuple for ``push_boxes_apart``."""
        from simetri.helpers.box_solver import Box as SolverBox

        return SolverBox(self.x, self.y, self.width, self.height)

    def move_to(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        half_width = self.width / 2
        half_height = self.height / 2
        self.min_x = self.x - half_width
        self.min_y = self.y - half_height

    @property
    def as_shape(self):
        box = sg.Rectangle((self.x, self.y), self.width, self.height)

        return box


points = [(0, 0), (40, 0), (40, 40), (0, 40)]
square = sg.Shape(points, closed=True)
squares = square.translate(20, 30, reps=1).translate(50, 0, reps=2)
squares.translate(-10, 50, reps=2)


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
            i,
            *item["position"],
            *estimate_vertex_coord_label_bbox(
                format_vertex_coord(*vertices[i]), vertex_font_size
            ),
        )
        for i, item in enumerate(layout)
    ]


def _axis_overlap(rect_a, rect_b, direction: str = "x") -> float:
    """Return overlap depth along one axis (0 if separated on that axis)."""
    if direction == "x":
        half_a = rect_a.width / 2
        half_b = rect_b.width / 2
        delta = rect_a.x - rect_b.x
    else:
        half_a = rect_a.height / 2
        half_b = rect_b.height / 2
        delta = rect_a.y - rect_b.y
    return max(0.0, half_a + half_b - abs(delta))


def _boxes_collide(a: Box, b: Box) -> bool:
    """True when two centered boxes overlap in 2D."""
    return _axis_overlap(a, b, "x") > 0 and _axis_overlap(a, b, "y") > 0


def _group_has_collision(group: list[Box]) -> bool:
    """True when any pair in ``group`` overlaps in 2D."""
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            if _boxes_collide(group[i], group[j]):
                return True
    return False


def resolve_collision(rect_a, rect_b):
    """
    Checks collision between two AABBs and calculates the MTV.

    ``rect_a`` and ``rect_b`` use center ``(x, y)`` plus ``width`` and ``height``.
    Returns: (is_colliding, mtv_vector) where MTV is the translation for ``rect_a``.
    """
    # 1. Calculate half-sizes
    half_w_a = rect_a.width / 2
    half_w_b = rect_b.width / 2
    half_h_a = rect_a.height / 2
    half_h_b = rect_b.height / 2

    # 2. Get center points
    center_x_a = rect_a.x
    center_x_b = rect_b.x
    center_y_a = rect_a.y
    center_y_b = rect_b.y

    # 3. Calculate distance between centers
    delta_x = center_x_a - center_x_b
    delta_y = center_y_a - center_y_b

    # 4. Calculate minimal distance required to avoid overlap
    min_dist_x = half_w_a + half_w_b
    min_dist_y = half_h_a + half_h_b

    # 5. Check for separating axes
    overlap_x = min_dist_x - abs(delta_x)
    overlap_y = min_dist_y - abs(delta_y)

    # If either overlap is less than or equal to zero, there is no collision
    if overlap_x <= 0 or overlap_y <= 0:
        return False, [0.0, 0.0]

    # 6. Resolve along the axis with the SMALLEST overlap
    if overlap_x < overlap_y:
        # Push along X-axis. Determine direction based on relative position.
        push_dir = 1.0 if delta_x > 0 else -1.0
        mtv = [overlap_x * push_dir, 0.0]
    else:
        # Push along Y-axis. Determine direction based on relative position.
        push_dir = 1.0 if delta_y > 0 else -1.0
        mtv = [0.0, overlap_y * push_dir]

    return True, mtv


def resolve_collision_xy(rect_a, rect_b, direction="x"):
    """
    Checks collision between two AABBs and calculates the MTV (min. translation vector)
    in one direction only.

    ``rect_a`` and ``rect_b`` use center ``(x, y)`` plus ``width`` and ``height``.
    Returns: (is_colliding, mtv_vector) where MTV is the translation for ``rect_a``.
    """
    if direction == "x":
        half_w_a = rect_a.width / 2
        half_w_b = rect_b.width / 2

        center_x_a = rect_a.x
        center_x_b = rect_b.x
        delta_x = center_x_a - center_x_b
        min_dist_x = half_w_a + half_w_b
        overlap_x = min_dist_x - abs(delta_x)

        if overlap_x <= 0:
            res = False, [0.0, 0.0]
        else:
            push_dir = 1.0 if delta_x > 0 else -1.0
            mtv = [overlap_x * push_dir, 0.0]
            res = True, mtv

    else:
        half_h_a = rect_a.height / 2
        half_h_b = rect_b.height / 2
        center_y_a = rect_a.y
        center_y_b = rect_b.y
        delta_y = center_y_a - center_y_b
        min_dist_y = half_h_a + half_h_b
        overlap_y = min_dist_y - abs(delta_y)

        if overlap_y <= 0:
            res = False, [0.0, 0.0]
        else:
            push_dir = 1.0 if delta_y > 0 else -1.0
            mtv = [0.0, overlap_y * push_dir]
            res = True, mtv

    return res


def _overlap_groups(boxes: list[Box]) -> list[list[Box]]:
    """Group boxes that overlap transitively."""
    n = len(boxes)
    if n == 0:
        return []

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
            if _boxes_collide(boxes[i], boxes[j]):
                union(i, j)

    groups: dict[int, list[Box]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(boxes[i])
    return list(groups.values())


def _separate_along_x(group: list[Box], gap: float) -> None:
    if len(group) <= 1 or not _group_has_collision(group):
        return

    sorted_x = sorted(group, key=lambda box: box.x)
    n = len(sorted_x)
    j = n // 2
    i = j - 1
    box_left, box_right = sorted_x[i], sorted_x[j]
    overlap = _axis_overlap(box_left, box_right, "x")
    if overlap > 0:
        dx = (overlap + gap) / 2
        box_left.x -= dx
        box_right.x += dx

    for si in range(i, 0, -1):
        left, right = sorted_x[si - 1], sorted_x[si]
        overlap = _axis_overlap(left, right, "x")
        if overlap > 0:
            left.x -= overlap + gap

    for si in range(j, n - 1):
        left, right = sorted_x[si], sorted_x[si + 1]
        overlap = _axis_overlap(left, right, "x")
        if overlap > 0:
            right.x += overlap + gap


def _separate_along_y(group: list[Box], gap: float) -> None:
    if len(group) <= 1 or not _group_has_collision(group):
        return

    sorted_y = sorted(group, key=lambda box: box.y)
    n = len(sorted_y)
    j = n // 2
    i = j - 1
    box_bottom, box_top = sorted_y[i], sorted_y[j]
    overlap = _axis_overlap(box_bottom, box_top, "y")
    if overlap > 0:
        dy = (overlap + gap) / 2
        box_bottom.y -= dy
        box_top.y += dy

    # in -y direction
    for si in range(i, 0, -1):
        bottom, top = sorted_y[si - 1], sorted_y[si]
        overlap = _axis_overlap(bottom, top, "y")
        if overlap > 0:
            bottom.y -= overlap + gap

    # in +y directions
    for si in range(j, n - 1):
        bottom, top = sorted_y[si], sorted_y[si + 1]
        overlap = _axis_overlap(bottom, top, "y")
        if overlap > 0:
            top.y += overlap + gap


def resolve_overlaps(boxes: list[Box], gap: float = 4) -> list[Box]:
    """Separate overlapping boxes using minimum translation vectors."""
    for group in _overlap_groups(boxes):
        if len(group) <= 1:
            continue
        _separate_along_y(group, gap)
        _separate_along_x(group, gap)

    return boxes


def get_box(id, square):
    x, y = square.midpoint

    return Box(id, x, y, square.width, square.height)


def get_time(start, end, with_units=True):
    ms = (end - start) / 1e6
    units = ""
    if ms < 900:
        elapsed = ms
        if with_units:
            units = "milliseconds"
    else:
        elapsed = ms / 1000

        if with_units:
            units = "seconds"

    return f"{elapsed:.2f} {units}"


def resolve_all_overlaps(rectangles, gap=3, max_iterations=10):
    if gap:
        buffer = 2 * gap
        for rect in rectangles:
            rect.width += buffer
            rect.height += buffer
    for _ in range(max_iterations):
        any_collisions_resolved = False
        n = len(rectangles)

        # Check every unique pair combination
        for i in range(n):
            for j in range(i + 1, n):
                rect1 = rectangles[i]
                rect2 = rectangles[j]
                collision, mtv = resolve_collision(rect1, rect2)
                if collision:
                    dx, dy = mtv
                    rect1.x += dx / 2
                    rect2.x -= dx / 2
                    rect1.y += dy / 2
                    rect2.y -= dy / 2
                    any_collisions_resolved = True

        # If an entire pass happens with zero overlaps, stop early
        if not any_collisions_resolved:
            break
    if gap:
        buffer = 2 * gap
        for rect in rectangles:
            rect.width -= buffer
            rect.height -= buffer


# boxes = [get_box(i, squares[i]) for i in range(4)]
boxes = [get_box(i, square) for (i, square) in enumerate(squares)]
canvas = sg.Canvas()

canvas.draw([b.as_shape for b in boxes], fill=False, vertices=True)
canvas.draw_CS()

# print("before separation y")
# for i, box in enumerate(boxes):
#     print(f"box{i}.y", box.y)

canvas.translate(0, -300)
# _separate_along_y(boxes, gap=2)
start = perf_counter_ns()
resolve_all_overlaps(boxes)
end = perf_counter_ns()
print(get_time(start, end))
canvas.draw([b.as_shape for b in boxes], fill=False)
canvas.draw_CS()
# print("after separation")
# for i, box in enumerate(boxes):
#     print(f"box{i}.y", box.y)

# _separate_along_y([box1, box2], gap=5)
# print("after separation y")
# print("box1.y", box1.y)
# print("box2.y", box2.y)
# canvas.draw([box1.as_shape, box2.as_shape], fill=False)

# canvas = sg.Canvas()
# all_boxes = []
# index = 0
# for sqr in squares:
#     boxes = get_vertex_label_bboxes(sqr)
#     for box in boxes:
#         all_boxes.append(box)
#         canvas.rectangle((box.x, box.y), box.width, box.height, fill=False)
#         index += 1

# canvas.draw(
#     squares, fill=False, vertices=True, line_width=0.5, line_color=sg.gray
# )
# canvas.translate(0, -squares.height * 1.5)
# canvas.draw(squares, fill=False, line_width=0.5, line_color=sg.gray)

# _separate_along_x(all_boxes, gap=2)
# _separate_along_y(all_boxes, gap=2)

# # _separate_along_x(all_boxes, gap=2)
# _separate_along_y(all_boxes, gap=2)

# for box in all_boxes:
#     canvas.rectangle((box.x, box.y), box.width, box.height, fill=False)

# # _separate_along_y(all_boxes)
# # for box in all_boxes:
# #     canvas.rectangle((box.x, box.y), box.width, box.height, fill=False)

canvas.save("c:/tmp/rectangle_overlap_test.svg", overwrite=True)

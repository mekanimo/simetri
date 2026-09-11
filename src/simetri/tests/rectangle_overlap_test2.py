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


def resolve_all_overlaps(rectangles, gap=1, max_iters=2):
    if gap:
        buffer = 2 * gap
        for rect in rectangles:
            rect.width += buffer
            rect.height += buffer
    for _ in range(max_iters):
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
# boxes = [get_box(i, square) for (i, square) in enumerate(squares)]
boxes = []
for sqr in squares:
    boxes.extend(get_vertex_label_bboxes(sqr))

canvas = sg.Canvas()

canvas.draw(
    squares,
    line_width=0.5,
    line_color=sg.gray,
    fill=True,
    # indices=True,
    # vertices=True,
    vertex_on_hull=True,
    alpha=0.5,
)

canvas.draw(
    squares[0],
    line_width=0.5,
    line_color=sg.gray,
    fill=False,
    indices=True,
    vertices=True,
    # vertex_on_hull=True,
)
# canvas.draw([b.as_shape for b in boxes], fill=False)
canvas.draw_CS()

# canvas.translate(0, -300)

# start = perf_counter_ns()
################################
# resolve_all_overlaps(boxes, gap=1, max_iters=2)
################################
# end = perf_counter_ns()
# print(get_time(start, end))

# canvas.draw(squares, line_width=0.5, line_color=sg.gray, fill=False)
# canvas.draw([b.as_shape for b in boxes], fill=False)
# canvas.draw_CS()

# canvas.translate(0, -300)
# canvas.draw(squares, fill=False, vertices=True)

canvas.save("c:/tmp/rectangle_overlap_test.pdf", overwrite=True)

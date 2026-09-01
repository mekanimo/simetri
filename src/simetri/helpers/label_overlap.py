"""Vertex label overlap resolution via centered AABB MTV separation."""

from collections.abc import Sequence


class LabelRect:
    """Mutable centered label bbox for overlap resolution."""

    __slots__ = ("sketch", "kind", "vertex_index", "x", "y", "width", "height")

    def __init__(
        self,
        sketch,
        kind: str,
        vertex_index: int,
        x: float,
        y: float,
        width: float,
        height: float,
    ):
        self.sketch = sketch
        self.kind = kind
        self.vertex_index = vertex_index
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return (
            f"LabelRect(kind={self.kind!r}, vertex_index={self.vertex_index}, "
            f"x={self.x}, y={self.y}, width={self.width}, height={self.height})"
        )


def resolve_collision(rect_a: LabelRect, rect_b: LabelRect) -> tuple[bool, tuple[float, float]]:
    """Return whether two centered boxes overlap and the MTV for ``rect_a``."""
    half_w_a = rect_a.width / 2
    half_w_b = rect_b.width / 2
    half_h_a = rect_a.height / 2
    half_h_b = rect_b.height / 2

    delta_x = rect_a.x - rect_b.x
    delta_y = rect_a.y - rect_b.y

    overlap_x = half_w_a + half_w_b - abs(delta_x)
    overlap_y = half_h_a + half_h_b - abs(delta_y)

    if overlap_x <= 0 or overlap_y <= 0:
        return False, (0.0, 0.0)

    if overlap_x < overlap_y:
        push_dir = 1.0 if delta_x > 0 else -1.0
        mtv = (overlap_x * push_dir, 0.0)
    else:
        push_dir = 1.0 if delta_y > 0 else -1.0
        mtv = (0.0, overlap_y * push_dir)

    return True, mtv


def labels_collide(a: LabelRect, b: LabelRect) -> bool:
    """Return True when two label boxes overlap in 2D."""
    return resolve_collision(a, b)[0]


def resolve_all_overlaps(
    rectangles: Sequence[LabelRect],
    gap: float = 1,
    max_iters: int = 2,
) -> None:
    """Separate overlapping label boxes in place."""
    if gap:
        buffer = 2 * gap
        for rect in rectangles:
            rect.width += buffer
            rect.height += buffer

    for _ in range(max_iters):
        any_collisions_resolved = False
        n = len(rectangles)
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
        if not any_collisions_resolved:
            break

    if gap:
        buffer = 2 * gap
        for rect in rectangles:
            rect.width -= buffer
            rect.height -= buffer

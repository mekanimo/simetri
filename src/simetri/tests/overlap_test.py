"""Compare scaled-bbox overlap vs gap-only MTV (local copies; library untouched)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import simetri.graphics as sg
from simetri.config.settings import defaults
from simetri.extensions.polyominoes import iter_polyominoes
from simetri.helpers.illustration import (
    default_font_size_pt,
    estimate_vertex_coord_label_bbox,
    format_vertex_coord,
    vert_label_layout,
)

# --- local copies (edit freely; library untouched) ---


@dataclass
class LabelRect:
    kind: str
    vertex_index: int
    x: float
    y: float
    width: float
    height: float

    @property
    def south(self) -> float:
        return self.y - self.height / 2

    @property
    def north(self) -> float:
        return self.y + self.height / 2

    def copy(self) -> LabelRect:
        return LabelRect(
            self.kind,
            self.vertex_index,
            self.x,
            self.y,
            self.width,
            self.height,
        )


def resolve_collision(a: LabelRect, b: LabelRect):
    half_w_a, half_w_b = a.width / 2, b.width / 2
    half_h_a, half_h_b = a.height / 2, b.height / 2
    dx, dy = a.x - b.x, a.y - b.y
    ox = half_w_a + half_w_b - abs(dx)
    oy = half_h_a + half_h_b - abs(dy)
    if ox <= 0 or oy <= 0:
        return False, (0.0, 0.0)
    if ox < oy:
        push = 1.0 if dx > 0 else -1.0
        return True, (ox * push, 0.0)
    push = 1.0 if dy > 0 else -1.0
    return True, (0.0, oy * push)


def resolve_all_overlaps(rects, gap=0.0, max_iters=2):
    if gap:
        buf = 2 * gap
        for r in rects:
            r.width += buf
            r.height += buf
    for _ in range(max_iters):
        moved = False
        n = len(rects)
        for i in range(n):
            for j in range(i + 1, n):
                hit, mtv = resolve_collision(rects[i], rects[j])
                if hit:
                    dx, dy = mtv
                    rects[i].x += dx / 2
                    rects[j].x -= dx / 2
                    rects[i].y += dy / 2
                    rects[j].y -= dy / 2
                    moved = True
        if not moved:
            break
    if gap:
        buf = 2 * gap
        for r in rects:
            r.width -= buf
            r.height -= buf


def build_initial_rects(shape, offset=None) -> list[LabelRect]:
    if offset is None:
        offset = defaults["vertex_offset"]
    layout = vert_label_layout(shape, offset)
    font = default_font_size_pt("vertex_font_size")
    rects = []
    for i, item in enumerate(layout):
        x, y = item["position"][:2]
        vx, vy = shape.vertices[i][:2]
        text = format_vertex_coord(vx, vy)
        w, h = estimate_vertex_coord_label_bbox(text, font)
        rects.append(LabelRect("vertex", i, x, y, w, h))
    return rects


def _centered_vertex_tag(text: str, pos, font) -> sg.Tag:
    """Tag matching vertex-label SVG centering (not default LEFT tag_align).

    Frame stays stroked off (Tag default) so only MTV rectangles show boxes.
    """
    tag = sg.Tag(
        text, pos=pos, font_size=font, align=sg.Align.CENTER, fill=False
    )
    tag.frame.inner_sep = 0
    return tag


def build_initial_rects_from_text_bounds(
    shape, offset=None
) -> list[LabelRect]:
    """Sizes/centers from ``Tag.b_box`` after centered ``text_bounds`` measure.

    Default Tag align is LEFT, which left-edges text at ``pos`` while MTV used a
    centered AABB — that mismatch made boxes look shifted. Here we force CENTER
    and ``inner_sep=0`` so the box midpoint matches ``pos`` like vertex labels.
    """
    if offset is None:
        offset = defaults["vertex_offset"]
    layout = vert_label_layout(shape, offset)
    font = default_font_size_pt("vertex_font_size")
    rects = []
    for i, item in enumerate(layout):
        x, y = item["position"][:2]
        vx, vy = shape.vertices[i][:2]
        text = format_vertex_coord(vx, vy)
        tag = _centered_vertex_tag(text, (x, y), font)
        bb = tag.b_box
        cx, cy = bb.midpoint[:2]
        rects.append(LabelRect("vertex", i, cx, cy, bb.width, bb.height))
    return rects

def apply_scale(rects: list[LabelRect], scale: float) -> list[LabelRect]:
    out = [r.copy() for r in rects]
    for r in out:
        r.width *= scale
        r.height *= scale
    return out


def edge_gaps_vertical(sorted_by_y: list[LabelRect]) -> list[float]:
    """Gap between south of upper and north of lower (positive = separation)."""
    gaps = []
    for a, b in pairwise(sorted_by_y):
        lower, upper = (a, b) if a.y <= b.y else (b, a)
        gaps.append(upper.south - lower.north)
    return gaps


def report(tag: str, initial: list[LabelRect], final: list[LabelRect]):
    print(f"\n=== {tag} ===")
    for i, (a, b) in enumerate(zip(initial, final)):
        print(
            f"  v{i}: center ({a.x:.2f},{a.y:.2f}) -> ({b.x:.2f},{b.y:.2f})  "
            f"d=({b.x - a.x:.2f},{b.y - a.y:.2f})  "
            f"N/S [{b.south:.2f}, {b.north:.2f}]  h={b.height:.2f} w={b.width:.2f}"
        )
    by_y = sorted(final, key=lambda r: r.y)
    vg = edge_gaps_vertical(by_y)
    print(f"  vertical edge gaps (adjacent by y): {[round(g, 2) for g in vg]}")
    max_shift = max(
        abs(b.x - a.x) + abs(b.y - a.y) for a, b in zip(initial, final)
    )
    print(f"  max |shift| = {max_shift:.2f}")


def label_texts(shape) -> list[str]:
    return [format_vertex_coord(*v[:2]) for v in shape.vertices]


def print_size_compare(shape) -> None:
    estimated = build_initial_rects(shape)
    from_bounds = build_initial_rects_from_text_bounds(shape)
    texts = label_texts(shape)
    print("\n=== estimate vs Tag.text_bounds sizes ===")
    for i, (est, tb) in enumerate(zip(estimated, from_bounds)):
        print(
            f"  v{i} {texts[i]!r}: "
            f"estimate w={est.width:.2f} h={est.height:.2f}  "
            f"text_bounds w={tb.width:.2f} h={tb.height:.2f}"
        )


def draw_resolved(
    shape,
    rects: list[LabelRect],
    path: str,
    title: str,
    mtv_color,
) -> None:
    """Draw shape, MTV boxes, and Tags (no Tag frames)."""
    canvas = sg.Canvas()
    canvas.draw(shape, fill=False, line_width=0.5, line_color=sg.black)
    font = default_font_size_pt("vertex_font_size")
    texts = label_texts(shape)
    boxes = []
    tags = []
    for r in rects:
        boxes.append(sg.Rectangle((r.x, r.y), r.width, r.height))
        tags.append(
            _centered_vertex_tag(texts[r.vertex_index], (r.x, r.y), font)
        )
    canvas.draw(boxes, fill=False, line_width=0.5, line_color=mtv_color)
    canvas.draw(tags)
    canvas.save(path, overwrite=True)
    print(f"saved {path} ({title}); MTV boxes={mtv_color}")


def run_compare(
    shape,
    scale=0.7,
    gap=1.0,
    max_iters=2,
    scale_path="c:/tmp/overlap_scale.svg",
    gap_path="c:/tmp/overlap_gap.svg",
    text_bounds_path="c:/tmp/overlap_text_bounds.svg",
):
    print_size_compare(shape)
    print(
        "\nColor legend (MTV boxes only; Tag frames stay off):\n"
        "  scale:        blue\n"
        "  full+gap:     orange\n"
        "  text_bounds:  green"
    )

    initial = build_initial_rects(shape)
    print(f"\nn_labels={len(initial)}  estimated sizes from illustration helpers")

    # A: old-style shrink, no gap
    scaled = apply_scale(initial, scale)
    resolve_all_overlaps(scaled, gap=0.0, max_iters=max_iters)
    report(f"scale={scale}, gap=0", initial, scaled)
    draw_resolved(
        shape,
        scaled,
        scale_path,
        f"scale={scale}, gap=0",
        mtv_color=sg.blue,
    )

    # B: full boxes, gap only
    full = [r.copy() for r in initial]
    resolve_all_overlaps(full, gap=gap, max_iters=max_iters)
    report(f"scale=1 (full), gap={gap}", initial, full)
    draw_resolved(
        shape,
        full,
        gap_path,
        f"scale=1, gap={gap}",
        mtv_color=sg.orange,
    )

    # C: Tag.text_bounds sizes, gap only
    tb_initial = build_initial_rects_from_text_bounds(shape)
    tb = [r.copy() for r in tb_initial]
    resolve_all_overlaps(tb, gap=gap, max_iters=max_iters)
    report(f"Tag.text_bounds, gap={gap}", tb_initial, tb)
    draw_resolved(
        shape,
        tb,
        text_bounds_path,
        f"Tag.text_bounds, gap={gap}",
        mtv_color=sg.green,
    )


if __name__ == "__main__":
    polyo = next(iter_polyominoes(n=6, size=15))[0]
    run_compare(
        polyo,
        scale=0.7,
        gap=float(defaults["vertices_label_overlap_gap"]),
        scale_path="c:/tmp/overlap_scale.svg",
        gap_path="c:/tmp/overlap_gap.svg",
        text_bounds_path="c:/tmp/overlap_text_bounds.svg",
    )

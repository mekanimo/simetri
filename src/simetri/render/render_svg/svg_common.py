"""Shared helper utilities for SVG rendering modules."""

from ...base.all_enums import Extent
from .svg_mask import has_mask_style
from .svg_sketch_utils import sketch_attrib


def _clip_line_to_rect(start, end, rect, draw_type):
    """Clip a line segment/ray/infinite line to a rectangle.

    Args:
        start: Line start point.
        end: Line end point.
        rect: ``(xmin, ymin, xmax, ymax)`` clip rectangle, or None.
        draw_type: ``Extent`` value.

    Returns:
        tuple: Possibly clipped ``(start, end)`` points.
    """
    if rect is None:
        return start, end

    x1, y1 = start[:2]
    x2, y2 = end[:2]
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return start, end

    xmin, ymin, xmax, ymax = rect
    t_min = float("-inf")
    t_max = float("inf")

    if abs(dx) < 1e-12:
        if x1 < xmin or x1 > xmax:
            return start, end
    else:
        tx1 = (xmin - x1) / dx
        tx2 = (xmax - x1) / dx
        t_min = max(t_min, min(tx1, tx2))
        t_max = min(t_max, max(tx1, tx2))

    if abs(dy) < 1e-12:
        if y1 < ymin or y1 > ymax:
            return start, end
    else:
        ty1 = (ymin - y1) / dy
        ty2 = (ymax - y1) / dy
        t_min = max(t_min, min(ty1, ty2))
        t_max = min(t_max, max(ty1, ty2))

    if t_min > t_max:
        return start, end

    if draw_type == Extent.INFINITE:
        t0, t1 = t_min, t_max
    elif draw_type == Extent.RAY:
        t0, t1 = max(0.0, t_min), t_max
        if t0 > t1:
            return start, end
    else:
        return start, end

    p0 = (x1 + t0 * dx, y1 + t0 * dy)
    p1 = (x1 + t1 * dx, y1 + t1 * dy)
    return p0, p1


def get_clip_mask_attrs(sketch):
    """Build SVG ``clip-path`` / ``mask`` attribute strings for a sketch.

    Args:
        sketch: Sketch that may reference a clip path or opacity mask.

    Returns:
        tuple: ``(clip_attr, mask_attr)`` strings (possibly empty).
    """
    clip_attr = ""
    clip = sketch_attrib(sketch, "clip")
    mask = sketch_attrib(sketch, "mask")
    if clip is True and mask is not None:
        clippath_id = f"clippath_{id(sketch)}"
        clip_attr = f' clip-path="url(#{clippath_id})"'

    mask_attr = ""
    if mask is not None and (clip is not True):
        mask_attr = f' mask="url(#mask_{sketch.id})"'
    elif has_mask_style(sketch) and (clip is not True):
        mask_attr = f' mask="url(#mask_{sketch.id})"'

    return clip_attr, mask_attr

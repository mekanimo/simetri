"""Shared TikZ helpers used across split TikZ modules."""

from __future__ import annotations

from math import atan2, degrees
from types import SimpleNamespace

from ...geom.bbox import bounding_box
from ...base.all_enums import Anchor, BackStyle, Extent, Types
from ...config.settings import defaults
from .tikz_utils import _get_gradient_shading_options, get_clip_code


def anchor_to_tikz(anchor: Anchor | None) -> str | None:
    """Convert an ``Anchor`` enum value to a TikZ anchor name.

    Args:
        anchor: Simetri anchor, or None.

    Returns:
        TikZ anchor string (for example ``"north east"``), or None.
    """
    if anchor is None:
        return None

    anchor_map = {
        Anchor.BASE_EAST: "base east",
        Anchor.BASE_WEST: "base west",
        Anchor.BOTTOM: "south",
        Anchor.LEFT: "west",
        Anchor.NORTHEAST: "north east",
        Anchor.NORTHWEST: "north west",
        Anchor.RIGHT: "east",
        Anchor.SOUTHEAST: "south east",
        Anchor.SOUTHWEST: "south west",
        Anchor.TOP: "north",
    }
    return anchor_map.get(anchor, anchor.value)


def _pgf_gray(transparency: int) -> str:
    if transparency <= 0:
        return "white"
    if transparency >= 100:
        return "black"
    return f"black!{transparency}"


def _parse_offset(offset):
    if isinstance(offset, (int, float)):
        return float(offset)
    if isinstance(offset, str) and offset.endswith("%"):
        return float(offset[:-1]) / 100.0
    return float(offset)


def _effective_alpha_from_stop(stop):
    if isinstance(stop, dict):
        offset = _parse_offset(stop["offset"])
        stop_opacity = stop.get(
            "stop-opacity", stop.get("stop_opacity", stop.get("opacity", 1.0))
        )
        alpha = max(0.0, min(1.0, float(stop_opacity)))
        return offset, alpha

    offset = _parse_offset(stop[0])
    if len(stop) >= 3:
        stop_opacity = stop[2]
    elif isinstance(stop[1], (int, float)):
        stop_opacity = stop[1]
    else:
        stop_opacity = 1.0
    alpha = max(0.0, min(1.0, float(stop_opacity)))
    return offset, alpha


def _build_fading_code(fade_id, stops, x1, y1, x2, y2):
    parsed_stops = [_effective_alpha_from_stop(stop) for stop in stops]
    parsed_stops.sort(key=lambda value: value[0])
    if not parsed_stops:
        parsed_stops = [(0.0, 1.0), (1.0, 1.0)]

    shade_id = f"{fade_id}Shade"
    color_stops = []
    for offset, alpha in parsed_stops:
        offset = max(0.0, min(1.0, float(offset)))
        position = int(round(offset * 100))
        transparency = int(round((1.0 - float(alpha)) * 100))
        color_stops.append(f"color({position}bp)=({_pgf_gray(transparency)})")

    if parsed_stops[0][0] > 0.0:
        first_transparency = int(round((1.0 - float(parsed_stops[0][1])) * 100))
        color_stops.insert(0, f"color(0bp)=({_pgf_gray(first_transparency)})")
    if parsed_stops[-1][0] < 1.0:
        last_transparency = int(round((1.0 - float(parsed_stops[-1][1])) * 100))
        color_stops.append(f"color(100bp)=({_pgf_gray(last_transparency)})")

    shading_decl = "; ".join(color_stops)
    angle = degrees(atan2(y2 - y1, x2 - x1))
    return (
        f"\\pgfdeclarehorizontalshading{{{shade_id}}}{{100bp}}{{{shading_decl}}}\n"
        f"\\tikzfadingfrompicture[name={fade_id}]\n"
        f"  \\shade[shading={shade_id}, shading angle={angle:.2f}] (0, 0) rectangle (100bp, 100bp);\n"
        f"\\endtikzfadingfrompicture\n"
    )


def _get_scope_fading_path(mask_shape, fade_id):
    bbox = mask_shape.b_box
    x1, y1 = bbox.southwest[:2]
    x2, y2 = bbox.northeast[:2]
    return f"\\path [scope fading={fade_id}] ({x1}, {y1}) rectangle ({x2}, {y2});\n"


def _mask_scope_parts(sketch, fade_id=None):
    if sketch.subtype == Types.MASKED_SKETCH:
        mask_data = sketch.mask
        mask = mask_data.shape
        clip = mask_data.opacity >= 1.0 and mask_data.stops is None
        mask_opacity = mask_data.opacity
        mask_stops = mask_data.stops
        mask_axis = mask_data.axis
        if mask_stops is not None and mask_axis is None:
            mask_axis = defaults["mask_axis"]
    else:
        if "mask" not in sketch.__dict__:
            return "", ""
        mask_data = sketch.mask
        if mask_data is None:
            return "", ""

        if mask_data.type == Types.MASK:
            mask = mask_data.shape
            clip = mask_data.opacity >= 1.0 and mask_data.stops is None
            mask_opacity = mask_data.opacity
            mask_stops = mask_data.stops
            mask_axis = mask_data.axis
            if mask_stops is not None and mask_axis is None:
                mask_axis = defaults["mask_axis"]
        else:
            mask = mask_data
            clip = True
            mask_opacity = 1.0
            mask_stops = None
            mask_axis = None

    if mask is None:
        return "", ""
    clip_code = get_clip_code(SimpleNamespace(mask=mask))

    if clip:
        return f"\\begin{{scope}}\n{clip_code}", "\\end{scope}\n"

    if mask_stops is not None:
        axis_start, axis_end = mask_axis
        mask_bbox = mask.b_box
        bbox_x, bbox_y = mask_bbox.southwest[:2]
        bbox_width = mask_bbox.width
        bbox_height = mask_bbox.height
        x1 = bbox_x + float(axis_start[0]) * bbox_width
        y1 = bbox_y + float(axis_start[1]) * bbox_height
        x2 = bbox_x + float(axis_end[0]) * bbox_width
        y2 = bbox_y + float(axis_end[1]) * bbox_height
        fade_name = fade_id or f"simetriMaskFade{id(sketch)}"
        start_code = _build_fading_code(fade_name, mask_stops, x1, y1, x2, y2)
        start_code += "\\begin{scope}[transparency group,blend mode=normal]\n"
        start_code += _get_scope_fading_path(mask, fade_name)
        start_code += clip_code
        return start_code, "\\end{scope}\n"

    if mask_opacity not in [None, 1]:
        return (
            f"\\begin{{scope}}[opacity={mask_opacity}]\n{clip_code}",
            "\\end{scope}\n",
        )

    return "", ""


def get_draw(sketch):
    """Return the TikZ draw command for sketches."""
    decision_table = {
        (True, True, True, True): "\\shadedraw",
        (True, True, True, False): "\\filldraw",
        (True, True, False, True): "\\shade",
        (True, True, False, False): "\\fill",
        (True, False, True, True): "\\draw",
        (True, False, True, False): "\\draw",
        (True, False, False, True): False,
        (True, False, False, False): False,
        (False, True, True, True): "\\draw",
        (False, True, True, False): "\\draw",
        (False, True, False, True): False,
        (False, True, False, False): False,
        (False, False, True, True): "\\draw",
        (False, False, True, False): "\\draw",
        (False, False, False, True): False,
        (False, False, False, False): False,
    }
    if hasattr(sketch, "markers_only") and sketch.markers_only:
        result = "\\draw"
    else:
        gradient_options = _get_gradient_shading_options(sketch)
        has_gradient = bool(gradient_options)
        if hasattr(sketch, "back_style"):
            shading = sketch.back_style == BackStyle.SHADING or has_gradient
        else:
            shading = has_gradient
        if not hasattr(sketch, "closed"):
            closed = False
        else:
            closed = sketch.closed
        if not hasattr(sketch, "fill"):
            fill = False
        else:
            fill = sketch.fill
            if fill is None:
                fill = False
        if not hasattr(sketch, "stroke"):
            stroke = False
        else:
            stroke = sketch.stroke
            if stroke is None:
                stroke = False

        result = decision_table[(closed, fill, stroke, shading)]

    return result


def get_begin_scope(ind=None):
    """Return \\begin{scope}[every node/.append style=nodestyle{ind}]."""
    if ind is None:
        result = ""
    else:
        result = f"\\begin{{scope}}[every node/.append style=nodestyle{ind}]\n"

    return result


def get_end_scope():
    """Return \\end{scope}."""
    return "\\end{scope}\n"


def _line_limits(canvas):
    if canvas is None:
        return None
    limits = None
    if canvas.limits is not None:
        limits = tuple(canvas.limits)
    elif canvas._all_vertices:
        bbox = bounding_box(canvas._all_vertices)
        southwest_x, southwest_y = bbox.southwest[:2]
        northeast_x, northeast_y = bbox.northeast[:2]
        limits = (southwest_x, southwest_y, northeast_x, northeast_y)

    page = canvas.active_page
    sketches = page.sketches
    for sketch in sketches:
        if (
            sketch.subtype == Types.HELPLINES_SKETCH
            and hasattr(sketch, "pos")
            and hasattr(sketch, "width")
            and hasattr(sketch, "height")
        ):
            x, y = sketch.pos[:2]
            candidate = (x, y, x + sketch.width, y + sketch.height)
            if limits is None:
                limits = candidate
            else:
                limits = (
                    min(limits[0], candidate[0]),
                    min(limits[1], candidate[1]),
                    max(limits[2], candidate[2]),
                    max(limits[3], candidate[3]),
                )

    return limits


def _clip_line_to_rect(start, end, rect, draw_type):
    if rect is None:
        return start, end

    x1, y1 = start[:2]
    x2, y2 = end[:2]
    delta_x = x2 - x1
    delta_y = y2 - y1
    if abs(delta_x) < 1e-12 and abs(delta_y) < 1e-12:
        return start, end

    xmin, ymin, xmax, ymax = rect
    t_min = float("-inf")
    t_max = float("inf")

    if abs(delta_x) < 1e-12:
        if x1 < xmin or x1 > xmax:
            return start, end
    else:
        tx1 = (xmin - x1) / delta_x
        tx2 = (xmax - x1) / delta_x
        t_min = max(t_min, min(tx1, tx2))
        t_max = min(t_max, max(tx1, tx2))

    if abs(delta_y) < 1e-12:
        if y1 < ymin or y1 > ymax:
            return start, end
    else:
        ty1 = (ymin - y1) / delta_y
        ty2 = (ymax - y1) / delta_y
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

    point0 = (x1 + t0 * delta_x, y1 + t0 * delta_y)
    point1 = (x1 + t1 * delta_x, y1 + t1 * delta_y)
    return point0, point1

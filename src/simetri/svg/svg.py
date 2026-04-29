from __future__ import annotations

import io
import re
from math import degrees, cos, sin, ceil
from typing import List, Union
from dataclasses import dataclass, field
from types import SimpleNamespace
import warnings
import html

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..graphics.common import common_properties
from ..graphics.all_enums import (
    BackStyle,
    FontSize,
    FontFamily,
    MarkerType,
    ShadeType,
    Types,
    TexLoc,
    FrameShape,
    DocumentClass,
    LineCap,
    LineJoin,
    Align,
    Anchor,
    ArrowLine,
    BlendMode,
    get_enum_value,
    LineWidth,
    LineDashArray,
    Extent,
    SvgUnits,
    SvgMaskType,
)
from ..graphics.bbox import bounding_box
from ..geometry.geometry import (
    homogenize,
    round_point,
    close_points2,
    vert_label_positions,
)
from ..colors.colors import black, white
from ..settings.settings import defaults, svg_defaults
from ..canvas.style_map import shape_style_map, line_style_map, marker_style_map
from ..graphics.sketch import TagSketch, ShapeSketch, MaskSketch, ScopeGroup
from .filters import SVG_Filter
from .svg_sketch import SVG_Mask
from .svg_utils import round_corners

from ..colors.colors import Color, check_color
from .svg_colors import color_to_matplotlib, color_to_svg


from PIL import ImageFont


def sketch_attrib(sketch, attrib):
    try:
        return object.__getattribute__(sketch, attrib)
    except AttributeError:
        return defaults.get(attrib)


def get_alpha(sketch, alpha_attrib):
    return sketch_attrib(sketch, alpha_attrib)


def get_color(sketch, color_attrib):
    return sketch_attrib(sketch, color_attrib)


def get_text_size(text, font_name, font_size):
    """Get accurate text dimensions using PIL.

    Args:
        text: The text to measure
        font_name: Font family name or FontFamily enum
        font_size: Font size in points

    Returns:
        tuple: (width, height) of the text
    """
    mult = 1.0  # Scaling multiplier for default font

    # Check if font_name is a FontFamily enum (generic font)
    if isinstance(font_name, FontFamily):
        # For generic font families, use default font with scaling
        font = ImageFont.load_default()
        mult = font_size / 10  # Default font is ~10 pixels
    else:
        # Try to load specific font file
        try:
            font = ImageFont.truetype(f"{font_name}.ttf", size=font_size)
        except OSError as e:
            # If specific font not found, use default font with scaling
            warnings.warn(
                f"Could not load font '{font_name}.ttf': {e}. Using default font with scaling."
            )
            font = ImageFont.load_default()
            mult = font_size / 10  # Default font is ~10 pixels

    # getbbox gives the most accurate bounding box
    bbox = font.getbbox(text)
    width = (bbox[2] - bbox[0]) * mult
    height = (bbox[3] - bbox[1]) * mult

    return (width, height)


def get_text_size2(text, font_path, font_size):
    font = ImageFont.truetype(font_path, font_size)
    _, descent = font.getmetrics()
    text_width = font.getmask(text).getbbox()[2]
    text_height = font.getmask(text).getbbox()[3] + descent
    return text_width, text_height


def draw_shape_sketch(sketch, ind=None):
    """Draws a shape sketch.

    Args:
        sketch: The shape sketch object.
        ind: Optional index for the shape sketch.

    Returns:
        str: The TikZ code for the shape sketch.
    """
    d_subtype_draw = {
        Types.ARC_SKETCH: draw_arc_sketch,
        Types.BEZIER_SKETCH: draw_bezier_sketch,
        Types.CIRCLE_SKETCH: draw_circle_sketch,
        Types.ELLIPSE_SKETCH: draw_ellipse_sketch,
        Types.LINE_SKETCH: draw_line_sketch,
    }
    subtype = sketch_attrib(sketch, "subtype")
    draw_markers = sketch_attrib(sketch, "draw_markers")
    marker_type = sketch_attrib(sketch, "marker_type")
    indices = sketch_attrib(sketch, "indices")
    smooth = sketch_attrib(sketch, "smooth")

    if subtype in d_subtype_draw:
        res = d_subtype_draw[subtype](sketch)
    elif (draw_markers and marker_type == MarkerType.INDICES) or indices:
        res = draw_shape_sketch_with_indices(sketch, ind)
    elif draw_markers or smooth:
        res = draw_shape_sketch_with_markers(sketch)
    else:
        res = draw_sketch(sketch)

    return res


def draw_sketch(sketch):
    """Draws a plain shape sketch.

    Args:
        sketch: The shape sketch object.

    Returns:
        str: The TikZ code for the plain shape sketch.
    """
    res = draw_sketch(sketch)
    if not res:
        return ""
    options = []
    back_style = sketch_attrib(sketch, "back_style")
    fill = sketch_attrib(sketch, "fill")
    closed = sketch_attrib(sketch, "closed")
    smooth = sketch_attrib(sketch, "smooth")

    if back_style == BackStyle.PATTERN and fill and closed:
        options += get_pattern_options(sketch)
    if sketch_attrib(sketch, "stroke"):
        options += get_line_style_options(sketch)
    if closed and fill:
        options += get_fill_style_options(sketch)
    if smooth:
        options += ["smooth"]
    if back_style == BackStyle.SHADING and fill and closed:
        options += get_shading_options(sketch)
    options = ", ".join(options)
    if options:
        res += f"[{options}]"
    vertices = sketch_attrib(sketch, "vertices")
    n = len(vertices)
    str_lines = [f"{vertices[0]}"]
    for i, vertex in enumerate(vertices[1:]):
        if (i + 1) % 8 == 0:
            if i == n - 1:
                str_lines.append(f"-- {vertex} \n")
            else:
                str_lines.append(f"\n\t-- {vertex} ")
        else:
            str_lines.append(f"-- {vertex} ")
    if closed:
        str_lines.append("-- cycle;\n")
    else:
        str_lines.append(";\n")
    if res:
        res += "".join(str_lines)
    else:
        res = "".join(str_lines)
    return res


def append_non_default_style_options(options, sketch, style_map):
    """Append CSS style options for sketch attributes that differ from defaults.

    Args:
        options: List of style option strings.
        sketch: Sketch object.
        style_map: Dictionary mapping sketch attribute name to (css_name, default_key).
    """
    sketch_dict = sketch_attrib(sketch, "__dict__")
    for attrib_name, (css_name, default_key) in style_map.items():
        if attrib_name not in sketch_dict:
            continue
        value = sketch_dict[attrib_name]
        default_value = defaults[default_key]
        if value != default_value:
            options.append(f"{css_name}: {value};")


def get_line_style_options(sketch, exceptions=None):
    """Returns the options for the line style.

    Args:
        sketch: The sketch object.
        exceptions: Optional exceptions for the line style options.

    Returns:
        list: The line style options as a list.
    """

    options = []
    stroke = sketch_attrib(sketch, "stroke")
    if stroke:
        line_color = color_to_svg(get_color(sketch, "line_color"), "line_color")
    else:
        line_color = "none"
    options.append(f"stroke: {line_color};")

    line_alpha = get_alpha(sketch, "line_alpha")
    if line_alpha is not None and line_alpha != defaults["line_alpha"]:
        options.append(f"stroke-opacity: {line_alpha};")

    style_map = {
        "line_width": ("stroke-width", "line_width"),
        "line_cap": ("stroke-linecap", "line_cap"),
        "line_join": ("stroke-linejoin", "line_join"),
    }
    append_non_default_style_options(options, sketch, style_map)

    # miter limit
    miter_limit = sketch_attrib(sketch, "miter_limit")
    if miter_limit:
        options.append(f"stroke-miterlimit: {miter_limit}")

    # dash pattern
    line_dash_array = sketch_attrib(sketch, "line_dash_array")
    if line_dash_array:
        options.append(f"stroke-dasharray: {get_dash_pattern(line_dash_array)}")

    return " ".join(options)


def get_fill_style_options(sketch, shape_type, exceptions=None, frame=False):
    """Returns the options for the fill style.

    Args:
        sketch: The sketch object.
        exceptions: Optional exceptions for the fill style options.
        frame: Optional flag for frame fill style.

    Returns:
        list: The fill style options as a list.
    """

    options = []
    fill = sketch_attrib(sketch, "fill")
    if fill and shape_type != "polyline":
        fill_color = color_to_svg(get_color(sketch, "fill_color"), "fill_color")
    else:
        fill_color = "none"

    options.append(f"fill: {fill_color};")

    fill_alpha = get_alpha(sketch, "fill_alpha")
    if fill_alpha != defaults["fill_alpha"]:
        options.append(f"fill-opacity: {fill_alpha};")

    if sketch_attrib(sketch, "even_odd"):
        options.append("fill-rule: evenodd;")

    return " ".join(options)


def get_dash_pattern(line_dash_array):
    """Returns the dash pattern for a line.

    Args:
        line_dash_array: The dash array for the line.

    Returns:
        str: The dash pattern as a string.
    """

    return " ".join([str(x) for x in line_dash_array])


def _line_limits(canvas):
    if canvas is None:
        return None
    limits = None
    if canvas.limits is not None:
        limits = tuple(canvas.limits)
    elif canvas._all_vertices:
        bbox = bounding_box(canvas._all_vertices)
        limits = (*bbox.southwest, *bbox.northeast)

    page = canvas.active_page
    sketches = page.sketches
    for sketch in sketches:
        if sketch_attrib(sketch, "subtype") == Types.HELPLINES_SKETCH:
            x, y = sketch_attrib(sketch, "pos")[:2]
            candidate = (
                x,
                y,
                x + sketch_attrib(sketch, "width"),
                y + sketch_attrib(sketch, "height"),
            )
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


def draw_line_sketch(sketch, canvas):
    vertices = sketch_attrib(sketch, "vertices")
    start = vertices[0]
    end = vertices[1]
    extent = sketch_attrib(sketch, "extent")
    if not isinstance(extent, Extent) and extent is not None:
        extent = Extent(extent)

    if extent in [Extent.RAY, Extent.INFINITE]:
        limits = _line_limits(canvas)
        start, end = _clip_line_to_rect(start, end, limits, extent)

    style = get_line_style_options(sketch)
    clip_attr, mask_attr = get_clip_mask_attrs(sketch)

    return (
        f'<line x1="{start[0]}" y1="{start[1]}" '
        f'x2="{end[0]}" y2="{end[1]}" style="{style}"{clip_attr}{mask_attr} />'
    )


def get_clip_mask_attrs(sketch):
    clip_attr = ""
    clip = sketch_attrib(sketch, "clip")
    mask = sketch_attrib(sketch, "mask")
    if clip is True and mask is not None:
        clippath_id = f"clippath_{id(sketch)}"
        clip_attr = f' clip-path="url(#{clippath_id})"'

    mask_attr = ""
    if mask is not None and (clip is not True):
        mask_id = sketch_attrib(sketch, "_mask_context_id")
        if mask_id is not None:
            mask_attr = f' mask="url(#{mask_id})"'
    elif has_mask_style(sketch) and (clip is not True):
        mask_id = sketch_attrib(sketch, "_mask_context_id")
        if mask_id is not None:
            mask_attr = f' mask="url(#{mask_id})"'

    return clip_attr, mask_attr


def get_marker_options(sketch):
    """Returns the options for the markers.

    Args:
        sketch: The sketch object.

    Returns:
        list: The marker options as a list.
    """
    attrib_map = {
        # 'marker': 'mark',
        "marker_size": "mark size",
        "marker_angle": "rotate",
        # 'fill_color': 'color',
        "marker_color": "color",
        "marker_fill": "fill",
        "marker_opacity": "opacity",
        "marker_repeat": "mark repeat",
        "marker_phase": "mark phase",
        "marker_tension": "tension",
        "marker_line_width": "line width",
        "marker_line_style": "style",
        # 'line_color': 'line color',
    }
    # if mark_stroke is false make line color same as fill color
    if sketch_attrib(sketch, "draw_markers"):
        res = sg_to_tikz(sketch, marker_style_map.keys(), attrib_map)
    else:
        res = []

    return res


def draw_shape_sketch_with_indices(sketch, index=0):
    """Draws a shape sketch with index numbers at each vertex for SVG.

    Args:
        sketch: The shape sketch object.
        index: The index.

    Returns:
        str: The SVG code for the shape sketch with indices.
    """
    vertices = sketch_attrib(sketch, "vertices")

    shape_type = "polygon" if sketch_attrib(sketch, "closed") else "polyline"

    line_style = get_line_style_options(sketch)
    fill_style = get_fill_style_options(sketch, shape_type)
    style = f"{line_style} {fill_style}".strip()
    style_attr = f'style="{style}"' if style else ""

    verts = " ".join([f"{vertex[0]},{vertex[1]}" for vertex in vertices])
    shape_svg = f'<{shape_type} points="{verts}" {style_attr}/>'

    # Determine offset for label positioning
    if hasattr(sketch, "ind_offset"):
        offset = sketch_attrib(sketch, "ind_offset")
    else:
        offset = defaults["ind_offset"]

    # Compute label positions using vert_label_positions
    label_positions = vert_label_positions(sketch, offset)

    font_size = defaults["font_size"]
    elements = [shape_svg]
    for i, (lx, ly) in enumerate(label_positions):
        elements.append(
            f'<g transform="translate({lx} {ly}) scale(1,-1)">'
            f'<text x="0" y="0" text-anchor="middle" dominant-baseline="middle"'
            f' font-size="{font_size}">{i}</text>'
            f"</g>"
        )

    content = "\n".join(elements)
    clip_attr, mask_attr = get_clip_mask_attrs(sketch)
    if clip_attr or mask_attr:
        return f"<g{clip_attr}{mask_attr}>\n{content}\n</g>"
    return content


"""
<polyline
points = "x1 y1, x2 y2, x3, y3"
style = "
stroke: black;
stroke-opacity: 1;
stroke-width: 1;
stroke-linecap: round;
stroke-linejoin: round;
stroke-miterlimit: 10;
fill: none
"/>
"""

"""
<polygon
points = "x1 y1, x2 y2, x3, y3"
style = "
stroke: black;
stroke-opacity: 1;
stroke-width: 1;
stroke-linecap: round;
stroke-linejoin: round;
stroke-miterlimit: 10;
fill: blue;
fill-opacity: 0.5;
fill-rule: evenodd;
"/>
"""

"""
1- Get shape type (line, circle, rectangle, polygon etc.)
2- Get coordinates (for line: x1, y1, x2, y2; for circle: cx, cy, r; for rectangle: x, y, width, height; for polygon: points etc.)
3- Get style options (stroke, fill, opacity, line width, line dash array etc.)

f'<{shape_type}
{coordinates}
style="{style_options}"/>'
"""


def draw_tag_sketch(sketch):
    """Converts a TagSketch to SVG code.

    Args:
        sketch: The TagSketch object.

    Returns:
        str: The SVG code for the TagSketch.
    """
    x, y = sketch_attrib(sketch, "pos")[:2]
    elements = []

    # Calculate text properties
    font_size = sketch_attrib(sketch, "font_size")
    font_family = sketch_attrib(sketch, "font_family")
    font_color = sketch_attrib(sketch, "font_color")
    text = sketch_attrib(sketch, "text")
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    text = text.replace(r"\_", "_")
    escaped_text = html.escape(html.unescape(text), quote=False)

    # Text styling
    font_weight = "bold" if sketch_attrib(sketch, "bold") else "normal"
    font_style = "italic" if sketch_attrib(sketch, "italic") else "normal"
    text_anchor = "middle"  # Default for centered text

    # Handle anchor positioning
    anchor = sketch_attrib(sketch, "anchor")
    if anchor:
        if anchor in [Anchor.WEST, Anchor.SOUTHWEST, Anchor.NORTHWEST]:
            text_anchor = "start"
        elif anchor in [Anchor.EAST, Anchor.SOUTHEAST, Anchor.NORTHEAST]:
            text_anchor = "end"

    # Draw frame if needed
    if sketch_attrib(sketch, "draw_frame"):
        frame_shape = sketch_attrib(sketch, "frame_shape")
        fill_color = (
            sketch_attrib(sketch, "frame_back_color")
            if sketch_attrib(sketch, "fill")
            else "none"
        )
        stroke_color = (
            sketch_attrib(sketch, "line_color")
            if sketch_attrib(sketch, "stroke")
            else "none"
        )
        stroke_width = sketch_attrib(sketch, "line_width")
        inner_sep = sketch_attrib(sketch, "frame_inner_sep")
        minimum_width = sketch_attrib(sketch, "minimum_width")

        # Get accurate text dimensions using PIL
        text_width, text_height = get_text_size(text, font_family, font_size)

        if minimum_width and text_width < minimum_width:
            text_width = minimum_width

        # Add padding
        bbox_width = text_width + 2 * inner_sep
        bbox_height = text_height + 2 * inner_sep
        bbox_x = x - bbox_width / 2
        bbox_y = y - bbox_height / 2

        if isinstance(fill_color, Color):
            fill_color = color_to_svg(fill_color)
        if isinstance(stroke_color, Color):
            stroke_color = color_to_svg(stroke_color)

        # Draw frame based on shape
        if frame_shape == FrameShape.CIRCLE:
            radius = max(bbox_width, bbox_height) / 2
            elements.append(
                f'<circle cx="{x}" cy="{y}" r="{radius}" '
                f'fill="{fill_color}" stroke="{stroke_color}" stroke-width="{stroke_width}" />'
            )
        elif frame_shape == FrameShape.ELLIPSE:
            rx = bbox_width / 2
            ry = bbox_height / 2
            elements.append(
                f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" '
                f'fill="{fill_color}" stroke="{stroke_color}" stroke-width="{stroke_width}" />'
            )
        else:  # RECTANGLE or other shapes default to rectangle
            elements.append(
                f'<rect x="{bbox_x}" y="{bbox_y}" width="{bbox_width}" height="{bbox_height}" '
                f'fill="{fill_color}" stroke="{stroke_color}" stroke-width="{stroke_width}" />'
            )

    # Draw text
    if isinstance(font_color, Color):
        font_color = color_to_svg(font_color)

    # SVG text element
    text_decoration = ""
    if sketch_attrib(sketch, "small_caps"):
        text_decoration = 'font-variant="small-caps" '

    align = sketch_attrib(sketch, "align")
    if align in (Align.LEFT, Align.FLUSH_LEFT):
        text_anchor = "start"
    elif align in (Align.RIGHT, Align.FLUSH_RIGHT):
        text_anchor = "end"
    elif align in (Align.CENTER, Align.FLUSH_CENTER):
        text_anchor = "middle"

    text_width = sketch_attrib(sketch, "text_width")
    text_width_attr = ""
    if text_width:
        text_width_attr = f'textLength="{text_width}" '

    # Wrap text in a transform group to flip y-axis (prevents upside-down text)
    elements.append(f'<g transform="translate({x} {y}) scale(1,-1)">')
    elements.append(
        f'<text x="0" y="0" '
        f'font-family="{font_family}" '
        f'font-size="{font_size}" '
        f'font-weight="{font_weight}" '
        f'font-style="{font_style}" '
        f"{text_decoration}"
        f'fill="{font_color}" '
        f'text-anchor="{text_anchor}" '
        f'dominant-baseline="middle" '
        f"{text_width_attr}"
        f">{escaped_text}</text>"
    )
    elements.append("</g>")

    content = "\n".join(elements)
    clip_attr, mask_attr = get_clip_mask_attrs(sketch)
    if clip_attr or mask_attr:
        return f"<g{clip_attr}{mask_attr}>\n{content}\n</g>"
    return content


def draw_helplines_sketch(sketch):
    x, y = sketch_attrib(sketch, "pos")[:2]
    width = sketch_attrib(sketch, "width")
    height = sketch_attrib(sketch, "height")
    spacing = sketch_attrib(sketch, "spacing")
    cs_size = sketch_attrib(sketch, "cs_size")
    kwargs = dict(sketch_attrib(sketch, "kwargs"))

    if "line_width" not in kwargs:
        kwargs["line_width"] = defaults["grid_line_width"]
    if "line_color" not in kwargs:
        kwargs["line_color"] = defaults["grid_line_color"]
    if "line_dash_array" not in kwargs:
        kwargs["line_dash_array"] = defaults["grid_line_dash_array"]
    if "line_alpha" not in kwargs:
        if "alpha" in kwargs:
            kwargs["line_alpha"] = kwargs["alpha"]
        else:
            kwargs["line_alpha"] = defaults["line_alpha"]
    if "line_cap" not in kwargs:
        kwargs["line_cap"] = defaults["line_cap"]
    if "line_join" not in kwargs:
        kwargs["line_join"] = defaults["line_join"]
    if "line_miter_limit" not in kwargs:
        kwargs["line_miter_limit"] = defaults["line_miter_limit"]

    # Match draw.grid defaults
    grid_line_width = kwargs["line_width"]
    grid_line_color = kwargs["line_color"]
    grid_line_dash_array = kwargs["line_dash_array"]
    line_alpha = kwargs["line_alpha"]
    line_cap = kwargs["line_cap"]
    line_join = kwargs["line_join"]
    line_miter_limit = kwargs["line_miter_limit"]

    def _line_style(line_color, line_width, line_dash_array=None, alpha=None):
        style_obj = SimpleNamespace(
            stroke=True,
            line_color=line_color,
            line_width=line_width,
            line_dash_array=line_dash_array,
            line_alpha=line_alpha,
            line_cap=line_cap,
            line_join=line_join,
            miter_limit=line_miter_limit,
        )
        return get_line_style_options(style_obj)

    elements = []

    # Grid lines (horizontal + vertical)
    n_h = int(height / spacing)
    n_v = int(width / spacing)
    grid_style = _line_style(
        grid_line_color, grid_line_width, grid_line_dash_array
    )

    for i in range(n_h + 1):
        yi = y + i * spacing
        elements.append(
            f'<line x1="{x}" y1="{yi}" x2="{x + width}" y2="{yi}" style="{grid_style}" />'
        )

    for i in range(n_v + 1):
        xi = x + i * spacing
        elements.append(
            f'<line x1="{xi}" y1="{y}" x2="{xi}" y2="{y + height}" style="{grid_style}" />'
        )

    # Coordinate system axes + origin marker
    if cs_size and cs_size > 0:
        if "colors" not in kwargs:
            kwargs["colors"] = (defaults["CS_x_color"], defaults["CS_y_color"])
        x_color, y_color = kwargs["colors"]

        if "line_width" not in kwargs:
            kwargs["line_width"] = defaults["CS_line_width"]
        cs_line_width = kwargs["line_width"]

        x_axis_style = _line_style(
            x_color, cs_line_width, kwargs["line_dash_array"]
        )
        y_axis_style = _line_style(
            y_color, cs_line_width, kwargs["line_dash_array"]
        )

        elements.append(
            f'<line x1="0" y1="0" x2="{cs_size}" y2="0" style="{x_axis_style}" />'
        )
        elements.append(
            f'<line x1="0" y1="0" x2="0" y2="{cs_size}" style="{y_axis_style}" />'
        )

        origin_color = kwargs["line_color"]
        origin_color_svg = (
            color_to_svg(origin_color)
            if isinstance(origin_color, Color)
            else origin_color
        )
        elements.append(
            f'<circle cx="0" cy="0" r="{defaults["CS_origin_size"]}" '
            f'fill="{origin_color_svg}" stroke="{origin_color_svg}" />'
        )

    content = "\n".join(elements)
    clip_attr, mask_attr = get_clip_mask_attrs(sketch)
    if clip_attr or mask_attr:
        return f"<g{clip_attr}{mask_attr}>\n{content}\n</g>"
    return content


def draw_image_sketch(sketch):
    """Converts an ImageSketch to SVG code.

    Args:
        sketch: The ImageSketch object.

    Returns:
        str: The SVG code for the ImageSketch.
    """
    x, y = sketch_attrib(sketch, "pos")[:2]
    size = sketch_attrib(sketch, "size")
    width, height = size if size else (100, 100)
    angle_value = sketch_attrib(sketch, "angle")
    angle = degrees(angle_value) if angle_value != 0 else 0

    # Get scale - can be tuple or single value
    scale = sketch_attrib(sketch, "scale")
    if isinstance(scale, (tuple, list)):
        sx, sy = scale
    else:
        sx = sy = scale

    # Get anchor offset
    anchor = sketch_attrib(sketch, "anchor")

    # Calculate anchor offset
    # In SVG, image x,y is at top-left, so we need to adjust based on anchor
    anchor_offsets = {
        Anchor.CENTER: (-width / 2, -height / 2),
        Anchor.NORTH: (-width / 2, 0),
        Anchor.SOUTH: (-width / 2, -height),
        Anchor.EAST: (-width, -height / 2),
        Anchor.WEST: (0, -height / 2),
        Anchor.NORTHEAST: (-width, 0),
        Anchor.NORTHWEST: (0, 0),
        Anchor.SOUTHEAST: (-width, -height),
        Anchor.SOUTHWEST: (0, -height),
    }
    dx, dy = anchor_offsets[anchor]

    # Build transform string
    transforms = []
    transforms.append(f"translate({x}, {y})")
    if angle != 0:
        transforms.append(f"rotate({angle})")
    if sx != 1 or sy != 1:
        transforms.append(f"scale({sx}, {sy})")
    if dx != 0 or dy != 0:
        transforms.append(f"translate({dx}, {dy})")

    # Images are rendered inside a globally flipped SVG group (scale(1,-1)).
    # Apply a local counter-flip so raster images remain upright.
    transforms.append(f"translate(0, {height})")
    transforms.append("scale(1, -1)")

    transform_attr = (
        f' transform="{" ".join(transforms)}"' if transforms else ""
    )

    # Use href (modern SVG) or xlink:href (legacy)
    file_path = sketch_attrib(sketch, "file_path")

    clip_attr, mask_attr = get_clip_mask_attrs(sketch)
    return f'<image x="0" y="0" width="{width}" height="{height}" href="{file_path}"{transform_attr}{clip_attr}{mask_attr} />'


def draw_latex_sketch(sketch):
    """Renders a LaTeX math formula to inline SVG using matplotlib mathtext.

    No TeX compiler required — uses matplotlib's built-in mathtext engine.
    The formula SVG fragment is counter-flipped to appear correctly inside
    simetri's global scale(1,-1) coordinate system.

    Args:
        sketch: A LatexSketch with formula, pos, font_size, and anchor.

    Returns:
        str: SVG code for the formula positioned at the canvas anchor point.
    """
    # Friendly name → matplotlib mathtext.fontset mapping
    _FONTSET_MAP = {
        "computer modern": "cm",
        "cm": "cm",
        "stix": "stix",
        "stix sans": "stixsans",
        "stixsans": "stixsans",
        "dejavu sans": "dejavusans",
        "dejavusans": "dejavusans",
        "dejavu": "dejavusans",
        "dejavu serif": "dejavuserif",
        "dejavuserif": "dejavuserif",
    }

    formula = sketch_attrib(sketch, "formula")
    x, y = sketch_attrib(sketch, "pos")[:2]
    font_size = sketch_attrib(sketch, "font_size")
    if font_size is None:
        font_size = defaults["font_size"]
    font_family = sketch_attrib(sketch, "font_family")
    font_color = sketch_attrib(sketch, "font_color")
    if font_color is None:
        font_color = defaults["font_color"]
    font_color = check_color(font_color)
    bold = sketch_attrib(sketch, "bold")
    if bold is None:
        bold = defaults["bold"]
    anchor = sketch_attrib(sketch, "anchor")
    if anchor is None:
        anchor = defaults["anchor"]

    # Optionally auto-wrap the entire formula in \boldsymbol{} for convenience.
    # \boldsymbol preserves the italic math style (bold italic), unlike \mathbf
    # which switches to bold upright — matching standard LaTeX \boldsymbol behaviour.
    if bold:
        formula = rf"\boldsymbol{{{formula}}}"

    # Silently map unsupported LaTeX text-mode commands to their math-mode
    # equivalents that matplotlib mathtext does support:
    _TEXT_MODE_MAP = [
        (r"\texttt", r"\mathtt"),  # monospace / typewriter
        (r"\textrm", r"\mathrm"),  # roman (serif)
        (r"\textbf", r"\mathbf"),  # bold
        (r"\textit", r"\mathit"),  # italic
        (r"\textsf", r"\mathsf"),  # sans-serif
    ]
    for src, dst in _TEXT_MODE_MAP:
        formula = formula.replace(src, dst)

    # Auto-select STIX when \mathbf or \boldsymbol appears anywhere in the
    # formula and no font_family has been explicitly set.
    if not font_family and (r"\mathbf" in formula or r"\boldsymbol" in formula):
        font_family = "stix"

    # Resolve the matplotlib fontset name (default: leave rcParams unchanged)
    fontset = _FONTSET_MAP.get((font_family or "").strip().lower())
    rc_overrides = {"mathtext.fontset": fontset} if fontset else {}

    color_arg = color_to_matplotlib(font_color)

    # Render with matplotlib mathtext (no LaTeX compiler required)
    with matplotlib.rc_context(rc_overrides):
        fig = plt.figure(figsize=(0.01, 0.01), dpi=72)
        fig.text(
            0,
            0,
            f"${formula}$",
            fontsize=font_size,
            usetex=False,
            color=color_arg,
        )
        buf = io.StringIO()
        fig.savefig(
            buf,
            format="svg",
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.05,
        )
        plt.close(fig)

    svg_str = buf.getvalue()

    # Use cached dimensions from draw_latex (avoids re-parsing) or fall back to regex
    cached_size = sketch_attrib(sketch, "formula_size")
    if cached_size:
        W, H = cached_size
        vb_match = re.search(r'<svg[^>]*\bviewBox="([^"]+)"', svg_str)
        vb = vb_match.group(1) if vb_match else f"0 0 {W} {H}"
    else:
        w_match = re.search(r'<svg[^>]*\bwidth="([\d.]+)pt"', svg_str)
        h_match = re.search(r'<svg[^>]*\bheight="([\d.]+)pt"', svg_str)
        vb_match = re.search(r'<svg[^>]*\bviewBox="([^"]+)"', svg_str)
        W = float(w_match.group(1)) if w_match else 100.0
        H = float(h_match.group(1)) if h_match else 20.0
        vb = vb_match.group(1) if vb_match else f"0 0 {W} {H}"

    # Extract inner SVG content (strip the outer <svg ...>...</svg> wrapper)
    inner_match = re.search(r"<svg[^>]*>(.*?)</svg>", svg_str, re.DOTALL)
    inner_svg = inner_match.group(1).strip() if inner_match else ""

    # Strip matplotlib boilerplate that is redundant when embedded inline:
    #   <metadata>...</metadata>  — RDF/Dublin Core copyright blocks
    #   <!-- ... -->              — XML comments (version stamps etc.)
    inner_svg = re.sub(
        r"<metadata>.*?</metadata>", "", inner_svg, flags=re.DOTALL
    )
    inner_svg = re.sub(r"<!--.*?-->", "", inner_svg, flags=re.DOTALL)
    # Collapse runs of blank lines left behind by the removals
    inner_svg = re.sub(r"\n{3,}", "\n\n", inner_svg).strip()

    # Anchor offset: distance from the formula's SW corner to the given anchor point,
    # measured in canvas/formula coordinate space (W wide, H tall).
    # The formula's SW corner is at its left edge and visual bottom edge.
    anchor_offsets = {
        Anchor.SOUTHWEST: (0, 0),
        Anchor.SOUTH: (W / 2, 0),
        Anchor.SOUTHEAST: (W, 0),
        Anchor.WEST: (0, H / 2),
        Anchor.CENTER: (W / 2, H / 2),
        Anchor.EAST: (W, H / 2),
        Anchor.NORTHWEST: (0, H),
        Anchor.NORTH: (W / 2, H),
        Anchor.NORTHEAST: (W, H),
    }
    ax, ay = anchor_offsets.get(anchor, (0, 0))

    # The main SVG group has transform="translate(0,dy) scale(1,-1)".
    # Inside this group, a sub-group with "translate(x,y) scale(1,-1)" restores normal
    # (y-up) canvas orientation.  In sub-group space, the formula occupies:
    #   left/right x: -ax .. W-ax
    #   bottom/top y: ay-H .. ay     (sub-group y-down → positive y = visually down)
    # The nested <svg> is placed at (sub_x, sub_y) = (-ax, ay-H).
    sub_x = -ax
    sub_y = ay - H

    clip_attr, mask_attr = get_clip_mask_attrs(sketch)

    return (
        f'<g transform="translate({x},{y}) scale(1,-1)"'
        f"{clip_attr}{mask_attr}>\n"
        f'  <svg x="{sub_x:.4f}" y="{sub_y:.4f}" width="{W:.4f}" height="{H:.4f}"'
        f' viewBox="{vb}" xmlns="http://www.w3.org/2000/svg"'
        f' xmlns:xlink="http://www.w3.org/1999/xlink">\n'
        f"{inner_svg}\n"
        f"  </svg>\n"
        f"</g>"
    )


def get_marker_path(marker_type, size):
    """Get the SVG path data for a specific marker type.

    Args:
        marker_type: The MarkerType enum value
        size: Size scaling factor

    Returns:
        str: SVG path data and attributes for the marker
    """
    # Normalize size
    s = size

    marker_paths = {
        MarkerType.CIRCLE: ("circle", f'<circle cx="0" cy="0" r="{s}"/>'),
        MarkerType.FCIRCLE: ("circle", f'<circle cx="0" cy="0" r="{s}"/>'),
        MarkerType.SQUARE: (
            "path",
            f'<rect x="{-s}" y="{-s}" width="{2 * s}" height="{2 * s}"/>',
        ),
        MarkerType.SQUARE_F: (
            "path",
            f'<rect x="{-s}" y="{-s}" width="{2 * s}" height="{2 * s}"/>',
        ),
        MarkerType.DIAMOND: (
            "path",
            f'<path d="M 0,{s} L {s},0 L 0,{-s} L {-s},0 Z"/>',
        ),
        MarkerType.DIAMOND_F: (
            "path",
            f'<path d="M 0,{s} L {s},0 L 0,{-s} L {-s},0 Z"/>',
        ),
        MarkerType.TRIANGLE: (
            "path",
            f'<path d="M 0,{s * 1.2} L {s * 1.04},{-s * 0.6} L {-s * 1.04},{-s * 0.6} Z"/>',
        ),
        MarkerType.TRIANGLE_F: (
            "path",
            f'<path d="M 0,{s * 1.2} L {s * 1.04},{-s * 0.6} L {-s * 1.04},{-s * 0.6} Z"/>',
        ),
        MarkerType.PLUS: (
            "path",
            f'<path d="M 0,{s} L 0,{-s} M {s},0 L {-s},0"/>',
        ),
        MarkerType.CROSS: (
            "path",
            f'<path d="M {s},{s} L {-s},{-s} M {s},{-s} L {-s},{s}"/>',
        ),
        MarkerType.ASTERISK: (
            "path",
            f'<path d="M 0,{s} L 0,{-s} M {s * 0.866},{s * 0.5} L {-s * 0.866},{-s * 0.5} M {s * 0.866},{-s * 0.5} L {-s * 0.866},{s * 0.5}"/>',
        ),
        MarkerType.STAR: (
            "path",
            f'<path d="M 0,{s} L {s * 0.224},{s * 0.309} L {s * 0.951},{s * 0.309} L {s * 0.363},{-s * 0.118} L {s * 0.588},{-s * 0.809} L 0,{-s * 0.382} L {-s * 0.588},{-s * 0.809} L {-s * 0.363},{-s * 0.118} L {-s * 0.951},{s * 0.309} L {-s * 0.224},{s * 0.309} Z"/>',
        ),
        MarkerType.PENTAGON: (
            "path",
            f'<path d="M 0,{s} L {s * 0.951},{s * 0.309} L {s * 0.588},{-s * 0.809} L {-s * 0.588},{-s * 0.809} L {-s * 0.951},{s * 0.309} Z"/>',
        ),
        MarkerType.PENTAGON_F: (
            "path",
            f'<path d="M 0,{s} L {s * 0.951},{s * 0.309} L {s * 0.588},{-s * 0.809} L {-s * 0.588},{-s * 0.809} L {-s * 0.951},{s * 0.309} Z"/>',
        ),
        MarkerType.HEXAGON: (
            "path",
            f'<path d="M {s},0 L {s * 0.5},{s * 0.866} L {-s * 0.5},{s * 0.866} L {-s},0 L {-s * 0.5},{-s * 0.866} L {s * 0.5},{-s * 0.866} Z"/>',
        ),
        MarkerType.HEXAGON_F: (
            "path",
            f'<path d="M {s},0 L {s * 0.5},{s * 0.866} L {-s * 0.5},{s * 0.866} L {-s},0 L {-s * 0.5},{-s * 0.866} L {s * 0.5},{-s * 0.866} Z"/>',
        ),
        MarkerType.BAR: ("path", f'<path d="M 0,{s} L 0,{-s}"/>'),
        MarkerType.MINUS: ("path", f'<path d="M {s},0 L {-s},0"/>'),
        MarkerType.OPLUS: (
            "g",
            f'<circle cx="0" cy="0" r="{s}"/><path d="M 0,{s * 0.7} L 0,{-s * 0.7} M {s * 0.7},0 L {-s * 0.7},0"/>',
        ),
        MarkerType.OPLUS_F: (
            "g",
            f'<circle cx="0" cy="0" r="{s}"/><path d="M 0,{s * 0.7} L 0,{-s * 0.7} M {s * 0.7},0 L {-s * 0.7},0"/>',
        ),
        MarkerType.O_TIMES: (
            "g",
            f'<circle cx="0" cy="0" r="{s}"/><path d="M {s * 0.7},{s * 0.7} L {-s * 0.7},{-s * 0.7} M {s * 0.7},{-s * 0.7} L {-s * 0.7},{s * 0.7}"/>',
        ),
        MarkerType.O_TIMES_F: (
            "g",
            f'<circle cx="0" cy="0" r="{s}"/><path d="M {s * 0.7},{s * 0.7} L {-s * 0.7},{-s * 0.7} M {s * 0.7},{-s * 0.7} L {-s * 0.7},{s * 0.7}"/>',
        ),
        MarkerType.HALF_CIRCLE: (
            "path",
            f'<path d="M 0,{s} A {s} {s} 0 0 1 0,{-s} L 0,{s} Z"/>',
        ),
        MarkerType.HALF_CIRCLE_F: (
            "path",
            f'<path d="M 0,{s} A {s} {s} 0 0 1 0,{-s} L 0,{s} Z"/>',
        ),
        MarkerType.HALF_SQUARE: (
            "path",
            f'<path d="M 0,{s} L {s},{s} L {s},{-s} L 0,{-s} Z"/>',
        ),
        MarkerType.HALF_SQUARE_F: (
            "path",
            f'<path d="M 0,{s} L {s},{s} L {s},{-s} L 0,{-s} Z"/>',
        ),
        MarkerType.HALF_DIAMOND: (
            "path",
            f'<path d="M 0,{s} L {s},0 L 0,{-s} Z"/>',
        ),
        MarkerType.HALF_DIAMOND_F: (
            "path",
            f'<path d="M 0,{s} L {s},0 L 0,{-s} Z"/>',
        ),
    }

    return marker_paths[marker_type]


def generate_marker_def(
    marker_id, marker_type, sketch, canvas=None, styles_dict=None
):
    """Generate SVG marker definition.

    Args:
        marker_id: Unique ID for this marker
        marker_type: The MarkerType enum value
        sketch: The sketch object with marker styling
        canvas: The canvas object (needed for custom shapes)
        styles_dict: Styles dictionary (needed for custom shapes)

    Returns:
        str: SVG <marker> element
    """
    marker_size = sketch_attrib(sketch, "marker_size")
    marker_color = sketch_attrib(sketch, "marker_color")
    marker_alpha = sketch_attrib(sketch, "marker_alpha")

    # Handle custom shape markers
    if marker_type == MarkerType.SHAPE:
        marker_shape = sketch_attrib(sketch, "marker_shape")
        from ..canvas.draw import create_sketch  # noqa: PLC0415 — circular import
        marker_sketch = create_sketch(marker_shape, canvas)

        if marker_sketch is None:
            raise ValueError("marker_shape sketch could not be created.")

        if isinstance(marker_color, Color):
            marker_color_svg = color_to_svg(marker_color)
        else:
            marker_color_svg = marker_color

        shape_type = get_shape_type(marker_sketch)
        if shape_type == "polygon" or shape_type == "polyline":
            marker_shape_fill = sketch_attrib(marker_sketch, "fill")
            marker_shape_fill_enabled = (
                defaults["fill"]
                if marker_shape_fill is None
                else bool(marker_shape_fill)
            )

            if marker_shape_fill_enabled:
                fill_color = marker_sketch.fill_color
                if isinstance(fill_color, Color):
                    fill_color = color_to_svg(fill_color)
                fill_attr = f'fill="{fill_color}" fill-opacity="{marker_alpha}"'
                stroke_attr = 'stroke="none"'
            else:
                fill_attr = 'fill="none"'
                stroke_attr = f'stroke="{marker_color_svg}" stroke-width="1" stroke-opacity="{marker_alpha}"'

            shape_svg = (
                f'<{shape_type} points="{points}" {fill_attr} {stroke_attr}/>'
            )
        else:
            shape_svg = svg_shape(marker_sketch, styles_dict)

        bbox = bounding_box(marker_sketch.vertices)
        vb_width = bbox.width
        vb_height = bbox.height
        vb_cx = (bbox.southwest[0] + bbox.northeast[0]) / 2
        vb_cy = (bbox.southwest[1] + bbox.northeast[1]) / 2
        scale_factor = marker_size / max(vb_width, vb_height, 1)
        vb_width *= scale_factor * 1.5
        vb_height *= scale_factor * 1.5

        return f'''  <marker id="{marker_id}" markerWidth="{vb_width}" markerHeight="{vb_height}"
      refX="{vb_cx}" refY="{vb_cy}" viewBox="{vb_cx - vb_width / 2} {vb_cy - vb_height / 2} {vb_width} {vb_height}" orient="auto">
    {shape_svg}
  </marker>'''

    # Handle predefined marker types
    # Get marker properties from sketch
    marker_fill = sketch_attrib(sketch, "marker_fill")
    marker_line_width = sketch_attrib(sketch, "marker_line_width")
    marker_fill_enabled = (
        defaults["fill"] if marker_fill is None else bool(marker_fill)
    )

    # Convert color
    if isinstance(marker_color, Color):
        marker_color_svg = color_to_svg(marker_color)
    else:
        marker_color_svg = marker_color

    # Determine fill and stroke based on marker type
    is_filled = marker_type.value.endswith("*") or marker_type in [
        MarkerType.FCIRCLE,
        MarkerType.SQUARE_F,
        MarkerType.DIAMOND_F,
        MarkerType.TRIANGLE_F,
        MarkerType.PENTAGON_F,
        MarkerType.HEXAGON_F,
        MarkerType.OPLUS_F,
        MarkerType.O_TIMES_F,
        MarkerType.HALF_CIRCLE_F,
        MarkerType.HALF_SQUARE_F,
        MarkerType.HALF_DIAMOND_F,
    ]

    if is_filled and marker_fill_enabled:
        fill_attr = f'fill="{marker_color_svg}" fill-opacity="{marker_alpha}"'
        stroke_attr = 'stroke="none"'
    else:
        fill_attr = 'fill="none"'
        stroke_attr = f'stroke="{marker_color_svg}" stroke-width="{marker_line_width}" stroke-opacity="{marker_alpha}"'

    # Get marker path
    elem_type, path_data = get_marker_path(marker_type, marker_size)

    return f'''  <marker id="{marker_id}" markerWidth="{marker_size * 2}" markerHeight="{marker_size * 2}"
      refX="0" refY="0" viewBox="{-marker_size} {-marker_size} {marker_size * 2} {marker_size * 2}" markerUnits="userSpaceOnUse" orient="auto">
    <g {fill_attr} {stroke_attr}>
      {path_data}
    </g>
  </marker>'''


def draw_shape_sketch_with_markers(sketch):
    """Draws a shape sketch with markers for SVG.

    Args:
        sketch: The shape sketch object.

    Returns:
        str: The SVG code for the shape sketch with markers.
    """
    # Get vertices
    vertices = sketch_attrib(sketch, "vertices")
    closed = sketch_attrib(sketch, "closed")
    if closed and not close_points2(vertices[0], vertices[-1]):
        vertices = list(vertices) + [vertices[0]]

    # Get marker type
    marker_type = sketch_attrib(sketch, "marker_type")
    if not isinstance(marker_type, MarkerType):
        # Try to convert by value first, then by name
        try:
            marker_type = MarkerType(marker_type)
        except ValueError:
            # If that fails, try converting by name (for backward compatibility)
            if isinstance(marker_type, str):
                marker_type = MarkerType[
                    marker_type.upper().replace(" ", "_").replace("-", "_")
                ]
            else:
                raise

    # Create unique marker ID for this sketch
    marker_id = f"marker_{id(sketch)}"

    # Check if markers only (no line)
    markers_only = sketch_attrib(sketch, "markers_only")

    # Build path for the line
    if not markers_only:
        # Draw the line/shape
        path_data = f"M {vertices[0][0]},{vertices[0][1]}"
        for v in vertices[1:]:
            path_data += f" L {v[0]},{v[1]}"

        # Get line styling
        line_color = get_color(sketch, "line_color")
        if isinstance(line_color, Color):
            line_color = color_to_svg(line_color)

        line_width = sketch_attrib(sketch, "line_width")
        line_alpha = get_alpha(sketch, "line_alpha")

        # Get fill styling if closed
        fill_str = '"none"'
        if sketch_attrib(sketch, "fill") and closed:
            fill_color = get_color(sketch, "fill_color")
            if isinstance(fill_color, Color):
                fill_color = color_to_svg(fill_color)
            fill_alpha = get_alpha(sketch, "fill_alpha")
            fill_str = f'"{fill_color}" fill-opacity="{fill_alpha}"'

        fill_rule_attr = ""
        if "even_odd" in sketch_attrib(sketch, "__dict__") and sketch_attrib(
            sketch, "even_odd"
        ):
            fill_rule_attr = ' fill-rule="evenodd"'

        # Build path element with marker reference
        path_element = f'<path d="{path_data}" stroke="{line_color}" stroke-width="{line_width}" '
        path_element += (
            f'stroke-opacity="{line_alpha}" fill={fill_str}{fill_rule_attr} '
        )
        path_element += f'marker-start="url(#{marker_id})" marker-mid="url(#{marker_id})" marker-end="url(#{marker_id})"/>'

        clip_attr, mask_attr = get_clip_mask_attrs(sketch)
        if clip_attr or mask_attr:
            return f"<g{clip_attr}{mask_attr}>\n{path_element}\n</g>"
        return path_element
    else:
        # Markers only - draw one degenerate path per vertex and attach marker-start.
        # This uses the same <defs>/<marker> pipeline as non-markers-only rendering,
        # so custom MarkerType.SHAPE markers work consistently.
        elements = []
        for v in vertices:
            x, y = v[0], v[1]
            elements.append(
                f'<path d="M {x},{y} L {x},{y}" stroke="none" fill="none" '
                f'marker-start="url(#{marker_id})"/>'
            )

        content = "\n".join(elements)
        clip_attr, mask_attr = get_clip_mask_attrs(sketch)
        if clip_attr or mask_attr:
            return f"<g{clip_attr}{mask_attr}>\n{content}\n</g>"
        return content


def get_svg_shapes(canvas: "Canvas", styles_dict: dict) -> str:
    """Convert the sketches in the Canvas to SVG code.

    Args:
        canvas ("Canvas"): The canvas object.

    Returns:
        str: The SVG code.
    """

    def get_sketch_code(sketch, canvas, ind):
        """Get the SVG code for a sketch.

        Args:
            sketch: The sketch object.
            canvas: The canvas object.
            ind: The index.

        Returns:
            tuple: The SVG code and the updated index.
        """

        subtype = sketch_attrib(sketch, "subtype")
        draw_markers = sketch_attrib(sketch, "draw_markers")
        indices = sketch_attrib(sketch, "indices")

        if subtype == Types.TAG_SKETCH:
            code = draw_tag_sketch(sketch)
        elif subtype == Types.CLIPPED_SKETCH:
            clippath_id = f"clippath_{id(sketch)}"
            child_codes = []
            for sketch_list in sketch.sketches:
                for clipped_sketch in sketch_list:
                    child_codes.append(get_sketch_code(clipped_sketch, canvas, ind))
            content = "\n".join(child_codes)
            code = f'<g clip-path="url(#{clippath_id})">\n{content}\n</g>'
        elif subtype == Types.MASKED_SKETCH:
            mask_id = f"mask_{id(sketch)}"
            child_codes = []
            for sketch_list in sketch.sketches:
                for masked_sketch in sketch_list:
                    child_codes.append(get_sketch_code(masked_sketch, canvas, ind))
            content = "\n".join(child_codes)
            code = f'<g mask="url(#{mask_id})">\n{content}\n</g>'
        elif subtype == Types.TEX_SKETCH:
            # TexSketch is for TikZ/LaTeX output, skip in SVG
            code = ""
        elif subtype == Types.MASK_SKETCH:
            code = ""
        elif subtype == Types.LATEX_SKETCH:
            code = draw_latex_sketch(sketch)
        elif subtype == Types.IMAGE_SKETCH:
            code = draw_image_sketch(sketch)
        elif subtype == Types.HELPLINES_SKETCH:
            code = draw_helplines_sketch(sketch)
        elif subtype == Types.LINE_SKETCH:
            code = draw_line_sketch(sketch, canvas)
        elif (
            draw_markers
            and sketch_attrib(sketch, "marker_type") == MarkerType.INDICES
        ) or indices:
            code = draw_shape_sketch_with_indices(sketch)
        elif draw_markers:
            # Use marker rendering for shapes with markers enabled
            code = draw_shape_sketch_with_markers(sketch)
        else:
            code = svg_shape(sketch, styles_dict)

        sketch_dict = sketch_attrib(sketch, "__dict__")
        sketch_filter = sketch_attrib(sketch, "filter")
        if "filter" in sketch_dict and sketch_filter is not None:
            filter_id = sketch_filter.id
            code = f'<g filter="url(#{filter_id})">\n{code}\n</g>'
        return code

    pages = canvas.pages

    if pages:
        for i, page in enumerate(pages):
            canvas.active_page = page
            sketches = page.sketches
            if i == 0:
                if page.back_color:
                    code = [color_to_svg(page.back_color)]
                else:
                    code = []
            else:
                code.append(defaults["end_SVG"])
                code.append("\\newpage")
                code.append(defaults["begin_SVG"])
            sketches_to_populate = list(sketches)
            while sketches_to_populate:
                sketch = sketches_to_populate.pop()
                subtype = sketch_attrib(sketch, "subtype")
                if subtype in [Types.CLIPPED_SKETCH, Types.MASKED_SKETCH]:
                    for sketch_list in sketch.sketches:
                        sketches_to_populate.extend(sketch_list)
                elif subtype == Types.HELPLINES_SKETCH:
                    sketch.populate(canvas)
                elif subtype == Types.LINE_SKETCH:
                    sketch.populate(canvas)
            ind = 0
            scope_opens = {}
            scope_closes = {}
            for scope_group in page.scope_groups:
                if scope_group.sketch_list:
                    first_sketch_id = id(scope_group.sketch_list[0])
                    last_sketch_id = id(scope_group.sketch_list[-1])
                    if first_sketch_id not in scope_opens:
                        scope_opens[first_sketch_id] = []
                    scope_opens[first_sketch_id].append(scope_group)
                    if last_sketch_id not in scope_closes:
                        scope_closes[last_sketch_id] = []
                    scope_closes[last_sketch_id].append(scope_group)
            for sketch in sketches:
                sketch_id = id(sketch)
                if sketch_id in scope_opens:
                    for scope_group in scope_opens[sketch_id]:
                        if scope_group.subtype == Types.CLIP_GROUP:
                            code.append(f'<g clip-path="url(#clippath_{scope_group.id})">')
                        elif scope_group.subtype == Types.MASK_GROUP:
                            code.append(f'<g mask="url(#{scope_group._mask_context_id})">')
                sketch_code = get_sketch_code(sketch, canvas, ind)
                code.append(sketch_code)
                if sketch_id in scope_closes:
                    for scope_group in scope_closes[sketch_id]:
                        code.append('</g>')

        code = "\n".join(code)
    else:
        raise ValueError("No pages found in the canvas.")
    return code


d_shape_types = {
    Types.LINE_SKETCH: "line",
    Types.BBOX_SKETCH: "shape",
    Types.CIRCLE_SKETCH: "circle",
    Types.ELLIPSE_SKETCH: "ellipse",
    Types.RECTANGLE_SKETCH: "rect",
    Types.HANDLE: "shape",
    Types.SHAPE_SKETCH: "shape",
    Types.TAG_SKETCH: "tag",
}


def get_shape_type(sketch):
    shape_type = d_shape_types[sketch_attrib(sketch, "subtype")]
    if shape_type == "shape":
        if sketch_attrib(sketch, "closed"):
            shape_type = "polygon"
        else:
            shape_type = "polyline"

    return shape_type


def get_coordinates(sketch, shape_type):
    if shape_type in ["polygon", "polyline"]:
        vertices = sketch_attrib(sketch, "vertices")
        verts = ", ".join([f"{x} {y}" for x, y in vertices])

        res = f'points = "{verts}"'
    elif shape_type == "rect":
        vertices = sketch_attrib(sketch, "vertices")
        if vertices:
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            x = min(xs)
            y = min(ys)
            width = max(xs) - x
            height = max(ys) - y
        else:
            x, y = sketch_attrib(sketch, "pos")[:2]
            width = sketch_attrib(sketch, "width")
            height = sketch_attrib(sketch, "height")

        res = f'x = "{x}" y = "{y}" width = "{width}" height = "{height}"'
    elif shape_type == "circle":
        cx, cy = sketch_attrib(sketch, "center")
        r = sketch_attrib(sketch, "radius")

        res = f'cx = "{cx}" cy = "{cy}" r = "{r}"'
    elif shape_type == "ellipse":
        cx, cy = sketch_attrib(sketch, "center")
        rx = sketch_attrib(sketch, "x_radius")
        ry = sketch_attrib(sketch, "y_radius")

        res = f'cx = "{cx}" cy = "{cy}" rx = "{rx}" ry = "{ry}"'

    return res


def get_style(sketch, shape_type):
    line_style = get_line_style_options(sketch)
    res = [line_style]
    if shape_type in ["circle", "ellipse", "polygon", "polyline", "rect"]:
        fill_style = get_fill_style_options(sketch, shape_type)
        res.append(fill_style)

    return "; ".join(res)


def get_styles_dict(canvas):
    """Get all line and fill styles from the sketches and create a dictionary.
      Name them line_style_1, line_style_2, ...
      fill_style_1, fill_style_2, ...
      Then create a style selector class section:
      <style type="text/css"><![CDATA[
      .line_style_1 {line_width: 2; stroke-dasharray: 2, 4;}
      .fill_style_1 { fill: yellow; stroke: red; }
      .fill_style_2 { fill-opacity: 0.25; fill-rule: evenodd; }
    ]]></style>"""

    def parse_style_string(style_string):
        """Parse a style string into a dictionary."""
        style_dict = {}
        if not style_string:
            return style_dict

        # Split by semicolon and process each property
        parts = style_string.split(";")
        for part in parts:
            part = part.strip()
            if ":" in part:
                key, value = part.split(":", 1)
                style_dict[key.strip()] = value.strip()

        return style_dict

    line_styles = {}
    fill_styles = {}
    line_counter = 1
    fill_counter = 1

    def collect_sketch_styles(sketch):
        nonlocal line_counter, fill_counter
        subtype = sketch_attrib(sketch, "subtype")

        if subtype in [Types.CLIPPED_SKETCH, Types.MASKED_SKETCH]:
            for sketch_list in sketch.sketches:
                for child_sketch in sketch_list:
                    collect_sketch_styles(child_sketch)
            return

        # Skip non-shape sketches (like TexSketch)
        if subtype not in d_shape_types:
            return

        # Get line style
        line_style_str = get_line_style_options(sketch)
        if line_style_str:
            line_style_dict = parse_style_string(line_style_str)
            # Check if this style already exists (compare as frozenset of items)
            style_exists = any(
                set(existing.items()) == set(line_style_dict.items())
                for existing in line_styles.values()
            )
            if not style_exists:
                line_styles[f"line_style_{line_counter}"] = line_style_dict
                line_counter += 1

        # Get fill style for sketches with a mapped SVG shape type
        shape_type = get_shape_type(sketch)
        if shape_type in [
            "circle",
            "ellipse",
            "polygon",
            "polyline",
            "rect",
        ]:
            fill_style_str = get_fill_style_options(sketch, shape_type)
            if fill_style_str:
                fill_style_dict = parse_style_string(fill_style_str)
                # Check if this style already exists
                style_exists = any(
                    set(existing.items()) == set(fill_style_dict.items())
                    for existing in fill_styles.values()
                )
                if not style_exists:
                    fill_styles[f"fill_style_{fill_counter}"] = fill_style_dict
                    fill_counter += 1

    pages = canvas.pages
    if pages:
        for page in pages:
            for sketch in page.sketches:
                collect_sketch_styles(sketch)

    # Combine all styles
    all_styles = {**line_styles, **fill_styles}
    return all_styles


def get_style_class(sketch, shape_type, styles_dict, skip_fill=False):
    """Find the style class names that match the sketch's styles.

    Args:
        sketch: The sketch object.
        shape_type: The SVG shape type.
        styles_dict: Dictionary of style class names to style dictionaries.
        skip_fill: If True, skip adding fill style class (used when gradient/pattern is applied).

    Returns:
        str: Space-separated class names.
    """

    def parse_style_string(style_string):
        """Parse a style string into a dictionary."""
        style_dict = {}
        if not style_string:
            return style_dict

        parts = style_string.split(";")
        for part in parts:
            part = part.strip()
            if ":" in part:
                key, value = part.split(":", 1)
                style_dict[key.strip()] = value.strip()

        return style_dict

    class_names = []

    # Get line style and find matching class
    line_style_str = get_line_style_options(sketch)
    if line_style_str:
        line_style_dict = parse_style_string(line_style_str)
        for class_name, style_values in styles_dict.items():
            if class_name.startswith("line_style_") and set(
                style_values.items()
            ) == set(line_style_dict.items()):
                class_names.append(class_name)
                break

    # Get fill style and find matching class (skip if gradient/pattern is used)
    if not skip_fill and shape_type in [
        "circle",
        "ellipse",
        "polygon",
        "polyline",
        "rect",
    ]:
        fill_style_str = get_fill_style_options(sketch, shape_type)
        if fill_style_str:
            fill_style_dict = parse_style_string(fill_style_str)
            for class_name, style_values in styles_dict.items():
                if class_name.startswith("fill_style_") and set(
                    style_values.items()
                ) == set(fill_style_dict.items()):
                    class_names.append(class_name)
                    break

    return " ".join(class_names)


def svg_shape(sketch, styles_dict):
    shape_type = get_shape_type(sketch)
    style_shape_type = shape_type
    coordinates = get_coordinates(sketch, shape_type)

    draw_fillets = sketch_attrib(sketch, "draw_fillets")
    fillet_radius = sketch_attrib(sketch, "fillet_radius")
    if (
        shape_type in ["polygon", "polyline"]
        and draw_fillets
        and fillet_radius is not None
        and fillet_radius > 0
    ):
        vertices = [vertex[:2] for vertex in sketch_attrib(sketch, "vertices")]
        path_data = round_corners(
            vertices,
            radius=fillet_radius,
            closed=sketch_attrib(sketch, "closed"),
        )
        shape_type = "path"
        coordinates = f'd="{path_data}"'

    # Check for pattern or gradient fill
    fill_attr = ""
    skip_fill_style = False
    if sketch_attrib(sketch, "tile_svg") is not None:
        pattern_id = f"pattern_{id(sketch)}"
        fill_attr = f'fill="url(#{pattern_id})"'
        skip_fill_style = True
    elif has_gradient(sketch):
        gradient_id = sketch_attrib(sketch, "_gradient_context_id")
        if gradient_id is None:
            gradient_id = f"gradient_{id(sketch)}"
        fill_attr = f'fill="url(#{gradient_id})"'
        skip_fill_style = True

    # Check for clip property (clip=True with mask holding the clipping shape)
    clip_attr = ""
    clip = sketch_attrib(sketch, "clip")
    mask = sketch_attrib(sketch, "mask")
    if clip is True and mask is not None:
        clippath_id = f"clippath_{id(sketch)}"
        clip_attr = f' clip-path="url(#{clippath_id})"'

    # Check for opacity mask property (mask shape + clip is not enabled)
    mask_attr = ""
    if mask is not None and (clip is not True):
        mask_id = sketch_attrib(sketch, "_mask_context_id")
        mask_attr = f' mask="url(#{mask_id})"'
    elif has_mask_style(sketch) and (clip is not True):
        mask_id = sketch_attrib(sketch, "_mask_context_id")
        mask_attr = f' mask="url(#{mask_id})"'

    # Get style class, skipping fill style if gradient/pattern is used
    style_class = get_style_class(
        sketch, style_shape_type, styles_dict, skip_fill=skip_fill_style
    )
    fill_attr_str = f" {fill_attr}" if fill_attr else ""
    fill_rule_attr = ""
    if "even_odd" in sketch_attrib(sketch, "__dict__") and sketch_attrib(
        sketch, "even_odd"
    ):
        fill_rule_attr = ' fill-rule="evenodd"'

    draw_double = sketch_attrib(sketch, "draw_double")
    if draw_double:
        line_width = sketch_attrib(sketch, "line_width")
        if line_width is None:
            line_width = defaults["line_width"]
        double_distance = sketch_attrib(sketch, "double_distance")
        if double_distance is None:
            double_distance = defaults["double_distance"]
        outer_stroke_width = 2 * line_width + double_distance
        line_color = get_color(sketch, "line_color")
        if line_color is None:
            line_color = defaults["line_color"]
        outer_stroke = color_to_svg(check_color(line_color))
        fill_style = get_fill_style_options(sketch, style_shape_type)
        outer_style = f"stroke: {outer_stroke}; stroke-width: {outer_stroke_width}; {fill_style}"
        outer_element = (
            f'<{shape_type}\n'
            f'style="{outer_style}"{fill_rule_attr}{clip_attr}{mask_attr}\n'
            f'{coordinates}\n'
            f'/>'
        )
        double_color = sketch_attrib(sketch, "double_color")
        if double_color is None:
            double_color = defaults["double_color"]
        gap_stroke = color_to_svg(check_color(double_color))
        gap_style = f"stroke: {gap_stroke}; stroke-width: {double_distance}; fill: none;"
        gap_element = (
            f'<{shape_type}\n'
            f'style="{gap_style}"\n'
            f'{coordinates}\n'
            f'/>'
        )
        return f'{outer_element}\n{gap_element}'

    return f'''<{shape_type}
class= "{style_class}"{fill_attr_str}{fill_rule_attr}{clip_attr}{mask_attr}
{coordinates}
/>'''


def has_gradient(sketch):
    """Check if a sketch has gradient configuration.

    Args:
        sketch: The sketch object to check

    Returns:
        bool: True if sketch has gradient configuration
    """
    gradient = sketch_attrib(sketch, "gradient")
    if gradient is None:
        return False
    return gradient.stops is not None


def has_mask_style(sketch):
    """Check if a sketch has mask style configuration."""
    try:
        mask_style = sketch_attrib(sketch, "style").fill_style.mask_style
        if mask_style.stops is not None:
            return True
    except AttributeError:
        pass
    return sketch_attrib(sketch, "msk_stops") is not None


def collect_patterns_and_gradients(canvas):
    """Collect all patterns and gradients from shapes in the canvas.

    Returns:
        tuple: (patterns_dict, gradients_dict) where keys are shape ids
    """
    patterns = {}
    gradients = {}

    if canvas.pages:
        for page in canvas.pages:
            sketches = list(page.sketches)
            while sketches:
                sketch = sketches.pop()
                if sketch_attrib(sketch, "subtype") in [Types.CLIPPED_SKETCH, Types.MASKED_SKETCH]:
                    for sketch_list in sketch.sketches:
                        sketches.extend(sketch_list)
                    continue
                if sketch_attrib(sketch, "tile_svg"):
                    patterns[id(sketch)] = sketch
                if has_gradient(sketch):
                    gradient_key = sketch_attrib(sketch, "_gradient_context_id")
                    if gradient_key is None:
                        gradient_key = f"gradient_{id(sketch)}"
                    if gradient_key not in gradients:
                        gradients[gradient_key] = sketch

    return patterns, gradients


def collect_markers(canvas):
    """Collect all shapes that have markers from the canvas.

    Returns:
        dict: Dictionary mapping sketch id to sketch for shapes with markers
    """
    markers = {}

    def _collect_from_sketch(sketch):
        if sketch_attrib(sketch, "subtype") in [Types.CLIPPED_SKETCH, Types.MASKED_SKETCH]:
            for sketch_list in sketch.sketches:
                for child_sketch in sketch_list:
                    _collect_from_sketch(child_sketch)
            return
        if (
            sketch_attrib(sketch, "draw_markers")
            and sketch_attrib(sketch, "marker_type") != MarkerType.INDICES
        ):
            markers[id(sketch)] = sketch

    if canvas.pages:
        for page in canvas.pages:
            for sketch in page.sketches:
                _collect_from_sketch(sketch)

    return markers


def collect_clip_paths(canvas):
    """Collect all shapes that have clip property from the canvas.

    Returns:
        dict: Dictionary mapping sketch id to (sketch, clip_shape) for shapes with clip property
    """
    clip_paths = {}

    if canvas.pages:
        for page in canvas.pages:
            sketches = list(page.sketches)
            while sketches:
                sketch = sketches.pop()
                subtype = sketch_attrib(sketch, "subtype")
                if subtype == Types.CLIPPED_SKETCH:
                    clip_paths[id(sketch)] = (sketch, sketch.clipper)
                    for sketch_list in sketch.sketches:
                        sketches.extend(sketch_list)
                    continue
                if subtype == Types.MASKED_SKETCH:
                    for sketch_list in sketch.sketches:
                        sketches.extend(sketch_list)
                    continue
                mask = sketch_attrib(sketch, "mask")
                if sketch_attrib(sketch, "clip") is True and mask is not None:
                    clip_paths[id(sketch)] = (sketch, mask)

    return clip_paths


def collect_masks(canvas):
    """Collect all shapes that have opacity mask property from the canvas.

    Returns:
        dict: Dictionary mapping sketch id to (sketch, mask_shape) for shapes with mask property
    """
    masks = {}

    if canvas.pages:
        for page in canvas.pages:
            sketches = list(page.sketches)
            while sketches:
                sketch = sketches.pop()
                subtype = sketch_attrib(sketch, "subtype")
                if subtype == Types.CLIPPED_SKETCH:
                    for sketch_list in sketch.sketches:
                        sketches.extend(sketch_list)
                    continue
                if subtype == Types.MASKED_SKETCH:
                    masks[id(sketch)] = (sketch, sketch.mask)
                    for sketch_list in sketch.sketches:
                        sketches.extend(sketch_list)
                    continue
                mask = sketch_attrib(sketch, "mask")
                clip = sketch_attrib(sketch, "clip")
                if mask is not None and (clip is not True):
                    mask_key = sketch_attrib(sketch, "_mask_context_id")
                    if mask_key not in masks:
                        masks[mask_key] = (sketch, mask)
                elif has_mask_style(sketch):
                    mask_key = sketch_attrib(sketch, "_mask_context_id")
                    if mask_key not in masks:
                        masks[mask_key] = (sketch, None)

    return masks


def get_limits_clippath(canvas):
    """Generate SVG clipPath for canvas limits or inset.

    This is the SVG equivalent of tikz.get_limits_code().

    Args:
        canvas: The canvas object

    Returns:
        tuple: (clippath_id, clippath_def) or (None, None) if no limits
    """
    limits = canvas.limits
    inset = canvas.inset

    if limits is None and inset == 0:
        return None, None

    # Calculate the clip rectangle
    if limits is not None:
        xmin, ymin, xmax, ymax = limits
    elif inset != 0:
        vertices = canvas._all_vertices
        g = inset
        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        xmin = min(x) + g
        xmax = max(x) - g
        ymin = min(y) + g
        ymax = max(y) - g
    else:
        return None, None

    # Create the clip path points
    points = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]

    # Apply transformation matrix if it exists
    if canvas.xform_matrix is not None:
        vertices = homogenize(points) @ canvas.xform_matrix
    else:
        vertices = points

    # Generate SVG polygon points string
    points_str = " ".join([f"{v[0]},{v[1]}" for v in vertices])

    clippath_id = "canvas_limits_clip"
    clippath_def = f'  <clipPath id="{clippath_id}">\n    <polygon points="{points_str}"/>\n  </clipPath>'

    return clippath_id, clippath_def


def generate_pattern_def(sketch, pattern_id, canvas, styles_dict):
    """Generate SVG pattern definition for a shape's tile_svg.

    Args:
        sketch: The shape that has tile_svg property
        pattern_id: Unique ID for this pattern
        canvas: The canvas object for property resolution
        styles_dict: Styles dictionary for rendering the pattern content

    Returns:
        str: SVG <pattern> element
    """
    tile = sketch_attrib(sketch, "tile_svg")
    width = sketch_attrib(sketch, "tile_width")
    height = sketch_attrib(sketch, "tile_height")
    units = sketch_attrib(sketch, "tile_units")

    # Get transformation attributes
    angle = sketch_attrib(sketch, "tile_angle")
    x_shift = sketch_attrib(sketch, "tile_x_shift")
    y_shift = sketch_attrib(sketch, "tile_y_shift")
    scale_x = sketch_attrib(sketch, "tile_scale_x")
    scale_y = sketch_attrib(sketch, "tile_scale_y")

    # Build pattern transform if needed
    transforms = []
    if x_shift != 0 or y_shift != 0:
        transforms.append(f"translate({x_shift}, {y_shift})")
    if angle != 0:
        transforms.append(f"rotate({angle})")
    if scale_x != 1.0 or scale_y != 1.0:
        transforms.append(f"scale({scale_x}, {scale_y})")

    pattern_transform = (
        f' patternTransform="{" ".join(transforms)}"' if transforms else ""
    )

    # Convert tile shape to sketch using canvas
    from ..canvas.draw import create_sketch  # noqa: PLC0415 — circular import
    if tile.type == Types.BATCH:
        # Handle batch - multiple shapes in pattern
        tile_contents = []
        for shape in tile.shapes:
            tile_sketch = create_sketch(shape, canvas)
            if tile_sketch:
                tile_contents.append(svg_shape(tile_sketch, styles_dict))
        tile_content = "\n    ".join(tile_contents)
    else:
        # Single shape - create sketch directly
        tile_sketch = create_sketch(tile, canvas)
        if tile_sketch:
            tile_content = svg_shape(tile_sketch, styles_dict)
        else:
            tile_content = ""

    return f'''  <pattern id="{pattern_id}" x="0" y="0" width="{width}" height="{height}" patternUnits="{units}"{pattern_transform}>
    {tile_content}
  </pattern>'''


def generate_gradient_def(sketch, gradient_id):
    """Generate SVG gradient definition for a shape's gradient.

    Args:
        sketch: The shape that has gradient configuration via style.fill_style.gradient_style
        gradient_id: Unique ID for this gradient

    Returns:
        str: SVG <linearGradient> or <radialGradient> element
    """
    gradient = sketch_attrib(sketch, "gradient")
    gradient_type = gradient.gradient_type
    if gradient.units:
        units = gradient.units.value
    else:
        units = defaults["gradient_units"]
    spread_method = gradient.spread_method
    transform = gradient.transform
    stops = gradient.stops

    context_bbox = sketch_attrib(sketch, "_gradient_context_bbox")

    transform_attr = f' gradientTransform="{transform}"' if transform else ""

    if gradient_type.value == "linear":
        x1, y1 = gradient.axis[0]
        x2, y2 = gradient.axis[1]

        if context_bbox is not None and units == "objectBoundingBox":
            bx, by, bw, bh = context_bbox
            x1 = bx + x1 * bw
            y1 = by + y1 * bh
            x2 = bx + x2 * bw
            y2 = by + y2 * bh
            units = "userSpaceOnUse"

        gradient_start = f'  <linearGradient id="{gradient_id}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" gradientUnits="{units}" spreadMethod="{spread_method}"{transform_attr}>'
        gradient_end = "  </linearGradient>"
    else:  # radial
        cx, cy = gradient.center
        fx, fy = gradient.focal
        r = gradient.radius

        if context_bbox is not None and units == "objectBoundingBox":
            bx, by, bw, bh = context_bbox
            cx = bx + cx * bw
            cy = by + cy * bh
            r = r * min(bw, bh)
            fx = bx + fx * bw
            fy = by + fy * bh
            units = "userSpaceOnUse"

        gradient_start = f'  <radialGradient id="{gradient_id}" cx="{cx}" cy="{cy}" r="{r}" fx="{fx}" fy="{fy}" gradientUnits="{units}" spreadMethod="{spread_method}"{transform_attr}>'
        gradient_end = "  </radialGradient>"

    # Generate color stops
    # Supported stop formats:
    # - (offset, color)
    # - (offset, color, opacity)
    # - {"offset": ..., "color": ..., "opacity": ...} / {"stop_opacity": ...}
    stops_svg = []
    if stops:
        for stop in stops:
            if stop.color is None:
                continue

            color_svg = (
                color_to_svg(stop.color)
                if isinstance(stop.color, (Color, tuple, list, np.ndarray))
                else stop.color
            )
            stop_opacity_attr = (
                f' stop-opacity="{stop.opacity}"'
                if stop.opacity is not None
                else ""
            )
            stops_svg.append(
                f'    <stop offset="{stop.offset}" stop-color="{color_svg}"{stop_opacity_attr} />'
            )

    stops_str = "\n".join(stops_svg)
    return f"{gradient_start}\n{stops_str}\n{gradient_end}"


def generate_clippath_def(sketch, clip_shape, clippath_id, canvas, styles_dict):
    """Generate SVG clipPath definition for a shape's clip property.

    Args:
        sketch: The shape that has clip property
        clip_shape: The shape to use for clipping
        clippath_id: Unique ID for this clipPath
        canvas: The canvas object for property resolution
        styles_dict: Styles dictionary for rendering the clip shape

    Returns:
        str: SVG <clipPath> element
    """
    from ..canvas.draw import create_sketch  # noqa: PLC0415 — circular import
    if isinstance(clip_shape, list):
        clip_contents = []
        for clip_sketch in clip_shape:
            clip_contents.append(svg_shape(clip_sketch, styles_dict))
        clip_content = "\n    ".join(clip_contents)
    elif clip_shape.type == Types.BATCH:
        # Handle batch - multiple shapes in clipPath
        clip_contents = []
        for shape in clip_shape.shapes:
            clip_sketch = create_sketch(shape, canvas)
            if clip_sketch:
                clip_contents.append(svg_shape(clip_sketch, styles_dict))
        clip_content = "\n    ".join(clip_contents)
    else:
        # Single shape - create sketch directly
        clip_sketch = create_sketch(clip_shape, canvas)
        if clip_sketch:
            clip_content = svg_shape(clip_sketch, styles_dict)
        else:
            clip_content = ""

    return f'  <clipPath id="{clippath_id}">\n    {clip_content}\n  </clipPath>'


def generate_mask_def(sketch, mask_shape, mask_id, canvas, styles_dict):
    """Generate SVG mask definition for a shape's mask property.

    Args:
        sketch: The shape that has mask property
        mask_shape: The shape/batch used for masking
        mask_id: Unique ID for this mask
        canvas: The canvas object for property resolution
        styles_dict: Styles dictionary for rendering the mask shape

    Returns:
        str: SVG <mask> element
    """
    from ..canvas.draw import create_sketch  # noqa: PLC0415 — circular import

    def get_mask_stop(stop):
        stop_offset = f"{float(stop.offset) * 100}%"

        if isinstance(stop_color, Color):
            stop_color = color_to_svg(stop_color)

        return stop_offset, stop_color, stop.opacity

    def _stops_to_svg(stops, indent="      "):
        stop_lines = []
        has_color = False
        for stop in stops or []:
            offset, stop_color, stop_opacity = get_mask_stop(stop)
            has_color = stop_color is not None
            if stop_color != color_to_svg(defaults["stop_color"]):
                has_color = True
            stop_opacity_attr = (
                f' stop-opacity="{stop_opacity}"'
                if stop_opacity is not None
                else ""
            )
            stop_lines.append(
                f'{indent}<stop offset="{offset}" stop-color="{stop_color}"{stop_opacity_attr} />'
            )
        return "\n".join(stop_lines), has_color

    def _normalize_svg_units(value):
        if isinstance(value, SvgUnits):
            return value
        if value is None:
            return _normalize_svg_units(
                defaults.get("mask_units", SvgUnits.USER_SPACE_ON_USE.value)
            )
        text = str(value).strip()
        lowered = text.lower()
        if lowered in {"userspaceonuse", "usersapceonuse"}:
            return SvgUnits.USER_SPACE_ON_USE
        if lowered == "objectboundingbox":
            return SvgUnits.OBJECT_BOUNDING_BOX
        if text == SvgUnits.USER_SPACE_ON_USE.value:
            return SvgUnits.USER_SPACE_ON_USE
        if text == SvgUnits.OBJECT_BOUNDING_BOX.value:
            return SvgUnits.OBJECT_BOUNDING_BOX
        return SvgUnits.USER_SPACE_ON_USE

    def _mask_bounds_in_user_space():
        if canvas._all_vertices:
            canvas_bbox = bounding_box(canvas._all_vertices)
            border = (
                defaults["border"] if canvas.border is None else canvas.border
            )
            x = canvas_bbox.southwest[0] - border
            y = canvas_bbox.southwest[1] - border
            width = canvas_bbox.width + 2 * border
            height = canvas_bbox.height + 2 * border
            return x, y, width, height

        mask_bbox = mask_shape.b_box
        return (
            mask_bbox.southwest[0],
            mask_bbox.southwest[1],
            mask_bbox.width,
            mask_bbox.height,
        )

    if mask_shape is None and has_mask_style(sketch):
        msk = sketch_attrib(sketch, "style").fill_style.mask_style

        mask_type = msk.mask_type
        units = msk.units
        spread_method = msk.spread_method
        transform = msk.transform
        stops = msk.stops
        mask_units = msk.mask_units
        mask_content_units = msk.mask_content_units
        mask_units = _normalize_svg_units(mask_units)
        mask_content_units = _normalize_svg_units(mask_content_units)

        transform_attr = (
            f' gradientTransform="{transform}"' if transform else ""
        )

        if mask_type == "linear":
            x1, y1 = msk.axis[0][:2]
            x2, y2 = msk.axis[1][:2]
            gradient_start = (
                f'    <linearGradient id="{mask_id}_gradient" x1="{x1}" y1="{y1}" '
                f'x2="{x2}" y2="{y2}" gradientUnits="{units}" '
                f'spreadMethod="{spread_method}"{transform_attr}>'
            )
            gradient_end = "    </linearGradient>"
        else:
            cx, cy = msk.center[:2]
            r = msk.radius
            fx, fy = msk.focal[:2]
            gradient_start = (
                f'    <radialGradient id="{mask_id}_gradient" cx="{cx}" cy="{cy}" '
                f'r="{r}" fx="{fx}" fy="{fy}" gradientUnits="{units}" '
                f'spreadMethod="{spread_method}"{transform_attr}>'
            )
            gradient_end = "    </radialGradient>"

        stops_str, has_color_stops = _stops_to_svg(stops, indent="      ")
        mask_type_attr = (
            SvgMaskType.LUMINANCE if has_color_stops else SvgMaskType.ALPHA
        )

        context_bbox = sketch_attrib(sketch, "_mask_context_bbox")
        bbox = sketch_attrib(sketch, "b_box")
        if context_bbox is not None:
            x, y, width, height = context_bbox
        elif bbox is not None:
            mask_bbox = bbox
            x = mask_bbox.southwest[0]
            y = mask_bbox.southwest[1]
            width = mask_bbox.width
            height = mask_bbox.height
        else:
            mask_bbox = bounding_box(
                np.array(sketch_attrib(sketch, "vertices"))
            )
            x = mask_bbox.southwest[0]
            y = mask_bbox.southwest[1]
            width = mask_bbox.width
            height = mask_bbox.height

        return (
            f'  <mask id="{mask_id}" maskUnits="{mask_units.value}" '
            f'maskContentUnits="{mask_content_units.value}" mask-type="{mask_type_attr.value}">\n'
            f"{gradient_start}\n"
            f"{stops_str}\n"
            f"{gradient_end}\n"
            f'    <rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'fill="url(#{mask_id}_gradient)" />\n'
            f"  </mask>"
        )

    def _mask_shape_svg(mask_sketch, fill_value="white", fill_opacity=None):
        shape_type = get_shape_type(mask_sketch)
        coordinates = get_coordinates(mask_sketch, shape_type)
        fill_opacity_attr = (
            f' fill-opacity="{fill_opacity}"'
            if fill_opacity is not None
            else ""
        )
        fill_rule_attr = ""
        if "even_odd" in mask_sketch.__dict__ and mask_sketch.even_odd:
            fill_rule_attr = ' fill-rule="evenodd"'

        return (
            f'<{shape_type} fill="{fill_value}"{fill_opacity_attr} '
            f'stroke="none"{fill_rule_attr} {coordinates} />'
        )

    if mask_shape.type == Types.MASK:
        mask_data = mask_shape
        mask_shape = mask_data.shape
        mask_opacity = mask_data.opacity
        mask_stops = mask_data.stops
        mask_axis = mask_data.axis
        if mask_stops is not None and mask_axis is None:
            mask_axis = defaults["mask_axis"]
        mask_units = _normalize_svg_units(None)
        mask_content_units = _normalize_svg_units(None)
    else:
        mask_opacity = sketch_attrib(sketch, "_mask_opacity")
        if mask_opacity is None:
            mask_opacity = 1.0
        mask_stops = sketch_attrib(sketch, "_mask_stops")
        mask_axis = sketch_attrib(sketch, "_mask_axis")
        mask_units = _normalize_svg_units(sketch_attrib(sketch, "_mask_units"))
        mask_content_units = _normalize_svg_units(
            sketch_attrib(sketch, "_mask_content_units")
        )

    gradient_opacity = mask_stops is not None

    if gradient_opacity:
        if hasattr(mask_axis, "start") and hasattr(mask_axis, "end"):
            axis_start = mask_axis.start
            axis_end = mask_axis.end
        else:
            axis_start, axis_end = mask_axis
        mask_bbox = mask_shape.b_box
        bbox_x = mask_bbox.southwest[0]
        bbox_y = mask_bbox.southwest[1]
        bbox_width = mask_bbox.width
        bbox_height = mask_bbox.height
        x1 = bbox_x + float(axis_start[0]) * bbox_width
        y1 = bbox_y + float(axis_start[1]) * bbox_height
        x2 = bbox_x + float(axis_end[0]) * bbox_width
        y2 = bbox_y + float(axis_end[1]) * bbox_height
        mask_x, mask_y, mask_width, mask_height = _mask_bounds_in_user_space()
        gradient_id = f"{mask_id}_opacity_gradient"
        if mask_shape.type == Types.BATCH:
            gradient_contents = []
            for shape in mask_shape.shapes:
                mask_sketch = create_sketch(shape, canvas)
                if mask_sketch:
                    gradient_contents.append(
                        _mask_shape_svg(
                            mask_sketch, fill_value=f"url(#{gradient_id})"
                        )
                    )
            gradient_mask_content = "\n      ".join(gradient_contents)
        else:
            mask_sketch = create_sketch(mask_shape, canvas)
            if mask_sketch:
                gradient_mask_content = _mask_shape_svg(
                    mask_sketch, fill_value=f"url(#{gradient_id})"
                )
            else:
                gradient_mask_content = ""

        stops_str, has_color_stops = _stops_to_svg(
            mask_stops, indent="        "
        )
        mask_type_attr = (
            SvgMaskType.LUMINANCE if has_color_stops else SvgMaskType.ALPHA
        )

        return (
            f'  <linearGradient id="{gradient_id}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" gradientUnits="userSpaceOnUse">\n'
            f"{stops_str}\n"
            f"  </linearGradient>\n"
            f'  <mask id="{mask_id}" maskUnits="{mask_units.value}" '
            f'maskContentUnits="{mask_content_units.value}" mask-type="{mask_type_attr.value}" '
            f'x="{mask_x}" y="{mask_y}" width="{mask_width}" height="{mask_height}">\n'
            f"    {gradient_mask_content}\n"
            f"  </mask>"
        )

    if mask_shape.type == Types.BATCH:
        mask_contents = []
        for shape in mask_shape.shapes:
            mask_sketch = create_sketch(shape, canvas)
            if mask_sketch:
                mask_contents.append(
                    _mask_shape_svg(mask_sketch, fill_opacity=mask_opacity)
                )
        mask_content = "\n    ".join(mask_contents)
    else:
        mask_sketch = create_sketch(mask_shape, canvas)
        if mask_sketch:
            mask_content = _mask_shape_svg(
                mask_sketch, fill_opacity=mask_opacity
            )
        else:
            mask_content = ""

    mask_x, mask_y, mask_width, mask_height = _mask_bounds_in_user_space()

    return (
        f'  <mask id="{mask_id}" maskUnits="{mask_units.value}" '
        f'maskContentUnits="{mask_content_units.value}" mask-type="{SvgMaskType.ALPHA.value}" '
        f'x="{mask_x}" y="{mask_y}" width="{mask_width}" height="{mask_height}">\n'
        f"    {mask_content}\n"
        f"  </mask>"
    )


def generate_defs(canvas, styles_dict):
    """Generate SVG <defs> section with patterns, gradients, clipPaths, and markers.

    Args:
        canvas: The canvas object
        styles_dict: Styles dictionary for rendering pattern content

    Returns:
        str: SVG <defs> section or empty string if no defs needed
    """

    def _canvas_mask_scope_sketch(canvas):
        for sketch in reversed(canvas.active_page.sketches):
            if "_canvas_mask_scope" in sketch.__dict__ and sketch._canvas_mask_scope:
                return sketch
        return None

    patterns, gradients = collect_patterns_and_gradients(canvas)
    markers = collect_markers(canvas)
    clip_paths = collect_clip_paths(canvas)
    masks = collect_masks(canvas)
    filters = collect_filters(canvas)
    limits_clippath_id, limits_clippath_def = get_limits_clippath(canvas)
    canvas_mask_scope = _canvas_mask_scope_sketch(canvas)
    if canvas_mask_scope is not None:
        canvas_mask = canvas_mask_scope.mask
        canvas_clip = bool(canvas_mask_scope.clip)
        canvas_mask_opacity = canvas_mask_scope._mask_opacity
        canvas_mask_stops = canvas_mask_scope._mask_stops
        canvas_mask_axis = canvas_mask_scope._mask_axis
    else:
        canvas_mask = None
        canvas_clip = False
        canvas_mask_opacity = 1.0
        canvas_mask_stops = None
        canvas_mask_axis = None
    canvas_gradient_opacity = canvas_mask_stops is not None
    canvas_mask_clippath_id = None
    canvas_mask_mask_id = None
    if canvas_clip and canvas_mask is not None:
        if canvas_mask_opacity >= 1.0 and not canvas_gradient_opacity:
            canvas_mask_clippath_id = "canvas_mask_clip"
        else:
            canvas_mask_mask_id = "canvas_mask_alpha"

    if (
        not patterns
        and not gradients
        and not markers
        and not clip_paths
        and not masks
        and not filters
        and not limits_clippath_def
        and not canvas_mask_clippath_id
        and not canvas_mask_mask_id
    ):
        return ""

    defs_content = []

    # Generate canvas limits clipPath first (if exists)
    if limits_clippath_def:
        defs_content.append(limits_clippath_def)

    # Generate canvas-level clipPath from canvas._mask (if exists)
    if canvas_mask_clippath_id is not None:
        defs_content.append(
            generate_clippath_def(
                None, canvas_mask, canvas_mask_clippath_id, canvas, styles_dict
            )
        )
    if canvas_mask_mask_id is not None:
        canvas_mask_sketch = MaskSketch(
            mask_opacity=canvas_mask_opacity,
            mask_stops=canvas_mask_stops,
            mask_axis=canvas_mask_axis,
        )
        defs_content.append(
            generate_mask_def(
                canvas_mask_sketch,
                canvas_mask,
                canvas_mask_mask_id,
                canvas,
                styles_dict,
            )
        )

    # Generate clipPath definitions from scope_groups (CLIP_GROUP)
    for page in canvas.pages:
        for scope_group in page.scope_groups:
            if scope_group.subtype == Types.CLIP_GROUP and scope_group.mask is not None:
                clippath_id = f"clippath_{scope_group.id}"
                defs_content.append(
                    generate_clippath_def(
                        None, scope_group.mask, clippath_id, canvas, styles_dict
                    )
                )

    # Generate mask definitions from scope_groups (MASK_GROUP)
    for page in canvas.pages:
        for scope_group in page.scope_groups:
            if scope_group.subtype == Types.MASK_GROUP and scope_group.mask is not None:
                defs_content.append(
                    generate_mask_def(
                        scope_group,
                        scope_group.mask,
                        scope_group._mask_context_id,
                        canvas,
                        styles_dict,
                    )
                )

    # Generate clipPath definitions (must come before shapes that use them)
    for sketch_id, (sketch, clip_shape) in clip_paths.items():
        clippath_id = f"clippath_{sketch_id}"
        defs_content.append(
            generate_clippath_def(
                sketch, clip_shape, clippath_id, canvas, styles_dict
            )
        )

    # Generate mask definitions
    for sketch_id, (sketch, mask_shape) in masks.items():
        mask_id = str(sketch_id)
        if not mask_id.startswith("mask_"):
            mask_id = f"mask_{mask_id}"
        defs_content.append(
            generate_mask_def(sketch, mask_shape, mask_id, canvas, styles_dict)
        )

    # Generate pattern definitions
    for sketch_id, sketch in patterns.items():
        pattern_id = f"pattern_{sketch_id}"
        defs_content.append(
            generate_pattern_def(sketch, pattern_id, canvas, styles_dict)
        )

    # Generate gradient definitions
    for sketch_id, sketch in gradients.items():
        gradient_id = str(sketch_id)
        if not gradient_id.startswith("gradient_"):
            gradient_id = f"gradient_{gradient_id}"
        defs_content.append(generate_gradient_def(sketch, gradient_id))

    # Generate marker definitions
    for sketch_id, sketch in markers.items():
        marker_id = f"marker_{sketch_id}"
        marker_type = sketch_attrib(sketch, "marker_type")
        if not isinstance(marker_type, MarkerType):
            # Try to convert by value first, then by name
            try:
                marker_type = MarkerType(marker_type)
            except ValueError:
                # If that fails, try converting by name (for backward compatibility)
                if isinstance(marker_type, str):
                    marker_type = MarkerType[
                        marker_type.upper().replace(" ", "_").replace("-", "_")
                    ]
                else:
                    raise
        defs_content.append(
            generate_marker_def(
                marker_id, marker_type, sketch, canvas, styles_dict
            )
        )

    # Generate filter definitions
    for sketch_id, svg_filter in filters.items():
        defs_content.append(generate_filter_def(sketch_id, svg_filter))

    defs_str = "\n".join(defs_content)
    return f"  <defs>\n{defs_str}\n  </defs>"


def collect_filters(canvas):
    """Collect all sketches that have an SVG filter configured."""
    filters = {}

    if canvas.pages:
        for page in canvas.pages:
            sketches = list(page.sketches)
            while sketches:
                sketch = sketches.pop()
                if sketch_attrib(sketch, "subtype") in [Types.CLIPPED_SKETCH, Types.MASKED_SKETCH]:
                    for sketch_list in sketch.sketches:
                        sketches.extend(sketch_list)
                    continue
                sketch_dict = sketch_attrib(sketch, "__dict__")
                sketch_filter = sketch_attrib(sketch, "filter")
                if "filter" in sketch_dict and sketch_filter is not None:
                    if not isinstance(sketch_filter, SVG_Filter):
                        raise TypeError(
                            "filter must be an SVG_Filter instance."
                        )
                    if sketch_filter.id is None:
                        sketch_filter.id = f"filter_{id(sketch)}"
                    filters[id(sketch)] = sketch_filter
                    _expand_vertices_for_filter(sketch, sketch_filter, canvas)

    return filters


def _expand_vertices_for_filter(sketch, filter_obj, canvas):
    """Expand canvas._all_vertices to include SVG filter region corners.

    Called at SVG render time so that the filter region is included in
    the viewBox bounding-box calculation.
    """
    filter_units = filter_obj.filterUnits
    if filter_units is not None and str(filter_units) != "userSpaceOnUse":
        return

    def _to_numeric(value):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if text.endswith("%"):
                return None
            try:
                return float(text)
            except ValueError:
                return None
        return None

    region_x = _to_numeric(filter_obj.x)
    region_y = _to_numeric(filter_obj.y)
    region_width = _to_numeric(filter_obj.width)
    region_height = _to_numeric(filter_obj.height)

    if (
        region_x is not None
        and region_y is not None
        and region_width is not None
        and region_height is not None
    ):
        region_corners = [
            (region_x, region_y),
            (region_x + region_width, region_y),
            (region_x + region_width, region_y + region_height),
            (region_x, region_y + region_height),
        ]
        xform_matrix = sketch_attrib(sketch, "xform_matrix")
        if xform_matrix is None:
            return
        transformed_corners = [
            vertex[:2]
            for vertex in homogenize(region_corners) @ xform_matrix
        ]
        canvas._all_vertices.extend(transformed_corners)


def generate_filter_def(sketch_id, svg_filter):
    """Generate SVG filter definition for a sketch filter."""
    filter_svg = svg_filter.to_string(
        pretty=True, include_defs=False, include_xmlns=False
    )
    lines = filter_svg.split("\n")
    indented = "\n".join([f"  {line}" if line else line for line in lines])
    return indented


def header(
    width: int,
    height: int,
    vbox_x,
    vbox_y,
    vbox_width,
    vbox_height,
    color,
    dy,
    styles,
    defs="",
):
    back_color = color_to_svg(color)
    defs_section = f"\n{defs}" if defs else ""
    return rf'''<svg
    xmlns="http://www.w3.org/2000/svg"
    width="{width}pt"
    height="{height}pt"
    viewBox="{vbox_x} {vbox_y} {vbox_width} {vbox_height}">{defs_section}
    {styles}
    <g transform="translate(0 {dy}) scale(1,-1)">
    <rect x="{vbox_x}" y="{vbox_y}" width="{width}" height="{height}" fill="{back_color}" />


'''


def footer():
    return r"""   </g>
</svg>
"""


def get_styles(canvas, styles_dict):
    styles_lines = []
    for key, value_dict in styles_dict.items():
        # Convert dictionary to CSS string
        css_properties = "; ".join(
            [f"{prop}: {val}" for prop, val in value_dict.items()]
        )
        styles_lines.append(f".{key} {{{css_properties}}}")
    lines = "\n".join(styles_lines)
    styles = f"<style>\n{lines}\n</style>"

    return styles


def get_svg_code(canvas):
    vertices = canvas._all_vertices
    if canvas.border is None:
        border = defaults["border"]
    else:
        border = canvas.border
    color = canvas.back_color
    if color is None:
        color = white
    styles_dict = get_styles_dict(canvas)
    styles = get_styles(canvas, styles_dict)
    defs = generate_defs(canvas, styles_dict)

    if not canvas.active_page.sketches or not vertices:
        warnings.warn(
            "Canvas has no drawings/sketches. Writing empty SVG output."
        )
        if canvas.size is not None:
            width, height = canvas.size
            minx, miny = canvas.origin
        else:
            width = 2 * border
            height = 2 * border
            minx = -border
            miny = -border
        dy = 2 * miny + height
        code = [
            header(
                width,
                height,
                minx,
                miny,
                width,
                height,
                color,
                dy,
                styles,
                defs,
            )
        ]
        code.append("<!-- Canvas has no drawings/sketches. -->")
        code.append(footer())
        return "\n".join(code)

    bbox = bounding_box(vertices)
    width = bbox.width + 2 * border
    height = bbox.height + 2 * border

    minx = min([v[0] for v in vertices]) - border
    miny = min([v[1] for v in vertices]) - border
    dy = 2 * miny + height

    # Check if canvas has limits that require clipping
    limits_clippath_id, _ = get_limits_clippath(canvas)
    canvas_mask_scope = None
    for sketch in reversed(canvas.active_page.sketches):
        if "_canvas_mask_scope" in sketch.__dict__ and sketch._canvas_mask_scope:
            canvas_mask_scope = sketch
            break

    if canvas_mask_scope is not None:
        canvas_mask = canvas_mask_scope.mask
        canvas_clip = canvas_mask_scope.clip
        canvas_mask_opacity = canvas_mask_scope._mask_opacity
        canvas_mask_stops = canvas_mask_scope._mask_stops
    else:
        canvas_mask = None
        canvas_clip = False
        canvas_mask_opacity = 1.0
        canvas_mask_stops = None

    canvas_gradient_opacity = canvas_mask_stops is not None
    canvas_mask_clippath_id = None
    canvas_mask_mask_id = None
    if canvas_clip and canvas_mask is not None:
        if canvas_mask_opacity >= 1.0 and not canvas_gradient_opacity:
            canvas_mask_clippath_id = "canvas_mask_clip"
        else:
            canvas_mask_mask_id = "canvas_mask_alpha"

    code = [
        header(
            width, height, minx, miny, width, height, color, dy, styles, defs
        )
    ]

    shapes_code = get_svg_shapes(canvas, styles_dict)

    if limits_clippath_id:
        shapes_code = f'  <g clip-path="url(#{limits_clippath_id})">\n{shapes_code}\n  </g>'

    if canvas_mask_clippath_id:
        shapes_code = f'  <g clip-path="url(#{canvas_mask_clippath_id})">\n{shapes_code}\n  </g>'
    elif canvas_mask_mask_id:
        shapes_code = (
            f'  <g mask="url(#{canvas_mask_mask_id})">\n{shapes_code}\n  </g>'
        )

    code.append(shapes_code)

    code.append(footer())

    return "\n".join(code)

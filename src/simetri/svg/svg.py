from __future__ import annotations

from math import degrees, cos, sin, ceil
from typing import List, Union
from dataclasses import dataclass, field
import warnings

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
    FilterType,
)
from ..graphics.bbox import bounding_box
from ..geometry.geometry import (
    homogenize,
    polar_to_cartesian,
    cartesian_to_polar,
    round_point,
    close_points2,
)
from ..colors.colors import black, white
from ..settings.settings import defaults, svg_defaults
from ..canvas.style_map import shape_style_map, line_style_map, marker_style_map
from ..graphics.sketch import TagSketch, ShapeSketch

from ..colors.colors import Color
from .svg_colors import color_to_svg


from PIL import ImageFont


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
            warnings.warn(f"Could not load font '{font_name}.ttf': {e}. Using default font with scaling.")
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
    if sketch.subtype in d_subtype_draw:
        res = d_subtype_draw[sketch.subtype](sketch)
    elif (
        hasattr(sketch, "draw_markers")
        and sketch.draw_markers
        and sketch.marker_type == MarkerType.INDICES
    ) or (hasattr(sketch, "indices") and sketch.indices):
        res = draw_shape_sketch_with_indices(sketch, ind)
    elif (
        hasattr(sketch, "draw_markers")
        and sketch.draw_markers
        or (hasattr(sketch, "smooth") and sketch.smooth)
    ):
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

    if sketch.back_style == BackStyle.PATTERN and sketch.fill and sketch.closed:
        options += get_pattern_options(sketch)
    if sketch.stroke:
        options += get_line_style_options(sketch)
    if sketch.closed and sketch.fill:
        options += get_fill_style_options(sketch)
    if sketch.smooth:
        options += ["smooth"]
    if sketch.back_style == BackStyle.SHADING and sketch.fill and sketch.closed:
        options += get_shading_options(sketch)
    options = ", ".join(options)
    if options:
        res += f"[{options}]"
    vertices = sketch.vertices
    n = len(vertices)
    str_lines = [f"{vertices[0]}"]
    for i, vertice in enumerate(vertices[1:]):
        if (i + 1) % 8 == 0:
            if i == n - 1:
                str_lines.append(f"-- {vertice} \n")
            else:
                str_lines.append(f"\n\t-- {vertice} ")
        else:
            str_lines.append(f"-- {vertice} ")
    if sketch.closed:
        str_lines.append("-- cycle;\n")
    else:
        str_lines.append(";\n")
    if res:
        res += "".join(str_lines)
    else:
        res = "".join(str_lines)
    return res


def get_line_style_options(sketch, exceptions=None):
    """Returns the options for the line style.

    Args:
        sketch: The sketch object.
        exceptions: Optional exceptions for the line style options.

    Returns:
        list: The line style options as a list.
    """

    options = []
    # line color - use color as fallback if line_color is not set
    if sketch.stroke:
        effective_line_color = sketch.line_color
        if effective_line_color is None and hasattr(sketch, 'color') and sketch.color is not None:
            effective_line_color = sketch.color
        line_color = color_to_svg(effective_line_color)
    else:
        line_color = "none"
    options.append(f"stroke: {line_color};")

    # line opacity - use alpha as fallback if line_alpha is not set
    effective_line_alpha = sketch.line_alpha
    if effective_line_alpha == 1 and hasattr(sketch, 'alpha') and sketch.alpha not in [None, 1]:
        effective_line_alpha = sketch.alpha

    if effective_line_alpha != 1:
        options.append(f"stroke-opacity: {effective_line_alpha};")

    # line width
    if sketch.line_width != 1:
        options.append(f"stroke-width: {sketch.line_width};")

    # linecap
    if sketch.line_cap != LineCap.BUTT:
        options.append(f"stroke-linecap: {sketch.line_cap};")

    # linejoin
    if sketch.line_join != LineJoin.MITER:
        options.append(f"stroke-linejoin: {sketch.line_join};")

    # miter limit
    if hasattr(sketch, "miter_limit") and sketch.miter_limit != 4:
        options.append(f"stroke-miterlimit: {sketch.miter_limit}")

    # dash pattern
    if hasattr(sketch, "line_dash_array") and sketch.line_dash_array:
        options.append(f"stroke-dasharray: {get_dash_pattern(sketch.line_dash_array)}")

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
    # fill color - use color as fallback if fill_color is not set
    if sketch.fill and shape_type != "polyline":
        effective_fill_color = sketch.fill_color
        if effective_fill_color is None and hasattr(sketch, 'color') and sketch.color is not None:
            effective_fill_color = sketch.color
        fill_color = color_to_svg(effective_fill_color)
    else:
        fill_color = "none"

    options.append(f"fill: {fill_color};")

    # fill opacity - use alpha as fallback if fill_alpha is not set
    effective_fill_alpha = sketch.fill_alpha
    if effective_fill_alpha == 1 and hasattr(sketch, 'alpha') and sketch.alpha not in [None, 1]:
        effective_fill_alpha = sketch.alpha

    if effective_fill_alpha != 1:
        options.append(f"fill-opacity: {effective_fill_alpha};")

    if hasattr(sketch, "even_odd") and sketch.even_odd:
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
    if sketch.draw_markers:
        res = sg_to_tikz(sketch, marker_style_map.keys(), attrib_map)
    else:
        res = []

    return res


def draw_shape_sketch_with_indices(sketch, index=0):
    """Draws a shape sketch with circle markers with index numbers in them.

    Args:
        sketch: The shape sketch object.
        index: The index.

    Returns:
        str: The TikZ code for the shape sketch with indices.
    """
    begin_scope = get_begin_scope(index)
    body = get_draw(sketch)
    if body:
        options = get_line_style_options(sketch)
        if sketch.fill and sketch.closed:
            options += get_fill_style_options(sketch)
        if sketch.smooth:
            if sketch.closed:
                options += ["smooth cycle"]
            else:
                options += ["smooth"]
        options = ", ".join(options)
        body += f"[{options}]"
    else:
        body = ""
    vertices = sketch.vertices
    indices = None
    if hasattr(sketch, "ind_offset"):
        offset = sketch.ind_offset
        if isinstance(offset[0], float):
            dx, dy = offset[:2]
            indices = [str((x + dx, y + dy)) for (x, y) in vertices]
        else:
            # offset is in polar coordinates
            center, offset_val = offset
            indices = []
            for vert in vertices:
                x, y = vert[:2]
                r, theta = cartesian_to_polar(x, y, center)
                new_x, new_y = polar_to_cartesian(r + offset_val, theta, center)
                indices.append(f"({new_x}, {new_y})")
    else:
        indices = [str(x) for x in vertices]
    vertices = [str(x) for x in vertices]
    str_lines = [vertices[0]]
    n = len(vertices)
    for i, vertice in enumerate(vertices[1:]):
        if (i + 1) % 6 == 0:
            if i == n - 1:
                str_lines.append(f" -- {vertice}\n")
            else:
                str_lines.append(f"\n\t-- {vertice}")
        else:
            str_lines.append(f"-- {vertice}")

    if body:
        if sketch.closed:
            str_lines.append(" -- cycle;\n")
        str_lines.append(";\n")
    if indices:
        str_lines.append(f"\\node at {indices[0]} {{{0}}};\n")
        for i, pos in enumerate(indices[1:]):
            str_lines.append(f"\\node at {pos} {{{i + 1}}};\n")

    end_scope = get_end_scope()
    if begin_scope == "\\begin{scope}[]\n":
        res = body + "".join(str_lines)
    else:
        res = begin_scope + body + "".join(str_lines) + end_scope

    return res

# This is what TikZ renderer uses
# def draw_shape_sketch_with_markers(sketch):
#     """Draws a shape sketch with markers.

#     Args:
#         sketch: The shape sketch object.

#     Returns:
#         str: The TikZ code for the shape sketch with markers.
#     """
#     # begin_scope = get_begin_scope()
#     body = get_draw(sketch)
#     if body:
#         options = get_line_style_options(sketch)
#         if sketch.fill and sketch.closed:
#             options += get_fill_style_options(sketch)
#         if sketch.smooth and sketch.closed:
#             options += ["smooth cycle"]
#         elif sketch.smooth:
#             options += ["smooth"]
#         options = ", ".join(options)
#         if options:
#             body += f"[{options}]"
#     else:
#         body = ""

#     if sketch.draw_markers:
#         marker_options = ", ".join(get_marker_options(sketch))
#     else:
#         marker_options = ""

#     if sketch.closed and not close_points2(vertices[0], vertices[1]):
#         vertices = [str(x) for x in sketch.vertices + [sketch.vertices[0]]]
#     else:
#         vertices = [str(x) for x in sketch.vertices]

#     str_lines = [vertices[0]]
#     for i, vertice in enumerate(vertices[1:]):
#         if (i + 1) % 6 == 0:
#             str_lines.append(f"\n\t{vertice} ")
#         else:
#             str_lines.append(f" {vertice} ")
#     coordinates = "".join(str_lines)

#     marker = get_enum_value(MarkerType, sketch.marker_type)
#     # marker = sketch.marker_type.value
#     if sketch.markers_only:
#         markers_only = "only marks ,"
#     else:
#         markers_only = ""
#     if sketch.draw_markers and marker_options:
#         body += (
#             f" plot[mark = {marker}, {markers_only}mark options = {{{marker_options}}}] "
#             f"\ncoordinates {{{coordinates}}};\n"
#         )
#     elif sketch.draw_markers:
#         body += (
#             f" plot[mark = {marker}, {markers_only}] coordinates {{{coordinates}}};\n"
#         )
#     else:
#         body += f" plot[tension=.5] coordinates {{{coordinates}}};\n"

#     return body


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
    x, y = sketch.pos[:2]
    elements = []

    # Calculate text properties
    font_size = getattr(sketch, 'font_size', defaults['font_size'])
    font_family = getattr(sketch, 'font_family', defaults['font_family'])
    font_color = getattr(sketch, 'font_color', defaults['font_color'])
    text = getattr(sketch, 'text', None)

    # Text styling
    font_weight = 'bold' if getattr(sketch, 'bold', None) else 'normal'
    font_style = 'italic' if getattr(sketch, 'italic', None) else 'normal'
    text_anchor = 'middle'  # Default for centered text

    # Handle anchor positioning
    anchor = getattr(sketch, 'anchor', None)
    if anchor:
        if anchor in [Anchor.WEST, Anchor.SOUTHWEST, Anchor.NORTHWEST]:
            text_anchor = 'start'
        elif anchor in [Anchor.EAST, Anchor.SOUTHEAST, Anchor.NORTHEAST]:
            text_anchor = 'end'

    # Draw frame if needed
    if getattr(sketch, 'draw_frame', None):
        frame_shape = getattr(sketch, 'frame_shape', defaults['frame_shape'])
        fill_color = getattr(sketch, 'frame_back_color', defaults['frame_back_color']) if getattr(sketch, 'fill', None) else 'none'
        stroke_color = getattr(sketch, 'line_color', defaults['line_color']) if getattr(sketch, 'stroke', None) else 'none'
        stroke_width = getattr(sketch, 'line_width', defaults['line_width'])
        inner_sep = getattr(sketch, 'frame_inner_sep', defaults['frame_inner_sep'])
        minimum_width = getattr(sketch, 'minimum_width', None)

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
    text_decoration = ''
    if getattr(sketch, 'small_caps', None):
        text_decoration = 'font-variant="small-caps" '

    align = getattr(sketch, 'align', None)
    text_width_attr = ''
    if hasattr(sketch, 'text_width') and sketch.text_width:
        text_width_attr = f'textLength="{sketch.text_width}" '

    # Wrap text in a transform group to flip y-axis (prevents upside-down text)
    elements.append(f'<g transform="translate({x} {y}) scale(1,-1)">')
    elements.append(
        f'<text x="0" y="0" '
        f'font-family="{font_family}" '
        f'font-size="{font_size}" '
        f'font-weight="{font_weight}" '
        f'font-style="{font_style}" '
        f'{text_decoration}'
        f'fill="{font_color}" '
        f'text-anchor="{text_anchor}" '
        f'dominant-baseline="middle" '
        f'{text_width_attr}'
        f'>{text}</text>'
    )
    elements.append('</g>')

    return '\n'.join(elements)


def draw_image_sketch(sketch):
    """Converts an ImageSketch to SVG code.

    Args:
        sketch: The ImageSketch object.

    Returns:
        str: The SVG code for the ImageSketch.
    """
    x, y = sketch.pos[:2]
    width, height = sketch.size if sketch.size else (100, 100)
    angle = degrees(sketch.angle) if sketch.angle != 0 else 0

    # Get scale - can be tuple or single value
    if isinstance(sketch.scale, (tuple, list)):
        sx, sy = sketch.scale
    else:
        sx = sy = sketch.scale

    # Get anchor offset
    anchor = sketch.anchor if hasattr(sketch, 'anchor') else Anchor.CENTER

    # Calculate anchor offset
    # In SVG, image x,y is at top-left, so we need to adjust based on anchor
    dx = 0
    dy = 0
    if anchor == Anchor.CENTER:
        dx = -width / 2
        dy = -height / 2
    elif anchor == Anchor.NORTH:
        dx = -width / 2
        dy = 0
    elif anchor == Anchor.SOUTH:
        dx = -width / 2
        dy = -height
    elif anchor == Anchor.EAST:
        dx = -width
        dy = -height / 2
    elif anchor == Anchor.WEST:
        dx = 0
        dy = -height / 2
    elif anchor == Anchor.NORTHEAST:
        dx = -width
        dy = 0
    elif anchor == Anchor.NORTHWEST:
        dx = 0
        dy = 0
    elif anchor == Anchor.SOUTHEAST:
        dx = -width
        dy = -height
    elif anchor == Anchor.SOUTHWEST:
        dx = 0
        dy = -height

    # Build transform string
    transforms = []
    transforms.append(f"translate({x}, {y})")
    if angle != 0:
        transforms.append(f"rotate({angle})")
    if sx != 1 or sy != 1:
        transforms.append(f"scale({sx}, {sy})")
    if dx != 0 or dy != 0:
        transforms.append(f"translate({dx}, {dy})")

    transform_attr = f' transform="{" ".join(transforms)}"' if transforms else ''

    # Use href (modern SVG) or xlink:href (legacy)
    file_path = sketch.file_path if hasattr(sketch, 'file_path') and sketch.file_path else ''

    return f'<image x="0" y="0" width="{width}" height="{height}" href="{file_path}"{transform_attr} />'


def get_marker_path(marker_type, size=1.0):
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
        MarkerType.CIRCLE: ('circle', f'<circle cx="0" cy="0" r="{s}"/>'),
        MarkerType.FCIRCLE: ('circle', f'<circle cx="0" cy="0" r="{s}"/>'),
        MarkerType.SQUARE: ('path', f'<rect x="{-s}" y="{-s}" width="{2*s}" height="{2*s}"/>'),
        MarkerType.SQUARE_F: ('path', f'<rect x="{-s}" y="{-s}" width="{2*s}" height="{2*s}"/>'),
        MarkerType.DIAMOND: ('path', f'<path d="M 0,{s} L {s},0 L 0,{-s} L {-s},0 Z"/>'),
        MarkerType.DIAMOND_F: ('path', f'<path d="M 0,{s} L {s},0 L 0,{-s} L {-s},0 Z"/>'),
        MarkerType.TRIANGLE: ('path', f'<path d="M 0,{s*1.2} L {s*1.04},{-s*0.6} L {-s*1.04},{-s*0.6} Z"/>'),
        MarkerType.TRIANGLE_F: ('path', f'<path d="M 0,{s*1.2} L {s*1.04},{-s*0.6} L {-s*1.04},{-s*0.6} Z"/>'),
        MarkerType.PLUS: ('path', f'<path d="M 0,{s} L 0,{-s} M {s},0 L {-s},0"/>'),
        MarkerType.CROSS: ('path', f'<path d="M {s},{s} L {-s},{-s} M {s},{-s} L {-s},{s}"/>'),
        MarkerType.ASTERISK: ('path', f'<path d="M 0,{s} L 0,{-s} M {s*0.866},{s*0.5} L {-s*0.866},{-s*0.5} M {s*0.866},{-s*0.5} L {-s*0.866},{s*0.5}"/>'),
        MarkerType.STAR: ('path', f'<path d="M 0,{s} L {s*0.224},{s*0.309} L {s*0.951},{s*0.309} L {s*0.363},{-s*0.118} L {s*0.588},{-s*0.809} L 0,{-s*0.382} L {-s*0.588},{-s*0.809} L {-s*0.363},{-s*0.118} L {-s*0.951},{s*0.309} L {-s*0.224},{s*0.309} Z"/>'),
        MarkerType.PENTAGON: ('path', f'<path d="M 0,{s} L {s*0.951},{s*0.309} L {s*0.588},{-s*0.809} L {-s*0.588},{-s*0.809} L {-s*0.951},{s*0.309} Z"/>'),
        MarkerType.PENTAGON_F: ('path', f'<path d="M 0,{s} L {s*0.951},{s*0.309} L {s*0.588},{-s*0.809} L {-s*0.588},{-s*0.809} L {-s*0.951},{s*0.309} Z"/>'),
        MarkerType.HEXAGON: ('path', f'<path d="M {s},0 L {s*0.5},{s*0.866} L {-s*0.5},{s*0.866} L {-s},0 L {-s*0.5},{-s*0.866} L {s*0.5},{-s*0.866} Z"/>'),
        MarkerType.HEXAGON_F: ('path', f'<path d="M {s},0 L {s*0.5},{s*0.866} L {-s*0.5},{s*0.866} L {-s},0 L {-s*0.5},{-s*0.866} L {s*0.5},{-s*0.866} Z"/>'),
        MarkerType.BAR: ('path', f'<path d="M 0,{s} L 0,{-s}"/>'),
        MarkerType.MINUS: ('path', f'<path d="M {s},0 L {-s},0"/>'),
        MarkerType.OPLUS: ('g', f'<circle cx="0" cy="0" r="{s}"/><path d="M 0,{s*0.7} L 0,{-s*0.7} M {s*0.7},0 L {-s*0.7},0"/>'),
        MarkerType.OPLUS_F: ('g', f'<circle cx="0" cy="0" r="{s}"/><path d="M 0,{s*0.7} L 0,{-s*0.7} M {s*0.7},0 L {-s*0.7},0"/>'),
        MarkerType.O_TIMES: ('g', f'<circle cx="0" cy="0" r="{s}"/><path d="M {s*0.7},{s*0.7} L {-s*0.7},{-s*0.7} M {s*0.7},{-s*0.7} L {-s*0.7},{s*0.7}"/>'),
        MarkerType.O_TIMES_F: ('g', f'<circle cx="0" cy="0" r="{s}"/><path d="M {s*0.7},{s*0.7} L {-s*0.7},{-s*0.7} M {s*0.7},{-s*0.7} L {-s*0.7},{s*0.7}"/>'),
        MarkerType.HALF_CIRCLE: ('path', f'<path d="M 0,{s} A {s} {s} 0 0 1 0,{-s} L 0,{s} Z"/>'),
        MarkerType.HALF_CIRCLE_F: ('path', f'<path d="M 0,{s} A {s} {s} 0 0 1 0,{-s} L 0,{s} Z"/>'),
        MarkerType.HALF_SQUARE: ('path', f'<path d="M 0,{s} L {s},{s} L {s},{-s} L 0,{-s} Z"/>'),
        MarkerType.HALF_SQUARE_F: ('path', f'<path d="M 0,{s} L {s},{s} L {s},{-s} L 0,{-s} Z"/>'),
        MarkerType.HALF_DIAMOND: ('path', f'<path d="M 0,{s} L {s},0 L 0,{-s} Z"/>'),
        MarkerType.HALF_DIAMOND_F: ('path', f'<path d="M 0,{s} L {s},0 L 0,{-s} Z"/>'),
    }

    # Default to circle if marker type not found
    if marker_type not in marker_paths:
        return marker_paths.get(MarkerType.CIRCLE, ('circle', f'<circle cx="0" cy="0" r="{s}"/>'))

    return marker_paths[marker_type]


def generate_marker_def(marker_id, marker_type, sketch, canvas=None, styles_dict=None):
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
    # Handle custom shape markers
    if marker_type == MarkerType.SHAPE:
        marker_shape = getattr(sketch, 'marker_shape', None)
        if marker_shape is None:
            # Fallback to circle if no custom shape provided
            marker_type = MarkerType.CIRCLE
        else:
            # Convert marker_shape to sketch
            from ..canvas.draw import create_sketch
            marker_sketch = create_sketch(marker_shape, canvas)

            if marker_sketch is None:
                # Fallback to circle if sketch creation failed
                marker_type = MarkerType.CIRCLE
            else:
                # Get marker size and color from the main sketch
                marker_size = getattr(sketch, 'marker_size', defaults.get('marker_size', 3))
                marker_color = getattr(sketch, 'marker_color', getattr(sketch, 'line_color', defaults.get('line_color', black)))
                marker_alpha = getattr(sketch, 'marker_alpha', getattr(sketch, 'alpha', 1.0))

                if isinstance(marker_color, Color):
                    marker_color_svg = color_to_svg(marker_color)
                else:
                    marker_color_svg = marker_color

                # Get the shape type and coordinates from the custom marker shape
                shape_type = get_shape_type(marker_sketch)

                # Build inline SVG element with fill/stroke attributes
                if shape_type == "polygon" or shape_type == "polyline":
                    vertices = marker_sketch.vertices
                    points = " ".join([f"{x},{y}" for x, y in vertices])

                    # Check if marker shape should be filled
                    if hasattr(marker_sketch, 'fill') and marker_sketch.fill:
                        fill_color = getattr(marker_sketch, 'fill_color', marker_color_svg)
                        if isinstance(fill_color, Color):
                            fill_color = color_to_svg(fill_color)
                        fill_attr = f'fill="{fill_color}" fill-opacity="{marker_alpha}"'
                        stroke_attr = 'stroke="none"'
                    else:
                        fill_attr = 'fill="none"'
                        stroke_attr = f'stroke="{marker_color_svg}" stroke-width="1" stroke-opacity="{marker_alpha}"'

                    shape_svg = f'<{shape_type} points="{points}" {fill_attr} {stroke_attr}/>'
                else:
                    # For other shape types, use a basic path rendering
                    fill_attr = f'fill="{marker_color_svg}" fill-opacity="{marker_alpha}"'
                    stroke_attr = 'stroke="none"'
                    shape_svg = svg_shape(marker_sketch, styles_dict or {})

                # Calculate bounding box of the marker shape
                from ..graphics.bbox import bounding_box
                if hasattr(marker_sketch, 'vertices') and marker_sketch.vertices:
                    bbox = bounding_box(marker_sketch.vertices)
                    vb_width = bbox.width
                    vb_height = bbox.height
                    # Use southwest and northeast corners
                    vb_cx = (bbox.southwest[0] + bbox.northeast[0]) / 2
                    vb_cy = (bbox.southwest[1] + bbox.northeast[1]) / 2

                    # Scale viewBox based on marker_size
                    scale_factor = marker_size / max(vb_width, vb_height, 1)
                    vb_width *= scale_factor * 1.5  # Add some padding
                    vb_height *= scale_factor * 1.5
                else:
                    # Fallback dimensions
                    vb_width = vb_height = marker_size * 2.5
                    vb_cx = vb_cy = 0

                return f'''  <marker id="{marker_id}" markerWidth="{vb_width}" markerHeight="{vb_height}"
      refX="{vb_cx}" refY="{vb_cy}" viewBox="{vb_cx - vb_width/2} {vb_cy - vb_height/2} {vb_width} {vb_height}" orient="auto">
    {shape_svg}
  </marker>'''

    # Handle predefined marker types
    # Get marker properties from sketch
    marker_size = getattr(sketch, 'marker_size', defaults.get('marker_size', 3))
    marker_color = getattr(sketch, 'marker_color', getattr(sketch, 'line_color', defaults.get('line_color', black)))
    marker_alpha = getattr(sketch, 'marker_alpha', getattr(sketch, 'alpha', 1.0))
    marker_fill = getattr(sketch, 'marker_fill', True)
    marker_line_width = getattr(sketch, 'marker_line_width', 1)

    # Convert color
    if isinstance(marker_color, Color):
        marker_color_svg = color_to_svg(marker_color)
    else:
        marker_color_svg = marker_color

    # Determine fill and stroke based on marker type
    is_filled = marker_type.value.endswith('*') or marker_type in [
        MarkerType.FCIRCLE, MarkerType.SQUARE_F, MarkerType.DIAMOND_F,
        MarkerType.TRIANGLE_F, MarkerType.PENTAGON_F, MarkerType.HEXAGON_F,
        MarkerType.OPLUS_F, MarkerType.O_TIMES_F, MarkerType.HALF_CIRCLE_F,
        MarkerType.HALF_SQUARE_F, MarkerType.HALF_DIAMOND_F
    ]

    if is_filled and marker_fill:
        fill_attr = f'fill="{marker_color_svg}" fill-opacity="{marker_alpha}"'
        stroke_attr = 'stroke="none"'
    else:
        fill_attr = 'fill="none"'
        stroke_attr = f'stroke="{marker_color_svg}" stroke-width="{marker_line_width}" stroke-opacity="{marker_alpha}"'

    # Get marker path
    elem_type, path_data = get_marker_path(marker_type, marker_size)

    # Calculate viewBox size (marker_size * 2.5 gives good spacing)
    vb_size = marker_size * 2.5

    return f'''  <marker id="{marker_id}" markerWidth="{vb_size}" markerHeight="{vb_size}"
      refX="0" refY="0" viewBox="{-vb_size} {-vb_size} {vb_size*2} {vb_size*2}" orient="auto">
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
    vertices = sketch.vertices
    if sketch.closed and not close_points2(vertices[0], vertices[-1]):
        vertices = list(vertices) + [vertices[0]]

    # Get marker type
    marker_type = sketch.marker_type
    if not isinstance(marker_type, MarkerType):
        # Try to convert by value first, then by name
        try:
            marker_type = MarkerType(marker_type)
        except ValueError:
            # If that fails, try converting by name (for backward compatibility)
            if isinstance(marker_type, str):
                marker_type = MarkerType[marker_type.upper().replace(' ', '_').replace('-', '_')]
            else:
                raise

    # Check if markers only (no line)
    markers_only = getattr(sketch, 'markers_only', False)

    # Build path for the line
    if not markers_only:
        # Draw the line/shape
        path_data = f'M {vertices[0][0]},{vertices[0][1]}'
        for v in vertices[1:]:
            path_data += f' L {v[0]},{v[1]}'

        # Get line styling
        line_color = getattr(sketch, 'line_color', defaults.get('line_color', black))
        if isinstance(line_color, Color):
            line_color = color_to_svg(line_color)

        line_width = getattr(sketch, 'line_width', defaults.get('line_width', 1))
        line_alpha = getattr(sketch, 'line_alpha', getattr(sketch, 'alpha', 1.0))

        # Get fill styling if closed
        fill_str = '"none"'
        if sketch.fill and sketch.closed:
            fill_color = getattr(sketch, 'fill_color', defaults.get('fill_color', white))
            if isinstance(fill_color, Color):
                fill_color = color_to_svg(fill_color)
            fill_alpha = getattr(sketch, 'fill_alpha', getattr(sketch, 'alpha', 1.0))
            fill_str = f'"{fill_color}" fill-opacity="{fill_alpha}"'

        # Create unique marker ID for this sketch
        marker_id = f'marker_{id(sketch)}'

        # Build path element with marker reference
        path_element = f'<path d="{path_data}" stroke="{line_color}" stroke-width="{line_width}" '
        path_element += f'stroke-opacity="{line_alpha}" fill={fill_str} '
        path_element += f'marker-start="url(#{marker_id})" marker-mid="url(#{marker_id})" marker-end="url(#{marker_id})"/>'

        return path_element
    else:
        # Markers only - draw individual markers at each vertex
        marker_size = getattr(sketch, 'marker_size', defaults.get('marker_size', 3))
        marker_color = getattr(sketch, 'marker_color', getattr(sketch, 'line_color', defaults.get('line_color', black)))
        if isinstance(marker_color, Color):
            marker_color = color_to_svg(marker_color)
        marker_alpha = getattr(sketch, 'marker_alpha', getattr(sketch, 'alpha', 1.0))
        marker_fill = getattr(sketch, 'marker_fill', True)

        # Handle custom shape markers
        if marker_type == MarkerType.SHAPE:
            marker_shape = getattr(sketch, 'marker_shape', None)
            if marker_shape is not None:
                # Import here to avoid circular dependency
                from ..canvas.draw import create_sketch
                # We'll need canvas to create the sketch, but it's not available here
                # For now, render a simple placeholder or fallback
                # TODO: This needs canvas context - consider refactoring
                # For now, fallback to circle
                elem_type, path_data = get_marker_path(MarkerType.CIRCLE, marker_size)
            else:
                elem_type, path_data = get_marker_path(MarkerType.CIRCLE, marker_size)
        else:
            # Get marker path for predefined types
            elem_type, path_data = get_marker_path(marker_type, marker_size)

        # Determine fill and stroke
        is_filled = marker_type.value.endswith('*') or marker_type in [
            MarkerType.FCIRCLE, MarkerType.SQUARE_F, MarkerType.DIAMOND_F
        ]

        if is_filled and marker_fill:
            fill_attr = f'fill="{marker_color}" fill-opacity="{marker_alpha}"'
            stroke_attr = 'stroke="none"'
        else:
            fill_attr = 'fill="none"'
            stroke_attr = f'stroke="{marker_color}" stroke-width="1" stroke-opacity="{marker_alpha}"'

        # Draw markers at each vertex
        elements = []
        for v in vertices:
            x, y = v[0], v[1]
            elements.append(f'<g transform="translate({x},{y})" {fill_attr} {stroke_attr}>')
            elements.append(f'  {path_data}')
            elements.append('</g>')

        return '\n'.join(elements)


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
        if sketch.subtype == Types.TAG_SKETCH:
            code = draw_tag_sketch(sketch)
        elif sketch.subtype == Types.TEX_SKETCH:
            # TexSketch is for TikZ/LaTeX output, skip in SVG
            code = ""
        elif sketch.subtype == Types.IMAGE_SKETCH:
            code = draw_image_sketch(sketch)
        # elif sketch.subtype == Types.PDF_SKETCH:
        #     code = draw_pdf_sketch(sketch)
        # elif sketch.subtype == Types.BBOX_SKETCH:
        #     code = draw_bbox_sketch(sketch)
        # elif sketch.subtype == Types.PATTERN_SKETCH:
        #     code = draw_pattern_sketch(sketch)
        elif hasattr(sketch, 'draw_markers') and sketch.draw_markers:
            # Use marker rendering for shapes with markers enabled
            code = draw_shape_sketch_with_markers(sketch)
        else:
            code = svg_shape(sketch, styles_dict)

        if 'filter' in sketch.__dict__ and sketch.filter is not None:
            filter_id = f'filter_{id(sketch)}'
            code = f'<g filter="url(#{filter_id})">\n{code}\n</g>'
        return code

    pages = canvas.pages

    if pages:
        for i, page in enumerate(pages):
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
            ind = 0
            for sketch in sketches:
                sketch_code = get_sketch_code(sketch, canvas, ind)
                code.append(sketch_code)

        code = "\n".join(code)
    else:
        raise ValueError("No pages found in the canvas.")
    return code


d_shape_types = {
    Types.LINE_SKETCH: "line",
    Types.CIRCLE_SKETCH: "circle",
    Types.ELLIPSE_SKETCH: "ellipse",
    Types.RECTANGLE_SKETCH: "rect",
    Types.SHAPE_SKETCH: "shape",
    Types.TAG_SKETCH: "tag"
}


def get_shape_type(sketch):
    shape_type = d_shape_types[sketch.subtype]
    if shape_type == "shape":
        if sketch.closed:
            shape_type = "polygon"
        else:
            shape_type = "polyline"

    return shape_type


def get_coordinates(sketch, shape_type):
    if shape_type in ["polygon", "polyline"]:
        vertices = sketch.vertices
        verts = ", ".join([f"{x} {y}" for x, y in vertices])

        res = f'points = "{verts}"'
    elif shape_type == "rect":
        pass
    elif shape_type == "circle":
        cx, cy = sketch.center
        r = sketch.radius

        res = f'cx = "{cx}" cy = "{cy}" r = "{r}"'
    elif shape_type == "ellipse":
        cx, cy = sketch.center
        rx = sketch.x_radius
        ry = sketch.y_radius

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
    ''' Get all line and fill styles from the sketches and create a dictionary.
    Name them line_style_1, line_style_2, ...
    fill_style_1, fill_style_2, ...
    Then create a style selector class section:
    <style type="text/css"><![CDATA[
    .line_style_1 {line_width: 2; stroke-dasharray: 2, 4;}
    .fill_style_1 { fill: yellow; stroke: red; }
    .fill_style_2 { fill-opacity: 0.25; fill-rule: evenodd; }
  ]]></style>'''

    def parse_style_string(style_string):
        """Parse a style string into a dictionary."""
        style_dict = {}
        if not style_string:
            return style_dict

        # Split by semicolon and process each property
        parts = style_string.split(';')
        for part in parts:
            part = part.strip()
            if ':' in part:
                key, value = part.split(':', 1)
                style_dict[key.strip()] = value.strip()

        return style_dict

    line_styles = {}
    fill_styles = {}
    line_counter = 1
    fill_counter = 1

    pages = canvas.pages
    if pages:
        for page in pages:
            sketches = page.sketches
            for sketch in sketches:
                # Skip non-shape sketches (like TexSketch)
                if not hasattr(sketch, 'stroke'):
                    continue

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
                if sketch.subtype not in d_shape_types:
                    continue

                shape_type = get_shape_type(sketch)
                if shape_type in ["circle", "ellipse", "polygon", "polyline", "rect"]:
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

        parts = style_string.split(';')
        for part in parts:
            part = part.strip()
            if ':' in part:
                key, value = part.split(':', 1)
                style_dict[key.strip()] = value.strip()

        return style_dict

    class_names = []

    # Get line style and find matching class
    line_style_str = get_line_style_options(sketch)
    if line_style_str:
        line_style_dict = parse_style_string(line_style_str)
        for class_name, style_values in styles_dict.items():
            if class_name.startswith("line_style_") and set(style_values.items()) == set(line_style_dict.items()):
                class_names.append(class_name)
                break

    # Get fill style and find matching class (skip if gradient/pattern is used)
    if not skip_fill and shape_type in ["circle", "ellipse", "polygon", "polyline", "rect"]:
        fill_style_str = get_fill_style_options(sketch, shape_type)
        if fill_style_str:
            fill_style_dict = parse_style_string(fill_style_str)
            for class_name, style_values in styles_dict.items():
                if class_name.startswith("fill_style_") and set(style_values.items()) == set(fill_style_dict.items()):
                    class_names.append(class_name)
                    break

    return " ".join(class_names)

def svg_shape(sketch, styles_dict):
    shape_type = get_shape_type(sketch)
    coordinates = get_coordinates(sketch, shape_type)

    # Check for pattern or gradient fill
    fill_attr = ''
    skip_fill_style = False
    if hasattr(sketch, 'tile_svg') and sketch.tile_svg is not None:
        pattern_id = f'pattern_{id(sketch)}'
        fill_attr = f'fill="url(#{pattern_id})"'
        skip_fill_style = True
    elif has_gradient(sketch):
        gradient_id = f'gradient_{id(sketch)}'
        fill_attr = f'fill="url(#{gradient_id})"'
        skip_fill_style = True

    # Check for clip property (clip=True with mask holding the clipping shape)
    clip_attr = ''
    if hasattr(sketch, 'clip') and sketch.clip is True and hasattr(sketch, 'mask') and sketch.mask is not None:
        clippath_id = f'clippath_{id(sketch)}'
        clip_attr = f' clip-path="url(#{clippath_id})"'

    # Get style class, skipping fill style if gradient/pattern is used
    style_class = get_style_class(sketch, shape_type, styles_dict, skip_fill=skip_fill_style)
    fill_attr_str = f' {fill_attr}' if fill_attr else ''

    return f'''<{shape_type}
class= "{style_class}"{fill_attr_str}{clip_attr}
{coordinates}
/>'''


def has_gradient(sketch):
    """Check if a sketch has gradient configuration.

    Args:
        sketch: The sketch object to check

    Returns:
        bool: True if sketch has gradient configuration
    """
    try:
        grad_style = sketch.style.fill_style.gradient_style
        return grad_style.stops is not None
    except AttributeError:
        return False
    return False


def collect_patterns_and_gradients(canvas):
    """Collect all patterns and gradients from shapes in the canvas.

    Returns:
        tuple: (patterns_dict, gradients_dict) where keys are shape ids
    """
    patterns = {}
    gradients = {}

    if canvas.pages:
        for page in canvas.pages:
            for sketch in page.sketches:
                # Check for tile_svg pattern
                if hasattr(sketch, 'tile_svg') and sketch.tile_svg is not None:
                    patterns[id(sketch)] = sketch
                # Check for gradient
                if has_gradient(sketch):
                    gradients[id(sketch)] = sketch

    return patterns, gradients


def collect_markers(canvas):
    """Collect all shapes that have markers from the canvas.

    Returns:
        dict: Dictionary mapping sketch id to sketch for shapes with markers
    """
    markers = {}

    if canvas.pages:
        for page in canvas.pages:
            for sketch in page.sketches:
                # Check for draw_markers and not INDICES type (INDICES are handled separately)
                if (hasattr(sketch, 'draw_markers') and sketch.draw_markers and
                    hasattr(sketch, 'marker_type') and sketch.marker_type != MarkerType.INDICES):
                    markers[id(sketch)] = sketch

    return markers


def collect_clip_paths(canvas):
    """Collect all shapes that have clip property from the canvas.

    Returns:
        dict: Dictionary mapping sketch id to (sketch, clip_shape) for shapes with clip property
    """
    clip_paths = {}

    if canvas.pages:
        for page in canvas.pages:
            for sketch in page.sketches:
                # Check for clip=True with mask holding the clipping shape
                if hasattr(sketch, 'clip') and sketch.clip is True and hasattr(sketch, 'mask') and sketch.mask is not None:
                    clip_paths[id(sketch)] = (sketch, sketch.mask)

    return clip_paths


def get_limits_clippath(canvas):
    """Generate SVG clipPath for canvas limits or inset.

    This is the SVG equivalent of tikz.get_limits_code().

    Args:
        canvas: The canvas object

    Returns:
        tuple: (clippath_id, clippath_def) or (None, None) if no limits
    """
    if not hasattr(canvas, 'limits') and not hasattr(canvas, 'inset'):
        return None, None

    limits = getattr(canvas, 'limits', None)
    inset = getattr(canvas, 'inset', 0)

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
    if hasattr(canvas, 'xform_matrix') and canvas.xform_matrix is not None:
        vertices = homogenize(points) @ canvas.xform_matrix
    else:
        vertices = points

    # Generate SVG polygon points string
    points_str = ' '.join([f'{v[0]},{v[1]}' for v in vertices])

    clippath_id = 'canvas_limits_clip'
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
    tile = sketch.tile_svg
    width = getattr(sketch, 'tile_width', defaults['tile_width'])
    height = getattr(sketch, 'tile_height', defaults['tile_height'])
    units = getattr(sketch, 'tile_units', defaults['tile_units'])

    # Get transformation attributes
    angle = getattr(sketch, 'tile_angle', defaults['tile_angle'])
    x_shift = getattr(sketch, 'tile_x_shift', defaults['tile_x_shift'])
    y_shift = getattr(sketch, 'tile_y_shift', defaults['tile_y_shift'])
    scale_x = getattr(sketch, 'tile_scale_x', defaults['tile_scale_x'])
    scale_y = getattr(sketch, 'tile_scale_y', defaults['tile_scale_y'])

    # Build pattern transform if needed
    transforms = []
    if x_shift != 0 or y_shift != 0:
        transforms.append(f"translate({x_shift}, {y_shift})")
    if angle != 0:
        transforms.append(f"rotate({angle})")
    if scale_x != 1.0 or scale_y != 1.0:
        transforms.append(f"scale({scale_x}, {scale_y})")

    pattern_transform = f' patternTransform="{" ".join(transforms)}"' if transforms else ''

    # Convert tile shape to sketch using canvas
    from ..graphics.all_enums import Types as GTypes
    from ..canvas.draw import create_sketch

    if hasattr(tile, 'type') and tile.type == GTypes.BATCH:
        # Handle batch - multiple shapes in pattern
        tile_contents = []
        for shape in tile.shapes:
            tile_sketch = create_sketch(shape, canvas)
            if tile_sketch:
                tile_contents.append(svg_shape(tile_sketch, styles_dict))
        tile_content = '\n    '.join(tile_contents)
    else:
        # Single shape - create sketch directly
        tile_sketch = create_sketch(tile, canvas)
        if tile_sketch:
            tile_content = svg_shape(tile_sketch, styles_dict)
        else:
            tile_content = ''

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
    grad = sketch.style.fill_style.gradient_style

    gradient_type = grad.gradient_type if grad.gradient_type is not None else defaults['gradient_type']
    units = grad.units if grad.units is not None else defaults['gr_units']
    spread_method = grad.spread_method if grad.spread_method is not None else defaults['gradient_spread_method']
    transform = grad.transform if grad.transform is not None else defaults['gradient_transform']
    stops = grad.stops if grad.stops is not None else defaults['gr_stops']

    transform_attr = f' gradientTransform="{transform}"' if transform else ''

    if gradient_type == 'linear':
        x1 = grad.x1 if grad.x1 is not None else defaults['gr_x1']
        y1 = grad.y1 if grad.y1 is not None else defaults['gr_y1']
        x2 = grad.x2 if grad.x2 is not None else defaults['gr_x2']
        y2 = grad.y2 if grad.y2 is not None else defaults['gr_y2']

        gradient_start = f'  <linearGradient id="{gradient_id}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" gradientUnits="{units}" spreadMethod="{spread_method}"{transform_attr}>'
        gradient_end = '  </linearGradient>'
    else:  # radial
        cx = grad.cx if grad.cx is not None else defaults['gr_cx']
        cy = grad.cy if grad.cy is not None else defaults['gr_cy']
        r = grad.r if grad.r is not None else defaults['gr_r']
        fx = grad.fx if grad.fx is not None else defaults['gr_fx']
        fy = grad.fy if grad.fy is not None else defaults['gr_fy']

        if fx is None:
            fx = cx
        if fy is None:
            fy = cy

        gradient_start = f'  <radialGradient id="{gradient_id}" cx="{cx}" cy="{cy}" r="{r}" fx="{fx}" fy="{fy}" gradientUnits="{units}" spreadMethod="{spread_method}"{transform_attr}>'
        gradient_end = '  </radialGradient>'

    # Generate color stops
    stops_svg = []
    if stops:
        for offset, color in stops:
            color_svg = color_to_svg(color) if hasattr(color, '__iter__') and not isinstance(color, str) else color
            stops_svg.append(f'    <stop offset="{offset}" stop-color="{color_svg}" />')

    stops_str = '\n'.join(stops_svg)
    return f'{gradient_start}\n{stops_str}\n{gradient_end}'


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
    from ..graphics.all_enums import Types as GTypes
    from ..canvas.draw import create_sketch

    # Convert clip shape to sketch
    if hasattr(clip_shape, 'type') and clip_shape.type == GTypes.BATCH:
        # Handle batch - multiple shapes in clipPath
        clip_contents = []
        for shape in clip_shape.shapes:
            clip_sketch = create_sketch(shape, canvas)
            if clip_sketch:
                clip_contents.append(svg_shape(clip_sketch, styles_dict))
        clip_content = '\n    '.join(clip_contents)
    else:
        # Single shape - create sketch directly
        clip_sketch = create_sketch(clip_shape, canvas)
        if clip_sketch:
            clip_content = svg_shape(clip_sketch, styles_dict)
        else:
            clip_content = ''

    return f'  <clipPath id="{clippath_id}">\n    {clip_content}\n  </clipPath>'


def generate_defs(canvas, styles_dict):
    """Generate SVG <defs> section with patterns, gradients, clipPaths, and markers.

    Args:
        canvas: The canvas object
        styles_dict: Styles dictionary for rendering pattern content

    Returns:
        str: SVG <defs> section or empty string if no defs needed
    """
    patterns, gradients = collect_patterns_and_gradients(canvas)
    markers = collect_markers(canvas)
    clip_paths = collect_clip_paths(canvas)
    filters = collect_filters(canvas)
    limits_clippath_id, limits_clippath_def = get_limits_clippath(canvas)

    if not patterns and not gradients and not markers and not clip_paths and not filters and not limits_clippath_def:
        return ''

    defs_content = []

    # Generate canvas limits clipPath first (if exists)
    if limits_clippath_def:
        defs_content.append(limits_clippath_def)

    # Generate clipPath definitions (must come before shapes that use them)
    for sketch_id, (sketch, clip_shape) in clip_paths.items():
        clippath_id = f'clippath_{sketch_id}'
        defs_content.append(generate_clippath_def(sketch, clip_shape, clippath_id, canvas, styles_dict))

    # Generate pattern definitions
    for sketch_id, sketch in patterns.items():
        pattern_id = f'pattern_{sketch_id}'
        defs_content.append(generate_pattern_def(sketch, pattern_id, canvas, styles_dict))

    # Generate gradient definitions
    for sketch_id, sketch in gradients.items():
        gradient_id = f'gradient_{sketch_id}'
        defs_content.append(generate_gradient_def(sketch, gradient_id))

    # Generate marker definitions
    for sketch_id, sketch in markers.items():
        marker_id = f'marker_{sketch_id}'
        marker_type = sketch.marker_type
        if not isinstance(marker_type, MarkerType):
            # Try to convert by value first, then by name
            try:
                marker_type = MarkerType(marker_type)
            except ValueError:
                # If that fails, try converting by name (for backward compatibility)
                if isinstance(marker_type, str):
                    marker_type = MarkerType[marker_type.upper().replace(' ', '_').replace('-', '_')]
                else:
                    raise
        defs_content.append(generate_marker_def(marker_id, marker_type, sketch, canvas, styles_dict))

    # Generate filter definitions
    for sketch_id, filter_type in filters.items():
        defs_content.append(generate_filter_def(sketch_id, filter_type))

    defs_str = '\n'.join(defs_content)
    return f'  <defs>\n{defs_str}\n  </defs>'


def collect_filters(canvas):
    """Collect all sketches that have an SVG filter configured."""
    filters = {}

    if canvas.pages:
        for page in canvas.pages:
            for sketch in page.sketches:
                if 'filter' in sketch.__dict__ and sketch.filter is not None:
                    filters[id(sketch)] = sketch.filter

    return filters


def generate_filter_def(sketch_id, filter_type):
    """Generate SVG filter definition for a sketch filter."""
    if isinstance(filter_type, FilterType):
        filter_tag = filter_type.value
    else:
        filter_tag = FilterType(filter_type).value

    filter_id = f'filter_{sketch_id}'
    return f'  <filter id="{filter_id}">\n    <{filter_tag} />\n  </filter>'


def header(
    width: int, height: int, vbox_x, vbox_y, vbox_width, vbox_height, color, dy, styles, defs=''
):
    back_color = color_to_svg(color)
    defs_section = f'\n{defs}' if defs else ''
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
        css_properties = '; '.join([f'{prop}: {val}' for prop, val in value_dict.items()])
        styles_lines.append(f'.{key} {{{css_properties}}}')
    lines = '\n'.join(styles_lines)
    styles = f"<style>\n{lines}\n</style>"

    return styles

def get_svg_code(canvas):
    vertices = canvas._all_vertices
    bbox = bounding_box(vertices)
    if canvas.border is None:
        border = defaults["border"]
    else:
        border = canvas.border
    width = bbox.width + 2 * border
    height = bbox.height + 2 * border

    minx = min([v[0] for v in vertices]) - border
    miny = min([v[1] for v in vertices]) - border
    color = canvas.back_color
    if color is None:
        color = white
    dy = 2 * miny + height
    styles_dict = get_styles_dict(canvas)
    styles = get_styles(canvas, styles_dict)
    defs = generate_defs(canvas, styles_dict)

    # Check if canvas has limits that require clipping
    limits_clippath_id, _ = get_limits_clippath(canvas)

    code = [header(width, height, minx, miny, width, height, color, dy, styles, defs)]

    # Wrap shapes in a clipping group if limits are defined
    if limits_clippath_id:
        code.append(f'  <g clip-path="url(#{limits_clippath_id})">')
        code.append(get_svg_shapes(canvas, styles_dict))
        code.append('  </g>')
    else:
        code.append(get_svg_shapes(canvas, styles_dict))

    code.append(footer())

    return "\n".join(code)

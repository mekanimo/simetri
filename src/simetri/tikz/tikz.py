"""TikZ exporter. Draws shapes using the TikZ package for LaTeX.
Sketch objects are converted to TikZ code."""

# This is a proof of concept.
# To do: This whole module needs to be restructured.

from __future__ import annotations

from math import degrees, cos, sin, ceil
from typing import List, Union
from dataclasses import dataclass, field
import warnings

import numpy as np

import simetri.graphics as sg
from ..graphics.common import common_properties
from ..graphics.bbox import bounding_box
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
    Align,
    Anchor,
    ArrowLine,
    BlendMode,
    get_enum_value,
    LineWidth,
    LineDashArray,
    Extent,
)
from ..canvas.style_map import shape_style_map, line_style_map, marker_style_map
from ..settings.settings import defaults, tikz_defaults
from ..geometry.geometry import (
    homogenize,
    polar_to_cartesian,
    cartesian_to_polar,
    round_point,
    close_points2
)
from ..graphics.sketch import TagSketch, ShapeSketch
from ..helpers.utilities import detokenize

from ..colors.colors import Color


NumberOrTex = Union[int, float, str]

np.set_printoptions(legacy="1.21")
array = np.array


enum_map = {}


def anchor_to_tikz(anchor: Anchor | None) -> str | None:
    """Convert Anchor enum values to TikZ-compatible anchor names."""
    if anchor is None:
        return None

    anchor_map = {
        Anchor.BASE_EAST: "base east",
        Anchor.BASE_WEST: "base west",
        Anchor.NORTHEAST: "north east",
        Anchor.NORTHWEST: "north west",
        Anchor.SOUTHEAST: "south east",
        Anchor.SOUTHWEST: "south west",
    }
    return anchor_map.get(anchor, anchor.value)


def _canvas_mask_scope_sketch(canvas):
    page = getattr(canvas, "active_page", None)
    sketches = getattr(page, "sketches", []) if page is not None else []
    for sketch in reversed(sketches):
        if getattr(sketch, "_canvas_mask_scope", False):
            return sketch
    return None


def scope_code_required(canvas: "Canvas") -> bool:
    """Check if canvas-level mask scope sketch exists."""
    return _canvas_mask_scope_sketch(canvas) is not None



def get_back_grid_code(grid: Grid, canvas: "Canvas") -> str:
    """Return the TikZ background grid code.

    Args:
        grid (Grid): The grid object.
        canvas (Canvas): The canvas object.

    Returns:
        str: The background grid code.
    """
    # \usetikzlibrary{backgrounds}
    # \begin{scope}[on background layer]
    # \fill[gray] (current bounding box.south west) rectangle
    # (current bounding box.north east);
    # \draw[white,step=.5cm] (current bounding box.south west) grid
    # (current bounding box.north east);
    # \end{scope}
    grid = canvas.active_page.grid
    back_color = color2tikz(grid.back_color)
    line_color = color2tikz(grid.line_color)
    step = grid.spacing
    lines = ["\\begin{scope}[on background layer]\n"]
    lines.append(f"\\fill[color={back_color}] (current bounding box.south west) ")
    lines.append("rectangle (current bounding box.north east);\n")
    options = []
    if grid.line_dash_array is not None:
        options.append(f"dashed, dash pattern={get_dash_pattern(grid.line_dash_array)}")
    if grid.line_width is not None:
        options.append(f"line width={grid.line_width}")
    if options:
        options = ",".join(options)
        lines.append(f"\\draw[color={line_color}, step={step}, {options}]")
    else:
        lines.append(f"\\draw[color={line_color},step={step}]")
    lines.append("(current bounding box.south west)")
    lines.append(" grid (current bounding box.north east);\n")
    lines.append("\\end{scope}\n")

    return "".join(lines)


def get_limits_code(canvas: "Canvas") -> str:
    """Get the limits of the canvas for clipping.

    Args:
        canvas ("Canvas"): The canvas object.

    Returns:
        str: The limits code for clipping.
    """
    if canvas.limits is not None:
        xmin, ymin, xmax, ymax = canvas.limits
    elif canvas.inset != 0:
        vertices = canvas._all_vertices
        g = canvas.inset
        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        xmin = min(x) + g
        xmax = max(x) - g
        ymin = min(y) + g
        ymax = max(y) - g

    points = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
    vertices = homogenize(points) @ canvas.xform_matrix
    coords = " ".join([f"({v[0]}, {v[1]})" for v in vertices])

    return f"\\clip plot[] coordinates {{{coords}}};\n"


def get_back_code(canvas: "Canvas") -> str:
    """Get the background code for the canvas.

    Args:
        canvas ("Canvas"): The canvas object.

    Returns:
        str: The background code.
    """
    back_color = color2tikz(canvas.back_color)
    return f"\\pagecolor{back_color}\n"


def get_tex_code(canvas: "Canvas") -> str:
    """Convert the sketches in the Canvas to TikZ code.

    Args:
        canvas ("Canvas"): The canvas object.

    Returns:
        str: The TikZ code.
    """

    def get_sketch_code(sketch, canvas, ind):
        """Get the TikZ code for a sketch.

        Args:
            sketch: The sketch object.
            canvas: The canvas object.
            ind: The index.

        Returns:
            tuple: The TikZ code and the updated index.
        """
        if sketch.subtype == Types.TAG_SKETCH:
            code = draw_tag_sketch(sketch)
        elif sketch.subtype == Types.IMAGE_SKETCH:
            code = draw_image_sketch(sketch)
        elif sketch.subtype == Types.HELPLINES_SKETCH:
            code = draw_helplines_sketch(sketch)
        elif sketch.subtype == Types.PDF_SKETCH:
            code = draw_pdf_sketch(sketch)
        elif sketch.subtype == Types.BBOX_SKETCH:
            code = draw_bbox_sketch(sketch)
        elif sketch.subtype == Types.PATTERN_SKETCH:
            code = draw_pattern_sketch(sketch)
        elif sketch.subtype == Types.TEX_SKETCH:
            if sketch.location == TexLoc.NONE:
                code = sketch.code
            else:
                code = ""
        elif sketch.subtype == Types.LATEX_SKETCH:
            code = draw_latex_sketch(sketch)
        elif sketch.subtype == Types.MASK_SKETCH:
            code = ""
        elif sketch.subtype == Types.BATCH_SKETCH:
            parts = []
            for sub_sketch in sketch.sketches:
                sub_code, ind = get_sketch_code(sub_sketch, canvas, ind)
                parts.append(sub_code)
            code = "".join(parts)
        else:
            if (
                hasattr(sketch, "draw_markers")
                and sketch.draw_markers
                and sketch.marker_type == MarkerType.INDICES
            ):
                code = draw_shape_sketch(sketch, ind, canvas)
                ind += 1
            else:
                code = draw_shape_sketch(sketch, canvas=canvas)

        return code, ind

    pages = canvas.pages
    has_sketches = any(page.sketches for page in pages)

    if not has_sketches:
        warnings.warn(
            "Canvas has no drawings/sketches. Writing empty TeX output."
        )
        return canvas.tex.tex_code(
            canvas,
            "% Canvas has no drawings/sketches.\n",
        )

    if pages:
        for i, page in enumerate(pages):
            canvas.active_page = page
            sketches = page.sketches
            back_color = f"\\pagecolor{color2tikz(page.back_color)}"
            if i == 0:
                if page.back_color:
                    code = [back_color]
                else:
                    code = []
            else:
                code.append(defaults["end_tikz"])
                code.append("\\newpage")
                code.append(defaults["begin_tikz"])

            # check for deferred helplines
            helplines = [sk for sk in sketches if sk.subtype == Types.HELPLINES_SKETCH]
            for helpline in helplines:
                helpline.populate(canvas)

            # check for deferred line clipping (RAY / INFINITE)
            lines = [sk for sk in sketches if sk.subtype == Types.LINE_SKETCH]
            for line in lines:
                if hasattr(line, "populate"):
                    line.populate(canvas)

            ind = 0
            for sketch in sketches:
                sketch_code, ind = get_sketch_code(sketch, canvas, ind)
                code.append(sketch_code)

        code = "\n".join(code)
    else:
        raise ValueError("No pages found in the canvas.")
    return canvas.tex.tex_code(canvas, code)


def draw_helplines_sketch(sketch):
    """Draw deferred help lines (grid + optional coordinate system) for TikZ output."""
    x, y = sketch.pos[:2]
    width = sketch.width
    height = sketch.height
    spacing = sketch.spacing
    cs_size = sketch.cs_size
    kwargs = dict(getattr(sketch, "kwargs", {}) or {})

    if spacing in [None, 0]:
        spacing = defaults["help_lines_spacing"]

    # Match draw.grid defaults
    grid_line_width = kwargs.get("line_width", defaults["grid_line_width"])
    grid_line_color = kwargs.get("line_color", defaults["grid_line_color"])
    grid_line_dash_array = kwargs.get("line_dash_array", defaults["grid_line_dash_array"])
    line_alpha = kwargs.get("line_alpha", kwargs.get("alpha", defaults["line_alpha"]))

    def _line_options(line_color, line_width, line_dash_array=None, draw_opacity=None):
        options = [
            f"color={color2tikz(line_color)}",
            f"line width={line_width}",
        ]
        if line_dash_array is not None:
            options.append(f"dash pattern={get_dash_pattern(line_dash_array)}")
        if draw_opacity not in [None, 1]:
            options.append(f"draw opacity={draw_opacity}")
        return ", ".join(options)

    lines = []

    # Grid lines (horizontal + vertical)
    n_h = int(height / spacing)
    n_v = int(width / spacing)
    grid_opts = _line_options(grid_line_color, grid_line_width, grid_line_dash_array, line_alpha)

    for i in range(n_h + 1):
        yi = y + i * spacing
        lines.append(f"\\draw[{grid_opts}] ({x}, {yi}) -- ({x + width}, {yi});")

    for i in range(n_v + 1):
        xi = x + i * spacing
        lines.append(f"\\draw[{grid_opts}] ({xi}, {y}) -- ({xi}, {y + height});")

    # Coordinate system axes + origin marker
    if cs_size and cs_size > 0:
        if "colors" in kwargs:
            x_color, y_color = kwargs["colors"]
        else:
            x_color = defaults["CS_x_color"]
            y_color = defaults["CS_y_color"]

        cs_line_width = kwargs.get("line_width", defaults["CS_line_width"])
        cs_dash = kwargs.get("line_dash_array", None)
        cs_alpha = kwargs.get("line_alpha", kwargs.get("alpha", defaults["line_alpha"]))

        x_axis_opts = _line_options(x_color, cs_line_width, cs_dash, cs_alpha)
        y_axis_opts = _line_options(y_color, cs_line_width, cs_dash, cs_alpha)

        lines.append(f"\\draw[{x_axis_opts}] (0, 0) -- ({cs_size}, 0);")
        lines.append(f"\\draw[{y_axis_opts}] (0, 0) -- (0, {cs_size});")

        origin_color = kwargs.get("line_color", defaults["CS_origin_color"])
        origin_size = defaults["CS_origin_size"]
        lines.append(
            f"\\filldraw[draw={color2tikz(origin_color)}, fill={color2tikz(origin_color)}] "
            f"(0, 0) circle ({origin_size});"
        )

    return "\n".join(lines) + "\n"


class Grid(sg.Shape):
    """Grid shape.

    Args:
        p1: (x_min, y_min)
        p2: (x_max, y_max)
        dx: x step
        dy: y step
    """

    def __init__(self, p1, p2, dx, dy, **kwargs):
        """
        Args:
            p1: (x_min, y_min)
            p2: (x_max, y_max)
            dx: x step
            dy: y step
        """
        self.p1 = p1
        self.p2 = p2
        self.dx = dx
        self.dy = dy
        self.primary_points = sg.Points([p1, p2])
        self.closed = False
        self.fill = False
        self.stroke = True
        self._b_box = None
        super().__init__([p1, p2], xform_matrix=None, subtype=sg.Types.GRID, **kwargs)


def get_min_size(sketch: ShapeSketch) -> str:
    """Returns the minimum size of the tag node.

    Args:
        sketch (ShapeSketch): The shape sketch object.

    Returns:
        str: The minimum size of the tag node.
    """
    options = []
    if sketch.frame_shape == "rectangle":
        if sketch.frame_min_width is None:
            width = defaults["min_width"]
        else:
            width = sketch.frame_min_width
        if sketch.frame_min_height is None:
            height = defaults["min_height"]
        else:
            height = sketch.frame_min_height
        options.append(f"minimum width = {width}")
        options.append(f"minimum height = {height}")
    else:
        if sketch.frame_min_size is None:
            min_size = defaults["min_size"]
        else:
            min_size = sketch.frame_min_size
        options.append(f"minimum size = {min_size}")

    return options


def frame_options(sketch: TagSketch) -> List[str]:
    """Returns the options for the frame of the tag node.

    Args:
        sketch (TagSketch): The tag sketch object.

    Returns:
        List[str]: The options for the frame of the tag node.
    """
    options = []
    if sketch.draw_frame:
        options.append(sketch.frame_shape)
        line_options = get_line_style_options(sketch, frame=True)
        if line_options:
            options.extend(line_options)
        fill_options = get_fill_style_options(sketch, frame=True)
        if fill_options:
            options.extend(fill_options)
        if sketch.text in [None, ""]:
            min_size = get_min_size(sketch)
            if min_size:
                options.extend(min_size)

    return options


def color2tikz(color):
    """Converts a Color object to a TikZ color string.

    Args:
        color (Color): The color object.

    Returns:
        str: The TikZ color string.
    """
    # \usepackage{xcolor}
    # \tikz\node[rounded corners, fill={rgb,255:red,21; green,66; blue,128},
    #                                    text=white, draw=black] {hello world};
    # \definecolor{mycolor}{rgb}{1,0.2,0.3}
    # \definecolor{mycolor}{R_g_b}{255,51,76}
    # \definecolor{mypink1}{rgb}{0.858, 0.188, 0.478}
    # \definecolor{mypink2}{R_g_b}{219, 48, 122}
    # \definecolor{mypink3}{cmyk}{0, 0.7808, 0.4429, 0.1412}
    # \definecolor{mygray}{gray}{0.6}
    if color is None:
        r, g, b, _ = 255, 255, 255, 255
        return f"{{rgb,255:red,{r}; green,{g}; blue,{b}}}"
    r, g, b = color.rgb255
    alpha = color.alpha
    if alpha is not None and alpha < 1:
        return f"{{rgb,255:red,{r}; green,{g}; blue,{b}}}, opacity={alpha}"
    else:
        return f"{{rgb,255:red,{r}; green,{g}; blue,{b}}}"


def get_scope_options(sketch: "Sketch") -> str:
    """Build TikZ scope options from sketch attributes.

    Args:
        sketch ("Sketch"): The sketch to get scope options for.

    Returns:
        str: The scope options as a string.
    """
    options = []

    blend_group = getattr(sketch, "blend_group", False)
    blend_mode = getattr(sketch, "blend_mode", None)
    fill_alpha = getattr(sketch, "fill_alpha", None)
    line_alpha = getattr(sketch, "line_alpha", None)
    text_alpha = getattr(sketch, "text_alpha", None)
    alpha = getattr(sketch, "alpha", None)
    even_odd_rule = getattr(sketch, "even_odd_rule", False)
    transparency_group = getattr(sketch, "transparency_group", False)

    if blend_group:
        options.append(f"blend group={blend_mode}")
    elif blend_mode:
        options.append(f"blend mode={blend_mode}")
    if fill_alpha not in [None, 1]:
        options.append(f"fill opacity={fill_alpha}")
    if line_alpha not in [None, 1]:
        options.append(f"draw opacity={line_alpha}")
    if text_alpha not in [None, 1]:
        options.append(f"text opacity={alpha}")
    if alpha not in [None, 1]:
        options.append(f"opacity={alpha}")
    if even_odd_rule:
        options.append("even odd rule")
    if transparency_group:
        options.append("transparency group")

    return ",".join(options)


def get_clip_code(sketch: "Sketch") -> str:
    """Returns the clip code for a sketch.

    Args:
        sketch ("Sketch"): The sketch to get clip code for.

    Returns:
        str: The clip code as a string.
    """
    mask = getattr(sketch, "mask", None)

    if mask is None:
        return ""

    if mask.subtype == Types.CIRCLE:
        x, y = mask.center[:2]
        res = f"\\clip({x}, {y}) circle ({mask.radius});\n"
    elif mask.subtype == Types.RECTANGLE:
        corners = mask.b_box.corners
        x1, y1 = corners[1][:2]
        x2, y2 = corners[3][:2]
        res = f"\\clip({x1}, {y1}) rectangle ({x2}, {y2});\n"

    elif mask.subtype == Types.SHAPE:
        vertices = getattr(mask, "vertices", None)
        if vertices:
            coords = " -- ".join(f"({x}, {y})" for x, y in vertices)
            if getattr(mask, "closed", False):
                res = f"\\clip {coords} -- cycle;\n"
            else:
                res = f"\\clip {coords};\n"
        else:
            res = ""
    else:
        res = ""

    return res


def get_canvas_scope(canvas):
    """Returns the TikZ code for the canvas scope.

    Args:
        canvas: The canvas object.

    Returns:
        str: The TikZ code for the canvas scope.
    """
    option_list = []
    canvas_mask_scope = _canvas_mask_scope_sketch(canvas)

    if canvas_mask_scope is not None:
        canvas_mask = getattr(canvas_mask_scope, "mask", None)
        canvas_clip = bool(getattr(canvas_mask_scope, "clip", False))
        canvas_mask_opacity = getattr(canvas_mask_scope, "_mask_opacity", 1.0)
        canvas_mask_stops = getattr(canvas_mask_scope, "_mask_stops", None)
        canvas_mask_fade_id = getattr(canvas_mask_scope, "_mask_fade_id", None)
    else:
        canvas_mask = None
        canvas_clip = False
        canvas_mask_opacity = 1.0
        canvas_mask_stops = None
        canvas_mask_fade_id = None

    if canvas_mask_stops is not None and canvas_mask_fade_id:
        option_list.extend([
            "transparency group",
            "blend mode=normal",
            f"scope fading={canvas_mask_fade_id}",
            "fit fading=true",
        ])
    elif canvas_mask_opacity not in [None, 1]:
        option_list.append(f"opacity={canvas_mask_opacity}")

    if option_list:
        res = f"\\begin{{scope}}[{','.join(option_list)}]\n"
    else:
        res = "\\begin{scope}\n"

    if canvas_clip and canvas_mask:
        res += get_clip_code(canvas_mask_scope)

    return res


def draw_batch_sketch(sketch, canvas):
    """Converts a BatchSketch to TikZ code.

    Args:
        sketch: The BatchSketch object.
        canvas: The canvas object.

    Returns:
        str: The TikZ code for the BatchSketch.
    """
    options = get_scope_options(sketch)
    if options:
        res = f"\\begin{{scope}}[{options}]\n"
    else:
        res = ""
    if getattr(sketch, 'clip', None) and getattr(sketch, 'mask', None):
        res += get_clip_code(sketch)
    for item in sketch.sketches:
        if item.subtype in d_sketch_draw:
            res += d_sketch_draw[item.subtype](item, canvas)
        else:
            res += draw_shape_sketch(item, canvas=canvas)

    if getattr(sketch, 'clip', None) and getattr(sketch, 'mask', None):
        res += get_clip_code(sketch)
    if options:
        res += "\\end{scope}\n"

    return res


def draw_bbox_sketch(sketch):
    """Converts a BBoxSketch to TikZ code.

    Args:
        sketch: The BBoxSketch object.
        canvas: The canvas object.

    Returns:
        str: The TikZ code for the BBoxSketch.
    """
    attrib_map = {
        "line_color": "color",
        "line_width": "line width",
        "line_dash_array": "dash pattern",
        "double_color": "double",
        "double_distance": "double distance",
    }
    attrib_list = ["line_color", "line_width", "line_dash_array"]
    options = sg_to_tikz(sketch, attrib_list, attrib_map)
    options = ", ".join(options)
    res = f"\\draw[{options}]"
    x1, y1 = sketch.vertices[1][:2]
    x2, y2 = sketch.vertices[3][:2]
    res += f"({x1}, {y1}) rectangle ({x2}, {y2});\n"

    return res


def draw_lace_sketch(item):
    """Converts a LaceSketch to TikZ code.

    Args:
        item: The LaceSketch object.

    Returns:
        str: The TikZ code for the LaceSketch.
    """
    if item.draw_fragments:
        for fragment in item.fragments:
            draw_shape_sketch(fragment)
    if item.draw_plaits:
        for plait in item.plaits:
            plait.fill = True
            draw_shape_sketch(plait)


def get_draw(sketch):
    """Returns the draw command for sketches.

    Args:
        sketch: The sketch object.

    Returns:
        str: The draw command as a string.
    """
    # sketch.closed, sketch.fill, sketch.stroke, shading
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
        res = "\\draw"
    else:
        has_svg_gradient = bool(getattr(sketch, "gr_stops", None))
        if hasattr(sketch, "back_style"):
            shading = sketch.back_style == BackStyle.SHADING or has_svg_gradient
        else:
            shading = has_svg_gradient
        if not hasattr(sketch, "closed"):
            closed = False
        else:
            closed = sketch.closed
        if not hasattr(sketch, "fill"):
            fill = False
        else:
            fill = sketch.fill
            # Safety check: if fill is still None (edge case), convert to False
            if fill is None:
                fill = False
        if not hasattr(sketch, "stroke"):
            stroke = False
        else:
            stroke = sketch.stroke
            # Safety check: if stroke is still None (edge case), convert to False
            if stroke is None:
                stroke = False

        res = decision_table[(closed, fill, stroke, shading)]

    return res


def _extract_gradient_stop_color(stop):
    if isinstance(stop, dict):
        return stop.get("stop-color", stop.get("stop_color", "black"))

    if isinstance(stop, (list, tuple)) and len(stop) >= 2:
        second = stop[1]
        if isinstance(second, (int, float)) and len(stop) >= 3:
            return stop[2]
        if isinstance(second, (int, float)):
            return "black"
        return second

    return "black"


def _extract_gradient_stop_offset(stop):
    if isinstance(stop, dict):
        offset = stop.get("offset", 0.0)
    elif isinstance(stop, (list, tuple)) and len(stop) >= 1:
        offset = stop[0]
    else:
        offset = 0.0

    if isinstance(offset, str) and offset.endswith("%"):
        try:
            return float(offset[:-1]) / 100.0
        except Exception:
            return 0.0

    try:
        return float(offset)
    except Exception:
        return 0.0


def _resolve_color_token(color_value):
    if isinstance(color_value, Color):
        return color_value

    if isinstance(color_value, str):
        name = color_value.strip()
        if name.startswith("#") and len(name) == 7:
            try:
                r = int(name[1:3], 16)
                g = int(name[3:5], 16)
                b = int(name[5:7], 16)
                return Color(r, g, b)
            except Exception:
                return None

        named = getattr(sg, name, None)
        if isinstance(named, Color):
            return named

    return None


def _color_at_offset(stops, t):
    parsed = []
    for stop in stops:
        offset = max(0.0, min(1.0, _extract_gradient_stop_offset(stop)))
        col = _resolve_color_token(_extract_gradient_stop_color(stop))
        if col is not None:
            parsed.append((offset, col))

    if not parsed:
        return None

    parsed.sort(key=lambda x: x[0])
    t = max(0.0, min(1.0, float(t)))

    if t <= parsed[0][0]:
        return parsed[0][1]
    if t >= parsed[-1][0]:
        return parsed[-1][1]

    for i in range(len(parsed) - 1):
        o1, c1 = parsed[i]
        o2, c2 = parsed[i + 1]
        if o1 <= t <= o2:
            if o2 <= o1:
                return c1
            ratio = (t - o1) / (o2 - o1)
            r1, g1, b1 = c1.rgb
            r2, g2, b2 = c2.rgb
            return Color(
                r1 + (r2 - r1) * ratio,
                g1 + (g2 - g1) * ratio,
                b1 + (b2 - b1) * ratio,
            )

    return parsed[-1][1]


def _shape_bbox(sketch):
    if hasattr(sketch, "vertices") and getattr(sketch, "vertices", None):
        xs = [v[0] for v in sketch.vertices]
        ys = [v[1] for v in sketch.vertices]
        return min(xs), min(ys), max(xs), max(ys)

    if hasattr(sketch, "center") and hasattr(sketch, "radius"):
        cx, cy = sketch.center[:2]
        r = sketch.radius
        return cx - r, cy - r, cx + r, cy + r

    if hasattr(sketch, "center") and hasattr(sketch, "width") and hasattr(sketch, "height"):
        cx, cy = sketch.center[:2]
        w = sketch.width
        h = sketch.height
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    return None


def _user_space_t_span(sketch, x1, y1, x2, y2):
    bbox = _shape_bbox(sketch)
    if bbox is None:
        return None

    bx1, by1, bx2, by2 = bbox
    corners = [(bx1, by1), (bx1, by2), (bx2, by1), (bx2, by2)]

    vx = float(x2) - float(x1)
    vy = float(y2) - float(y1)
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return None

    t_values = [((px - float(x1)) * vx + (py - float(y1)) * vy) / denom for px, py in corners]
    return min(t_values), max(t_values)


def _get_svg_gradient_shading_options(sketch):
    stops = getattr(sketch, "gr_stops", None)
    if not isinstance(stops, (list, tuple)) or len(stops) < 2:
        return None

    first_color = _extract_gradient_stop_color(stops[0])
    last_color = _extract_gradient_stop_color(stops[-1])

    try:
        left = color2tikz(first_color) if isinstance(first_color, Color) else str(first_color)
        right = color2tikz(last_color) if isinstance(last_color, Color) else str(last_color)
    except Exception:
        left, right = "black", "white"

    x1 = getattr(sketch, "gr_x1", 0.0)
    y1 = getattr(sketch, "gr_y1", 0.0)
    x2 = getattr(sketch, "gr_x2", 1.0)
    y2 = getattr(sketch, "gr_y2", 0.0)
    units = getattr(sketch, "gr_units", None)

    if units == "userSpaceOnUse":
        t_span = _user_space_t_span(sketch, x1, y1, x2, y2)
        if t_span is not None:
            t0, t1 = t_span
            c0 = _color_at_offset(stops, t0)
            c1 = _color_at_offset(stops, t1)
            if c0 is not None and c1 is not None:
                left = color2tikz(c0)
                right = color2tikz(c1)

    try:
        angle = degrees(np.arctan2(float(y2) - float(y1), float(x2) - float(x1)))
    except Exception:
        angle = 0.0

    options = [f"left color={left}", f"right color={right}"]
    if abs(angle) > 1e-12:
        options.append(f"shading angle={angle:.2f}")
    return options


def get_frame_options(sketch):
    """Returns the options for the frame of a TagSketch.

    Args:
        sketch: The TagSketch object.

    Returns:
        list: The options for the frame of the TagSketch.
    """
    options = get_line_style_options(sketch)
    options += get_fill_style_options(sketch)
    if sketch.text in [None, ""]:
        if sketch.frame.frame_shape == "rectangle":
            width = sketch.frame.min_width
            height = sketch.frame.min_height
            if not width:
                width = defaults["min_width"]
            if not height:
                height = defaults["min_height"]
            options += "minimum width = {width}, minimum height = {height}"
        else:
            size = sketch.frame.min_size
            if not size:
                size = defaults["min_size"]
            options += f"minimum size = {size}"
    return options


def draw_tag_sketch(sketch):
    """Converts a TagSketch to TikZ code.

    Args:
        sketch: The TagSketch object.
        canvas: The canvas object.

    Returns:
        str: The TikZ code for the TagSketch.
    """

    # \node at (0,0) {some text};
    def get_font_family(sketch):
        default_fonts = [
            defaults["main_font"],
            defaults["sans_font"],
            defaults["mono_font"],
        ]

        if sketch.font_family in default_fonts:
            if sketch.font_family == defaults["main_font"]:
                res = "tex_family", ""
            elif sketch.font_family == defaults["sans_font"]:
                res = "tex_family", "textsf"
            else:  # defaults['mono_font']
                res = "tex_family", "texttt"
        elif sketch.font_family:
            if isinstance(sketch.font_family, FontFamily):
                if sketch.font_family == FontFamily.SANSSERIF:
                    res = "tex_family", "textsf"
                elif sketch.font_family == FontFamily.MONOSPACE:
                    res = "tex_family", "texttt"
                else:
                    res = "tex_family", "textrm"

            elif isinstance(sketch.font_family, str):
                res = "new_family", sketch.font_family.replace(" ", "")

            else:
                raise ValueError(f"Font family {sketch.font_family} not supported.")
        else:
            res = "no_family", None

        return res

    def get_font_size(sketch):
        if sketch.font_size:
            if isinstance(sketch.font_size, FontSize):
                res = "tex_size", sketch.font_size.value
            else:
                res = "num_size", sketch.font_size
        else:
            res = "no_size", None

        return res

    res = []
    text_value = detokenize(sketch.text)
    x, y = sketch.pos[:2]

    options = ""
    if sketch.draw_frame:
        options += "draw"
        if sketch.stroke:
            if sketch.frame_shape != FrameShape.RECTANGLE:
                options += f", {sketch.frame_shape}, "
            line_style_options = get_line_style_options(sketch)
            if line_style_options:
                options += ", " + ", ".join(line_style_options)
            if sketch.frame_inner_sep:
                options += f", inner sep={sketch.frame_inner_sep}"
            if sketch.minimum_width:
                options += f", minimum width={sketch.minimum_width}"
            if sketch.smooth and sketch.frame_shape not in [
                FrameShape.CIRCLE,
                FrameShape.ELLIPSE,
            ]:
                options += ", smooth"

    if sketch.fill and sketch.back_color:
        options += f", fill={color2tikz(sketch.frame_back_color)}"
    if sketch.anchor:
        options += f", anchor={anchor_to_tikz(sketch.anchor)}"
    if sketch.back_style == BackStyle.SHADING and sketch.fill:
        shading_options = get_shading_options(sketch)[0]
        options += ", " + shading_options
    if sketch.back_style == BackStyle.PATTERN and sketch.fill:
        pattern_options = get_pattern_options(sketch)[0]
        options += ", " + pattern_options
    if sketch.align != defaults["tag_align"]:
        options += f", align={sketch.align.value}"
    if sketch.text_width:
        options += f", text width={sketch.text_width}"

    # no_family, tex_family, new_family
    # no_size, tex_size, num_size

    # num_size and new_family {\fontsize{20}{24} \selectfont \Verdana ABCDEFG Hello, World! 25}
    # tex_size and new_family {\large{\selectfont \Verdana ABCDEFG Hello, World! 50}}
    # no_size and new_family {\selectfont \Verdana ABCDEFG Hello, World! 50}

    # tex_family {\textsc{\textit{\textbf{\Huge{\texttt{ABCDG Just a test -50}}}}}};

    # no_family {\textsc{\textit{\textbf{\Huge{ABCDG Just a test -50}}}}};

    if sketch.font_color is not None and sketch.font_color != defaults["font_color"]:
        options += f", text={color2tikz(sketch.font_color)}"
    family, font_family = get_font_family(sketch)
    size, font_size = get_font_size(sketch)
    tex_text = ""
    if sketch.small_caps:
        tex_text += "\\textsc{"

    if sketch.italic:
        tex_text += "\\textit{"

    if sketch.bold:
        tex_text += "\\textbf{"

    if size == "num_size":
        f_size = font_size
        f_size2 = ceil(font_size * 1.2)
        tex_text += f"\\fontsize{{{f_size}}}{{{f_size2}}}\\selectfont "

    elif size == "tex_size":
        tex_text += f"\\{font_size}{{\\selectfont "

    else:
        tex_text += "\\selectfont "

    if family == "new_family":
        tex_text += f"\\{font_family} {text_value}}}"

    elif family == "tex_family":
        if font_family:
                tex_text += f"\\{font_family}{{ {text_value}}}}}"
        else:
            tex_text += f"{{ {text_value}}}"
    else:  # no_family
        tex_text += f"{{ {text_value}}}"

    tex_text = "{" + tex_text

    open_braces = tex_text.count("{")
    close_braces = tex_text.count("}")
    tex_text = tex_text + "}" * (open_braces - close_braces)

    res.append(f"\\node[{options}] at ({x}, {y}) {tex_text};\n")

    return "".join(res)


def draw_latex_sketch(sketch):
    """Convert a LatexSketch to TikZ code."""
    x, y = sketch.pos[:2]
    formula = sketch.formula
    if sketch.bold:
        formula = rf"\mathbf{{{formula}}}"

    options = []
    if sketch.anchor and sketch.anchor != Anchor.CENTER:
        options.append(f"anchor={anchor_to_tikz(sketch.anchor)}")
    if sketch.font_color is not None and sketch.font_color != defaults["font_color"]:
        options.append(f"text={color2tikz(sketch.font_color)}")

    font_size = sketch.font_size or defaults["font_size"]
    baseline_skip = ceil(font_size * 1.2)
    tex_formula = rf"{{\fontsize{{{font_size}}}{{{baseline_skip}}}\selectfont ${formula}$}}"
    option_str = f"[{', '.join(options)}]" if options else ""
    return f"\\node{option_str} at ({x}, {y}) {tex_formula};\n"


def get_dash_pattern(line_dash_array):
    """Returns the dash pattern for a line.

    Args:
        line_dash_array: The dash array for the line.

    Returns:
        str: The dash pattern as a string.
    """
    dash_pattern = []
    for i, dash in enumerate(line_dash_array):
        if i % 2 == 0:
            dash_pattern.extend(["on", f"{dash}pt"])
        else:
            dash_pattern.extend(["off", f"{dash}pt"])

    return " ".join(dash_pattern)


def sg_to_tikz(sketch, attrib_list, attrib_map, conditions=None, exceptions=None):
    """Converts the attributes of a sketch to TikZ options.

    Args:
        sketch: The sketch object.
        attrib_list: The list of attributes to convert.
        attrib_map: The map of attributes to TikZ options.
        conditions: Optional conditions for the attributes.
        exceptions: Optional exceptions for the attributes.

    Returns:
        list: The TikZ options as a list.
    """
    skip = ["marker_color", "fill_color"]
    tikz_way = {"line_width": LineWidth, "line_dash_array": LineDashArray}
    if exceptions:
        skip += exceptions
    d_converters = {
        "line_color": color2tikz,
        "fill_color": color2tikz,
        "double_color": color2tikz,
        "draw": color2tikz,
        "line_dash_array": get_dash_pattern,
    }
    options = []
    for attrib in attrib_list:
        if attrib not in attrib_map:
            continue
        if conditions and attrib in conditions and not conditions[attrib]:
            continue
        if attrib in tikz_way:
            value = getattr(sketch, attrib)
            if isinstance(value, tikz_way[attrib]):
                option = value.value
                options.append(option)
                continue
            if isinstance(value, str):
                if value in tikz_way[attrib]:
                    options.append(value)
                continue

        tikz_attrib = attrib_map[attrib]
        if hasattr(sketch, attrib):
            value = getattr(sketch, attrib)
            if value is not None and tikz_attrib in list(attrib_map.values()):
                if attrib in ["smooth", "draw_double"]:  # boolean values
                    if value:
                        options.append(attrib)

                elif attrib in skip:
                    value = color2tikz(getattr(sketch, attrib))
                    options.append(f"{tikz_attrib}={value}")

                elif value != tikz_defaults[tikz_attrib]:
                    if attrib in d_converters:
                        value = d_converters[attrib](value)
                    options.append(f"{tikz_attrib}={value}")

    return options


# Tex class went to tex.py

def get_line_style_options(sketch, exceptions=None):
    """Returns the options for the line style.

    Args:
        sketch: The sketch object.
        exceptions: Optional exceptions for the line style options.

    Returns:
        list: The line style options as a list.
    """
    attrib_map = {
        "double_color": "double",
        "double_distance": "double distance",
        "line_color": "color",
        "line_width": "line width",
        "line_dash_array": "dash pattern",
        "line_cap": "line cap",
        "line_join": "line join",
        "line_miter_limit": "miter limit",
        "line_dash_phase": "dash phase",
        "line_alpha": "draw opacity",
        "smooth": "smooth",
        "fillet_radius": "rounded corners",
    }
    attribs = list(line_style_map.keys())
    if hasattr(sketch, "stroke") and sketch.stroke:
        if exceptions and "draw_fillets" not in exceptions:
            conditions = {"fillet_radius": sketch.draw_fillets}
        else:
            conditions = None
        if sketch.line_alpha in [None, 1]:
            attribs.remove("line_alpha")
        if not sketch.draw_double:
            attribs.remove("double_color")
            attribs.remove("double_distance")
        if not sketch.smooth:
            attribs.remove("smooth")
        res = sg_to_tikz(sketch, attribs, attrib_map, conditions, exceptions)
    else:
        res = []

    return res


def get_fill_style_options(sketch, exceptions=None, frame=False):
    """Returns the options for the fill style.

    Args:
        sketch: The sketch object.
        exceptions: Optional exceptions for the fill style options.
        frame: Optional flag for frame fill style.

    Returns:
        list: The fill style options as a list.
    """
    attrib_map = {
        "fill_color": "fill",
        "fill_alpha": "fill opacity",
        #'fill_mode': 'even odd rule',
        "blend_mode": "blend mode",
        "frame_back_color": "fill",
    }
    attribs = list(shape_style_map.keys())
    if sketch.fill_alpha in [None, 1]:
        attribs.remove("fill_alpha")
    if sketch.fill and not sketch.back_style == BackStyle.PATTERN:
        res = sg_to_tikz(sketch, attribs, attrib_map, exceptions=exceptions)
        if frame:
            res = [f"fill = {color2tikz(getattr(sketch, 'back_color'))}"] + res
    else:
        res = []

    return res


def get_axis_shading_colors(sketch):
    """Returns the shading colors for the axis.

    Args:
        sketch: The sketch object.

    Returns:
        str: The shading colors for the axis.
    """

    def get_color(color, color_key):
        if isinstance(color, Color):
            res = color2tikz(color)
        else:
            res = defaults[color_key]

        return res

    left = get_color(sketch.shade_left_color, "shade_left_color")
    right = get_color(sketch.shade_right_color, "shade_right_color")
    top = get_color(sketch.shade_top_color, "shade_top_color")
    bottom = get_color(sketch.shade_bottom_color, "shade_bottom_color")
    middle = get_color(sketch.shade_middle_color, "shade_middle_color")

    axis_colors = {
        ShadeType.AXIS_BOTTOM_MIDDLE: f"bottom color={bottom}, middle color={middle}",
        ShadeType.AXIS_LEFT_MIDDLE: f"left color={left}, middle color={middle}",
        ShadeType.AXIS_RIGHT_MIDDLE: f"right color={right}, middle color={middle}",
        ShadeType.AXIS_TOP_MIDDLE: f"top color={top}, middle color={middle}",
        ShadeType.AXIS_LEFT_RIGHT: f"left color={left}, right color={right}",
        ShadeType.AXIS_TOP_BOTTOM: f"top color={top}, bottom color={bottom}",
    }

    res = axis_colors[sketch.shade_type]
    return res


def get_bilinear_shading_colors(sketch):
    """Returns the shading colors for the bilinear shading.

    Args:
        sketch: The sketch object.

    Returns:
        str: The shading colors for the bilinear shading.
    """
    res = []
    if sketch.shade_upper_left_color:
        res.append(f"upper left = {color2tikz(sketch.shade_upper_left_color)}")
    if sketch.shade_upper_right_color:
        res.append(f"upper right = {color2tikz(sketch.shade_upper_right_color)}")
    if sketch.shade_lower_left_color:
        res.append(f"lower left = {color2tikz(sketch.shade_lower_left_color)}")
    if sketch.shade_lower_right_color:
        res.append(f"lower right = {color2tikz(sketch.shade_lower_right_color)}")

    return ", ".join(res)


def get_radial_shading_colors(sketch):
    """Returns the shading colors for the radial shading.

    Args:
        sketch: The sketch object.

    Returns:
        str: The shading colors for the radial shading.
    """
    res = []
    if sketch.shade_type == ShadeType.RADIAL_INNER:
        res.append(f"inner color = {color2tikz(sketch.shade_inner_color)}")
    elif sketch.shade_type == ShadeType.RADIAL_OUTER:
        res.append(f"outer color = {color2tikz(sketch.shade_outer_color)}")
    elif sketch.shade_type == ShadeType.RADIAL_INNER_OUTER:
        res.append(f"inner color = {color2tikz(sketch.shade_inner_color)}")
        res.append(f"outer color = {color2tikz(sketch.shade_outer_color)}")

    return ", ".join(res)


axis_shading_types = [
    ShadeType.AXIS_BOTTOM_MIDDLE,
    ShadeType.AXIS_LEFT_MIDDLE,
    ShadeType.AXIS_RIGHT_MIDDLE,
    ShadeType.AXIS_TOP_MIDDLE,
    ShadeType.AXIS_LEFT_RIGHT,
    ShadeType.AXIS_TOP_BOTTOM,
]

radial_shading_types = [
    ShadeType.RADIAL_INNER,
    ShadeType.RADIAL_OUTER,
    ShadeType.RADIAL_INNER_OUTER,
]


def get_shading_options(sketch):
    """Returns the options for the shading.

    Args:
        sketch: The sketch object.

    Returns:
        list: The shading options as a list.
    """
    shade_type = sketch.shade_type
    if shade_type in axis_shading_types:
        res = get_axis_shading_colors(sketch)
        if sketch.shade_axis_angle:
            res += f", shading angle={sketch.shade_axis_angle}"
    elif shade_type == ShadeType.BILINEAR:
        res = get_bilinear_shading_colors(sketch)
    elif shade_type in radial_shading_types:
        res = get_radial_shading_colors(sketch)
    elif shade_type == ShadeType.BALL:
        res = f"ball color = {color2tikz(sketch.shade_ball_color)}"
    elif shade_type == ShadeType.COLORWHEEL:
        res = "shading=color wheel"
    elif shade_type == ShadeType.COLORWHEEL_BLACK:
        res = "shading=color wheel black center"
    elif shade_type == ShadeType.COLORWHEEL_WHITE:
        res = "shading=color wheel white center"

    return [res]


def get_pattern_options(sketch):
    """Returns the options for the patterns.

    Args:
        sketch: The sketch object.

    Returns:
        list: The pattern options as a list.
    """
    pattern_type = sketch.pattern_type
    if pattern_type:
        distance = sketch.pattern_distance
        options = f"pattern={{{pattern_type}[distance={distance}, "
        angle = degrees(sketch.pattern_angle)
        if angle:
            options += f"angle={angle}, "
        line_width = sketch.pattern_line_width
        if line_width:
            options += f"line width={line_width}, "
        x_shift = sketch.pattern_x_shift
        if x_shift:
            options += f"xshift={x_shift}, "
        y_shift = sketch.pattern_y_shift
        if y_shift:
            options += f"yshift={y_shift}, "
        if pattern_type in ["Stars", "Dots"]:
            radius = sketch.pattern_radius
            if radius:
                options += f"radius={radius}, "
            if pattern_type == "Stars":
                points = sketch.pattern_points
                if points:
                    options += f"points={points}, "
        options = options.strip()
        if options.endswith(","):
            options = options[:-1]
        options += "]"
        color = sketch.pattern_color
        if color and color != sg.black:
            options += f", pattern color={color2tikz(color)}, "

        options += "}"
        res = [options]
    else:
        res = []

    return res


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
    for i, vertex in enumerate(vertices[1:]):
        if (i + 1) % 6 == 0:
            if i == n - 1:
                str_lines.append(f" -- {vertex}\n")
            else:
                str_lines.append(f"\n\t-- {vertex}")
        else:
            str_lines.append(f"-- {vertex}")

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


def draw_shape_sketch_with_markers(sketch):
    """Draws a shape sketch with markers.

    Args:
        sketch: The shape sketch object.

    Returns:
        str: The TikZ code for the shape sketch with markers.
    """
    # begin_scope = get_begin_scope()
    body = get_draw(sketch)
    if body:
        options = get_line_style_options(sketch)
        if sketch.fill and sketch.closed:
            options += get_fill_style_options(sketch)
        if sketch.smooth and sketch.closed:
            options += ["smooth cycle"]
        elif sketch.smooth:
            options += ["smooth"]
        options = ", ".join(options)
        if options:
            body += f"[{options}]"
    else:
        body = ""

    if sketch.draw_markers:
        marker_options = ", ".join(get_marker_options(sketch))
    else:
        marker_options = ""

    if sketch.closed and not close_points2(sketch.vertices[0], sketch.vertices[-1]):
        vertices = [str(x) for x in sketch.vertices + [sketch.vertices[0]]]
    else:
        vertices = [str(x) for x in sketch.vertices]

    str_lines = [vertices[0]]
    for i, vertex in enumerate(vertices[1:]):
        if (i + 1) % 6 == 0:
            str_lines.append(f"\n\t{vertex} ")
        else:
            str_lines.append(f" {vertex} ")
    coordinates = "".join(str_lines)

    marker_type = sketch.marker_type

    # Handle custom shape markers
    if marker_type == MarkerType.SHAPE:
        marker_shape = getattr(sketch, 'marker_shape', None)
        if marker_shape is not None:
            # For custom shapes in TikZ, we need to manually place the shape at each vertex
            # since plot[mark=...] only supports predefined marker types
            # Draw the path first (if not markers_only)
            if not sketch.markers_only:
                if body:
                    body += f" plot coordinates {{{coordinates}}};\n"
                else:
                    body = f"\\draw plot coordinates {{{coordinates}}};\n"

            # TODO: Add code to place custom marker_shape at each vertex
            # This requires defining a pic or using nodes
            # For now, we'll use a note comment
            body += f"% Custom marker shape at vertices (not yet implemented in TikZ)\n"
            for vertex in sketch.vertices:
                x, y = vertex[0], vertex[1]
                body += f"% Marker at ({x}, {y})\n"

            return body
        else:
            # Fallback to default marker if no shape provided
            marker = get_enum_value(MarkerType, MarkerType.FCIRCLE)
    else:
        marker = get_enum_value(MarkerType, marker_type)

    # Standard marker handling for predefined types
    if sketch.markers_only:
        markers_only = "only marks ,"
    else:
        markers_only = ""
    if sketch.draw_markers and marker_options:
        body += (
            f" plot[mark = {marker}, {markers_only}mark options = {{{marker_options}}}] "
            f"\ncoordinates {{{coordinates}}};\n"
        )
    elif sketch.draw_markers:
        body += (
            f" plot[mark = {marker}, {markers_only}] coordinates {{{coordinates}}};\n"
        )
    else:
        body += f" plot[tension=.5] coordinates {{{coordinates}}};\n"

    return body


def get_begin_scope(ind=None):
    """Returns \begin{scope}[every node/.append style=nodestyle{ind}].

    Args:
        ind: Optional index for the scope.

    Returns:
        str: The begin scope string.
    """
    if ind is None:
        res = "\\begin{scope}[]\n"
    else:
        res = f"\\begin{{scope}}[every node/.append style=nodestyle{ind}]\n"

    return res


def get_end_scope():
    """Returns \\end{scope}.

    Returns:
        str: The end scope string.
    """
    return "\\end{scope}\n"


def draw_pattern_sketch(sketch):
    """Draws a pattern sketch.

    Args:
        sketch: The pattern sketch object.

    Returns:
        str: The TikZ code for the pattern sketch.
    """
    begin_scope = "\\begin{scope}"

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
        begin_scope += f"[{options}]\n"
    end_scope = get_end_scope()

    draw = get_draw(sketch)
    if not draw:
        return ""
    all_vertices = sketch.kernel_vertices @ sketch.all_matrices
    vertices_list = np.hsplit(all_vertices, sketch.count)
    shapes = []
    for vertices in vertices_list:
        vertices @= sketch.xform_matrix
        vertices = [tuple(vert) for vert in vertices[:, :2].tolist()]
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
        if sketch.closed:
            str_lines.append("-- cycle;\n")
        else:
            str_lines.append(";\n")
        shapes.append(draw + "".join(str_lines))

    return begin_scope + f"[{options}]\n" + "\n".join(shapes) + end_scope


def draw_sketch(sketch):
    """Draws a plain shape sketch.

    Args:
        sketch: The shape sketch object.

    Returns:
        str: The TikZ code for the plain shape sketch.
    """
    res = get_draw(sketch)
    if not res:
        return ""
    options = []
    svg_gradient_options = _get_svg_gradient_shading_options(sketch)
    has_svg_gradient = bool(svg_gradient_options) and sketch.fill and sketch.closed

    if sketch.back_style == BackStyle.PATTERN and sketch.fill and sketch.closed:
        options += get_pattern_options(sketch)
    if sketch.stroke:
        options += get_line_style_options(sketch)
    if sketch.closed and sketch.fill and not has_svg_gradient:
        options += get_fill_style_options(sketch)
    if sketch.smooth:
        options += ["smooth"]
    if sketch.back_style == BackStyle.SHADING and sketch.fill and sketch.closed:
        options += get_shading_options(sketch)
    elif has_svg_gradient:
        options += svg_gradient_options
    options = ", ".join(options)
    if options:
        res += f"[{options}]"
    vertices = sketch.vertices
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
    if sketch.closed:
        str_lines.append("-- cycle;\n")
    else:
        str_lines.append(";\n")
    if res:
        res += "".join(str_lines)
    else:
        res = "".join(str_lines)
    return res


def draw_tex_sketch(sketch):
    """Draws a TeX sketch.

    Args:
        sketch: The TeX sketch object.

    Returns:
        str: The TeX code for the TeX sketch.
    """

    return sketch.code


def _format_scalar(value: NumberOrTex) -> str:
    if isinstance(value, str):
        return value
    # Keep output stable/compact for TeX.
    return f"{float(value):.12g}"


def _format_translation(value: NumberOrTex, unit: str) -> str:
    if isinstance(value, str):
        return value
    # TikZ translation components are a TeX dimension; add an explicit unit.
    return f"{float(value):.12g}{unit}"

def transform_image(
    transform_matrix,
    image_url: str,
    *,
    translation_unit: str = "" # "bp",
) -> str:
    """Return a TikZ node string for an \\includegraphics transformed by a 3x3 matrix.

    The input matrix is interpreted as a 2D homogeneous transform in *row-major* form.

        Assumptions (based on your pipeline):
        - `transform_matrix` is always a 3x3 row-major matrix.
        - Points are row-vectors and you apply transforms as `points_mat @ transform_mat`.
            That implies the affine 3x3 is in the row-vector convention:

                [[m11, m12,   0],
                 [m21, m22,   0],
                 [m31, m32,   1]]

    TikZ/PGF's `cm={a,b,c,d,(tx,ty)}` corresponds to:
        x' = a x + c y + tx
        y' = b x + d y + ty

    Mapping to TikZ `cm={a,b,c,d,(tx,ty)}`:
        a=m11, b=m12, c=m21, d=m22, tx=m31, ty=m32

    Notes:
    - Numeric translations are emitted with `translation_unit` (default `bp`).
      Use `bp` for PostScript points (1in=72bp) and `pt` for TeX points (1in=72.27pt).
      Pass strings (e.g. "1cm") in the matrix to control units per-value.
    """
    # See notes: TikZ cm
    # Row-vector convention: [[m11,m12,0],[m21,m22,0],[tx,ty,1]]
    m11 = transform_matrix[0, 0]
    m12 = transform_matrix[0, 1]
    m21 = transform_matrix[1, 0]
    m22 = transform_matrix[1, 1]
    m31 = transform_matrix[2, 0]
    m32 = transform_matrix[2, 1]

    a = _format_scalar(m11)
    b = _format_scalar(m12)
    c = _format_scalar(m21)
    d = _format_scalar(m22)
    tx = _format_translation(m31, translation_unit)
    ty = _format_translation(m32, translation_unit)

    # `transform shape` ensures the \\includegraphics is transformed (not just the anchor).
    return (
        "\\node[inner sep=0pt, transform shape, "
        f"cm={{{a},{b},{c},{d},({tx},{ty})}}] "
        f"{{\\includegraphics{{{image_url}}}}};"
    )

def draw_image_sketch(sketch):
    """Draws an image sketch.

    Args:
        sketch: The image sketch object.

    Returns:
        str: The TikZ code for the image sketch.
    """
    begin_scope = get_begin_scope()
    options = get_line_style_options(sketch)
    options += get_fill_style_options(sketch, frame=True)
    # options = ", ".join(options)
    # if options:
    #     res += f"[{options}]"
    x, y = sketch.pos[:2]
    # res += f" ({x}, {y}) "
    if sketch.angle != 0:
        angle = degrees(sketch.angle)
        options.append(f"rotate = {angle}")

    if sketch.scale != (1, 1):
        sx, sy = sketch.scale
        options.append(f"xscale = {sx}, yscale = {sy}")

    # res += f"node[anchor={sketch.anchor.value}, rotate={angle}] {{\\includegraphics{{{sketch.file_path}}}}};\n"
    if sketch.anchor != Anchor.CENTER:
        options.append(f"anchor = {anchor_to_tikz(sketch.anchor)}")

    # res = f"\\node[draw, {', '.join(options)}]at({x}, {y}) {{\\includegraphics{{{sketch.file_path}}}}};\n"
    res = f"\\node[{', '.join(options)}]at({x}, {y}) {{\\includegraphics{{{sketch.file_path}}}}};\n"
    end_scope = get_end_scope()

    return begin_scope + res + end_scope


def draw_pdf_sketch(sketch):
    """Draws a PDF sketch.

    Args:
        sketch: The PDF sketch object.

    Returns:
        str: The TikZ code for the image sketch.
    """
    begin_scope = get_begin_scope()
    options = get_line_style_options(sketch)
    # options += get_fill_style_options(sketch, frame=True)
    x, y = sketch.pos[:2]
    if sketch.angle != 0:
        angle = degrees(sketch.angle)
        options.append(f"rotate = {angle}")

    if sketch.scale != 1:
        scale = sketch.scale
        options.append(f"xscale = {scale}, yscale = {scale}")

    if sketch.anchor != Anchor.CENTER:
        options.append(f"anchor = {anchor_to_tikz(sketch.anchor)}")

    res = f"\\node[{', '.join(options)}]at({x}, {y}) {{\\includegraphics{{{sketch.file_path}}}}};\n"
    end_scope = get_end_scope()

    return begin_scope + res + end_scope


def draw_shape_sketch(sketch, ind=None, canvas=None):
    """Draws a shape sketch.

    Args:
        sketch: The shape sketch object.
        ind: Optional index for the shape sketch.

    Returns:
        str: The TikZ code for the shape sketch.
    """
    d_subtype_draw = {
        sg.Types.ARC_SKETCH: draw_arc_sketch,
        sg.Types.BEZIER_SKETCH: draw_bezier_sketch,
        sg.Types.CIRCLE_SKETCH: draw_circle_sketch,
        sg.Types.ELLIPSE_SKETCH: draw_ellipse_sketch,
    }
    if sketch.subtype == sg.Types.LINE_SKETCH:
        res = draw_line_sketch(sketch, canvas)
    elif sketch.subtype in d_subtype_draw:
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


def _line_limits(canvas):
    if canvas is None:
        return None
    limits = None
    if canvas.limits is not None:
        limits = tuple(canvas.limits)
    elif canvas._all_vertices:
        bbox = bounding_box(canvas._all_vertices)
        limits = (*bbox.southwest, *bbox.northeast)

    page = getattr(canvas, "active_page", None)
    sketches = getattr(page, "sketches", []) if page is not None else []
    for sketch in sketches:
        if (
            getattr(sketch, "subtype", None) == Types.HELPLINES_SKETCH
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


def draw_line_sketch(sketch, canvas=None):
    """Draws a line sketch.

    Args:
        sketch: The line sketch object.

    Returns:
        str: The TikZ code for the line sketch.
    """
    begin_scope = get_begin_scope()
    res = "\\draw"
    options = get_line_style_options(sketch)

    start = sketch.vertices[0]
    end = sketch.vertices[1]
    extent = getattr(sketch, "extent", getattr(sketch, "draw_type", Extent.SEGMENT))
    if not isinstance(extent, Extent) and extent is not None:
        extent = Extent(extent)
    if extent in [Extent.RAY, Extent.INFINITE]:
        limits = _line_limits(canvas)
        start, end = _clip_line_to_rect(start, end, limits, extent)

    options = ", ".join(options)
    res += f"[{options}]"
    res += f" {start[:2]} -- {end[:2]};\n"
    end_scope = get_end_scope()

    return begin_scope + res + end_scope


def draw_circle_sketch(sketch):
    """Draws a circle sketch.

    Args:
        sketch: The circle sketch object.

    Returns:
        str: The TikZ code for the circle sketch.
    """
    begin_scope = get_begin_scope()
    res = get_draw(sketch)
    if not res:
        return ""
    options = get_line_style_options(sketch)
    svg_gradient_options = _get_svg_gradient_shading_options(sketch)
    has_svg_gradient = bool(svg_gradient_options) and sketch.fill
    if not has_svg_gradient:
        fill_options = get_fill_style_options(sketch)
        options += fill_options
    else:
        options += svg_gradient_options
    if sketch.smooth:
        options += ["smooth"]
    options = ", ".join(options)
    if options:
        res += f"[{options}]"
    x, y = sketch.center[:2]
    res += f"({x}, {y}) circle ({sketch.radius});\n"
    end_scope = get_end_scope()

    return begin_scope + res + end_scope


def draw_rect_sketch(sketch):
    """Draws a rectangle sketch.

    Args:
        sketch: The rectangle sketch object.

    Returns:
        str: The TikZ code for the rectangle sketch.
    """
    begin_scope = get_begin_scope()
    res = get_draw(sketch)
    if not res:
        return ""
    options = get_line_style_options(sketch)
    svg_gradient_options = _get_svg_gradient_shading_options(sketch)
    has_svg_gradient = bool(svg_gradient_options) and sketch.fill
    if not has_svg_gradient:
        fill_options = get_fill_style_options(sketch)
        options += fill_options
    else:
        options += svg_gradient_options
    if sketch.smooth:
        options += ["smooth"]
    options = ", ".join(options)
    res += f"[{options}]"
    x, y = sketch.center[:2]
    width, height = sketch.width, sketch.height
    res += f"({x}, {y}) rectangle ({width}, {height});\n"
    end_scope = get_end_scope()

    return begin_scope + res + end_scope


def draw_ellipse_sketch(sketch):
    """Draws an ellipse sketch.

    Args:
        sketch: The ellipse sketch object.

    Returns:
        str: The TikZ code for the ellipse sketch.
    """
    begin_scope = get_begin_scope()
    res = get_draw(sketch)
    if not res:
        return ""
    options = get_line_style_options(sketch)
    svg_gradient_options = _get_svg_gradient_shading_options(sketch)
    has_svg_gradient = bool(svg_gradient_options) and sketch.fill
    if not has_svg_gradient:
        fill_options = get_fill_style_options(sketch)
        options += fill_options
    else:
        options += svg_gradient_options
    if sketch.smooth:
        options += ["smooth"]
    angle = degrees(sketch.angle)
    x, y = sketch.center[:2]
    if angle:
        options += [f"rotate around= {{{angle}:({x},{y})}}"]
    options = ", ".join(options)
    res += f"[{options}]"
    a = sketch.x_radius
    b = sketch.y_radius

    res += f"({x}, {y}) ellipse ({a} and {b});\n"
    end_scope = get_end_scope()

    return begin_scope + res + end_scope


def draw_arc_sketch(sketch):
    """Draws an arc sketch.

    Args:
        sketch: The arc sketch object.

    Returns:
        str: The TikZ code for the arc sketch.
    """
    res = get_draw(sketch)
    if not res:
        return ""
    if sketch.closed:
        options = ["smooth cycle"]
    else:
        options = ["smooth"]

    if sketch.back_style == BackStyle.PATTERN and sketch.fill and sketch.closed:
        options += get_pattern_options(sketch)
    if sketch.stroke:
        options += get_line_style_options(sketch)
    svg_gradient_options = _get_svg_gradient_shading_options(sketch)
    has_svg_gradient = bool(svg_gradient_options) and sketch.fill and sketch.closed
    if sketch.closed and sketch.fill and not has_svg_gradient:
        options += get_fill_style_options(sketch)

    if sketch.back_style == BackStyle.SHADING and sketch.fill and sketch.closed:
        options += get_shading_options(sketch)
    elif has_svg_gradient:
        options += svg_gradient_options
    options = ", ".join(options)
    if options:
        res += f"[{options}] plot[tension=.8] coordinates" + "{"
    vertices = [round_point(v) for v in sketch.vertices]
    n = len(vertices)
    str_lines = [f"{vertices[0]}"]
    for i, vertex in enumerate(vertices[1:]):
        if (i + 1) % 8 == 0:
            if i == n - 1:
                str_lines.append(f" {vertex} \n")
            else:
                str_lines.append(f"\n\t {vertex} ")
        else:
            str_lines.append(f" {vertex} ")
    if sketch.closed:
        str_lines.append(" cycle;\n")
    else:
        str_lines.append("};\n")
    if res:
        res += "".join(str_lines)
    else:
        res = "".join(str_lines)
    return res


def draw_bezier_sketch(sketch):
    """Draws a Bezier curve sketch.

    Args:
        sketch: The Bezier curve sketch object.

    Returns:
        str: The TikZ code for the Bezier curve sketch.
    """
    begin_scope = get_begin_scope()
    res = get_draw(sketch)
    if not res:
        return ""
    options = get_line_style_options(sketch)
    options = ", ".join(options)
    res += f"[{options}]"
    p1, cp1, cp2, p2 = sketch.control_points
    x1, y1 = p1[:2]
    x2, y2 = cp1[:2]
    x3, y3 = cp2[:2]
    x4, y4 = p2[:2]
    res += f" ({x1}, {y1}) .. controls ({x2}, {y2}) and ({x3}, {y3}) .. ({x4}, {y4});\n"
    end_scope = get_end_scope()

    return begin_scope + res + end_scope


def draw_line(line):
    """Tikz code for a line.

    Args:
        line: The line object.

    Returns:
        str: The TikZ code for the line.
    """
    p1 = line.start[:2]
    p2 = line.end[:2]
    options = []
    if line.line_width is not None:
        options.append(line.line_width)
    if line.color is not None:
        color = color2tikz(line.color)
        options.append(color)
    if line.dash_array is not None:
        options.append(line.dash_array)
    # options = [line.width, line.color, line.dash_array, line.cap, line.join]
    if line.line_width == 0:
        res = f"\\path[{', '.join(options)}] {p1} -- {p2};\n"
    else:
        res = f"\\draw[{', '.join(options)}] {p1} -- {p2};\n"

    return res


def is_stroked(shape: Shape) -> bool:
    """Returns True if the shape is stroked.

    Args:
        shape (Shape): The shape object.

    Returns:
        bool: True if the shape is stroked, False otherwise.
    """
    return shape.stroke and shape.line_color is not None and shape.line_width > 0


d_sketch_draw = {
    sg.Types.ARC: draw_arc_sketch,
    sg.Types.BATCH: draw_batch_sketch,
    sg.Types.CIRCLE: draw_circle_sketch,
    sg.Types.ELLIPSE: draw_shape_sketch,
    sg.Types.IMAGE_SKETCH: draw_image_sketch,
    sg.Types.LACESKETCH: draw_lace_sketch,
    sg.Types.LINE: draw_line_sketch,
    sg.Types.SHAPE: draw_shape_sketch,
    sg.Types.TAG_SKETCH: draw_tag_sketch,
}

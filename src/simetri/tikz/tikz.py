"""TikZ exporter. Draws shapes using the TikZ package for LaTeX.
Sketch objects are converted to TikZ code."""

from __future__ import annotations

from math import atan2, degrees
from types import SimpleNamespace

import numpy as np

from ..canvas.pre_render import (
    collect_tikz_preamble_requirements_for_sketch,
    set_styles,
    style_properties,
)
from ..geometry.geometry import homogenize
from ..graphics.all_enums import (
    Anchor,
    BackStyle,
    Extent,
    MarkerType,
    ShadeType,
    TexLoc,
    Types,
)
from ..graphics.bbox import bounding_box
from ..graphics.points import Points
from ..graphics.shape import Shape
from ..settings.settings import defaults, issue_warning
from . import tikz_sketch as tikz_sketch_module
from .tikz_mask import *
from .tikz_mask import _effective_alpha_from_stop, _pgf_gray
from .tikz_sketch import *
from .tikz_sketch import _canvas_mask_scope_sketch
from .tikz_utils import *
from .tikz_utils import _get_gradient_shading_options

NumberOrTex = int | float | str

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


def scope_code_required(canvas: Canvas) -> bool:
    """Check if canvas-level mask scope sketch exists."""
    return _canvas_mask_scope_sketch(canvas) is not None


def get_back_grid_code(grid: Grid, canvas: Canvas) -> str:
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
    back_color = color_to_tikz(grid.back_color, "grid_back_color")
    line_color = color_to_tikz(grid.line_color, "grid_line_color")
    step = grid.spacing
    lines = ["\\begin{scope}[on background layer]\n"]
    lines.append(
        f"\\fill[color={back_color}] (current bounding box.south west) "
    )
    lines.append("rectangle (current bounding box.north east);\n")
    options = []
    if grid.line_dash_array is not None:
        options.append(
            f"dashed, dash pattern={get_dash_pattern(grid.line_dash_array)}"
        )
    if grid.line_width is not None:
        options.append(f"line width={grid.line_width}")
    if options:
        options = ",".join(options)
        lines.append(f"\\draw[draw={line_color}, step={step}, {options}]")
    else:
        lines.append(f"\\draw[draw={line_color},step={step}]")
    lines.append("(current bounding box.south west)")
    lines.append(" grid (current bounding box.north east);\n")
    lines.append("\\end{scope}\n")

    return "".join(lines)


def get_limits_code(canvas: Canvas) -> str:
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
    if canvas.limits is not None:
        vertices = points
    else:
        vertices = homogenize(points) @ canvas.xform_matrix
    coords = " ".join([f"({v[0]}, {v[1]})" for v in vertices])

    return (
        f"\\path[use as bounding box] plot[] coordinates {{{coords}}};\n"
        f"\\clip plot[] coordinates {{{coords}}};\n"
    )


def get_back_code(canvas: Canvas) -> str:
    """Get the background code for the canvas.

    Args:
        canvas ("Canvas"): The canvas object.

    Returns:
        str: The background code.
    """
    back_color = color_to_tikz(canvas.back_color, "back_color")
    return f"\\pagecolor{back_color}\n"


def get_tex_code(canvas: Canvas) -> str:
    """Convert the sketches in the Canvas to TikZ code.

    Args:
        canvas ("Canvas"): The canvas object.

    Returns:
        str: The TikZ code.
    """

    render_style_ids = {}

    styleable_subtypes = [
        Types.ARC_SKETCH,
        Types.BEZIER_SKETCH,
        Types.CIRCLE_SKETCH,
        Types.ELLIPSE_SKETCH,
        Types.LINE_SKETCH,
        Types.PATH_SKETCH,
        Types.PATTERN_SKETCH,
        Types.RECTANGLE_SKETCH,
        Types.SHAPE_SKETCH,
    ]
    tikz_libraries = []
    tikz_packages = ["tikz", "pgf"]

    def render_sketches(sketches, ind):
        code = []
        for sketch in sketches:
            sketch_code, ind = get_sketch_code(
                sketch, canvas, ind, style_properties
            )
            code.append(sketch_code)
        return "".join(code), ind

    def get_sketch_code(sketch, canvas, ind, suppressed_style_keys):
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
            code = draw_pattern_sketch(sketch, exceptions=suppressed_style_keys)
        elif sketch.subtype == Types.TEX_SKETCH:
            if sketch.location == TexLoc.NONE:
                code = sketch.code
            else:
                code = ""
        elif sketch.subtype == Types.LATEX_SKETCH:
            code = draw_latex_sketch(sketch)
        elif sketch.subtype == Types.MASK_SKETCH:
            code = ""
        elif sketch.subtype == Types.CLIPPED_SKETCH:
            clip_code = get_clip_code(sketch.clipper)
            clipped_code = ["\\begin{scope}\n", clip_code]
            child_sketches = []
            for sketch_list in sketch.sketches:
                child_sketches.extend(sketch_list)
            child_code, ind = render_sketches(child_sketches, ind)
            clipped_code.append(child_code)
            clipped_code.append("\\end{scope}\n")
            code = "".join(clipped_code)
        elif sketch.subtype == Types.MASKED_SKETCH:
            mask_start, mask_end = _mask_scope_parts(sketch)
            masked_code = [mask_start]
            child_sketches = []
            for sketch_list in sketch.sketches:
                child_sketches.extend(sketch_list)
            child_code, ind = render_sketches(child_sketches, ind)
            masked_code.append(child_code)
            masked_code.append(mask_end)
            code = "".join(masked_code)
        elif sketch.subtype == Types.COMPOSITE_SKETCH:
            child_code, ind = render_sketches(sketch.sketches, ind)
            code = child_code
        else:
            if (
                hasattr(sketch, "draw_markers")
                and sketch.draw_markers
                and sketch.marker_type == MarkerType.INDICES
            ):
                code = draw_shape_sketch(sketch, ind, canvas)
                ind += 1
            else:
                code = draw_shape_sketch(
                    sketch,
                    canvas=canvas,
                    exceptions=suppressed_style_keys,
                )

        return code, ind

    pages = canvas.pages
    has_sketches = any(page.sketches for page in pages)

    if not has_sketches:
        issue_warning(
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
            back_color = (
                f"\\pagecolor{color_to_tikz(page.back_color, 'back_color')}"
            )
            if i == 0:
                if page.back_color:
                    code = [back_color]
                else:
                    code = []
            else:
                code.append(defaults["end_tikz"])
                code.append("\\newpage")
                code.append(defaults["begin_tikz"])
                if canvas.limits is not None or canvas.inset != 0:
                    code.append(get_limits_code(canvas))

            sketches_to_populate = list(sketches)
            style_sketches = []
            while sketches_to_populate:
                sketch = sketches_to_populate.pop()
                collect_tikz_preamble_requirements_for_sketch(
                    sketch, tikz_libraries, tikz_packages
                )
                if sketch.subtype in [
                    Types.CLIPPED_SKETCH,
                    Types.MASKED_SKETCH,
                ]:
                    for sketch_list in sketch.sketches:
                        sketches_to_populate.extend(sketch_list)
                elif sketch.subtype == Types.COMPOSITE_SKETCH:
                    sketches_to_populate.extend(sketch.sketches)
                elif sketch.subtype == Types.HELPLINES_SKETCH:
                    sketch.populate(canvas)
                elif sketch.subtype == Types.LINE_SKETCH:
                    if hasattr(sketch, "populate"):
                        sketch.populate(canvas)
                if sketch.subtype in styleable_subtypes:
                    style_sketches.append(sketch)

            d_styles, d_sketch_style = set_styles(style_sketches)
            render_style_ids = {}
            tikz_sketch_module.set_active_tikz_style_ids(render_style_ids)
            style_sketch_dict = {}
            for sketch in style_sketches:
                if sketch.id not in d_sketch_style:
                    continue
                style_id = d_sketch_style[sketch.id]
                if style_id not in style_sketch_dict:
                    style_sketch_dict[style_id] = [sketch]
                else:
                    style_sketch_dict[style_id].append(sketch)

            if d_styles:
                style_lines = ["\\tikzset{"]
                has_style_lines = False
                for style_id in d_styles:
                    style_sketches_for_id = style_sketch_dict[style_id]
                    sketch = style_sketches_for_id[0]
                    options = []
                    if sketch.stroke:
                        options += get_line_style_options(sketch)
                    fill_style_sketch = None
                    for candidate_sketch in style_sketches_for_id:
                        if candidate_sketch.fill and (
                            candidate_sketch.subtype == Types.PATH_SKETCH
                            or (
                                "closed" in candidate_sketch.__dict__
                                and candidate_sketch.closed
                            )
                        ):
                            fill_style_sketch = candidate_sketch
                            break
                    if fill_style_sketch is not None:
                        options += get_fill_style_options(fill_style_sketch)
                    if not options:
                        continue
                    for style_sketch in style_sketches_for_id:
                        render_style_ids[style_sketch.id] = style_id
                    style_lines.append(
                        f"{style_id}/.style={{{', '.join(options)}}},"
                    )
                    has_style_lines = True
                style_lines.append("}")
                if has_style_lines:
                    code.append("\n".join(style_lines))

            ind = 0
            page_code, ind = render_sketches(sketches, ind)
            tikz_sketch_module.set_active_tikz_style_ids({})
            code.append(page_code)

        code = "\n".join(code)
    else:
        raise ValueError("No pages found in the canvas.")
    canvas.tex.tikz_libraries = tikz_libraries
    canvas.tex.packages = tikz_packages
    return canvas.tex.tex_code(canvas, code)


class Grid(Shape):
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
        self.primary_points = Points([p1, p2])
        self.closed = False
        self.fill = False
        self.stroke = True
        self._b_box = None
        super().__init__(
            [p1, p2], xform_matrix=None, subtype=Types.GRID, **kwargs
        )


def _build_fading_code(fade_id, stops, x1, y1, x2, y2):
    parsed_stops = [_effective_alpha_from_stop(stop) for stop in stops]
    parsed_stops.sort(key=lambda item: item[0])
    if not parsed_stops:
        parsed_stops = [(0.0, 1.0), (1.0, 1.0)]

    shade_id = f"{fade_id}Shade"
    color_stops = []
    for offset, alpha in parsed_stops:
        offset = max(0.0, min(1.0, float(offset)))
        position = round(offset * 100)
        transparency = round((1.0 - float(alpha)) * 100)
        color_stops.append(f"color({position}bp)=({_pgf_gray(transparency)})")

    if parsed_stops[0][0] > 0.0:
        first_transparency = round((1.0 - float(parsed_stops[0][1])) * 100)
        color_stops.insert(0, f"color(0bp)=({_pgf_gray(first_transparency)})")
    if parsed_stops[-1][0] < 1.0:
        last_transparency = round((1.0 - float(parsed_stops[-1][1])) * 100)
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
        elif sketch.subtype == Types.MASK_SKETCH:
            mask = mask_data
            clip = sketch.clip
            mask_opacity = sketch.mask_opacity
            mask_stops = sketch.mask_stops
            mask_axis = sketch.mask_axis
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
        bbox_x = mask_bbox.southwest[0]
        bbox_y = mask_bbox.southwest[1]
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

    if mask_opacity not in (None, 1):
        return (
            f"\\begin{{scope}}[opacity={mask_opacity}]\n{clip_code}",
            "\\end{scope}\n",
        )

    return "", ""


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
        canvas_mask = canvas_mask_scope.mask
        canvas_clip = canvas_mask_scope.clip
        canvas_mask_opacity = canvas_mask_scope.mask_opacity
        canvas_mask_stops = canvas_mask_scope.mask_stops
        canvas_mask_fade_id = f"simetriCanvasMaskFade{canvas_mask_scope.id}"
    else:
        canvas_mask = None
        canvas_clip = False
        canvas_mask_opacity = 1.0
        canvas_mask_stops = None
        canvas_mask_fade_id = None

    if canvas_mask_stops is not None and canvas_mask is not None:
        fade_name = (
            canvas_mask_fade_id
            or f"simetriCanvasMaskFade{id(canvas_mask_scope)}"
        )
        start_code, _ = _mask_scope_parts(canvas_mask_scope, fade_name)
        return start_code

    if canvas_mask_opacity not in (None, 1) and canvas_mask is not None:
        start_code, _ = _mask_scope_parts(canvas_mask_scope)
        return start_code

    if canvas_clip and canvas_mask:
        start_code, _ = _mask_scope_parts(canvas_mask_scope)
        return start_code

    if option_list:
        return f"\\begin{{scope}}[{','.join(option_list)}]\n"
    return "\\begin{scope}\n"


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


def _shape_bbox(sketch):
    if hasattr(sketch, "vertices") and getattr(sketch, "vertices", None):
        xs = [v[0] for v in sketch.vertices]
        ys = [v[1] for v in sketch.vertices]
        return min(xs), min(ys), max(xs), max(ys)

    if hasattr(sketch, "center") and hasattr(sketch, "radius"):
        cx, cy = sketch.center[:2]
        r = sketch.radius
        return cx - r, cy - r, cx + r, cy + r

    if (
        hasattr(sketch, "center")
        and hasattr(sketch, "width")
        and hasattr(sketch, "height")
    ):
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

    t_values = [
        ((px - float(x1)) * vx + (py - float(y1)) * vy) / denom
        for px, py in corners
    ]
    return min(t_values), max(t_values)


# Tex class went to tex.py


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

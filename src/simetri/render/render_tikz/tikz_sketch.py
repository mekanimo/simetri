"""Serialize individual sketch types to TikZ/PGF markup."""

from dataclasses import dataclass
from math import ceil, degrees

import numpy as np

import simetri.graphics as sg

from ..geom.points.point_utils import close_points2
from ..base.all_enums import (
    Align,
    Anchor,
    BackStyle,
    Extent,
    FontFamily,
    FontSize,
    FrameShape,
    MarkerType,
    TexLoc,
    Types,
    get_enum_value,
)
from ..geom.points.point_utils import round_point
from ..helpers.illustration import (
    label_halo_color,
    label_halo_stroke_width,
    prepare_shape_index_labels,
    prepare_shape_vertex_coord_labels,
    sketch_label_font_color,
    sketch_label_font_size_pt,
)
from ..helpers.utilities import detokenize
from ..config.settings import defaults
from .tikz_common import (
    _clip_line_to_rect,
    _line_limits,
    _mask_scope_parts,
    anchor_to_tikz,
    get_begin_scope,
    get_draw,
    get_end_scope,
)
from .tikz_utils import (
    _get_gradient_shading_options,
    color_to_tikz,
    get_dash_pattern,
    get_fill_style_options,
    get_line_style_options,
    get_marker_options,
    get_pattern_options,
    get_shading_options,
    sg_to_tikz,
)


@dataclass
class TexSketch:
    """TexSketch is a dataclass for inserting code into the tex file.

    Attributes:
        code (str, optional): The code to be inserted. Defaults to None.
        location (TexLoc, optional): The location of the code. Defaults to TexLoc.NONE.

    Returns:
        None
    """

    code: str = None
    location: TexLoc = TexLoc.NONE

    def __post_init__(self):
        """Initialize the TexSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.TEX_SKETCH


"""TikZ Sketch"""


_active_tikz_style_ids = {}


def set_active_tikz_style_ids(style_ids):
    """Set the active sketch-id to TikZ style-id mapping.

    Args:
        style_ids: Mapping of sketch id to TikZ style id.
    """
    global _active_tikz_style_ids
    _active_tikz_style_ids = style_ids


def get_active_tikz_style_id(sketch):
    """Return the TikZ style id for ``sketch``, if any.

    Args:
        sketch: Sketch whose style id is requested.

    Returns:
        The active style id, or None.
    """
    if sketch.id in _active_tikz_style_ids:
        return _active_tikz_style_ids[sketch.id]
    return None


def _canvas_mask_scope_sketch(canvas):
    page = canvas.active_page
    sketches = page.sketches
    for sketch in reversed(sketches):
        if sketch.subtype == Types.MASK_SKETCH:
            return sketch
    return None


def draw_helplines_sketch(sketch):
    """Draw deferred help lines (grid + optional coordinate system) for TikZ output."""
    x, y = sketch.pos[:2]
    width = sketch.width
    height = sketch.height
    spacing = sketch.spacing
    cs_size = sketch.cs_size
    kwargs = dict(sketch.kwargs)

    if spacing in (None, 0):
        spacing = defaults["help_lines_spacing"]

    # Match draw.grid defaults
    grid_line_width = kwargs.get("line_width", defaults["grid_line_width"])
    grid_line_color = kwargs.get("line_color", defaults["grid_line_color"])
    grid_line_dash_array = kwargs.get(
        "line_dash_array", defaults["grid_line_dash_array"]
    )
    line_alpha = kwargs.get(
        "line_alpha", kwargs.get("alpha", defaults["line_alpha"])
    )

    def _line_options(
        line_color, line_width, line_dash_array=None, draw_opacity=None
    ):
        options = [
            f"draw={color_to_tikz(line_color)}",
            f"line width={line_width}",
        ]
        if line_dash_array is not None:
            options.append(f"dash pattern={get_dash_pattern(line_dash_array)}")
        if draw_opacity not in (None, 1):
            options.append(f"draw opacity={draw_opacity}")
        return ", ".join(options)

    lines = []

    # Grid lines (horizontal + vertical)
    n_h = int(height / spacing)
    n_v = int(width / spacing)
    grid_opts = _line_options(
        grid_line_color, grid_line_width, grid_line_dash_array, line_alpha
    )

    for i in range(n_h + 1):
        yi = y + i * spacing
        lines.append(f"\\draw[{grid_opts}] ({x}, {yi}) -- ({x + width}, {yi});")

    for i in range(n_v + 1):
        xi = x + i * spacing
        lines.append(
            f"\\draw[{grid_opts}] ({xi}, {y}) -- ({xi}, {y + height});"
        )

    # Coordinate system axes + origin marker
    if cs_size and cs_size > 0:
        if "colors" in kwargs:
            x_color, y_color = kwargs["colors"]
        else:
            x_color = defaults["CS_x_color"]
            y_color = defaults["CS_y_color"]

        cs_line_width = kwargs.get("line_width", defaults["CS_line_width"])
        cs_dash = kwargs.get("line_dash_array", None)
        cs_alpha = kwargs.get(
            "line_alpha", kwargs.get("alpha", defaults["line_alpha"])
        )

        x_axis_opts = _line_options(x_color, cs_line_width, cs_dash, cs_alpha)
        y_axis_opts = _line_options(y_color, cs_line_width, cs_dash, cs_alpha)

        lines.append(f"\\draw[{x_axis_opts}] (0, 0) -- ({cs_size}, 0);")
        lines.append(f"\\draw[{y_axis_opts}] (0, 0) -- (0, {cs_size});")

        origin_color = kwargs.get("line_color", defaults["CS_origin_color"])
        origin_size = defaults["CS_origin_size"]
        lines.append(
            f"\\filldraw[draw={color_to_tikz(origin_color)}, fill={color_to_tikz(origin_color)}] "
            f"(0, 0) circle ({origin_size});"
        )

    return "\n".join(lines) + "\n"


def draw_bbox_sketch(sketch):
    """Converts a BBoxSketch to TikZ code.

    Args:
        sketch: The BBoxSketch object.
        canvas: The canvas object.

    Returns:
        str: The TikZ code for the BBoxSketch.
    """
    attrib_map = {
        "line_color": "draw",
        "line_width": "line width",
        "line_dash_array": "dash pattern",
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
                raise ValueError(
                    f"Font family {sketch.font_family} not supported."
                )
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
    if sketch.draw_frame and sketch.line_width > 0:
        options += "draw"
        if sketch.stroke:
            if sketch.frame_shape != FrameShape.RECTANGLE:
                options += f", {sketch.frame_shape}, "
            line_style_options = get_line_style_options(sketch)
            if line_style_options:
                options += ", " + ", ".join(line_style_options)
            if sketch.frame_inner_sep:
                options += f", inner sep={sketch.frame_inner_sep}"
            else:
                options += ", inner sep=0pt"
            if sketch.minimum_width:
                options += f", minimum width={sketch.minimum_width}"
            if sketch.smooth and sketch.frame_shape not in [
                FrameShape.CIRCLE,
                FrameShape.ELLIPSE,
            ]:
                options += ", smooth"
    else:
        options = "inner sep=0pt"

    if sketch.fill and sketch.back_color:
        options += f", fill={color_to_tikz(sketch.frame_back_color, 'frame_back_color')}"
    effective_anchor = (
        sketch.anchor if sketch.anchor is not None else defaults["anchor"]
    )
    if (
        sketch.align in (Align.LEFT, Align.FLUSH_LEFT)
        and effective_anchor == defaults["anchor"]
    ):
        options += f", anchor={anchor_to_tikz(Anchor.WEST)}"
    elif (
        sketch.align in (Align.RIGHT, Align.FLUSH_RIGHT)
        and effective_anchor == defaults["anchor"]
    ):
        options += f", anchor={anchor_to_tikz(Anchor.EAST)}"
    elif sketch.anchor:
        options += f", anchor={anchor_to_tikz(sketch.anchor)}"
    if sketch.back_style == BackStyle.SHADING and sketch.fill:
        shading_options = get_shading_options(sketch)[0]
        options += ", " + shading_options
    if sketch.back_style == BackStyle.PATTERN and sketch.fill:
        pattern_options = get_pattern_options(sketch)[0]
        options += ", " + pattern_options
    if sketch.align.value:
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

    if (
        sketch.font_color is not None
        and sketch.font_color != defaults["font_color"]
    ):
        options += f", text={color_to_tikz(sketch.font_color)}"
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
    if (
        sketch.font_color is not None
        and sketch.font_color != defaults["font_color"]
    ):
        options.append(f"text={color_to_tikz(sketch.font_color)}")

    font_size = sketch.font_size or defaults["font_size"]
    baseline_skip = ceil(font_size * 1.2)
    tex_formula = rf"{{\fontsize{{{font_size}}}{{{baseline_skip}}}\selectfont ${formula}$}}"
    option_str = f"[{', '.join(options)}]" if options else ""
    return f"\\node{option_str} at ({x}, {y}) {tex_formula};\n"


def _label_font_tikz(sketch, label_kind: str) -> str:
    """TikZ node font option for index or vertex-coordinate labels."""
    family = defaults["indices_font_family"]
    pt = sketch_label_font_size_pt(sketch, label_kind)
    baseline = ceil(pt * 1.2)
    return f"font=\\{family}\\fontsize{{{pt}}}{{{baseline}}}\\selectfont"


def _tikz_halo_label_lines(x, y, text, label_kind: str, sketch) -> list[str]:
    """TikZ node with a contour halo for label readability."""
    font = _label_font_tikz(sketch, label_kind)
    text_color = color_to_tikz(sketch_label_font_color(sketch, label_kind))
    halo_color = color_to_tikz(label_halo_color())
    pt = sketch_label_font_size_pt(sketch, label_kind)
    contour_len = label_halo_stroke_width(pt)
    label_text = str(text)
    content = (
        f"\\contourlength{{{contour_len}pt}}"
        f"\\contour{halo_color}{{\\textcolor{text_color}{{{label_text}}}}}"
    )
    return [
        f"\\node[{font}, inner sep=0pt] at ({x}, {y}) {{{content}}};\n",
    ]


def draw_shape_sketch_with_indices(sketch, index=0, exceptions=None):
    """Draw a shape sketch with optional vertex indices and coordinate labels.

    When ``sketch.indices`` is truthy, index numbers are drawn at offset
    label positions. When ``sketch.show_vertex_coords`` is True, ``(x, y)``
    coordinate labels are drawn similarly.

    Args:
        sketch: The shape sketch object.
        index: The index.

    Returns:
        str: The TikZ code for the shape sketch with vertex labels.
    """
    begin_scope = get_begin_scope(index)
    body = get_draw(sketch)
    if body:
        options = []
        style_id = get_active_tikz_style_id(sketch)
        if style_id is not None:
            options.append(style_id)
        options += get_line_style_options(sketch, exceptions=exceptions)
        if sketch.fill and sketch.closed:
            options += get_fill_style_options(sketch, exceptions=exceptions)
        if sketch.smooth:
            if sketch.closed:
                options += ["smooth cycle"]
            else:
                options += ["smooth"]
        options = ", ".join(options)
        body += f"[{options}]"
    else:
        body = ""
    vertex_coords = sketch.vertices

    vertex_font_size = sketch_label_font_size_pt(sketch, "vertex")
    index_font_size = sketch_label_font_size_pt(sketch, "index")

    vertices = [str(x) for x in vertex_coords]
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

    index_draw = prepare_shape_index_labels(sketch)
    if index_draw is not None:
        index_positions, index_labels = index_draw
        for (lx, ly), label in zip(index_positions, index_labels):
            str_lines.extend(
                _tikz_halo_label_lines(lx, ly, label, "index", sketch)
            )

    vertex_draw = prepare_shape_vertex_coord_labels(sketch)
    if vertex_draw is not None:
        coord_positions, coord_labels = vertex_draw
        for (lx, ly), text in zip(coord_positions, coord_labels):
            str_lines.extend(
                _tikz_halo_label_lines(lx, ly, text, "vertex", sketch)
            )

    end_scope = get_end_scope()
    if not begin_scope:
        res = body + "".join(str_lines)
    else:
        res = begin_scope + body + "".join(str_lines) + end_scope

    return res


def draw_shape_sketch_with_markers(sketch, exceptions=None):
    """Draws a shape sketch with markers.

    Args:
        sketch: The shape sketch object.

    Returns:
        str: The TikZ code for the shape sketch with markers.
    """
    # begin_scope = get_begin_scope()
    body = get_draw(sketch)
    if body:
        options = []
        style_id = get_active_tikz_style_id(sketch)
        if style_id is not None:
            options.append(style_id)
        options += get_line_style_options(sketch, exceptions=exceptions)
        if sketch.fill and sketch.closed:
            options += get_fill_style_options(sketch, exceptions=exceptions)
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

    if sketch.closed and not close_points2(
        sketch.vertices[0], sketch.vertices[-1]
    ):
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
        marker_shape = sketch.marker_shape
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
            body += "% Custom marker shape at vertices (not yet implemented in TikZ)\n"
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
        body += f" plot[mark = {marker}, {markers_only}] coordinates {{{coordinates}}};\n"
    else:
        body += f" plot[tension=.5] coordinates {{{coordinates}}};\n"

    return body


def draw_pattern_sketch(sketch, exceptions=None):
    """Draws a pattern sketch.

    Args:
        sketch: The pattern sketch object.

    Returns:
        str: The TikZ code for the pattern sketch.
    """
    options = []
    style_id = get_active_tikz_style_id(sketch)
    if style_id is not None:
        options.append(style_id)

    if sketch.back_style == BackStyle.PATTERN and sketch.fill and sketch.closed:
        options += get_pattern_options(sketch)
    if sketch.stroke:
        options += get_line_style_options(sketch, exceptions=exceptions)
    if sketch.closed and sketch.fill:
        options += get_fill_style_options(sketch, exceptions=exceptions)
    if sketch.smooth:
        options += ["smooth"]
    if sketch.back_style == BackStyle.SHADING and sketch.fill and sketch.closed:
        options += get_shading_options(sketch)
    option_text = f"[{', '.join(options)}]" if options else ""

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
        shapes.append(draw + option_text + "".join(str_lines))

    return "\n".join(shapes)


def draw_sketch(sketch, exceptions=None):
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
    style_id = get_active_tikz_style_id(sketch)
    if style_id is not None:
        options.append(style_id)
    gradient_options = _get_gradient_shading_options(sketch)
    has_gradient = bool(gradient_options) and sketch.fill and sketch.closed

    if sketch.back_style == BackStyle.PATTERN and sketch.fill and sketch.closed:
        options += get_pattern_options(sketch)
    if sketch.stroke:
        options += get_line_style_options(sketch, exceptions=exceptions)
    if sketch.closed and sketch.fill and not has_gradient:
        options += get_fill_style_options(sketch, exceptions=exceptions)
    if sketch.smooth:
        options += ["smooth"]
    if sketch.back_style == BackStyle.SHADING and sketch.fill and sketch.closed:
        options += get_shading_options(sketch)
    elif has_gradient:
        options += gradient_options
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


def draw_image_sketch(sketch, exceptions=None):
    """Draws an image sketch.

    Args:
        sketch: The image sketch object.

    Returns:
        str: The TikZ code for the image sketch.
    """
    begin_scope = get_begin_scope()
    options = get_line_style_options(sketch, exceptions=exceptions)
    options += get_fill_style_options(sketch, exceptions=exceptions, frame=True)
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
    if begin_scope:
        return begin_scope + res + end_scope
    return res


def draw_pdf_sketch(sketch, exceptions=None):
    """Draws a PDF sketch.

    Args:
        sketch: The PDF sketch object.

    Returns:
        str: The TikZ code for the image sketch.
    """
    begin_scope = get_begin_scope()
    options = get_line_style_options(sketch, exceptions=exceptions)
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
    if begin_scope:
        return begin_scope + res + end_scope
    return res


def draw_shape_sketch(sketch, ind=None, canvas=None, exceptions=None):
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
        sg.Types.PATH_SKETCH: draw_path_sketch,
    }
    if sketch.subtype == sg.Types.LINE_SKETCH:
        res = draw_line_sketch(sketch, canvas, exceptions=exceptions)
    elif sketch.subtype in d_subtype_draw:
        res = d_subtype_draw[sketch.subtype](sketch, exceptions=exceptions)
    elif (
        (
            hasattr(sketch, "draw_markers")
            and sketch.draw_markers
            and sketch.marker_type == MarkerType.INDICES
        )
        or (hasattr(sketch, "indices") and sketch.indices)
        or (hasattr(sketch, "show_vertex_coords") and sketch.show_vertex_coords)
    ):
        res = draw_shape_sketch_with_indices(sketch, ind, exceptions=exceptions)
    elif (
        hasattr(sketch, "draw_markers")
        and sketch.draw_markers
        or (hasattr(sketch, "smooth") and sketch.smooth)
    ):
        res = draw_shape_sketch_with_markers(sketch, exceptions=exceptions)
    else:
        res = draw_sketch(sketch, exceptions=exceptions)

    mask_start, mask_end = _mask_scope_parts(sketch)
    if mask_start or mask_end:
        res = mask_start + res + mask_end

    return res


def draw_path_sketch(sketch, exceptions=None):
    """Draw a path sketch using PGF's SVG path parser."""
    res = get_draw(sketch)
    if not res:
        return ""
    options = []
    style_id = get_active_tikz_style_id(sketch)
    if style_id is not None:
        options.append(style_id)
    if sketch.stroke:
        options.extend(get_line_style_options(sketch, exceptions=exceptions))
    if sketch.fill:
        options.extend(get_fill_style_options(sketch, exceptions=exceptions))
    if sketch.back_style == BackStyle.SHADING and sketch.fill and sketch.closed:
        options.extend(get_shading_options(sketch))
    if sketch.back_style == BackStyle.PATTERN and sketch.fill and sketch.closed:
        options.extend(get_pattern_options(sketch))
    option_text = f"[{', '.join(options)}]" if options else ""
    return f"{res}{option_text} svg {{{sketch.path_data}}};\n"


def draw_line_sketch(sketch, canvas=None, exceptions=None):
    """Draws a line sketch.

    Args:
        sketch: The line sketch object.

    Returns:
        str: The TikZ code for the line sketch.
    """
    res = "\\draw"
    options = []
    style_id = get_active_tikz_style_id(sketch)
    if style_id is not None:
        options.append(style_id)
    options += get_line_style_options(sketch, exceptions=exceptions)

    start = sketch.vertices[0]
    end = sketch.vertices[1]
    extent = sketch.extent
    if not isinstance(extent, Extent) and extent is not None:
        extent = Extent(extent)
    if extent in (Extent.RAY, Extent.INFINITE):
        limits = _line_limits(canvas)
        start, end = _clip_line_to_rect(start, end, limits, extent)

    options = ", ".join(options)
    if options:
        res += f"[{options}]"
    res += f" {start[:2]} -- {end[:2]};\n"
    return res


def draw_circle_sketch(sketch, exceptions=None):
    """Draws a circle sketch.

    Args:
        sketch: The circle sketch object.

    Returns:
        str: The TikZ code for the circle sketch.
    """
    res = get_draw(sketch)
    if not res:
        return ""
    options = []
    style_id = get_active_tikz_style_id(sketch)
    if style_id is not None:
        options.append(style_id)
    options += get_line_style_options(sketch, exceptions=exceptions)
    gradient_options = _get_gradient_shading_options(sketch)
    has_gradient = bool(gradient_options) and sketch.fill
    if not has_gradient:
        fill_options = get_fill_style_options(sketch, exceptions=exceptions)
        options += fill_options
    else:
        options += gradient_options
    if sketch.smooth:
        options += ["smooth"]
    options = ", ".join(options)
    if options:
        res += f"[{options}]"
    x, y = sketch.center[:2]
    res += f"({x}, {y}) circle ({sketch.radius});\n"
    return res


def draw_rect_sketch(sketch, exceptions=None):
    """Draws a rectangle sketch.

    Args:
        sketch: The rectangle sketch object.

    Returns:
        str: The TikZ code for the rectangle sketch.
    """
    res = get_draw(sketch)
    if not res:
        return ""
    options = []
    style_id = get_active_tikz_style_id(sketch)
    if style_id is not None:
        options.append(style_id)
    options += get_line_style_options(sketch, exceptions=exceptions)
    gradient_options = _get_gradient_shading_options(sketch)
    has_gradient = bool(gradient_options) and sketch.fill
    if not has_gradient:
        fill_options = get_fill_style_options(sketch, exceptions=exceptions)
        options += fill_options
    else:
        options += gradient_options
    if sketch.smooth:
        options += ["smooth"]
    options = ", ".join(options)
    if options:
        res += f"[{options}]"
    x, y = sketch.center[:2]
    width, height = sketch.width, sketch.height
    res += f"({x}, {y}) rectangle ({width}, {height});\n"
    return res


def draw_ellipse_sketch(sketch, exceptions=None):
    """Draws an ellipse sketch.

    Args:
        sketch: The ellipse sketch object.

    Returns:
        str: The TikZ code for the ellipse sketch.
    """
    res = get_draw(sketch)
    if not res:
        return ""
    options = []
    style_id = get_active_tikz_style_id(sketch)
    if style_id is not None:
        options.append(style_id)
    options += get_line_style_options(sketch, exceptions=exceptions)
    gradient_options = _get_gradient_shading_options(sketch)
    has_gradient = bool(gradient_options) and sketch.fill
    if not has_gradient:
        fill_options = get_fill_style_options(sketch, exceptions=exceptions)
        options += fill_options
    else:
        options += gradient_options
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
    return res


def draw_arc_sketch(sketch, exceptions=None):
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
    style_id = get_active_tikz_style_id(sketch)
    if style_id is not None:
        options.insert(0, style_id)

    if sketch.back_style == BackStyle.PATTERN and sketch.fill and sketch.closed:
        options += get_pattern_options(sketch)
    if sketch.stroke:
        options += get_line_style_options(sketch, exceptions=exceptions)
    gradient_options = _get_gradient_shading_options(sketch)
    has_gradient = bool(gradient_options) and sketch.fill and sketch.closed
    if sketch.closed and sketch.fill and not has_gradient:
        options += get_fill_style_options(sketch, exceptions=exceptions)

    if sketch.back_style == BackStyle.SHADING and sketch.fill and sketch.closed:
        options += get_shading_options(sketch)
    elif has_gradient:
        options += gradient_options
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


def draw_bezier_sketch(sketch, exceptions=None):
    """Draws a Bezier curve sketch.

    Args:
        sketch: The Bezier curve sketch object.

    Returns:
        str: The TikZ code for the Bezier curve sketch.
    """
    res = get_draw(sketch)
    if not res:
        return ""
    options = []
    style_id = get_active_tikz_style_id(sketch)
    if style_id is not None:
        options.append(style_id)
    options += get_line_style_options(sketch, exceptions=exceptions)
    options = ", ".join(options)
    if options:
        res += f"[{options}]"
    p1, cp1, cp2, p2 = sketch.control_points
    x1, y1 = p1[:2]
    x2, y2 = cp1[:2]
    x3, y3 = cp2[:2]
    x4, y4 = p2[:2]
    res += f" ({x1}, {y1}) .. controls ({x2}, {y2}) and ({x3}, {y3}) .. ({x4}, {y4});\n"
    return res


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
        color = color_to_tikz(line.color)
        options.append(f"draw={color}")
    if line.dash_array is not None:
        options.append(line.dash_array)
    # options = [line.width, line.color, line.dash_array, line.cap, line.join]
    if line.line_width == 0:
        res = f"\\path[{', '.join(options)}] {p1} -- {p2};\n"
    else:
        res = f"\\draw[{', '.join(options)}] {p1} -- {p2};\n"

    return res

"""SVG related sketches are handled here."""

from dataclasses import dataclass
from math import degrees
from types import SimpleNamespace
import html
import io
import re

import matplotlib
import matplotlib.pyplot as plt

from ..colors.colors import Color, check_color
from ..geometry.geometry import close_points2, vert_label_positions
from ..graphics.all_enums import Align, Anchor, Extent, FrameShape, MarkerType
from ..settings.settings import defaults
from .svg_colors import color_to_matplotlib, color_to_svg
from .svg_common import _clip_line_to_rect, get_clip_mask_attrs
from .svg_sketch_utils import (
    _line_limits,
    get_fill_style_options,
    get_line_style_options,
    get_text_size,
    has_gradient,
    sketch_attrib,
)


@dataclass
class SVG_Mask:
    """SVG_Mask is used to configure SVG opacity mask attributes.

    Attributes mirror SVG gradient-style masks for alpha/luminance masking.
    """

    mask_type: str = None  # 'linear' or 'radial'
    x1: float = None
    y1: float = None
    x2: float = None
    y2: float = None
    cx: float = None
    cy: float = None
    r: float = None
    fx: float = None
    fy: float = None
    units: str = None  # gradient units
    spread_method: str = None
    transform: str = None
    stops: object = (
        None  # mask stops: [(offset, opacity)] or [(offset, color, opacity)]
    )
    mask_units: str = None  # maskUnits
    mask_content_units: str = None  # maskContentUnits

    def __str__(self):
        return f"SVG_Mask: {self.id}"

    def __repr__(self):
        return f"SVG_Mask: {self.id}"


def draw_line_sketch(sketch, canvas, exceptions=None):
    vertices = sketch_attrib(sketch, "vertices")
    start = vertices[0]
    end = vertices[1]
    extent = sketch_attrib(sketch, "extent")
    if not isinstance(extent, Extent) and extent is not None:
        extent = Extent(extent)

    if extent in [Extent.RAY, Extent.INFINITE]:
        limits = _line_limits(canvas)
        start, end = _clip_line_to_rect(start, end, limits, extent)

    style = get_line_style_options(sketch, exceptions=exceptions)
    clip_attr, mask_attr = get_clip_mask_attrs(sketch)

    style_attr = ""
    if style:
        style_attr = f' style="{style}"'

    return (
        f'<line x1="{start[0]}" y1="{start[1]}" '
        f'x2="{end[0]}" y2="{end[1]}"{style_attr}{clip_attr}{mask_attr} />'
    )


def draw_arc_sketch(sketch, exceptions=None):
    """Draw an arc sketch as an SVG path."""
    vertices = sketch_attrib(sketch, "vertices")
    closed = sketch_attrib(sketch, "closed")
    if closed and not close_points2(vertices[0], vertices[-1]):
        vertices = list(vertices) + [vertices[0]]

    path_parts = [f"M {vertices[0][0]},{vertices[0][1]}"]
    for vertex in vertices[1:]:
        path_parts.append(f"L {vertex[0]},{vertex[1]}")
    if closed:
        path_parts.append("Z")
    path_data = " ".join(path_parts)

    style_shape_type = "path" if closed else "polyline"
    line_style = get_line_style_options(sketch, exceptions=exceptions)

    fill_attr = ""
    skip_fill_style = False
    if sketch_attrib(sketch, "tile_svg") is not None:
        pattern_id = f"pattern_{id(sketch)}"
        fill_attr = f' fill="url(#{pattern_id})"'
        skip_fill_style = True
    elif has_gradient(sketch):
        gradient_id = sketch_attrib(sketch, "_gradient_context_id")
        if gradient_id is None:
            gradient_id = f"gradient_{id(sketch)}"
        fill_attr = f' fill="url(#{gradient_id})"'
        skip_fill_style = True

    style = line_style
    if not skip_fill_style:
        fill_style = get_fill_style_options(
            sketch, style_shape_type, exceptions=exceptions
        )
        style = f"{line_style} {fill_style}".strip()

    fill_rule_attr = ""
    if skip_fill_style and sketch_attrib(sketch, "even_odd"):
        fill_rule_attr = ' fill-rule="evenodd"'

    clip_attr, mask_attr = get_clip_mask_attrs(sketch)
    style_attr = ""
    if style:
        style_attr = f' style="{style}"'

    return (
        f'<path d="{path_data}"{style_attr}'
        f"{fill_attr}{fill_rule_attr}{clip_attr}{mask_attr} />"
    )


def draw_path_sketch(sketch, exceptions=None):
    """Draw a LinPath sketch as an SVG path without geometry conversion."""
    path_data = sketch_attrib(sketch, "path_data")
    line_style = get_line_style_options(sketch, exceptions=exceptions)

    fill_attr = ""
    skip_fill_style = False
    if sketch_attrib(sketch, "tile_svg") is not None:
        pattern_id = f"pattern_{id(sketch)}"
        fill_attr = f' fill="url(#{pattern_id})"'
        skip_fill_style = True
    elif has_gradient(sketch):
        gradient_id = sketch_attrib(sketch, "_gradient_context_id")
        if gradient_id is None:
            gradient_id = f"gradient_{id(sketch)}"
        fill_attr = f' fill="url(#{gradient_id})"'
        skip_fill_style = True

    style = line_style
    if not skip_fill_style:
        fill_style = get_fill_style_options(
            sketch, "path", exceptions=exceptions
        )
        style = f"{line_style} {fill_style}".strip()

    fill_rule_attr = ""
    if skip_fill_style and sketch_attrib(sketch, "even_odd"):
        fill_rule_attr = ' fill-rule="evenodd"'

    clip_attr, mask_attr = get_clip_mask_attrs(sketch)
    style_attr = ""
    if style:
        style_attr = f' style="{style}"'

    return (
        f'<path d="{path_data}"{style_attr}'
        f"{fill_attr}{fill_rule_attr}{clip_attr}{mask_attr} />"
    )


def draw_shape_sketch_with_indices(sketch, index=0, exceptions=None):
    """Draws a shape sketch with index numbers at each vertex for SVG.

    Args:
        sketch: The shape sketch object.
        index: The index.

    Returns:
        str: The SVG code for the shape sketch with indices.
    """
    vertices = sketch_attrib(sketch, "vertices")

    shape_type = "polygon" if sketch_attrib(sketch, "closed") else "polyline"

    line_style = get_line_style_options(sketch, exceptions=exceptions)
    fill_style = get_fill_style_options(
        sketch, shape_type, exceptions=exceptions
    )
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
    if isinstance(sketch.indices, bool):
        labels = range(len(vertices))
    else:
        labels = sketch.indices

    font_size = defaults["font_size"]
    elements = [shape_svg]
    for i, (lx, ly) in enumerate(label_positions):
        elements.append(
            f'<g transform="translate({lx} {ly}) scale(1,-1)">'
            f'<text x="0" y="0" text-anchor="middle" dominant-baseline="middle"'
            f' font-size="{font_size}">{labels[i]}</text>'
            f"</g>"
        )

    content = "\n".join(elements)
    clip_attr, mask_attr = get_clip_mask_attrs(sketch)
    if clip_attr or mask_attr:
        return f"<g{clip_attr}{mask_attr}>\n{content}\n</g>"
    return content


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


def draw_shape_sketch_with_markers(sketch, exceptions=None):
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

    has_marker_symbol = marker_type != MarkerType.EMPTY

    # Use stable sketch.id so marker refs match defs across scoped render clones.
    marker_id = f"marker_{sketch.id}"

    # Check if markers only (no line)
    markers_only = sketch_attrib(sketch, "markers_only")

    # Build path for the line
    if not markers_only:
        # Draw the line/shape
        path_data = f"M {vertices[0][0]},{vertices[0][1]}"
        for v in vertices[1:]:
            path_data += f" L {v[0]},{v[1]}"

        # Get line styling
        line_color = sketch_attrib(sketch, "line_color")
        if isinstance(line_color, Color):
            line_color = color_to_svg(line_color)
        if exceptions is not None and "stroke" in exceptions:
            line_color = "none"

        line_width = sketch_attrib(sketch, "line_width")
        if exceptions is not None and "line_width" in exceptions:
            line_width = defaults["line_width"]
        line_alpha = sketch_attrib(sketch, "line_alpha")
        if exceptions is not None and "line_alpha" in exceptions:
            line_alpha = defaults["line_alpha"]

        # Get fill styling if closed
        fill_str = '"none"'
        if (
            sketch_attrib(sketch, "fill")
            and closed
            and not (exceptions is not None and "fill" in exceptions)
        ):
            fill_color = sketch_attrib(sketch, "fill_color")
            if isinstance(fill_color, Color):
                fill_color = color_to_svg(fill_color)
            fill_alpha = sketch_attrib(sketch, "fill_alpha")
            if exceptions is not None and "fill_alpha" in exceptions:
                fill_alpha = defaults["fill_alpha"]
            fill_str = f'"{fill_color}" fill-opacity="{fill_alpha}"'

        fill_rule_attr = ""
        if (
            not (exceptions is not None and "even_odd" in exceptions)
            and "even_odd" in sketch_attrib(sketch, "__dict__")
            and sketch_attrib(
            sketch, "even_odd"
            )
        ):
            fill_rule_attr = ' fill-rule="evenodd"'

        # Build path element with marker reference
        path_element = f'<path d="{path_data}" stroke="{line_color}" stroke-width="{line_width}" '
        path_element += (
            f'stroke-opacity="{line_alpha}" fill={fill_str}{fill_rule_attr} '
        )
        if has_marker_symbol:
            path_element += (
                f'marker-start="url(#{marker_id})" '
                f'marker-mid="url(#{marker_id})" '
                f'marker-end="url(#{marker_id})"/>'
            )
        else:
            path_element += "/>"

        clip_attr, mask_attr = get_clip_mask_attrs(sketch)
        if clip_attr or mask_attr:
            return f"<g{clip_attr}{mask_attr}>\n{path_element}\n</g>"
        return path_element
    else:
        # Markers only - draw one degenerate path per vertex and attach marker-start.
        # This uses the same <defs>/<marker> pipeline as non-markers-only rendering,
        # so custom MarkerType.SHAPE markers work consistently.
        elements = []
        if not has_marker_symbol:
            return ""

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

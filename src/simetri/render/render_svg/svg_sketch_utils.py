"""Helpers that assemble and update SVG sketch elements."""

import numpy as np
from PIL import ImageFont

from ..pre_render import set_styles
from ...coloring.colors import Color
from ...base.all_enums import FontFamily, MarkerType, Types
from ...geom.bbox import bounding_box
from ...config.settings import defaults, issue_warning
from .svg_colors import color_to_svg


def svg_shape(*args, **kwargs):
    """Placeholder replaced by ``svg`` at import time.

    Raises:
        RuntimeError: If called before SVG module initialization.
    """
    raise RuntimeError(
        "svg_shape must be initialized by simetri.render.render_svg.svg before use."
    )


_active_svg_style_ids = {}


def set_active_svg_style_ids(style_ids):
    """Set the active sketch-id to CSS-class-id mapping.

    Args:
        style_ids: Mapping of sketch id to CSS class id.
    """
    global _active_svg_style_ids
    _active_svg_style_ids = style_ids


def get_active_svg_style_id(sketch):
    """Return the CSS class id for ``sketch``, if any.

    Args:
        sketch: Sketch whose style id is requested.

    Returns:
        str | None: Active style id, or None.
    """
    if sketch.id in _active_svg_style_ids:
        return _active_svg_style_ids[sketch.id]
    return None


d_shape_types = {
    Types.LINE_SKETCH: "line",
    Types.BBOX_SKETCH: "shape",
    Types.CIRCLE_SKETCH: "circle",
    Types.ELLIPSE_SKETCH: "ellipse",
    Types.RECTANGLE_SKETCH: "rect",
    Types.HANDLE: "shape",
    Types.SHAPE_SKETCH: "shape",
    Types.TAG_SKETCH: "tag",
    Types.ARC_SKETCH: "path",
    Types.BEZIER_SKETCH: "path",
    Types.PATH_SKETCH: "path",
}


def sketch_attrib(sketch, attrib):
    """Read a sketch attribute, falling back to library defaults.

    Args:
        sketch: Sketch object.
        attrib: Attribute name.

    Returns:
        The attribute value, or the matching default when missing.
    """
    try:
        return object.__getattribute__(sketch, attrib)
    except AttributeError:
        return defaults.get(attrib)


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
            issue_warning(
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
    """Measure text size using a TrueType font file path.

    Args:
        text: Text to measure.
        font_path: Path to a ``.ttf`` (or similar) font file.
        font_size: Font size in points.

    Returns:
        tuple: ``(width, height)`` of the text.
    """
    font = ImageFont.truetype(font_path, font_size)
    _, descent = font.getmetrics()
    text_width = font.getmask(text).getbbox()[2]
    text_height = font.getmask(text).getbbox()[3] + descent
    return text_width, text_height


def get_line_style_options(sketch, exceptions=None):
    """Returns the options for the line style.

    Args:
        sketch: The sketch object.
        exceptions: Optional exceptions for the line style options.

    Returns:
        list: The line style options as a list.
    """

    merged_exceptions = []
    if exceptions is not None:
        for exception in exceptions:
            merged_exceptions.append(exception)
    sketch_dict = sketch_attrib(sketch, "__dict__")

    options = []
    if "stroke" not in merged_exceptions:
        stroke = sketch_attrib(sketch, "stroke")
        if stroke:
            line_color = color_to_svg(
                sketch_attrib(sketch, "line_color"), "line_color"
            )
        else:
            line_color = "none"
        options.append(f"stroke: {line_color};")

    if "line_alpha" not in merged_exceptions:
        line_alpha = sketch_attrib(sketch, "line_alpha")
        if line_alpha is not None and line_alpha != defaults["line_alpha"]:
            options.append(f"stroke-opacity: {line_alpha};")

    if "line_width" not in merged_exceptions and "line_width" in sketch_dict:
        line_width = sketch.line_width
        if line_width != defaults["line_width"]:
            options.append(f"stroke-width: {line_width};")
    if "line_cap" not in merged_exceptions and "line_cap" in sketch_dict:
        line_cap = sketch.line_cap
        if line_cap != defaults["line_cap"]:
            options.append(f"stroke-linecap: {line_cap};")
    if "line_join" not in merged_exceptions and "line_join" in sketch_dict:
        line_join = sketch.line_join
        if line_join != defaults["line_join"]:
            options.append(f"stroke-linejoin: {line_join};")

    if "line_miter_limit" not in merged_exceptions:
        miter_limit = sketch_attrib(sketch, "miter_limit")
        if miter_limit:
            options.append(f"stroke-miterlimit: {miter_limit}")

    if "line_dash_array" not in merged_exceptions:
        line_dash_array = sketch_attrib(sketch, "line_dash_array")
        if line_dash_array:
            options.append(
                f"stroke-dasharray: {get_dash_pattern(line_dash_array)};"
            )

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

    merged_exceptions = []
    if exceptions is not None:
        for exception in exceptions:
            merged_exceptions.append(exception)

    options = []
    if "fill" not in merged_exceptions:
        fill = sketch_attrib(sketch, "fill")
        if fill and shape_type != "polyline":
            fill_color = color_to_svg(
                sketch_attrib(sketch, "fill_color"), "fill_color"
            )
        else:
            fill_color = "none"
        options.append(f"fill: {fill_color};")

    if "fill_alpha" not in merged_exceptions:
        fill_alpha = sketch_attrib(sketch, "fill_alpha")
        if fill_alpha != defaults["fill_alpha"]:
            options.append(f"fill-opacity: {fill_alpha};")

    if "even_odd" not in merged_exceptions and sketch_attrib(
        sketch, "even_odd"
    ):
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
        if marker_shape is None:
            marker_type = MarkerType.FCIRCLE
        else:
            from ..draw import create_sketch  # noqa: PLC0415 — circular import

            marker_sketch = create_sketch(marker_shape, canvas)

            if marker_sketch is None:
                raise ValueError("marker_shape sketch could not be created.")

            if isinstance(marker_color, Color):
                marker_color_svg = color_to_svg(marker_color)
            else:
                marker_color_svg = marker_color

            shape_type = get_shape_type(marker_sketch)
            if shape_type == "polygon" or shape_type == "polyline":
                coordinates = get_coordinates(marker_sketch, shape_type)
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
                    fill_attr = (
                        f'fill="{fill_color}" fill-opacity="{marker_alpha}"'
                    )
                    stroke_attr = 'stroke="none"'
                else:
                    fill_attr = 'fill="none"'
                    stroke_attr = f'stroke="{marker_color_svg}" stroke-width="1" stroke-opacity="{marker_alpha}"'

                shape_svg = (
                    f"<{shape_type} {coordinates} {fill_attr} {stroke_attr}/>"
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
    _, path_data = get_marker_path(marker_type, marker_size)

    return f'''  <marker id="{marker_id}" markerWidth="{marker_size * 2}" markerHeight="{marker_size * 2}"
      refX="0" refY="0" viewBox="{-marker_size} {-marker_size} {marker_size * 2} {marker_size * 2}" markerUnits="userSpaceOnUse" orient="auto">
    <g {fill_attr} {stroke_attr}>
      {path_data}
    </g>
  </marker>'''


def get_shape_type(sketch):
    """Map a sketch type enum to an SVG shape category string.

    Args:
        sketch: Sketch whose type is inspected.

    Returns:
        str: Shape category such as ``line``, ``circle``, or ``shape``.
    """
    shape_type = d_shape_types[sketch_attrib(sketch, "subtype")]
    if shape_type == "shape":
        if sketch_attrib(sketch, "closed"):
            shape_type = "polygon"
        else:
            shape_type = "polyline"

    return shape_type


def get_coordinates(sketch, shape_type):
    """Build SVG coordinate attributes for a sketch and shape type.

    Args:
        sketch: Sketch providing geometry.
        shape_type: Category from ``get_shape_type``.

    Returns:
        str: Attribute fragment such as ``x1=... y1=...`` or path ``d``.
    """
    if shape_type in ("polygon", "polyline"):
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
    """Build a CSS style attribute string for a sketch.

    Args:
        sketch: Sketch to style.
        shape_type: SVG shape category from ``get_shape_type``.

    Returns:
        str: Semicolon-separated CSS declarations.
    """
    line_style = get_line_style_options(sketch)
    res = [line_style]
    if shape_type in ("circle", "ellipse", "polygon", "polyline", "rect"):
        fill_style = get_fill_style_options(sketch, shape_type)
        res.append(fill_style)

    return "; ".join(res)


def get_style_maps(canvas):
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

    style_sketches = []

    def collect_sketch_styles(sketch):
        subtype = sketch_attrib(sketch, "subtype")

        if subtype in (Types.CLIPPED_SKETCH, Types.MASKED_SKETCH):
            for sketch_list in sketch.sketches:
                for child_sketch in sketch_list:
                    collect_sketch_styles(child_sketch)
            return

        if subtype == Types.COMPOSITE_SKETCH:
            for child_sketch in sketch.sketches:
                collect_sketch_styles(child_sketch)
            return

        # Skip non-shape sketches (like TexSketch)
        if subtype not in d_shape_types:
            return

        style_sketches.append(sketch)

    pages = canvas.pages
    if pages:
        for page in pages:
            for sketch in page.sketches:
                collect_sketch_styles(sketch)

    if not style_sketches:
        return {}, {}

    d_styles, sketch_style_ids = set_styles(style_sketches)
    sketch_by_id = {sketch.id: sketch for sketch in style_sketches}
    style_sketch_dict = {}
    special_fill_sketch_ids = set()

    for sketch in style_sketches:
        sketch_dict = sketch.__dict__
        if "tile_svg" in sketch_dict and sketch.tile_svg is not None:
            special_fill_sketch_ids.add(sketch.id)
        if "gradient" in sketch_dict and has_gradient(sketch):
            special_fill_sketch_ids.add(sketch.id)

    for sketch in style_sketches:
        if sketch.id not in sketch_style_ids:
            continue
        style_id = sketch_style_ids[sketch.id]
        if style_id not in style_sketch_dict:
            style_sketch_dict[style_id] = [sketch.id]
        else:
            style_sketch_dict[style_id].append(sketch.id)

    css_styles = {}
    for style_id in d_styles:
        sketch = sketch_by_id[style_sketch_dict[style_id][0]]
        shape_type = get_shape_type(sketch)
        style_parts = [get_line_style_options(sketch)]
        if shape_type != "line" and sketch.id not in special_fill_sketch_ids:
            style_parts.append(get_fill_style_options(sketch, shape_type))
        css_styles[style_id] = parse_style_string(" ".join(style_parts))

    return css_styles, sketch_style_ids


def get_styles_dict(canvas):
    """Return the CSS class dictionary for a canvas.

    Args:
        canvas: Canvas whose sketches are styled.

    Returns:
        dict: Mapping of CSS class name to property dictionaries.
    """
    css_styles, _ = get_style_maps(canvas)
    return css_styles


def get_style_class(
    sketch, shape_type, styles_dict, skip_fill=False, exceptions=None
):
    """Find the style class names that match the sketch's styles.

    Args:
        sketch: The sketch object.
        shape_type: The SVG shape type.
        styles_dict: Dictionary of style class names to style dictionaries.
        skip_fill: If True, skip adding fill style class (used when gradient/pattern is applied).

    Returns:
        str: Space-separated class names.
    """

    if exceptions:
        return ""
    style_id = get_active_svg_style_id(sketch)
    if style_id is not None:
        return style_id
    return ""


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
    from ..draw import create_sketch  # noqa: PLC0415 — circular import

    if tile.type == Types.GROUP:
        # Handle group - multiple shapes in pattern
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
        x1, y1 = gradient.axis[0][:2]
        x2, y2 = gradient.axis[1][:2]

        if units == "objectBoundingBox":
            y1 = 1 - y1
            y2 = 1 - y2

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
        cx, cy = gradient.center[:2]
        fx, fy = gradient.focal[:2]
        r = gradient.radius

        if units == "objectBoundingBox":
            cy = 1 - cy
            fy = 1 - fy

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
    from ..draw import create_sketch  # noqa: PLC0415 — circular import

    if isinstance(clip_shape, list):
        clip_contents = []
        for clip_sketch in clip_shape:
            clip_contents.append(svg_shape(clip_sketch, styles_dict))
        clip_content = "\n    ".join(clip_contents)
    elif clip_shape.type == Types.GROUP:
        # Handle group - multiple shapes in clipPath
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

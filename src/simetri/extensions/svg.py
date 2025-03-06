import webbrowser
from typing import Sequence, Dict, List, Tuple, Union, Any, Optional

from simetri.common import Type, FillMode, LineCap, LineJoin, MarkerPos
from simetri.geometry import close_points
from simetri.settings import (
    NDIGITSSVG,
    SHOWBROWSER,
    TOL,
    defaults,
    default_style,
)
from simetri import colors

from simetri.utilities import analyze_path
from simetri.affine import translation_matrix, mirror_matrix
import simetri.graphics as sg
from simetri.palettes import (
    seq_DEEP_256,
)


def get_header(width: float, height: float, title: Optional[str] = None, back_color: Optional[str] = None) -> str:
    """Return a string of the header of an SVG file.

    Args:
        width: Width of the SVG canvas in points.
        height: Height of the SVG canvas in points.
        title: Optional title for the SVG document.
        back_color: Optional background color for the SVG canvas.

    Returns:
        str: SVG header string with optional title and background.
    """
    header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}pt" height="{height}pt"'
    )
    if title:
        header += f' title="{title}"'
    header += ">"
    if back_color:
        header += f'<rect x="0" y="0" width="{width}" height="{height}"'
        header += f' style="fill:{back_color}" />\n'
    return header


def get_footer() -> str:
    """Return a string of the footer of an SVG file.

    Returns:
        str: SVG closing tag.
    """
    return "</svg>\n"


def __setStyle(canvas: "sg.Canvas", item: Any, **kwargs) -> None:
    """Set style attributes for SVG elements based on a priority cascade.

    This function determines the styling for SVG elements following this priority:
    1. Use value from kwargs if provided
    2. Use item's style attribute if available
    3. Use canvas style if available
    4. Fall back to default style if none of the above exist

    Args:
        canvas: The canvas object containing default style information.
        item: The graphical item to style.
        **kwargs: Style overrides to apply, which take highest precedence.

    Note:
        For polygons, supported styles include:
        - fill_color: Color to fill polygon with
        - line_color: Color for outline
        - line_width: Width of outline
        - line_join: Join style for line segments
        - line_cap: Cap style for line ends
        - line_dash_array: Pattern for dashed lines
        - stroke: Whether to draw outline (True->1, False->0)
        - even_odd: Fill rule (True->1 even-odd, False->0 nonzero)
        - fill: Whether to fill polygon (True->1, False->0)

        For polylines:
        - line_color: Color for line
        - line_width: Width of line
        - line_join: Join style for line segments
        - line_cap: Cap style for line ends
        - line_dash_array: Pattern for dashed lines
        - stroke: Whether to draw line (True->1, False->0)
    """
    pass


def get_style(shape: Any, tol: float = TOL) -> str:
    """Return a string of style attributes for the given shape.

    Args:
        shape: The shape object to extract styling from.
        tol: Tolerance value for numerical comparisons.

    Returns:
        str: CSS style string for SVG elements.
    """
    dict_line_style = {
        "line_width": "stroke-width",
        "line_color": "stroke",
        "line_alpha": "stroke-opacity",
        "line_miter_limit": "stroke-miterlimit",
        "line_join": "stroke-linejoin",
        "line_cap": "stroke-linecap",
        "marker": "marker-end",
    }

    # for polygons use dict_style
    dict_style = {
        "fill_color": "fill",
        "fill_alpha": "fill-opacity",
        "fill_mode": "fill-rule",
        "line_width": "stroke-width",
        "line_color": "stroke",
        "line_alpha": "stroke-opacity",
        "line_miter_limit": "stroke-miterlimit",
        "line_join": "stroke-linejoin",
        "line_cap": "stroke-linecap",
    }

    dict_line_join = {
        LineJoin.MITER: "miter",
        LineJoin.ROUND: "round",
        LineJoin.BEVEL: "bevel",
        LineJoin.ARCS: "arcs",
        LineJoin.MITER_CLIP: "miter-clip",
    }

    dict_line_cap = {
        LineCap.BUTT: "butt",
        LineCap.ROUND: "round",
        LineCap.SQUARE: "square",
    }
    style_elements = []
    if not sg.draw_filled(shape, tol):
        dict_style = dict_line_style
        style_elements.append("fill: none;")

    for key, value in vars(shape).items():
        if key in dict_style:
            if value is None:
                if key in default_style.attribs:
                    value = getattr(default_style, key)
                else:
                    print(f"Warning: {key} is None")

            if key == "fill_mode":
                if value == FillMode.NONZERO:
                    value = "nonzero"
                    style_elements.append(f"{dict_style[key]}: {value};")
            elif key == "line_join":
                if value != LineJoin.MITER:
                    value = dict_line_join[value]
                    style_elements.append(f"{dict_style[key]}: {value};")
            elif key == "line_cap":
                if value != LineCap.BUTT:
                    value = dict_line_cap[value]
                    style_elements.append(f"{dict_style[key]}: {value};")
            elif key in ("fill_color", "line_color"):
                if isinstance(value, (tuple, list)):
                    v1, v2, v3 = value
                    if v1 > 0 or v2 > 0 or v3 > 0:
                        red = v1
                        green = v2
                        blue = v3
                    else:
                        red = round(value[0] * 255)
                        green = round(value[1] * 255)
                        blue = round(value[2] * 255)
                    value = f"rgb({red}, {green}, {blue})"
                else:
                    red = round(value.red * 255)
                    green = round(value.green * 255)
                    blue = round(value.blue * 255)

                value = f"rgb({red}, {green}, {blue})"
                style_elements.append(f"{dict_style[key]}: {value};")
            elif key in ("fill_alpha", "line_alpha"):
                if value != 1:
                    value = round(value, NDIGITSSVG)
                    style_elements.append(f"{dict_style[key]}: {value};")
            elif key == "line_width":
                value = str(round(value, NDIGITSSVG)) + "pt"
                style_elements.append(f"{dict_style[key]}: {value};")
            elif key == "line_miter_limit":
                if value != 4:
                    value = str(round(value, NDIGITSSVG))
                    style_elements.append(f"{dict_style[key]}: {value};")
            elif key == "marker":
                if value:
                    style_elements.append(f"{dict_style[key]}: url(#{value});")
            else:
                style_elements.append(f"{dict_style[key]}: {value};")

    style = " ".join(style_elements)

    return f"{style}"


def draw_circle(
    canvas: "sg.Canvas",
    cx: float,
    cy: float,
    radius: float,
    fill_color: colors.Color = defaults['fill_color'],
    fill_alpha: float = defaults['fill_alpha'],
    line_width: float = defaults['line_width'],
    line_color: colors.Color = defaults['line_color'],
    line_alpha: float = defaults['line_alpha'],
    line_cap: LineCap = defaults['line_cap'],
    line_join: LineJoin = defaults['line_join'],
    line_miter_limit: float = defaults['line_miter_limit'],
    line_dash_array: Sequence = defaults['line_dash_array'],
) -> str:
    """Generate SVG markup for a circle.

    Args:
        canvas: The canvas to draw on.
        cx: X-coordinate of the circle center.
        cy: Y-coordinate of the circle center.
        radius: Radius of the circle.
        fill_color: Color to fill the circle with.
        fill_alpha: Opacity level for the fill (0.0-1.0).
        line_width: Width of the circle's outline.
        line_color: Color of the circle's outline.
        line_alpha: Opacity level for the outline (0.0-1.0).
        line_cap: Style for the endpoints of the outline.
        line_join: Style for the joins between line segments.
        line_miter_limit: Limit for the miter join style.
        line_dash_array: Pattern for dashed lines.

    Returns:
        str: SVG circle element.
    """
    style = f"""stroke={line_color} stroke-width={line_width} fill={fill_color}
        fill-opacity={fill_alpha} stroke-opacity={line_alpha}
        stroke-linecap={line_cap} stroke-linejoin={line_join}
        stroke-miterlimit={line_miter_limit} stroke-dasharray={line_dash_array}"""
    return f'<circle cx="{cx}" cy="{cy}" r="{radius}" style="{style}" />\n'


def draw_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    line_width: float = defaults['line_width'],
    line_color: colors.Color = defaults['line_color'],
    line_alpha: float = defaults['line_alpha'],
    line_cap: LineCap = defaults['line_cap'],
    line_join: LineJoin = defaults['line_join'],
    line_miter_limit: float = defaults['line_miter_limit'],
    line_dash_array: Sequence = defaults['line_dash_array'],
) -> str:
    """Generate SVG markup for a line.

    Args:
        x1: X-coordinate of the start point.
        y1: Y-coordinate of the start point.
        x2: X-coordinate of the end point.
        y2: Y-coordinate of the end point.
        line_width: Width of the line.
        line_color: Color of the line.
        line_alpha: Opacity level for the line (0.0-1.0).
        line_cap: Style for the endpoints of the line.
        line_join: Style for the joins between line segments.
        line_miter_limit: Limit for the miter join style.
        line_dash_array: Pattern for dashed lines.

    Returns:
        str: SVG line element.
    """
    style = f"""stroke={line_color} stroke-width={line_width}
        stroke-opacity={line_alpha} stroke-linecap={line_cap}
        stroke-linejoin={line_join} stroke-miterlimit={line_miter_limit}
        stroke-dasharray={line_dash_array}"""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" style="{style}" />\n'


def draw_lines(
    points: str,
    line_width: float = defaults['line_width'],
    line_color: colors.Color = defaults['line_color'],
    line_alpha: float = defaults['line_alpha'],
    line_cap: LineCap = defaults['line_cap'],
    line_join: LineJoin = defaults['line_join'],
    line_miter_limit: float = defaults['line_miter_limit'],
    line_dash_array: Sequence = defaults['line_dash_array'],
) -> str:
    """Generate SVG markup for a series of connected lines.

    Args:
        points: String representation of points coordinates.
        line_width: Width of the lines.
        line_color: Color of the lines.
        line_alpha: Opacity level for the lines (0.0-1.0).
        line_cap: Style for the endpoints of the lines.
        line_join: Style for the joins between line segments.
        line_miter_limit: Limit for the miter join style.
        line_dash_array: Pattern for dashed lines.

    Returns:
        str: SVG polyline element.
    """
    style = f"""stroke={line_color} stroke-width={line_width}
        stroke-opacity={line_alpha} stroke-linecap={line_cap}
        stroke-linejoin={line_join} stroke-miterlimit={line_miter_limit}
        stroke-dasharray={line_dash_array}"""
    return f'<polyline points="{points}" style="{style}" />\n'


def draw_polygon(
    points: str,
    fill_color: colors.Color = defaults['fill_color'],
    fill_alpha: float = defaults['fill_alpha'],
    line_width: float = defaults['line_width'],
    line_color: colors.Color = defaults['line_color'],
    line_alpha: float = defaults['line_alpha'],
    line_cap: LineCap = defaults['line_cap'],
    line_join: LineJoin = defaults['line_join'],
    line_miter_limit: float = defaults['line_miter_limit'],
    line_dash_array: Sequence = defaults['line_dash_array'],
) -> str:
    """Generate SVG markup for a polygon.

    Args:
        points: String representation of points coordinates.
        fill_color: Color to fill the polygon with.
        fill_alpha: Opacity level for the fill (0.0-1.0).
        line_width: Width of the polygon outline.
        line_color: Color of the polygon outline.
        line_alpha: Opacity level for the outline (0.0-1.0).
        line_cap: Style for the endpoints of the outline.
        line_join: Style for the joins between line segments.
        line_miter_limit: Limit for the miter join style.
        line_dash_array: Pattern for dashed lines.

    Returns:
        str: SVG polygon element.
    """
    style = f"""stroke={line_color} stroke-width={line_width} fill={fill_color}
        fill-opacity={fill_alpha} stroke-opacity={line_alpha}
        stroke-linecap={line_cap} stroke-linejoin={line_join}
        stroke-miterlimit={line_miter_limit} stroke-dasharray={line_dash_array}"""
    return f'<polygon points="{points}" style="{style}" />\n'


def draw_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill_color: colors.Color = defaults['fill_color'],
    fill_alpha: float = defaults['fill_alpha'],
    line_width: float = defaults['line_width'],
    line_color: colors.Color = defaults['line_color'],
    line_alpha: float = defaults['line_alpha'],
    line_cap: LineCap = defaults['line_cap'],
    line_join: LineJoin = defaults['line_join'],
    line_miter_limit: float = defaults['line_miter_limit'],
    line_dash_array: Sequence = defaults['line_dash_array'],
) -> str:
    """Generate SVG markup for a rectangle.

    Args:
        x: X-coordinate of the top-left corner.
        y: Y-coordinate of the top-left corner.
        width: Width of the rectangle.
        height: Height of the rectangle.
        fill_color: Color to fill the rectangle with.
        fill_alpha: Opacity level for the fill (0.0-1.0).
        line_width: Width of the rectangle outline.
        line_color: Color of the rectangle outline.
        line_alpha: Opacity level for the outline (0.0-1.0).
        line_cap: Style for the endpoints of the outline.
        line_join: Style for the joins between line segments.
        line_miter_limit: Limit for the miter join style.
        line_dash_array: Pattern for dashed lines.

    Returns:
        str: SVG rectangle element.
    """
    style = f"""stroke={line_color} stroke-width={line_width} fill={fill_color}
        fill-opacity={fill_alpha} stroke-opacity={line_alpha}
        stroke-linecap={line_cap} stroke-linejoin={line_join}
        stroke-miterlimit={line_miter_limit} stroke-dasharray={line_dash_array}"""
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" style="{style}" />\n'
    )


def draw_text(
    canvas: "sg.Canvas",
    x: float,
    y: float,
    text: str,
    font_name: str = "Helvetica",
    font_size: int = 11,
    fill_color: colors.Color = colors.black,
    angle: float = 0,
    anchor: str = "sw",
) -> str:
    """Generate SVG markup for text.

    Args:
        canvas: The canvas to draw on.
        x: X-coordinate for text placement.
        y: Y-coordinate for text placement.
        text: The text to draw.
        font_name: Name of the font to use.
        font_size: Font size in points.
        fill_color: Color of the text.
        angle: Rotation angle of the text in degrees.
        anchor: Text anchor position ("sw", "n", "e", etc.).

    Returns:
        str: SVG text element.
    """
    style = None
    return f'<text x="{x}" y="{y}" style="{style}">{text}</text>\n'


def create_SVG(code: List[str], canvas: "sg.Canvas", dict_ID_obj: Dict[str, Any], tol: float = TOL) -> str:
    """Create the complete SVG representation of the canvas.

    Args:
        code: List of code instructions to execute.
        canvas: The canvas containing drawing information.
        dict_ID_obj: Dictionary mapping IDs to drawing objects.
        tol: Tolerance value for numerical comparisons.

    Returns:
        str: Complete SVG document.
    """
    translate = translation_matrix(0, canvas.height)
    reflect = mirror_matrix([(0, canvas.height), (1, canvas.height)])
    svg_transform = translate @ reflect  # svg origin is at top left

    def draw(item_ID, code_list, tol=TOL):
        item = dict_ID_obj[item_ID]
        if item.type == Type.SHAPE:
            coords = []
            vertices = item.final_coords @ svg_transform
            for vert in vertices[:, :2]:
                x, y = [round(x, NDIGITSSVG) for x in vert]
                coords.extend([str(x), str(y)])
            coords = ", ".join(coords)
            same_points = close_points(item.vertices[0], item.vertices[-1], tol)
            closed = item.closed
            style = get_style(item)
            if style:
                style = f'\nstyle="{style}"'
            if same_points or closed:
                code_list.append(f'<polygon points = "{coords}" {style} />')
            else:
                code_list.append(f'<polyline points = "{coords}" {style} />')
        elif item.type == Type.BATCH:
            if item.subtype == Type.LACE:
                if item.draw_fragments:
                    groups = item.group_fragments()
                    if item.swatch:
                        swatch = item.swatch
                    else:
                        swatch = seq_DEEP_256  # use DEFAULT_SWATCH
                    for i, group in enumerate(groups):
                        color = swatch[i * 5]
                        for fragment in group:
                            fragment.fill_color = color
                            draw(fragment.id, code_list)
                            if fragment.inner_lines:
                                for line in fragment.inner_lines:
                                    draw(line.id, code_list)
                for plait in item.plaits:
                    draw(plait.id, code_list)
                    if plait.inner_lines:
                        for line in plait.inner_lines:
                            draw(line.id, code_list)
                if item.draw_markers:
                    marker = item.marker
                    if item.marker_pos == MarkerPos.CONVEXHULL:
                        for vert in item.convex_hull:
                            marker.move_to(*vert)
                            draw(marker.id, code_list)
                    elif item.marker_pos == MarkerPos.MAINX:
                        for vert in item.iter_main_intersections():
                            marker.move_to(*vert.point)
                            draw(marker.id, code_list)
                    elif item.marker_pos == MarkerPos.OFFSETX:
                        for vert in item.iter_offset_intersections():
                            marker.move_to(*vert.point)
                            draw(marker.id, code_list)
            elif item.subtype == Type.SKETCH:
                if item.draw_fragments:
                    for fragment in item.fragments:
                        draw(fragment.id, code_list)
                if item.draw_plaits:
                    for plait in item.plaits:
                        plait.fill = True
                        draw(plait.id, code_list)
            elif item.subtype == Type.OVERLAP:
                for section in item.sections:
                    draw(section.id, code_list)
            else:
                for element in item.all_elements:
                    draw(element.id, code_list)

    code_list = [get_header(canvas.width, canvas.height)]

    for line in code:
        exec(line)

    code_list.append(get_footer())

    return "\n".join(code_list)


def _draw_background(self) -> None:
    """Draw the background for the canvas.

    Args:
        self: The canvas object.
    """
    self._drawing.add(gs.Rect(0, 0, self.width, self.height, fill_color=self.back_color))


def save_SVG(canvas: "sg.Canvas", file_path: str, dict_ID_obj: Dict[str, Any], show: bool = SHOWBROWSER) -> None:
    """Save the canvas as an SVG file.

    Args:
        canvas: The canvas to save.
        file_path: Path where the SVG file will be saved.
        dict_ID_obj: Dictionary mapping IDs to drawing objects.
        show: If True, open the saved SVG file in a web browser.

    Raises:
        RuntimeError: If the file path is invalid.
    """
    valid, error_message, extension = analyze_path(file_path)
    if valid:
        if extension == ".svg":
            f = open(file_path, "w")
            code = [line.replace(")", ", code_list)") for line in canvas._code]
            svg_Code = create_SVG(code, canvas, dict_ID_obj)
            f.writelines(svg_Code)
            f.close()
    else:
        raise RuntimeError(error_message)
    if show:
        webbrowser.open(file_path)

import webbrowser
from typing import Sequence

from simetri.common import Type, FillMode, LineCap, LineJoin, MarkerPos
from simetri.geometry import close_points
from simetri.settings import (
    NDIGITSSVG,
    SHOWBROWSER,
    TOL,
    logging,
    Style,
    defaults,
    default_style,
)
from simetri import colors

from simetri.utilities import analyze_path
from simetri.affine import translation_matrix, mirror_matrix
import simetri.graphics as sg
from simetri.palettes import (
    seq_DEEP_256,
    seq_LINEARL_256,
    seq_MATTER_256,
    seq_BATLOW_256,
)


def get_header(width, height, title=None, back_color=None):
    """Return a string of the header of an SVG file."""

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


def get_footer():
    """Return a string of the footer of an SVG file."""
    return "</svg>\n"


def __setStyle(canvas, item, **kwargs):
    """
    check if kwargs has the attribute
        if it does
            use that value
        if it doesn't
            check if the item has a style attribute
                if it does
                    use that value
                if it doesn't
                    use the canvas style
                    if the canvas style is None,
                        use the default style


    For polygons:
    fill_color -> _rlCanvas.setFillColor(color)
    line_color -> _rlCanvas.setLineColor(color)
    line_width -> _rlCanvas.setLineWidth(width)
    line_join -> _rlCanvas.setLineJoin(join)
    line_cap -> _rlCanvas.setLineCap(cap)
    line_dash_array -> _rlCanvas.setLineDash(dashArray, phase)
    stroke -> _rlCanvas.stroke 0 no lines 1 lines (True->1, False->0)
    even_odd -> _rlCanvas.evenOddFill 0 nonzero 1 even_odd (True->1, False->0)
    fill -> _rlCanvas.fill 0 no fill 1 fill (True->1, False->0)


    For polylines:
    line_color -> _rlCanvas.setLineColor(color)
    line_width -> _rlCanvas.setLineWidth(width)
    line_join -> _rlCanvas.setLineJoin(join)
    line_cap -> _rlCanvas.setLineCap(cap)
    line_dash_array -> _rlCanvas.setLineDash(dashArray, phase)
    stroke -> _rlCanvas.stroke 0 no lines 1 lines (True->1, False->0)
    """


def get_style(shape, tol=TOL):
    """Return a string of style attributes for the given shape."""
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
        "fill_mode": "fill-rule",
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
):
    style = f"""stroke={line_color} stroke-width={line_width} fill={fill_color}
        fill-opacity={fill_alpha} stroke-opacity={line_alpha}
        stroke-linecap={line_cap} stroke-linejoin={line_join}
        stroke-miterlimit={line_miter_limit} stroke-dasharray={line_dash_array}"""
    """Return a string of an SVG circle."""
    return f'<circle cx="{cx}" cy="{cy}" r="{radius}" style="{style}" />\n'


def draw_line(
    x1,
    y1,
    x2,
    y2,
    line_width: float = defaults['line_width'],
    line_color: colors.Color = defaults['line_color'],
    line_alpha: float = defaults['line_alpha'],
    line_cap: LineCap = defaults['line_cap'],
    line_join: LineJoin = defaults['line_join'],
    line_miter_limit: float = defaults['line_miter_limit'],
    line_dash_array: Sequence = defaults['line_dash_array'],
):
    """Return a string of an SVG line."""
    style = f"""stroke={line_color} stroke-width={line_width}
        stroke-opacity={line_alpha} stroke-linecap={line_cap}
        stroke-linejoin={line_join} stroke-miterlimit={line_miter_limit}
        stroke-dasharray={line_dash_array}"""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" style="{style}" />\n'


def draw_lines(
    points,
    line_width: float = defaults['line_width'],
    line_color: colors.Color = defaults['line_color'],
    line_alpha: float = defaults['line_alpha'],
    line_cap: LineCap = defaults['line_cap'],
    line_join: LineJoin = defaults['line_join'],
    line_miter_limit: float = defaults['line_miter_limit'],
    line_dash_array: Sequence = defaults['line_dash_array'],
):
    """Draw a series of connected lines."""
    style = f"""stroke={line_color} stroke-width={line_width}
        stroke-opacity={line_alpha} stroke-linecap={line_cap}
        stroke-linejoin={line_join} stroke-miterlimit={line_miter_limit}
        stroke-dasharray={line_dash_array}"""
    return f'<polyline points="{points}" style="{style}" />\n'


def draw_polygon(
    points,
    fill_color: colors.Color = defaults['fill_color'],
    fill_alpha: float = defaults['fill_alpha'],
    line_width: float = defaults['line_width'],
    line_color: colors.Color = defaults['line_color'],
    line_alpha: float = defaults['line_alpha'],
    line_cap: LineCap = defaults['line_cap'],
    line_join: LineJoin = defaults['line_join'],
    line_miter_limit: float = defaults['line_miter_limit'],
    line_dash_array: Sequence = defaults['line_dash_array'],
):
    """Draw a polygon with points."""
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
):
    """Draw a rectangle."""
    style = f"""stroke={line_color} stroke-width={line_width} fill={fill_color}
        fill-opacity={fill_alpha} stroke-opacity={line_alpha}
        stroke-linecap={line_cap} stroke-linejoin={line_join}
        stroke-miterlimit={line_miter_limit} stroke-dasharray={line_dash_array}"""
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" style="{style}" />\n'
    )


def draw_text(
    canvas,
    x,
    y,
    text,
    font_name="Helvetica",
    font_size=11,
    fill_color=colors.black,
    angle=0,
    anchor="sw",
):
    """Draw text on the canvas."""
    style = None
    return f'<text x="{x}" y="{y}" style="{style}">{text}</text>\n'


def create_SVG(code, canvas, dict_ID_obj, tol=TOL):
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
    self._drawing.add(gs.Rect(0, 0, self.width, self.height, fill_color=self.back_color))


def save_SVG(canvas, file_path, dict_ID_obj, show=SHOWBROWSER):
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

"""TikZ helper functions for styles, colors, paths, and shading."""

from math import degrees
from typing import List

import numpy as np
import simetri.graphics as sg

from ..canvas.style_map import line_style_map, marker_style_map, shape_style_map
from ..colors.colors import Color, check_color
from ..core.all_enums import (
    BackStyle,
    LineDashArray,
    LineWidth,
    ShadeType,
    Types,
)
from ..graphics.shape import Shape
from ..canvas.sketch import ShapeSketch, TagSketch
from ..settings.settings import defaults, tikz_defaults

axis_shading_types = [
    ShadeType.AXIS_LEFT_RIGHT,
    ShadeType.AXIS_TOP_BOTTOM,
    ShadeType.AXIS_LEFT_MIDDLE,
    ShadeType.AXIS_RIGHT_MIDDLE,
    ShadeType.AXIS_TOP_MIDDLE,
    ShadeType.AXIS_BOTTOM_MIDDLE,
]

radial_shading_types = [
    ShadeType.BALL,
    ShadeType.RADIAL_INNER,
    ShadeType.RADIAL_OUTER,
    ShadeType.RADIAL_INNER_OUTER,
]

NumberOrTex = int | float | str


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


def frame_options(sketch: TagSketch) -> list[str]:
    """Returns the options for the frame of the tag node.

    Args:
        sketch (TagSketch): The tag sketch object.

    Returns:
        list[str]: The options for the frame of the tag node.
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
        if sketch.text in (None, ""):
            min_size = get_min_size(sketch)
            if min_size:
                options.extend(min_size)

    return options


def color_to_tikz(color, property_name=None):
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
        color = defaults[property_name]
    if isinstance(color, str):
        color = check_color(color)
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

    blend_group = sketch.blend_group
    blend_mode = sketch.blend_mode
    fill_alpha = sketch.fill_alpha
    line_alpha = sketch.line_alpha
    text_alpha = sketch.text_alpha
    alpha = sketch.alpha
    even_odd_rule = sketch.even_odd_rule
    transparency_group = sketch.transparency_group

    if blend_group:
        options.append(f"blend group={blend_mode}")
    elif blend_mode:
        options.append(f"blend mode={blend_mode}")
    if fill_alpha not in (None, 1):
        options.append(f"fill opacity={fill_alpha}")
    if line_alpha not in (None, 1):
        options.append(f"draw opacity={line_alpha}")
    if text_alpha not in (None, 1):
        options.append(f"text opacity={alpha}")
    if alpha not in (None, 1):
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
    try:
        mask = sketch.mask
    except AttributeError:
        mask = sketch

    if mask is None:
        return ""

    if isinstance(mask, list):
        res = []
        for clip_sketch in mask:
            res.append(get_clip_code(clip_sketch))
        return "".join(res)

    if mask.subtype in (Types.CIRCLE, Types.CIRCLE_SKETCH):
        x, y = mask.center[:2]
        res = f"\\clip({x}, {y}) circle ({mask.radius});\n"
    elif mask.subtype in [Types.ELLIPSE, Types.ELLIPSE_SKETCH]:
        x, y = mask.center[:2]
        res = (
            f"\\clip({x}, {y}) ellipse ({mask.x_radius} and {mask.y_radius});\n"
        )
    elif mask.subtype in [Types.RECTANGLE, Types.RECTANGLE_SKETCH]:
        try:
            vertices = mask.vertices
            xs = [vertex[:2][0] for vertex in vertices]
            ys = [vertex[:2][1] for vertex in vertices]
            x1 = min(xs)
            y1 = min(ys)
            x2 = max(xs)
            y2 = max(ys)
        except AttributeError:
            corners = mask.b_box.corners
            x1, y1 = corners[1][:2]
            x2, y2 = corners[3][:2]
        res = f"\\clip({x1}, {y1}) rectangle ({x2}, {y2});\n"

    elif mask.subtype in [Types.SHAPE, Types.SHAPE_SKETCH, Types.BBOX_SKETCH]:
        vertices = mask.vertices
        if vertices:
            coords = " -- ".join(
                f"({vertex[:2][0]}, {vertex[:2][1]})" for vertex in vertices
            )
            if mask.closed:
                res = f"\\clip {coords} -- cycle;\n"
            else:
                res = f"\\clip {coords};\n"
        else:
            res = ""
    else:
        res = ""

    return res


def _parse_mask_offset(offset):
    if isinstance(offset, (int, float)):
        return float(offset)
    if isinstance(offset, str) and offset.endswith("%"):
        return float(offset[:-1]) / 100.0
    return float(offset)


def _luminance_from_stop_color(stop_color):
    if isinstance(stop_color, Color):
        red, green, blue = stop_color.rgb255
    elif isinstance(stop_color, str):
        color_text = stop_color.strip().lower()
        if color_text == "white":
            return 1.0
        if color_text == "black":
            return 0.0
        if color_text.startswith("#") and len(color_text) == 7:
            red = int(color_text[1:3], 16)
            green = int(color_text[3:5], 16)
            blue = int(color_text[5:7], 16)
        else:
            return 1.0
    else:
        return 1.0

    return (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0


def _effective_alpha_from_stop(stop):
    if isinstance(stop, dict):
        offset = _parse_mask_offset(stop["offset"])
        stop_color = stop.get("stop-color", stop.get("stop_color", "white"))
        stop_opacity = stop.get(
            "stop-opacity",
            stop.get("stop_opacity", stop.get("opacity", 1.0)),
        )
    elif hasattr(stop, "offset") and hasattr(stop, "color"):
        offset = _parse_mask_offset(stop.offset)
        stop_color = stop.color if stop.color is not None else "white"
        stop_opacity = stop.opacity if stop.opacity is not None else 1.0
    else:
        offset = _parse_mask_offset(stop[0])
        second_value = stop[1]
        if len(stop) >= 3:
            stop_color = second_value
            stop_opacity = stop[2]
        elif isinstance(second_value, (int, float)):
            stop_color = "white"
            stop_opacity = second_value
        else:
            stop_color = second_value
            stop_opacity = 1.0

    luminance = _luminance_from_stop_color(stop_color)
    alpha = max(0.0, min(1.0, float(stop_opacity) * luminance))
    return offset, alpha


def _pgf_gray(transparency: int) -> str:
    if transparency <= 0:
        return "white"
    if transparency >= 100:
        return "black"
    return f"black!{transparency}"


def _extract_gradient_stop_color(stop):
    if isinstance(stop, sg.Stop):
        return stop.color

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
    if isinstance(stop, sg.Stop):
        return stop.offset

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


def sg_to_tikz(
    sketch, attrib_list, attrib_map, conditions=None, exceptions=None
):
    """Convert resolved sketch attributes to TikZ options."""
    boolean_attribs = ["smooth"]
    tikz_enum_attribs = {
        "line_width": LineWidth,
        "line_dash_array": LineDashArray,
    }
    converters = {
        "line_color": color_to_tikz,
        "fill_color": color_to_tikz,
        "double_color": color_to_tikz,
        "draw": color_to_tikz,
        "marker_color": color_to_tikz,
        "line_dash_array": get_dash_pattern,
    }

    options = []
    for attrib_name in attrib_list:
        if exceptions is not None and attrib_name in exceptions:
            continue

        if (
            conditions is not None
            and attrib_name in conditions
            and not conditions[attrib_name]
        ):
            continue
        if attrib_name not in attrib_map:
            continue
        tikz_attrib = attrib_map[attrib_name]
        value = sketch.__dict__[attrib_name]

        if value is None:
            continue

        if attrib_name in tikz_enum_attribs:
            enum_type = tikz_enum_attribs[attrib_name]
            if isinstance(value, enum_type):
                options.append(value.value)
                continue
            if isinstance(value, str) and value in enum_type:
                options.append(value)
                continue

        if attrib_name in boolean_attribs:
            if value:
                options.append(tikz_attrib)
            continue

        if tikz_attrib in tikz_defaults and value == tikz_defaults[tikz_attrib]:
            continue

        if attrib_name in converters and value is not None:
            value = converters[attrib_name](value)

        options.append(f"{tikz_attrib}={value}")

    return options


def get_line_style_options(sketch, exceptions=None):
    """Returns the options for the line style.

    Args:
        sketch: The sketch object.
        exceptions: Optional exceptions for the line style options.

    Returns:
        list: The line style options as a list.
    """
    if exceptions is None:
        exceptions = []

    attrib_map = {
        "double_color": "double",
        "double_distance": "double distance",
        "line_color": "draw",
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
    if "exclusive" in sketch.__dict__ and sketch.exclusive is not None:
        attribs = [
            style_name
            for style_name in attribs
            if style_name in sketch.exclusive
        ]
    for style_key in exceptions:
        if style_key in attribs:
            attribs.remove(style_key)
    if sketch.stroke:
        if (
            "fillet_radius" in attribs
            and exceptions
            and "draw_fillets" not in exceptions
        ):
            conditions = {"fillet_radius": sketch.draw_fillets}
        else:
            conditions = None
        if "line_alpha" in attribs and sketch.line_alpha in (None, 1):
            attribs.remove("line_alpha")
        if "double_color" in attribs and not sketch.draw_double:
            if "double_color" in attribs:
                attribs.remove("double_color")
            if "double_distance" in attribs:
                attribs.remove("double_distance")
        if "smooth" in attribs and not sketch.smooth:
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
    if exceptions is None:
        exceptions = []

    attrib_map = {
        "fill_color": "fill",
        "fill_alpha": "fill opacity",
        #'fill_mode': 'even odd rule',
        "blend_mode": "blend mode",
        "frame_back_color": "fill",
    }
    attribs = list(shape_style_map.keys())
    if "exclusive" in sketch.__dict__ and sketch.exclusive is not None:
        attribs = [
            style_name
            for style_name in attribs
            if style_name in sketch.exclusive
        ]

    for style_key in exceptions:
        if style_key in attribs:
            attribs.remove(style_key)
    if "fill_alpha" in attribs and sketch.fill_alpha in (None, 1):
        attribs.remove("fill_alpha")
    if sketch.fill and not sketch.back_style == BackStyle.PATTERN:
        res = sg_to_tikz(sketch, attribs, attrib_map, exceptions=exceptions)
        if frame:
            res = [
                f"fill = {color_to_tikz(getattr(sketch, 'back_color'), 'back_color')}"
            ] + res
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
            res = color_to_tikz(color)
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
        res.append(
            f"upper left = {color_to_tikz(sketch.shade_upper_left_color)}"
        )
    if sketch.shade_upper_right_color:
        res.append(
            f"upper right = {color_to_tikz(sketch.shade_upper_right_color)}"
        )
    if sketch.shade_lower_left_color:
        res.append(
            f"lower left = {color_to_tikz(sketch.shade_lower_left_color)}"
        )
    if sketch.shade_lower_right_color:
        res.append(
            f"lower right = {color_to_tikz(sketch.shade_lower_right_color)}"
        )

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
        res.append(f"inner color = {color_to_tikz(sketch.shade_inner_color)}")
    elif sketch.shade_type == ShadeType.RADIAL_OUTER:
        res.append(f"outer color = {color_to_tikz(sketch.shade_outer_color)}")
    elif sketch.shade_type == ShadeType.RADIAL_INNER_OUTER:
        res.append(f"inner color = {color_to_tikz(sketch.shade_inner_color)}")
        res.append(f"outer color = {color_to_tikz(sketch.shade_outer_color)}")

    return ", ".join(res)


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
        res = f"ball color = {color_to_tikz(sketch.shade_ball_color)}"
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
        if pattern_type in ("Stars", "Dots"):
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
            options += f", pattern color={color_to_tikz(color)}, "

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
    translation_unit: str = "",  # "bp",
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


def is_stroked(shape: Shape) -> bool:
    """Returns True if the shape is stroked.

    Args:
        shape (Shape): The shape object.

    Returns:
        bool: True if the shape is stroked, False otherwise.
    """
    return (
        shape.stroke and shape.line_color is not None and shape.line_width > 0
    )


def get_frame_options(sketch):
    """Returns the options for the frame of a TagSketch.

    Args:
        sketch: The TagSketch object.

    Returns:
        list: The options for the frame of the TagSketch.
    """
    options = get_line_style_options(sketch)
    options += get_fill_style_options(sketch)
    if sketch.text in (None, ""):
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


def _get_gradient_shading_options(sketch):
    if "gradient" not in sketch.__dict__:
        return None
    gradient = sketch.gradient
    if gradient is None or gradient.gradient_type != sg.GradientType.LINEAR:
        return None

    stops = gradient.stops
    if not isinstance(stops, (list, tuple)) or len(stops) < 2:
        return None

    first_color = _extract_gradient_stop_color(stops[0])
    last_color = _extract_gradient_stop_color(stops[-1])

    try:
        left = (
            color_to_tikz(first_color)
            if isinstance(first_color, Color)
            else str(first_color)
        )
        right = (
            color_to_tikz(last_color)
            if isinstance(last_color, Color)
            else str(last_color)
        )
    except Exception:
        left, right = "black", "white"

    axis = gradient.axis
    x1, y1 = axis[0]
    x2, y2 = axis[1]

    try:
        angle = degrees(
            np.arctan2(float(y2) - float(y1), float(x2) - float(x1))
        )
    except Exception:
        angle = 0.0

    options = [f"left color={left}", f"right color={right}"]
    if abs(angle) > 1e-12:
        options.append(f"shading angle={angle:.2f}")
    return options

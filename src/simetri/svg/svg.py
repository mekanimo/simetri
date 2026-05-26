from __future__ import annotations


from ..graphics.all_enums import (
    BackStyle,
    MarkerType,
    Types,
)
from ..graphics.bbox import bounding_box
from ..geometry.geometry import (
    homogenize,
)
from ..colors.colors import black, white
from ..settings.settings import defaults, issue_warning, svg_defaults
from ..canvas.style_map import  marker_style_map
from ..canvas.pre_render import (
    render_svg_scope_loop,
)
from ..graphics.sketch import MaskSketch
from .filters import SVG_Filter
from .svg_sketch_utils import *
from .svg_sketch import *
from .svg_mask import *
from .svg_mask import _canvas_mask_scope_sketch
from .svg_utils import *

from ..colors.colors import check_color
from .svg_colors import color_to_svg


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
    elif (draw_markers and marker_type != MarkerType.EMPTY) or smooth:
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


def get_svg_shapes(canvas: "Canvas", styles_dict: dict) -> str:
    """Convert the sketches in the Canvas to SVG code.

    Args:
        canvas ("Canvas"): The canvas object.

    Returns:
        str: The SVG code.
    """

    line_scope_style_keys = [
        "stroke",
        "draw_double",
        "double_color",
        "double_distance",
        "line_color",
        "line_alpha",
        "line_width",
        "line_cap",
        "line_join",
        "line_miter_limit",
        "line_dash_array",
    ]
    fill_scope_style_keys = [
        "fill",
        "fill_color",
        "fill_alpha",
        "even_odd",
    ]
    scope_style_keys = line_scope_style_keys + fill_scope_style_keys
    excluded_subtypes = [
        Types.CLIPPED_SKETCH,
        Types.HELPLINES_SKETCH,
        Types.IMAGE_SKETCH,
        Types.LATEX_SKETCH,
        Types.MASKED_SKETCH,
        Types.TAG_SKETCH,
        Types.TEX_SKETCH,
    ]

    def get_scope_style(scope_group, scope_sketch):
        scope_keys = list(scope_group.style_data.keys())
        line_exceptions = []
        for style_key in line_scope_style_keys:
            if style_key not in scope_keys:
                line_exceptions.append(style_key)
        fill_exceptions = []
        for style_key in fill_scope_style_keys:
            if style_key not in scope_keys:
                fill_exceptions.append(style_key)
        scope_line_style = get_line_style_options(
            scope_sketch, exceptions=line_exceptions
        )
        scope_fill_style = ""
        if scope_sketch.subtype in d_shape_types:
            scope_fill_style = get_fill_style_options(
                scope_sketch,
                get_shape_type(scope_sketch),
                exceptions=fill_exceptions,
            )
        elif scope_sketch.subtype in [Types.ARC_SKETCH, Types.PATH_SKETCH]:
            scope_fill_style = get_fill_style_options(
                scope_sketch, "path", exceptions=fill_exceptions
            )
        return f"{scope_line_style} {scope_fill_style}".strip()

    def format_manual_scope_open(scope_group, scope_sketch):
        if scope_group.subtype == Types.CLIP_GROUP:
            return f'<g clip-path="url(#clippath_{scope_group.id})">'
        if scope_group.subtype == Types.MASK_GROUP:
            return f'<g mask="url(#{scope_group._mask_context_id})">'
        if scope_group.subtype == Types.SCOPE_GROUP:
            scope_style = get_scope_style(scope_group, scope_sketch)
            if scope_style:
                return f'<g style="{scope_style}">'
            return "<g>"
        return ""

    def format_auto_scope_open(scope_group, scope_sketch):
        scope_style = get_scope_style(scope_group, scope_sketch)
        if scope_style:
            return f'<g style="{scope_style}">'
        return "<g>"

    def is_svg_scope_eligible(sketch):
        if sketch_attrib(sketch, "tile_svg") is not None:
            return False
        if has_gradient(sketch):
            return False
        return True

    def render_sketches(sketches, ind, scope_groups=None):
        if scope_groups is None:
            scope_groups = []

        return render_svg_scope_loop(
            sketches,
            ind,
            scope_groups,
            scope_style_keys,
            excluded_subtypes,
            lambda sketch, local_ind, suppressed_style_keys: get_sketch_code(
                sketch, canvas, local_ind, suppressed_style_keys
            ),
            format_manual_scope_open,
            format_auto_scope_open,
            is_svg_scope_eligible,
        )

    def get_sketch_code(sketch, canvas, ind, suppressed_style_keys):
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
            child_sketches = []
            for sketch_list in sketch.sketches:
                child_sketches.extend(sketch_list)
            content = render_sketches(child_sketches, ind)
            code = f'<g clip-path="url(#{clippath_id})">\n{content}\n</g>'
        elif subtype == Types.MASKED_SKETCH:
            mask_id = f"mask_{id(sketch)}"
            child_sketches = []
            for sketch_list in sketch.sketches:
                child_sketches.extend(sketch_list)
            content = render_sketches(child_sketches, ind)
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
            code = draw_line_sketch(
                sketch, canvas, exceptions=suppressed_style_keys
            )
        elif subtype == Types.ARC_SKETCH:
            code = draw_arc_sketch(sketch, exceptions=suppressed_style_keys)
        elif subtype == Types.PATH_SKETCH:
            code = draw_path_sketch(sketch, exceptions=suppressed_style_keys)
        elif (
            draw_markers
            and sketch_attrib(sketch, "marker_type") == MarkerType.INDICES
        ) or indices:
            code = draw_shape_sketch_with_indices(
                sketch, exceptions=suppressed_style_keys
            )
        elif draw_markers:
            # Use marker rendering for shapes with markers enabled
            code = draw_shape_sketch_with_markers(
                sketch, exceptions=suppressed_style_keys
            )
        else:
            code = svg_shape(
                sketch, styles_dict, exceptions=suppressed_style_keys
            )

        sketch_dict = sketch_attrib(sketch, "__dict__")
        sketch_filter = sketch_attrib(sketch, "filter")
        if "filter" in sketch_dict and sketch_filter is not None:
            filter_id = sketch_filter.id
            code = f'<g filter="url(#{filter_id})">\n{code}\n</g>'
        return code

    page = canvas.active_page

    if page is None:
        raise ValueError("No active page found in the canvas.")

    sketches = page.sketches
    if page.back_color:
        code = [color_to_svg(page.back_color)]
    else:
        code = []
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
    rendered_code = render_sketches(sketches, 0, page.scope_groups)
    code.append(rendered_code)

    code = "\n".join(code)
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


def svg_shape(sketch, styles_dict, exceptions=None):
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
        sketch,
        style_shape_type,
        styles_dict,
        skip_fill=skip_fill_style,
        exceptions=exceptions,
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
        line_color = sketch_attrib(sketch, "line_color")
        if line_color is None:
            line_color = defaults["line_color"]
        outer_stroke = color_to_svg(check_color(line_color))
        fill_style = get_fill_style_options(
            sketch, style_shape_type, exceptions=exceptions
        )
        outer_style = f"stroke: {outer_stroke}; stroke-width: {outer_stroke_width}; {fill_style}"
        outer_element = (
            f"<{shape_type}\n"
            f'style="{outer_style}"{fill_rule_attr}{clip_attr}{mask_attr}\n'
            f"{coordinates}\n"
            f"/>"
        )
        double_color = sketch_attrib(sketch, "double_color")
        if double_color is None:
            double_color = defaults["double_color"]
        gap_stroke = color_to_svg(check_color(double_color))
        gap_style = f"stroke: {gap_stroke}; stroke-width: {double_distance}; fill: none;"
        gap_element = f'<{shape_type}\nstyle="{gap_style}"\n{coordinates}\n/>'
        return f"{outer_element}\n{gap_element}"

    class_attr = ""
    if style_class:
        class_attr = f' class= "{style_class}"'

    return f"""<{shape_type}
{class_attr}{fill_attr_str}{fill_rule_attr}{clip_attr}{mask_attr}
{coordinates}
/>"""


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
                if sketch_attrib(sketch, "subtype") in [
                    Types.CLIPPED_SKETCH,
                    Types.MASKED_SKETCH,
                ]:
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
        dict: Dictionary mapping stable sketch.id to sketch for shapes with markers
    """
    markers = {}

    def _collect_from_sketch(sketch):
        if sketch_attrib(sketch, "subtype") in [
            Types.CLIPPED_SKETCH,
            Types.MASKED_SKETCH,
        ]:
            for sketch_list in sketch.sketches:
                for child_sketch in sketch_list:
                    _collect_from_sketch(child_sketch)
            return
        if (
            sketch_attrib(sketch, "draw_markers")
            and sketch_attrib(sketch, "marker_type") != MarkerType.EMPTY
            and sketch_attrib(sketch, "marker_type") != MarkerType.INDICES
        ):
            markers[sketch.id] = sketch

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
            if (
                scope_group.subtype == Types.CLIP_GROUP
                and scope_group.mask is not None
            ):
                clippath_id = f"clippath_{scope_group.id}"
                defs_content.append(
                    generate_clippath_def(
                        None, scope_group.mask, clippath_id, canvas, styles_dict
                    )
                )

    # Generate mask definitions from scope_groups (MASK_GROUP)
    for page in canvas.pages:
        for scope_group in page.scope_groups:
            if (
                scope_group.subtype == Types.MASK_GROUP
                and scope_group.mask is not None
            ):
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
                if sketch_attrib(sketch, "subtype") in [
                    Types.CLIPPED_SKETCH,
                    Types.MASKED_SKETCH,
                ]:
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
            vertex[:2] for vertex in homogenize(region_corners) @ xform_matrix
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
    if canvas.page_size is None:
        if canvas.border is None:
            border_left = defaults["border"]
            border_bottom = defaults["border"]
            border_right = defaults["border"]
            border_top = defaults["border"]
        elif isinstance(canvas.border, (int, float)):
            border_left = canvas.border
            border_bottom = canvas.border
            border_right = canvas.border
            border_top = canvas.border
        elif (
            isinstance(canvas.border, (list, tuple)) and len(canvas.border) == 4
        ):
            border_left, border_bottom, border_right, border_top = canvas.border
        else:
            raise ValueError(
                "Canvas.border must be a positive numeric value or a tuple of 4 positive numeric values."
            )
    else:
        border_left = 0
        border_bottom = 0
        border_right = 0
        border_top = 0
    color = canvas.back_color
    if color is None:
        color = white
    styles_dict = get_styles_dict(canvas)
    styles = get_styles(canvas, styles_dict)
    defs = generate_defs(canvas, styles_dict)

    if not canvas.active_page.sketches or not vertices:
        issue_warning(
            "Canvas has no drawings/sketches. Writing empty SVG output."
        )
        if canvas.page_size is not None:
            width, height = canvas.page_size
            minx, miny = canvas.page_origin
        else:
            width = border_left + border_right
            height = border_bottom + border_top
            minx = -border_left
            miny = -border_bottom
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

    if canvas.page_size is not None:
        minx, miny, maxx, maxy = canvas.limits
        width = maxx - minx
        height = maxy - miny
    else:
        bbox = bounding_box(vertices)
        width = bbox.width + border_left + border_right
        height = bbox.height + border_bottom + border_top
        x_coords = []
        y_coords = []
        for vertex in vertices:
            x_coord, y_coord = vertex[:2]
            x_coords.append(x_coord)
            y_coords.append(y_coord)
        minx = min(x_coords) - border_left
        miny = min(y_coords) - border_bottom

    dy = 2 * miny + height

    # Check if canvas has limits that require clipping
    limits_clippath_id, _ = get_limits_clippath(canvas)
    canvas_mask_scope = None
    for sketch in reversed(canvas.active_page.sketches):
        if (
            "_canvas_mask_scope" in sketch.__dict__
            and sketch._canvas_mask_scope
        ):
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

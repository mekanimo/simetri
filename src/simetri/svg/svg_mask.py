"""SVG opacity-mask and clip-path definition helpers."""

import numpy as np

from ..colors.colors import Color
from ..core.all_enums import SvgMaskType, SvgUnits, Types
from ..geometry.bbox import bounding_box
from ..settings.settings import defaults
from .svg_colors import color_to_svg
from .svg_sketch_utils import get_coordinates, get_shape_type, sketch_attrib


def generate_mask_def(sketch, mask_shape, mask_id, canvas, styles_dict):
    """Generate SVG mask definition for a shape's mask property.

    Args:
        sketch: The shape that has mask property
        mask_shape: The shape/group used for masking
        mask_id: Unique ID for this mask
        canvas: The canvas object for property resolution
        styles_dict: Styles dictionary for rendering the mask shape

    Returns:
        str: SVG <mask> element
    """
    from ..canvas.draw import create_sketch  # noqa: PLC0415 — circular import

    def get_mask_stop(stop):
        stop_offset = f"{float(stop.offset) * 100}%"

        if isinstance(stop_color, Color):
            stop_color = color_to_svg(stop_color)

        return stop_offset, stop_color, stop.opacity

    def _stops_to_svg(stops, indent="      "):
        stop_lines = []
        has_color = False
        for stop in stops or []:
            offset, stop_color, stop_opacity = get_mask_stop(stop)
            has_color = stop_color is not None
            if stop_color != color_to_svg(defaults["stop_color"]):
                has_color = True
            stop_opacity_attr = (
                f' stop-opacity="{stop_opacity}"'
                if stop_opacity is not None
                else ""
            )
            stop_lines.append(
                f'{indent}<stop offset="{offset}" stop-color="{stop_color}"{stop_opacity_attr} />'
            )
        return "\n".join(stop_lines), has_color

    def _normalize_svg_units(value):
        if isinstance(value, SvgUnits):
            return value
        if value is None:
            return _normalize_svg_units(
                defaults.get("mask_units", SvgUnits.USER_SPACE_ON_USE.value)
            )
        text = str(value).strip()
        lowered = text.lower()
        if lowered in ("userspaceonuse", "usersapceonuse"):
            return SvgUnits.USER_SPACE_ON_USE
        if lowered == "objectboundingbox":
            return SvgUnits.OBJECT_BOUNDING_BOX
        if text == SvgUnits.USER_SPACE_ON_USE.value:
            return SvgUnits.USER_SPACE_ON_USE
        if text == SvgUnits.OBJECT_BOUNDING_BOX.value:
            return SvgUnits.OBJECT_BOUNDING_BOX
        return SvgUnits.USER_SPACE_ON_USE

    def _mask_bounds_in_user_space():
        if canvas.page_size is not None:
            x_min, y_min, x_max, y_max = canvas.limits
            return x_min, y_min, x_max - x_min, y_max - y_min

        if canvas._all_vertices:
            canvas_bbox = bounding_box(canvas._all_vertices)
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
                isinstance(canvas.border, (list, tuple))
                and len(canvas.border) == 4
            ):
                border_left, border_bottom, border_right, border_top = (
                    canvas.border
                )
            else:
                raise ValueError(
                    "Canvas.border must be a positive numeric value or a tuple of 4 positive numeric values."
                )
            x = canvas_bbox.southwest[0] - border_left
            y = canvas_bbox.southwest[1] - border_bottom
            width = canvas_bbox.width + border_left + border_right
            height = canvas_bbox.height + border_bottom + border_top
            return x, y, width, height

        mask_bbox = mask_shape.b_box
        return (
            mask_bbox.southwest[0],
            mask_bbox.southwest[1],
            mask_bbox.width,
            mask_bbox.height,
        )

    if mask_shape is None and has_mask_style(sketch):
        msk = sketch_attrib(sketch, "style").fill_style.mask_style

        mask_type = msk.mask_type
        units = msk.units
        spread_method = msk.spread_method
        transform = msk.transform
        stops = msk.stops
        mask_units = msk.mask_units
        mask_content_units = msk.mask_content_units
        mask_units = _normalize_svg_units(mask_units)
        mask_content_units = _normalize_svg_units(mask_content_units)

        transform_attr = (
            f' gradientTransform="{transform}"' if transform else ""
        )

        if mask_type == "linear":
            x1, y1 = msk.axis[0][:2]
            x2, y2 = msk.axis[1][:2]
            gradient_start = (
                f'    <linearGradient id="{mask_id}_gradient" x1="{x1}" y1="{y1}" '
                f'x2="{x2}" y2="{y2}" gradientUnits="{units}" '
                f'spreadMethod="{spread_method}"{transform_attr}>'
            )
            gradient_end = "    </linearGradient>"
        else:
            cx, cy = msk.center[:2]
            r = msk.radius
            fx, fy = msk.focal[:2]
            gradient_start = (
                f'    <radialGradient id="{mask_id}_gradient" cx="{cx}" cy="{cy}" '
                f'r="{r}" fx="{fx}" fy="{fy}" gradientUnits="{units}" '
                f'spreadMethod="{spread_method}"{transform_attr}>'
            )
            gradient_end = "    </radialGradient>"

        stops_str, has_color_stops = _stops_to_svg(stops, indent="      ")
        mask_type_attr = (
            SvgMaskType.LUMINANCE if has_color_stops else SvgMaskType.ALPHA
        )

        context_bbox = sketch_attrib(sketch, "_mask_context_bbox")
        bbox = sketch_attrib(sketch, "b_box")
        if context_bbox is not None:
            x, y, width, height = context_bbox
        elif bbox is not None:
            mask_bbox = bbox
            x = mask_bbox.southwest[0]
            y = mask_bbox.southwest[1]
            width = mask_bbox.width
            height = mask_bbox.height
        else:
            mask_bbox = bounding_box(
                np.array(sketch_attrib(sketch, "vertices"))
            )
            x = mask_bbox.southwest[0]
            y = mask_bbox.southwest[1]
            width = mask_bbox.width
            height = mask_bbox.height

        return (
            f'  <mask id="{mask_id}" maskUnits="{mask_units.value}" '
            f'maskContentUnits="{mask_content_units.value}" mask-type="{mask_type_attr.value}">\n'
            f"{gradient_start}\n"
            f"{stops_str}\n"
            f"{gradient_end}\n"
            f'    <rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'fill="url(#{mask_id}_gradient)" />\n'
            f"  </mask>"
        )

    def _mask_shape_svg(mask_sketch, fill_value="white", fill_opacity=None):
        shape_type = get_shape_type(mask_sketch)
        coordinates = get_coordinates(mask_sketch, shape_type)
        fill_opacity_attr = (
            f' fill-opacity="{fill_opacity}"'
            if fill_opacity is not None
            else ""
        )
        fill_rule_attr = ""
        if "even_odd" in mask_sketch.__dict__ and mask_sketch.even_odd:
            fill_rule_attr = ' fill-rule="evenodd"'

        return (
            f'<{shape_type} fill="{fill_value}"{fill_opacity_attr} '
            f'stroke="none"{fill_rule_attr} {coordinates} />'
        )

    if mask_shape.type == Types.MASK:
        mask_data = mask_shape
        mask_shape = mask_data.shape
        mask_opacity = mask_data.opacity
        mask_stops = mask_data.stops
        mask_axis = mask_data.axis
        if mask_stops is not None and mask_axis is None:
            mask_axis = defaults["mask_axis"]
        mask_units = _normalize_svg_units(None)
        mask_content_units = _normalize_svg_units(None)
    elif sketch.subtype == Types.MASK_SKETCH:
        mask_opacity = sketch.mask_opacity
        mask_stops = sketch.mask_stops
        mask_axis = sketch.mask_axis
        mask_units = _normalize_svg_units(sketch.mask_units)
        mask_content_units = _normalize_svg_units(sketch.mask_content_units)
    else:
        mask_opacity = sketch_attrib(sketch, "_mask_opacity")
        if mask_opacity is None:
            mask_opacity = 1.0
        mask_stops = sketch_attrib(sketch, "_mask_stops")
        mask_axis = sketch_attrib(sketch, "_mask_axis")
        mask_units = _normalize_svg_units(sketch_attrib(sketch, "_mask_units"))
        mask_content_units = _normalize_svg_units(
            sketch_attrib(sketch, "_mask_content_units")
        )

    gradient_opacity = mask_stops is not None

    if gradient_opacity:
        if hasattr(mask_axis, "start") and hasattr(mask_axis, "end"):
            axis_start = mask_axis.start
            axis_end = mask_axis.end
        else:
            axis_start, axis_end = mask_axis
        mask_bbox = mask_shape.b_box
        bbox_x = mask_bbox.southwest[0]
        bbox_y = mask_bbox.southwest[1]
        bbox_width = mask_bbox.width
        bbox_height = mask_bbox.height
        x1 = bbox_x + float(axis_start[0]) * bbox_width
        y1 = bbox_y + float(axis_start[1]) * bbox_height
        x2 = bbox_x + float(axis_end[0]) * bbox_width
        y2 = bbox_y + float(axis_end[1]) * bbox_height
        mask_x, mask_y, mask_width, mask_height = _mask_bounds_in_user_space()
        gradient_id = f"{mask_id}_opacity_gradient"
        if mask_shape.type == Types.GROUP:
            gradient_contents = []
            for shape in mask_shape.shapes:
                mask_sketch = create_sketch(shape, canvas)
                if mask_sketch:
                    gradient_contents.append(
                        _mask_shape_svg(
                            mask_sketch, fill_value=f"url(#{gradient_id})"
                        )
                    )
            gradient_mask_content = "\n      ".join(gradient_contents)
        else:
            mask_sketch = create_sketch(mask_shape, canvas)
            if mask_sketch:
                gradient_mask_content = _mask_shape_svg(
                    mask_sketch, fill_value=f"url(#{gradient_id})"
                )
            else:
                gradient_mask_content = ""

        stops_str, has_color_stops = _stops_to_svg(
            mask_stops, indent="        "
        )
        mask_type_attr = (
            SvgMaskType.LUMINANCE if has_color_stops else SvgMaskType.ALPHA
        )

        return (
            f'  <linearGradient id="{gradient_id}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" gradientUnits="userSpaceOnUse">\n'
            f"{stops_str}\n"
            f"  </linearGradient>\n"
            f'  <mask id="{mask_id}" maskUnits="{mask_units.value}" '
            f'maskContentUnits="{mask_content_units.value}" mask-type="{mask_type_attr.value}" '
            f'x="{mask_x}" y="{mask_y}" width="{mask_width}" height="{mask_height}">\n'
            f"    {gradient_mask_content}\n"
            f"  </mask>"
        )

    if mask_shape.type == Types.GROUP:
        mask_contents = []
        for shape in mask_shape.shapes:
            mask_sketch = create_sketch(shape, canvas)
            if mask_sketch:
                mask_contents.append(
                    _mask_shape_svg(mask_sketch, fill_opacity=mask_opacity)
                )
        mask_content = "\n    ".join(mask_contents)
    else:
        mask_sketch = create_sketch(mask_shape, canvas)
        if mask_sketch:
            mask_content = _mask_shape_svg(
                mask_sketch, fill_opacity=mask_opacity
            )
        else:
            mask_content = ""

    mask_x, mask_y, mask_width, mask_height = _mask_bounds_in_user_space()

    return (
        f'  <mask id="{mask_id}" maskUnits="{mask_units.value}" '
        f'maskContentUnits="{mask_content_units.value}" mask-type="{SvgMaskType.ALPHA.value}" '
        f'x="{mask_x}" y="{mask_y}" width="{mask_width}" height="{mask_height}">\n'
        f"    {mask_content}\n"
        f"  </mask>"
    )


def has_mask_style(sketch):
    """Check if a sketch has mask style configuration."""
    try:
        mask_style = sketch_attrib(sketch, "style").fill_style.mask_style
        if mask_style.stops is not None:
            return True
    except AttributeError:
        pass
    return sketch_attrib(sketch, "msk_stops") is not None


def _canvas_mask_scope_sketch(canvas):
    for sketch in reversed(canvas.active_page.sketches):
        if sketch.subtype == Types.MASK_SKETCH:
            return sketch
    return None

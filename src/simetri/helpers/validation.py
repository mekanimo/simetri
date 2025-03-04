"""Validation functions for the user entered argument values and kwargs."""

import re
from strenum import enum
from typing import Any, Dict


from numpy import ndarray
from ..graphics import all_enums, __version__
from ..graphics.all_enums import *
from ..colors import Color


# Validation functions. They return True if the value is valid, False otherwise.


class VersionConflict(Exception):
    """Exception raised for version conflicts."""


def check_version(required_version):
    """Check if the current version is compatible with the required version."""

    def version_value(str_version):
        digits = str_version.split(".")
        return int(digits[0]) * 100 + int(digits[1]) * 10 + int(digits[2])

    if version_value(required_version) > version_value(__version__):
        msg = (
            f"Version conflict: Minimum required version is {required_version}. "
            f"This version is {__version__}\n"
            "Please update the simetri package using: pip install -U simetri"
        )
        raise VersionConflict(msg)

    return True


def check_str(value: Any) -> bool:
    """Check if the value is a string."""
    return isinstance(value, str)


def check_int(value: Any) -> bool:
    """Check if the value is an integer."""
    return isinstance(value, int)


def check_number(number: Any) -> bool:
    """Check if the number is a valid number."""
    return isinstance(number, (int, float))


def check_color(color: Any) -> bool:
    """Check if the color is a valid color."""
    return isinstance(color, (Color, str, tuple, list, ndarray))


def check_dash_array(dash_array: Any) -> bool:
    """Check if the dash array is a list of numbers
    or predefined."""
    if dash_array in LineDashArray:
        res = True
    elif dash_array is None:
        res = True
    else:
        res = isinstance(dash_array, (list, tuple, ndarray)) and all(
            isinstance(x, (int, float)) for x in dash_array
        )

    return res


def check_bool(value: Any) -> bool:
    """Check if the value is a boolean.
    Boolean values need to be explicitly set to True or False.
    None is not a valid boolean value.
    """
    return isinstance(value, bool)


def check_enum(value: Any, enum: Any) -> bool:
    """Check if the value is a valid enum value."""
    return value in enum


def check_blend_mode(blend_mode: Any) -> bool:
    """Check if the blend mode is a valid blend mode."""
    return blend_mode in BlendMode


# def check_shade_type(shade: Any) -> bool:
#     """Check if the shader is a valid shader."""
#     return shade in ShadeType


# def check_back_style(back_style: Any) -> bool:
#     """Check if the back style is a valid back style."""
#     return back_style in BackStyle


def check_position(pos: Any) -> bool:
    """Check if the position is a valid position."""
    return (
        isinstance(pos, (list, tuple, ndarray))
        and len(pos) >= 2
        and all(isinstance(x, (int, float)) for x in pos)
    )


def check_points(points: Any) -> bool:
    """Check if the points are a valid list of points."""
    return isinstance(points, (list, tuple, ndarray)) and all(
        isinstance(x, (list, tuple, ndarray)) for x in points
    )


def check_xform_matrix(matrix: Any) -> bool:
    """Check if the matrix is a valid transformation matrix."""
    return isinstance(matrix, (list, tuple, ndarray))


def check_subtype(subtype: Any) -> bool:
    """This check is done in Shape class."""
    return True


def check_mask(mask: Any) -> bool:
    """This check is done in Batch class."""
    return mask.type == Types.Shape


def check_line_width(line_width: Any) -> bool:
    """Check if the line width is a valid line width."""
    if isinstance(line_width, (int, float)):
        res = line_width >= 0
    elif line_width in all_enums.LineWidth:
        res = True
    else:
        res = False

    return res


def check_anchor(anchor: Any) -> bool:
    """Check if the anchor is a valid anchor."""
    return anchor in Anchor


# Create a dictionary of enums for validation.
items = (item for item in all_enums.__dict__.items() if item[0][0] != "_")
# from https://stackoverflow.com/questions/1175208/elegant-python-function-to-
# convert-camelcase-to-snake-case
pattern = re.compile(r"(?<!^)(?=[A-Z])")  # convert CamelCase to snake_case
enum_map = {}
exclude = [
    "TypeAlias",
    "Union",
    "StrEnum",
    "CI_StrEnum",
    "Comparable",
    "IUC",
    "drawable_types",
    "shape_types",
]
for item in items:
    name = item[0]
    if isinstance(item[1], enum.EnumMeta) and name not in exclude:
        key_ = pattern.sub("_", name).lower()
        enum_map[key_] = item[1]

d_validators = {
    "alpha": check_number,
    "clip": check_bool,
    "color": check_color,
    "dist_tol": check_number,
    "dist_tol2": check_number,
    "double_distance": check_number,
    "double_lines": check_bool,
    "draw_fillets": check_bool,
    "draw_frame": check_bool,
    "draw_markers": check_bool,
    "even_odd_rule": check_bool,
    "fill": check_bool,
    "fill_alpha": check_number,
    "fill_blend_mode": check_blend_mode,
    "fill_color": check_color,
    "fillet_radius": check_number,
    "font_color": check_color,
    "font_family": check_str,
    "frame_inner_sep": check_number,
    "frame_inner_xsep": check_number,
    "frame_inner_ysep": check_number,
    "frame_min_height": check_number,
    "frame_min_width": check_number,
    "grid_alpha": check_number,
    "grid_back_color": check_color,
    "grid_line_color": check_color,
    "grid_line_width": check_number,
    "line_alpha": check_number,
    "line_blend_mode": check_blend_mode,
    "line_color": check_color,
    "line_dash_array": check_dash_array,
    "line_dash_phase": check_number,
    "line_miter_limit": check_number,
    "line_width": check_line_width,
    "marker_color": check_color,
    "marker_radius": check_number,
    "marker_size": check_number,
    "markers_only": check_bool,
    "mask": check_mask,
    "pattern_angle": check_number,
    "pattern_color": check_color,
    "pattern_distance": check_number,
    "pattern_line_width": check_number,
    "pattern_points": check_int,
    "pattern_radius": check_number,
    "pattern_xshift": check_number,
    "pattern_yshift": check_number,
    "points": check_points,
    "pos": check_position,
    "radius": check_number,
    "shade_axis_angle": check_number,
    "shade_color_wheel": check_number,
    "shade_color_wheel_black": check_bool,
    "shade_color_wheel_white": check_bool,
    "shade_bottom_color": check_color,
    "shade_inner_color": check_color,
    "shade_left_color": check_color,
    "shade_lower_left_color": check_color,
    "shade_lower_right_color": check_color,
    "shade_middle_color": check_color,
    "shade_outer_color": check_color,
    "shade_right_color": check_color,
    "shade_top_color": check_color,
    "shade_upper_left_color": check_color,
    "shade_upper_right_color": check_color,
    "smooth": check_bool,
    "stroke": check_bool,
    "xform_matrix": check_xform_matrix,
    "subtype": check_subtype,
    "text_alpha": check_number,
    "transparency_group": check_bool,
}


def validate_args(args: Dict[str, Any], valid_args: list[str]) -> None:
    """Validate the user entered arguments.
    Returns None. Raises a ValueError if an invalid key or value is found.
    """
    for key, value in args.items():
        if (key not in valid_args) and (key not in d_validators):
            raise ValueError(f"Invalid key: {key}")
        if key in d_validators:
            if not d_validators[key](value):
                raise ValueError(f"Invalid value for {key}: {value}")
        elif key in enum_map:
            if value not in enum_map[key]:
                raise ValueError(f"Invalid value for {key}: {value}")
        elif not d_validators[key](value):
            raise ValueError(f"Invalid value for {key}: {value}")

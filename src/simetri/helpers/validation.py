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


def check_version(required_version: str) -> bool:
    """
    Check if the current version is compatible with the required version.

    :param required_version: The required version as a string.
    :type required_version: str
    :raises VersionConflict: If the current version is lower than the required version.
    :return: True if the current version is compatible.
    :rtype: bool
    """
    def version_value(str_version: str) -> int:
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
    """
    Check if the value is a string.

    :param value: The value to check.
    :type value: Any
    :return: True if the value is a string, False otherwise.
    :rtype: bool
    """
    return isinstance(value, str)


def check_int(value: Any) -> bool:
    """
    Check if the value is an integer.

    :param value: The value to check.
    :type value: Any
    :return: True if the value is an integer, False otherwise.
    :rtype: bool
    """
    return isinstance(value, int)


def check_number(number: Any) -> bool:
    """
    Check if the number is a valid number.

    :param number: The number to check.
    :type number: Any
    :return: True if the number is a valid number, False otherwise.
    :rtype: bool
    """
    return isinstance(number, (int, float))


def check_color(color: Any) -> bool:
    """
    Check if the color is a valid color.

    :param color: The color to check.
    :type color: Any
    :return: True if the color is a valid color, False otherwise.
    :rtype: bool
    """
    return isinstance(color, (Color, str, tuple, list, ndarray))


def check_dash_array(dash_array: Any) -> bool:
    """
    Check if the dash array is a list of numbers or predefined.

    :param dash_array: The dash array to check.
    :type dash_array: Any
    :return: True if the dash array is valid, False otherwise.
    :rtype: bool
    """
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
    """
    Check if the value is a boolean.

    Boolean values need to be explicitly set to True or False.
    None is not a valid boolean value.

    :param value: The value to check.
    :type value: Any
    :return: True if the value is a boolean, False otherwise.
    :rtype: bool
    """
    return isinstance(value, bool)


def check_enum(value: Any, enum: Any) -> bool:
    """
    Check if the value is a valid enum value.

    :param value: The value to check.
    :type value: Any
    :param enum: The enum to check against.
    :type enum: Any
    :return: True if the value is a valid enum value, False otherwise.
    :rtype: bool
    """
    return value in enum


def check_blend_mode(blend_mode: Any) -> bool:
    """
    Check if the blend mode is a valid blend mode.

    :param blend_mode: The blend mode to check.
    :type blend_mode: Any
    :return: True if the blend mode is valid, False otherwise.
    :rtype: bool
    """
    return blend_mode in BlendMode


def check_position(pos: Any) -> bool:
    """
    Check if the position is a valid position.

    :param pos: The position to check.
    :type pos: Any
    :return: True if the position is valid, False otherwise.
    :rtype: bool
    """
    return (
        isinstance(pos, (list, tuple, ndarray))
        and len(pos) >= 2
        and all(isinstance(x, (int, float)) for x in pos)
    )


def check_points(points: Any) -> bool:
    """
    Check if the points are a valid list of points.

    :param points: The points to check.
    :type points: Any
    :return: True if the points are valid, False otherwise.
    :rtype: bool
    """
    return isinstance(points, (list, tuple, ndarray)) and all(
        isinstance(x, (list, tuple, ndarray)) for x in points
    )


def check_xform_matrix(matrix: Any) -> bool:
    """
    Check if the matrix is a valid transformation matrix.

    :param matrix: The matrix to check.
    :type matrix: Any
    :return: True if the matrix is valid, False otherwise.
    :rtype: bool
    """
    return isinstance(matrix, (list, tuple, ndarray))


def check_subtype(subtype: Any) -> bool:
    """
    This check is done in Shape class.

    :param subtype: The subtype to check.
    :type subtype: Any
    :return: True
    :rtype: bool
    """
    return True


def check_mask(mask: Any) -> bool:
    """
    This check is done in Batch class.

    :param mask: The mask to check.
    :type mask: Any
    :return: True if the mask is valid, False otherwise.
    :rtype: bool
    """
    return mask.type == Types.Shape


def check_line_width(line_width: Any) -> bool:
    """
    Check if the line width is a valid line width.

    :param line_width: The line width to check.
    :type line_width: Any
    :return: True if the line width is valid, False otherwise.
    :rtype: bool
    """
    if isinstance(line_width, (int, float)):
        res = line_width >= 0
    elif line_width in all_enums.LineWidth:
        res = True
    else:
        res = False

    return res


def check_anchor(anchor: Any) -> bool:
    """
    Check if the anchor is a valid anchor.

    :param anchor: The anchor to check.
    :type anchor: Any
    :return: True if the anchor is valid, False otherwise.
    :rtype: bool
    """
    return anchor in Anchor


def validate_args(args: Dict[str, Any], valid_args: list[str]) -> None:
    """
    Validate the user entered arguments.

    :param args: The arguments to validate.
    :type args: Dict[str, Any]
    :param valid_args: The list of valid argument keys.
    :type valid_args: list[str]
    :raises ValueError: If an invalid key or value is found.
    :return: None
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
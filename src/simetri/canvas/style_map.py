"""This module contains the Style classes used to set the style of shapes, lines, text,
and tags. Shape and Tag objects use the maps to create aliases for style attributes.

Examples:
shape.style.line_style.color is aliased by shape.line_color
tag.style.fill_style.pattern_style.line_style.width is aliased by tag.pattern_line_width
Documentation list all aliases for each style class.
"""
# to do: Change this so that IDEs can find the classes and methods.

from typing import List, Optional, Sequence, Union
from dataclasses import dataclass

from ..settings.settings import defaults
from ..graphics.common import get_unique_id, VOID
from ..graphics.all_enums import (
    Align,
    Anchor,
    BackStyle,
    BlendMode,
    FillMode,
    FontFamily,
    FontSize,
    FrameShape,
    LineCap,
    LineJoin,
    MarkerType,
    PatternType,
    ShadeType,
    Types,
)
from ..colors import Color


def _set_style_args(obj, attribs, exact=None, prefix=None):
    """Set the style arguments for the given object.

    Args:
        obj: The object to set the style arguments for.
        attribs: List of attributes to set.
        exact: List of exact attributes to set.
        prefix: Prefix to use for the attributes.
    """
    for attrib in attribs:
        if exact and attrib in exact:
            default = defaults.get(attrib, VOID)
            if default != VOID:
                setattr(obj, attrib, default)
        else:
            if prefix:
                setattr(obj, attrib, defaults[f"{prefix}_{attrib}"])
            else:
                default = defaults.get(attrib, VOID)
                if default != VOID:
                    setattr(obj, attrib, default)


def _get_style_attribs(style: Types.STYLE, prefix: str = None, exact: list = None, exclude: list = None) -> List[str]:
    """Get the list of attributes from the given Style object.

    Args:
        style (Types.STYLE): The style object to get attributes from.
        prefix (str, optional): The prefix to use for the attributes. Defaults to None.
        exact (list, optional): List of exact attributes to include. Defaults to None.
        exclude (list, optional): List of attributes to exclude. Defaults to None.

    Returns:
        List[str]: List of attributes.
    """
    attribs = style.__dict__.keys()
    res = []
    for attrib in attribs:
        if attrib in exclude:
            continue
        if attrib in exact:
            res.append(attrib)
        else:
            res.append(f"{prefix}_{attrib}")
    return res


@dataclass
class FontStyle:
    """FontStyle is used to set the font, color, and style of text.

    Attributes:
        font_family (str): The font family.
        color (Color): The color of the font.
        family (Union[FontFamily, str]): The font family.
        size (Union[FontSize, float]): The size of the font.
        bold (bool): Whether the font is bold.
        italic (bool): Whether the font is italic.
        small_caps (bool): Whether the font uses small caps.
        old_style_nums (bool): Whether the font uses old style numbers.
        overline (bool): Whether the font has an overline.
        strike_through (bool): Whether the font has a strike through.
        underline (bool): Whether the font is underlined.
        blend_mode (BlendMode): The blend mode of the font.
        alpha (float): The alpha value of the font.
    """

    font_family: str = None
    color: Color = None
    family: Union[FontFamily, str] = None
    size: Union[FontSize, float] = None
    bold: bool = None
    italic: bool = None
    small_caps: bool = None
    old_style_nums: bool = None
    overline: bool = None
    strike_through: bool = None
    underline: bool = None
    blend_mode: BlendMode = None
    alpha: float = None

    def __post_init__(self):
        """Initialize the FontStyle object."""
        exact = [
            "bold",
            "italic",
            "small_caps",
            "old_style_nums",
            "overline",
            "strike_through",
            "underline",
            "draw_frame",
            "font_family",
        ]
        exclude = []

        _style_init(
            self,
            exact=exact,
            exclude=exclude,
            prefix="font",
            subtype=Types.FONT_STYLE,
        )
        self._exact = exact
        self._exclude = exclude


@dataclass
class GridStyle:
    """GridStyle is used to set the grid color, alpha, width, and pattern.

    Attributes:
        line_color (Color): The color of the grid lines.
        line_width (float): The width of the grid lines.
        alpha (float): The alpha value of the grid.
        back_color (Color): The background color of the grid.
    """

    line_color: Color = None
    line_width: float = None
    alpha: float = None
    # width: float = None
    # height: float = None
    back_color: Color = None

    def __post_init__(self):
        """Initialize the GridStyle object."""
        exact = []
        exclude = []

        _style_init(
            self,
            exact=exact,
            exclude=exclude,
            prefix="grid",
            subtype=Types.GRID_STYLE,
        )
        self._exact = exact
        self._exclude = exclude

    def __str__(self):
        """Return a string representation of the GridStyle object."""
        return f"GridStyle: {self.id}"

    def __repr__(self):
        """Return a string representation of the GridStyle object."""
        return f"GridStyle: {self.id}"

    def _get_attributes(self):
        """Get the attributes of the GridStyle object."""
        attribs = [x for x in self.__dict__ if not x.startswith("_")]
        res = []
        for attrib in attribs:
            if attrib in self._exact:
                res.append(attrib)
            else:
                res.append(f"grid_{attrib}")


@dataclass
class MarkerStyle:
    """MarkerStyle is used to set the marker type, size, and color of a shape.

    Attributes:
        marker_type (MarkerType): The type of the marker.
        size (float): The size of the marker.
        color (Color): The color of the marker.
        radius (float): The radius of the marker.
    """

    marker_type: MarkerType = None
    size: float = None
    color: Color = None
    radius: float = None

    def __post_init__(self):
        """Initialize the MarkerStyle object."""
        exact = ["marker_type"]
        exclude = []
        _style_init(
            self,
            exact=exact,
            exclude=exclude,
            prefix="marker",
            subtype=Types.MARKER_STYLE,
        )
        self._exact = exact
        self._exclude = exclude

    def __str__(self):
        """Return a string representation of the MarkerStyle object."""
        return f"Marker: {self.type}"


@dataclass
class LineStyle:
    """LineStyle is used to set the line color, alpha, width, and pattern of a shape.

    Attributes:
        color (Color): The color of the line.
        alpha (float): The alpha value of the line.
        width (int): The width of the line.
        dash_array (Optional[Sequence[float]]): The dash array of the line.
        dash_phase (float): The dash phase of the line.
        cap (LineCap): The cap style of the line.
        join (LineJoin): The join style of the line.
        miter_limit (float): The miter limit of the line.
        fillet_radius (float): The fillet radius of the line.
        marker_style (MarkerStyle): The marker style of the line.
        smooth (bool): Whether the line is smooth.
        stroke (bool): Whether the line is stroked.
        draw_markers (bool): Whether to draw markers on the line.
        draw_fillets (bool): Whether to draw fillets on the line.
        markers_only (bool): Whether to draw only markers on the line.
        double_lines (bool): Whether to draw double lines.
        double_distance (float): The distance between double lines.
    """

    # To do: Add support for arrows
    color: Color = None
    alpha: float = None
    width: int = None
    dash_array: Optional[Sequence[float]] = None
    dash_phase: float = None
    cap: LineCap = None
    join: LineJoin = None
    miter_limit: float = None
    fillet_radius: float = None
    marker_style: MarkerStyle = None
    smooth: bool = None
    stroke: bool = None
    draw_markers: bool = None
    draw_fillets: bool = None
    markers_only: bool = None
    double_lines: bool = None
    double_distance: float = None

    def __post_init__(self):
        """Initialize the LineStyle object."""
        exact = [
            "smooth",
            "stroke",
            "fillet_radius",
            "draw_fillets",
            "draw_markers",
            "markers_only",
            "double",
            "double_distance",
            "double_lines",
        ]
        exclude = ["marker_style"]
        _style_init(
            self, exact, exclude, prefix="line", subtype=Types.LINE_STYLE
        )
        self._exact = exact
        self._exclude = exclude
        self.marker_style = MarkerStyle()

    def __str__(self):
        """Return a string representation of the LineStyle object."""
        return f"LineStyle: {self.id}"


def _style_init(style, exact=None, exclude=None, prefix="", subtype=None):
    """Initialize the style object.

    Args:
        style: The style object to initialize.
        exact: List of exact attributes to include.
        exclude: List of attributes to exclude.
        prefix: Prefix to use for the attributes.
        subtype: The subtype of the style.
    """
    if exclude is None:
        exclude = []
    if exact is None:
        exact = []
    attribs = [x for x in style.__dict__ if x not in exclude]
    _set_style_args(style, attribs, exact, prefix=prefix)

    style.attribs = _get_style_attribs(style, prefix=prefix, exact=exact, exclude=exclude)
    style.id = get_unique_id(style)
    style.type = Types.STYLE
    style.subtype = subtype


@dataclass  # used for creating patterns
class PatternStyle:
    """PatternStyle is used to set the pattern type, color, distance, angle, shift, line width, radius, and points.

    Attributes:
        pattern_type (PatternType): The type of the pattern.
        color (Color): The color of the pattern.
        distance (float): The distance between pattern elements.
        angle (float): The angle of the pattern.
        x_shift (float): The x-axis shift of the pattern.
        y_shift (float): The y-axis shift of the pattern.
        line_width (float): The line width of the pattern.
        radius (float): The radius of the pattern elements.
        points (int): The number of points in the pattern.
    """

    pattern_type: PatternType = None  # LINES, HATCH, DOTS, STARS
    color: Color = None
    distance: float = None
    angle: float = None
    x_shift: float = None
    y_shift: float = None
    line_width: float = None
    radius: float = None  # used for dots, stars
    points: int = None  # number of petals. Used for stars

    def __post_init__(self):
        """Initialize the PatternStyle object."""
        exact = ["stroke", "pattern_type"]
        exclude = []
        _style_init(
            self,
            exact=exact,
            exclude=exclude,
            prefix="pattern",
            subtype=Types.PATTERN_STYLE,
        )
        self._exact = exact
        self._exclude = exclude

    def __str__(self):
        """Return a string representation of the PatternStyle object."""
        return f"Pattern: {self.type}"


# \usetikzlibrary{shadings}
@dataclass
class ShadeStyle:
    """ShadeStyle uses TikZ shading library to create colors with gradients.

    Attributes:
        shade_type (ShadeType): The type of the shade.
        axis_angle (float): The axis angle of the shade.
        ball_color (Color): The color of the ball.
        bottom_color (Color): The bottom color of the shade.
        color_wheel (Color): The color wheel of the shade.
        color_wheel_black (bool): Whether the color wheel includes black.
        color_wheel_white (bool): Whether the color wheel includes white.
        inner_color (Color): The inner color of the shade.
        left_color (Color): The left color of the shade.
        lower_left_color (Color): The lower left color of the shade.
        lower_right_color (Color): The lower right color of the shade.
        middle_color (Color): The middle color of the shade.
        outer_color (Color): The outer color of the shade.
        right_color (Color): The right color of the shade.
        top_color (Color): The top color of the shade.
        upper_left_color (Color): The upper left color of the shade.
        upper_right_color (Color): The upper right color of the shade.
    """

    shade_type: ShadeType = None
    axis_angle: float = None
    ball_color: Color = None
    bottom_color: Color = None
    color_wheel: Color = None
    color_wheel_black: bool = None
    color_wheel_white: bool = None
    inner_color: Color = None
    left_color: Color = None
    lower_left_color: Color = None
    lower_right_color: Color = None
    middle_color: Color = None
    outer_color: Color = None
    right_color: Color = None
    top_color: Color = None
    upper_left_color: Color = None
    upper_right_color: Color = None

    def __post_init__(self):
        """Initialize the ShadeStyle object."""
        exact = [
            "shade_type",
            "axis_angle",
            "color_wheel_black",
            "color_wheel_white",
            "top_color",
            "bottom_color",
            "left_color",
            "right_color",
            "middle_color",
            "inner_color",
            "outer_color",
            "upper_left_color",
            "upper_right_color",
            "lower_left_color",
            "lower_right_color",
            "color_wheel",
        ]
        exact = ["shade_type"]
        exclude = []
        _style_init(
            self,
            exact=exact,
            exclude=exclude,
            prefix="shade",
            subtype=Types.SHADE_STYLE,
        )
        self._exact = exact
        self._exclude = exclude


@dataclass
class FillStyle:
    """FillStyle is used to set the fill color, alpha, and pattern of a shape.

    Attributes:
        color (Color): The fill color.
        alpha (float): The alpha value of the fill.
        fill (bool): Whether the shape is filled.
        back_style (BackStyle): The back style of the fill.
        mode (FillMode): The fill mode.
        pattern_style (PatternStyle): The pattern style of the fill.
        shade_style (ShadeStyle): The shade style of the fill.
        grid_style (GridStyle): The grid style of the fill.
    """

    color: Color = None
    alpha: float = None
    fill: bool = None
    back_style: BackStyle = None
    mode: FillMode = None
    pattern_style: PatternStyle = None
    shade_style: ShadeStyle = None
    grid_style: GridStyle = None

    def __post_init__(self):
        """Initialize the FillStyle object."""
        self.shade_style = ShadeStyle()
        self.grid_style = GridStyle()
        self.pattern_style = PatternStyle()
        exact = ["fill", "back_style"]
        exclude = ["pattern_style", "shade_style", "grid_style"]
        _style_init(
            self, exact, exclude, prefix="fill", subtype=Types.FILL_STYLE
        )
        self._exact = exact
        self._exclude = exclude

    def __str__(self):
        """Return a string representation of the FillStyle object."""
        return f"FillStyle: {self.id}"

    def __repr__(self):
        """Return a string representation of the FillStyle object."""
        return f"FillStyle: {self.id}"

    def _get_attributes(self):
        """Get the attributes of the FillStyle object."""
        attribs = [x for x in self.__dict__ if not x.startswith("_")]
        res = []
        for attrib in attribs:
            if attrib in self._exact:
                res.append(attrib)
            else:
                res.append(f"fill_{attrib}")


@dataclass
class ShapeStyle:
    """ShapeStyle is used to set the fill and line style of a shape.

    Attributes:
        line_style (LineStyle): The line style of the shape.
        fill_style (FillStyle): The fill style of the shape.
        alpha (float): The alpha value of the shape.
    """

    line_style: LineStyle = None
    fill_style: FillStyle = None
    alpha: float = None

    def __post_init__(self):
        """Initialize the ShapeStyle object."""
        self.line_style = LineStyle()
        self.fill_style = FillStyle()
        self.marker_style = MarkerStyle()
        self.alpha = defaults["alpha"]
        exact = ["alpha"]
        exclude = ["line_style", "fill_style"]
        _style_init(
            self,
            exact=exact,
            exclude=exclude,
            prefix="",
            subtype=Types.SHAPE_STYLE,
        )
        self._exact = exact
        self._exclude = exclude

    def __str__(self):
        """Return a string representation of the ShapeStyle object."""
        return f"ShapeStyle: {self.id}"

    def __repr__(self):
        """Return a string representation of the ShapeStyle object."""
        return f"ShapeStyle: {self.id}"


@dataclass
class FrameStyle:
    """FrameStyle is used to set the frame shape, line style, fill style, and size of a shape.

    Attributes:
        shape (FrameShape): The shape of the frame.
        line_style (LineStyle): The line style of the frame.
        fill_style (FillStyle): The fill style of the frame.
        inner_sep (float): The inner separation of the frame.
        inner_xsep (float): The inner x-axis separation of the frame.
        inner_ysep (float): The inner y-axis separation of the frame.
        outer_sep (float): The outer separation of the frame.
        min_width (float): The minimum width of the frame.
        min_height (float): The minimum height of the frame.
        min_size (float): The minimum size of the frame.
        alpha (float): The alpha value of the frame.
    """

    shape: FrameShape = None
    line_style: LineStyle = None
    fill_style: FillStyle = None
    inner_sep: float = None
    inner_xsep: float = None
    inner_ysep: float = None
    outer_sep: float = None
    min_width: float = None
    min_height: float = None
    min_size: float = None
    alpha: float = None

    def __post_init__(self):
        """Initialize the FrameStyle object."""
        self.line_style = LineStyle()
        self.fill_style = FillStyle()
        exact = []
        exclude = ["line_style", "fill_style"]
        _style_init(
            self,
            exact=exact,
            exclude=exclude,
            prefix="frame",
            subtype=Types.FRAME_STYLE,
        )
        self._exact = exact
        self._exclude = exclude


@dataclass
class TagStyle:
    """TagStyle is used to set the font, color, and style of tag objects.

    Attributes:
        align (Align): The alignment of the tag.
        alpha (float): The alpha value of the tag.
        bold (bool): Whether the tag is bold.
        italic (bool): Whether the tag is italic.
        anchor (Anchor): The anchor of the tag.
        blend_mode (BlendMode): The blend mode of the tag.
        draw_frame (bool): Whether to draw a frame around the tag.
        font_style (FontStyle): The font style of the tag.
        frame_style (FrameStyle): The frame style of the tag.
        text_width (float): The text width of the tag.
    """

    align: Align = None
    alpha: float = None
    bold: bool = None
    italic: bool = None
    anchor: Anchor = None
    blend_mode: BlendMode = None
    draw_frame: bool = None
    font_style: FontStyle = None
    frame_style: FrameStyle = None
    text_width: float = None

    def __post_init__(self):
        """Initialize the TagStyle object."""
        self.font_style = FontStyle()
        self.frame_style = FrameStyle()
        self.alpha = defaults["tag_alpha"]
        self.bold = defaults["bold"]
        self.italic = defaults["italic"]
        self.align = defaults["tag_align"]
        self.blend_mode = defaults["tag_blend_mode"]
        self.text_width = defaults["text_width"]
        exact = [
            "alpha",
            "blend_mode",
            "draw_frame",
            "anchor",
            "bold",
            "italic",
            "text_width"
        ]
        exclude = ["font_style", "frame_style"]

        _style_init(
            self,
            exact=exact,
            exclude=exclude,
            prefix="tag",
            subtype=Types.TAG_STYLE,
        )
        self._exact = exact
        self._exclude = exclude

    def __str__(self):
        """Return a string representation of the TagStyle object."""
        return f"TagStyle: {self.id}"

    def __repr__(self):
        """Return a string representation of the TagStyle object."""
        return f"TagStyle: {self.id}"


# frame_style_map = {}

frame_style_map = {
    'alpha': ('frame_style', 'alpha'),
    'back_style': ('frame_style.fill_style', 'back_style'),
    'double_distance': ('frame_style.line_style', 'double_distance'),
    'double_lines': ('frame_style.line_style', 'double_lines'),
    'draw_fillets': ('frame_style.line_style', 'draw_fillets'),
    'draw_markers': ('frame_style.line_style', 'draw_markers'),
    'fill': ('frame_style.fill_style', 'fill'),
    'fill_alpha': ('frame_style.fill_style', 'alpha'),
    'fill_color': ('frame_style.fill_style', 'color'),
    'fill_mode': ('frame_style.fill_style', 'mode'),
    'fillet_radius': ('frame_style.line_style', 'fillet_radius'),
    'inner_sep': ('frame_style', 'inner_sep'),
    'line_alpha': ('frame_style.line_style', 'alpha'),
    'line_cap': ('frame_style.line_style', 'cap'),
    'line_color': ('frame_style.line_style', 'color'),
    'line_dash_array': ('frame_style.line_style', 'dash_array'),
    'line_dash_phase': ('frame_style.line_style', 'dash_phase'),
    'line_join': ('frame_style.line_style', 'join'),
    'line_miter_limit': ('frame_style.line_style', 'miter_limit'),
    'line_width': ('frame_style.line_style', 'width'),
    'markers_only': ('frame_style.line_style', 'markers_only'),
    'min_height': ('frame_style', 'min_height'),
    'min_size': ('frame_style', 'min_size'),
    'min_width': ('frame_style', 'min_width'),
    'outer_sep': ('frame_style', 'outer_sep'),
    'shape': ('frame_style', 'shape'),
    'smooth': ('frame_style.line_style', 'smooth'),
    'stroke': ('frame_style.line_style', 'stroke'),
}


def _set_frame_style_alias_map(debug=False):
    """Set the frame style alias map.

    Args:
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        dict: The frame style alias map.
    """
    line_style = LineStyle()
    fill_style = FillStyle()

    styles = [line_style, fill_style]
    paths = ["frame_style.line_style", "frame_style.fill_style"]
    prefixes = ["line", "fill"]

    _set_style_alias_map(frame_style_map, styles, paths, prefixes, debug=debug)
    frame_style_map["shape"] = ("frame_style", "shape")
    frame_style_map["inner_sep"] = ("frame_style", "inner_sep")
    frame_style_map["outer_sep"] = ("frame_style", "outer_sep")
    frame_style_map["min_width"] = ("frame_style", "min_width")
    frame_style_map["min_height"] = ("frame_style", "min_height")
    frame_style_map["min_size"] = ("frame_style", "min_size")
    frame_style_map["alpha"] = ("frame_style", "alpha")

    return frame_style_map


# marker_style_map = {}

marker_style_map = {
    'back_style': ('marker_style.fill_style', 'back_style'),
    'double_distance': ('marker_style.line_style', 'double_distance'),
    'double_lines': ('marker_style.line_style', 'double_lines'),
    'draw_fillets': ('marker_style.line_style', 'draw_fillets'),
    'draw_markers': ('marker_style.line_style', 'draw_markers'),
    'fill': ('marker_style.fill_style', 'fill'),
    'fill_alpha': ('marker_style.fill_style', 'alpha'),
    'fill_color': ('marker_style.fill_style', 'color'),
    'fill_mode': ('marker_style.fill_style', 'mode'),
    'fillet_radius': ('marker_style.line_style', 'fillet_radius'),
    'line_alpha': ('marker_style.line_style', 'alpha'),
    'line_cap': ('marker_style.line_style', 'cap'),
    'line_color': ('marker_style.line_style', 'color'),
    'line_dash_array': ('marker_style.line_style', 'dash_array'),
    'line_dash_phase': ('marker_style.line_style', 'dash_phase'),
    'line_join': ('marker_style.line_style', 'join'),
    'line_miter_limit': ('marker_style.line_style', 'miter_limit'),
    'line_width': ('marker_style.line_style', 'width'),
    'marker_alpha': ('marker_style', 'alpha'),
    'marker_radius': ('marker_style', 'radius'),
    'marker_size': ('marker_style', 'size'),
    'marker_type': ('marker_style', 'marker_type'),
    'markers_only': ('marker_style.line_style', 'markers_only'),
    'smooth': ('marker_style.line_style', 'smooth'),
    'stroke': ('marker_style.line_style', 'stroke'),
}

def _set_marker_style_alias_map(debug=False):
    """Set the marker style alias map.

    Args:
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        dict: The marker style alias map.
    """
    line_style = LineStyle()
    fill_style = FillStyle()

    styles = [line_style, fill_style]
    paths = ["marker_style.line_style", "marker_style.fill_style"]
    prefixes = ["line", "fill"]

    _set_style_alias_map(marker_style_map, styles, paths, prefixes, debug=debug)

    marker_style_map["marker_type"] = ("marker_style", "marker_type")
    marker_style_map["marker_alpha"] = ("marker_style", "alpha")
    marker_style_map["marker_size"] = ("marker_style", "size")
    marker_style_map["marker_radius"] = ("marker_style", "radius")

    return marker_style_map


# tag_style_map = {}
tag_style_map = {
    'align': ('style', 'align'),
    'alpha': ('style', 'alpha'),
    'back_color': ('style.frame_style.fill_style', 'color'),
    'back_style': ('style.frame_style.fill_style', 'back_style'),
    'blend_mode': ('style', 'blend_mode'),
    'bold': ('style.font_style', 'bold'),
    'double_distance': ('style.frame_style.line_style', 'double_distance'),
    'double_lines': ('style.frame_style.line_style', 'double_lines'),
    'draw_fillets': ('style.frame_style.line_style', 'draw_fillets'),
    'draw_frame': ('style', 'draw_frame'),
    'draw_markers': ('style.frame_style.line_style', 'draw_markers'),
    'fill': ('style.frame_style.fill_style', 'fill'),
    'fill_alpha': ('style.frame_style.fill_style', 'alpha'),
    'fill_color': ('style.frame_style.fill_style', 'color'),
    'fill_mode': ('style.frame_style.fill_style', 'mode'),
    'fillet_radius': ('style.frame_style.line_style', 'fillet_radius'),
    'font_alpha': ('style.font_style', 'alpha'),
    'font_blend_mode': ('style.font_style', 'blend_mode'),
    'font_color': ('style.font_style', 'color'),
    'font_family': ('style.font_style', 'font_family'),
    'font_size': ('style.font_style', 'size'),
    'frame_alpha': ('style.frame_style', 'alpha'),
    'frame_inner_sep': ('style.frame_style', 'inner_sep'),
    'frame_inner_xsep': ('style.frame_style', 'inner_xsep'),
    'frame_inner_ysep': ('style.frame_style', 'inner_ysep'),
    'frame_min_height': ('style.frame_style', 'min_height'),
    'frame_min_size': ('style.frame_style', 'min_size'),
    'frame_min_width': ('style.frame_style', 'min_width'),
    'frame_outer_sep': ('style.frame_style', 'outer_sep'),
    'frame_shape': ('style.frame_style', 'shape'),
    'grid_alpha': ('style.frame_style.fill_style.grid_style', 'alpha'),
    'grid_back_color': ('style.frame_style.fill_style.grid_style', 'back_color'),
    'grid_line_color': ('style.frame_style.fill_style.grid_style', 'line_color'),
    'grid_line_width': ('style.frame_style.fill_style.grid_style', 'line_width'),
    'italic': ('style.font_style', 'italic'),
    'line_alpha': ('style.frame_style.line_style', 'alpha'),
    'line_cap': ('style.frame_style.line_style', 'cap'),
    'line_color': ('style.frame_style.line_style', 'color'),
    'line_dash_array': ('style.frame_style.line_style', 'dash_array'),
    'line_dash_phase': ('style.frame_style.line_style', 'dash_phase'),
    'line_join': ('style.frame_style.line_style', 'join'),
    'line_miter_limit': ('style.frame_style.line_style', 'miter_limit'),
    'line_width': ('style.frame_style.line_style', 'width'),
    'marker_color': ('style.frame_style.line_style.marker_style', 'color'),
    'marker_radius': ('style.frame_style.line_style.marker_style', 'radius'),
    'marker_size': ('style.frame_style.line_style.marker_style', 'size'),
    'marker_type': ('style.frame_style.line_style.marker_style', 'marker_type'),
    'markers_only': ('style.frame_style.line_style', 'markers_only'),
    'old_style_nums': ('style.font_style', 'old_style_nums'),
    'overline': ('style.font_style', 'overline'),
    'pattern_angle': ('style.frame_style.fill_style.pattern_style', 'angle'),
    'pattern_color': ('style.frame_style.fill_style.pattern_style', 'color'),
    'pattern_distance': ('style.frame_style.fill_style.pattern_style', 'distance'),
    'pattern_line_width': ('style.frame_style.fill_style.pattern_style', 'line_width'),
    'pattern_points': ('style.frame_style.fill_style.pattern_style', 'points'),
    'pattern_radius': ('style.frame_style.fill_style.pattern_style', 'radius'),
    'pattern_type': ('style.frame_style.fill_style.pattern_style', 'pattern_type'),
    'pattern_x_shift': ('style.frame_style.fill_style.pattern_style', 'x_shift'),
    'pattern_y_shift': ('style.frame_style.fill_style.pattern_style', 'y_shift'),
    'shade_axis_angle': ('style.frame_style.fill_style.shade_style', 'axis_angle'),
    'shade_ball_color': ('style.frame_style.fill_style.shade_style', 'ball_color'),
    'shade_bottom_color': ('style.frame_style.fill_style.shade_style', 'bottom_color'),
    'shade_color_wheel': ('style.frame_style.fill_style.shade_style', 'color_wheel'),
    'shade_color_wheel_black': ('style.frame_style.fill_style.shade_style', 'color_wheel_black'),
    'shade_color_wheel_white': ('style.frame_style.fill_style.shade_style', 'color_wheel_white'),
    'shade_inner_color': ('style.frame_style.fill_style.shade_style', 'inner_color'),
    'shade_left_color': ('style.frame_style.fill_style.shade_style', 'left_color'),
    'shade_lower_left_color': ('style.frame_style.fill_style.shade_style', 'lower_left_color'),
    'shade_lower_right_color': ('style.frame_style.fill_style.shade_style', 'lower_right_color'),
    'shade_middle_color': ('style.frame_style.fill_style.shade_style', 'middle_color'),
    'shade_outer_color': ('style.frame_style.fill_style.shade_style', 'outer_color'),
    'shade_right_color': ('style.frame_style.fill_style.shade_style', 'right_color'),
    'shade_top_color': ('style.frame_style.fill_style.shade_style', 'top_color'),
    'shade_type': ('style.frame_style.fill_style.shade_style', 'shade_type'),
    'shade_upper_left_color': ('style.frame_style.fill_style.shade_style', 'upper_left_color'),
    'shade_upper_right_color': ('style.frame_style.fill_style.shade_style', 'upper_right_color'),
    'small_caps': ('style.font_style', 'small_caps'),
    'smooth': ('style.frame_style.line_style', 'smooth'),
    'strike_through': ('style.font_style', 'strike_through'),
    'stroke': ('style.frame_style.line_style', 'stroke'),
    'underline': ('style.font_style', 'underline'),
}


def _set_tag_style_alias_map(debug=False):
    """Set the tag style alias map.

    Args:
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        dict: The tag style alias map.
    """
    font_style = FontStyle()
    frame_style = FrameStyle()
    fill_style = FillStyle()
    line_style = LineStyle()
    pattern_style = PatternStyle()
    shade_style = ShadeStyle()
    grid_style = GridStyle()
    marker_style = MarkerStyle()

    styles = [
        font_style,
        line_style,
        fill_style,
        marker_style,
        frame_style,
        pattern_style,
        shade_style,
        grid_style,
    ]
    paths = [
        "style.font_style",
        "style.frame_style.line_style",
        "style.frame_style.fill_style",
        "style.frame_style.line_style.marker_style",
        "style.frame_style",
        "style.frame_style.fill_style.pattern_style",
        "style.frame_style.fill_style.shade_style",
        "style.frame_style.fill_style.grid_style",
    ]
    prefixes = ["font", "line", "fill", "marker", "frame", "pattern", "shade", "grid"]

    _set_style_alias_map(tag_style_map, styles, paths, prefixes, debug=debug)
    tag_style_map["alpha"] = ("style", "alpha")
    tag_style_map["align"] = ("style", "align")
    tag_style_map["blend_mode"] = ("style", "blend_mode")
    tag_style_map["draw_frame"] = ("style", "draw_frame")
    tag_style_map["back_color"] = ("style.frame_style.fill_style", "color")
    tag_style_map["align"] = ("style", "align")
    return tag_style_map


# fill_style_map = {}

fill_style_map = {
    'alpha': ('fill_style', 'alpha'),
    'color': ('fill_style', 'color'),
    'grid_alpha': ('fill_style.grid_style', 'alpha'),
    'grid_back_color': ('fill_style.grid_style', 'back_color'),
    'grid_line_color': ('fill_style.grid_style', 'line_color'),
    'grid_line_width': ('fill_style.grid_style', 'line_width'),
    'mode': ('fill_style', 'mode'),
    'pattern_angle': ('fill_style.pattern_style', 'angle'),
    'pattern_color': ('fill_style.pattern_style', 'color'),
    'pattern_distance': ('fill_style.pattern_style', 'distance'),
    'pattern_line_width': ('fill_style.pattern_style', 'line_width'),
    'pattern_points': ('fill_style.pattern_style', 'points'),
    'pattern_radius': ('fill_style.pattern_style', 'radius'),
    'pattern_type': ('fill_style.pattern_style', 'pattern_type'),
    'pattern_x_shift': ('fill_style.pattern_style', 'x_shift'),
    'pattern_y_shift': ('fill_style.pattern_style', 'y_shift'),
    'shade_axis_angle': ('fill_style.shade_style', 'axis_angle'),
    'shade_ball_color': ('fill_style.shade_style', 'ball_color'),
    'shade_bottom_color': ('fill_style.shade_style', 'bottom_color'),
    'shade_color_wheel': ('fill_style.shade_style', 'color_wheel'),
    'shade_color_wheel_black': ('fill_style.shade_style', 'color_wheel_black'),
    'shade_color_wheel_white': ('fill_style.shade_style', 'color_wheel_white'),
    'shade_inner_color': ('fill_style.shade_style', 'inner_color'),
    'shade_left_color': ('fill_style.shade_style', 'left_color'),
    'shade_lower_left_color': ('fill_style.shade_style', 'lower_left_color'),
    'shade_lower_right_color': ('fill_style.shade_style', 'lower_right_color'),
    'shade_middle_color': ('fill_style.shade_style', 'middle_color'),
    'shade_outer_color': ('fill_style.shade_style', 'outer_color'),
    'shade_right_color': ('fill_style.shade_style', 'right_color'),
    'shade_top_color': ('fill_style.shade_style', 'top_color'),
    'shade_type': ('fill_style.shade_style', 'shade_type'),
    'shade_upper_left_color': ('fill_style.shade_style', 'upper_left_color'),
    'shade_upper_right_color': ('fill_style.shade_style', 'upper_right_color'),
}

def _set_fill_style_alias_map(debug=False):
    """Set the fill style alias map.

    Args:
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        dict: The fill style alias map.
    """
    pattern_style = PatternStyle()
    shade_style = ShadeStyle()
    grid_style = GridStyle()

    styles = [pattern_style, shade_style, grid_style]
    paths = [
        "fill_style.pattern_style",
        "fill_style.shade_style",
        "fill_style.grid_style",
    ]
    prefixes = ["pattern", "shade", "grid"]

    _set_style_alias_map(fill_style_map, styles, paths, prefixes, debug=debug)
    fill_style_map["alpha"] = ("fill_style", "alpha")
    fill_style_map["color"] = ("fill_style", "color")
    fill_style_map["mode"] = ("fill_style", "mode")

    return fill_style_map


# pattern_style_map = {}

pattern_style_map = {
'alpha': ('pattern_style', 'alpha'),
'pattern_angle': ('pattern_style', 'angle'),
'pattern_color': ('pattern_style', 'color'),
'pattern_distance': ('pattern_style', 'distance'),
'pattern_line_width': ('pattern_style', 'line_width'),
'pattern_points': ('pattern_style', 'points'),
'pattern_radius': ('pattern_style', 'radius'),
'pattern_type': ('pattern_style', 'pattern_type'),
'pattern_x_shift': ('pattern_style', 'x_shift'),
'pattern_y_shift': ('pattern_style', 'y_shift'),
}

def _set_pattern_style_alias_map(debug=False):
    """Set the pattern style alias map.

    Args:
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        dict: The pattern style alias map.
    """
    pattern_style = PatternStyle()

    styles = [pattern_style]
    paths = ["pattern_style"]
    prefixes = ["pattern"]

    _set_style_alias_map(pattern_style_map, styles, paths, prefixes, debug=debug)
    pattern_style_map["alpha"] = ("pattern_style", "alpha")

    return pattern_style_map


# line_style_map = {}
line_style_map = {
    'double_distance': ('line_style', 'double_distance'),
    'double_lines': ('line_style', 'double_lines'),
    'draw_fillets': ('line_style', 'draw_fillets'),
    'draw_markers': ('line_style', 'draw_markers'),
    'fillet_radius': ('line_style', 'fillet_radius'),
    'line_alpha': ('line_style', 'alpha'),
    'line_cap': ('line_style', 'cap'),
    'line_color': ('line_style', 'color'),
    'line_dash_array': ('line_style', 'dash_array'),
    'line_dash_phase': ('line_style', 'dash_phase'),
    'line_join': ('line_style', 'join'),
    'line_miter_limit': ('line_style', 'miter_limit'),
    'line_width': ('line_style', 'width'),
    'marker_color': ('line_style.marker_style', 'color'),
    'marker_radius': ('line_style.marker_style', 'radius'),
    'marker_size': ('line_style.marker_style', 'size'),
    'marker_type': ('line_style.marker_style', 'marker_type'),
    'markers_only': ('line_style', 'markers_only'),
    'smooth': ('line_style', 'smooth'),
    'stroke': ('line_style', 'stroke'),
}


def _set_line_style_alias_map(debug=False):
    """Set the line style alias map.

    Args:
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        dict: The line style alias map.
    """
    line_style = LineStyle()
    marker_style = MarkerStyle()

    styles = [line_style, marker_style]
    paths = ["line_style", "line_style.marker_style"]
    prefixes = ["line", "marker"]

    _set_style_alias_map(line_style_map, styles, paths, prefixes, debug=debug)

    return line_style_map


shape_style_map_ = {} # if any of the styles are changed, the alias-map must be updated!!!

shape_style_map = {
    'alpha': ('style', 'alpha'),
    'back_style': ('style.fill_style', 'back_style'),
    'double_distance': ('style.line_style', 'double_distance'),
    'double_lines': ('style.line_style', 'double_lines'),
    'draw_fillets': ('style.line_style', 'draw_fillets'),
    'draw_markers': ('style.line_style', 'draw_markers'),
    'fill': ('style.fill_style', 'fill'),
    'fill_alpha': ('style.fill_style', 'alpha'),
    'fill_color': ('style.fill_style', 'color'),
    'fill_mode': ('style.fill_style', 'mode'),
    'fillet_radius': ('style.line_style', 'fillet_radius'),
    'grid_alpha': ('style.fill_style.grid_style', 'alpha'),
    'grid_back_color': ('style.fill_style.grid_style', 'back_color'),
    'grid_line_color': ('style.fill_style.grid_style', 'line_color'),
    'grid_line_width': ('style.fill_style.grid_style', 'line_width'),
    'line_alpha': ('style.line_style', 'alpha'),
    'line_cap': ('style.line_style', 'cap'),
    'line_color': ('style.line_style', 'color'),
    'line_dash_array': ('style.line_style', 'dash_array'),
    'line_dash_phase': ('style.line_style', 'dash_phase'),
    'line_join': ('style.line_style', 'join'),
    'line_miter_limit': ('style.line_style', 'miter_limit'),
    'line_width': ('style.line_style', 'width'),
    'marker_color': ('style.line_style.marker_style', 'color'),
    'marker_radius': ('style.line_style.marker_style', 'radius'),
    'marker_size': ('style.line_style.marker_style', 'size'),
    'marker_type': ('style.line_style.marker_style', 'marker_type'),
    'markers_only': ('style.line_style', 'markers_only'),
    'pattern_angle': ('style.fill_style.pattern_style', 'angle'),
    'pattern_color': ('style.fill_style.pattern_style', 'color'),
    'pattern_distance': ('style.fill_style.pattern_style', 'distance'),
    'pattern_line_width': ('style.fill_style.pattern_style', 'line_width'),
    'pattern_points': ('style.fill_style.pattern_style', 'points'),
    'pattern_radius': ('style.fill_style.pattern_style', 'radius'),
    'pattern_type': ('style.fill_style.pattern_style', 'pattern_type'),
    'pattern_x_shift': ('style.fill_style.pattern_style', 'x_shift'),
    'pattern_y_shift': ('style.fill_style.pattern_style', 'y_shift'),
    'shade_axis_angle': ('style.fill_style.shade_style', 'axis_angle'),
    'shade_ball_color': ('style.fill_style.shade_style', 'ball_color'),
    'shade_bottom_color': ('style.fill_style.shade_style', 'bottom_color'),
    'shade_color_wheel': ('style.fill_style.shade_style', 'color_wheel'),
    'shade_color_wheel_black': ('style.fill_style.shade_style', 'color_wheel_black'),
    'shade_color_wheel_white': ('style.fill_style.shade_style', 'color_wheel_white'),
    'shade_inner_color': ('style.fill_style.shade_style', 'inner_color'),
    'shade_left_color': ('style.fill_style.shade_style', 'left_color'),
    'shade_lower_left_color': ('style.fill_style.shade_style', 'lower_left_color'),
    'shade_lower_right_color': ('style.fill_style.shade_style', 'lower_right_color'),
    'shade_middle_color': ('style.fill_style.shade_style', 'middle_color'),
    'shade_outer_color': ('style.fill_style.shade_style', 'outer_color'),
    'shade_right_color': ('style.fill_style.shade_style', 'right_color'),
    'shade_top_color': ('style.fill_style.shade_style', 'top_color'),
    'shade_type': ('style.fill_style.shade_style', 'shade_type'),
    'shade_upper_left_color': ('style.fill_style.shade_style', 'upper_left_color'),
    'shade_upper_right_color': ('style.fill_style.shade_style', 'upper_right_color'),
    'smooth': ('style.line_style', 'smooth'),
    'stroke': ('style.line_style', 'stroke')}


def _set_shape_style_alias_map(debug=False):
    """Set the shape style alias map.

    Args:
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        dict: The shape style alias map.
    """
    line_style = LineStyle()
    fill_style = FillStyle()
    marker_style = MarkerStyle()
    pattern_style = PatternStyle()
    shade_style = ShadeStyle()
    grid_style = GridStyle()

    styles = [
        line_style,
        fill_style,
        marker_style,
        pattern_style,
        shade_style,
        grid_style,
    ]
    paths = [
        "style.line_style",
        "style.fill_style",
        "style.line_style.marker_style",
        "style.fill_style.pattern_style",
        "style.fill_style.shade_style",
        "style.fill_style.grid_style",
    ]
    prefixes = ["line", "fill", "marker", "pattern", "shade", "grid"]

    _set_style_alias_map(shape_style_map, styles, paths, prefixes, debug=debug)
    shape_style_map["alpha"] = ("style", "alpha")
    return shape_style_map


def _set_style_alias_map(map_dict, styles, paths, prefixes, debug=False):
    """Set the style alias map.

    Args:
        map_dict (dict): The dictionary to store the alias map.
        styles (list): List of style objects.
        paths (list): List of paths to the style objects.
        prefixes (list): List of prefixes for the style attributes.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Returns:
        dict: The style alias map.
    """
    for i, style in enumerate(styles):
        style_attribs = style.attribs
        exact = style._exact
        exclude = style._exclude
        style_path = paths[i]
        prefix = prefixes[i]

        for alias in style_attribs:
            if alias in exact or alias in exclude:
                attrib = alias
            else:
                attrib = alias.replace(f"{prefix}_", "")
            if debug:
                if alias in map_dict:
                    print(f"Duplicate style attribute: {alias}")
                print(f"{style_path}.{attrib}", alias)
            map_dict[alias] = (style_path, attrib)

    return map_dict


# shape_args = []

shape_args = [
    'alpha',
    'back_style',
    'dist_tol',
    'double_distance',
    'double_lines',
    'draw_fillets',
    'draw_markers',
    'fill',
    'fill_alpha',
    'fill_color',
    'fill_mode',
    'fillet_radius',
    'grid_alpha',
    'grid_back_color',
    'grid_line_color',
    'grid_line_width',
    'line_alpha',
    'line_cap',
    'line_color',
    'line_dash_array',
    'line_dash_phase',
    'line_join',
    'line_miter_limit',
    'line_width',
    'marker_color',
    'marker_radius',
    'marker_size',
    'marker_type',
    'markers_only',
    'pattern_angle',
    'pattern_color',
    'pattern_distance',
    'pattern_line_width',
    'pattern_points',
    'pattern_radius',
    'pattern_type',
    'pattern_x_shift',
    'pattern_y_shift',
    'points',
    'shade_axis_angle',
    'shade_ball_color',
    'shade_bottom_color',
    'shade_color_wheel',
    'shade_color_wheel_black',
    'shade_color_wheel_white',
    'shade_inner_color',
    'shade_left_color',
    'shade_lower_left_color',
    'shade_lower_right_color',
    'shade_middle_color',
    'shade_outer_color',
    'shade_right_color',
    'shade_top_color',
    'shade_type',
    'shade_upper_left_color',
    'shade_upper_right_color',
    'smooth',
    'stroke',
    'subtype',
    'xform_matrix',
]


def _set_shape_args(debug=False):
    shape_args.extend(list(shape_style_map.keys()))
    shape_args.extend(["subtype", "xform_matrix", "points", "dist_tol"])

# These are applicable to Canvas and Batch objects. They are set in \begin{scope}[...].
group_args = [
    "blend_mode",
    "clip",
    "mask",
    "even_odd_rule",
    "transparency_group",
    "blend_group",
    "alpha",
    "line_alpha",
    "fill_alpha",
    "text_alpha",
]

# batch_args = []

batch_args = [
'alpha',
'back_style',
'blend_group',
'blend_mode',
'clip',
'dist_tol',
'double_distance',
'double_lines',
'draw_fillets',
'draw_markers',
'even_odd_rule',
'fill',
'fill_alpha',
'fill_color',
'fill_mode',
'fillet_radius',
'grid_alpha',
'grid_back_color',
'grid_line_color',
'grid_line_width',
'line_alpha',
'line_cap',
'line_color',
'line_dash_array',
'line_dash_phase',
'line_join',
'line_miter_limit',
'line_width',
'marker_color',
'marker_radius',
'marker_size',
'marker_type',
'markers_only',
'mask',
'modifiers',
'pattern_angle',
'pattern_color',
'pattern_distance',
'pattern_line_width',
'pattern_points',
'pattern_radius',
'pattern_type',
'pattern_x_shift',
'pattern_y_shift',
'shade_axis_angle',
'shade_ball_color',
'shade_bottom_color',
'shade_color_wheel',
'shade_color_wheel_black',
'shade_color_wheel_white',
'shade_inner_color',
'shade_left_color',
'shade_lower_left_color',
'shade_lower_right_color',
'shade_middle_color',
'shade_outer_color',
'shade_right_color',
'shade_top_color',
'shade_type',
'shade_upper_left_color',
'shade_upper_right_color',
'smooth',
'stroke',
'subtype',
'text_alpha',
'transparency_group',
]


def _set_batch_args(debug=False):
    batch_args.extend(list(shape_style_map.keys()))
    batch_args.extend(["subtype", "dist_tol", "modifiers", "dist_tol"])
    batch_args.extend(group_args)
    print()

canvas_args = ["size", "back_color", "title", "border"]
canvas_args.extend(group_args)

shape_aliases_dict = {}


def _set_shape_aliases_dict(shape):
    """Set the shape aliases dictionary.

    Args:
        shape: The shape object to set aliases for.
    """
    for alias, path_attrib in shape_style_map.items():
        style_path, attrib = path_attrib
        obj = shape
        for attrib_name in style_path.split("."):
            obj = obj.__dict__[attrib_name]

            if obj is not shape:
                shape_aliases_dict[alias] = (obj, attrib)
    self.__dict__["_aliasses"] = _aliasses


# From: https://tikz.dev/library-patterns#pgf.patterns

# LINES pattern example
# \usetikzlibrary {patterns,patterns.meta}
# \begin{tikzpicture}
#   \draw[pattern={horizontal lines},pattern color=orange]
#     (0,0) rectangle +(1,1);
#   \draw[pattern={Lines[yshift=.5pt]},pattern color=blue]
#     (0,0) rectangle +(1,1);

#   \draw[pattern={vertical lines},pattern color=orange]
#     (1,0) rectangle +(1,1);
#   \draw[pattern={Lines[angle=90,yshift=-.5pt]},pattern color=blue]
#     (1,0) rectangle +(1,1);

#   \draw[pattern={north east lines},pattern color=orange]
#     (0,1) rectangle +(1,1);
#   \draw[pattern={Lines[angle=45,distance={3pt/sqrt(2)}]},pattern color=blue]
#     (0,1) rectangle +(1,1);

#   \draw[pattern={north west lines},pattern color=orange]
#     (1,1) rectangle +(1,1);
#   \draw[pattern={Lines[angle=-45,distance={3pt/sqrt(2)}]},pattern color=blue]
#     (1,1) rectangle +(1,1);
# \end{tikzpicture}

# HATCH pattern example
# \usetikzlibrary {patterns,patterns.meta}
# \begin{tikzpicture}
#   \draw[pattern={grid},pattern color=orange]
#     (0,0) rectangle +(1,1);
#   \draw[pattern={Hatch},pattern color=blue]
#     (0,0) rectangle +(1,1);

#   \draw[pattern={crosshatch},pattern color=orange]
#     (1,0) rectangle +(1,1);
#   \draw[pattern={Hatch[angle=45,distance={3pt/sqrt(2)},xshift=.1pt]},
#     pattern color=blue] (1,0) rectangle +(1,1);
# \end{tikzpicture}

# DOTS pattern example
# \usetikzlibrary {patterns,patterns.meta}
# \begin{tikzpicture}
#   \draw[pattern={dots},pattern color=orange]
#     (0,0) rectangle +(1,1);
#   \draw[pattern={Dots},pattern color=blue]
#     (0,0) rectangle +(1,1);

#   \draw[pattern={crosshatch dots},pattern color=orange]
#     (1,0) rectangle +(1,1);
#   \draw[pattern={Dots[angle=45,distance={3pt/sqrt(2)}]},
#     pattern color=blue] (1,0) rectangle +(1,1);
# \end{tikzpicture}

# STARS pattern example
# \usetikzlibrary {patterns,patterns.meta}
# \begin{tikzpicture}
#   \draw[pattern={fivepointed stars},pattern color=orange]
#     (0,0) rectangle +(1,1);
#   \draw[pattern={Stars},pattern color=blue]
#     (0,0) rectangle +(1,1);

#   \draw[pattern={sixpointed stars},pattern color=orange]
#     (1,0) rectangle +(1,1);
#   \draw[pattern={Stars[points=6]},pattern color=blue]
#     (1,0) rectangle +(1,1);
# \end{tikzpicture}

# Declare pattern custom pattern
# \tikzdeclarepattern{⟨config⟩}

# A pattern declared with \pgfdeclarepattern can only execute pgf code. This
# command extends the functionality to also allow TikZ code. All the same keys
# of \pgfdeclarepattern are valid, but some of them have been overloaded to give
# a more natural TikZ syntax.

# /tikz/patterns/bottom left=⟨point⟩
# (no default)

# Instead of a pgf name point, this key takes a TikZ point, e.g. (-.1,-.1).

# /tikz/patterns/top right=⟨point⟩
# (no default)

# Instead of a pgf name point, this key takes a TikZ point, e.g. (3.1,3.1).

# /tikz/patterns/tile size=⟨point⟩
# (no default)

# Instead of a pgf name point, this key takes a TikZ point, e.g. (3,3).

# /tikz/patterns/tile transformation=⟨transformation⟩
# (no default)

# Instead of a pgf transformation, this key takes a list of keys and value and
# extracts the resulting transformation from them, e.g. rotate=30.

# In addition to the overloaded keys, some new keys have been added.

# /tikz/patterns/bounding box=⟨point⟩ and ⟨point⟩
# (no default)

# This is a shorthand to set the bounding box. It will assign the first point to
# bottom left and the second point to top right.

# /tikz/patterns/infer tile bounding box=⟨dimension⟩
# (default 0pt)

# Instead of specifying the bounding box by hand, you can ask TikZ to infer the
# size of the bounding box for you. The ⟨dimension⟩ parameter is padding that is
# added around the bounding box.

# Declare pattern example 1
# \usetikzlibrary {patterns.meta}
# \tikzdeclarepattern{
#   name=flower,
#   type=colored,
#   bottom left={(-.1pt,-.1pt)},
#   top right={(10.1pt,10.1pt)},
#   tile size={(10pt,10pt)},
#   code={
#     \tikzset{x=1pt,y=1pt}
#     \path [draw=green] (5,2.5) -- (5, 7.5);
#     \foreach \i in {0,60,...,300}
#       \path [fill=pink, shift={(5,7.5)}, rotate=-\i]
#         (0,0) .. controls ++(120:4) and ++(60:4) .. (0,0);
#     \path [fill=red] (5,7.5) circle [radius=1];
#     \foreach \i in {-45,45}
#       \path [fill=green, shift={(5,2.5)}, rotate=-\i]
#         (0,0) .. controls ++(120:4) and ++(60:4) .. (0,0);
#   }
# }

# \tikz\draw [pattern=flower] circle [radius=1];


# Declare pattern example 2
# \usetikzlibrary {patterns.meta}
# \tikzdeclarepattern{
#   name=mystars,
#   type=uncolored,
#   bounding box={(-5pt,-5pt) and (5pt,5pt)},
#   tile size={(\tikztilesize,\tikztilesize)},
#   parameters={\tikzstarpoints,\tikzstarradius,\tikzstarrotate,\tikztilesize},
#   tile transformation={rotate=\tikzstarrotate},
#   defaults={
#     points/.store in=\tikzstarpoints,points=5,
#     radius/.store in=\tikzstarradius,radius=3pt,
#     rotate/.store in=\tikzstarrotate,rotate=0,
#     tile size/.store in=\tikztilesize,tile size=10pt,
#   },
#   code={
#     \pgfmathparse{180/\tikzstarpoints}\let\a=\pgfmathresult
#     \fill (90:\tikzstarradius) \foreach \i in {1,...,\tikzstarpoints}{
#       -- (90+2*\i*\a-\a:\tikzstarradius/2) -- (90+2*\i*\a:\tikzstarradius)
#     } -- cycle;
#   }
# }

# \begin{tikzpicture}
#  \draw[pattern=mystars,pattern color=blue]               (0,0) rectangle +(2,2);
#  \draw[pattern={mystars[points=7,tile size=15pt]}]       (2,0) rectangle +(2,2);
#  \draw[pattern={mystars[rotate=45]},pattern color=red]   (0,2) rectangle +(2,2);
#  \draw[pattern={mystars[rotate=30,points=4,radius=5pt]}] (2,2) rectangle +(2,2);
# \end{tikzpicture}

# Declare pattern example 3
# \usetikzlibrary {patterns.meta}
# \tikzdeclarepattern{
#   name=mylines,
#   parameters={
#       \pgfkeysvalueof{/pgf/pattern keys/size},
#       \pgfkeysvalueof{/pgf/pattern keys/angle},
#       \pgfkeysvalueof{/pgf/pattern keys/line width},
#   },
#   bounding box={
#     (0,-0.5*\pgfkeysvalueof{/pgf/pattern keys/line width}) and
#     (\pgfkeysvalueof{/pgf/pattern keys/size},
# 0.5*\pgfkeysvalueof{/pgf/pattern keys/line width})},
#   tile size={(\pgfkeysvalueof{/pgf/pattern keys/size},
# \pgfkeysvalueof{/pgf/pattern keys/size})},
#   tile transformation={rotate=\pgfkeysvalueof{/pgf/pattern keys/angle}},
#   defaults={
#     size/.initial=5pt,
#     angle/.initial=45,
#     line width/.initial=.4pt,
#   },
#   code={
#       \draw [line width=\pgfkeysvalueof{/pgf/pattern keys/line width}]
#         (0,0) -- (\pgfkeysvalueof{/pgf/pattern keys/size},0);
#   },
# }

# \begin{tikzpicture}
#   \draw[pattern={mylines[size=10pt,line width=.8pt,angle=10]},
#         pattern color=red]    (0,0) rectangle ++(2,2);
#   \draw[pattern={mylines[size= 5pt,line width=.8pt,angle=40]},
#         pattern color=blue]   (2,0) rectangle ++(2,2);
#   \draw[pattern={mylines[size=10pt,line width=.4pt,angle=90]},
#         pattern color=green]  (0,2) rectangle ++(2,2);
#   \draw[pattern={mylines[size= 2pt,line width= 1pt,angle=70]},
#         pattern color=orange] (2,2) rectangle ++(2,2);
# \end{tikzpicture}


# style.line_style.color line_color
# style.line_style.alpha line_alpha
# style.line_style.width line_width
# style.line_style.dash_array line_dash_array
# style.line_style.dash_phase line_dash_phase
# style.line_style.cap line_cap
# style.line_style.join line_join
# style.line_style.miter_limit line_miter_limit
# style.line_style.fillet_radius fillet_radius
# style.line_style.smooth smooth
# style.line_style.stroke stroke
# style.line_style.draw_markers draw_markers
# style.line_style.draw_fillets draw_fillets
# style.line_style.markers_only markers_only
# style.line_style.double double
# style.line_style.double_distance double_distance
# style.fill_style.color fill_color
# style.fill_style.alpha fill_alpha
# style.fill_style.fill fill
# style.fill_style.back_style back_style
# style.fill_style.mode fill_mode
# style.line_style.marker_style.marker_type marker_type
# style.line_style.marker_style.size marker_size
# style.line_style.marker_style.color marker_color
# style.fill_style.pattern_style.pattern_type pattern_type
# style.fill_style.pattern_style.color pattern_color
# style.fill_style.pattern_style.distance pattern_distance
# style.fill_style.pattern_style.angle pattern_angle
# style.fill_style.pattern_style.x_shift pattern_x_shift
# style.fill_style.pattern_style.y_shift pattern_y_shift
# style.fill_style.pattern_style.line_width pattern_line_width
# style.fill_style.pattern_style.radius pattern_radius
# style.fill_style.pattern_style.points pattern_points
# style.fill_style.shade_style.top_color shade_top_color
# style.fill_style.shade_style.bottom_color shade_bottom_color
# style.fill_style.shade_style.left_color shade_left_color
# style.fill_style.shade_style.right_color shade_right_color
# style.fill_style.shade_style.middle_color shade_middle_color
# style.fill_style.shade_style.inner_color shade_inner_color
# style.fill_style.shade_style.outer_color shade_outer_color
# style.fill_style.shade_style.upper_left_color shade_upper_left_color
# style.fill_style.shade_style.upper_right_color shade_upper_right_color
# style.fill_style.shade_style.lower_left_color shade_lower_left_color
# style.fill_style.shade_style.lower_right_color shade_lower_right_color
# style.fill_style.shade_style.color_wheel shade_color_wheel
# style.fill_style.shade_style.color_wheel_black shade_color_wheel_black
# style.fill_style.shade_style.color_wheel_white shade_color_wheel_white
# style.fill_style.grid_style.line_color grid_line_color
# style.fill_style.grid_style.line_width grid_line_width
# style.fill_style.grid_style.alpha grid_alpha
# style.fill_style.grid_style.back_color grid_back_color


# style.font_style.font_family font_family
# style.font_style.color font_color
# style.font_style.family font_family
# style.font_style.size font_size
# style.font_style.bold bold
# style.font_style.italic italic
# style.font_style.small_caps small_caps
# style.font_style.old_style_nums old_style_nums
# style.font_style.overline overline
# style.font_style.strike_through strike_through
# style.font_style.underline underline
# style.font_style.alpha font_alpha
# style.frame_style.shape frame_shape
# style.frame_style.line_style line_style
# style.frame_style.fill_style fill_style
# style.frame_style.inner_sep frame_inner_sep
# style.frame_style.outer_sep frame_outer_sep
# style.frame_style.min_width frame_min_width
# style.frame_style.min_height frame_min_height
# style.frame_style.min_size frame_min_size
# style.frame_style.alpha frame_alpha
# style.frame_style.blend_mode frame_blend_mode
# style.frame_style.fill_style.color fill_color
# style.frame_style.fill_style.alpha fill_alpha
# style.frame_style.fill_style.fill fill
# style.frame_style.fill_style.back_style back_style
# style.frame_style.fill_style.mode fill_mode
# style.frame_style.fill_style.blend_mode fill_blend_mode
# style.frame_style.line_style.color line_color
# style.frame_style.line_style.alpha line_alpha
# style.frame_style.line_style.width line_width
# style.frame_style.line_style.dash_array line_dash_array
# style.frame_style.line_style.dash_phase line_dash_phase
# style.frame_style.line_style.cap line_cap
# style.frame_style.line_style.join line_join
# style.frame_style.line_style.miter_limit line_miter_limit
# style.frame_style.line_style.fillet_radius fillet_radius
# style.frame_style.line_style.smooth smooth
# style.frame_style.line_style.stroke stroke
# style.frame_style.line_style.blend_mode line_blend_mode
# style.frame_style.line_style.draw_markers draw_markers
# style.frame_style.line_style.draw_fillets draw_fillets
# style.frame_style.line_style.markers_only markers_only
# style.frame_style.line_style.double double

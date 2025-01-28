"""This module contains the Style classes used to set the style of shapes, lines, text,
    and tags. Shape and Tag objects use the maps to create aliases for style attributes.
    Examples:
    shape.style.line_style.color is aliased by shape.line_color
    tag.style.fill_style.pattern_style.line_style.width is aliased by tag.pattern_line_width
    Documentation list all aliases for each style class.
    """

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


# To do: Remove print statements
def _set_style_args(obj, attribs, exact=None, prefix=None):
    """Set the style arguments for the given object."""
    for attrib in attribs:
        if exact and attrib in exact:
            default = defaults.get(attrib, VOID)
            if default != VOID:
                setattr(obj, attrib, default)
            else:
                print(prefix, "_set_style_args missing default:", attrib)
        else:
            if prefix:
                setattr(obj, attrib, defaults[f"{prefix}_{attrib}"])
            else:
                default = defaults.get(attrib, VOID)
                if default != VOID:
                    setattr(obj, attrib, default)
                else:
                    print(prefix, "_set_style_args missing default:", attrib)


def _get_style_attribs(style: Types.STYLE, prefix: str = None) -> List[str]:
    """Get the list of attributes from the given Style object."""
    attribs = [x for x in style.__dict__.keys() if not x.startswith("_")]
    res = []
    for attrib in attribs:
        if attrib in style._exclude:
            continue
        if attrib in style._exact:
            res.append(attrib)
        else:
            res.append(f"{prefix}_{attrib}")
    return res


@dataclass
class FontStyle:
    """FontStyle is used to set the font, color, and style of text."""

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
        self._exact = [
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
        self._exclude = []

        _style_init(
            self,
            exact=self._exact,
            exclude=self._exclude,
            prefix="font",
            subtype=Types.FONTSTYLE,
        )


@dataclass
class GridStyle:
    """GridStyle is used to set the grid color, alpha, width, and pattern."""

    line_color: Color = None
    line_width: float = None
    alpha: float = None
    # width: float = None
    # height: float = None
    back_color: Color = None

    def __post_init__(self):
        self._exact = []
        self._exclude = []

        _style_init(
            self,
            exact=self._exact,
            exclude=self._exclude,
            prefix="grid",
            subtype=Types.GRIDSTYLE,
        )

    def __str__(self):
        return f"GridStyle: {self.id}"

    def __repr__(self):
        return f"GridStyle: {self.id}"

    def _get_attributes(self):
        attribs = [x for x in self.__dict__ if not x.startswith("_")]
        res = []
        for attrib in attribs:
            if attrib in self._exact:
                res.append(attrib)
            else:
                res.append(f"grid_{attrib}")


@dataclass
class MarkerStyle:
    """Marker is used to set the marker type, size, and color of a shape."""

    marker_type: MarkerType = None
    size: float = None
    color: Color = None
    radius: float = None

    def __post_init__(self):
        self._exact = ["marker_type"]
        self._exclude = []
        _style_init(
            self,
            exact=self._exact,
            exclude=self._exclude,
            prefix="marker",
            subtype=Types.MARKERSTYLE,
        )

    def __str__(self):
        return f"Marker: {self.type}"


@dataclass
class LineStyle:
    """LineStyle is used to set the line color, alpha, width, and pattern of a shape."""

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
        self._exact = exact
        self._exclude = exclude
        _style_init(
            self, self._exact, self._exclude, prefix="line", subtype=Types.LINESTYLE
        )
        self.marker_style = MarkerStyle()

    def __str__(self):
        return f"LineStyle: {self.id}"


def _style_init(style, exact=None, exclude=None, prefix="", subtype=None):
    if exclude is None:
        exclude = []
    if exact is None:
        exact = []
    attribs = [x for x in style.__dict__ if not x.startswith("_") and x not in exclude]
    _set_style_args(style, attribs, exact, prefix=prefix)

    style.attribs = _get_style_attribs(style, prefix=prefix)
    style.id = get_unique_id(style)
    style.type = Types.STYLE
    style.subtype = subtype


@dataclass  # used for creating patterns
class PatternStyle:
    """PatternStyle is used to set the pattern type, color, distance,
    angle, shift, line width, radius, and points.
    Patterns come form the TikZ library patterns.meta."""

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
        self._exact = ["stroke", "pattern_type"]
        self._exclude = []
        _style_init(
            self,
            exact=self._exact,
            exclude=self._exclude,
            prefix="pattern",
            subtype=Types.PATTERNSTYLE,
        )

    def __str__(self):
        return f"Pattern: {self.type}"


# \usetikzlibrary{shadings}
@dataclass
class ShadeStyle:
    """ShadeStyle uses TikZ shading library to create colors with gradients."""

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
        self._exact = exact
        self._exclude = exclude
        _style_init(
            self, exact=exact, exclude=exclude, prefix="shade", subtype=Types.SHADESTYLE
        )


@dataclass
class FillStyle:
    """FillStyle is used to set the fill color, alpha, and pattern of a shape."""

    color: Color = None
    alpha: float = None
    fill: bool = None
    back_style: BackStyle = None
    mode: FillMode = None
    pattern_style: PatternStyle = None
    shade_style: ShadeStyle = None
    grid_style: GridStyle = None

    def __post_init__(self):
        self.shade_style = ShadeStyle()
        self.grid_style = GridStyle()
        self.pattern_style = PatternStyle()
        self._exact = ["fill", "back_style"]
        self._exclude = ["pattern_style", "shade_style", "grid_style"]
        _style_init(
            self, self._exact, self._exclude, prefix="fill", subtype=Types.FILLSTYLE
        )

    def __str__(self):
        return f"FillStyle: {self.id}"

    def __repr__(self):
        return f"FillStyle: {self.id}"

    def _get_attributes(self):
        attribs = [x for x in self.__dict__ if not x.startswith("_")]
        res = []
        for attrib in attribs:
            if attrib in self._exact:
                res.append(attrib)
            else:
                res.append(f"fill_{attrib}")


@dataclass
class ShapeStyle:
    """ShapeStyle is used to set the fill and line style of a shape."""

    line_style: LineStyle = None
    fill_style: FillStyle = None
    alpha: float = None

    def __post_init__(self):
        self.line_style = LineStyle()
        self.fill_style = FillStyle()
        self.marker_style = MarkerStyle()
        self.alpha = defaults["alpha"]
        self._exact = ["alpha"]
        self._exclude = ["line_style", "fill_style"]
        _style_init(
            self,
            exact=self._exact,
            exclude=self._exclude,
            prefix="",
            subtype=Types.SHAPESTYLE,
        )

    def __str__(self):
        return f"ShapeStyle: {self.id}"

    def __repr__(self):
        return f"ShapeStyle: {self.id}"


@dataclass
class FrameStyle:
    """FrameStyle is used to set the frame shape, line style, fill style,
    and size of a shape."""

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
        self.line_style = LineStyle()
        self.fill_style = FillStyle()
        self._exact = []
        self._exclude = ["line_style", "fill_style"]
        _style_init(
            self,
            exact=self._exact,
            exclude=self._exclude,
            prefix="frame",
            subtype=Types.FRAMESTYLE,
        )


@dataclass
class TagStyle:
    """TagStyle is used to set the font, color, and style of tag objects."""

    align: Align = None
    alpha: float = None
    anchor: Anchor = None
    blend_mode: BlendMode = None
    draw_frame: bool = None
    font_style: FontStyle = None
    frame_style: FrameStyle = None

    def __post_init__(self):
        self.font_style = FontStyle()
        self.frame_style = FrameStyle()
        self.alpha = defaults["tag_alpha"]
        self.align = defaults["tag_align"]
        self.blend_mode = defaults["tag_blend_mode"]
        self._exact = [
            "alpha",
            "blend_mode",
            "draw_frame",
            "anchor",
        ]
        self._exclude = ["font_style", "frame_style"]

        _style_init(
            self,
            exact=self._exact,
            exclude=self._exclude,
            prefix="tag",
            subtype=Types.TAGSTYLE,
        )

    def __str__(self):
        return f"TagStyle: {self.id}"

    def __repr__(self):
        return f"TagStyle: {self.id}"


frame_style_map = {}


def _set_frame_style_alias_map(debug=False):
    """Set the frame style alias map."""
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


marker_style_map = {}


def _set_marker_style_alias_map(debug=False):
    """Set the marker style alias map."""
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


tag_style_map = {}


def _set_tag_style_alias_map(debug=False):
    """Set the tag style alias map."""
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


fill_style_map = {}


def _set_fill_style_alias_map(debug=False):
    """Set the fill style alias map."""
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


pattern_style_map = {}


def _set_pattern_style_alias_map(debug=False):
    """Set the pattern style alias map."""
    pattern_style = PatternStyle()

    styles = [pattern_style]
    paths = ["pattern_style"]
    prefixes = ["pattern"]

    _set_style_alias_map(pattern_style_map, styles, paths, prefixes, debug=debug)
    pattern_style_map["alpha"] = ("pattern_style", "alpha")

    return pattern_style_map


line_style_map = {}


def _set_line_style_alias_map(debug=False):
    """Set the line style alias map."""
    line_style = LineStyle()
    marker_style = MarkerStyle()

    styles = [line_style, marker_style]
    paths = ["line_style", "line_style.marker_style"]
    prefixes = ["line", "marker"]

    _set_style_alias_map(line_style_map, styles, paths, prefixes, debug=debug)

    return line_style_map


shape_style_map = {}


def _set_shape_style_alias_map(debug=False):
    """Set the shape style alias map."""
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
    """Set the style alias map."""
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


shape_args = []


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

batch_args = []


def _set_batch_args(debug=False):
    batch_args.extend(list(shape_style_map.keys()))
    batch_args.extend(["subtype", "dist_tol", "modifiers", "dist_tol"])
    batch_args.extend(group_args)


canvas_args = ["size", "back_color", "title", "border"]
canvas_args.extend(group_args)

shape_aliases_dict = {}


def _set_shape_aliases_dict(shape):
    """Set the shape aliases dictionary."""
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

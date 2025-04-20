"""All enumerations."""

from typing import Union
from typing_extensions import TypeAlias
from strenum import StrEnum


def get_enum_value(enum_class: StrEnum, value: str) -> str:
    """Get the value of an enumeration."""
    if isinstance(value, enum_class):
        res = value.value
    else:
        res = enum_class[value.upper()].value

    return res


class Align(StrEnum):
    """Align is used to set the alignment of the text in tags.
    Used for Tag and Text objects. This is based on TikZ.

    Valid values are: NONE, CENTER, FLUSH_CENTER, FLUSH_LEFT, FLUSH_RIGHT, JUSTIFY, LEFT, RIGHT.
    """

    NONE = ""
    CENTER = "center"
    FLUSH_CENTER = "flush center"
    FLUSH_LEFT = "flush left"
    FLUSH_RIGHT = "flush right"
    JUSTIFY = "justify"
    LEFT = "left"
    RIGHT = "right"


# Anchor points
# Used for TikZ. VaLUeS are case sensitive.
class Anchor(StrEnum):
    """Anchor is used to set the anchor point of the shapes
    relative to the boundary box of shapes/batches or
    frames of tag objects.

    Valid values are: BASE, BASE_EAST, BASE_WEST, BOTTOM, CENTER, EAST, LEFT, MID, MIDEAST, MIDWEST, NORTH,
    NORTHEAST, NORTHWEST, RIGHT, SOUTH, SOUTHEAST, SOUTHWEST, TEXT, TOP, WEST.
    """

    BASE = "base"  # FOR TAGS ONLY
    BASE_EAST = "base east"  # FOR TAGS ONLY
    BASE_WEST = "base west"  # FOR TAGS ONLY
    BOTTOM = "bottom"
    MIDPOINT = "midpoint"
    EAST = "east"
    LEFT = "left"
    CENTER = "center"
    MID = "mid"
    MIDEAST = "mid east"
    MIDWEST = "mid west"
    NORTH = "north"
    NORTHEAST = "north east"
    NORTHWEST = "north west"
    RIGHT = "right"
    SOUTH = "south"
    SOUTHEAST = "south east"
    SOUTHWEST = "south west"
    TEXT = "text"
    TOP = "top"
    WEST = "west"


class ArrowLine(StrEnum):
    """ArrowLine is used to set the type of arrow line.

    Valid values are: FLATBASE_END, FLATBASE_MIDDLE, FLATBASE_START, FLATBOTH_END, FLATBOTH_MIDDLE,
    FLATBOTH_START, FLATTOP_END, FLATTOP_MIDDLE, FLATTOP_START, STRAIGHT_END, STRAIGHT_MIDDLE, STRAIGHT_START.
    """

    FLATBASE_END = "flatbase end"  # FLAT BASE, ARROW AT THE END
    FLATBASE_MIDDLE = "flatbase middle"  # FLAT BASE, ARROW AT THE MIDDLE
    FLATBASE_START = "flatbase start"  # FLAT BASE, ARROW AT THE START
    FLATBOTH_END = "flatboth end"
    FLATBOTH_MIDDLE = "flatboth middle"
    FLATBOTH_START = "flatboth start"
    FLATTOP_END = "flattop end"  #  FLAT TOP, ARROW AT THE END
    FLATTOP_MIDDLE = "flattop middle"
    FLATTOP_START = "flattop start"
    STRAIGHT_END = "straight end"  # DEFAULT, STRAIGHT LINE, ARROW AT THE END
    STRAIGHT_MIDDLE = "straight middle"  # STRAIGHT LINE, ARROW AT THE MIDDLE
    STRAIGHT_START = "straight start"  # STRAIGHT LINE, ARROW AT THE START


class Axis(StrEnum):
    """Cartesian coordinate system axes.

    Valid values are: X, Y.
    """

    X = "x"
    Y = "y"


# Used for shapes, canvas, and  tags
class BackStyle(StrEnum):
    """BackStyle is used to set the background style of a shape or tag.
    If shape.fill is True, then background will be drawn according to
    the shape.back_style value.

    Valid values are: COLOR, COLOR_AND_GRID, EMPTY, GRIDLINES, PATTERN, SHADING, SHADING_AND_GRID.
    """

    COLOR = "COLOR"
    COLOR_AND_GRID = "COLOR_AND_GRID"
    EMPTY = "EMPTY"
    GRIDLINES = "GRIDLINES"
    PATTERN = "PATTERN"
    SHADING = "SHADING"
    SHADING_AND_GRID = "SHADING_AND_GRID"


class BlendMode(StrEnum):
    """BlendMode is used to set the blend mode of the colors.

    Valid values are: COLOR, COLORBURN, COLORDODGE, DARKEN, DIFFERENCE, EXCLUSION, HARDLIGHT, HUE, LIGHTEN,
    LUMINOSITY, MULTIPLY, NORMAL, OVERLAY, SATURATION, SCREEN, SOFTLIGHT.
    """

    COLOR = "color"
    COLORBURN = "colorburn"
    COLORDODGE = "colordodge"
    DARKEN = "darken"
    DIFFERENCE = "difference"
    EXCLUSION = "exclusion"
    HARDLIGHT = "hardlight"
    HUE = "hue"
    LIGHTEN = "lighten"
    LUMINOSITY = "luminosity"
    MULTIPLY = "multiply"
    NORMAL = "normal"
    OVERLAY = "overlay"
    SATURATION = "saturation"
    SCREEN = "screen"
    SOFTLIGHT = "softlight"


class ColorSpace(StrEnum):
    """ColorSpace is used to set the color space of the colors.

    Valid values are: CMYK, GRAY, HCL, HLS, HSV, LAB, RGB, YIQ.
    """

    CMYK = "CMYK"
    GRAY = "GRAY"
    HCL = "HCL"
    HLS = "HLS"
    HSV = "HSV"
    LAB = "LAB"
    RGB = "RGB"
    YIQ = "YIQ"


class Connection(StrEnum):
    """Connection is used for identifying how two line segments are related
    to each other. Intersection check uses some of these values.

    Valid values are: CHAIN, COINCIDENT, COLL_CHAIN, CONGRUENT, CONTAINS, COVERS, DISJOINT, END_END, END_START,
    FLIPPED, INTERSECT, NONE, OVERLAPS, PARALLEL, START_END, START_START, TOUCHES, WITHIN, YJOINT.
    """

    CHAIN = "CHAIN"
    COINCIDENT = "COINCIDENT"
    COLL_CHAIN = "COLL_CHAIN"
    CONGRUENT = "CONGRUENT"
    CONTAINS = "CONTAINS"
    COVERS = "COVERS"
    DISJOINT = "DISJOINT"
    END_END = "END_END"
    END_START = "END_START"
    FLIPPED = "FLIPPED"
    INTERSECT = "INTERSECT"
    NONE = "NONE"
    OVERLAPS = "OVERLAPS"
    PARALLEL = "PARALLEL"
    START_END = "START_END"
    START_START = "START_START"
    TOUCHES = "TOUCHES"
    WITHIN = "WITHIN"
    YJOINT = "YJOINT"


class Connector(StrEnum):
    """Connector is used to set the way connecting lines are drawn.
    This is not used yet.

    Valid values are: ARC, ARROW_LINE, CURVE, LINE, DOUBLE_LINE, SQUIGLY, ZIGZAG, SQUIGLY_ARROW, ZIGZAG_ARROW,
    DOUBLE_ARROW, DOUBLE_SQUIGLY.
    """

    ARC = "ARC"
    ARROW_LINE = "ARROW_LINE"
    CURVE = "CURVE"
    LINE = "LINE"
    DOUBLE_LINE = "DOUBLE_LINE"
    SQUIGLY = "squigly"
    ZIGZAG = "zigzag"
    SQUIGLY_ARROW = "squigly_arrow"
    ZIGZAG_ARROW = "zigzag_arrow"
    DOUBLE_ARROW = "double_arrow"
    DOUBLE_SQUIGLY = "double_squigly"


class ConstraintType(StrEnum):
    """Constraint types are used with the 2D geometric constraint solver.

    Valid values are: COLLINEAR, DISTANCE, LINE_ANGLE, PARALLEL, PERPENDICULAR, EQUAL_SIZE, EQUAL_VALUE,
    INNER_TANGENT, OUTER_TANGENT.
    """

    COLLINEAR = "COLLINEAR"
    DISTANCE = "DISTANCE"
    LINE_ANGLE = "LINE_ANGLE"
    PARALLEL = "PARALLEL"
    PERPENDICULAR = "PERPENDICULAR"
    EQUAL_SIZE = "EQUAL_SIZE"
    EQUAL_VALUE = "EQUAL_VALUE"
    INNER_TANGENT = "INNER_TANGENT"
    OUTER_TANGENT = "OUTER_TANGENT"


class Compiler(StrEnum):
    """Used for the LaTeX compiler.
    Currently, only XELATEX is used.

    Valid values are: LATEX, PDFLATEX, XELATEX, LUALATEX.
    """

    LATEX = "LATEX"
    PDFLATEX = "PDFLATEX"
    XELATEX = "XELATEX"
    LUALATEX = "LUALATEX"


class Control(StrEnum):
    """Used with the modifiers.

    Valid values are: INITIAL, PAUSE, RESTART, RESUME, STOP.
    """

    INITIAL = "INITIAL"
    PAUSE = "PAUSE"
    RESTART = "RESTART"
    RESUME = "RESUME"
    STOP = "STOP"


class Conway(StrEnum):
    """Frieze groups in Conway notation.

    Valid values are: HOP, JUMP, SIDLE, SPINNING_HOP, SPINNING_JUMP, SPINNING_SIDLE, STEP.
    """

    HOP = "HOP"
    JUMP = "JUMP"
    SIDLE = "SIDLE"
    SPINNING_HOP = "SPINNING_HOP"
    SPINNING_JUMP = "SPINNING_JUMP"
    SPINNING_SIDLE = "SPINNING_SIDLE"
    STEP = "STEP"


class CurveMode(StrEnum):
    """CurveMode is used to set how arc objects are drawn.

    Valid values are: OPEN, CHORD, PIE.
    """

    OPEN = "OPEN"
    CHORD = "CHORD"
    PIE = "PIE"


class Dep(StrEnum):
    """Depend may be used in the future to set the dependency of the shapes
    when they are copied. Dependent copies share the same underlying data.

    Valid values are: FALSE, TRUE, GEOM, STYLE.
    """

    FALSE = "FALSE"  # Independent
    TRUE = "TRUE"  # Both geometry and style are dependent
    GEOM = "GEOM"  # Only geometry is dependent
    STYLE = "STYLE"  # Only style is dependent


class DocumentClass(StrEnum):
    """DocumentClass is used to set the class of the LaTeX document.

    Valid values are: ARTICLE, BEAMER, BOOK, IEEETRAN, LETTER, REPORT, SCRARTCL, SLIDES, STANDALONE.
    """

    ARTICLE = "article"
    BEAMER = "beamer"
    BOOK = "book"
    IEEETRAN = "ieeetran"
    LETTER = "letter"
    REPORT = "report"
    SCRARTCL = "scrartcl"
    SLIDES = "slides"
    STANDALONE = "standalone"


class FillMode(StrEnum):
    """FillMode is used to set the fill mode of the shape.

    Valid values are: EVENODD, NONZERO.
    """

    EVENODD = "even odd"
    NONZERO = "non zero"


class FontFamily(StrEnum):
    """FontFamily is used to set the family of the font.

    Valid values are: MONOSPACE, SERIF, SANSSERIF.
    """

    MONOSPACE = "monospace"  # \ttfamily, \texttt
    SERIF = "serif"  # serif \rmfamily, \textrm
    SANSSERIF = "sansserif"  # \sffamily, \textsf


class FontSize(StrEnum):
    """FontSize is used to set the size of the font.

    Valid values are: FOOTNOTESIZE, HUGE, HUGE2, LARGE, LARGE2, LARGE3, NORMAL, SCRIPTSIZE, SMALL, TINY.
    """

    FOOTNOTESIZE = "footnotesize"
    HUGE = "huge"  # \huge
    HUGE2 = "Huge"  # \Huge
    LARGE = "large"  # \large
    LARGE2 = "Large"  # \Large
    LARGE3 = "LARGE"  # \LARGE
    NORMAL = "normalsize"  # \normalsize
    SCRIPTSIZE = "scriptsize"  # \scriptsize
    SMALL = "small"  # \small
    TINY = "tiny"  # \tiny


class FontStretch(StrEnum):
    """FontStretch is used to set the stretch of the font.
    These come from LaTeX.

    Valid values are: CONDENSED, EXPANDED, EXTRA_CONDENSED, EXTRA_EXPANDED, NORMAL, SEMI_CONDENSED,
    SEMI_EXPANDED, ULTRA_CONDENSED, ULTRA_EXPANDED.
    """

    CONDENSED = "condensed"
    EXPANDED = "expanded"
    EXTRA_CONDENSED = "extracondensed"
    EXTRA_EXPANDED = "extraexpanded"
    NORMAL = "normal"
    SEMI_CONDENSED = "semicondensed"
    SEMI_EXPANDED = "semiexpanded"
    ULTRA_CONDENSED = "ultracondensed"
    ULTRA_EXPANDED = "ultraexpanded"


class FontStrike(StrEnum):
    """FontStrike is used to set the strike of the font.

    Valid values are: OVERLINE, THROUGH, UNDERLINE.
    """

    OVERLINE = "overline"
    THROUGH = "through"
    UNDERLINE = "underline"


class FontWeight(StrEnum):
    """FontWeight is used to set the weight of the font.

    Valid values are: BOLD, MEDIUM, NORMAL.
    """

    BOLD = "bold"
    MEDIUM = "medium"
    NORMAL = "normal"


class FrameShape(StrEnum):
    """FrameShape is used to set the shape of the frame.
    Frames are used for the tags.

    Valid values are: CIRCLE, DIAMOND, ELLIPSE, FORBIDDEN, PARALLELOGRAM, POLYGON, RECTANGLE, RHOMBUS,
    SPLITCIRCLE, SQUARE, STAR, TRAPEZOID.
    """

    CIRCLE = "circle"
    DIAMOND = "diamond"
    ELLIPSE = "ellipse"
    FORBIDDEN = "forbidden"
    PARALLELOGRAM = "parallelogram"
    POLYGON = "polygon"
    RECTANGLE = "rectangle"
    RHOMBUS = "rhombus"
    SPLITCIRCLE = "split circle"
    SQUARE = "square"
    STAR = "star"
    TRAPEZOID = "trapezoid"


class Graph(StrEnum):
    """Graph is used to set the type of graph.

    Valid values are: DIRECTED, DIRECTEDWEIGHTED, UNDIRECTED, UNDIRECTEDWEIGHTED.
    """

    DIRECTED = "DIRECTED"
    DIRECTEDWEIGHTED = "DIRECTEDWEIGHTED"
    UNDIRECTED = "UNDIRECTED"
    UNDIRECTEDWEIGHTED = "UNDIRECTEDWEIGHTED"


class GridType(StrEnum):
    """GridType is used to set the type of grid.
    Grids are used for creating star patterns.

    Valid values are: CIRCULAR, SQUARE, HEXAGONAL, MIXED.
    """

    CIRCULAR = "CIRCULAR"
    SQUARE = "SQUARE"
    HEXAGONAL = "HEXAGONAL"
    MIXED = "MIXED"  # CIRCULAR + SQUARE


# arrow head positions
class HeadPos(StrEnum):
    """Arrow head positions.

    Valid values are: BOTH, END, MIDDLE, START, NONE.
    """

    BOTH = "BOTH"
    END = "END"
    MIDDLE = "MIDDLE"
    START = "START"
    NONE = "NONE"


class IUC(StrEnum):
    """IUC notation for frieze groups.

    Valid values are: P1, P11G, P11M, P1M1, P2, P2MG, P2MM.
    """

    P1 = "P1"
    P11G = "P11G"
    P11M = "P11M"
    P1M1 = "P1M1"
    P2 = "P2"
    P2MG = "P2MG"
    P2MM = "P2MM"


class LineCap(StrEnum):
    """LineCap is used to set the type of line cap.

    Valid values are: BUTT, ROUND, SQUARE.
    """

    BUTT = "butt"
    ROUND = "round"
    SQUARE = "square"


class LineDashArray(StrEnum):
    """LineDashArray is used to set the type of dashed-line.

    Valid values are: DASHDOT, DASHDOTDOT, DASHED, DENSELY_DASHED, DENSELY_DOTTED, DOTTED, LOOSELY_DASHED,
    LOOSELY_DOTTED, SOLID.
    """

    DASHDOT = "dash dot"
    DASHDOTDOT = "dash dot dot"
    DASHED = "dashed"
    DENSELY_DASHED = "densely dashed"
    DENSELY_DOTTED = "densely dotted"
    DOTTED = "dotted"
    LOOSELY_DASHED = "loosely dashed"
    LOOSELY_DOTTED = "loosely dotted"
    SOLID = "solid"


class LineJoin(StrEnum):
    """LineJoin is used to set the type of line join.

    Valid values are: BEVEL, MITER, ROUND.
    """

    BEVEL = "bevel"
    MITER = "miter"
    ROUND = "round"


class LineWidth(StrEnum):
    """LineWidth is used to set the width of the line.

    Valid values are: SEMITHICK, THICK, THIN, ULTRA_THICK, ULTRA_THIN, VERY_THICK, VERY_THIN.
    """

    SEMITHICK = "semithick"
    THICK = "thick"
    THIN = "thin"
    ULTRA_THICK = "ultra thick"
    ULTRA_THIN = "ultra thin"
    VERY_THICK = "very thick"
    VERY_THIN = "very thin"


class MarkerPos(StrEnum):
    """MarkerPos is used to set the position of the marker.

    Valid values are: CONCAVEHULL, CONVEXHULL, MAINX, OFFSETX.
    """

    CONCAVEHULL = "CONCAVEHULL"
    CONVEXHULL = "CONVEXHULL"
    MAINX = "MAINX"
    OFFSETX = "OFFSETX"


class MarkerType(StrEnum):
    """MarkerType is used to set the type of marker.

    Valid values are: ASTERISK, BAR, CIRCLE, CROSS, DIAMOND, DIAMOND_F, EMPTY, FCIRCLE, HALF_CIRCLE,
    HALF_CIRCLE_F, HALF_DIAMOND, HALF_DIAMOND_F, HALF_SQUARE, HALF_SQUARE_F, HEXAGON, HEXAGON_F, INDICES,
    MINUS, OPLUS, OPLUS_F, O_TIMES, O_TIMES_F, PENTAGON, PENTAGON_F, PLUS, SQUARE, SQUARE_F, STAR, STAR2,
    STAR3, TEXT, TRIANGLE, TRIANGLE_F.
    """

    ASTERISK = "asterisk"
    BAR = "|"
    CIRCLE = "o"
    CROSS = "x"
    DIAMOND = "diamond"
    DIAMOND_F = "diamond*"
    EMPTY = ""
    FCIRCLE = "*"
    HALF_CIRCLE = "halfcircle"
    HALF_CIRCLE_F = "halfcircle*"
    HALF_DIAMOND = "halfdiamond"
    HALF_DIAMOND_F = "halfdiamond*"
    HALF_SQUARE = "halfsquare"
    HALF_SQUARE_F = "halfsquare*"
    HEXAGON = "hexagon"
    HEXAGON_F = "hexagon*"
    INDICES = "indices"
    MINUS = "-"
    OPLUS = "oplus"
    OPLUS_F = "oplus*"
    O_TIMES = "otimes"
    O_TIMES_F = "otimes*"
    PENTAGON = "pentagon"
    PENTAGON_F = "pentagon*"
    PLUS = "+"
    SQUARE = "square"
    SQUARE_F = "square*"
    STAR = "star"
    STAR2 = "star2"
    STAR3 = "star3"
    TEXT = "text"
    TRIANGLE = "triangle"
    TRIANGLE_F = "triangle*"


class MusicScale(StrEnum):
    """MusicScale is used for musical note scales.
    This is used for audio generation for animations.
    Not implemented yet!!!

    Valid values are: MAJOR, MINOR, CHROMATIC, PENTATONIC, IONIC, DORIAN, PHRYGIAN, LYDIAN, MIXOLYDIAN,
    AEOLIAN, LOCRIAN.
    """

    MAJOR = "major"
    MINOR = "minor"
    CHROMATIC = "chromatic"
    PENTATONIC = "pentatonic"
    IONIC = "ionic"
    DORIAN = "dorian"
    PHRYGIAN = "phrygian"
    LYDIAN = "lydian"
    MIXOLYDIAN = "mixolydian"
    AEOLIAN = "aeolian"
    LOCRIAN = "locrian"


class Orientation(StrEnum):
    """Orientation is used to set the orientation of the dimension
    lines.

    Valid values are: ANGLED, HORIZONTAL, VERTICAL.
    """

    ANGLED = "ANGLED"
    HORIZONTAL = "HORIZONTAL"
    VERTICAL = "VERTICAL"


class PageMargins(StrEnum):
    """Page margins for the LaTeX documents.
    Used in Page class.

    Valid values are: CUSTOM, NARROW, STANDARD, WIDE.
    """

    CUSTOM = "custom"
    NARROW = "narrow"
    STANDARD = "standard"
    WIDE = "wide"


class PageNumbering(StrEnum):
    """Page numbering style for the LaTeX documents.
    Used in Page class.

    Valid values are: ALPH, ALPHUPPER, ARABIC, NONE, ROMAN, ROMAN_UPPER.
    """

    ALPH = "alph"
    ALPHUPPER = "ALPH"
    ARABIC = "arabic"
    NONE = "none"
    ROMAN = "roman"
    ROMAN_UPPER = "ROMAN"


class PageNumberPosition(StrEnum):
    """Page number positions for the LaTeX documents.
    Used in Page class.

    Valid values are: BOTTOM_CENTER, BOTTOM_LEFT, BOTTOM_RIGHT, CUSTOM, TOP_CENTER, TOP_LEFT, TOP_RIGHT.
    """

    BOTTOM_CENTER = "bottom"
    BOTTOM_LEFT = "bottom left"
    BOTTOM_RIGHT = "bottom right"
    CUSTOM = "custom"
    TOP_CENTER = "top"
    TOP_LEFT = "top left"
    TOP_RIGHT = "top right"


class PageOrientation(StrEnum):
    """Page orientations for the LaTeX documents.
    Used in Page class.

    Valid values are: LANDSCAPE, PORTRAIT.
    """

    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"


class PageSize(StrEnum):
    """Page sizes for the LaTeX documents.
    Used in Page class.

    Valid values are: LETTER, LEGAL, EXECUTIVE, B0, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13,
    A0, A1, A2, A3, A4, A5, A6.
    """

    LETTER = "letterpaper"
    LEGAL = "legalpaper"
    EXECUTIVE = "executivepaper"
    B0 = "b0paper"
    B1 = "b1paper"
    B2 = "b2paper"
    B3 = "b3paper"
    B4 = "b4paper"
    B5 = "b5paper"
    B6 = "b6paper"
    B7 = "b7paper"
    B8 = "b8paper"
    B9 = "b9paper"
    B10 = "b10paper"
    B11 = "b11paper"
    B12 = "b12paper"
    B13 = "b13paper"
    A0 = "a0paper"
    A1 = "a1paper"
    A2 = "a2paper"
    A3 = "a3paper"
    A4 = "a4paper"
    A5 = "a5paper"
    A6 = "a6paper"


class PathOperation(StrEnum):
    """PathOperation is used to set the type of path operation.
    Used with LinPath objects.

    Valid values are: ARC, ARC_TO, BLEND_ARC, BLEND_CUBIC, BLEND_QUAD, BLEND_SINE, CLOSE, CUBIC_TO, FORWARD,
    HOBBY_TO, H_LINE, LINE_TO, MOVE_TO, QUAD_TO, R_LINE, R_MOVE, SEGMENTS, SINE, V_LINE.
    """

    ARC = "ARC"
    ARC_TO = "ARC_TO"
    BLEND_ARC = "BLEND_ARC"
    BLEND_CUBIC = "BLEND_CUBIC"
    BLEND_QUAD = "BLEND_QUAD"
    BLEND_SINE = "BLEND_SINE"
    CLOSE = "CLOSE"
    CUBIC_TO = "CUBIC_TO"
    FORWARD = "FORWARD"
    HOBBY_TO = "HOBBY_TO"
    H_LINE = "H_LINE"
    LINE_TO = "LINE_TO"
    MOVE_TO = "MOVE_TO"
    QUAD_TO = "QUAD_TO"
    R_LINE = "RLINE"
    R_MOVE = "RMOVE"
    SEGMENTS = "SEGMENTS"
    SINE = "SINE"
    V_LINE = "V_LINE"


class PatternType(StrEnum):
    """PatternType is used to set the type of pattern.
    Used with closed shapes and Tag objects.

    Valid values are: BRICKS, CHECKERBOARD, CROSSHATCH, CROSSHATCH_DOTS, DOTS, FIVE_POINTED_STARS, GRID,
    HORIZONTAL_LINES, NORTHEAST, NORTHWEST, SIX_POINTED_STARS, VERTICAL_LINES.
    """

    BRICKS = "bricks"
    CHECKERBOARD = "checkerboard"
    CROSSHATCH = "crosshatch"
    CROSSHATCH_DOTS = "crosshatch dots"
    DOTS = "dots"
    FIVE_POINTED_STARS = "fivepointed stars"
    GRID = "grid"
    HORIZONTAL_LINES = "horizontal lines"
    NORTHEAST = "north east lines"
    NORTHWEST = "north west lines"
    SIX_POINTED_STARS = "sixpointed stars"
    VERTICAL_LINES = "vertical lines"


class Placement(StrEnum):
    """Placement is used to set the placement of the tags
    relative to another object.

    Valid values are: ABOVE, ABOVE_LEFT, ABOVE_RIGHT, BELOW, BELOW_LEFT, BELOW_RIGHT, CENTERED, INSIDE,
    LEFT, OUTSIDE, RIGHT.
    """

    ABOVE = "above"
    ABOVE_LEFT = "above left"
    ABOVE_RIGHT = "above right"
    BELOW = "below"
    BELOW_LEFT = "below left"
    BELOW_RIGHT = "below right"
    CENTERED = "centered"
    INSIDE = "inside"
    LEFT = "left"
    OUTSIDE = "outside"
    RIGHT = "right"


class Render(StrEnum):
    """Render is used to set the type of rendering.

    Valid values are: EPS, PDF, SVG, TEX.
    """

    EPS = "EPS"
    PDF = "PDF"
    SVG = "SVG"
    TEX = "TEX"


class Result(StrEnum):
    """Result is used for the return values of the functions.

    Valid values are: FAILURE, GO, NOPAGES, OVERWRITE, SAVED, STOP, SUCCESS.
    """

    FAILURE = "FAILURE"
    GO = "GO"
    NOPAGES = "NO_PAGES"
    OVERWRITE = "OVERWRITE"
    SAVED = "SAVED"
    STOP = "STOP"
    SUCCESS = "SUCCESS"


class ShadeType(StrEnum):
    """ShadeType is used to set the type of shading.

    Valid values are: AXIS_LEFT_RIGHT, AXIS_TOP_BOTTOM, AXIS_LEFT_MIDDLE, AXIS_RIGHT_MIDDLE, AXIS_TOP_MIDDLE,
    AXIS_BOTTOM_MIDDLE, BALL, BILINEAR, COLORWHEEL, COLORWHEEL_BLACK, COLORWHEEL_WHITE, RADIAL_INNER,
    RADIAL_OUTER, RADIAL_INNER_OUTER.
    """

    AXIS_LEFT_RIGHT = "axis left right"
    AXIS_TOP_BOTTOM = "axis top bottom"
    AXIS_LEFT_MIDDLE = "axis left middle"
    AXIS_RIGHT_MIDDLE = "axis right middle"
    AXIS_TOP_MIDDLE = "axis top middle"
    AXIS_BOTTOM_MIDDLE = "axis bottom middle"
    BALL = "ball"
    BILINEAR = "bilinear"
    COLORWHEEL = "color wheel"
    COLORWHEEL_BLACK = "color wheel black center"
    COLORWHEEL_WHITE = "color wheel white center"
    RADIAL_INNER = "radial inner"
    RADIAL_OUTER = "radial outer"
    RADIAL_INNER_OUTER = "radial inner outer"


# Anchor lines are called sides.
class Side(StrEnum):
    """Side is used to with boundary box offset lines.
    They determine the position of the offset lines.

    Valid values are: BASE, BOTTOM, DIAGONAL1, DIAGONAL2, H_CENTERLINE, LEFT, MID, RIGHT, TOP, V_CENTERLINE.
    """

    BASE = "BASE"
    BOTTOM = "BOTTOM"
    DIAGONAL1 = "DIAGONAL1"
    DIAGONAL2 = "DIAGONAL2"
    H_CENTERLINE = "H_CENTERLINE"
    LEFT = "LEFT"
    MID = "MID"
    RIGHT = "RIGHT"
    TOP = "TOP"
    V_CENTERLINE = "V_CENTERLINE"


class State(StrEnum):
    """State is used for modifiers.
    Not implemented yet.

    Valid values are: INITIAL, PAUSED, RESTARTING, RUNNING, STOPPED.
    """

    INITIAL = "INITIAL"
    PAUSED = "PAUSED"
    RESTARTING = "RESTARTING"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"


class TexLoc(StrEnum):
    """TexLoc is used to set the location of the TeX related
    objects.

    Valid values are: DOCUMENT, PICTURE, PREAMBLE, NONE.
    """

    DOCUMENT = "DOCUMENT"  # BETWEEN \BEGIN{DOCUMENT} AND \BEGIN{TIKZPICTURE}
    PICTURE = "PICTURE"  # AFTER \BEGIN{PICTURE}
    PREAMBLE = "PREAMBLE"  # BEFORE \BEGIN{DOCUMENT}
    NONE = "NONE"  # Anywhere in the picture.


class Topology(StrEnum):
    """Topology is used for geometry classification.

    Valid values are: CLOSED, COLLINEAR, CONGRUENT, FOLDED, INTERSECTING, OPEN, SELF_INTERSECTING, SIMPLE,
    YJOINT.
    """

    CLOSED = "CLOSED"
    COLLINEAR = "COLLINEAR"
    CONGRUENT = "CONGRUENT"
    FOLDED = "FOLDED"
    INTERSECTING = "INTERSECTING"
    OPEN = "OPEN"
    SELF_INTERSECTING = "SELF_INTERSECTING"
    SIMPLE = "SIMPLE"
    YJOINT = "YJOINT"


class Transformation(StrEnum):
    """Transformation is used to set the type of a transformation.

    Valid values are: GLIDE, MIRROR, ROTATE, SCALE, SHEAR, TRANSFORM, TRANSLATE.
    """

    GLIDE = "GLIDE"
    MIRROR = "MIRROR"
    ROTATE = "ROTATE"
    SCALE = "SCALE"
    SHEAR = "SHEAR"
    TRANSFORM = "TRANSFORM"
    TRANSLATE = "TRANSLATE"


class Types(StrEnum):
    """All objects in simetri.graphics has type and subtype properties.
    Types are mostly Batch and Shape,  and subtypes are listed here.
    """

    ANGULAR_DIMENSION = "ANGULAR DIMENSION"
    ANNOTATION = "ANNOTATION"
    ARC = "ARC"
    ARC_ARROW = "ARC_ARROW"
    ARC_SKETCH = "ARC_SKETCH"
    ARROW = "ARROW"
    ARROW_HEAD = "ARROW_HEAD"
    AXIS = "AXIS"
    BATCH = "BATCH"
    BATCH_SKETCH = "BATCH_SKETCH"
    BEZIER = "BEZIER"
    BEZIER_SKETCH = "BEZIER_SKETCH"
    BBOX_SKETCH = "BBOX_SKETCH"
    BOUNDING_BOX = "BOUNDING_BOX"
    BRACE = "BRACE"
    CANVAS = "CANVAS"
    CIRCLE = "CIRCLE"
    CIRCLE_SKETCH = "CIRCLE_SKETCH"
    CIRCULAR_GRID = "CIRCULAR_GRID"
    COLOR = "COLOR"
    CS = "CS"
    CURVE = "CURVE"
    CURVE_SKETCH = "CURVE_SKETCH"
    DIMENSION = "DIMENSION"
    DIRECTED = "DIRECTED_GRAPH"
    DIVISION = "DIVISION"
    DOT = "DOT"
    DOTS = "DOTS"
    EDGE = "EDGE"
    ELLIPSE = "ELLIPSE"
    ELLIPSE_SKETCH = "ELLIPSE_SKETCH"
    ELLIPTIC_ARC = "ELLIPTIC_ARC"
    FILL_STYLE = "FILL_STYLE"
    FONT = "FONT"
    FONT_SKETCH = "FONT_SKETCH"
    FONT_STYLE = "FONT_STYLE"
    FRAGMENT = "FRAGMENT"
    FRAGMENT_SKETCH = "FRAGMENT_SKETCH"
    FRAME = "FRAME"
    FRAME_SKETCH = "FRAME_SKETCH"
    FRAME_STYLE = "FRAME_STYLE"
    GRADIENT = "GRADIENT"
    GRID = "GRID"
    GRID_STYLE = "GRID_STYLE"
    HANDLE = "HANDLE"
    HEXAGONAL = "HEXAGONAL"
    HEX_GRID = "HEX_GRID"
    ICANVAS = "ICANVAS"
    IMAGE = "IMAGE"
    IMAGE_SKETCH = "IMAGE_SKETCH"
    INTERSECTION = "INTERSECTION"
    LABEL = "LABEL"
    LACE = "LACE"
    LACESKETCH = "LACE_SKETCH"
    LINE = "LINE"
    LINEAR = "LINEAR"
    LINE_SKETCH = "LINE_SKETCH"
    LINE_STYLE = "LINE_STYLE"
    LINPATH = "LINPATH"
    LOOM = "LOOM"
    MARKER = "MARKER"
    MARKER_STYLE = "MARKER_STYLE"
    MASK = "MASK"
    MIXED_GRID = "MIXED_GRID"
    NONE = "NONE"
    OBLIQUE = "OBLIQUE"
    OUTLINE = "OUTLINE"
    OVERLAP = "OVERLAP"
    PAGE = "PAGE"
    PAGE_GRID = "PAGE_GRID"
    PARALLEL_POLYLINE = "PARALLEL_POLYLINE"
    PART = "PART"
    PATH_OPERATION = "PATH_OPERATION"
    PATH_SKETCH = "PATH_SKETCH"
    PATTERN = "PATTERN"
    PATTERN_SKETCH = "PATTERN_SKETCH"
    PATTERN_STYLE = "PATTERN_STYLE"
    PETAL = "PETAL"
    PLAIT = "PLAIT"
    PLAIT_SKETCH = "PLAIT_SKETCH"
    POINT = "POINT"
    POINTS = "POINTS"
    POLYLINE = "POLYLINE"
    Q_BEZIER = "Q_BEZIER"
    RADIAL = "RADIAL"
    RECT_SKETCH = "RECT_SKETCH"
    RECTANGLE = "RECTANGLE"
    RECTANGULAR = "RECTANGULAR"
    REG_POLY = "REGPOLY"
    REG_POLY_SKETCH = "REGPOLY_SKETCH"
    REGULAR_POLYGON = "REGULAR_POLYGON"
    RHOMBIC = "RHOMBIC"
    SECTION = "SECTION"
    SEGMENT = "SEGMENT"
    SEGMENTS = "SEGMENTS"
    SHADE_STYLE = "SHADE_STYLE"
    SHAPE = "SHAPE"
    SHAPE_SKETCH = "SHAPE_SKETCH"
    SHAPE_STYLE = "SHAPE_STYLE"
    SINE_WAVE = "SINE_WAVE"
    SKETCH = "SKETCH"
    SKETCH_STYLE = "SKETCH_STYLE"
    SQUARE = "SQUARE"
    SQUARE_GRID = "SQUARE_GRID"
    STAR = "STAR"
    STYLE = "STYLE"
    SVG_PATH = "SVG_PATH"
    SVG_PATH_SKETCH = "SVG_PATH_SKETCH"
    TAG = "TAG"
    TAG_SKETCH = "TAG_SKETCH"
    TAG_STYLE = "TAG_STYLE"
    TEX = "TEX"  # USED FOR GENERATING OUTPUTFILE.TEX
    TEX_SKETCH = "TEX_SKETCH"
    TEXT = "TEXT"
    TEXTANCHOR = "TEXT_ANCHOR"
    TEXT_ANCHOR_LINE = "TEXT_ANCHORLINE"
    TEXT_ANCHOR_POINT = "TEXT_ANCHORPOINT"
    THREAD = "THREAD"
    TRANSFORM = "TRANSFORM"
    TRANSFORMATION = "TRANSFORMATION"
    TRIANGLE = "TRIANGLE"
    TURTLE = "TURTLE"
    UNDIRECTED = "UNDIRECTED_GRAPH"
    VERTEX = "VERTEX"
    WARP = "WARP"
    WEFT = "WEFT"
    WEIGHTED = "WEIGHTED_GRAPH"


drawable_types = [
    Types.ARC,
    Types.ARC_ARROW,
    Types.ARROW,
    Types.ARROW_HEAD,
    Types.BATCH,
    Types.BEZIER,
    Types.BOUNDING_BOX,
    Types.CIRCLE,
    Types.CIRCULAR_GRID,
    Types.DIMENSION,
    Types.DIVISION,
    Types.DOT,
    Types.DOTS,
    Types.EDGE,
    Types.ELLIPSE,
    Types.FRAGMENT,
    Types.HEX_GRID,
    Types.IMAGE,
    Types.INTERSECTION,
    Types.LACE,
    Types.LINPATH,
    Types.MIXED_GRID,
    Types.OUTLINE,
    Types.OVERLAP,
    Types.PARALLEL_POLYLINE,
    Types.PATTERN,
    Types.PLAIT,
    Types.POLYLINE,
    Types.Q_BEZIER,
    Types.RECTANGLE,
    Types.SECTION,
    Types.SEGMENT,
    Types.SHAPE,
    Types.SINE_WAVE,
    Types.SQUARE_GRID,
    Types.STAR,
    Types.SVG_PATH,
    Types.TAG,
    Types.TURTLE,
]

shape_types = [
    Types.ARC,
    Types.ARROW_HEAD,
    Types.BEZIER,
    Types.BRACE,
    Types.CIRCLE,
    Types.CURVE,
    Types.DIVISION,
    Types.ELLIPSE,
    Types.FRAME,
    Types.INTERSECTION,
    Types.LINE,
    Types.POLYLINE,
    Types.Q_BEZIER,
    Types.SECTION,
    Types.SHAPE,
    Types.SINE_WAVE,
]

batch_types = [
    Types.ANGULAR_DIMENSION,
    Types.ANNOTATION,
    Types.ARC_ARROW,
    Types.ARROW,
    Types.BATCH,
    Types.CIRCULAR_GRID,
    Types.DIMENSION,
    Types.DOTS,
    Types.HEX_GRID,
    Types.LACE,
    Types.LINPATH,
    Types.MARKER,
    Types.MIXED_GRID,
    Types.OVERLAP,
    Types.PARALLEL_POLYLINE,
    Types.PATTERN,
    Types.SQUARE_GRID,
    Types.STAR,
    Types.SVG_PATH,
    Types.TURTLE,
]

# Python Version 3.9 cannot handle Union[*drawable_types]
Drawable: TypeAlias = Union[
    Types.ARC,
    Types.ARC_ARROW,
    Types.ARROW,
    Types.ARROW_HEAD,
    Types.BATCH,
    Types.CIRCLE,
    Types.CIRCULAR_GRID,
    Types.DIMENSION,
    Types.DOT,
    Types.DOTS,
    Types.EDGE,
    Types.ELLIPSE,
    Types.FRAGMENT,
    Types.HEX_GRID,
    Types.IMAGE,
    Types.INTERSECTION,
    Types.LACE,
    Types.LINPATH,
    Types.MIXED_GRID,
    Types.OUTLINE,
    Types.OVERLAP,
    Types.PARALLEL_POLYLINE,
    Types.PATTERN,
    Types.PLAIT,
    Types.POLYLINE,
    Types.RECTANGLE,
    Types.SECTION,
    Types.SEGMENT,
    Types.SHAPE,
    Types.SINE_WAVE,
    Types.SQUARE_GRID,
    Types.STAR,
    Types.SVG_PATH,
    Types.TAG,
    Types.TURTLE,
]


anchors = [
    "southeast",
    "southwest",
    "northeast",
    "northwest",
    "south",
    "north",
    "east",
    "west",
    "center",
    "midpoint",
    "left",
    "right",
    "top",
    "bottom",
    "diagonal1",
    "diagonal2",
    "horiz_centerline",
    "vert_centerline",
    "left_of",
    "right_of",
    "above",
    "below",
    "above_left",
    "above_right",
    "below_left",
    "below_right",
    "centered",
    "polar_pos",
    "s",
    "n",
    "e",
    "w",
    "sw",
    "se",
    "nw",
    "ne",
    "c",
    "d1",
    "d",
    "corners",
    "all_anchors",
    "width",
    "height",
]

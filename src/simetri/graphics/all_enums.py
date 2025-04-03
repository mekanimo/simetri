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


# Tag text alignment options
# Used for TikZ. VaLUeS are case sensitive.
class Align(StrEnum):
    """Align is used to set the alignment of the text in tags."""
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
    frames of tag objects."""

    BASE = "base" # FOR TAGS ONLY
    BASE_EAST = "base east" # FOR TAGS ONLY
    BASE_WEST = "base west" # FOR TAGS ONLY
    BOTTOM = "bottom"
    CENTER = "center"
    EAST = "east"
    LEFT = "left"
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
    """ArrowLine is used to set the type of arrow line."""

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
    """Cartesian coordinate system axes."""

    X = "x"
    Y = "y"


# Used for shapes, canvas, and  tags
class BackStyle(StrEnum):
    """BackStyle is used to set the background style of a shape or tag.
    If shape.fill is True, then background will be drawn according to
    the shape.back_style value.
    """

    COLOR = "COLOR"
    COLOR_AND_GRID = "COLOR_AND_GRID"
    EMPTY = "EMPTY"
    GRIDLINES = "GRIDLINES"
    PATTERN = "PATTERN"
    SHADING = "SHADING"
    SHADING_AND_GRID = "SHADING_AND_GRID"


class BlendMode(StrEnum):
    """BlendMode is used to set the blend mode of the colors."""

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
    """ColorSpace is used to set the color space of the colors."""

    CMYK = "CMYK"
    GRAY = "GRAY"
    HCL = "HCL"
    HLS = "HLS"
    HSV = "HSV"
    LAB = "LAB"
    RGB = "RGB"
    YIQ = "YIQ"


class Connection(StrEnum):
    """Connection is used to set the connection type of the shapes."""

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
    """Connector is used to set the connector type of the shapes."""

    ARC = "ARC"
    ARROW_LINE = "ARROW_LINE"
    CURVE = "CURVE"
    ELLIPSE = "ELLIPSE"
    LINE = "LINE"
    DOUBLE_LINE = "DOUBLE_LINE"
    # squigly
    # zigzag
    # squigly_arrow
    # zigzag_arrow
    # double_arrow
    # double_squigly

class ConstraintType(StrEnum):
    """Constraint is used to set the constraint type of the shapes."""

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
    """Used for the LaTeX compiler."""

    LATEX = "LATEX"
    PDFLATEX = "PDFLATEX"
    XELATEX = "XELATEX"
    LUALATEX = "LUALATEX"


class Control(StrEnum):
    """Used for the modifiers of a bounding box"""

    INITIAL = "INITIAL"
    PAUSE = "PAUSE"
    RESTART = "RESTART"
    RESUME = "RESUME"
    STOP = "STOP"


class Conway(StrEnum):
    """Frieze groups in Conway notation."""

    HOP = "HOP"
    JUMP = "JUMP"
    SIDLE = "SIDLE"
    SPINNING_HOP = "SPINNING_HOP"
    SPINNING_JUMP = "SPINNING_JUMP"
    SPINNING_SIDLE = "SPINNING_SIDLE"
    STEP = "STEP"


class CurveMode(StrEnum):
    OPEN = "OPEN"
    CHORD = "CHORD"
    PIE = "PIE"


class Dep(StrEnum):
    """Depend is used to set the dependency of the shapes.
    This is used when shapes are copied or transformed.
    """

    FALSE = "FALSE" # Independent
    TRUE = "TRUE" # Both geometry and style are dependent
    GEOM = "GEOM" # Only geometry is dependent
    STYLE = "STYLE" # Only style is dependent

# Document classes for the output files
# These come from LaTeX
# Canvas uses these classes to generate output files
class DocumentClass(StrEnum):
    """DocumentClass is used to set the class of the document."""

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
    """FillMode is used to set the fill mode of the shape."""

    EVENODD = "even odd"
    NONZERO = "non zero"


class FontFamily(StrEnum):
    """FontFamily is used to set the family of the font."""

    MONOSPACE = "monospace" # \ttfamily, \texttt
    SERIF = "serif"  # serif \rmfamily, \textrm
    SANSSERIF = "sansserif" # \sffamily, \textsf


class FontSize(StrEnum):
    """FontSize is used to set the size of the font."""

    FOOTNOTESIZE = "footnotesize"
    HUGE = "huge" # \huge
    HUGE2 = "Huge" # \Huge
    LARGE = "large" # \large
    LARGE2 = "Large" # \Large
    LARGE3 = "LARGE" # \LARGE
    NORMAL = "normalsize" # \normalsize
    SCRIPTSIZE = "scriptsize" # \scriptsize
    SMALL = "small" # \small
    TINY = "tiny" # \tiny


class FontStretch(StrEnum):
    """FontStretch is used to set the stretch of the font."""

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
    """FontStrike is used to set the strike of the font."""

    OVERLINE = "overline"
    THROUGH = "through"
    UNDERLINE = "underline"


class FontWeight(StrEnum):
    """FontWeight is used to set the weight of the font."""

    BOLD = "bold"
    MEDIUM = "medium"
    NORMAL = "normal"


class FrameShape(StrEnum):
    """FrameShape is used to set the shape of the frame."""

    # frame can be a rectangle, circle, ellipse
    # size is width and height for rectangle,
    # radius for circle
    # (radius_x, radius_y) for ellipse
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
    """Graph is used to set the type of graph."""

    DIRECTED = "DIRECTED"
    DIRECTEDWEIGHTED = "DIRECTEDWEIGHTED"
    UNDIRECTED = "UNDIRECTED"
    UNDIRECTEDWEIGHTED = "UNDIRECTEDWEIGHTED"


# arrow head positions
class HeadPos(StrEnum):
    """Arrow head positions."""

    BOTH = "BOTH"
    END = "END"
    MIDDLE = "MIDDLE"
    START = "START"
    NONE = "NONE"


class IUC(StrEnum):
    """IUC notation for frieze groups."""

    P1 = "P1"
    P11G = "P11G"
    P11M = "P11M"
    P1M1 = "P1M1"
    P2 = "P2"
    P2MG = "P2MG"
    P2MM = "P2MM"


class LineCap(StrEnum):
    """LineCap is used to set the type of line cap."""

    BUTT = "butt"
    ROUND = "round"
    SQUARE = "square"


class LineDashArray(StrEnum):
    """LineDashArray is used to set the type of dashed-line."""

    DASHDOT = "dashdot"
    DASHDOTDOT = "dashdotdot"
    DASHED = "dashed"
    DENSELY_DASHED = "densely dashed"
    DENSELY_DOTTED = "densely dotted"
    DOTTED = "dotted"
    LOOSELY_DASHED = "loosely dashed"
    LOOSELY_DOTTED = "loosely dotted"
    SOLID = "solid"


class LineJoin(StrEnum):
    """LineJoin is used to set the type of line join."""

    BEVEL = "bevel"
    MITER = "miter"
    ROUND = "round"

class LineWidth(StrEnum):
    '''LineWidth is used to set the width of the line.'''
    SEMITHICK = "semithick"
    THICK = "thick"
    THIN = "thin"
    ULTRA_THICK = "ultra thick"
    ULTRA_THIN = "ultra thin"
    VERY_THICK = "very thick"
    VERY_THIN = "very thin"


class MarkerPos(StrEnum):
    """MarkerPos is used to set the position of the marker."""

    CONCAVEHULL = "CONCAVEHULL"
    CONVEXHULL = "CONVEXHULL"
    MAINX = "MAINX"
    OFFSETX = "OFFSETX"


class MarkerType(StrEnum):
    """MarkerType is used to set the type of marker."""

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
    lines."""

    ANGLED = "ANGLED"
    HORIZONTAL = "HORIZONTAL"
    VERTICAL = "VERTICAL"


# Page margins for the output files
# These come from LaTeX
# Canvas uses these margins to generate output files
# Used in Page class


class PageMargins(StrEnum):
    """Page margins for the LaTeX documents."""

    CUSTOM = "custom"
    NARROW = "narrow"
    STANDARD = "standard"
    WIDE = "wide"


# Page numbering for the output files
# These come from LaTeX
# Canvas uses these numbering to generate output files
# Used in Page class


class PageNumbering(StrEnum):
    """Page numbering style for the LaTeX documents."""

    ALPH = "alph"
    ALPHUPPER = "ALPH"
    ARABIC = "arabic"
    NONE = "none"
    ROMAN = "roman"
    ROMAN_UPPER = "ROMAN"


# Page number position for the output files
# These come from LaTeX
# Canvas uses these positions to generate output files
# Used in Page class


class PageNumberPosition(StrEnum):
    """Page number positions for the LaTeX documents."""

    BOTTOM_CENTER = "bottom"
    BOTTOM_LEFT = "bottom left"
    BOTTOM_RIGHT = "bottom right"
    CUSTOM = "custom"
    TOP_CENTER = "top"
    TOP_LEFT = "top left"
    TOP_RIGHT = "top right"


# Page orientations for the output files
# These come from LaTeX
# Canvas uses these orientations to generate output files
# Used in Page class


class PageOrientation(StrEnum):
    """Page orientations for the LaTeX documents."""

    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"


# Page sizes for the output files
# These come from LaTeX
# Canvas uses these sizes to generate output files
# Used in Page class


class PageSize(StrEnum):
    """Page sizes for the LaTeX documents."""

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
    """PathOperation is used to set the type of path operation."""

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
    """PatternType is used to set the type of pattern."""

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

# Tag placement options
class Placement(StrEnum):
    """Placement is used to set the placement of the tags
    relative to another object."""

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
    """Render is used to set the type of rendering."""

    EPS = "EPS"
    PDF = "PDF"
    SVG = "SVG"
    TEX = "TEX"


class Result(StrEnum):
    """Result is used for the return values of the functions."""

    FAILURE = "FAILURE"
    GO = "GO"
    NOPAGES = "NO_PAGES"
    OVERWRITE = "OVERWRITE"
    SAVED = "SAVED"
    STOP = "STOP"
    SUCCESS = "SUCCESS"


class ShadeType(StrEnum):
    """ShadeType is used to set the type of shading."""

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
    """Side is used to with boundary boxes."""

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
    Not implemented yet."""

    INITIAL = "INITIAL"
    PAUSED = "PAUSED"
    RESTARTING = "RESTARTING"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"


# Not implemented yet.
class TexLoc(StrEnum):
    """TexLoc is used to set the location of the TeX related
    objects."""

    DOCUMENT = "DOCUMENT"  # BETWEEN \BEGIN{DOCUMENT} AND \BEGIN{TIKZPICTURE}
    PICTURE = "PICTURE"  # AFTER \BEGIN{PICTURE}
    PREAMBLE = "PREAMBLE"  # BEFORE \BEGIN{DOCUMENT}


class Topology(StrEnum):
    """Topology is used to set the type of topology."""

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
    """Transformation is used to set the type of transformation."""

    GLIDE = "GLIDE"
    MIRROR = "MIRROR"
    ROTATE = "ROTATE"
    SCALE = "SCALE"
    SHEAR = "SHEAR"
    TRANSFORM = "TRANSFORM"
    TRANSLATE = "TRANSLATE"


# object types and subtypes in simetri.graphics
class Types(StrEnum):
    """All objects in simetri.graphics has type and subtype
    properties."""
    # to do: use snake-case for the names
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
    FONTSKETCH = "FONT_SKETCH"
    FONT_STYLE = "FONT_STYLE"
    FRAGMENT = "FRAGMENT"
    FRAGMENT_SKETCH = "FRAGMENT_SKETCH"
    FRAME = "FRAME"
    FRAMESKETCH = "FRAME_SKETCH"
    FRAME_STYLE = "FRAME_STYLE"
    GRADIENT = "GRADIENT"
    GRID = "GRID"
    GRID_STYLE = "GRID_STYLE"
    HANDLE = "HANDLE"
    HEXAGONAL = "HEXAGONAL"
    ICANVAS = "ICANVAS"
    INTERSECTION = "INTERSECTION"
    LABEL = "LABEL"
    LACE = "LACE"
    LACESKETCH = "LACE_SKETCH"
    LINE = "LINE"
    LINEAR = "LINEAR"
    LINE_SKETCH = "LINE_SKETCH"
    LINE_STYLE = "LINE_STYLE"
    LOOM = "LOOM"
    MARKER = "MARKER"
    MARKER_STYLE = "MARKER_STYLE"
    MASK = "MASK"
    NONE = "NONE"
    OBLIQUE = "OBLIQUE"
    OUTLINE = "OUTLINE"
    OVERLAP = "OVERLAP"
    PAGE = "PAGE"
    PAGE_GRID = "PAGE_GRID"
    PARALLEL_POLYLINE = "PARALLEL_POLYLINE"
    PART = "PART"
    PATH = "PATH"
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
    Types.DIMENSION,
    Types.DIVISION,
    Types.DOT,
    Types.DOTS,
    Types.EDGE,
    Types.ELLIPSE,
    Types.FRAGMENT,
    Types.INTERSECTION,
    Types.LACE,
    Types.OUTLINE,
    Types.OVERLAP,
    Types.PARALLEL_POLYLINE,
    Types.PATH,
    Types.PATTERN,
    Types.PLAIT,
    Types.POLYLINE,
    Types.Q_BEZIER,
    Types.RECTANGLE,
    Types.SECTION,
    Types.SEGMENT,
    Types.SHAPE,
    Types.SINE_WAVE,
    Types.STAR,
    Types.SVG_PATH,
    Types.TAG,
    Types.TURTLE
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
    Types.DIMENSION,
    Types.DOTS,
    Types.LACE,
    Types.MARKER,
    Types.OVERLAP,
    Types.PARALLEL_POLYLINE,
    Types.PATH,
    Types.PATTERN,
    Types.STAR,
    Types.SVG_PATH,
    Types.TURTLE
]

# Python Version 3.9 cannot handle Union[*drawable_types]
Drawable: TypeAlias = Union[
    Types.ARC,
    Types.ARC_ARROW,
    Types.ARROW,
    Types.ARROW_HEAD,
    Types.BATCH,
    Types.CIRCLE,
    Types.DIMENSION,
    Types.DOT,
    Types.DOTS,
    Types.EDGE,
    Types.ELLIPSE,
    Types.FRAGMENT,
    Types.INTERSECTION,
    Types.LACE,
    Types.OUTLINE,
    Types.OVERLAP,
    Types.PARALLEL_POLYLINE,
    Types.PATH,
    Types.PATTERN,
    Types.PLAIT,
    Types.POLYLINE,
    Types.RECTANGLE,
    Types.SECTION,
    Types.SEGMENT,
    Types.SHAPE,
    Types.SINE_WAVE,
    Types.STAR,
    Types.SVG_PATH,
    Types.TAG,
    Types.TURTLE
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
            "left",
            "right",
            "top",
            "bottom",
            "diagonal1",
            "diagonal2",
            "horiz_centerline",
            "vert_centerline",
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
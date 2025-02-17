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
    LEFT = "left"
    RIGHT = "right"
    FLUSH_RIGHT = "flush right"
    FLUSH_LEFT = "flush_left"
    FLUSH_CENTER = "flush center"
    JUSTIFY = "justify"

# Anchor points
# Used for TikZ. VaLUeS are case sensitive.
class Anchor(StrEnum):
    """Anchor is used to set the anchor point of the shapes
    relative to the boundary box of shapes/batches or
    frames of tag objects."""

    BASE = "base" # FOR TAGS ONLY
    BASE_EAST = "base_east" # FOR TAGS ONLY
    BASE_WEST = "base_west" # FOR TAGS ONLY
    BOTTOM = "bottom"
    CENTER = "center"
    EAST = "east"
    LEFT = "left"
    MID = "mid"
    MIDEAST = "mideast"
    MIDWEST = "midwest"
    NORTH = "north"
    NORTHEAST = "northeast"
    NORTHWEST = "northwest"
    RIGHT = "right"
    SOUTH = "south"
    SOUTHEAST = "southeast"
    SOUTHWEST = "southwest"
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
    COLORANDGRID = "COLOR_AND_GRID"
    EMPTY = "EMPTY"
    GRIDLINES = "GRIDLINES"
    PATTERN = "PATTERN"
    SHADING = "SHADING"
    SHADINGANDGRID = "SHADING_AND_GRID"


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
    ENDEND = "ENDEND"
    ENDSTART = "ENDSTART"
    FLIPPED = "FLIPPED"
    INTERSECT = "INTERSECT"
    NONE = "NONE"
    OVERLAPS = "OVERLAPS"
    PARALLEL = "PARALLEL"
    STARTEND = "STARTEND"
    STARTSTART = "STARTSTART"
    TOUCHES = "TOUCHES"
    WITHIN = "WITHIN"
    YJOINT = "YJOINT"


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
    SPINNINGHOP = "SPINNINGHOP"
    SPINNINGJUMP = "SPINNINGJUMP"
    SPINNINGSIDLE = "SPINNINGSIDLE"
    STEP = "STEP"


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

    EVENODD = "even_odd"
    NONZERO = "non_zero"


class FontFamily(StrEnum):
    """FontFamily is used to set the family of the font."""

    CURSIVE = "cursive"
    FANTASY = "fantasy"
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
    EXTRACONDENSED = "extracondensed"
    EXTRAEXPANDED = "extraexpanded"
    NORMAL = "normal"
    SEMICONDENSED = "semicondensed"
    SEMIEXPANDED = "semiexpanded"
    ULTRACONDENSED = "ultracondensed"
    ULTRAEXPANDED = "ultraexpanded"


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
    SPLITCIRCLE = "split_circle"
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


class LineJoin(StrEnum):
    """LineJoin is used to set the type of line join."""

    BEVEL = "bevel"
    MITER = "miter"
    ROUND = "round"


class LineWidth(StrEnum):
    ULTRA_THIN = "ultra thin"
    VERY_THIN = "very thin"
    THIN = "thin"
    SEMITHICK = "semithick"
    THICK = "thick"
    VERY_THICK = "very thick"
    ULTRA_THICK = "ultra thick"


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
    HALFCIRCLE = "halfcircle"
    HALFCIRCLE_F = "halfcircle*"
    HALFDIAMOND = "halfdiamond"
    HALFDIAMOND_F = "halfdiamond*"
    HALFSQUARE = "halfsquare"
    HALFSQUARE_F = "halfsquare*"
    HEXAGON = "hexagon"
    HEXAGON_F = "hexagon*"
    INDICES = "indices"
    MINUS = "-"
    OPLUS = "oplus"
    OPLUS_F = "oplus*"
    OTIMES = "otimes"
    OTIMES_F = "otimes*"
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
    CLOSE = "CLOSE"
    CURVE = "CURVE"
    CURVE2 = "CURVE2"
    HLINE = "HLINE"
    LINE = "LINE"
    MOVE = "MOVE"
    RLINE = "REL_LINE"
    RMOVE = "REL_MOVE"
    VLINE = "VLINE"

class PatternType(StrEnum):
    """PatternType is used to set the type of pattern."""

    BRICKS = "bricks"
    CHECKERBOARD = "checkerboard"
    CROSSHATCH = "crosshatch"
    CROSSHATCHDOTS = "crosshatch dots"
    DOTS = "dots"
    FIVEPOINTEDSTARS = "fivepointed stars"
    GRID = "grid"
    VERTICALLINES = "vertical lines"
    HORIZONTALLINES = "horizontal lines"
    NORTHEAST = "north east lines"
    NORTHWEST = "north west lines"
    SIXPOINTEDSTARS = "sixpointed stars"

# Tag placement options
class Placement(StrEnum):
    """Placement is used to set the placement of the tags
    relative to another object."""

    ABOVE = "above"
    BELOW = "below"
    LEFT = "left"
    RIGHT = "right"
    INSIDE = "inside"
    OUTSIDE = "outside"
    ABOVE_LEFT = "above left"
    ABOVE_RIGHT = "above right"
    BELOW_LEFT = "below left"
    BELOW_RIGHT = "below right"
    CENTERED = "centered"


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
    NOPAGES = "NO PAGES"
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
    HCENTER = "HCENTER"
    LEFT = "LEFT"
    MID = "MID"
    RIGHT = "RIGHT"
    TOP = "TOP"
    VCENTER = "VCENTER"


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


# object types and subtypes
class Types(StrEnum):
    """All objects in simetri.graphics has type and subtype
    properties."""

    ANGULARDIMENSION = "ANGULAR_DIMENSION"
    ANNOTATION = "ANNOTATION"
    ARC = "ARC"
    ARCARROW = "ARC_ARROW"
    ARCSKETCH = "ARC_SKETCH"
    ARROW = "ARROW"
    ARROWHEAD = "ARROWHEAD"
    AXIS = "AXIS"
    BATCH = "BATCH"
    BATCHSKETCH = "BATCH_SKETCH"
    BOUNDINGBOX = "BOUNDING_BOX"
    BRACE = "BRACE"
    CANVAS = "CANVAS"
    CIRCLE = "CIRCLE"
    CIRCLESKETCH = "CIRCLE_SKETCH"
    CIRCULARGRID = "CIRCULAR_GRID"
    COLOR = "COLOR"
    CS = "CS"
    CURVE = "CURVE"
    CURVESKETCH = "CURVE_SKETCH"
    DIMENSION = "DIMENSION"
    DIRECTED = "DIRECTED_GRAPH"
    DIVISION = "DIVISION"
    DOT = "DOT"
    DOTS = "DOTS"
    EDGE = "EDGE"
    ELLIPSE = "ELLIPSE"
    ELLIPSESKETCH = "ELLIPSE_SKETCH"
    ELLIPTICARC = "ELLIPTIC_ARC"
    FILLSTYLE = "FILL_STYLE"
    FONT = "FONT"
    FONTSKETCH = "FONT_SKETCH"
    FONTSTYLE = "FONT_STYLE"
    FRAGMENT = "FRAGMENT"
    FRAGMENTSKETCH = "FRAGMENT_SKETCH"
    FRAME = "FRAME"
    FRAMESKETCH = "FRAME_SKETCH"
    FRAMESTYLE = "FRAME_STYLE"
    GRADIENT = "GRADIENT"
    GRID = "GRID"
    GRIDSTYLE = "GRID_STYLE"
    HEXAGONAL = "HEXAGONAL"
    ICANVAS = "ICANVAS"
    INTERSECTION = "INTERSECTION"
    LABEL = "LABEL"
    LACE = "LACE"
    LACESKETCH = "LACE_SKETCH"
    LINE = "LINE"
    LINEAR = "LINEAR"
    LINESKETCH = "LINE_SKETCH"
    LINESSKETCH = "LINES_SKETCH"
    LINESTYLE = "LINE_STYLE"
    LOOM = "LOOM"
    MARKER = "MARKER"
    MARKERSTYLE = "MARKER_STYLE"
    MASK = "MASK"
    NONE = "NONE"
    OBLIQUE = "OBLIQUE"
    OUTLINE = "OUTLINE"
    OVERLAP = "OVERLAP"
    PAGE = "PAGE"
    PAGEGRID = "PAGE_GRID"
    PARALLELPOLYLINE = "PARALLEL_POLYLINE"
    PART = "PART"
    PATH = "PATH"
    PATTERNSKETCH = "PATTERN_SKETCH"
    PATTERNSTYLE = "PATTERN_STYLE"
    PETAL = "PETAL"
    PLAIT = "PLAIT"
    PLAITSKETCH = "PLAIT_SKETCH"
    POINT = "POINT"
    POINTS = "POINTS"
    POLYLINE = "POLYLINE"
    RADIAL = "RADIAL"
    RECTANGLE = "RECTANGLE"
    RECTANGULAR = "RECTANGULAR"
    REGPOLY = "REGPOLY"
    REGPOLYSKETCH = "REGPOLY_SKETCH"
    REGULARPOLYGON = "REGULAR_POLYGON"
    RHOMBIC = "RHOMBIC"
    SECTION = "SECTION"
    SEGMENT = "SEGMENT"
    SEGMENTS = "SEGMENTS"
    SHADESTYLE = "SHADE_STYLE"
    SHAPE = "SHAPE"
    SHAPESKETCH = "SHAPE_SKETCH"
    SHAPESTYLE = "SHAPE_STYLE"
    SKETCH = "SKETCH"
    SKETCHSTYLE = "SKETCH_STYLE"
    SQUARE = "SQUARE"
    STAR = "STAR"
    STYLE = "STYLE"
    TAG = "TAG"
    TAGSTYLE = "TAG_STYLE"
    TEX = "TEX"  # USED FOR GENERATING OUTPUTFILE.TEX
    TEXSKETCH = "TEX_SKETCH"
    TEXT = "TEXT"
    TEXTANCHOR = "TEXT_ANCHOR"
    TEXTANCHORLINE = "TEXT_ANCHORLINE"
    TEXTANCHORPOINT = "TEXT_ANCHORPOINT"
    TAGSKETCH = "TAG_SKETCH"
    THREAD = "THREAD"
    TRIANGLE = "TRIANGLE"
    TURTLE = "TURTLE"
    UNDIRECTED = "UNDIRECTED_GRAPH"
    VERTEX = "VERTEX"
    WARP = "WARP"
    WEFT = "WEFT"
    WEIGHTED = "WEIGHTED_GRAPH"


drawable_types = [
    Types.ARROW,
    Types.ARC,
    Types.ARCARROW,
    Types.ARROWHEAD,
    Types.BATCH,
    Types.CIRCLE,
    Types.DOT,
    Types.DOTS,
    Types.DIMENSION,
    Types.DIVISION,
    Types.EDGE,
    Types.ELLIPSE,
    Types.FRAGMENT,
    Types.INTERSECTION,
    Types.LACE,
    Types.OUTLINE,
    Types.OVERLAP,
    Types.PARALLELPOLYLINE,
    Types.PATH,
    Types.PLAIT,
    Types.POLYLINE,
    Types.RECTANGLE,
    Types.SECTION,
    Types.SEGMENT,
    Types.SHAPE,
    Types.STAR,
    Types.TAG,
    Types.TURTLE
]

shape_types = [
    Types.ARC,
    Types.BRACE,
    Types.CIRCLE,
    Types.CURVE,
    Types.DIVISION,
    Types.ELLIPSE,
    Types.FRAME,
    Types.INTERSECTION,
    Types.LINE,
    Types.POLYLINE,
    Types.SECTION,
    Types.SHAPE,
    Types.ARROWHEAD,
]

batch_types = [
    Types.ANGULARDIMENSION,
    Types.ANNOTATION,
    Types.ARCARROW,
    Types.ARROW,
    Types.BATCH,
    Types.DOTS,
    Types.LACE,
    Types.MARKER,
    Types.OVERLAP,
    Types.PARALLELPOLYLINE,
    Types.STAR,
    Types.DIMENSION,
    Types.TURTLE
]

# Python Version 3.9 cannot handle Union[*drawable_types]
Drawable: TypeAlias = Union[
    Types.ARROW,
    Types.ARC,
    Types.ARCARROW,
    Types.ARROWHEAD,
    Types.BATCH,
    Types.CIRCLE,
    Types.DOT,
    Types.DOTS,
    Types.DIMENSION,
    Types.EDGE,
    Types.ELLIPSE,
    Types.FRAGMENT,
    Types.INTERSECTION,
    Types.LACE,
    Types.OUTLINE,
    Types.OVERLAP,
    Types.PARALLELPOLYLINE,
    Types.PATH,
    Types.PLAIT,
    Types.POLYLINE,
    Types.RECTANGLE,
    Types.SECTION,
    Types.SEGMENT,
    Types.SHAPE,
    Types.STAR,
    Types.TAG,
    Types.TURTLE
]

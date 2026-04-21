"""
This module creates sketch objects with a neutral format for drawing.
Every other format is converted from this format.
If you need to save as a different format, you can use these
sketch objects to convert to the format you need.
Sketches are not meant to be modified.
They preserve the state of graphics objects at the time of drawing.
They are snapshots of the state of the objects and the Canvas at the time of drawing.
"""

from dataclasses import dataclass
from typing import List, Any, Union

import numpy as np
from numpy import ndarray

from ..colors import colors
from .affine import identity_matrix
from .common import common_properties, PointType
from .all_enums import Types, Anchor, FrameShape, CurveMode, TexLoc, Extent
from ..settings.settings import defaults
from ..geometry.geometry import homogenize
from ..helpers.utilities import decompose_transformations, round_symmetric
from .pattern import Pattern, Group
from ..image.image import Image
from ..graphics.bbox import bounding_box

Color = colors.Color

np.set_printoptions(legacy="1.21")


@dataclass
class CircleSketch:
    """CircleSketch is a dataclass for creating a circle sketch object.

    Attributes:
        center (tuple): The center of the circle.
        radius (float): The radius of the circle.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
    """

    center: tuple
    radius: float
    xform_matrix: ndarray = None

    def __post_init__(self):
        """Initialize the CircleSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.CIRCLE_SKETCH
        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()
            center = self.center
        else:
            center = homogenize([self.center])
            center = (center @ self.xform_matrix).tolist()[0][:2]
        self.center = center
        self.closed = True


@dataclass
class EllipseSketch:
    """EllipseSketch is a dataclass for creating an ellipse sketch object.

    Attributes:
        center (tuple): The center of the ellipse.
        x_radius (float): The x-axis radius of the ellipse.
        y_radius (float): The y-axis radius of the ellipse.
        angle (float, optional): The orientation angle. Defaults to 0.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
    """

    center: tuple
    x_radius: float
    y_radius: float
    angle: float = 0  # orientation angle
    xform_matrix: ndarray = None

    def __post_init__(self):
        """Initialize the EllipseSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.ELLIPSE_SKETCH
        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()
            center = self.center
        else:
            center = homogenize([self.center])
            center = (center @ self.xform_matrix).tolist()[0][:2]
        self.center = center
        self.closed = True


@dataclass
class RectangleSketch:
    """RectangleSketch is a dataclass for creating an rectangle sketch object.

    Attributes:
        lower_left (PointType): The center of the ellipse.
        width (PointType); Width of the rectangle
        height (PointType); Height of the rectangle
        angle (float, optional): The orientation angle. Defaults to 0.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
    """

    lower_left: PointType
    width: float
    height: float
    angle: float = 0  # orientation angle
    xform_matrix: ndarray = None

    def __post_init__(self):
        """Initialize the RectangleSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.RECTANGLE_SKETCH
        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()

        self.closed = True

@dataclass
class LineSketch:
    """LineSketch is a dataclass for creating a line sketch object.

    Attributes:
        vertices (list): The vertices of the line.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
    """

    vertices: list
    xform_matrix: ndarray = None

    def __post_init__(self):
        """Initialize the LineSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.LINE_SKETCH

        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()
            vertices = self.vertices
        else:
            vertices = homogenize(self.vertices)
            vertices = vertices @ self.xform_matrix
        self.vertices = [tuple(x) for x in vertices[:, :2]]
        self._raw_vertices = self.vertices[:]

    @staticmethod
    def _line_limits(canvas):
        if canvas is None:
            return None
        limits = None
        if canvas.limits is not None:
            limits = tuple(canvas.limits)
        elif canvas._all_vertices:
            bbox = bounding_box(canvas._all_vertices)
            limits = (*bbox.southwest, *bbox.northeast)

        page = getattr(canvas, "active_page", None)
        sketches = getattr(page, "sketches", []) if page is not None else []
        for sketch in sketches:
            if (
                getattr(sketch, "subtype", None) == Types.HELPLINES_SKETCH
                and hasattr(sketch, "pos")
                and hasattr(sketch, "width")
                and hasattr(sketch, "height")
            ):
                x, y = sketch.pos[:2]
                candidate = (x, y, x + sketch.width, y + sketch.height)
                if limits is None:
                    limits = candidate
                else:
                    limits = (
                        min(limits[0], candidate[0]),
                        min(limits[1], candidate[1]),
                        max(limits[2], candidate[2]),
                        max(limits[3], candidate[3]),
                    )

        return limits

    @staticmethod
    def _clip_line_to_rect(start, end, rect, draw_type):
        if rect is None:
            return start, end

        x1, y1 = start[:2]
        x2, y2 = end[:2]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return start, end

        xmin, ymin, xmax, ymax = rect
        t_min = float("-inf")
        t_max = float("inf")

        if abs(dx) < 1e-12:
            if x1 < xmin or x1 > xmax:
                return start, end
        else:
            tx1 = (xmin - x1) / dx
            tx2 = (xmax - x1) / dx
            t_min = max(t_min, min(tx1, tx2))
            t_max = min(t_max, max(tx1, tx2))

        if abs(dy) < 1e-12:
            if y1 < ymin or y1 > ymax:
                return start, end
        else:
            ty1 = (ymin - y1) / dy
            ty2 = (ymax - y1) / dy
            t_min = max(t_min, min(ty1, ty2))
            t_max = min(t_max, max(ty1, ty2))

        if t_min > t_max:
            return start, end

        if draw_type == Extent.INFINITE:
            t0, t1 = t_min, t_max
        elif draw_type == Extent.RAY:
            t0, t1 = max(0.0, t_min), t_max
            if t0 > t1:
                return start, end
        else:
            return start, end

        p0 = (x1 + t0 * dx, y1 + t0 * dy)
        p1 = (x1 + t1 * dx, y1 + t1 * dy)
        return p0, p1

    def populate(self, canvas):
        """Populate rendered vertices for deferred draw types (RAY/INFINITE)."""
        extent = getattr(self, "extent", getattr(self, "draw_type", Extent.SEGMENT))
        if not isinstance(extent, Extent) and extent is not None:
            extent = Extent(extent)

        self.vertices = self._raw_vertices[:]
        if extent not in [Extent.RAY, Extent.INFINITE]:
            return

        limits = self._line_limits(canvas)
        start, end = self._clip_line_to_rect(
            self._raw_vertices[0], self._raw_vertices[1], limits, extent
        )
        self.vertices = [start, end]


@dataclass
class PatternSketch:
    """PatternSketch is a dataclass for creating a pattern sketch object.

    Attributes:
        pattern Pattern: The pattern object.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
    """

    pattern: Pattern = None
    xform_matrix: ndarray = None

    def __post_init__(self):
        """Initialize the PatternSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.PATTERN_SKETCH
        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()
        self.kernel_vertices = self.pattern.kernel.final_coords
        self.all_matrices = self.pattern.composite
        self.count = self.pattern.count
        self.closed = self.pattern.closed


@dataclass
class GroupSketch:
    """GroupSketch is a dataclass for creating a group sketch object.
    Group sketches do not freeze anything. sketch.group is used during
    canvas.save or canvas.display.

    Attributes:
        group Group: The group object.

    """

    group: Group = None
    xform_matrix: ndarray = None

    def __post_init__(self):
        """Initialize the PatternSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.GROUP_SKETCH
        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()


@dataclass
class ImageSketch:
    """ImageSketch is a dataclass for creating an image sketch object.

    Attributes:
        image Image: The Image object.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
    """

    image: Image
    pos: PointType = None
    angle: float = None
    scale: Union[tuple, float] = None
    anchor: Anchor = None
    size: tuple = None
    file_path: str = None
    anchor: Anchor = None
    xform_matrix: ndarray = None

    def __post_init__(self):
        """Initialize the ImageSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.IMAGE_SKETCH

        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()
            self.pos = self.image.pos
            self.size = self.image.size
        else:
            pos = homogenize([self.image.pos])
            self.pos = (pos @ self.xform_matrix).tolist()[0][:2]
            _, _, scale = decompose_transformations(self.xform_matrix)
            w, h = self.image.size
            self.size = scale[0] * w, scale[1] * h
        self.image = self.image.copy()


@dataclass
class LatexSketch:
    """LatexSketch renders a LaTeX math formula inline in SVG via matplotlib mathtext.

    Attributes:
        formula (str): LaTeX math string, e.g. r'\frac{a}{b}'.  Do NOT wrap in $..$.            Text-mode commands are silently mapped to their math-mode equivalents:
            \\texttt → \\mathtt, \\textrm → \\mathrm, \\textbf → \\mathbf,
            \\textit → \\mathit, \\textsf → \\mathsf.        pos (PointType): Canvas-space position (x, y) of the formula's anchor point.
        font_size (int): Font size in points. Defaults to 14.
        font_family (str, optional): Mathtext fontset name. Accepted values (case-insensitive):
            'computer modern'/'cm', 'stix', 'stix sans'/'stixsans',
            'dejavu sans'/'dejavusans' (default), 'dejavu serif'/'dejavuserif'.
            When bold=True and font_family is omitted, defaults to 'stix'.
        font_color: Formula colour — a simetri Color, an (r,g,b) tuple, or any
            matplotlib-accepted colour string (e.g. 'red', '#ff0000'). Defaults to black.
        bold (bool): If True, wraps the entire formula in \\mathbf{}. Defaults to False.
            For partial bold (e.g. only some symbols), put \\mathbf{} directly in the
            formula string instead — STIX will be selected automatically.
        anchor (Anchor): Anchor point of the rendered box. Defaults to Anchor.SOUTHWEST.
        xform_matrix (ndarray, optional): The canvas transformation matrix.
    """

    formula: str
    pos: PointType
    font_size: int = 14
    font_family: str = None
    font_color: object = None
    bold: bool = False
    anchor: Anchor = None
    xform_matrix: ndarray = None
    formula_size: tuple = None  # (W, H) in points, filled by draw_latex

    def __post_init__(self):
        """Initialize the LatexSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.LATEX_SKETCH
        if self.anchor is None:
            self.anchor = Anchor.SOUTHWEST
        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()
            pos = self.pos
        else:
            pos = homogenize([self.pos])
            pos = (pos @ self.xform_matrix).tolist()[0][:2]
        self.pos = pos


@dataclass
class MaskSketch:
    """Sketch-like container for canvas-level mask scope metadata."""

    mask: Any = None
    clip: bool = True
    mask_opacity: float = 1.0
    mask_stops: list = None
    mask_axis: Any = ((0.0, 0.0), (1.0, 0.0))
    mask_units: Any = None
    mask_content_units: Any = None

    def __post_init__(self):
        self.type = Types.SKETCH
        self.subtype = Types.MASK_SKETCH
        self.code = ""
        self.location = TexLoc.NONE
        self._canvas_mask_scope = True
        self._mask_opacity = self.mask_opacity
        self._mask_stops = self.mask_stops
        self._mask_axis = self.mask_axis
        self._mask_units = self.mask_units
        self._mask_content_units = self.mask_content_units


@dataclass
class ShapeSketch:
    """ShapeSketch is a neutral format for drawing.

    It contains geometry (only vertices for shapes) and style properties.
    Style properties are not assigned during initialization.
    They are not meant to be transformed, only to be drawn.
    Sketches have no methods, only data.
    They do not check anything, they just store data.
    They are populated during sketch creation.
    You should make sure the data is correct before creating a sketch.

    Attributes:
        vertices (list, optional): The vertices of the shape. Defaults to None.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
    """

    vertices: list = None
    xform_matrix: ndarray = None

    def __post_init__(self):
        """Initialize the ShapeSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.SHAPE_SKETCH
        if self.xform_matrix is None:
            vertices = self.vertices
            self.xform_matrix = identity_matrix()
        else:
            vertices = homogenize(self.vertices)
            vertices = vertices @ self.xform_matrix
        self.vertices = [tuple(x) for x in vertices[:, :2]]


@dataclass
class BezierSketch:
    """BezierSketch is a dataclass for creating a bezier sketch object.

    Attributes:
        control_points (list): The control points of the bezier curve.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
        mode (CurveMode, optional): The mode of the curve. Defaults to CurveMode.OPEN.
    """

    control_points: list
    xform_matrix: ndarray = None
    mode: CurveMode = CurveMode.OPEN

    def __post_init__(self):
        """Initialize the BezierSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.BEZIER_SKETCH

        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()
            control_points = self.control_points
        else:
            control_points = homogenize(self.control_points)
            control_points = control_points @ self.xform_matrix
        self.control_points = [tuple(x) for x in control_points[:, :3]]
        self.closed = False


@dataclass
class ArcSketch:
    """ArcSketch is a dataclass for creating an arc sketch object.

    Attributes:
        vertices (list, optional): The vertices of the shape. Defaults to None.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
        mode (CurveMode, optional): The mode of the curve. Defaults to CurveMode.OPEN.
    """

    vertices: list = None
    xform_matrix: ndarray = None
    mode: CurveMode = CurveMode.OPEN

    def __post_init__(self):
        """Initialize the ArcSketch object."""
        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()
            vertices = self.vertices
        else:
            vertices = homogenize(self.vertices)
            vertices = vertices @ self.xform_matrix
        self.vertices = [tuple(x) for x in vertices[:, :2]]

        self.type = Types.SKETCH
        self.subtype = Types.ARC_SKETCH
        self.closed = self.mode != CurveMode.OPEN


@dataclass
class BatchSketch:
    """BatchSketch is a dataclass for creating a batch sketch object.

    Attributes:
        sketches (List[Types.SKETCH]): The list of sketches.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
    """

    sketches: List[Types.SKETCH]
    xform_matrix: ndarray = None

    def __post_init__(self):
        """Initialize the BatchSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.BATCH_SKETCH
        self.sketches = self.sketches


@dataclass
class PathSketch:
    """PathSketch is a dataclass for creating a path sketch object.

    Attributes:
        sketches (List[Types.SKETCH]): The list of sketches.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
    """

    sketches: List[Types.SKETCH]
    xform_matrix: ndarray = None

    def __post_init__(self):
        """Initialize the PathSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.PATH_SKETCH
        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()


@dataclass
class LaceSketch:
    """LaceSketch is a dataclass for creating a lace sketch object.

    Attributes:
        fragment_sketches (List[ShapeSketch]): The list of fragment sketches.
        plait_sketches (List[ShapeSketch]): The list of plait sketches.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
    """

    fragment_sketches: List[ShapeSketch]
    plait_sketches: List[ShapeSketch]
    xform_matrix: ndarray = None

    def __post_init__(self):
        """Initialize the LaceSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.LACESKETCH
        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()


@dataclass
class FrameSketch:
    """FrameSketch is a dataclass for creating a frame sketch object.

    Attributes:
        frame_shape (FrameShape, optional): The shape of the frame. Defaults to "rectangle".
        line_width (float, optional): The width of the line. Defaults to 1.
        line_dash_array (list, optional): The dash array for the line. Defaults to None.
        line_color (Color, optional): The color of the line. Defaults to colors.black.
        back_color (Color, optional): The background color. Defaults to colors.white.
        fill (bool, optional): Whether to fill the frame. Defaults to False.
        stroke (bool, optional): Whether to stroke the frame. Defaults to True.
        draw_double (bool, optional): Whether to draw a double line. Defaults to False.
        double (Color, optional): Color of the double lines.
        double_distance (float, optional): The distance between double lines. Defaults to 2.
        inner_sep (float, optional): The inner separation. Defaults to 10.
        outer_sep (float, optional): The outer separation. Defaults to 10.
        smooth (bool, optional): Whether to smooth the frame. Defaults to False.
        rounded_corners (bool, optional): Whether to round the corners. Defaults to False.
        fillet_radius (float, optional): The radius of the fillet. Defaults to 10.
        draw_fillets (bool, optional): Whether to draw fillets. Defaults to False.
        blend_mode (str, optional): The blend mode. Defaults to None.
        gradient (str, optional): The gradient. Defaults to None.
        pattern (str, optional): The pattern. Defaults to None.
        visible (bool, optional): Whether the frame is visible. Defaults to True.
        min_width (float, optional): The minimum width. Defaults to 0.
        min_height (float, optional): The minimum height. Defaults to 0.
        min_radius (float, optional): The minimum radius. Defaults to 0.
    """

    frame_shape: FrameShape = (
        "rectangle"  # default value cannot be FrameShape.RECTANGLE!
    )
    line_width: float = 1
    line_dash_array: list = None
    line_color: Color = colors.black
    back_color: Color = colors.white
    fill: bool = False
    stroke: bool = True
    draw_double: bool = False
    double_color: Color = None
    double_distance: float = 2
    inner_sep: float = 10
    outer_sep: float = 10
    smooth: bool = False
    rounded_corners: bool = False
    fillet_radius: float = 10
    draw_fillets: bool = False
    blend_mode: str = None
    gradient: str = None
    pattern: str = None
    visible: bool = True
    min_width: float = 0
    min_height: float = 0
    min_radius: float = 0

    def __post_init__(self):
        """Initialize the FrameSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.FRAME_SKETCH
        common_properties(self)


@dataclass
class TagSketch:
    """TagSketch is a dataclass for creating a tag sketch object.

    Attributes:
        text (str, optional): The text of the tag. Defaults to None.
        pos (PointType, optional): The position of the tag. Defaults to None.
        anchor (Anchor, optional): The anchor of the tag. Defaults to None.
        font_family (str, optional): The font family. Defaults to None.
        font_size (float, optional): The font size. Defaults to None.
        minimum_width (float, optional): The minimum width. Defaults to None.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
    """

    text: str = None
    pos: PointType = None
    anchor: Anchor = None
    font_family: str = None
    font_size: float = None
    minimum_width: float = None
    xform_matrix: ndarray = None

    def __post_init__(self):
        """Initialize the TagSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.TAG_SKETCH
        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()
            pos = self.pos
        else:
            pos = homogenize([self.pos])
            pos = (pos @ self.xform_matrix).tolist()[0][:2]
        self.pos = pos


@dataclass
class PDFSketch:
    """PDFSketch is a dataclass for creating a PDF sketch object.

    Attributes:
        pdf_path (str): The path to the PDF file.
        pos (PointType, optional): The position of the PDF. Defaults to None.
        scale (float, optional): The scale of the PDF. Defaults to 1.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
    """

    file_path: str
    pos: PointType = None
    scale: float = 1
    angle: float = 0
    size: tuple = None
    anchor: Anchor = Anchor.CENTER
    xform_matrix: ndarray = None

    def __post_init__(self):
        """Initialize the PDFSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.PDF_SKETCH


@dataclass
class RectSketch:
    """RectSketch is a dataclass for creating a rectangle sketch object.

    Attributes:
        pos (PointType): The position of the rectangle.
        width (float): The width of the rectangle.
        height (float): The height of the rectangle.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
    """

    pos: PointType
    width: float
    height: float
    xform_matrix: ndarray = None

    def __post_init__(self):
        """Initialize the RectSketch object.

        Args:
            pos (PointType): The position of the rectangle.
            width (float): The width of the rectangle.
            height (float): The height of the rectangle.
            xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
        """
        self.type = Types.SKETCH
        self.subtype = Types.RECTANGLE_SKETCH
        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()
            pos = self.pos
        else:
            pos = homogenize([self.pos])
            pos = (pos @ self.xform_matrix).tolist()[0][:2]
        self.pos = pos
        h2 = self.height / 2
        w2 = self.width / 2
        self.vertices = [
            (pos[0] - w2, pos[1] - h2),
            (pos[0] + w2, pos[1] - h2),
            (pos[0] + w2, pos[1] + h2),
            (pos[0] - w2, pos[1] + h2),
        ]
        self.closed = True

@dataclass
class HelpLinesSketch:
    spacing: float
    cs_size: float

    def __post_init__(self):
        """Initialize the ShapeSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.HELPLINES_SKETCH

    def populate(self, canvas):
        bbox = bounding_box(canvas._all_vertices)
        x, y = bbox.southwest
        width = bbox.width
        height = bbox.height
        spacing = self.spacing
        if spacing in [None, 0]:
            spacing = defaults["help_lines_spacing"]
            self.spacing = spacing

        d = defaults["help_lines_margin"]
        x1 = round_symmetric(x - d, self.spacing)
        y1 = round_symmetric(y - d, self.spacing)
        w = round_symmetric(width + 2 * d, self.spacing)
        h = round_symmetric(height + 2 * d, self.spacing)
        self.pos = (x1, y1)
        self.width = w
        self.height = h

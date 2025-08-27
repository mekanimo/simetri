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
from .common import common_properties, Point
from .all_enums import Types, Anchor, FrameShape, CurveMode, TexLoc
from ..settings.settings import defaults
from ..geometry.geometry import homogenize
from ..helpers.utilities import decompose_transformations
from .pattern import Pattern
from ..image.image import Image

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
class ImageSketch:
    """ImageSketch is a dataclass for creating an image sketch object.

    Attributes:
        image Image: The Image object.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
    """

    image: Image
    pos: Point = None
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
class TexSketch:
    """TexSketch is a dataclass for inserting code into the tex file.

    Attributes:
        code (str, optional): The code to be inserted. Defaults to None.
        location (TexLoc, optional): The location of the code. Defaults to TexLoc.NONE.

    Returns:
        None
    """

    code: str = None
    location: TexLoc = TexLoc.NONE

    def __post_init__(self):
        """Initialize the TexSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.TEX_SKETCH


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
        pos (Point, optional): The position of the tag. Defaults to None.
        anchor (Anchor, optional): The anchor of the tag. Defaults to None.
        font_family (str, optional): The font family. Defaults to None.
        font_size (float, optional): The font size. Defaults to None.
        minimum_width (float, optional): The minimum width. Defaults to None.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
    """

    text: str = None
    pos: Point = None
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
        pos (Point, optional): The position of the PDF. Defaults to None.
        scale (float, optional): The scale of the PDF. Defaults to 1.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
    """

    file_path: str
    pos: Point = None
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
        pos (Point): The position of the rectangle.
        width (float): The width of the rectangle.
        height (float): The height of the rectangle.
        xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
    """

    pos: Point
    width: float
    height: float
    xform_matrix: ndarray = None

    def __post_init__(self):
        """Initialize the RectSketch object.

        Args:
            pos (Point): The position of the rectangle.
            width (float): The width of the rectangle.
            height (float): The height of the rectangle.
            xform_matrix (ndarray, optional): The transformation matrix. Defaults to None.
        """
        self.type = Types.SKETCH
        self.subtype = Types.RECT_SKETCH
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

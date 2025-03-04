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
from typing import List, Any

import numpy as np
from numpy import ndarray

from ..colors import colors
from .affine import identity_matrix
from .common import common_properties, Point
from .all_enums import Types, Anchor, FrameShape, CurveMode
from ..settings.settings import defaults
from ..geometry.geometry import homogenize
from ..helpers.utilities import decompose_transformations

Color = colors.Color

np.set_printoptions(legacy="1.21")


def get_property2(shape, canvas, prop, batch_attrib=None):
    """To get a property from a shape
    1- Check if the shape has the property assigned (not None)
    2- If not, check if the Canvas has the property assigned (not None)
    3- If not, use the default value"""

    res = getattr(shape, prop)

    if res is None and batch_attrib is not None:
        res = batch_attrib

    elif res is None and canvas is not None:
        res = getattr(canvas, prop)

    if res is None:
        res = defaults[prop]

    return res


@dataclass
class CircleSketch:
    """CircleSketch is a dataclass for creating a circle sketch object."""

    center: tuple
    radius: float
    xform_matrix: ndarray = None

    def __post_init__(self):
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
    """EllipseSketch is a dataclass for creating an ellipse sketch object."""

    center: tuple
    x_radius: float
    y_radius: float
    angle: float = 0 # orientation angle
    xform_matrix: ndarray = None

    def __post_init__(self):
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
    """LineSketch is a dataclass for creating a line sketch object."""

    vertices: list
    xform_matrix: ndarray = None

    def __post_init__(self):
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
class ShapeSketch:
    """Sketch is a neutral format for drawing.
    It contains geometry (only vertices for shapes) and style
    properties.
    Style properties are not assigned during initialization.
    They are not meant to be transformed, only to be drawn.
    Sketches have no methods, only data.
    They do not check anything, they just store data.
    They are populated during sketch creation.
    You should make sure the data is correct before creating a sketch.
    """

    vertices: list = None
    xform_matrix: ndarray = None

    def __post_init__(self):
        self.type = Types.SKETCH
        self.subtype = Types.SHAPE_SKETCH
        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()
            vertices = self.vertices
        else:
            vertices = homogenize(self.vertices)
            vertices = vertices @ self.xform_matrix
        self.vertices = [tuple(x) for x in vertices[:, :2]]

@dataclass
class BezierSketch:
    """BezierSketch is a dataclass for creating a bezier sketch object."""

    control_points: list
    xform_matrix: ndarray = None
    mode: CurveMode = CurveMode.OPEN

    def __post_init__(self):
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
    """ArcSketch is a dataclass for creating an arc sketch object."""

    center: tuple
    start_angle: float
    end_angle: float
    radius: float
    radius2: float = None
    rot_angle: float = 0
    start_point: tuple = None
    xform_matrix: ndarray = None

    mode: CurveMode = CurveMode.OPEN

    def __post_init__(self):
        self.type = Types.SKETCH
        self.subtype = Types.ARC_SKETCH

        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()
            center = self.center
            scale = 1
        else:
            center = homogenize([self.center])
            center = (center @ self.xform_matrix).tolist()[0][:2]
            _, _, scale = decompose_transformations(self.xform_matrix)

            scale = scale[0]
        n = defaults["tikz_nround"]
        self.radius *= round(scale, n)
        if self.radius2 is not None:
            self.radius2 *= round(scale, n)
        self.start_point = round(self.start_point[0], n), round(self.start_point[1], n)
        self.closed = False


@dataclass
class BatchSketch:
    """BatchSketch is a dataclass for creating a batch sketch object."""

    sketches: List[Types.SKETCH]
    xform_matrix: ndarray = None

    def __post_init__(self):
        self.type = Types.SKETCH
        self.subtype = Types.BATCH_SKETCH
        self.sketches = self.sketches


@dataclass
class PathSketch:
    """PathSketch is a dataclass for creating a path sketch object."""

    sketches: List[Types.SKETCH]
    xform_matrix: ndarray = None

    def __post_init__(self):
        self.type = Types.SKETCH
        self.subtype = Types.PATH_SKETCH
        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()


@dataclass
class LaceSketch:
    """LaceSketch is a dataclass for creating a lace sketch object."""

    fragment_sketches: List[ShapeSketch]
    plait_sketches: List[ShapeSketch]
    xform_matrix: ndarray = None

    def __post_init__(self):
        self.type = Types.SKETCH
        self.subtype = Types.LACESKETCH
        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()


@dataclass
class FrameSketch:
    """FrameSketch is a dataclass for creating a frame sketch object."""

    frame_shape: FrameShape = (
        "rectangle"  # default value cannot be FrameShape.RECTANGLE!
    )
    line_width: float = 1
    line_dash_array: list = None
    line_color: Color = colors.black
    back_color: Color = colors.white
    fill: bool = False
    stroke: bool = True
    double: bool = False
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
        self.type = Types.SKETCH
        self.subtype = Types.FRAME_SKETCH
        common_properties(self)


@dataclass
class TagSketch:
    """TagSketch is a dataclass for creating a tag sketch object."""

    text: str = None
    pos: Point = None
    anchor: Anchor = None
    font_family: str = None
    font_size: float = None
    minimum_width: float = None
    xform_matrix: ndarray = None

    def __post_init__(self):
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
class RectSketch:
    """RectSketch is a dataclass for creating a rectangle sketch object."""

    pos: Point
    width: float
    height: float
    xform_matrix: ndarray = None

    def __post_init__(self):
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
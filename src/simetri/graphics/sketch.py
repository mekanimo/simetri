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
from .all_enums import Types, Anchor, FrameShape
from ..settings.settings import defaults
from ..helpers.geometry import homogenize
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
        self.subtype = Types.CIRCLESKETCH
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
    width: float
    height: float
    xform_matrix: ndarray = None

    def __post_init__(self):
        self.type = Types.SKETCH
        self.subtype = Types.ELLIPSESKETCH
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
        self.subtype = Types.LINESKETCH

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
        self.subtype = Types.SHAPESKETCH
        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()
            vertices = self.vertices
        else:
            vertices = homogenize(self.vertices)
            vertices = vertices @ self.xform_matrix
        self.vertices = [tuple(x) for x in vertices[:, :2]]


@dataclass
class ArcSketch:
    """ArcSketch is a dataclass for creating an arc sketch object."""

    center: tuple
    radius: float
    start_angle: float
    end_angle: float
    xform_matrix: ndarray = None

    def __post_init__(self):
        self.type = Types.SKETCH
        self.subtype = Types.ARCSKETCH

        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()
            center = self.center
            scale = 1
        else:
            center = homogenize([self.center])
            center = (center @ self.xform_matrix).tolist()[0][:2]
            _, _, scale = decompose_transformations(self.xform_matrix)

            scale = scale[0]

        self.radius *= scale
        self.center = center
        self.closed = False


@dataclass
class BatchSketch:
    """BatchSketch is a dataclass for creating a batch sketch object."""

    sketches: List[Any]

    def __post_init__(self):
        self.type = Types.SKETCH
        self.subtype = Types.BATCHSKETCH
        self.sketches = self.sketches


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
        self.subtype = Types.FRAMESKETCH
        common_properties(self)


@dataclass
class TagSketch:
    """TagSketch is a dataclass for creating a tag sketch object."""

    text: str = None
    pos: Point = None
    anchor: Anchor = None
    font_name: str = None
    font_size: float = None
    xform_matrix: ndarray = None

    def __post_init__(self):
        self.type = Types.SKETCH
        self.subtype = Types.TAGSKETCH
        if self.xform_matrix is None:
            self.xform_matrix = identity_matrix()
            pos = self.pos
        else:
            pos = homogenize([self.pos])
            pos = (pos @ self.xform_matrix).tolist()[0][:2]
        self.pos = pos

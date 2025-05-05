"""This module contains functions and classes for creating annotations,
arrows, dimensions, etc."""

from dataclasses import dataclass
from math import pi, atan2
from PIL import ImageFont

import fitz
import numpy as np

# from reportlab.pdfbase import pdfmetrics # to do: remove this

from ..graphics.core import Base
from ..graphics.bbox import bounding_box
from ..graphics.points import Points
from ..graphics.batch import Batch
from ..graphics.shape import Shape

from ..graphics.shapes import reg_poly_points_side_length
from ..graphics.common import get_defaults, common_properties, Point, _set_Nones
from ..graphics.all_enums import (
    Types,
    LineJoin,
    Anchor,
    FrameShape,
    HeadPos,
    ArrowLine,
    Placement,
    FontSize,
)
from ..canvas.style_map import shape_style_map, tag_style_map, TagStyle
from ..graphics.affine import identity_matrix
from ..geometry.ellipse import Arc
from ..geometry.geometry import (
    distance,
    line_angle,
    extended_line,
    line_by_point_angle_length,
    mid_point,
)
from .utilities import get_transform, detokenize
from ..colors.swatches import swatches_255
from ..settings.settings import defaults
from ..colors import colors
from .validation import validate_args

Color = colors.Color
array = np.array


def logo(scale=1):
    """Returns the Simetri logo.

    Args:
        scale (int, optional): Scale factor for the logo. Defaults to 1.

    Returns:
        Batch: A Batch object containing the logo shapes.
    """
    w = 10 * scale
    points = [
        (0, 0),
        (-4, 0),
        (-4, 6),
        (1, 6),
        (1, 2),
        (-2, 2),
        (-2, 4),
        (-1, 4),
        (-1, 3),
        (0, 3),
        (0, 5),
        (-3, 5),
        (-3, 1),
        (5, 1),
        (5, -10),
        (0, -10),
        (0, -6),
        (3, -6),
        (3, -8),
        (2, -8),
        (2, -7),
        (1, -7),
        (1, -9),
        (4, -9),
        (4, -5),
        (-4, -5),
        (-4, -1),
        (-1, -1),
        (-1, -3),
        (-2, -3),
        (-2, -2),
        (-3, -2),
        (-3, -4),
        (0, -4),
    ]

    points2 = [
        (1, 0),
        (1, -4),
        (4, -4),
        (4, -3),
        (2, -3),
        (2, -1),
        (3, -1),
        (3, -2),
        (4, -2),
        (4, 0),
    ]

    points = [(x * w, y * w) for x, y in points]
    points2 = [(x * w, y * w) for x, y in points2]
    kernel1 = Shape(points, closed=True)
    kernel2 = Shape(points2, closed=True)
    rad = 1
    line_width = 2
    kernel1.fillet_radius = rad
    kernel2.fillet_radius = rad
    kernel1.line_width = line_width
    kernel2.line_width = line_width
    fill_color = Color(*swatches_255[62][8])
    kernel1.fill_color = fill_color
    kernel2.fill_color = colors.white

    return Batch([kernel1, kernel2])


def convert_latex_font_size(latex_font_size: FontSize):
    """Converts LaTeX font size to a numerical value.

    Args:
        latex_font_size (FontSize): The LaTeX font size.

    Returns:
        int: The corresponding numerical font size.
    """
    d_font_size = {
        FontSize.TINY: 5,
        FontSize.SMALL: 7,
        FontSize.NORMAL: 10,
        FontSize.LARGE: 12,
        FontSize.LARGE2: 14,
        FontSize.LARGE3: 17,
        FontSize.HUGE: 20,
        FontSize.HUGE2: 25,
    }

    return d_font_size[latex_font_size]


def letter_F_points():
    """Returns the points of the capital letter F.

    Returns:
        list: A list of points representing the letter F.
    """
    return [
        (0.0, 0.0),
        (20.0, 0.0),
        (20.0, 40.0),
        (40.0, 40.0),
        (40.0, 60.0),
        (20.0, 60.0),
        (20.0, 80.0),
        (50.0, 80.0),
        (50.0, 100.0),
        (0.0, 100.0),
        (0.0, 0.0),
    ]


def letter_F(scale=1, **kwargs):
    """Returns a Shape object representing the capital letter F.

    Args:
        scale (int, optional): Scale factor for the letter. Defaults to 1.
        **kwargs: Additional keyword arguments for shape styling.

    Returns:
        Shape: A Shape object representing the letter F.
    """
    F = Shape(letter_F_points(), closed=True)
    if scale != 1:
        F.scale(scale)
    for k, v in kwargs.items():
        if k in shape_style_map:
            setattr(F, k, v)
        else:
            raise AttributeError(f"{k}. Invalid attribute!")
    return F


def cube(size: float = 100):
    """Returns a Batch object representing a cube.

    Args:
        size (float, optional): The size of the cube. Defaults to 100.

    Returns:
        Batch: A Batch object representing the cube.
    """
    points = reg_poly_points_side_length((0, 0), 6, size)
    center = (0, 0)
    face1 = Shape([points[0], center] + points[4:], closed=True)
    cube_ = face1.rotate(-2 * pi / 3, (0, 0), reps=2)
    cube_[0].fill_color = Color(0.3, 0.3, 0.3)
    cube_[1].fill_color = Color(0.4, 0.4, 0.4)
    cube_[2].fill_color = Color(0.6, 0.6, 0.6)

    return cube_


def pdf_to_svg(pdf_path, svg_path):
    """Converts a single-page PDF file to SVG.

    Args:
        pdf_path (str): The path to the PDF file.
        svg_path (str): The path to save the SVG file.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    svg = page.get_svg_image()
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)


# To do: use a different name for the Annotation class
# annotation is a label with an arrow
class Annotation(Batch):
    """An Annotation object is a label with an arrow pointing to a specific location.

    Args:
        text (str): The annotation text.
        pos (tuple): The position of the annotation.
        frame (FrameShape): The frame shape of the annotation.
        root_pos (tuple): The root position of the arrow.
        arrow_line (ArrowLine, optional): The type of arrow line. Defaults to ArrowLine.STRAIGHT_END.
        **kwargs: Additional keyword arguments for annotation styling.
    """

    def __init__(
        self, text, pos, frame, root_pos, arrow_line=ArrowLine.STRAIGHT_END, **kwargs
    ):
        self.text = text
        self.pos = pos
        self.frame = frame
        self.root_pos = root_pos
        self.arrow_line = arrow_line
        self.kwargs = kwargs

        super().__init__(subtype=Types.ANNOTATION, **kwargs)


@dataclass
class TagFrame:
    """Frame objects are used with Tag objects to create boxes.

    Args:
        frame_shape (FrameShape, optional): The shape of the frame. Defaults to "rectangle".
        line_width (float, optional): The width of the frame line. Defaults to 1.
        line_dash_array (list, optional): The dash pattern for the frame line. Defaults to None.
        line_join (LineJoin, optional): The line join style. Defaults to "miter".
        line_color (Color, optional): The color of the frame line. Defaults to colors.black.
        back_color (Color, optional): The background color of the frame. Defaults to colors.white.
        fill (bool, optional): Whether to fill the frame. Defaults to False.
        stroke (bool, optional): Whether to stroke the frame. Defaults to True.
        double (bool, optional): Whether to use a double line. Defaults to False.
        double_distance (float, optional): The distance between double lines. Defaults to 2.
        inner_sep (float, optional): The inner separation. Defaults to 10.
        outer_sep (float, optional): The outer separation. Defaults to 10.
        smooth (bool, optional): Whether to smooth the frame. Defaults to False.
        rounded_corners (bool, optional): Whether to use rounded corners. Defaults to False.
        fillet_radius (float, optional): The radius of the fillet. Defaults to 10.
        draw_fillets (bool, optional): Whether to draw fillets. Defaults to False.
        blend_mode (str, optional): The blend mode. Defaults to None.
        gradient (str, optional): The gradient. Defaults to None.
        pattern (str, optional): The pattern. Defaults to None.
        min_width (float, optional): The minimum width. Defaults to None.
        min_height (float, optional): The minimum height. Defaults to None.
        min_size (float, optional): The minimum size. Defaults to None.
    """

    frame_shape: FrameShape = "rectangle"
    line_width: float = 1
    line_dash_array: list = None
    line_join: LineJoin = "miter"
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
    min_width: float = None
    min_height: float = None
    min_size: float = None

    def __post_init__(self):
        self.type = Types.FRAME
        self.subtype = Types.FRAME
        common_properties(self, id_only=True)


class Tag(Base):
    """A Tag object is very similar to TikZ library's nodes. It is a text with a frame.

    Args:
        text (str): The text of the tag.
        pos (Point): The position of the tag.
        font_family (str, optional): The font family. Defaults to None.
        font_size (int, optional): The font size. Defaults to None.
        font_color (Color, optional): The font color. Defaults to None.
        anchor (Anchor, optional): The anchor point. Defaults to Anchor.CENTER.
        bold (bool, optional): Whether the text is bold. Defaults to False.
        italic (bool, optional): Whether the text is italic. Defaults to False.
        text_width (float, optional): The width of the text. Defaults to None.
        placement (Placement, optional): The placement of the tag. Defaults to None.
        minimum_size (float, optional): The minimum size of the tag. Defaults to None.
        minimum_width (float, optional): The minimum width of the tag. Defaults to None.
        minimum_height (float, optional): The minimum height of the tag. Defaults to None.
        frame (TagFrame, optional): The frame of the tag. Defaults to None.
        xform_matrix (array, optional): The transformation matrix. Defaults to None.
        **kwargs: Additional keyword arguments for tag styling.
    """

    def __init__(
        self,
        text: str,
        pos: Point,
        font_family: str = None,
        font_size: int = None,
        font_color: Color = None,
        anchor: Anchor = Anchor.CENTER,
        bold: bool = False,
        italic: bool = False,
        text_width: float = None,
        placement: Placement = None,
        minimum_size: float = None,
        minimum_width: float = None,
        minimum_height: float = None,
        frame=None,
        xform_matrix=None,
        **kwargs,
    ):
        self.__dict__["style"] = TagStyle()
        self.__dict__["_style_map"] = tag_style_map
        self._set_aliases()
        tag_attribs = list(tag_style_map.keys())
        tag_attribs.append("subtype")
        _set_Nones(
            self,
            ["font_family", "font_size", "font_color"],
            [font_family, font_size, font_color],
        )
        validate_args(kwargs, tag_attribs)
        x, y = pos[:2]
        self._init_pos = array([x, y, 1.0])

        self.text = detokenize(text)
        if frame is None:
            self.frame = TagFrame()
        self.type = Types.TAG
        self.subtype = Types.TAG
        self.style = TagStyle()
        self.style.draw_frame = True
        if font_family:
            self.font_family = font_family
        if font_size:
            self.font_size = font_size
        else:
            self.font_size = defaults["font_size"]
        if xform_matrix is None:
            self.xform_matrix = identity_matrix()
        else:
            self.xform_matrix = get_transform(xform_matrix)

        self.anchor = anchor
        self.bold = bold
        self.italic = italic
        self.text_width = text_width
        self.placement = placement
        self.minimum_size = minimum_size
        self.minimum_width = minimum_width
        self.minimum_height = minimum_height
        for k, v in kwargs.items():
            setattr(self, k, v)

        x1, y1, x2, y2 = self.text_bounds()
        w = x2 - x1
        h = y2 - y1
        self.points = Points([(0, 0, 1), (w, 0, 1), (w, h, 1), (0, h, 1)])
        common_properties(self)

    def __setattr__(self, name, value):
        obj, attrib = self.__dict__["_aliasses"].get(name, (None, None))
        if obj:
            setattr(obj, attrib, value)
        else:
            self.__dict__[name] = value

    def __getattr__(self, name):
        obj, attrib = self.__dict__["_aliasses"].get(name, (None, None))
        if obj:
            res = getattr(obj, attrib)
        else:
            try:
                res = super().__getattr__(name)
            except AttributeError:
                res = self.__dict__[name]

        return res

    def _set_aliases(self):
        _aliasses = {}

        for alias, path_attrib in self._style_map.items():
            style_path, attrib = path_attrib
            obj = self
            for attrib_name in style_path.split("."):
                obj = obj.__dict__[attrib_name]

            if obj is not self:
                _aliasses[alias] = (obj, attrib)
        self.__dict__["_aliasses"] = _aliasses

    def _update(self, xform_matrix, reps: int = 0, merge: bool = False):
        if reps == 0:
            self.xform_matrix = self.xform_matrix @ xform_matrix
            res = self
        else:
            tags = [self]
            tag = self
            for _ in range(reps):
                tag = tag.copy()
                tag._update(xform_matrix)
                tags.append(tag)
            res = Batch(tags)

        if merge and reps > 0:
            res = res.merge_shapes()

        return res

    @property
    def pos(self) -> Point:
        """Returns the position of the text.

        Returns:
            Point: The position of the text.
        """
        return (self._init_pos @ self.xform_matrix)[:2].tolist()

    def copy(self) -> "Tag":
        """Returns a copy of the Tag object.

        Returns:
            Tag: A copy of the Tag object.
        """
        tag = Tag(self.text, self.pos, xform_matrix=self.xform_matrix)
        tag._init_pos = self._init_pos
        tag.font_family = self.font_family
        tag.font_size = self.font_size
        tag.font_color = self.font_color
        tag.anchor = self.anchor
        tag.bold = self.bold
        tag.italic = self.italic
        tag.text_width = self.text_width
        tag.placement = self.placement
        tag.minimum_size = self.minimum_size
        tag.minimum_width = self.minimum_width

        return tag

    def text_bounds(self) -> tuple[float, float, float, float]:
        """Returns the bounds of the text.

        Returns:
            tuple: The bounds of the text (xmin, ymin, xmax, ymax).
        """
        d_font_size = {
            FontSize.TINY: 5,
            FontSize.SMALL: 7,
            FontSize.NORMAL: 10,
            FontSize.LARGE: 12,
            FontSize.LARGE2: 14,
            FontSize.LARGE3: 17,
            FontSize.HUGE: 20,
            FontSize.HUGE2: 25,
        }
        mult = 1 # font size multiplier
        if self.font_size is None:
            font_size = defaults["font_size"]
        elif type(self.font_size) in [int, float]:
            font_size = self.font_size
        elif self.font_size in FontSize:
            font_size = convert_latex_font_size(self.font_size)
        else:
            raise ValueError("Invalid font size.")
        try:
            font = ImageFont.truetype(f"{self.font_family}.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
            if self.font_size in d_font_size:
                font_size = d_font_size[self.font_size]
            else:
                font_size = self.font_size
            mult = font_size / 10
        xmin, ymin, xmax, ymax = font.getbbox(self.text)
        width = xmax - xmin
        height = ymax - ymin
        dx = (width * mult) / 2
        dy = (height * mult) / 2
        xmin -= dx
        xmax += dx
        ymin -= dy
        ymax += dy

        return xmin, ymin, xmax, ymax

    @property
    def final_coords(self):
        """Returns the final coordinates of the text.

        Returns:
            array: The final coordinates of the text.
        """
        return self.points.homogen_coords @ self.xform_matrix

    @property
    def b_box(self):
        """Returns the bounding box of the text.

        Returns:
            tuple: The bounding box of the text.
        """
        xmin, ymin, xmax, ymax = self.text_bounds()
        w2 = (xmax - xmin) / 2
        h2 = (ymax - ymin) / 2
        x, y = self.pos
        inner_sep = self.frame.inner_sep
        xmin = x - w2 - inner_sep
        xmax = x + w2 + inner_sep
        ymin = y - h2 - inner_sep
        ymax = y + h2 + inner_sep
        points = [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
        ]
        return bounding_box(points)
    def __str__(self) -> str:
        return f"Tag({self.text})"

    def __repr__(self) -> str:
        return f"Tag({self.text})"


class ArrowHead(Shape):
    """An ArrowHead object is a shape that represents the head of an arrow.

    Args:
        length (float, optional): The length of the arrow head. Defaults to None.
        width_ (float, optional): The width of the arrow head. Defaults to None.
        points (list, optional): The points defining the arrow head. Defaults to None.
        **kwargs: Additional keyword arguments for arrow head styling.
    """

    def __init__(
        self, length: float = None, width_: float = None, points: list = None, **kwargs
    ):
        length, width_ = get_defaults(
            ["arrow_head_length", "arrow_head_width"], [length, width_]
        )
        if points is None:
            w2 = width_ / 2
            points = [(0, 0), (0, -w2), (length, 0), (0, w2)]
        super().__init__(points, closed=True, subtype=Types.ARROW_HEAD, **kwargs)
        self.head_length = length
        self.head_width = width_

        self.kwargs = kwargs


def draw_cs_tiny(canvas, pos=(0, 0), width=25, height=25, neg_width=5, neg_height=5):
    """Draws a tiny coordinate system.

    Args:
        canvas: The canvas to draw on.
        pos (tuple, optional): The position of the coordinate system. Defaults to (0, 0).
        width (int, optional): The length of the x-axis. Defaults to 25.
        height (int, optional): The length of the y-axis. Defaults to 25.
        neg_width (int, optional): The negative length of the x-axis. Defaults to 5.
        neg_height (int, optional): The negative length of the y-axis. Defaults to 5.
    """
    x, y = pos[:2]
    canvas.circle((x, y), 2, fill=False, line_color=colors.gray)
    canvas.draw(Shape([(x - neg_width, y), (x + width, y)]), line_color=colors.gray)
    canvas.draw(Shape([(x, y - neg_height), (x, y + height)]), line_color=colors.gray)


def draw_cs_small(canvas, pos=(0, 0), width=80, height=100, neg_width=5, neg_height=5):
    """Draws a small coordinate system.

    Args:
        canvas: The canvas to draw on.
        pos (tuple, optional): The position of the coordinate system. Defaults to (0, 0).
        width (int, optional): The length of the x-axis. Defaults to 80.
        height (int, optional): The length of the y-axis. Defaults to 100.
        neg_width (int, optional): The negative length of the x-axis. Defaults to 5.
        neg_height (int, optional): The negative length of the y-axis. Defaults to 5.
    """
    x, y = pos[:2]
    x_axis = arrow(
        (-neg_width + x, y), (width + 10 + x, y), head_length=8, head_width=2
    )
    y_axis = arrow(
        (x, -neg_height + y), (x, height + 10 + y), head_length=8, head_width=2
    )
    canvas.draw(x_axis, line_width=1)
    canvas.draw(y_axis, line_width=1)


def arrow(
    p1,
    p2,
    head_length=10,
    head_width=4,
    line_width=1,
    line_color=colors.black,
    fill_color=colors.black,
    centered=False,
):
    """Return an arrow from p1 to p2.

    Args:
        p1 (tuple): The starting point of the arrow.
        p2 (tuple): The ending point of the arrow.
        head_length (int, optional): The length of the arrow head. Defaults to 10.
        head_width (int, optional): The width of the arrow head. Defaults to 4.
        line_width (int, optional): The width of the arrow line. Defaults to 1.
        line_color (Color, optional): The color of the arrow line. Defaults to colors.black.
        fill_color (Color, optional): The fill color of the arrow head. Defaults to colors.black.
        centered (bool, optional): Whether the arrow is centered. Defaults to False.

    Returns:
        Batch: A Batch object containing the arrow shapes.
    """
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    dx = x2 - x1
    dy = y2 - y1
    angle = atan2(dy, dx)
    body = Shape(
        [(x1, y1), (x2, y2)],
        closed=False,
        line_color=line_color,
        fill_color=fill_color,
        line_width=line_width,
    )
    w2 = head_width / 2
    head = Shape(
        [(-head_length, w2), (0, 0), (-head_length, -w2)],
        closed=True,
        line_color=line_color,
        fill_color=fill_color,
        line_width=line_width,
    )
    head.rotate(angle)
    if centered:
        head.translate(*mid_point((x1, y1), (x2, y2)))
    else:
        head.translate(x2, y2)
    return Batch([body, head])


class ArcArrow(Batch):
    """An ArcArrow object is an arrow with an arc.

    Args:
        center (Point): The center of the arc.
        radius (float): The radius of the arc.
        start_angle (float): The starting angle of the arc.
        end_angle (float): The ending angle of the arc.
        xform_matrix (array, optional): The transformation matrix. Defaults to None.
        **kwargs: Additional keyword arguments for arc arrow styling.
    """

    def __init__(
        self,
        center: Point,
        radius: float,
        start_angle: float,
        end_angle: float,
        xform_matrix: array = None,
        **kwargs,
    ):
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
        # create the arc
        self.arc = Arc(center, radius, start_angle, end_angle)
        self.arc.fill = False
        # create arrow_head1
        self.arrow_head1 = ArrowHead()
        # create arrow_head2
        self.arrow_head2 = ArrowHead()
        start = self.arc.start_point
        end = self.arc.end_point
        self.points = [center, start, end]

        self.arrow_head1.translate(-1 * self.arrow_head1.head_length, 0)
        self.arrow_head1.rotate(start_angle - pi / 2)
        self.arrow_head1.translate(*start)
        self.arrow_head2.translate(-1 * self.arrow_head2.head_length, 0)
        self.arrow_head2.rotate(end_angle + pi / 2)
        self.arrow_head2.translate(*end)
        items = [self.arc, self.arrow_head1, self.arrow_head2]
        super().__init__(items, subtype=Types.ARC_ARROW, **kwargs)
        for k, v in kwargs.items():
            if k in shape_style_map:
                setattr(self, k, v)  # we should check for valid values here
            else:
                raise AttributeError(f"{k}. Invalid attribute!")
        self.xform_matrix = get_transform(xform_matrix)


class Arrow(Batch):
    """An Arrow object is a line with an arrow head.

    Args:
        p1 (Point): The starting point of the arrow.
        p2 (Point): The ending point of the arrow.
        head_pos (HeadPos, optional): The position of the arrow head. Defaults to HeadPos.END.
        head (Shape, optional): The shape of the arrow head. Defaults to None.
        **kwargs: Additional keyword arguments for arrow styling.
    """

    def __init__(
        self,
        p1: Point,
        p2: Point,
        head_pos: HeadPos = HeadPos.END,
        head: Shape = None,
        **kwargs,
    ):
        self.p1 = p1
        self.p2 = p2
        self.head_pos = head_pos
        self.head = head
        self.kwargs = kwargs
        length = distance(p1, p2)
        angle = line_angle(p1, p2)
        self.line = Shape([(0, 0), (length, 0)])
        if head is None:
            self.head = ArrowHead()
        else:
            self.head = head
        if self.head_pos == HeadPos.END:
            x = length
            self.head.translate(x - self.head.head_length, 0)
            self.head.rotate(angle)
            self.line.rotate(angle)
            self.line.translate(*p1)
            self.head.translate(*p1)
            self.heads = [self.head]
        elif self.head_pos == HeadPos.START:
            self.head = [None]
        elif self.head_pos == HeadPos.BOTH:
            self.head2 = ArrowHead()
            self.head2.rotate(pi)
            self.head2.translate(self.head2.head_length, 0)
            self.head2.rotate(angle)
            self.head2.translate(*p1)
            x = length
            self.head.translate(x - self.head.head_length, 0)
            self.head.rotate(angle)
            self.line.rotate(angle)
            self.line.translate(*p1)
            self.head.translate(*p1)
            self.heads = [self.head, self.head2]
        elif self.head_pos == HeadPos.NONE:
            self.heads = [None]

        items = [self.line] + self.heads
        super().__init__(items, subtype=Types.ARROW, **kwargs)


class AngularDimension(Batch):
    """An AngularDimension object is a dimension that represents an angle.

    Args:
        center (Point): The center of the angle.
        radius (float): The radius of the angle.
        start_angle (float): The starting angle.
        end_angle (float): The ending angle.
        ext_angle (float): The extension angle.
        gap_angle (float): The gap angle.
        text_offset (float, optional): The text offset. Defaults to None.
        gap (float, optional): The gap. Defaults to None.
        **kwargs: Additional keyword arguments for angular dimension styling.
    """

    def __init__(
        self,
        center: Point,
        radius: float,
        start_angle: float,
        end_angle: float,
        ext_angle: float,
        gap_angle: float,
        text_offset: float = None,
        gap: float = None,
        **kwargs,
    ):
        text_offset, gap = get_defaults(["text_offset", "gap"], [text_offset, gap])
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.ext_angle = ext_angle
        self.gap_angle = gap_angle
        self.text_offset = text_offset
        self.gap = gap
        super().__init__(subtype=Types.ANGULAR_DIMENSION, **kwargs)


class Dimension(Batch):
    """A Dimension object is a line with arrows and a text.

    Args:
        text (str): The text of the dimension.
        p1 (Point): The starting point of the dimension.
        p2 (Point): The ending point of the dimension.
        ext_length (float): The length of the extension lines.
        ext_length2 (float, optional): The length of the second extension line. Defaults to None.
        orientation (Anchor, optional): The orientation of the dimension. Defaults to None.
        text_pos (Anchor, optional): The position of the text. Defaults to Anchor.CENTER.
        text_offset (float, optional): The offset of the text. Defaults to 0.
        gap (float, optional): The gap. Defaults to None.
        reverse_arrows (bool, optional): Whether to reverse the arrows. Defaults to False.
        reverse_arrow_length (float, optional): The length of the reversed arrows. Defaults to None.
        parallel (bool, optional): Whether the dimension is parallel. Defaults to False.
        ext1pnt (Point, optional): The first extension point. Defaults to None.
        ext2pnt (Point, optional): The second extension point. Defaults to None.
        scale (float, optional): The scale factor. Defaults to 1.
        font_size (int, optional): The font size. Defaults to 12.
        **kwargs: Additional keyword arguments for dimension styling.
    """

    # To do: This is too long and convoluted. Refactor it.
    def __init__(
        self,
        text: str,
        p1: Point,
        p2: Point,
        ext_length: float,
        ext_length2: float = None,
        orientation: Anchor = None,
        text_pos: Anchor = Anchor.CENTER,
        text_offset: float = 0,
        gap: float = None,
        reverse_arrows: bool = False,
        reverse_arrow_length: float = None,
        parallel: bool = False,
        ext1pnt: Point = None,
        ext2pnt: Point = None,
        scale: float = 1,
        font_size: int = 12,
        **kwargs,
    ):
        ext_length2, gap, reverse_arrow_length = get_defaults(
            ["ext_length2", "gap", "rev_arrow_length"],
            [ext_length2, gap, reverse_arrow_length],
        )
        if text == "":
            self.text = str(distance(p1, p2) / scale)
        else:
            self.text = text
        self.p1 = p1
        self.p2 = p2
        self.ext_length = ext_length
        self.ext_length2 = ext_length2
        self.orientation = orientation
        self.text_pos = text_pos
        self.text_offset = text_offset
        self.gap = gap
        self.reverse_arrows = reverse_arrows
        self.reverse_arrow_length = reverse_arrow_length
        self.kwargs = kwargs
        self.ext1 = None
        self.ext2 = None
        self.ext3 = None
        self.arrow1 = None
        self.arrow2 = None
        self.dim_line = None
        self.mid_line = None
        self.ext1pnt = ext1pnt
        self.ext2pnt = ext2pnt
        x1, y1 = p1[:2]
        x2, y2 = p2[:2]

        # px1_1 : extension1 point 1
        # px1_2 : extension1 point 2
        # px2_1 : extension2 point 1
        # px2_2 : extension2 point 2
        # px3_1 : extension3 point 1
        # px3_2 : extension3 point 2
        # pa1 : arrow point 1
        # pa2 : arrow point 2
        # ptext : text point
        super().__init__(subtype=Types.DIMENSION, **kwargs)
        dist_tol = defaults["dist_tol"]
        if font_size is not None:
            self.font_size = font_size
        if parallel:
            if orientation is None:
                orientation = Anchor.NORTHEAST

            if orientation == Anchor.NORTHEAST:
                angle = line_angle(p1, p2) + pi / 2
            elif orientation == Anchor.NORTHWEST:
                angle = line_angle(p1, p2) + pi / 2
            elif orientation == Anchor.SOUTHEAST:
                angle = line_angle(p1, p2) - pi / 2
            elif orientation == Anchor.SOUTHWEST:
                angle = line_angle(p1, p2) + pi / 2
            if self.ext1pnt is None:
                px1_1 = line_by_point_angle_length(p1, angle, self.gap)[1]
            else:
                px1_1 = self.ext1pnt
            px1_2 = line_by_point_angle_length(p1, angle, self.gap + self.ext_length)[1]
            if self.ext2pnt is None:
                px2_1 = line_by_point_angle_length(p2, angle, self.gap)[1]
            else:
                px2_1 = self.ext2pnt
            px2_2 = line_by_point_angle_length(p2, angle, self.gap + self.ext_length)[1]

            pa1 = line_by_point_angle_length(px1_2, angle, self.gap * -1.5)[1]
            pa2 = line_by_point_angle_length(px2_2, angle, self.gap * -1.5)[1]

            self.text_pos = mid_point(pa1, pa2)
            self.dim_line = Arrow(pa1, pa2, head_pos=HeadPos.BOTH)
            self.ext1 = Shape([px1_1, px1_2])
            self.ext2 = Shape([px2_1, px2_2])
            self.append(self.dim_line)
            self.append(self.ext1)
            self.append(self.ext2)

        else:
            if abs(x1 - x2) < dist_tol:
                # vertical line
                if self.orientation is None:
                    orientation = Anchor.EAST

                if orientation in [Anchor.WEST, Anchor.SOUTHWEST, Anchor.NORTHWEST]:
                    x = x1 - self.gap
                    px1_1 = (x, y1)
                    px1_2 = (x - ext_length, y1)
                    px2_1 = (x, y2)
                    px2_2 = (x - ext_length, y2)
                    x = px1_2[0] + self.gap * 1.5
                    pa1 = (x, y1)
                    pa2 = (x, y2)
                elif orientation in [Anchor.EAST, Anchor.SOUTHEAST, Anchor.NORTHEAST]:
                    x = x1 + self.gap
                    px1_1 = (x, y1)
                    px1_2 = (x + ext_length, y1)
                    px2_1 = (x, y2)
                    px2_2 = (x + ext_length, y2)
                    x = px1_2[0] - self.gap * 1.5
                    pa1 = (x, y1)
                    pa2 = (x, y2)
                elif orientation == Anchor.CENTER:
                    pa1 = (x1, y1)
                    pa2 = (x1, y2)
                x = pa1[0]
                if orientation in [Anchor.SOUTHWEST, Anchor.SOUTHEAST]:
                    px3_1 = pa2
                    y = y2 - self.ext_length2
                    px3_2 = (x, y)
                    self.ext3 = Shape([px3_1, px3_2])
                    self.text_pos = (x, y - self.text_offset)
                elif orientation in [Anchor.NORTHWEST, Anchor.NORTHEAST]:
                    px3_1 = pa1
                    y = y1 + self.ext_length2
                    px3_2 = (x, y)
                    self.ext3 = Shape([px3_1, px3_2])
                    self.text_pos = (x, y + self.text_offset)
                elif orientation == Anchor.SOUTH:
                    px3_1 = pa2
                    y = y2 - self.ext_length2
                    px3_2 = (x, y)
                    self.ext3 = Shape([px3_1, px3_2])
                    self.text_pos = (x, y - self.text_offset)
                elif orientation == Anchor.NORTH:
                    px3_2 = pa1
                    y = y2 + self.ext_length2
                    px3_1 = (x, y)
                    self.ext3 = Shape([px3_1, px3_2])
                    self.text_pos = (x, y + self.text_offset)
                else:
                    self.text_pos = (x, y1 - (y1 - y2) / 2)
                if orientation not in [Anchor.CENTER, Anchor.NORTH, Anchor.SOUTH]:
                    if self.ext1pnt is None:
                        self.ext1 = Shape([px1_1, px1_2])
                    else:
                        self.ext1 = Shape([ext1pnt, px1_2])
                    if self.ext2pnt is None:
                        self.ext2 = Shape([px2_1, px2_2])
                    else:
                        self.ext2 = Shape([ext2pnt, px2_2])
            elif abs(y1 - y2) < dist_tol:
                # horizontal line
                if self.orientation is None:
                    orientation = Anchor.SOUTH

                if orientation in [Anchor.SOUTH, Anchor.SOUTHWEST, Anchor.SOUTHEAST]:
                    y = y1 - self.gap
                    px1_1 = (x1, y)
                    px1_2 = (x1, y - ext_length)
                    px2_1 = (x2, y)
                    px2_2 = (x2, y - ext_length)
                    y = px1_2[1] + self.gap * 1.5
                    pa1 = (x1, y)
                    pa2 = (x2, y)
                elif orientation in [Anchor.NORTH, Anchor.NORTHWEST, Anchor.NORTHEAST]:
                    y = y1 + self.gap
                    px1_1 = (x1, y)
                    px1_2 = (x1, y + ext_length)
                    px2_1 = (x2, y)
                    px2_2 = (x2, y + ext_length)
                    y = px1_2[1] - self.gap * 1.5
                    pa1 = (x1, y)
                    pa2 = (x2, y)
                elif orientation in [Anchor.WEST, Anchor.EAST]:
                    pa1 = (x1, y1)
                    pa2 = (x2, y2)
                    if orientation == Anchor.WEST:
                        px3_1 = (pa1[0] - self.ext_length2, pa1[1])
                        px3_2 = pa1
                        self.text_pos = (px3_1[0] - self.text_offset, pa1[1])
                    else:
                        px3_1 = pa2
                        px3_2 = (pa2[0] + self.ext_length2, pa1[1])
                        self.text_pos = (px3_1[0] + self.text_offset, pa1[1])
                    self.ext3 = Shape([px3_1, px3_2])
                elif orientation == Anchor.CENTER:
                    pa1 = (x1, y1)
                    pa2 = (x2, y2)

                y = pa1[1]
                if orientation in [Anchor.SOUTHWEST, Anchor.NORTHWEST]:
                    px3_1 = pa1
                    x = x1 - self.ext_length2
                    px3_2 = (x, y)
                    self.ext3 = Shape([px3_1, px3_2])
                    self.text_pos = (x - self.text_offset, y)
                elif orientation in [Anchor.NORTHEAST, Anchor.SOUTHEAST]:
                    px3_1 = pa2
                    x = x2 + self.ext_length2
                    px3_2 = (x, y)
                    self.ext3 = Shape([px3_1, px3_2])
                    self.text_pos = (x + self.text_offset, y)
                elif orientation in [Anchor.CENTER, Anchor.NORTH, Anchor.SOUTH]:
                    self.text_pos = (x1 + (x2 - x1) / 2, y)

                if orientation not in [Anchor.CENTER, Anchor.WEST, Anchor.EAST]:
                    if self.ext1pnt is None:
                        self.ext1 = Shape([px1_1, px1_2])
                    else:
                        self.ext1Shape([ext1pnt, px1_2])
                    if self.ext2pnt is None:
                        self.ext2 = Shape([px2_1, px2_2])
                    else:
                        self.ext2 = Shape([ext2pnt, px2_2])

            if self.reverse_arrows:
                dist = self.reverse_arrow_length
                p2 = extended_line(dist, [pa1, pa2])[1]
                self.arrow1 = Arrow(p2, pa2)
                p2 = extended_line(dist, [pa2, pa1])[1]
                self.arrow2 = Arrow(p2, pa1)
                self.append(self.arrow1)
                self.append(self.arrow2)
                self.mid_line = Shape([pa1, pa2])
                self.append(self.mid_line)
                dist = self.text_offset + self.reverse_arrow_length
                if orientation in [Anchor.EAST, Anchor.NORTHEAST, Anchor.NORTH]:

                    self.text_pos = extended_line(dist, [pa1, pa2])[1]
                else:
                    self.text_pos = extended_line(dist, [pa2, pa1])[1]
            else:
                self.dim_line = Arrow(pa1, pa2, head_pos=HeadPos.BOTH)
                self.append(self.dim_line)
            if self.ext1 is not None:
                self.append(self.ext1)

            if self.ext2 is not None:
                self.append(self.ext2)

            if self.ext3 is not None:
                self.append(self.ext3)

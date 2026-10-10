"""Illustration helpers for tags, text, figures, and the logo."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import dataclass
from math import pi
from typing import Any

import numpy as np
import pymupdf as fitz
from numpy.typing import NDArray
from PIL import ImageFont

from ..base.all_enums import (
    Align,
    Anchor,
    FontFamily,
    FontSize,
    FrameShape,
    InPlace,
    LineJoin,
    Placement,
    TransformationType,
    Types,
)
from ..base.common import PointType
from ..base.core import Base, _next_xform_matrix, _Targets
from ..coloring import colors
from ..coloring.swatches import swatches_255
from ..config.settings import runtime_defaults
from ..geom.bbox import BoundingBox, bounding_box
from ..geom.homogenize import homogenize
from ..geom.matrices import identity_matrix
from ..geom.nonlinear.path import Path2D, shape_to_path2d
from ..group.batch import Group
from ..render.style_map import shape_style_map, tag_style_map
from ..shapes.geom_items import reg_poly_points_side_length
from ..shapes.points import Points
from ..shapes.shape import Shape
from .utilities import get_transform
from .validation import validate_args

# Flat Tag style attribute names (no StyleMixin / nested TagStyle aliases).
TAG_STYLE_ATTRS: tuple[str, ...] = tuple(tag_style_map.keys())
# Tag names stored on ``Tag.frame`` via properties (not a second copy on Tag).
_TAG_FRAME_PROPERTY_NAMES: frozenset[str] = frozenset(
    {
        "back_color",
        "double_color",
        "double_distance",
        "draw_double",
        "draw_fillets",
        "fill",
        "fill_color",
        "fillet_radius",
        "frame_inner_sep",
        "frame_min_height",
        "frame_min_size",
        "frame_min_width",
        "frame_outer_sep",
        "frame_shape",
        "line_color",
        "line_dash_array",
        "line_join",
        "line_width",
        "smooth",
        "stroke",
    }
)

Color = colors.Color
array = np.array

def logo(scale: int | float = 1) -> Group:
    """Returns the Simetri logo.

    Args:
        scale (int, optional): Scale factor for the logo. Defaults to 1.

    Returns:
        Group: A Group object containing the logo shapes.

    Examples:
        >>> import simetri.graphics as sg
        >>> logo = sg.logo(1)
        >>> len(logo)
        2
        >>> logo[1].vertices
        ((10.0, 0.0), (10.0, -40.0), (40.0, -40.0), (40.0, -30.0), (20.0, -30.0), (20.0, -10.0), (30.0, -10.0), (30.0, -20.0), (40.0, -20.0), (40.0, 0.0))
        >>> logo[0].vertices
        ((0.0, 0.0), (-40.0, 0.0), (-40.0, 60.0), (10.0, 60.0), (10.0, 20.0), (-20.0, 20.0), (-20.0, 40.0), (-10.0, 40.0), (-10.0, 30.0), (0.0, 30.0), (0.0, 50.0), (-30.0, 50.0), (-30.0, 10.0), (50.0, 10.0), (50.0, -100.0), (0.0, -100.0), (0.0, -60.0), (30.0, -60.0), (30.0, -80.0), (20.0, -80.0), (20.0, -70.0), (10.0, -70.0), (10.0, -90.0), (40.0, -90.0), (40.0, -50.0), (-40.0, -50.0), (-40.0, -10.0), (-10.0, -10.0), (-10.0, -30.0), (-20.0, -30.0), (-20.0, -20.0), (-30.0, -20.0), (-30.0, -40.0), (0.0, -40.0))
    """
    w = 10 * scale
    points = [  # noqa
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
        (3, -2),  # noqa
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
    kernel1.draw_fillets = True
    kernel2.draw_fillets = True
    kernel1.line_width = line_width
    kernel2.line_width = line_width
    fill_color = Color(*swatches_255[62][8])
    kernel1.fill_color = fill_color
    kernel2.fill_color = colors.white

    return Group([kernel1, kernel2])


def convert_latex_font_size(latex_font_size: FontSize) -> float:
    """Converts LaTeX font size to a numerical value.

    Args:
        latex_font_size (FontSize): The LaTeX font size.

    Returns:
        float: The corresponding numerical font size.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.convert_latex_font_size(sg.FontSize.TINY)
        5
    """
    return latex_font_size_to_pt(latex_font_size)


def latex_font_size_to_pt(latex_font_size: FontSize) -> float:
    """Convert a LaTeX font-size name to an approximate point size.

    Args:
        latex_font_size (FontSize): Named LaTeX font size.

    Returns:
        float: Approximate size in points.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.latex_font_size_to_pt(sg.FontSize.TINY)
        5
        >>> sg.latex_font_size_to_pt(sg.FontSize.NORMAL)
        10
    """
    d_font_size = {
        FontSize.MINISCULE: 4,
        FontSize.TINY: 5,
        FontSize.SCRIPTSIZE: 6,
        FontSize.FOOTNOTESIZE: 7,
        FontSize.SMALL: 8,
        FontSize.NORMAL: 10,
        FontSize.LARGE: 11,
        FontSize.LARGE2: 12,
        FontSize.LARGE3: 14,
        FontSize.HUGE: 17,
        FontSize.HUGE2: 20,
    }

    return d_font_size[latex_font_size]


def default_font_size_pt(key: str) -> float:
    """Point size for a ``defaults`` font-size entry (LaTeX name or number).

    Args:
        key (str): Key into ``defaults``.

    Returns:
        float: Font size in points.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.default_font_size_pt('index_font_size')
        8
    """
    size = runtime_defaults[key]
    if isinstance(size, (int, float)):
        return float(size)
    return latex_font_size_to_pt(FontSize(size))


def letter_F_points() -> list[tuple[float, float]]:
    """Returns the points of the capital letter F.

    Returns:
        list: A list of points representing the letter F.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.letter_F_points()
        [(0.0, 0.0), (20.0, 0.0), (20.0, 40.0), (40.0, 40.0), (40.0, 60.0), (20.0, 60.0), (20.0, 80.0), (50.0, 80.0), (50.0, 100.0), (0.0, 100.0), (0.0, 0.0)]
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


def letter_F(scale: int | float = 1, **kwargs: object) -> Shape:
    """Returns a Shape object representing the capital letter F.

    Args:
        scale (int, optional): Scale factor for the letter. Defaults to 1.
        **kwargs: Additional keyword arguments for shape styling.

    Returns:
        Shape: A Shape object representing the letter F.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.letter_F().vertices
        ((0.0, 0.0), (20.0, 0.0), (20.0, 40.0), (40.0, 40.0), (40.0, 60.0), (20.0, 60.0), (20.0, 80.0), (50.0, 80.0), (50.0, 100.0), (0.0, 100.0))
        >>> sg.letter_F(2).vertices
        ((0.0, 0.0), (40.0, 0.0), (40.0, 80.0), (80.0, 80.0), (80.0, 120.0), (40.0, 120.0), (40.0, 160.0), (100.0, 160.0), (100.0, 200.0), (0.0, 200.0))
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


def cube(size: float = 100) -> Group:
    """Returns a Group object representing a cube.

    Args:
        size (float, optional): The size of the cube. Defaults to 100.

    Returns:
        Group: A Group object representing the cube.

    Examples:
        >>> import simetri.graphics as sg
        >>> cube = sg.cube(50)
        >>> len(cube)
        3
        >>> [tuple(round(c, 6) for c in face.midpoint[:2]) for face in cube]
        [(12.5, -21.650635), (-25.0, 0.0), (12.5, 21.650635)]
    """
    points = reg_poly_points_side_length(6, size, (0, 0))
    center = (0, 0)
    face1 = Shape([points[0], center] + points[4:], closed=True)
    cube_ = face1.rotate(-2 * pi / 3, (0, 0), reps=2)
    cube_[0].fill_color = Color(0.3, 0.3, 0.3)
    cube_[1].fill_color = Color(0.4, 0.4, 0.4)
    cube_[2].fill_color = Color(0.6, 0.6, 0.6)

    return cube_


def get_pdf_dimensions(pdf_path: str) -> tuple[float, float] | None:
    """Return width and height in points for the first PDF page.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        ``(width, height)`` in points, or ``None`` on error.

    Examples:
        >>> import contextlib
        >>> import io
        >>> import simetri.graphics as sg
        >>> buf = io.StringIO()
        >>> with contextlib.redirect_stdout(buf):
        ...     result = sg.get_pdf_dimensions('__missing__.pdf')
        >>> result is None
        True
        >>> 'not found' in buf.getvalue().lower()
        True
    """
    try:
        doc = fitz.open(pdf_path)
        if not doc.page_count:
            print("PDF document contains no pages.")
            return None

        page = doc.load_page(0)  # Load the first page (index 0)
        width = page.rect.width
        height = page.rect.height
        doc.close()
        return width, height
    except fitz.FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    except (fitz.FileDataError, RuntimeError, ValueError) as e:
        print(f"An error occurred: {e}")
        return None


def get_image_dimensions_from_pdf_pages(
    pdf_path: str,
) -> list | None:
    """Extract image dimensions found in a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        list | None: Collected page image dimension lists, or ``None`` on error.

    Note:
        Current implementation initializes ``pages`` but only appends to
        per-page ``images`` lists; callers should treat this as incomplete.
    """
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page_num in range(len(doc)):
            images = []
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)

            for _, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)

                # Extract image dimensions from the extracted image data
                width = base_image["width"]
                height = base_image["height"]
                images.append((width, height))
        doc.close()
        return pages
    except (fitz.FileDataError, RuntimeError, ValueError) as e:
        print(f"An error occurred: {e}")


def pdf_to_svg(pdf_path: str, svg_path: str) -> None:
    """Converts a single-page PDF file to SVG.

    Args:
        pdf_path (str): The path to the PDF file.
        svg_path (str): The path to save the SVG file.

    Note:
        The example is skipped; it needs an existing PDF file.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.pdf_to_svg('a.pdf', 'b.svg')  # doctest: +SKIP
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(0)
    svg = page.get_svg_image()
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)



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
        draw_double (bool, optional): Whether to use a double line. Defaults to False.
        double_distance (float, optional): The distance between double lines. Defaults to 2.
        double (Color, optional): Color of the double lines.
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

    Examples:
        >>> import simetri.graphics as sg
        >>> frame = sg.TagFrame()
        >>> frame.frame_shape
        'rectangle'
        >>> frame.line_width
        1
        >>> frame.stroke
        True
    """

    frame_shape: FrameShape | str = "rectangle"
    line_width: float = 1
    line_dash_array: list | None = None
    line_join: LineJoin | str = "miter"
    line_color: Color = colors.black
    back_color: Color = colors.white
    fill: bool = False
    stroke: bool = True
    draw_double: bool = False
    double_distance: float = 2
    double: Color = colors.black
    inner_sep: float = 10
    outer_sep: float = 10
    smooth: bool = False
    rounded_corners: bool = False
    fillet_radius: float = 10
    draw_fillets: bool = False
    blend_mode: str | None = None
    gradient: str | None = None
    pattern: str | None = None
    min_width: float | None = None
    min_height: float | None = None
    min_size: float | None = None

    def __post_init__(self) -> None:
        """Set frame type metadata after dataclass initialization."""
        self.type = Types.FRAME
        self.subtype = Types.FRAME


class Tag(Base):
    """A Tag object is very similar to TikZ library's nodes. It is a text with a frame.

    Frame paint aliases (``fill``, ``stroke``, ``line_width``, ``fill_color``,
    …) are properties on ``self.frame``. Text styles stay on the Tag.
    See ``TAG_STYLE_ATTRS`` / ``tag_style_map`` for the full set.

    Args:
        text (str): The text of the tag.
        pos (PointType): The position of the tag.
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
        fill (bool, optional): Whether to fill the tag frame. Defaults to None
            (use the default).
        xform_matrix (array, optional): The transformation matrix. Defaults to None.
        **kwargs: Additional keyword arguments for tag styling.

    Examples:
        >>> import simetri.graphics as sg
        >>> tag = sg.Tag('Hello', (0, 0))
        >>> tag.text
        'Hello'
        >>> tag.pos
        [0.0, 0.0]
        >>> str(tag)
        'Tag(Hello)'
    """

    def __init__(
        self,
        text: str,
        pos: PointType,
        font_family: str | FontFamily | None = None,
        font_size: int | float | FontSize | None = None,
        font_color: Color | None = None,
        anchor: Anchor = Anchor.CENTER,
        bold: bool = False,
        italic: bool = False,
        text_width: float | None = None,
        placement: Placement | None = None,
        minimum_size: float | None = None,
        minimum_width: float | None = None,
        minimum_height: float | None = None,
        frame: TagFrame | None = None,
        fill: bool | None = None,
        xform_matrix: NDArray | None = None,
        **kwargs: object,
    ) -> None:
        """Create a framed text tag.

        See the class docstring for argument details.
        """
        tag_attribs = list(TAG_STYLE_ATTRS)
        tag_attribs.append("subtype")
        validate_args(kwargs, tag_attribs)

        x, y = pos[:2]
        self._init_pos = array([x, y, 1.0])
        self.text = text
        self.type = Types.TAG
        self.subtype = Types.TAG
        self.visible = True

        if frame is None:
            self.frame = TagFrame(
                stroke=False,
                inner_sep=runtime_defaults["frame_inner_sep"],
            )
        else:
            self.frame = frame

        for name in TAG_STYLE_ATTRS:
            if name in _TAG_FRAME_PROPERTY_NAMES:
                continue
            setattr(self, name, None)

        self.draw_frame = True
        self.alpha = runtime_defaults["tag_alpha"]
        self.align = runtime_defaults["tag_align"]
        self.blend_mode = runtime_defaults["tag_blend_mode"]

        if font_family is not None:
            self.font_family = font_family
        else:
            self.font_family = runtime_defaults["font_family"]
        if font_size is not None:
            self.font_size = font_size
        else:
            self.font_size = runtime_defaults["font_size"]
        self.font_color = font_color

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

        if fill is not None:
            self.fill = fill

        for key, value in kwargs.items():
            setattr(self, key, value)

        x1, y1, x2, y2 = self.text_bounds()
        w = x2 - x1
        h = y2 - y1
        self.points = Points([(0, 0, 1), (w, 0, 1), (w, h, 1), (0, h, 1)])

    @property
    def fill(self) -> bool:
        """Whether the tag frame is filled. Stored on ``self.frame.fill``."""
        return self.frame.fill

    @fill.setter
    def fill(self, value: object) -> None:
        self.frame.fill = value

    @property
    def stroke(self) -> bool:
        """Whether the tag frame is stroked. Stored on ``self.frame.stroke``."""
        return self.frame.stroke

    @stroke.setter
    def stroke(self, value: object) -> None:
        self.frame.stroke = value

    @property
    def line_width(self) -> float:
        """Frame line width. Stored on ``self.frame.line_width``."""
        return self.frame.line_width

    @line_width.setter
    def line_width(self, value: object) -> None:
        self.frame.line_width = value

    @property
    def line_color(self) -> Color:
        """Frame line color. Stored on ``self.frame.line_color``."""
        return self.frame.line_color

    @line_color.setter
    def line_color(self, value: object) -> None:
        self.frame.line_color = value

    @property
    def line_dash_array(self) -> list | None:
        """Frame dash pattern. Stored on ``self.frame.line_dash_array``."""
        return self.frame.line_dash_array

    @line_dash_array.setter
    def line_dash_array(self, value: object) -> None:
        self.frame.line_dash_array = value

    @property
    def line_join(self) -> LineJoin:
        """Frame line join. Stored on ``self.frame.line_join``."""
        return self.frame.line_join

    @line_join.setter
    def line_join(self, value: object) -> None:
        self.frame.line_join = value

    @property
    def back_color(self) -> Color:
        """Frame fill color. Stored on ``self.frame.back_color``."""
        return self.frame.back_color

    @back_color.setter
    def back_color(self, value: object) -> None:
        self.frame.back_color = value

    @property
    def fill_color(self) -> Color:
        """Alias of ``back_color`` / ``self.frame.back_color``."""
        return self.frame.back_color

    @fill_color.setter
    def fill_color(self, value: object) -> None:
        self.frame.back_color = value

    @property
    def draw_double(self) -> bool:
        """Whether the frame uses a double line. Stored on ``self.frame``."""
        return self.frame.draw_double

    @draw_double.setter
    def draw_double(self, value: object) -> None:
        self.frame.draw_double = value

    @property
    def double_distance(self) -> float:
        """Distance between double frame lines. Stored on ``self.frame``."""
        return self.frame.double_distance

    @double_distance.setter
    def double_distance(self, value: object) -> None:
        self.frame.double_distance = value

    @property
    def double_color(self) -> Color:
        """Color of double frame lines. Stored on ``self.frame.double``."""
        return self.frame.double

    @double_color.setter
    def double_color(self, value: object) -> None:
        self.frame.double = value

    @property
    def draw_fillets(self) -> bool:
        """Whether the frame draws fillets. Stored on ``self.frame``."""
        return self.frame.draw_fillets

    @draw_fillets.setter
    def draw_fillets(self, value: object) -> None:
        self.frame.draw_fillets = value

    @property
    def fillet_radius(self) -> float:
        """Frame fillet radius. Stored on ``self.frame.fillet_radius``."""
        return self.frame.fillet_radius

    @fillet_radius.setter
    def fillet_radius(self, value: object) -> None:
        self.frame.fillet_radius = value

    @property
    def smooth(self) -> bool:
        """Whether the frame is smoothed. Stored on ``self.frame.smooth``."""
        return self.frame.smooth

    @smooth.setter
    def smooth(self, value: object) -> None:
        self.frame.smooth = value

    @property
    def frame_shape(self) -> FrameShape:
        """Frame shape. Stored on ``self.frame.frame_shape``."""
        return self.frame.frame_shape

    @frame_shape.setter
    def frame_shape(self, value: object) -> None:
        self.frame.frame_shape = value

    @property
    def frame_inner_sep(self) -> float:
        """Frame inner separation. Stored on ``self.frame.inner_sep``."""
        return self.frame.inner_sep

    @frame_inner_sep.setter
    def frame_inner_sep(self, value: object) -> None:
        self.frame.inner_sep = value

    @property
    def frame_outer_sep(self) -> float:
        """Frame outer separation. Stored on ``self.frame.outer_sep``."""
        return self.frame.outer_sep

    @frame_outer_sep.setter
    def frame_outer_sep(self, value: object) -> None:
        self.frame.outer_sep = value

    @property
    def frame_min_width(self) -> float | None:
        """Frame minimum width. Stored on ``self.frame.min_width``."""
        return self.frame.min_width

    @frame_min_width.setter
    def frame_min_width(self, value: object) -> None:
        self.frame.min_width = value

    @property
    def frame_min_height(self) -> float | None:
        """Frame minimum height. Stored on ``self.frame.min_height``."""
        return self.frame.min_height

    @frame_min_height.setter
    def frame_min_height(self, value: object) -> None:
        self.frame.min_height = value

    @property
    def frame_min_size(self) -> float | None:
        """Frame minimum size. Stored on ``self.frame.min_size``."""
        return self.frame.min_size

    @frame_min_size.setter
    def frame_min_size(self, value: object) -> None:
        self.frame.min_size = value

    def _update(
        self,
        xform_matrix: NDArray[np.float64],
        reps: int = 0,
        take: slice | None = None,
        incr: float
        | tuple[float, float]
        | tuple[callable, Any]
        | tuple[InPlace, Any]
        | NDArray
        | Sequence[Sequence[float]]
        | None = None,
        dyn_ref: Callable | None = None,
        merge: bool = False,
        xform_type: TransformationType | None = None,
    ) -> Tag | Group:
        if take is not None:
            raise ValueError(
                "Tag._update does not support take=; transform the whole tag."
            )
        if reps == 0:
            self.xform_matrix = self.xform_matrix @ xform_matrix
            return self
        tags = [self]
        tag = self
        if dyn_ref:
            pattern = Group()
            pattern.elements = tags
            targets = _Targets(self, pattern)
        else:
            targets = None
        for i in range(reps):
            tag = tag.copy()
            if targets is not None:
                targets.active = tag
            xform_matrix = _next_xform_matrix(
                xform_matrix, xform_type, incr, dyn_ref, targets, i
            )
            tag._update(xform_matrix)
            tags.append(tag)
        res = Group(tags)
        if merge:
            res = res.merge_shapes()
        return res

    @property
    def pos(self) -> PointType:
        """Returns the position of the text.

        Returns:
            PointType: The position of the text.

        Examples:
            >>> import simetri.graphics as sg
            >>> tag = sg.Tag('A', (0, 0))
            >>> tag.pos
            [0.0, 0.0]
            >>> tag.translate(40, 0)
            Tag(A)
            >>> tag.pos
            [40.0, 0.0]
        """
        return (self._init_pos @ self.xform_matrix)[:2].tolist()

    def copy(self, **kwargs: object) -> Tag:
        """Returns a copy of the Tag object.

        Returns:
            Tag: A copy of the Tag object.

        Examples:
            >>> import simetri.graphics as sg
            >>> tag = sg.Tag('Hello', (0, 0))
            >>> copied = tag.copy()
            >>> copied.text
            'Hello'
            >>> copied.pos
            [0.0, 0.0]
            >>> copied is tag
            False
        """
        tag = Tag(self.text, self.pos, xform_matrix=self.xform_matrix)
        tag._init_pos = self._init_pos
        tag.frame = copy(self.frame)
        tag.placement = self.placement
        tag.minimum_size = self.minimum_size
        tag.minimum_width = self.minimum_width
        tag.minimum_height = self.minimum_height
        for name in TAG_STYLE_ATTRS:
            setattr(tag, name, getattr(self, name))

        for key, value in kwargs.items():
            setattr(tag, key, value)

        return tag

    def text_bounds(self) -> tuple[float, float, float, float]:
        """Returns the bounds of the text.

        Returns:
            tuple: The bounds of the text (xmin, ymin, xmax, ymax).
        """
        if self.font_size is None:
            font_size = runtime_defaults["font_size"]
        elif type(self.font_size) in [int, float]:
            font_size = self.font_size
        elif self.font_size in FontSize:
            font_size = convert_latex_font_size(self.font_size)
        else:
            raise ValueError("Invalid font size.")
        if isinstance(self.font_family, FontFamily):
            if self.font_family == FontFamily.MONOSPACE:
                font_name = runtime_defaults["mono_font"]
            elif self.font_family == FontFamily.SANSSERIF:
                font_name = runtime_defaults["sans_font"]
            else:
                font_name = runtime_defaults["main_font"]
        else:
            font_name = self.font_family

        if not isinstance(font_name, str) or not font_name:
            raise ValueError(f"Invalid font family for Tag: {font_name!r}")

        normalized_font_name = font_name.strip().lower()
        font_resource_map = {
            "courier new": "cour.ttf",
            "times new roman": "times.ttf",
            "arial": "arial.ttf",
        }
        if normalized_font_name in font_resource_map:
            font_resource = font_resource_map[normalized_font_name]
        elif normalized_font_name.endswith(".ttf"):
            font_resource = font_name
        else:
            font_resource = f"{font_name}.ttf"

        font = ImageFont.truetype(font_resource, int(font_size))
        xmin, ymin, xmax, ymax = font.getbbox(self.text)
        width = xmax - xmin
        height = ymax - ymin
        xmin, ymin, xmax, ymax = 0, 0, width, height

        return xmin, ymin, xmax, ymax

    @property
    def final_coords(self) -> NDArray[np.float64]:
        """Returns the final coordinates of the text.

        Returns:
            array: The final coordinates of the text.
        """
        return self.points.homogen_coords @ self.xform_matrix

    @property
    def b_box(self) -> BoundingBox:
        """Returns the bounding box of the text.

        Horizontal placement matches SVG/TikZ tag framing: west/east anchors
        first, otherwise ``align`` (default LEFT is left-edged at ``pos``),
        otherwise centered on ``pos``. Vertical extent is centered on ``pos``
        (``dominant-baseline="middle"``).

        Returns:
            BoundingBox: Axis-aligned box including ``frame.inner_sep``.
        """
        xmin, ymin, xmax, ymax = self.text_bounds()
        text_width = xmax - xmin
        text_height = ymax - ymin
        if self.text_width is not None:
            text_width = self.text_width
        if self.minimum_width is not None:
            text_width = max(text_width, self.minimum_width)

        w2 = text_width / 2
        h2 = text_height / 2
        x, y = self.pos[:2]
        inner_sep = self.frame.inner_sep
        effective_anchor = (
            self.anchor
            if self.anchor is not None
            else runtime_defaults["anchor"]
        )
        effective_align = (
            self.align
            if self.align is not None
            else runtime_defaults["tag_align"]
        )

        if effective_anchor in (
            Anchor.WEST,
            Anchor.SOUTHWEST,
            Anchor.NORTHWEST,
        ):
            xmin = x - inner_sep
            xmax = x + text_width + inner_sep
        elif effective_anchor in (
            Anchor.EAST,
            Anchor.SOUTHEAST,
            Anchor.NORTHEAST,
        ):
            xmin = x - text_width - inner_sep
            xmax = x + inner_sep
        elif effective_align in (Align.LEFT, Align.FLUSH_LEFT):
            xmin = x - inner_sep
            xmax = x + text_width + inner_sep
        elif effective_align in (Align.RIGHT, Align.FLUSH_RIGHT):
            xmin = x - text_width - inner_sep
            xmax = x + inner_sep
        else:
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

    @property
    def all_vertices(self) -> list[PointType]:
        """Returns all the vertices of the tag.

        Returns:
            list: A list of all the vertices of the tag.
        """
        bbox = self.b_box
        return bbox.corners

    def __str__(self) -> str:
        """Return a readable string representation.

        Returns:
            str: ``Tag(text)`` style string.

        Examples:
            >>> import simetri.graphics as sg
            >>> str(sg.Tag('Hello', (0, 0)))
            'Tag(Hello)'
        """
        return f"Tag({self.text})"

    def __repr__(self) -> str:
        """Return the official string representation.

        Returns:
            str: ``Tag(text)`` style string.

        Examples:
            >>> import simetri.graphics as sg
            >>> repr(sg.Tag('Hello', (0, 0)))
            'Tag(Hello)'
        """
        return f"Tag({self.text})"


def _path2d_for_text(path: Path2D | Shape) -> Path2D:
    """Return a Path2D copy of ``path`` for text-along-path drawing.

    Raises:
        TypeError: If ``path`` is not a ``Path2D`` or ``Shape``.
        ValueError: If the path has no segments.
    """
    if isinstance(path, Path2D):
        path2d = path.copy()
    elif isinstance(path, Shape):
        path2d = shape_to_path2d(path)
    else:
        raise TypeError("text_path path must be a Path2D or Shape.")
    if not path2d.operations:
        raise ValueError("text_path requires a path with at least one segment.")
    return path2d


class TextPath(Base):
    """Text laid out along a path (SVG ``textPath``, TikZ ``text along path``).

    The guide path is not stroked unless ``draw_path`` is True. Font
    kwargs match ``canvas.text``; Tag frames and anchors do not apply.

    Args:
        text: String to place along the path.
        path: A ``Path2D`` or a ``Shape`` (converted to a polyline path).
        font_family: Font family. ``None`` uses the default.
        font_size: Font size. ``None`` uses the default.
        font_color: Text color. ``None`` uses the default at draw time.
        bold: Bold type. Defaults to False.
        italic: Italic type. Defaults to False.
        draw_path: If True, also stroke the guide path. Defaults to False.
        xform_matrix: Optional extra transform. Defaults to None.
        **kwargs: Extra attributes stored on the object.

    Examples:
        >>> import simetri.graphics as sg
        >>> curve = sg.Path2D((0, 0)).quad_to((50, 40), (100, 0))
        >>> item = sg.TextPath("curve", curve)
        >>> item.text
        'curve'
        >>> item.draw_path
        False
        >>> canvas = sg.Canvas()
        >>> canvas.draw(item) is canvas
        True
        >>> canvas.active_page.sketches[-1].subtype.name
        'TEXT_PATH_SKETCH'
    """

    def __init__(
        self,
        text: str,
        path: Path2D | Shape,
        font_family: str | FontFamily | None = None,
        font_size: int | float | FontSize | None = None,
        font_color: Color | None = None,
        bold: bool = False,
        italic: bool = False,
        draw_path: bool = False,
        xform_matrix: NDArray | None = None,
        **kwargs: object,
    ) -> None:
        """Create text along a path. See the class docstring."""
        self.text = text
        self.path = _path2d_for_text(path)
        self.type = Types.TEXT_PATH
        self.subtype = Types.TEXT_PATH
        self.visible = True
        self.closed = self.path.closed
        self.draw_path = draw_path
        self.bold = bold
        self.italic = italic
        if font_family is not None:
            self.font_family = font_family
        else:
            self.font_family = runtime_defaults["font_family"]
        if font_size is not None:
            self.font_size = font_size
        else:
            self.font_size = runtime_defaults["font_size"]
        self.font_color = font_color
        if xform_matrix is None:
            self.xform_matrix = identity_matrix()
        else:
            self.xform_matrix = get_transform(xform_matrix)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _update(
        self,
        xform_matrix: NDArray[np.float64],
        reps: int = 0,
        take: slice | None = None,
        incr: float
        | tuple[float, float]
        | tuple[callable, Any]
        | tuple[InPlace, Any]
        | NDArray
        | Sequence[Sequence[float]]
        | None = None,
        dyn_ref: Callable | None = None,
        merge: bool = False,
        xform_type: TransformationType | None = None,
    ) -> TextPath | Group:
        if take is not None:
            raise ValueError(
                "TextPath._update does not support take=; "
                "transform the whole object."
            )
        if reps == 0:
            self.xform_matrix = self.xform_matrix @ xform_matrix
            return self
        items = [self]
        item = self
        if dyn_ref:
            pattern = Group()
            pattern.elements = items
            targets = _Targets(self, pattern)
        else:
            targets = None
        for i in range(reps):
            item = item.copy()
            if targets is not None:
                targets.active = item
            xform_matrix = _next_xform_matrix(
                xform_matrix, xform_type, incr, dyn_ref, targets, i
            )
            item._update(xform_matrix)
            items.append(item)
        res = Group(items)
        if merge:
            res = res.merge_shapes()
        return res

    @property
    def all_vertices(self) -> list[PointType]:
        """Path vertices after this object's transform."""
        vertices = self.path.all_vertices
        if not vertices:
            return []
        return [
            point[:2]
            for point in (homogenize(vertices) @ self.xform_matrix).tolist()
        ]

    @property
    def b_box(self) -> BoundingBox:
        """Bounding box of the guide path after this object's transform."""
        return bounding_box(self.all_vertices)

    def copy(self, **kwargs: object) -> TextPath:
        """Return a copy of this text-on-path object."""
        copied = TextPath(
            self.text,
            self.path.copy(),
            font_family=self.font_family,
            font_size=self.font_size,
            font_color=self.font_color,
            bold=self.bold,
            italic=self.italic,
            draw_path=self.draw_path,
            xform_matrix=self.xform_matrix.copy(),
        )
        for key, value in kwargs.items():
            setattr(copied, key, value)
        return copied

    def __str__(self) -> str:
        return f"TextPath({self.text})"

    def __repr__(self) -> str:
        return f"TextPath({self.text})"

def draw_cs_tiny(
    canvas: Any,
    pos: PointType = (0, 0),
    width: float = 25,
    height: float = 25,
    neg_width: float = 5,
    neg_height: float = 5,
) -> None:
    """Draws a tiny coordinate system.

    Args:
        canvas: The canvas to draw on.
        pos (tuple, optional): The position of the coordinate system. Defaults to (0, 0).
        width (int, optional): The length of the x-axis. Defaults to 25.
        height (int, optional): The length of the y-axis. Defaults to 25.
        neg_width (int, optional): The negative length of the x-axis. Defaults to 5.
        neg_height (int, optional): The negative length of the y-axis. Defaults to 5.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> sg.draw_cs_tiny(canvas, (0, 0))
        >>> [sketch.subtype.name for sketch in canvas.active_page.sketches]
        ['CIRCLE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH']
    """
    x, y = pos[:2]
    canvas.circle(2, (x, y), fill=False, line_color=colors.gray)
    canvas.draw(
        Shape([(x - neg_width, y), (x + width, y)]), line_color=colors.gray
    )
    canvas.draw(
        Shape([(x, y - neg_height), (x, y + height)]), line_color=colors.gray
    )

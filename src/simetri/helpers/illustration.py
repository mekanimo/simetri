"""Illustration helpers for annotations, tags, arrows, and dimensions.

Examples:
    >>> import simetri.graphics as sg
    >>> tag = sg.Tag("Hello", (0, 0))
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import atan2, hypot, pi

import fitz
import numpy as np
from numpy.typing import NDArray
from PIL import ImageFont

from ..render.style_map import TagStyle, shape_style_map, tag_style_map
from ..coloring import colors
from ..coloring.swatches import swatches_255
from ..geom.geometry import bbox_overlap
from ..geom.nonlinear.ellipse import Arc
from ..geom.geometry import (
    polar_to_cartesian,
)
from ..geom.segments.line_utils import line_by_point_angle_length
from ..geom.points.point_utils import distance, midpoint
from ..geom.segments.line_utils import extended_line, line_angle
from ..geom.vectors import Vector, perp_unit_vector, v_from_points
from ..geom.affine import identity_matrix
from ..base.all_enums import (
    Align,
    Anchor,
    ArrowLine,
    FontFamily,
    FontSize,
    FrameShape,
    HeadPos,
    LineJoin,
    Placement,
    Types,
)
from ..group.batch import Group
from ..geom.bbox import bounding_box
from ..base.common import (
    PointType,
    _set_Nones,
    get_defaults,
)

# from reportlab.pdfbase import pdfmetrics # to do: remove this
from ..base.core import Base, StyleMixin
from ..shapes.points import Points
from ..shapes.shape import Shape
from ..shapes.geom_items import reg_poly_points_side_length
from ..config.settings import defaults
from .label_overlap import LabelRect, resolve_all_overlaps
from .utilities import get_transform
from .validation import validate_args

Color = colors.Color
array = np.array


def logo(scale=1):
    """Returns the Simetri logo.

    Args:
        scale (int, optional): Scale factor for the logo. Defaults to 1.

    Returns:
        Group: A Group object containing the logo shapes.
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

    return Group([kernel1, kernel2])


def convert_latex_font_size(latex_font_size: FontSize):
    """Converts LaTeX font size to a numerical value.

    Args:
        latex_font_size (FontSize): The LaTeX font size.

    Returns:
        int: The corresponding numerical font size.
    """
    return latex_font_size_to_pt(latex_font_size)


def latex_font_size_to_pt(latex_font_size: FontSize) -> float:
    """Convert a LaTeX font-size name to an approximate point size.

    Args:
        latex_font_size (FontSize): Named LaTeX font size.

    Returns:
        float: Approximate size in points.
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
    """
    size = defaults[key]
    if isinstance(size, (int, float)):
        return float(size)
    return latex_font_size_to_pt(FontSize(size))


def sketch_label_font_size_pt(sketch, label_kind: str) -> float:
    """Label font size in points from sketch kwargs or defaults.

    Args:
        sketch: Sketch providing optional font-size attributes.
        label_kind (str): ``index`` or ``vertex``.

    Returns:
        float: Font size in points.
    """
    if label_kind == "index":
        attr = "index_font_size"
    else:
        attr = "vertex_font_size"
    if attr in sketch.__dict__:
        size = sketch.__dict__[attr]
        if size is not None:
            if isinstance(size, (int, float)):
                return float(size)
            return latex_font_size_to_pt(FontSize(size))
    return default_font_size_pt(attr)


def sketch_label_offset(sketch, label_kind: str) -> float:
    """Label radial offset in points from sketch kwargs or defaults.

    Args:
        sketch: Sketch providing optional offset attributes.
        label_kind (str): ``index`` or ``vertex``.

    Returns:
        float: Radial offset in points.
    """
    if label_kind == "index":
        attr = "index_offset"
    else:
        attr = "vertex_offset"
    try:
        return float(object.__getattribute__(sketch, attr))
    except AttributeError:
        return float(defaults[attr])


def sketch_label_font_color(sketch, label_kind: str):
    """Label text color from sketch kwargs or defaults.

    Args:
        sketch: Sketch providing optional font-color attributes.
        label_kind (str): ``index`` or ``vertex``.

    Returns:
        Color: Label text color.
    """
    if label_kind == "index":
        attr = "index_font_color"
    else:
        attr = "vertex_font_color"
    if attr in sketch.__dict__ and sketch.__dict__[attr] is not None:
        return sketch.__dict__[attr]
    return defaults[attr]


def label_halo_color():
    """Stroke/halo color behind vertex index and coordinate labels."""
    return defaults["label_halo_color"]


def label_halo_stroke_width(font_size_pt: float) -> float:
    """SVG halo stroke width / TikZ ``\\contourlength`` in points."""
    scale = float(defaults["label_halo_width_scale"])
    return max(0.2, font_size_pt * scale)


def label_halo_scale() -> float:
    """Legacy scale factor (SVG/TikZ use stroke width / contour length instead)."""
    return float(defaults["label_halo_scale"])


def svg_label_paint_attrs(fill_color, font_size_pt: float) -> str:
    """SVG fill/stroke attributes for halo-backed label text."""
    fill_r, fill_g, fill_b = fill_color.rgb255
    halo_r, halo_g, halo_b = label_halo_color().rgb255
    width = label_halo_stroke_width(font_size_pt)
    return (
        f'fill="rgb({fill_r}, {fill_g}, {fill_b})" '
        f'stroke="rgb({halo_r}, {halo_g}, {halo_b})" '
        f'stroke-width="{width}" paint-order="stroke fill"'
    )


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
    """Returns a Group object representing a cube.

    Args:
        size (float, optional): The size of the cube. Defaults to 100.

    Returns:
        Group: A Group object representing the cube.
    """
    points = reg_poly_points_side_length((0, 0), 6, size)
    center = (0, 0)
    face1 = Shape([points[0], center] + points[4:], closed=True)
    cube_ = face1.rotate(-2 * pi / 3, (0, 0), reps=2)
    cube_[0].fill_color = Color(0.3, 0.3, 0.3)
    cube_[1].fill_color = Color(0.4, 0.4, 0.4)
    cube_[2].fill_color = Color(0.6, 0.6, 0.6)

    return cube_


def get_pdf_dimensions(pdf_path):
    """
    Retrieves the width and height of the first page of a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        tuple: A tuple containing (width, height) in points, or None if an error occurs.
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
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_image_dimensions_from_pdf_pages(pdf_path):
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

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)

                # Extract image dimensions from the extracted image data
                width = base_image["width"]
                height = base_image["height"]
                images.append((width, height))
        doc.close()
        return pages
    except Exception as e:
        print(f"An error occurred: {e}")


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
class Annotation(Group):
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
        self,
        text,
        pos,
        frame,
        root_pos,
        arrow_line=ArrowLine.STRAIGHT_END,
        **kwargs,
    ):
        """Create a text annotation with an arrow.

        See the class docstring for argument details.
        """
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
    """

    frame_shape: FrameShape = "rectangle"
    line_width: float = 1
    line_dash_array: list = None
    line_join: LineJoin = "miter"
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
    blend_mode: str = None
    gradient: str = None
    pattern: str = None
    min_width: float = None
    min_height: float = None
    min_size: float = None

    def __post_init__(self):
        """Set frame type metadata after dataclass initialization."""
        self.type = Types.FRAME
        self.subtype = Types.FRAME


class Tag(Base, StyleMixin):
    """A Tag object is very similar to TikZ library's nodes. It is a text with a frame.

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
        xform_matrix (array, optional): The transformation matrix. Defaults to None.
        **kwargs: Additional keyword arguments for tag styling.
    """

    def __init__(
        self,
        text: str,
        pos: PointType,
        font_family: str | None = None,
        font_size: int | None = None,
        font_color: Color = None,
        anchor: Anchor = Anchor.CENTER,
        bold: bool = False,
        italic: bool = False,
        text_width: float | None = None,
        placement: Placement = None,
        minimum_size: float | None = None,
        minimum_width: float | None = None,
        minimum_height: float | None = None,
        frame=None,
        xform_matrix=None,
        **kwargs,
    ):
        """Create a framed text tag.

        See the class docstring for argument details.
        """
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

        self.text = text
        if frame is None:
            self.frame = TagFrame(stroke=False)
        self.type = Types.TAG
        self.subtype = Types.TAG
        # self.style = TagStyle()
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
        self.visible = True

    def __setattr__(self, name, value):
        """Set an attribute, routing style aliases when present.

        Args:
            name: Attribute name.
            value: Attribute value.
        """
        obj, attrib = self.__dict__["_aliases"].get(name, (None, None))
        if obj:
            setattr(obj, attrib, value)
        else:
            self.__dict__[name] = value

    def __getattr__(self, name):
        """Get an attribute, resolving style aliases when present.

        Args:
            name: Attribute name.

        Returns:
            Resolved attribute value, or ``None`` if missing.
        """
        obj, attrib = self.__dict__["_aliases"].get(name, (None, None))
        if obj:
            res = getattr(obj, attrib)
        else:
            try:
                res = super().__getattr__(name)
            except AttributeError:
                res = self.__dict__.get(name, None)

        return res

    def _set_aliases(self):
        _aliases = {}

        for alias, path_attrib in self._style_map.items():
            style_path, attrib = path_attrib
            obj = self
            for attrib_name in style_path.split("."):
                obj = obj.__dict__[attrib_name]

            if obj is not self:
                _aliases[alias] = (obj, attrib)
        self.__dict__["_aliases"] = _aliases

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
            res = Group(tags)

        if merge and reps > 0:
            res = res.merge_shapes()

        return res

    @property
    def pos(self) -> PointType:
        """Returns the position of the text.

        Returns:
            PointType: The position of the text.
        """
        return (self._init_pos @ self.xform_matrix)[:2].tolist()

    def copy(self, **kwargs) -> "Tag":
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

        for k, v in kwargs.items():
            setattr(tag, k, v)

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
        if self.font_size is None:
            font_size = defaults["font_size"]
        elif type(self.font_size) in [int, float]:
            font_size = self.font_size
        elif self.font_size in FontSize:
            font_size = convert_latex_font_size(self.font_size)
        else:
            raise ValueError("Invalid font size.")
        if isinstance(self.font_family, FontFamily):
            if self.font_family == FontFamily.MONOSPACE:
                font_name = defaults["mono_font"]
            elif self.font_family == FontFamily.SANSSERIF:
                font_name = defaults["sans_font"]
            else:
                font_name = defaults["main_font"]
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
    def final_coords(self):
        """Returns the final coordinates of the text.

        Returns:
            array: The final coordinates of the text.
        """
        return self.points.homogen_coords @ self.xform_matrix

    @property
    def b_box(self):
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
            text_width = max(text_width, self.text_width)

        w2 = text_width / 2
        h2 = text_height / 2
        x, y = self.pos[:2]
        inner_sep = self.frame.inner_sep
        effective_anchor = (
            self.anchor if self.anchor is not None else defaults["anchor"]
        )
        effective_align = (
            self.align if self.align is not None else defaults["tag_align"]
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
    def all_vertices(self):
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
        """
        return f"Tag({self.text})"

    def __repr__(self) -> str:
        """Return the official string representation.

        Returns:
            str: ``Tag(text)`` style string.
        """
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
        self,
        length: float | None = None,
        width_: float | None = None,
        points: list | None = None,
        **kwargs,
    ):
        """Create an arrow head shape.

        See the class docstring for argument details.
        """
        length, width_ = get_defaults(
            ["arrow_head_length", "arrow_head_width"], [length, width_]
        )
        if points is None:
            w2 = width_ / 2
            points = [(0, 0), (0, -w2), (length, 0), (0, w2)]
        super().__init__(
            points, closed=True, subtype=Types.ARROW_HEAD, **kwargs
        )
        self.head_length = length
        self.head_width = width_

        self.kwargs = kwargs


def draw_cs_tiny(
    canvas, pos=(0, 0), width=25, height=25, neg_width=5, neg_height=5
):
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
    canvas.circle(2, (x, y), fill=False, line_color=colors.gray)
    canvas.draw(
        Shape([(x - neg_width, y), (x + width, y)]), line_color=colors.gray
    )
    canvas.draw(
        Shape([(x, y - neg_height), (x, y + height)]), line_color=colors.gray
    )


def draw_cs_small(
    canvas, pos=(0, 0), width=80, height=100, neg_width=5, neg_height=5
):
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
        Group: A Group object containing the arrow shapes.
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
        head.translate(*midpoint((x1, y1), (x2, y2)))
    else:
        head.translate(x2, y2)
    return Group([body, head])


class ArcArrow(Group):
    """An ArcArrow object is an arrow with an arc.

    Args:
        center (PointType): The center of the arc.
        radius (float): The radius of the arc.
        start_angle (float): The starting angle of the arc.
        end_angle (float): The ending angle of the arc.
        xform_matrix (array, optional): The transformation matrix. Defaults to None.
        **kwargs: Additional keyword arguments for arc arrow styling.
    """

    def __init__(
        self,
        center: PointType,
        radius: float,
        start_angle: float,
        end_angle: float,
        xform_matrix: NDArray | None = None,
        **kwargs,
    ):
        """Create an arc with arrow heads at both ends.

        See the class docstring for argument details.

        Raises:
            AttributeError: If an invalid style keyword is provided.
        """
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
        # create the arc
        self.arc = Arc(
            center, radius, start_angle=start_angle, end_angle=end_angle
        )
        self.arc.fill = False
        # create arrow_head1
        self.arrow_head1 = ArrowHead()
        # create arrow_head2
        self.arrow_head2 = ArrowHead()
        start = self.arc[0]
        end = self.arc[-1]
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


class RadialDimension(Group):
    """A RadialDimension object is a dimension that represents a radius.

    Args:
        center (PointType): The center of the circle.
        radius (float): The radius of the circle.
        angle (float): The angle of the dimension line.
        text_offset (float, optional): The offset for the dimension text. Defaults to None.
        gap (float, optional): The gap between the dimension line and the text. Defaults to None.
        **kwargs: Additional keyword arguments for radial dimension styling.
    """

    def __init__(
        self,
        center: PointType,
        radius: float | None = None,
        angle: float = 0,
        text: str = "",
        text_offset: Sequence = (0, 0),
        ext_length: float = 10,
        reverse_arrow: bool = False,
        keep_inside: bool = True,
        gap: float | None = None,
        **kwargs,
    ):
        """Create a radial dimension annotation.

        See the class docstring for argument details.
        """
        text_offset, gap = get_defaults(
            ["text_offset", "gap"], [text_offset, gap]
        )
        self.center = center
        self.radius = radius
        self.angle = angle
        self.text = text
        self.text_offset = text_offset
        self.ext_length = ext_length
        self.reverse_arrow = reverse_arrow
        self.keep_inside = keep_inside
        self.gap = gap
        super().__init__(subtype=Types.RADIAL_DIMENSION, **kwargs)

        p2 = polar_to_cartesian(self.radius, self.angle, self.center)

        self.arrow = Arrow(center, p2)
        self.extension = None
        if self.reverse_arrow:
            self.extension = extended_line(ext_length, [center, p2])

        if self.text == "":
            if self.radius is not None:
                self.text = f"{self.radius:.2f}"
            else:
                self.text = f"r = {distance(center, p2):.2f}"

        self.tag = Tag(self.text, midpoint(center, p2))
        self._items = [self.arrow, self.tag]

        super().__init__(self._items, subtype=Types.RADIAL_DIMENSION, **kwargs)


class Arrow(Group):
    """An Arrow object is a line with an arrow head.

    Args:
        p1 (PointType): The starting point of the arrow.
        p2 (PointType): The ending point of the arrow.
        head_pos (HeadPos, optional): The position of the arrow head. Defaults to HeadPos.END.
        head (Shape, optional): The shape of the arrow head. Defaults to None.
        **kwargs: Additional keyword arguments for arrow styling.
    """

    def __init__(
        self,
        p1: PointType,
        p2: PointType,
        head_pos: HeadPos = HeadPos.END,
        head: Shape = None,
        line_width: float = 1,
        color: Color = colors.black,
        **kwargs,
    ):
        """Create a line arrow with one or more heads.

        See the class docstring for argument details.
        """
        self.p1 = p1
        self.p2 = p2
        self.head_pos = head_pos
        self.head = head
        self.line_width = line_width
        self.color = color
        self.kwargs = kwargs
        length = distance(p1, p2)
        angle = line_angle(p1, p2)
        self.line = Shape(
            [(0, 0), (length, 0)],
            line_width=line_width,
            line_color=self.color,
            **kwargs,
        )
        if head is None:
            self.head = ArrowHead()
            self.head.fill_color = color
            self.head.line_color = color
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


class AngularDimension(Group):
    """An AngularDimension object is a dimension that represents an angle.

    Args:
        center (PointType): The center of the angle.
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
        center: PointType,
        radius: float,
        start_angle: float,
        end_angle: float,
        ext_angle: float,
        gap_angle: float,
        text_offset: float | None = None,
        gap: float | None = None,
        **kwargs,
    ):
        """Create an angular dimension annotation.

        See the class docstring for argument details.
        """
        text_offset, gap = get_defaults(
            ["text_offset", "gap"], [text_offset, gap]
        )
        self.center = center
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.ext_angle = ext_angle
        self.gap_angle = gap_angle
        self.text_offset = text_offset
        self.gap = gap
        super().__init__(subtype=Types.ANGULAR_DIMENSION, **kwargs)


class Dimension(Group):
    """A Dimension object is a line with arrows and a text.

    Args:
        text (str): The text of the dimension.
        p1 (PointType): The starting point of the dimension.
        p2 (PointType): The ending point of the dimension.
        ext_length (float): The length of the extension lines.
        ext_length2 (float, optional): The length of the second extension line. Defaults to None.
        orientation (Anchor, optional): The orientation of the dimension. Defaults to None.
        text_pos (Anchor, optional): The position of the text. Defaults to Anchor.CENTER.
        text_offset (float, optional): The offset of the text. Defaults to 0.
        gap (float, optional): The gap. Defaults to None.
        reverse_arrows (bool, optional): Whether to reverse the arrows. Defaults to False.
        reverse_arrow_length (float, optional): The length of the reversed arrows. Defaults to None.
        parallel (bool, optional): Whether the dimension is parallel. Defaults to False.
        ext1pnt (PointType, optional): The first extension point. Defaults to None.
        ext2pnt (PointType, optional): The second extension point. Defaults to None.
        scale (float, optional): The scale factor. Defaults to 1.
        font_size (int, optional): The font size. Defaults to 12.
        keep_centered (bool, optional): Whether to keep the dimension centered. Defaults to False.
        **kwargs: Additional keyword arguments for dimension styling.
    """

    # To do: This is too long and convoluted. Refactor it.
    def __init__(
        self,
        p1: PointType,
        p2: PointType,
        ext_length: float,
        ext_length2: float | None = None,
        orientation: Anchor = None,
        text: str = "",
        text_pos: Anchor = Anchor.CENTER,
        text_offset: tuple = (0, 0),
        gap: float | None = None,
        reverse_arrows: bool = False,
        reverse_arrow_length: float | None = None,
        parallel: bool = False,
        ext1pnt: PointType = None,
        ext2pnt: PointType = None,
        scale: float = 1,
        font_size: int = 12,
        keep_centered: bool = False,
        text_side: Anchor = None,  # (Anchor.TOP, Anchor.BOTTOM, Anchor.LEFT, Anchor.RIGHT)
        **kwargs,
    ):
        """Create a linear dimension with extension lines and arrows.

        See the class docstring for argument details.
        """
        ext_length2, gap, reverse_arrow_length = get_defaults(
            ["ext_length2", "gap", "rev_arrow_length"],
            [ext_length2, gap, reverse_arrow_length],
        )
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
        self.parallel = parallel
        self.keep_centered = keep_centered
        self.text_side = text_side
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
        self.scale = scale
        x1, y1 = p1[:2]
        x2, y2 = p2[:2]
        min_x = min(x1, x2)
        max_x = max(x1, x2)
        min_y = min(y1, y2)
        max_y = max(y1, y2)
        text_dx, text_dy = self.text_offset

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
        space = gap * 0.75
        if font_size is not None:
            self.font_size = font_size
        if parallel:
            if orientation is None:
                orientation = Anchor.NORTHEAST
            elif (
                orientation == Anchor.NORTHEAST
                or orientation == Anchor.NORTHWEST
            ):
                angle = line_angle(p1, p2) + pi / 2
            elif orientation == Anchor.SOUTHEAST:
                angle = line_angle(p1, p2) - pi / 2
            elif orientation == Anchor.SOUTHWEST:
                angle = line_angle(p1, p2) + pi / 2
            else:
                angle = line_angle(p1, p2) - pi / 2
            if self.ext1pnt is None:
                px1_1 = line_by_point_angle_length(p1, angle, self.gap)[1]
            else:
                px1_1 = self.ext1pnt
            px1_2 = line_by_point_angle_length(
                p1, angle, self.gap + self.ext_length
            )[1]
            if self.ext2pnt is None:
                px2_1 = line_by_point_angle_length(p2, angle, self.gap)[1]
            else:
                px2_1 = self.ext2pnt
            px2_2 = line_by_point_angle_length(
                p2, angle, self.gap + self.ext_length
            )[1]

            pa1 = line_by_point_angle_length(px1_2, angle, -space)[1]
            pa2 = line_by_point_angle_length(px2_2, angle, -space)[1]

            tx, ty = midpoint(pa1, pa2)
            self.text_pos = (tx + text_dx, ty + text_dy)
            if self.text == "":
                self.text = f"{distance(p1, p2):.2f}"

            # Handle reverse_arrows for parallel dimensions
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
                dist = self.text_offset[0] + self.reverse_arrow_length
                if not self.keep_centered:
                    if orientation in [
                        Anchor.EAST,
                        Anchor.NORTHEAST,
                        Anchor.NORTH,
                    ]:
                        if self.text_side == Anchor.BOTTOM:
                            tx, ty = extended_line(dist, [pa2, pa1])[1]
                        else:
                            tx, ty = extended_line(dist, [pa1, pa2])[1]
                            self.text_pos = (tx + text_dx, ty + text_dy)
                    else:
                        tx, ty = extended_line(dist, [pa1, pa2])[1]
                        self.text_pos = (tx + text_dx, ty + text_dy)
            else:
                self.dim_line = Arrow(pa1, pa2, head_pos=HeadPos.BOTH)
                self.append(self.dim_line)

            self.ext1 = Shape([px1_1, px1_2])
            self.ext2 = Shape([px2_1, px2_2])
            self.append(self.ext1)
            self.append(self.ext2)

        else:
            if self.text == "":
                if orientation in (Anchor.NORTH, Anchor.SOUTH):
                    self.text = f"{(max_x - min_x / scale):.2f}"
                elif orientation in [Anchor.EAST, Anchor.WEST]:
                    self.text = f"{(max_y - min_y / scale):.2f}"
                else:
                    self.text = f"{distance(p1, p2):.2f}"
            if abs(x1 - x2) < dist_tol:
                # vertical line
                if self.orientation is None:
                    orientation = Anchor.EAST

                if orientation in [
                    Anchor.WEST,
                    Anchor.SOUTHWEST,
                    Anchor.NORTHWEST,
                ]:
                    x = x1 - self.gap
                    px1_1 = (x, y1)
                    px1_2 = (x - ext_length, y1)
                    px2_1 = (x, y2)
                    px2_2 = (x - ext_length, y2)
                    x = px1_2[0] + space
                    pa1 = (x, y1)
                    pa2 = (x, y2)
                elif orientation in [
                    Anchor.EAST,
                    Anchor.SOUTHEAST,
                    Anchor.NORTHEAST,
                ]:
                    x = x1 + self.gap
                    px1_1 = (x, y1)
                    px1_2 = (x + ext_length, y1)
                    px2_1 = (x, y2)
                    px2_2 = (x + ext_length, y2)
                    x = px1_2[0] - space
                    pa1 = (x, y1)
                    pa2 = (x, y2)
                elif orientation == Anchor.CENTER:
                    pa1 = (x1, y1)
                    pa2 = (x1, y2)
                x = pa1[0]
                if orientation in (Anchor.SOUTHWEST, Anchor.SOUTHEAST):
                    px3_1 = pa2
                    y = y2 - self.ext_length2
                    px3_2 = (x, y)
                    self.ext3 = Shape([px3_1, px3_2])
                    self.text_pos = (x + text_dx, y - text_dy)
                elif orientation in [Anchor.NORTHWEST, Anchor.NORTHEAST]:
                    px3_1 = pa1
                    y = y1 + self.ext_length2
                    px3_2 = (x, y)
                    self.ext3 = Shape([px3_1, px3_2])
                    self.text_pos = (x + text_dx, y + text_dy)
                elif orientation == Anchor.SOUTH:
                    px3_1 = pa2
                    y = y2 - self.ext_length2
                    px3_2 = (x, y)
                    self.ext3 = Shape([px3_1, px3_2])
                    self.text_pos = (x + text_dx, y - text_dy)
                elif orientation == Anchor.NORTH:
                    px3_2 = pa1
                    y = y2 + self.ext_length2
                    px3_1 = (x, y)
                    self.ext3 = Shape([px3_1, px3_2])
                    self.text_pos = (x + text_dx, y + text_dy)
                else:
                    self.text_pos = (x + text_dx, y1 - (y1 - y2) / 2 + text_dy)
                if orientation not in [
                    Anchor.CENTER,
                    Anchor.NORTH,
                    Anchor.SOUTH,
                ]:
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

                if orientation in [
                    Anchor.SOUTH,
                    Anchor.SOUTHWEST,
                    Anchor.SOUTHEAST,
                ]:
                    y = y1 - self.gap
                    px1_1 = (x1, y)
                    px1_2 = (x1, y - ext_length)
                    px2_1 = (x2, y)
                    px2_2 = (x2, y - ext_length)
                    y = px1_2[1] + space
                    pa1 = (x1, y)
                    pa2 = (x2, y)
                elif orientation in [
                    Anchor.NORTH,
                    Anchor.NORTHWEST,
                    Anchor.NORTHEAST,
                ]:
                    y = y1 + self.gap
                    px1_1 = (x1, y)
                    px1_2 = (x1, y + ext_length)
                    px2_1 = (x2, y)
                    px2_2 = (x2, y + ext_length)
                    y = px1_2[1] - space
                    pa1 = (x1, y)
                    pa2 = (x2, y)
                elif orientation in [Anchor.WEST, Anchor.EAST]:
                    pa1 = (x1, y1)
                    pa2 = (x2, y2)
                    if orientation == Anchor.WEST:
                        px3_1 = (pa1[0] - self.ext_length2, pa1[1])
                        px3_2 = pa1
                        self.text_pos = (px3_1[0] - text_dx, pa1[1] + text_dy)
                    else:
                        px3_1 = pa2
                        px3_2 = (pa2[0] + self.ext_length2, pa1[1])
                        self.text_pos = (px3_1[0] + text_dx, pa1[1] + text_dy)
                    self.ext3 = Shape([px3_1, px3_2])
                elif orientation == Anchor.CENTER:
                    pa1 = (x1, y1)
                    pa2 = (x2, y2)

                y = pa1[1]
                if orientation in (Anchor.SOUTHWEST, Anchor.NORTHWEST):
                    px3_1 = pa1
                    x = x1 - self.ext_length2
                    px3_2 = (x, y)
                    self.ext3 = Shape([px3_1, px3_2])
                    self.text_pos = (x + text_dx, y + text_dy)
                elif orientation in [Anchor.NORTHEAST, Anchor.SOUTHEAST]:
                    px3_1 = pa2
                    x = x2 + self.ext_length2
                    px3_2 = (x, y)
                    self.ext3 = Shape([px3_1, px3_2])
                    self.text_pos = (x + text_dx, y + text_dy)
                elif orientation in [Anchor.CENTER, Anchor.NORTH, Anchor.SOUTH]:
                    self.text_pos = (x1 + (x2 - x1) / 2, y)

                if orientation not in (Anchor.CENTER, Anchor.WEST, Anchor.EAST):
                    if self.ext1pnt is None:
                        self.ext1 = Shape([px1_1, px1_2])
                    else:
                        self.ext1 = Shape([ext1pnt, px1_2])
                    if self.ext2pnt is None:
                        self.ext2 = Shape([px2_1, px2_2])
                    else:
                        self.ext2 = Shape([ext2pnt, px2_2])
            else:
                if orientation is Anchor.WEST:
                    leftmost = min_x - self.gap - self.ext_length
                    px1_1 = (min_x - self.gap, min_y)
                    px1_2 = (leftmost, min_y)
                    px2_1 = (max_x + self.gap, max_y)
                    px2_2 = (leftmost, max_y)
                    pa1 = (leftmost + space, min_y)
                    pa2 = (leftmost + space, max_y)
                    self.ext1 = Shape([px1_1, px1_2])
                    self.ext2 = Shape([px2_1, px2_2])
                elif orientation is Anchor.EAST:
                    rightmost = max_x + self.gap + self.ext_length
                    px1_1 = (min_x + self.gap, min_y)
                    px1_2 = (rightmost, min_y)
                    px2_1 = (max_x + self.gap, max_y)
                    px2_2 = (rightmost, max_y)
                    pa1 = (rightmost - space, min_y)
                    pa2 = (rightmost - space, max_y)
                    self.ext1 = Shape([px1_1, px1_2])
                    self.ext2 = Shape([px2_1, px2_2])
                elif orientation is Anchor.NORTH:
                    topmost = max_y + self.gap + self.ext_length
                    if min_x == (x1, y1)[0]:
                        px1_1 = (min_x, y1 + self.gap)
                        px1_2 = (min_x, topmost)

                        px2_1 = (max_x, y2 + self.gap)
                        px2_2 = (max_x, topmost)
                    else:
                        px1_1 = (max_x, y2 + self.gap)
                        px1_2 = (max_x, topmost)

                        px2_1 = (min_x, y1 + self.gap)
                        px2_2 = (min_x, topmost)

                    pa1 = (min_x, topmost - space)
                    pa2 = (max_x, topmost - space)
                    self.ext1 = Shape([px1_1, px1_2])
                    self.ext2 = Shape([px2_1, px2_2])
                elif orientation is Anchor.SOUTH:
                    bottommost = min_y - self.gap - self.ext_length
                    # px1_1 = (min_x, max_y - self.gap)
                    px1_1 = (p1[0], p1[1] - self.gap)
                    # px1_2 = (min_x, bottommost)
                    px1_2 = (p1[0], bottommost)
                    # px2_1 = (max_x, min_y - self.gap)
                    px2_1 = (p2[0], p2[1] - self.gap)
                    # px2_2 = (max_x, bottommost)
                    px2_2 = (p2[0], bottommost)
                    pa1 = (min_x, bottommost + space)
                    pa2 = (max_x, bottommost + space)
                    self.ext1 = Shape([px1_1, px1_2])
                    self.ext2 = Shape([px2_1, px2_2])
                else:
                    pa1 = p1
                    pa2 = p2
                tx, ty = midpoint(pa1, pa2)
                self.text_pos = (tx + text_dx, ty + text_dy)

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
                dist = self.text_offset[0] + self.reverse_arrow_length
                if not keep_centered:
                    if orientation in [
                        Anchor.EAST,
                        Anchor.NORTHEAST,
                        Anchor.NORTH,
                    ]:
                        if self.text_side == Anchor.BOTTOM:
                            tx, ty = extended_line(dist, [pa2, pa1])[1]
                        else:
                            tx, ty = extended_line(dist, [pa1, pa2])[1]
                            self.text_pos = (tx + text_dx, ty + text_dy)
                    else:
                        tx, ty = extended_line(dist, [pa1, pa2])[1]
                        self.text_pos = (tx + text_dx, ty + text_dy)
            else:
                self.dim_line = Arrow(pa1, pa2, head_pos=HeadPos.BOTH)
                self.append(self.dim_line)
            if self.ext1 is not None:
                self.append(self.ext1)

            if self.ext2 is not None:
                self.append(self.ext2)

            if self.ext3 is not None:
                self.append(self.ext3)


def vert_label_layout(shape, offset):
    """Return label anchor, outward direction, and vertex for each vertex."""
    from simetri.geom.polygons.polygon import in_polygon

    vertices = list(shape.vertices)

    vec1 = v_from_points(vertices[0], vertices[-1])
    count = len(vertices)

    layout = []
    for i, vert in enumerate(vertices):
        prev = vertices[i - 1][:2]
        next = vertices[(i + 1) % count][:2]
        point = vert

        vec2 = v_from_points(point, next)
        vert_vec = Vector(point)

        bisector = vec1.bisector(vec2)
        if bisector.norm() < 1e-9:
            direction = Vector(perp_unit_vector((prev, next)))
        else:
            direction = bisector.normalize()

        test_point = vert_vec + direction
        if in_polygon(test_point, vertices):
            pos = vert_vec - direction * offset
        else:
            pos = vert_vec + direction * offset

        to_label = pos - vert_vec
        if to_label.norm() > 1e-9:
            placement_dir = to_label.normalize()
        else:
            placement_dir = direction

        layout.append(
            {
                "position": (pos.x, pos.y),
                "direction": (placement_dir.x, placement_dir.y),
                "vertex": tuple(vert[:2]),
            }
        )
        vec1 = -vec2

    return layout


# Rule-of-thumb tiers replaced by Tag.text_bounds (Pillow glyph bbox).


def _label_size_from_tag_text_bounds(
    text: str, font_size_pt: float
) -> tuple[float, float]:
    """Return ``(width, height)`` from ``Tag.text_bounds`` (no frame padding).

    Uses a centered Tag with ``inner_sep=0`` so the size matches the Pillow
    ink box used by successful overlap resolution experiments.
    """
    tag = Tag(
        str(text),
        pos=(0.0, 0.0),
        font_size=font_size_pt,
        align=Align.CENTER,
    )
    tag.frame.inner_sep = 0
    xmin, ymin, xmax, ymax = tag.text_bounds()
    return xmax - xmin, ymax - ymin


def estimate_index_label_bbox(
    label, font_size_pt: float
) -> tuple[float, float]:
    """Width/height for an index label from ``Tag.text_bounds``.

    Args:
        label: Index label value (converted with ``str``).
        font_size_pt (float): Font size in points.

    Returns:
        tuple[float, float]: ``(width, height)`` of the label box.
    """
    return _label_size_from_tag_text_bounds(str(label), font_size_pt)


def estimate_vertex_coord_label_bbox(
    text: str, font_size_pt: float
) -> tuple[float, float]:
    """Width/height for a vertex coordinate label from ``Tag.text_bounds``.

    Args:
        text (str): Coordinate label text.
        font_size_pt (float): Font size in points.

    Returns:
        tuple[float, float]: ``(width, height)`` of the label box.
    """
    return _label_size_from_tag_text_bounds(text, font_size_pt)


def _centered_label_bbox(
    center: tuple[float, float], size: tuple[float, float]
) -> tuple[float, float, float, float]:
    cx, cy = center
    w, h = size
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _label_bboxes_overlap(
    center_a: tuple[float, float],
    size_a: tuple[float, float],
    center_b: tuple[float, float],
    size_b: tuple[float, float],
    shrink: float = 1.0,
) -> bool:
    if shrink != 1.0:
        size_a = (size_a[0] * shrink, size_a[1] * shrink)
        size_b = (size_b[0] * shrink, size_b[1] * shrink)
    a = _centered_label_bbox(center_a, size_a)
    b = _centered_label_bbox(center_b, size_b)
    return bbox_overlap(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3])


def _label_axis_overlaps(
    center_a: tuple[float, float],
    size_a: tuple[float, float],
    center_b: tuple[float, float],
    size_b: tuple[float, float],
) -> tuple[float, float]:
    """Return horizontal and vertical overlap amounts (0 if disjoint)."""
    ax, ay = center_a
    bx, by = center_b
    aw, ah = size_a
    bw, bh = size_b
    overlap_h = min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2)
    overlap_v = min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2)
    return max(0.0, overlap_h), max(0.0, overlap_v)


def format_vertex_coord(x, y, ndigits=None) -> str:
    """Return ``(x, y)`` formatted for vertex-coordinate labels."""
    if ndigits is None:
        ndigits = defaults["n_vert_digits"]
    return f"({round(float(x), ndigits)}, {round(float(y), ndigits)})"


_RESOLVED_LABELS_KEY = "_simetri_resolved_vertex_labels"
_LABEL_RECTS_KEY = "_simetri_label_rects"
_LABEL_META_KEY = "_simetri_label_meta"


def _vertices_on_hull_points(
    vertices: Sequence,
    hull_pts: Sequence | None = None,
) -> list[int]:
    """Return vertex indices that lie on the given or computed convex hull."""
    from ..geom.polygons.convex_hull import convex_hull

    verts = [tuple(v[:2]) for v in vertices]
    if len(verts) <= 1:
        return list(range(len(verts)))

    if hull_pts is None:
        hull_pts = convex_hull(verts, on_edge=True)
    else:
        hull_pts = [tuple(p[:2]) for p in hull_pts]

    ndigits = int(defaults["n_vert_digits"])
    tol = 1e-6
    indices: list[int] = []
    for i, (vx, vy) in enumerate(verts):
        for hx, hy in hull_pts:
            if hypot(vx - hx, vy - hy) <= tol:
                indices.append(i)
                break
            if ndigits >= 0:
                if round(vx, ndigits) == round(hx, ndigits) and round(
                    vy, ndigits
                ) == round(hy, ndigits):
                    indices.append(i)
                    break
    return indices


def _hull_vertex_indices(vertices: Sequence) -> list[int]:
    """Return vertex indices on this shape's own convex hull."""
    return _vertices_on_hull_points(vertices)


def _iter_label_sketches(sketches):
    """Yield shape sketches that show vertex or index labels."""
    for sketch in sketches:
        subtype = getattr(sketch, "subtype", None)
        if subtype in (Types.CLIPPED_SKETCH, Types.MASKED_SKETCH):
            for sketch_list in sketch.sketches:
                yield from _iter_label_sketches(sketch_list)
        elif subtype == Types.COMPOSITE_SKETCH:
            yield from _iter_label_sketches(sketch.sketches)
        elif getattr(sketch, "indices", False) or getattr(
            sketch, "show_vertex_coords", False
        ):
            yield sketch


def _coord_label_vertex_indices(sketch, n: int) -> list[int]:
    """Vertex indices that receive coordinate (not index) labels."""
    if getattr(sketch, "vertex_on_hull", False):
        group_hull = getattr(sketch, "_group_hull_points", None)
        return _vertices_on_hull_points(sketch.vertices, group_hull)
    return list(range(n))


def _index_label_vertex_indices(sketch, n: int) -> list[int]:
    """Vertex indices that receive index labels (never hull-filtered)."""
    if isinstance(getattr(sketch, "indices", False), bool):
        return list(range(n))
    return list(sketch.indices)


def _build_shape_label_rects(sketch) -> list[LabelRect]:
    """Build centered label boxes at layout anchors (no overlap pass)."""
    existing = getattr(sketch, _LABEL_RECTS_KEY, None)
    if existing is not None:
        return existing

    has_index = bool(getattr(sketch, "indices", False))
    has_vertex = bool(getattr(sketch, "show_vertex_coords", False))
    vertices = sketch.vertices
    n = len(vertices)
    index_label_indices = (
        _index_label_vertex_indices(sketch, n) if has_index else []
    )
    coord_label_indices = (
        _coord_label_vertex_indices(sketch, n) if has_vertex else []
    )

    entries: list[
        tuple[str, int, tuple[float, float], tuple[float, float]]
    ] = []
    index_labels = None
    coord_texts = None

    if has_index:
        index_offset = sketch_label_offset(sketch, "index")
        index_layout = vert_label_layout(sketch, index_offset)
        if isinstance(sketch.indices, bool):
            index_labels = list(range(n))
        else:
            index_labels = list(sketch.indices)
        index_font = sketch_label_font_size_pt(sketch, "index")
        for i in index_label_indices:
            pos = index_layout[i]["position"]
            size = estimate_index_label_bbox(index_labels[i], index_font)
            entries.append(("index", i, pos, size))

    if has_vertex:
        vertex_offset = sketch_label_offset(sketch, "vertex")
        vertex_layout = vert_label_layout(sketch, vertex_offset)
        coord_texts = [format_vertex_coord(*vertices[i]) for i in range(n)]
        vertex_font = sketch_label_font_size_pt(sketch, "vertex")
        for i in coord_label_indices:
            pos = vertex_layout[i]["position"]
            size = estimate_vertex_coord_label_bbox(coord_texts[i], vertex_font)
            entries.append(("vertex", i, pos, size))

    rects = [
        LabelRect(sketch, kind, i, pos[0], pos[1], size[0], size[1])
        for kind, i, pos, size in entries
    ]
    setattr(sketch, _LABEL_RECTS_KEY, rects)
    setattr(
        sketch,
        _LABEL_META_KEY,
        {
            "n": n,
            "index_labels": index_labels,
            "coord_texts": coord_texts,
            "has_index": has_index,
            "has_vertex": has_vertex,
            "index_label_indices": index_label_indices,
            "coord_label_indices": coord_label_indices,
            "entries": entries,
        },
    )
    return rects


def _apply_label_rects_to_sketch(sketch) -> None:
    """Write label rect centers into the sketch resolved-label cache."""
    meta = getattr(sketch, _LABEL_META_KEY, None)
    rects = getattr(sketch, _LABEL_RECTS_KEY, None)
    if meta is None or rects is None:
        return

    n = meta["n"]
    has_index = meta["has_index"]
    has_vertex = meta["has_vertex"]
    index_labels = meta["index_labels"]
    coord_texts = meta["coord_texts"]
    index_label_indices = meta.get("index_label_indices", list(range(n)))
    coord_label_indices = meta.get("coord_label_indices", list(range(n)))

    index_positions = [None] * n if has_index else None
    coord_positions = [None] * n if has_vertex else None
    for rect in rects:
        pos = (rect.x, rect.y)
        if rect.kind == "index":
            index_positions[rect.vertex_index] = pos
        else:
            coord_positions[rect.vertex_index] = pos

    result: dict = {"index": None, "vertex": None}
    if has_index:
        result["index"] = (
            [index_positions[i] for i in index_label_indices],
            [index_labels[i] for i in index_label_indices],
        )
    if has_vertex:
        result["vertex"] = (
            [coord_positions[i] for i in coord_label_indices],
            [coord_texts[i] for i in coord_label_indices],
        )
    setattr(sketch, _RESOLVED_LABELS_KEY, result)


def resolve_page_vertex_labels(sketches) -> None:
    """Resolve overlaps for all vertex/index labels on a sketch list.

    Args:
        sketches: Sketches belonging to one page (or comparable group).

    Returns:
        None
    """
    label_sketches = list(_iter_label_sketches(sketches))
    all_rects: list[LabelRect] = []
    for sketch in label_sketches:
        all_rects.extend(_build_shape_label_rects(sketch))

    if defaults["vertices_label_avoid_overlap"] and len(all_rects) > 1:
        gap = float(defaults["vertices_label_overlap_gap"])
        max_iters = int(defaults["vertices_label_overlap_max_iters"])
        debug = any(bool(getattr(s, "debug", False)) for s in label_sketches)
        if debug:
            print(
                f"resolve_page_vertex_labels: {len(all_rects)} labels, "
                f"gap={gap}, max_iters={max_iters}"
            )
        resolve_all_overlaps(all_rects, gap=gap, max_iters=max_iters)

    for sketch in label_sketches:
        _apply_label_rects_to_sketch(sketch)


def _resolve_shape_labels(sketch) -> dict:
    """Return cached label layout for a shape sketch."""
    cached = getattr(sketch, _RESOLVED_LABELS_KEY, None)
    if cached is not None:
        return cached

    resolve_page_vertex_labels([sketch])
    return getattr(sketch, _RESOLVED_LABELS_KEY)


def prepare_shape_index_labels(
    sketch,
) -> tuple[list[tuple[float, float]], list] | None:
    """Return index label positions and values for a shape sketch.

    Uses ``index_offset`` from the sketch or defaults. When coordinate labels
    are also shown, overlap resolution considers both label types together.

    Args:
        sketch: Shape sketch that may request index labels.

    Returns:
        tuple | None: ``(positions, labels)`` or ``None`` if indices are off.
    """
    if not getattr(sketch, "indices", False):
        return None
    return _resolve_shape_labels(sketch)["index"]


def prepare_shape_vertex_coord_labels(
    sketch,
) -> tuple[list[tuple[float, float]], list[str]] | None:
    """Return vertex coordinate label positions and texts for a shape sketch.

    Uses ``vertex_offset`` from the sketch or defaults. When index labels are
    also shown, overlap resolution considers both label types together.

    Args:
        sketch: Shape sketch that may request coordinate labels.

    Returns:
        tuple | None: ``(positions, texts)`` or ``None`` if coords are off.
    """
    if not getattr(sketch, "show_vertex_coords", False):
        return None
    return _resolve_shape_labels(sketch)["vertex"]


def edge_label_positions(shape, offset):
    """Return edge-label positions using the given radial offset.

    Args:
        shape: Shape whose edges are labeled.
        offset: Distance from each edge midpoint to the label.

    Returns:
        list: Label positions for each edge.
    """
    from simetri.geom.polygons.polygon import in_polygon

    vertices = list(shape.vertices)
    count = len(vertices)
    num_edges = count if shape.closed else count - 1

    # Initialize with edge vector for edge 0
    edge_vec = v_from_points(vertices[0][:2], vertices[1][:2])
    positions = []
    for i in range(num_edges):
        prev_point = vertices[i][:2]
        next_point = vertices[(i + 1) % count][:2]
        point = midpoint(prev_point, next_point)

        mid_vec = Vector(point)
        direction = edge_vec.perp().normalize()

        test_point = mid_vec + direction
        if in_polygon(test_point, vertices):
            pos = mid_vec - direction * offset
        else:
            pos = mid_vec + direction * offset

        positions.append(pos[:])

        # Compute edge vector for next iteration (if there is one)
        if i < num_edges - 1:
            edge_vec = v_from_points(next_point, vertices[(i + 2) % count][:2])

    return positions


def edge_label_pos(shape, index, offset=10):
    """Returns the position of the edge label using the given
    edge index and label offset."""
    from simetri.geom.polygons.polygon import in_polygon

    vertices = shape.vertices
    count = len(vertices)
    prev_point = vertices[index][:2]
    next_point = vertices[(index + 1) % count][:2]
    point = midpoint(prev_point, next_point)

    vec1 = v_from_points(point, next_point)
    edge_vec = Vector(point)

    direction = vec1.perp().normalize()

    test_point = edge_vec + direction
    if in_polygon(test_point, shape.vertices):
        pos = edge_vec - direction * offset
    else:
        pos = edge_vec + direction * offset

    return (pos.x, pos.y)

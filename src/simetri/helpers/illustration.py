"""Illustration helpers for annotations, tags, arrows, and dimensions.

Examples:
    >>> import simetri.graphics as sg
    >>> tag = sg.Tag("Hello", (0, 0))
"""

from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin

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
    HeadPos,
    LineJoin,
    Placement,
    TransformationType,
    Types,
    WarningType,
)
from ..base.common import (
    PointType,
    VecType,
    _set_Nones,
    get_defaults,
)

# from reportlab.pdfbase import pdfmetrics # to do: remove this
from ..base.core import Base, _next_xform_matrix, _Targets
from ..coloring import colors
from ..coloring.swatches import swatches_255
from ..config.settings import defaults, issue_warning
from ..geom.bbox import bounding_box
from ..geom.geom_utils import midpoint
from ..geom.geometry import (
    bbox_overlap,
    polar_to_cartesian,
)
from ..geom.matrices import identity_matrix
from ..geom.nonlinear.ellipse import Arc
from ..geom.points.point_utils import distance
from ..geom.segments.line_utils import (
    extended_line,
    line_angle,
    line_by_point_angle_length,
)
from ..geom.vectors import Vector, perp_unit_vector, v_from_points
from ..group.batch import Group
from ..render.style_map import shape_style_map, tag_style_map
from ..shapes.geom_items import Line, reg_poly_points_side_length
from ..shapes.points import Points
from ..shapes.shape import Shape
from .label_overlap import LabelRect, resolve_all_overlaps
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


def sketch_label_font_family(sketch, label_kind: str):
    """Label font family from sketch kwargs or defaults.

    Args:
        sketch: Sketch providing optional font-family attributes.
        label_kind (str): ``index`` or ``vertex``.

    Returns:
        str | FontFamily: TeX switch name, CSS-ish name, or FontFamily.
    """
    if label_kind == "index":
        attr = "index_font_family"
    else:
        attr = "vertex_font_family"
    if attr in sketch.__dict__ and sketch.__dict__[attr] is not None:
        return sketch.__dict__[attr]
    return defaults[attr]


def label_font_family_tikz(family) -> str:
    """Map a label font-family value to a TeX font switch (no backslash).

    Args:
        family: ``FontFamily``, or a string such as ``ttfamily`` / ``monospace``.

    Returns:
        str: One of ``ttfamily``, ``rmfamily``, ``sffamily``.
    """
    if isinstance(family, FontFamily):
        if family == FontFamily.MONOSPACE:
            return "ttfamily"
        if family == FontFamily.SANSSERIF:
            return "sffamily"
        return "rmfamily"

    normalized = str(family).strip().lower().replace("-", "").replace("_", "")
    if normalized in ("ttfamily", "texttt", "monospace", "mono"):
        return "ttfamily"
    if normalized in ("sffamily", "textsf", "sansserif", "sans"):
        return "sffamily"
    if normalized in ("rmfamily", "textrm", "serif", "rm"):
        return "rmfamily"
    raise ValueError(
        f"Unsupported label font family for TikZ: {family!r}. "
        "Use FontFamily or ttfamily/rmfamily/sffamily."
    )


def label_font_family_svg(family) -> str:
    """Map a label font-family value to a CSS ``font-family`` keyword.

    Args:
        family: ``FontFamily``, or a string such as ``ttfamily`` / ``monospace``.

    Returns:
        str: ``monospace``, ``serif``, or ``sans-serif``.
    """
    if isinstance(family, FontFamily):
        if family == FontFamily.MONOSPACE:
            return "monospace"
        if family == FontFamily.SANSSERIF:
            return "sans-serif"
        return "serif"

    normalized = str(family).strip().lower().replace("-", "").replace("_", "")
    if normalized in ("ttfamily", "texttt", "monospace", "mono"):
        return "monospace"
    if normalized in ("sffamily", "textsf", "sansserif", "sans"):
        return "sans-serif"
    if normalized in ("rmfamily", "textrm", "serif", "rm"):
        return "serif"
    raise ValueError(
        f"Unsupported label font family for SVG: {family!r}. "
        "Use FontFamily or ttfamily/rmfamily/sffamily."
    )


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
    except (fitz.FileDataError, RuntimeError, ValueError) as e:
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


# annotation is a label with a broken leader and an arrow
class AnnotationArrow(Group):
    """A leader from a feature point to text or a circled number.

    The leader is two segments: an angled shaft from ``tip`` to ``elbow``,
    then a landing from ``elbow`` to ``landing``. The arrowhead sits at
    ``tip``. If only ``landing`` or only ``elbow`` is given, the missing
    point is inferred so the landing is horizontal.

    Args:
        tip (PointType): Feature point the arrowhead points at.
        text (str | int): Label text, or a number for a balloon. Defaults
            to "".
        landing (PointType, optional): End of the landing, at the label.
            Defaults to None.
        elbow (PointType, optional): Break between the angled shaft and
            the landing. Defaults to None.
        landing_length (float, optional): Horizontal landing length used
            when inferring ``elbow`` or ``landing``. Defaults to None
            (``defaults["landing_length"]``).
        circled (bool, optional): If True, the label is a balloon
            (circled number or text). Defaults to False.
        font_size (float, optional): Label font size. Defaults to None
            (``defaults["font_size"]``).
        **kwargs: Passed to the leader ``Arrow`` and landing ``Shape``.

    Examples:
        >>> import simetri.graphics as sg
        >>> note = sg.AnnotationArrow((0, 0), "A", landing=(40, 20))
        >>> note.tip
        (0, 0)
        >>> note.elbow
        (20, 20)
        >>> note.landing
        (40, 20)
        >>> balloon = sg.AnnotationArrow(
        ...     (0, 0), 1, landing=(40, 20), circled=True
        ... )
        >>> balloon.text
        '1'
        >>> balloon.tag.frame_shape == sg.FrameShape.CIRCLE
        True
        >>> sg.AnnotationArrow((0, 0), "A")
        Traceback (most recent call last):
            ...
        ValueError: AnnotationArrow requires landing or elbow.
    """

    def __init__(
        self,
        tip: PointType,
        text: str | int = "",
        landing: PointType | None = None,
        elbow: PointType | None = None,
        landing_length: float | None = None,
        circled: bool = False,
        font_size: float | None = None,
        **kwargs,
    ):
        """Create a broken-leader annotation arrow.

        See the class docstring for argument details.
        """
        if elbow is None and landing is None:
            raise ValueError("AnnotationArrow requires landing or elbow.")
        if elbow is not None and landing is not None:
            if landing_length is not None:
                raise ValueError(
                    "Do not pass landing_length when both elbow "
                    "and landing are given."
                )
        else:
            (landing_length,) = get_defaults(
                ["landing_length"], [landing_length]
            )

        (font_size,) = get_defaults(["font_size"], [font_size])
        tip_x, tip_y = tip[:2]
        self.tip = (tip_x, tip_y)
        self.circled = circled
        self.font_size = font_size
        self.text = str(text)

        if elbow is None:
            landing_x, landing_y = landing[:2]
            if landing_x >= tip_x:
                elbow_x = landing_x - landing_length
            else:
                elbow_x = landing_x + landing_length
            elbow_y = landing_y
            self.elbow = (elbow_x, elbow_y)
            self.landing = (landing_x, landing_y)
        elif landing is None:
            elbow_x, elbow_y = elbow[:2]
            if elbow_x >= tip_x:
                landing_x = elbow_x + landing_length
            else:
                landing_x = elbow_x - landing_length
            landing_y = elbow_y
            self.elbow = (elbow_x, elbow_y)
            self.landing = (landing_x, landing_y)
        else:
            elbow_x, elbow_y = elbow[:2]
            landing_x, landing_y = landing[:2]
            self.elbow = (elbow_x, elbow_y)
            self.landing = (landing_x, landing_y)

        self.arrow = Arrow(self.elbow, self.tip, **kwargs)
        items = [self.arrow]
        dist_tol = defaults["dist_tol"]
        landing_span = distance(self.elbow, self.landing)
        if landing_span > dist_tol:
            self.landing_line = Shape(
                [self.elbow, self.landing], fill=False, **kwargs
            )
            items.append(self.landing_line)
        else:
            self.landing_line = None

        land_dx = landing_x - elbow_x
        land_dy = landing_y - elbow_y
        land_len = hypot(land_dx, land_dy)
        if land_len > dist_tol:
            unit_x = land_dx / land_len
            unit_y = land_dy / land_len
        elif landing_x >= tip_x:
            unit_x, unit_y = 1.0, 0.0
        else:
            unit_x, unit_y = -1.0, 0.0

        text_gap = defaults["text_offset"]
        if circled:
            tag_x, tag_y = landing_x, landing_y
        else:
            tag_x = landing_x + unit_x * text_gap
            tag_y = landing_y + unit_y * text_gap
        self.tag = Tag(
            self.text,
            (tag_x, tag_y),
            font_size=font_size,
            align=Align.CENTER,
        )
        if circled:
            self.tag.frame_shape = FrameShape.CIRCLE
            self.tag.stroke = True
            self.tag.fill = True
            self.tag.fill_color = colors.white
        items.append(self.tag)
        super().__init__(items, subtype=Types.ANNOTATION)


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
    blend_mode: str | None = None
    gradient: str | None = None
    pattern: str | None = None
    min_width: float | None = None
    min_height: float | None = None
    min_size: float | None = None

    def __post_init__(self):
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
        fill: bool | None = None,
        xform_matrix=None,
        **kwargs,
    ):
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
                inner_sep=defaults["frame_inner_sep"],
            )
        else:
            self.frame = frame

        for name in TAG_STYLE_ATTRS:
            if name in _TAG_FRAME_PROPERTY_NAMES:
                continue
            setattr(self, name, None)

        self.draw_frame = True
        self.alpha = defaults["tag_alpha"]
        self.align = defaults["tag_align"]
        self.blend_mode = defaults["tag_blend_mode"]

        if font_family is not None:
            self.font_family = font_family
        else:
            self.font_family = defaults["font_family"]
        if font_size is not None:
            self.font_size = font_size
        else:
            self.font_size = defaults["font_size"]
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
    def fill(self):
        """Whether the tag frame is filled. Stored on ``self.frame.fill``."""
        return self.frame.fill

    @fill.setter
    def fill(self, value):
        self.frame.fill = value

    @property
    def stroke(self):
        """Whether the tag frame is stroked. Stored on ``self.frame.stroke``."""
        return self.frame.stroke

    @stroke.setter
    def stroke(self, value):
        self.frame.stroke = value

    @property
    def line_width(self):
        """Frame line width. Stored on ``self.frame.line_width``."""
        return self.frame.line_width

    @line_width.setter
    def line_width(self, value):
        self.frame.line_width = value

    @property
    def line_color(self):
        """Frame line color. Stored on ``self.frame.line_color``."""
        return self.frame.line_color

    @line_color.setter
    def line_color(self, value):
        self.frame.line_color = value

    @property
    def line_dash_array(self):
        """Frame dash pattern. Stored on ``self.frame.line_dash_array``."""
        return self.frame.line_dash_array

    @line_dash_array.setter
    def line_dash_array(self, value):
        self.frame.line_dash_array = value

    @property
    def line_join(self):
        """Frame line join. Stored on ``self.frame.line_join``."""
        return self.frame.line_join

    @line_join.setter
    def line_join(self, value):
        self.frame.line_join = value

    @property
    def back_color(self):
        """Frame fill color. Stored on ``self.frame.back_color``."""
        return self.frame.back_color

    @back_color.setter
    def back_color(self, value):
        self.frame.back_color = value

    @property
    def fill_color(self):
        """Alias of ``back_color`` / ``self.frame.back_color``."""
        return self.frame.back_color

    @fill_color.setter
    def fill_color(self, value):
        self.frame.back_color = value

    @property
    def draw_double(self):
        """Whether the frame uses a double line. Stored on ``self.frame``."""
        return self.frame.draw_double

    @draw_double.setter
    def draw_double(self, value):
        self.frame.draw_double = value

    @property
    def double_distance(self):
        """Distance between double frame lines. Stored on ``self.frame``."""
        return self.frame.double_distance

    @double_distance.setter
    def double_distance(self, value):
        self.frame.double_distance = value

    @property
    def double_color(self):
        """Color of double frame lines. Stored on ``self.frame.double``."""
        return self.frame.double

    @double_color.setter
    def double_color(self, value):
        self.frame.double = value

    @property
    def draw_fillets(self):
        """Whether the frame draws fillets. Stored on ``self.frame``."""
        return self.frame.draw_fillets

    @draw_fillets.setter
    def draw_fillets(self, value):
        self.frame.draw_fillets = value

    @property
    def fillet_radius(self):
        """Frame fillet radius. Stored on ``self.frame.fillet_radius``."""
        return self.frame.fillet_radius

    @fillet_radius.setter
    def fillet_radius(self, value):
        self.frame.fillet_radius = value

    @property
    def smooth(self):
        """Whether the frame is smoothed. Stored on ``self.frame.smooth``."""
        return self.frame.smooth

    @smooth.setter
    def smooth(self, value):
        self.frame.smooth = value

    @property
    def frame_shape(self):
        """Frame shape. Stored on ``self.frame.frame_shape``."""
        return self.frame.frame_shape

    @frame_shape.setter
    def frame_shape(self, value):
        self.frame.frame_shape = value

    @property
    def frame_inner_sep(self):
        """Frame inner separation. Stored on ``self.frame.inner_sep``."""
        return self.frame.inner_sep

    @frame_inner_sep.setter
    def frame_inner_sep(self, value):
        self.frame.inner_sep = value

    @property
    def frame_outer_sep(self):
        """Frame outer separation. Stored on ``self.frame.outer_sep``."""
        return self.frame.outer_sep

    @frame_outer_sep.setter
    def frame_outer_sep(self, value):
        self.frame.outer_sep = value

    @property
    def frame_min_width(self):
        """Frame minimum width. Stored on ``self.frame.min_width``."""
        return self.frame.min_width

    @frame_min_width.setter
    def frame_min_width(self, value):
        self.frame.min_width = value

    @property
    def frame_min_height(self):
        """Frame minimum height. Stored on ``self.frame.min_height``."""
        return self.frame.min_height

    @frame_min_height.setter
    def frame_min_height(self, value):
        self.frame.min_height = value

    @property
    def frame_min_size(self):
        """Frame minimum size. Stored on ``self.frame.min_size``."""
        return self.frame.min_size

    @frame_min_size.setter
    def frame_min_size(self, value):
        self.frame.min_size = value

    def _update(
        self,
        xform_matrix,
        reps: int = 0,
        take: slice | None = None,
        incr=None,
        dyn_ref: Callable | None = None,
        merge: bool = False,
        xform_type: TransformationType = None,
    ):
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
        """
        return (self._init_pos @ self.xform_matrix)[:2].tolist()

    def copy(self, **kwargs) -> "Tag":
        """Returns a copy of the Tag object.

        Returns:
            Tag: A copy of the Tag object.
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
            text_width = self.text_width
        if self.minimum_width is not None:
            text_width = max(text_width, self.minimum_width)

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
            self.head.rotate(pi)
            self.head.translate(self.head.head_length, 0)
            self.head.rotate(angle)
            self.line.rotate(angle)
            self.line.translate(*p1)
            self.head.translate(*p1)
            self.heads = [self.head]
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


def vec_arrow(
    vec: VecType,
    *,
    start: PointType | None = None,
    end: PointType | None = None,
    **kwargs,
) -> Arrow:
    """Return an ``Arrow`` from a displacement and a start or end point.

    ``vec`` is a displacement. It is not stored on the ``Arrow``. Extra
    keyword arguments are passed to ``Arrow``.

    Args:
        vec (VecType): Displacement from tail to tip.
        start (PointType, optional): Tail position. Defaults to None.
        end (PointType, optional): Tip position. Defaults to None.
        **kwargs: Passed to ``Arrow``.

    Returns:
        Arrow: Arrow from ``start`` to ``start + vec``, or from
        ``end - vec`` to ``end``.

    Raises:
        ValueError: If ``vec`` is zero, if neither ``start`` nor ``end``
            is given, or if both are given and ``end - start`` does not
            match ``vec``.

    Examples:
        >>> import simetri.graphics as sg
        >>> arrow = sg.vec_arrow(sg.Vector(3, 4), start=(10, 20))
        >>> arrow.p1
        (10, 20)
        >>> arrow.p2
        (13, 24)
        >>> arrow = sg.vec_arrow(sg.Vector(3, 4), end=(13, 24))
        >>> arrow.p1
        (10, 20)
        >>> arrow.p2
        (13, 24)
        >>> sg.vec_arrow(sg.Vector(0, 0), start=(1, 1))
        Traceback (most recent call last):
            ...
        ValueError: Cannot create an Arrow from a zero-length Vector.
        >>> sg.vec_arrow(sg.Vector(3, 4), start=(0, 0), end=(1, 0))
        Traceback (most recent call last):
            ...
        ValueError: start and end are not consistent with the Vector displacement (3, 4).
    """
    vector_x, vector_y = vec[:2]
    if vector_x == 0 and vector_y == 0:
        raise ValueError("Cannot create an Arrow from a zero-length Vector.")
    if start is None and end is None:
        raise ValueError("vec_arrow requires start or end.")
    if start is not None and end is not None:
        start_x, start_y = start[:2]
        end_x, end_y = end[:2]
        displacement_x = end_x - start_x
        displacement_y = end_y - start_y
        offset = hypot(displacement_x - vector_x, displacement_y - vector_y)
        if offset <= defaults["dist_tol"]:
            issue_warning(
                f"Duplicate position used for Vector({start}, {end}).",
                warning_type=WarningType.vector.duplicate,
            )
            return Arrow((start_x, start_y), (end_x, end_y), **kwargs)
        raise ValueError(
            "start and end are not consistent with the Vector "
            f"displacement {(vector_x, vector_y)}."
        )
    if start is not None:
        start_x, start_y = start[:2]
        return Arrow(
            (start_x, start_y),
            (start_x + vector_x, start_y + vector_y),
            **kwargs,
        )
    end_x, end_y = end[:2]
    return Arrow(
        (end_x - vector_x, end_y - vector_y),
        (end_x, end_y),
        **kwargs,
    )


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
    """A linear dimension: extension lines, dimension line, and a label.

    ``p1`` and ``p2`` decide the kind: same ``y`` is horizontal, same
    ``x`` is vertical, otherwise the dimension is aligned to the
    segment (diagonal). The label position is stored on ``text_pos``.

    Args:
        p1 (PointType): First feature point.
        p2 (PointType): Second feature point.
        side (str): Which side of the features the dimension line is
            on. Horizontal or diagonal: ``"up"`` or ``"down"``.
            Vertical: ``"left"`` or ``"right"``.
        text_offset (float): Distance from the gap to the dimension line.
        text (str, optional): Label text. ``None`` uses the measured
            length. Defaults to None.
        ext_line_extension (float, optional): How far each extension
            continues past the dimension line. ``None`` uses
            ``defaults["overshoot"]``.
        ext_line_offset (float, optional): Gap from the feature to the
            start of the extension. ``None`` uses ``defaults["gap"]``.
        text_horiz_offset (float, optional): Offset of the label along
            the dimension line when ``text_loc`` is ``"left"`` or
            ``"right"``. ``None`` uses ``defaults["ext_length2"]``.
        text_loc (str, optional): ``"middle"``, ``"left"``, or
            ``"right"``. Defaults to ``"middle"``.
        stub_length (float, optional): Outward shaft length when the
            label is not in the middle. Defaults to 15.
        **kwargs: Additional keyword arguments for dimension styling.
    """

    def __init__(
        self,
        p1: PointType,
        p2: PointType,
        side: str,
        text_offset: float,
        text: str | None = None,
        ext_line_extension: float | None = None,
        ext_line_offset: float | None = None,
        text_horiz_offset: float | None = None,
        text_loc: str = "middle",
        stub_length: float = 15,
        **kwargs,
    ):
        """Create a linear dimension with extension lines and arrows.

        See the class docstring for argument details.
        """
        (
            ext_line_extension,
            ext_line_offset,
            text_horiz_offset,
            font_size,
        ) = get_defaults(
            [
                "overshoot",
                "gap",
                "ext_length2",
                "font_size",
            ],
            [
                ext_line_extension,
                ext_line_offset,
                text_horiz_offset,
                None,
            ],
        )
        if text is None:
            text = str(distance(p1, p2))

        self.p1 = p1
        self.p2 = p2
        self.side = side
        self.text_offset = text_offset
        self.text = text
        self.ext_line_extension = ext_line_extension
        self.ext_line_offset = ext_line_offset
        self.text_horiz_offset = text_horiz_offset
        self.text_loc = text_loc
        self.stub_length = stub_length
        self.font_size = font_size
        self.kwargs = kwargs
        self.ext1 = None
        self.ext2 = None
        self.ext3 = None
        self.arrow1 = None
        self.arrow2 = None
        self.dim_line = None
        self.mid_line = None
        self.tag = None
        self.text_anchor = Anchor.CENTER
        self.text_align = Align.CENTER

        super().__init__(subtype=Types.DIMENSION, **kwargs)

        x1, y1 = p1[:2]
        x2, y2 = p2[:2]
        dist_tol = defaults["dist_tol"]
        if abs(x1 - x2) < dist_tol and abs(y1 - y2) < dist_tol:
            raise ValueError("Dimension points must be distinct.")

        if abs(y1 - y2) < dist_tol:
            dim_x1 = x1
            dim_x2 = x2
            if text_loc == "middle":
                text_x = (x1 + x2) / 2
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.CENTER
            elif text_loc == "left":
                text_x = x1 - text_horiz_offset
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.RIGHT
            else:
                text_x = x2 + text_horiz_offset
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.LEFT
            if side == "up":
                text_y = y1 + ext_line_offset + text_offset
                y_p1_start = y1 + ext_line_offset
                y_p1_end = text_y + ext_line_extension
                y_p2_start = y2 + ext_line_offset
                y_p2_end = y2 + ext_line_offset + text_offset + ext_line_extension
            else:
                text_y = y1 - ext_line_offset - text_offset
                y_p1_start = y1 - ext_line_offset
                y_p1_end = y1 - ext_line_offset - text_offset - ext_line_extension
                y_p2_start = y2 - ext_line_offset
                y_p2_end = y2 - ext_line_offset - text_offset - ext_line_extension
            ext1_start = (x1, y_p1_start)
            ext1_end = (x1, y_p1_end)
            ext2_start = (x2, y_p2_start)
            ext2_end = (x2, y_p2_end)
            dim1 = (dim_x1, text_y)
            dim2 = (dim_x2, text_y)
            stub1_tail = (dim_x1 - stub_length, text_y)
            stub2_tip = (dim_x2 + stub_length, text_y)
        elif abs(x1 - x2) < dist_tol:
            dim_y1 = y1
            dim_y2 = y2
            if text_loc == "middle":
                text_y = (y1 + y2) / 2
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.CENTER
            elif text_loc == "left":
                text_y = y1 - text_horiz_offset
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.RIGHT
            else:
                text_y = y2 + text_horiz_offset
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.LEFT
            if side == "right":
                text_x = x1 + ext_line_offset + text_offset
                x_p1_start = x1 + ext_line_offset
                x_p1_end = text_x + ext_line_extension
                x_p2_start = x2 + ext_line_offset
                x_p2_end = x2 + ext_line_offset + text_offset + ext_line_extension
            else:
                text_x = x1 - ext_line_offset - text_offset
                x_p1_start = x1 - ext_line_offset
                x_p1_end = x1 - ext_line_offset - text_offset - ext_line_extension
                x_p2_start = x2 - ext_line_offset
                x_p2_end = x2 - ext_line_offset - text_offset - ext_line_extension
            ext1_start = (x_p1_start, y1)
            ext1_end = (x_p1_end, y1)
            ext2_start = (x_p2_start, y2)
            ext2_end = (x_p2_end, y2)
            dim1 = (text_x, dim_y1)
            dim2 = (text_x, dim_y2)
            stub1_tail = (text_x, dim_y1 - stub_length)
            stub2_tip = (text_x, dim_y2 + stub_length)
        else:
            dim_angle = line_angle(p1, p2)
            if side == "up":
                normal_angle = dim_angle + pi / 2
            else:
                normal_angle = dim_angle - pi / 2
            along_x = cos(dim_angle)
            along_y = sin(dim_angle)
            dist_to_line = ext_line_offset + text_offset
            ext_end_length = dist_to_line + ext_line_extension
            ext1_start = line_by_point_angle_length(
                p1, normal_angle, ext_line_offset
            )[1]
            ext1_end = line_by_point_angle_length(
                p1, normal_angle, ext_end_length
            )[1]
            ext2_start = line_by_point_angle_length(
                p2, normal_angle, ext_line_offset
            )[1]
            ext2_end = line_by_point_angle_length(
                p2, normal_angle, ext_end_length
            )[1]
            dim1 = line_by_point_angle_length(p1, normal_angle, dist_to_line)[1]
            dim2 = line_by_point_angle_length(p2, normal_angle, dist_to_line)[1]
            dim1_x, dim1_y = dim1[:2]
            dim2_x, dim2_y = dim2[:2]
            if text_loc == "middle":
                text_x = (dim1_x + dim2_x) / 2
                text_y = (dim1_y + dim2_y) / 2
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.CENTER
            elif text_loc == "left":
                text_x = dim1_x - along_x * text_horiz_offset
                text_y = dim1_y - along_y * text_horiz_offset
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.RIGHT
            else:
                text_x = dim2_x + along_x * text_horiz_offset
                text_y = dim2_y + along_y * text_horiz_offset
                self.text_anchor = Anchor.CENTER
                self.text_align = Align.LEFT
            stub1_tail = (
                dim1_x - along_x * stub_length,
                dim1_y - along_y * stub_length,
            )
            stub2_tip = (
                dim2_x + along_x * stub_length,
                dim2_y + along_y * stub_length,
            )

        self.text_pos = (text_x, text_y)

        self.tag = Tag(
            text,
            pos=(text_x, text_y),
            fill=True,
            anchor=self.text_anchor,
            align=self.text_align,
            font_size=self.font_size,
        )
        self.ext1 = Line(ext1_start, ext1_end)
        self.ext2 = Line(ext2_start, ext2_end)
        self.append(self.ext1)
        self.append(self.ext2)
        if text_loc == "middle":
            self.dim_line = Arrow(dim1, dim2, head_pos=HeadPos.BOTH)
            self.append(self.dim_line)
        else:
            self.arrow1 = Arrow(stub1_tail, dim1, head_pos=HeadPos.END)
            self.arrow2 = Arrow(dim2, stub2_tip, head_pos=HeadPos.START)
            self.mid_line = Line(dim1, dim2)
            self.append(self.arrow1)
            self.append(self.arrow2)
            self.append(self.mid_line)


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
            if (
                ndigits >= 0
                and round(vx, ndigits) == round(hx, ndigits)
                and round(vy, ndigits) == round(hy, ndigits)
            ):
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

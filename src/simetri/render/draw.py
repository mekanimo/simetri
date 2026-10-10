"""Canvas object uses these methods to draw shapes and text."""

from __future__ import annotations

from collections.abc import Sequence
from math import pi, radians, sin
from types import SimpleNamespace
from typing import TYPE_CHECKING, Self

from ..base.all_enums import (
    Align,
    Anchor,
    BackStyle,
    Connection,
    Drawable,
    FillMode,
    FragmentColoring,
    MarkerType,
    PlaitStyle,
    SvgLoc,
    TexLoc,
    Types,
    drawable_types,
    get_enum_value,
)
from ..base.common import PointType, alias_argument
from ..base.common_style import coerce_style_overlay
from ..coloring import colors
from ..coloring.colors import Color, change_lightness
from ..config.settings import runtime_defaults
from ..geom.affine import (
    rotation_matrix,
    translation_matrix,
)
from ..geom.bbox import bounding_box
from ..geom.geom_utils import midpoint
from ..geom.points.point_utils import distance
from ..geom.homogenize import homogenize
from ..geom.matrices import identity_matrix
from ..geom.nonlinear.bezier import bezier_points
from ..geom.nonlinear.ellipse import (
    _elliptic_arc_points_from_signed,
    resolve_arc_sweep,
)
from ..geom.nonlinear.path import (
    Path2D,
    group_to_nonzero_path,
    path2d_to_svg_path,
)
from ..geom.polygons.convex_hull import convex_hull
from ..geom.polygons.polygon import offset_polygon, polygon_area
from ..geom.segments.line_utils import (
    inclination_angle,
    intersect,
    intersection,
)
from ..group.batch import Group
from ..helpers.arrows import _style_annotation
from ..helpers.illustration import Tag, TextPath
from ..helpers.utilities import (
    decompose_transformations,
    group_into_bins,
)
from ..shapes.shape import Shape, all_segments
from .render_svg.svg_sketch import SvgSketch
from .render_tikz.tikz_sketch import TexSketch
from .sketch import (
    ArcSketch,
    BezierSketch,
    CircleSketch,
    ClippedSketch,
    CompositeSketch,
    EllipseSketch,
    HelpLinesSketch,
    ImageSketch,
    LatexSketch,
    LineSketch,
    PathSketch,
    PatternSketch,
    PDFSketch,
    RectSketch,
    ShapeSketch,
    Sketch,
    TagSketch,
    TextPathSketch,
)

DrawStyleKwargs = dict[str, object]
from .style_map import (
    MarkerStyle,
    line_style_map,
    shape_style_map,
    tag_style_map,
)

if TYPE_CHECKING:
    from ..geom.bbox import BoundingBox
    from ..helpers.dimension import Dimension
    from ..images.image import PDF, Image
    from ..interlace.lace import Lace
    from ..patterns.pattern import Pattern
    from ..shapes.shape import Clipping
    from .canvas import Canvas


_PRECEDENCE_KEYS = frozenset(
    (
        "color",
        "line_color",
        "fill_color",
        "alpha",
        "line_alpha",
        "fill_alpha",
    )
)


def help_lines(
    self: Canvas,
    pos: PointType | None = None,
    width: float | None = None,
    height: float | None = None,
    spacing: float | None = None,
    cs_size: float | None = None,
    deferred: bool = False,
    **kwargs: object,
) -> Self:
    """Draw a square grid, and optionally the coordinate axes.

    When ``deferred`` is true, a ``HelpLinesSketch`` is stored and the
    grid is not drawn yet. Otherwise the grid is drawn immediately, and
    the coordinate system is drawn when ``cs_size`` is greater than 0.

    Args:
        pos: Lower-left corner of the grid.
        width: Length of the grid along the x-axis.
        height: Length of the grid along the y-axis.
        spacing: Distance between grid lines.
        cs_size: Length of the coordinate axes. Used when ``deferred``
            is false.
        deferred: If true, store a help-lines sketch instead of drawing.
        **kwargs: Style overrides for the grid and axes.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.help_lines((0, 0), 20, 20, spacing=40, cs_size=0, deferred=False) is canvas
        True
        >>> len(canvas.active_page.sketches) > 0
        True
    """
    if deferred:
        style_source = SimpleNamespace(type=Types.SKETCH)

        grid_kwargs = dict(kwargs)
        if "line_width" not in grid_kwargs:
            grid_kwargs["line_width"] = runtime_defaults["grid_line_width"]
        if "line_color" not in grid_kwargs and "color" not in grid_kwargs:
            grid_kwargs["line_color"] = runtime_defaults["grid_line_color"]
        if "line_dash_array" not in grid_kwargs:
            grid_kwargs["line_dash_array"] = runtime_defaults[
                "grid_line_dash_array"
            ]
        grid_style = self.resolve_style_properties(
            style_source,
            line_style_map,
            **grid_kwargs,
        )

        if "colors" in kwargs:
            x_axis_color, y_axis_color = kwargs["colors"]
        else:
            x_axis_color = runtime_defaults["CS_x_color"]
            y_axis_color = runtime_defaults["CS_y_color"]

        axis_kwargs = dict(kwargs)
        if "line_width" not in axis_kwargs:
            axis_kwargs["line_width"] = runtime_defaults["CS_line_width"]
        x_axis_kwargs = dict(axis_kwargs)
        x_axis_kwargs["line_color"] = x_axis_color
        x_axis_style = self.resolve_style_properties(
            style_source,
            line_style_map,
            **x_axis_kwargs,
        )
        y_axis_kwargs = dict(axis_kwargs)
        y_axis_kwargs["line_color"] = y_axis_color
        y_axis_style = self.resolve_style_properties(
            style_source,
            line_style_map,
            **y_axis_kwargs,
        )

        if "line_color" in kwargs:
            origin_color = kwargs["line_color"]
        elif "color" in kwargs:
            origin_color = kwargs["color"]
        else:
            origin_color = runtime_defaults["CS_origin_color"]
        origin_kwargs = dict(kwargs)
        origin_kwargs["color"] = origin_color
        origin_kwargs["fill"] = True
        origin_kwargs["stroke"] = True
        origin_style = self.resolve_style_properties(
            style_source,
            shape_style_map,
            **origin_kwargs,
        )

        sketch = HelpLinesSketch(
            spacing,
            cs_size,
            grid_style,
            x_axis_style,
            y_axis_style,
            origin_style,
            runtime_defaults["CS_origin_size"],
        )
        self.active_page.sketches.append(sketch)
    else:
        self.grid(pos, width, height, spacing, **kwargs)
        if cs_size > 0:
            self.draw_CS(cs_size, **kwargs)
    return self


@alias_argument({"radius_x": "rx", "radius_y": "ry"})
def arc(
    self: Canvas,
    center: PointType,
    radius_x: float,
    radius_y: float | None = None,
    start_angle: float = 0,
    span_angle: float | None = None,
    rot_angle: float = 0,
    n_points: int | None = None,
    *,
    end_angle: float | None = None,
    clockwise: bool = False,
    **kwargs: object,
) -> Self:
    """Draw an elliptic arc from ``start_angle`` through ``span_angle``
    or ``end_angle``.

    Args:
        center: Center of the arc.
        radius_x: Radius along the local x-axis.
        radius_y: Radius along the local y-axis; defaults to ``radius_x``.
        start_angle: Start angle in radians. Defaults to 0.
        span_angle: Sweep in radians. A negative value draws clockwise.
            Mutually exclusive with ``end_angle``. At least one of
            ``span_angle`` or ``end_angle`` is required.
        rot_angle: Rotation of the arc about ``center``, in radians.
        n_points: Number of samples along the arc.
        end_angle: Ending angle in radians. Mutually exclusive with
            ``span_angle``.
        clockwise: If True, the arc is drawn clockwise. Defaults to False.
        **kwargs: Style overrides for the arc sketch.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.arc((0, 0), 10, 10, 0, sg.pi / 2, 0) is canvas
        True
        >>> canvas.active_page.sketches[-1].subtype.name
        'ARC_SKETCH'
    """
    if radius_y is None:
        radius_y = radius_x
    signed_span = resolve_arc_sweep(
        start_angle,
        span_angle,
        end_angle,
        clockwise,
    )
    vertices = _elliptic_arc_points_from_signed(
        center, radius_x, radius_y, start_angle, signed_span, n_points
    )
    if rot_angle != 0:
        vertices = homogenize(vertices) @ rotation_matrix(rot_angle, center)
    self._all_vertices.extend(vertices.tolist() + [center])

    self._sketch_xform_matrix = self.xform_matrix
    sketch = ArcSketch(
        vertices=vertices, xform_matrix=self._sketch_xform_matrix
    )
    self._sketch_xform_matrix = identity_matrix()
    resolved = self.resolve_style_properties(sketch, shape_style_map, **kwargs)
    for attrib_name, attrib_value in resolved.items():
        setattr(sketch, attrib_name, attrib_value)

    for k, v in kwargs.items():
        if k not in _PRECEDENCE_KEYS:
            setattr(sketch, k, v)
    self.active_page.sketches.append(sketch)

    return self


def bezier(
    self: Canvas, control_points: Sequence[PointType], **kwargs: object
) -> Self:
    """Draw a Bezier curve through the given control points.

    Args:
        control_points: Control points of the curve, in walk order.
        **kwargs: Style overrides for the curve sketch.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.bezier([(0, 0), (20, 40), (40, 0)]) is canvas
        True
        >>> len(canvas.active_page.sketches)
        1
    """
    self._all_vertices.extend(control_points)
    self._sketch_xform_matrix = self.xform_matrix
    sketch = BezierSketch(control_points, self._sketch_xform_matrix)
    self._sketch_xform_matrix = identity_matrix()
    resolved = self.resolve_style_properties(sketch, shape_style_map, **kwargs)
    for attrib_name, attrib_value in resolved.items():
        setattr(sketch, attrib_name, attrib_value)

    for k, v in kwargs.items():
        if k not in _PRECEDENCE_KEYS:
            setattr(sketch, k, v)
    self.active_page.sketches.append(sketch)
    return self


def _circle_sketch(
    center: PointType, radius: float, matrix: object
) -> CircleSketch | EllipseSketch:
    """Return a circle sketch, or an ellipse when the axis scales differ.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.draw import _circle_sketch
        >>> _circle_sketch((0, 0), 10, sg.identity_matrix()).subtype.name
        'CIRCLE_SKETCH'
    """
    _, _, scale = decompose_transformations(matrix)
    scale_x = float(scale[0])
    scale_y = float(scale[1])
    if abs(scale_x - scale_y) > runtime_defaults["abs_tol"]:
        return EllipseSketch(center, radius, radius, 0, matrix)
    return CircleSketch(center, radius, matrix)


def circle(
    self: Canvas, radius: float, center: PointType = (0, 0), **kwargs: object
) -> Self:
    """Draw a circle with the given radius and optional center.

    Args:
        radius: Radius of the circle.
        center: Center of the circle. Defaults to ``(0, 0)``.
        **kwargs: Style overrides for the circle sketch.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.circle(10, (0, 0)) is canvas
        True
        >>> canvas.active_page.sketches[-1].subtype.name
        'CIRCLE_SKETCH'
    """
    x, y = center[:2]
    p1 = x - radius, y - radius
    p2 = x + radius, y + radius
    p3 = x - radius, y + radius
    p4 = x + radius, y - radius
    self._all_vertices.extend([p1, p2, p3, p4])
    self._sketch_xform_matrix = self.xform_matrix
    sketch = _circle_sketch(center, radius, self._sketch_xform_matrix)
    self._sketch_xform_matrix = identity_matrix()
    resolved = self.resolve_style_properties(sketch, shape_style_map, **kwargs)
    for attrib_name, attrib_value in resolved.items():
        setattr(sketch, attrib_name, attrib_value)

    for k, v in kwargs.items():
        if k not in _PRECEDENCE_KEYS:
            setattr(sketch, k, v)
    self.active_page.sketches.append(sketch)

    return self


def ellipse(
    self: Canvas,
    width: float,
    height: float,
    center: PointType = (0, 0),
    angle: float = 0,
    **kwargs: object,
) -> Self:
    """Draw an ellipse with the given width, height, and optional center.

    Args:
        width: Full width. The x-radius is ``width / 2``.
        height: Full height. The y-radius is ``height / 2``.
        center: Center of the ellipse. Defaults to ``(0, 0)``.
        angle: Rotation of the ellipse in radians. Defaults to 0.
        **kwargs: Style overrides for the ellipse sketch.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.ellipse(20, 40) is canvas
        True
        >>> canvas.active_page.sketches[-1].subtype.name
        'ELLIPSE_SKETCH'
    """
    x, y = center[:2]
    x_radius = width / 2
    y_radius = height / 2
    p1 = x - x_radius, y - y_radius
    p2 = x + x_radius, y + y_radius
    p3 = x - x_radius, y + y_radius
    p4 = x + x_radius, y - y_radius
    self._all_vertices.extend([p1, p2, p3, p4])
    self._sketch_xform_matrix = self.xform_matrix
    sketch = EllipseSketch(
        center, x_radius, y_radius, angle, self._sketch_xform_matrix
    )
    self._sketch_xform_matrix = identity_matrix()
    resolved = self.resolve_style_properties(sketch, shape_style_map, **kwargs)
    for attrib_name, attrib_value in resolved.items():
        setattr(sketch, attrib_name, attrib_value)

    for k, v in kwargs.items():
        if k not in _PRECEDENCE_KEYS:
            setattr(sketch, k, v)
    self.active_page.sketches.append(sketch)

    return self


def text(
    self: Canvas,
    txt: str,
    pos: PointType,
    font_family: str | None = None,
    font_size: int | None = None,
    font_color: Color | None = None,
    anchor: Anchor | None = None,
    align: Align | None = None,
    **kwargs: object,
) -> Self:
    """Draw text at the given position.

    Args:
        txt: Text to draw.
        pos: Position of the text.
        font_family: Font family. None uses the default.
        font_size: Font size. None uses the default.
        font_color: Color of the text. None uses the default.
        anchor: Anchor of the text. None uses the default.
        align: Alignment of the text. None uses the default.
        **kwargs: Style overrides forwarded to the text tag.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.text("A", (0, 0)) is canvas
        True
        >>> len(canvas.active_page.sketches)
        1
    """
    # first create a Tag object
    tag_obj = Tag(
        txt,
        pos,
        font_family=font_family,
        font_size=font_size,
        font_color=font_color,
        anchor=anchor,
        align=align,
        **kwargs,
    )
    tag_obj.draw_frame = False
    # extend vertices with the Tag's bounding box
    self._sketch_xform_matrix = self.xform_matrix
    extend_vertices(self, tag_obj)
    sketch = create_sketch(tag_obj, self, **kwargs)
    self._sketch_xform_matrix = identity_matrix()
    self.active_page.sketches.append(sketch)

    return self


def text_path(
    self: Canvas,
    txt: str,
    path: Path2D | Shape,
    font_family: str | None = None,
    font_size: int | None = None,
    font_color: Color | None = None,
    bold: bool = False,
    italic: bool = False,
    draw_path: bool = False,
    **kwargs: object,
) -> Self:
    """Draw text along a path.

    Args:
        txt: Text to place on the path.
        path: A ``Path2D`` or ``Shape``.
        font_family: Font family. None uses the default.
        font_size: Font size. None uses the default.
        font_color: Color of the text. None uses the default.
        bold: Bold type. Defaults to False.
        italic: Italic type. Defaults to False.
        draw_path: If True, also stroke the guide path. Defaults to False.
        **kwargs: Extra attributes stored on the ``TextPath``.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> curve = sg.Path2D((0, 0)).line_to((80, 0))
        >>> canvas.text_path("along", curve) is canvas
        True
        >>> canvas.active_page.sketches[-1].subtype.name
        'TEXT_PATH_SKETCH'
    """
    item = TextPath(
        txt,
        path,
        font_family=font_family,
        font_size=font_size,
        font_color=font_color,
        bold=bold,
        italic=italic,
        draw_path=draw_path,
        **kwargs,
    )
    self._sketch_xform_matrix = self.xform_matrix
    extend_vertices(self, item)
    sketch = create_sketch(item, self)
    self._sketch_xform_matrix = identity_matrix()
    self.active_page.sketches.append(sketch)
    return self


def line(
    self: Canvas, start: PointType, end: PointType, **kwargs: object
) -> Self:
    """Draw a line segment from start to end.

    Args:
        start: Starting point of the line.
        end: Ending point of the line.
        **kwargs: Style overrides for the line.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.line((0, 0), (40, 0)) is canvas
        True
        >>> len(canvas.active_page.sketches)
        1
    """
    self._sketch_xform_matrix = self.xform_matrix
    line_shape = Shape([start, end], closed=False, **kwargs)
    extend_vertices(self, line_shape)
    line_sketch = create_sketch(line_shape, self, **kwargs)
    self.active_page.sketches.append(line_sketch)
    self._sketch_xform_matrix = identity_matrix()
    return self


def rectangle(
    self: Canvas,
    width: float,
    height: float,
    center: PointType = (0, 0),
    angle: float = 0,
    **kwargs: object,
) -> Self:
    """Draw a rectangle with the given width, height, and optional center.

    Args:
        width: Width of the rectangle.
        height: Height of the rectangle.
        center: Center of the rectangle. Defaults to ``(0, 0)``.
        angle: Rotation about ``center``, in radians. Defaults to 0.
        **kwargs: Style overrides for the rectangle.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.rectangle(40, 60) is canvas
        True
        >>> len(canvas.active_page.sketches)
        1
    """
    x, y = center[:2]
    w2 = width / 2
    h2 = height / 2
    p1 = x - w2, y + h2
    p2 = x - w2, y - h2
    p3 = x + w2, y - h2
    p4 = x + w2, y + h2
    points = homogenize([p1, p2, p3, p4]) @ rotation_matrix(angle, center)
    rect_shape = Shape(points.tolist(), closed=True, **kwargs)
    self._sketch_xform_matrix = self.xform_matrix
    extend_vertices(self, rect_shape)
    rect_sketch = create_sketch(rect_shape, self, **kwargs)
    self.active_page.sketches.append(rect_sketch)
    self._sketch_xform_matrix = identity_matrix()

    return self


def draw_CS(self: Canvas, size: float | None = None, **kwargs: object) -> Self:
    """Draw the coordinate axes and an origin marker.

    ``size`` None uses ``runtime_defaults["CS_size"]``. ``kwargs["colors"]`` is
    ``(x_color, y_color)`` for the two axes.

    Args:
        size: Length of each axis.
        **kwargs: Style overrides. ``colors`` selects the axis colors.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.draw_CS(10) is canvas
        True
        >>> len(canvas.active_page.sketches) >= 2
        True
    """
    if size is None:
        size = runtime_defaults["CS_size"]
    if "colors" in kwargs:
        x_color, y_color = kwargs["colors"]
        del kwargs["colors"]
    else:
        x_color = runtime_defaults["CS_x_color"]
        y_color = runtime_defaults["CS_y_color"]
    if "line_width" not in kwargs:
        kwargs["line_width"] = runtime_defaults["CS_line_width"]
    self.line((0, 0), (size, 0), line_color=x_color, **kwargs)
    self.line((0, 0), (0, size), line_color=y_color, **kwargs)
    if "line_color" not in kwargs:
        kwargs["line_color"] = runtime_defaults["CS_origin_color"]
    self.circle(radius=runtime_defaults["CS_origin_size"], **kwargs)

    return self


def lines(self: Canvas, points: Sequence[PointType], **kwargs: object) -> Self:
    """Draw connected line segments through the given points.

    Args:
        points: Points in walk order.
        **kwargs: Style overrides for the line sketch.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.lines([(0, 0), (40, 0), (40, 20)]) is canvas
        True
        >>> len(canvas.active_page.sketches)
        1
    """
    self._all_vertices.extend(points)
    self._sketch_xform_matrix = self.xform_matrix
    sketch = LineSketch(points, self._sketch_xform_matrix, **kwargs)
    self._sketch_xform_matrix = identity_matrix()
    for attrib_name in line_style_map:
        attrib_value = self.resolve_property(sketch, attrib_name)
        setattr(sketch, attrib_name, attrib_value)
    self.active_page.sketches.append(sketch)

    return self


def _measure_latex_formula(
    formula: str,
    font_size: int,
    font_family: str | None,
    bold: bool,
) -> tuple[float, float]:
    """Return the rendered size of a math formula.

    Args:
        formula: LaTeX math string without surrounding dollar signs.
        font_size: Font size in points.
        font_family: Mathtext font family, or None for the current default.
        bold: If true, wrap the formula in ``\\boldsymbol``.

    Returns:
        tuple[float, float]: Width and height in points.
    """
    import io
    import re

    import matplotlib

    _FONTSET_MAP = {
        "computer modern": "cm",
        "cm": "cm",
        "stix": "stix",
        "stix sans": "stixsans",
        "stixsans": "stixsans",
        "dejavu sans": "dejavusans",
        "dejavusans": "dejavusans",
        "dejavu": "dejavusans",
        "dejavu serif": "dejavuserif",
        "dejavuserif": "dejavuserif",
    }
    _TEXT_MODE_MAP = [
        (r"\texttt", r"\mathtt"),
        (r"\textrm", r"\mathrm"),
        (r"\textbf", r"\mathbf"),
        (r"\textit", r"\mathit"),
        (r"\textsf", r"\mathsf"),
    ]

    if bold:
        formula = rf"\boldsymbol{{{formula}}}"
    for src, dst in _TEXT_MODE_MAP:
        formula = formula.replace(src, dst)
    if not font_family and (r"\mathbf" in formula or r"\boldsymbol" in formula):
        font_family = "stix"

    fontset = _FONTSET_MAP.get((font_family or "").strip().lower())
    rc_overrides = {"mathtext.fontset": fontset} if fontset else {}

    with matplotlib.rc_context(rc_overrides):
        fig = matplotlib.pyplot.figure(figsize=(0.01, 0.01), dpi=72)
        fig.text(0, 0, f"${formula}$", fontsize=font_size, usetex=False)
        buf = io.StringIO()
        fig.savefig(
            buf,
            format="svg",
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.05,
        )
        matplotlib.pyplot.close(fig)

    svg_str = buf.getvalue()
    w_match = re.search(r'<svg[^>]*\bwidth="([\d.]+)pt"', svg_str)
    h_match = re.search(r'<svg[^>]*\bheight="([\d.]+)pt"', svg_str)
    W = float(w_match.group(1)) if w_match else 100.0
    H = float(h_match.group(1)) if h_match else 20.0
    return W, H


def draw_latex(
    self: Canvas,
    formula: str,
    pos: PointType,
    font_size: int = 14,
    font_family: str | None = None,
    font_color: Color | str | Sequence[float] | None = None,
    bold: bool = False,
    anchor: Anchor | None = None,
    **kwargs: object,
) -> Self:
    """Draw a LaTeX math formula on the canvas using matplotlib mathtext (no TeX compiler needed).

    Args:
        formula (str): LaTeX math string without surrounding $. E.g. r'\\frac{a}{b}'.
            Text-mode commands are silently mapped to their math-mode equivalents:
            \\texttt → \\mathtt (monospace), \\textrm → \\mathrm, \\textbf → \\mathbf,
            \\textit → \\mathit, \\textsf → \\mathsf.
        pos (PointType): Canvas position of the formula anchor.
        font_size (int): Font size in points. Defaults to 14.
        font_family (str, optional): Mathtext fontset — 'computer modern'/'cm', 'stix',
            'stix sans'/'stixsans', 'dejavu sans'/'dejavusans', 'dejavu serif'/'dejavuserif'.
            If omitted and the formula contains \\mathbf{}, STIX is chosen automatically
            (closest to LaTeX output). Otherwise matplotlib's current default is used.
        font_color: Formula colour — simetri Color, (r,g,b) tuple, or matplotlib colour
            string (e.g. 'red', '#ff0000'). Defaults to black.
        bold (bool): Wrap the *entire* formula in \\mathbf{}. For partial bold, write
            \\mathbf{} directly in the formula string — STIX is still selected automatically.
            Defaults to False.
        anchor (Anchor, optional): Anchor point. Defaults to Anchor.SOUTHWEST.
        **kwargs: Style overrides applied to the formula sketch.

    Returns:
        Self: The canvas object.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.draw_latex("x", (0, 0), visible=False) is canvas
        True
        >>> canvas.active_page.sketches[-1].visible
        False
    """
    self._sketch_xform_matrix = self.xform_matrix
    sketch = LatexSketch(
        formula=formula,
        pos=pos,
        font_size=font_size,
        font_family=font_family,
        font_color=font_color,
        bold=bold,
        anchor=anchor,
        xform_matrix=self._sketch_xform_matrix,
    )
    self._sketch_xform_matrix = identity_matrix()
    for name, value in kwargs.items():
        setattr(sketch, name, value)
    # Measure the formula's rendered bounding box so we can register the
    # correct canvas extents in _all_vertices.
    W, H = _measure_latex_formula(
        formula, sketch.font_size, font_family, bold
    )
    sketch.formula_size = (W, H)

    # Compute the anchor offset within the formula box (same table as svg.py)
    _anchor_offsets = {
        Anchor.SOUTHWEST: (0, 0),
        Anchor.SOUTH: (W / 2, 0),
        Anchor.SOUTHEAST: (W, 0),
        Anchor.WEST: (0, H / 2),
        Anchor.CENTER: (W / 2, H / 2),
        Anchor.EAST: (W, H / 2),
        Anchor.NORTHWEST: (0, H),
        Anchor.NORTH: (W / 2, H),
        Anchor.NORTHEAST: (W, H),
    }
    resolved_anchor = anchor if anchor is not None else Anchor.SOUTHWEST
    ax, ay = _anchor_offsets.get(resolved_anchor, (0, 0))

    # sketch.pos is already in transformed canvas-space (xform_matrix applied)
    sx, sy = sketch.pos[:2]
    # All four corners of the formula box in canvas-space (y-up)
    self._all_vertices.extend(
        [
            (sx - ax, sy - ay),  # SW
            (sx + W - ax, sy - ay),  # SE
            (sx - ax, sy + H - ay),  # NW
            (sx + W - ax, sy + H - ay),  # NE
        ]
    )
    self.active_page.sketches.append(sketch)
    return self


def insert_svg(self: Canvas, code: str, location: SvgLoc = SvgLoc.NONE) -> Self:
    """Insert an SVG fragment at the given location.

    Args:
        code: SVG markup to insert.
        location: Where the snippet is placed. Defaults to ``SvgLoc.NONE``.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.insert_svg('<circle cx="0" cy="0" r="10"/>') is canvas
        True
        >>> len(canvas.active_page.sketches)
        1
    """
    active_sketches = self.active_page.sketches
    sketch = SvgSketch(code, location=location)
    active_sketches.append(sketch)

    return self


def insert_tex(self: Canvas, code: str, location: TexLoc = TexLoc.NONE) -> Self:
    """Insert a TeX snippet at the given location.

    Args:
        code: TeX source to insert.
        location: Where the snippet is placed. Defaults to ``TexLoc.NONE``.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.insert_tex("% note") is canvas
        True
        >>> len(canvas.active_page.sketches)
        1
    """
    active_sketches = self.active_page.sketches
    sketch = TexSketch(code, location=location)
    active_sketches.append(sketch)

    return self


def _bbox_feature_style(option: object) -> dict[str, object] | None:
    """Return draw kwargs for a bbox feature, or None to skip it.

    ``False`` skips. ``True`` uses default line color and width.
    A style dict or ``Style`` is applied as given.
    """
    if option is False:
        return None
    if option is True:
        return {
            "fill": False,
            "stroke": True,
            "line_color": runtime_defaults["line_color"],
            "line_width": runtime_defaults["line_width"],
        }
    return coerce_style_overlay(option)


def _merge_style(
    base: dict[str, object], extra: dict[str, object]
) -> dict[str, object]:
    """Return ``base`` with ``extra`` keys overwriting."""
    merged = dict(base)
    for key in extra:
        merged[key] = extra[key]
    return merged


def draw_bbox(
    self: Canvas,
    bbox: BoundingBox,
    border: bool | dict[str, object] = False,
    centerlines: bool | dict[str, object] = False,
    diagonals: bool | dict[str, object] = False,
    **kwargs: object,
) -> Self:
    """Draw a bounding box.

    If ``border``, ``centerlines``, and ``diagonals`` are all False, the
    box is drawn as a bounding-box sketch (same as ``canvas.draw(bbox)``).
    If a flag is True, those lines use the default line color and width.
    If a flag is a style dict, those lines use that style.

    Args:
        bbox: Bounding box to draw.
        border: Rectangle outline. Defaults to False.
        centerlines: Horizontal and vertical centerlines. Defaults to False.
        diagonals: Both diagonals. Defaults to False.
        **kwargs: Extra style forwarded to each drawn part, or to the
            bounding-box sketch when all flags are False.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> box = sg.BoundingBox((0, 0), (40, 20))
        >>> canvas.draw_bbox(box) is canvas
        True
        >>> canvas.active_page.sketches[0].subtype.name
        'BBOX_SKETCH'
        >>> len(canvas._all_vertices)
        4
        >>> canvas.draw_bbox(box, border=True) is canvas
        True
        >>> canvas.active_page.sketches[1].subtype.name
        'SHAPE_SKETCH'
        >>> before = len(canvas.active_page.sketches)
        >>> canvas.draw_bbox(box, centerlines=True, diagonals={"line_width": 2}) is canvas
        True
        >>> len(canvas.active_page.sketches) - before
        4
    """
    border_style = _bbox_feature_style(border)
    centerlines_style = _bbox_feature_style(centerlines)
    diagonals_style = _bbox_feature_style(diagonals)
    if (
        border_style is None
        and centerlines_style is None
        and diagonals_style is None
    ):
        extend_vertices(self, bbox)
        sketch = create_sketch(bbox, self, **kwargs)
        self.active_page.sketches.append(sketch)
        return self

    if border_style is not None:
        draw(
            self,
            Shape(bbox.corners, closed=True),
            **_merge_style(border_style, kwargs),
        )
    if centerlines_style is not None:
        part_style = _merge_style(centerlines_style, kwargs)
        vertical = bbox.vert_centerline
        horizontal = bbox.horiz_centerline
        draw(self, Shape([vertical[0], vertical[1]]), **part_style)
        draw(self, Shape([horizontal[0], horizontal[1]]), **part_style)
    if diagonals_style is not None:
        part_style = _merge_style(diagonals_style, kwargs)
        first = bbox.diagonal1
        second = bbox.diagonal2
        draw(self, Shape([first[0], first[1]]), **part_style)
        draw(self, Shape([second[0], second[1]]), **part_style)

    return self


_GEOMETRIC_GRID_TYPES = frozenset(
    (
        Types.CIRCULAR_GRID,
        Types.HEX_GRID,
        Types.MIXED_GRID,
        Types.SQUARE_GRID,
    )
)


def draw_geometric_grid(self: Canvas, grid: Group, **kwargs: object) -> Self:
    """Draw a geometric grid from ``connections``, ``skip``, ``border``,
    ``orthogonals``, ``diagonals``, and ``centerlines``.

    Args:
        grid: ``CircularGrid``, ``SquareGrid``, ``HexGrid``, or ``Grid``.
        **kwargs: Extra style forwarded to each drawn part. ``indices=True``
            labels the grid vertices only, not the chord endpoints.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> grid = sg.CircularGrid(n=6, radius=40)
        >>> grid.connections = [3]
        >>> grid.orthogonals = False
        >>> canvas.draw(grid) is canvas
        True
        >>> [sketch.subtype.name for sketch in canvas.active_page.sketches]
        ['SHAPE_SKETCH', 'SHAPE_SKETCH', 'SHAPE_SKETCH', 'CIRCLE_SKETCH']
        >>> canvas = sg.Canvas()
        >>> grid = sg.SquareGrid(n=16, cell_size=25)
        >>> canvas.draw(grid, indices=True) is canvas
        True
        >>> [
        ...     len(sketch.vertices)
        ...     for sketch in canvas.active_page.sketches
        ...     if "indices" in sketch.__dict__ and sketch.indices
        ... ]
        [16]
        >>> [
        ...     sketch.subtype.name
        ...     for sketch in canvas.active_page.sketches
        ...     if "indices" not in sketch.__dict__ or not sketch.indices
        ... ].count("SHAPE_SKETCH")
        22
        >>> canvas = sg.Canvas()
        >>> grid = sg.SquareGrid(n=16, cell_size=25)
        >>> grid.orthogonals = False
        >>> grid.diagonals = True
        >>> canvas.draw(grid) is canvas
        True
        >>> len(canvas.active_page.sketches)
        2
        >>> canvas = sg.Canvas()
        >>> grid = sg.SquareGrid(n=16, cell_size=25)
        >>> grid.orthogonals = False
        >>> grid.centerlines = True
        >>> canvas.draw(grid) is canvas
        True
        >>> len(canvas.active_page.sketches)
        2
    """
    points = list(grid.points)
    part_kwargs = dict(kwargs)
    if "indices" in part_kwargs:
        del part_kwargs["indices"]
    line_style = {
        "fill": False,
        "stroke": True,
        "line_color": runtime_defaults["grid_line_color"],
        "line_width": runtime_defaults["grid_line_width"],
    }
    line_style = _merge_style(line_style, part_kwargs)
    for first, second in grid.line_index_pairs():
        draw(
            self,
            Shape([points[first], points[second]]),
            **line_style,
        )

    ortho_style = _bbox_feature_style(grid.orthogonals)
    if ortho_style is not None:
        for first, second in grid.orthogonal_index_pairs():
            draw(
                self,
                Shape([points[first], points[second]]),
                **_merge_style(ortho_style, part_kwargs),
            )

    centerlines_style = _bbox_feature_style(grid.centerlines)
    if centerlines_style is not None and points:
        box = bounding_box(points)
        part_style = _merge_style(centerlines_style, part_kwargs)
        vertical = box.vert_centerline
        horizontal = box.horiz_centerline
        draw(self, Shape([vertical[0], vertical[1]]), **part_style)
        draw(self, Shape([horizontal[0], horizontal[1]]), **part_style)

    diag_style = _bbox_feature_style(grid.diagonals)
    if diag_style is not None:
        for first, second in grid.diagonal_index_pairs():
            draw(
                self,
                Shape([points[first], points[second]]),
                **_merge_style(diag_style, part_kwargs),
            )

    border_style = _bbox_feature_style(grid.border)
    if border_style is not None:
        draw(
            self,
            Shape(points, closed=True),
            **_merge_style(border_style, part_kwargs),
        )

    if "indices" in kwargs and kwargs["indices"]:
        index_style: dict[str, object] = {
            "fill": False,
            "stroke": False,
            "indices": True,
        }
        for key in kwargs:
            if key == "indices" or key.startswith("index_"):
                index_style[key] = kwargs[key]
        draw(self, Shape(points), **index_style)

    for element in grid:
        if element is grid._points:
            continue
        draw(self, element, **part_kwargs)
    return self


def draw_pattern(self: Canvas, pattern: Pattern, **kwargs: object) -> Self:
    """Draw a pattern by sketching each shape from ``get_shapes()``.

    Args:
        pattern: Pattern to expand and draw.
        **kwargs: Style overrides forwarded to ``get_sketches``.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> kernel = sg.Shape([(0, 0), (40, 0), (40, 40)], closed=True)
        >>> pat = sg.Pattern(kernel)
        >>> _ = pat.translate(40, 0, reps=1)
        >>> from simetri.render.draw import draw_pattern
        >>> draw_pattern(canvas, pat) is canvas
        True
    """
    active_sketches = self.active_page.sketches
    shapes = pattern.get_shapes()
    for shape in shapes:
        sketches = get_sketches(shape, self, **kwargs)
        if sketches:
            active_sketches.extend(sketches)

    return self


def draw_group(self: Canvas, group: Group, **kwargs: object) -> Self:
    """Draw a group.

    Args:
        group: Group to draw.
        **kwargs: Style overrides forwarded to the group sketch.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> group = sg.Group([sg.Shape([(0, 0), (40, 0), (40, 40)])])
        >>> canvas.draw(group) is canvas
        True
        >>> len(canvas.active_page.sketches)
        1
    """
    sketch = create_sketch(group, self, **kwargs)
    self.active_page.sketches.append(sketch)

    return self


def draw_widget(self: Canvas, item: Drawable, **kwargs: object) -> Self:
    """Draw an item that exposes ``draw_list`` as a composite sketch.

    Args:
        item: Drawable with a ``draw_list`` of drawables and/or functions.
        **kwargs: Style overrides applied to nested drawables.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> from types import SimpleNamespace
        >>> widget = SimpleNamespace(
        ...     draw_list=[sg.Shape([(0, 0), (40, 0), (40, 40)])],
        ... )
        >>> canvas = sg.Canvas()
        >>> canvas.draw_widget(widget) is canvas
        True
    """
    active_sketches = self.active_page.sketches
    first_sketch_index = len(active_sketches)

    for drawable_item in item.draw_list:
        if callable(drawable_item):
            drawable_item(self, item, **kwargs)
            continue
        draw(self, drawable_item, **kwargs)

    widget_sketches = active_sketches[first_sketch_index:]
    if widget_sketches:
        del active_sketches[first_sketch_index:]
        active_sketches.append(CompositeSketch(widget_sketches))

    return self


def draw_hobby(
    self: Canvas,
    points: Sequence[PointType],
    controls: Sequence[PointType],
    cyclic: bool = False,
    **kwargs: object,
) -> Self:
    """Draw a Hobby curve through the given points using the control points.

    Args:
        points (Sequence[PointType]): Points through which the curve passes.
        controls (Sequence[PointType]): Control points for the curve.
        cyclic (bool, optional): Whether the curve is cyclic. Defaults to False.
        **kwargs: Style overrides forwarded to each Bezier ``draw`` call.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.draw import draw_hobby
        >>> canvas = sg.Canvas()
        >>> draw_hobby(canvas, [(0, 0), (40, 0)], [(20, 20), (20, -20)]) is canvas
        True
    """
    n = len(points)
    if cyclic:
        for i in range(n):
            ind = i * 2
            bezier_pnts = bezier_points(
                points[i], *controls[ind : ind + 2], points[(i + 1) % n], 20
            )
            bezier_ = Shape(bezier_pnts)
            self.draw(bezier_, **kwargs)
    else:
        for i in range(len(points) - 1):
            ind = i * 2
            bezier_pnts = bezier_points(
                points[i], *controls[ind : ind + 2], points[i + 1], 20
            )
            bezier_ = Shape(bezier_pnts)
            self.draw(bezier_, **kwargs)
    return self


_LACE_DRAW_OPTION_NAMES = frozenset(
    (
        "draw_fragments",
        "draw_plaits",
        "fillet_radii",
        "fragment_coloring",
        "fragments",
        "line_widths",
        "palette",
        "percent_offsets",
        "plait_color",
        "plait_fill_color",
        "plait_style",
        "plaits",
        "shade_plaits",
        "swatch",
    )
)


def _lace_style_kwargs(kwargs: DrawStyleKwargs) -> DrawStyleKwargs:
    """Return draw kwargs with lace-only option names removed."""
    return {
        name: value
        for name, value in kwargs.items()
        if name not in _LACE_DRAW_OPTION_NAMES
    }


def _resolved_plait_fill_color(
    lace: Lace | None, kwargs: DrawStyleKwargs
) -> Color | str | Sequence[float] | Sequence[int]:
    """Return the plait fill color from draw options, then the lace, then defaults.

    Args:
        lace: Lace supplying ``plait_color`` when kwargs do not.
        kwargs: Draw keyword arguments (not mutated).

    Returns:
        Resolved fill color for plait drawing.
    """
    if "plait_color" in kwargs:
        return kwargs["plait_color"]
    if "plait_fill_color" in kwargs:
        return kwargs["plait_fill_color"]
    if "fill_color" in kwargs:
        return kwargs["fill_color"]
    if lace is not None:
        if lace.plait_color is not None:
            return lace.plait_color
    return runtime_defaults["plait_color"]


def _resolved_shade_plaits(kwargs: DrawStyleKwargs) -> bool:
    """Return whether plaits should be shaded for this draw call.

    Args:
        kwargs: Draw keyword arguments (not mutated).

    Returns:
        bool: ``shade_plaits`` from kwargs or ``defaults``.
    """
    if "shade_plaits" in kwargs:
        return kwargs["shade_plaits"]
    return runtime_defaults["shade_plaits"]


def shade_value(angle: float) -> float:
    """Return a shade weight from an angle.

    The weight is ``sin(angle)``. ``pi / 2`` returns 1. ``0`` and
    ``pi`` return 0.

    Args:
        angle: Angle in radians. Must be between 0 and ``2 * pi``.

    Returns:
        float: ``sin(angle)``.

    Raises:
        ValueError: If ``angle`` is outside ``[0, 2 * pi]``.

    Examples:
        >>> import simetri.graphics as sg
        >>> shade_value(sg.pi / 2)
        1.0
        >>> shade_value(0)
        0.0
    """
    if not 0 <= angle <= 2 * pi:
        raise ValueError("Angle must be between 0 and 2 pi radians.")

    return sin(angle)


def plait_emboss1(self: Canvas, lace: Lace, **kwargs: object) -> None:
    """Draw lace plaits with embossed quad shading (style 1).

    Args:
        lace: Lace object whose plaits are embossed (mutated).
        **kwargs: Style overrides such as ``fill_color``.

    Returns:
        None

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> lace = sg.Lace([(0, 0), (20, 0), (40, 40)], closed=True)  # doctest: +SKIP
        >>> canvas.plait_emboss1(lace)  # doctest: +SKIP
    """
    if "fill_color" not in kwargs:
        kwargs["fill_color"] = _resolved_plait_fill_color(lace, kwargs)
    lace._set_plait_ends()
    all_quads = []
    for plait in lace.plaits:
        vertices = plait.vertices
        n = len(vertices)
        ends = []
        emboss_points = []
        plait.emboss_quads = []
        for i, overlap in enumerate(plait.overlaps):
            quad = Shape([inter._point for inter in overlap[1].intersections])
            ind1 = plait.ends[i]
            ind2 = (ind1 + 1) % n
            p1 = vertices[ind1]
            p2 = vertices[ind2]
            mp = midpoint(p1, p2)
            emboss_points.append(((p1, ind1), (p2, ind2), mp))
            ends.append(p1)
            ends.append(p2)
        j = plait.ends[0]
        for i in range(1, int(n / 2) - 1):
            ind1 = (j + i + 1) % n
            ind2 = j - i
            p1 = vertices[ind1]
            p2 = vertices[ind2]
            mp = midpoint(p1, p2)
            emboss_points.append(((p1, ind1), (p2, ind2), mp))

        ep = emboss_points
        quads = plait.emboss_quads
        (_, ind1), (p_2, ind2), mp_1 = ep[0]
        (_, _), (p_4, ind4), mp_2 = ep[1]
        (_, _), (p5, ind5), mp3 = ep[2]
        p6 = vertices[(ind2 + 1) % n]
        p7 = vertices[(ind5 + 1) % n]
        quad1 = (mp_1, p_2, p6, mp3)
        quad2 = (mp_1, mp3, p5, p7)
        quads.append((quad1, quad2, (mp_1, mp3), (p_2, p6), (p5, p7)))
        n_iter = len(ep)
        for i in range(2, n_iter):
            (p1, ind1), (p2, ind2), mp = ep[i]
            if i == n_iter - 1:
                p3 = vertices[(ind1 + 1) % n]
                quad1 = (mp, p1, p3, mp_2)
                quad2 = (mp, mp_2, p_4, p2)
                quads.append((quad1, quad2, (mp, mp_2), (p1, p3), (p_4, p2)))
            else:
                (p3, _), (p4, ind4), mp_ = ep[(i + 1) % n]
                p5 = vertices[(ind4 + 1) % n]
                quad1 = (mp, p1, p3, mp_)
                quad2 = (mp, mp_, p4, p5)
                quads.append((quad1, quad2, (mp, mp_), (p1, p3), (p4, p5)))

        all_quads.extend(quads)

    style_kwargs = _lace_style_kwargs(kwargs)
    if _resolved_shade_plaits(kwargs):
        dist = lace.width * 3
        cx, cy = lace.midpoint
        far_point = cx - dist, cy + dist
        color = _resolved_plait_fill_color(lace, kwargs)
        for quad in all_quads:
            quad1 = Shape(quad[0], closed=True)
            quad2 = Shape(quad[1], closed=True)

            angle = inclination_angle(*quad[2])
            shade_angle = abs(radians(135) - angle)
            shade_factor = shade_value(shade_angle)

            mid1 = midpoint(*quad[2])
            line1 = (far_point, mid1)
            line2 = quad[3]
            line3 = quad[4]

            x1, _ = intersect(line1, line2)
            x2, _ = intersect(line1, line3)
            shade_step = 0.2

            if x2 < x1:
                color1 = change_lightness(color, shade_factor * -shade_step)
                color2 = change_lightness(color, shade_factor * shade_step)
            elif x1 < x2:
                color1 = change_lightness(color, shade_factor * shade_step)
                color2 = change_lightness(color, shade_factor * -shade_step)
            else:
                color1 = color2 = color
            quad1_kwargs = dict(style_kwargs)
            quad1_kwargs["fill_color"] = color1
            quad2_kwargs = dict(style_kwargs)
            quad2_kwargs["fill_color"] = color2
            draw(self, quad1, **quad1_kwargs)
            draw(self, quad2, **quad2_kwargs)
    else:
        for quad in all_quads:
            quad1 = Shape(quad[0], closed=True)
            quad2 = Shape(quad[1], closed=True)
            draw(self, quad1, **style_kwargs)
            draw(self, quad2, **style_kwargs)


def plait_emboss2(self: Canvas, lace: Lace, **kwargs: object) -> None:
    """Draw lace plaits with embossed quad shading (style 2).

    Args:
        lace: Lace object whose plaits are embossed (mutated).
        **kwargs: Style overrides such as ``fill_color``.

    Returns:
        None

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> lace = sg.Lace([(0, 0), (20, 0), (40, 40)], closed=True)  # doctest: +SKIP
        >>> canvas.plait_emboss2(lace)  # doctest: +SKIP
    """
    if "fill_color" not in kwargs:
        kwargs["fill_color"] = _resolved_plait_fill_color(lace, kwargs)
    quads = []
    for ppoly in lace.parallel_poly_list:
        for poly in ppoly.offset_poly_list:
            n = len(poly.sections)
            count = 0
            for j, sect in enumerate(poly.sections):
                if sect.is_overlap:
                    break
                count += 1

            if count:
                poly_sections = poly.sections[count:] + poly.sections[:count]
            else:
                poly_sections = poly.sections
            quad1 = []
            for j, sect in enumerate(poly_sections):
                overlap_sect = sect.is_overlap
                if overlap_sect:
                    mp = Shape(
                        [x._point for x in sect.overlap.intersections]
                    ).midpoint
                    quad1.append(mp)
                else:
                    p1 = sect.start._point
                    p2 = sect.end._point
                    twin = sect.twin
                    twin_p1 = twin.end._point
                    twin_p2 = twin.start._point
                    intersection_ = intersection((p1, twin_p1), (p2, twin_p2))
                    if intersection_[0] != Connection.INTERSECT:
                        twin_p1, twin_p2 = twin_p2, twin_p1
                    if not quad1:
                        quad1.append(midpoint(p1, twin_p2))
                    quad1.extend([p1, p2])
                    next_section = poly_sections[(j + 1) % n]
                    if next_section.is_overlap:
                        mp = Shape(
                            [
                                x._point
                                for x in next_section.overlap.intersections
                            ]
                        ).midpoint
                    else:
                        mp = midpoint(p2, twin_p1)
                    quad1.append(mp)
                    quad2 = [quad1[0], mp, twin_p1, twin_p2]

                    quads.append(
                        (
                            quad1,
                            quad2,
                            (quad1[0], mp),
                            (p1, p2),
                            (twin_p1, twin_p2),
                        )
                    )
                    quad1 = []
    style_kwargs = _lace_style_kwargs(kwargs)
    if _resolved_shade_plaits(kwargs):
        dist = lace.width * 3
        cx, cy = lace.midpoint
        far_point = cx - dist, cy + dist

        color = _resolved_plait_fill_color(lace, kwargs)
        for quad in quads:
            quad1 = Shape(quad[0], closed=True)
            quad2 = Shape(quad[1], closed=True)

            angle = inclination_angle(*quad[2])
            shade_angle = abs(radians(135) - angle)
            shade_factor = shade_value(shade_angle)

            mid1 = midpoint(*quad[2])
            line1 = (far_point, mid1)
            line2 = quad[3]
            line3 = quad[4]

            x1, _ = intersect(line1, line2)
            x2, _ = intersect(line1, line3)
            shade_step = 0.2

            if x2 < x1:
                color1 = change_lightness(color, shade_factor * -shade_step)
                color2 = change_lightness(color, shade_factor * shade_step)
            elif x1 < x2:
                color1 = change_lightness(color, shade_factor * shade_step)
                color2 = change_lightness(color, shade_factor * -shade_step)
            else:
                color1 = color2 = color
            quad1_kwargs = dict(style_kwargs)
            quad1_kwargs["fill_color"] = color1
            quad2_kwargs = dict(style_kwargs)
            quad2_kwargs["fill_color"] = color2
            draw(self, quad1, **quad1_kwargs)
            draw(self, quad2, **quad2_kwargs)
    else:
        for quad in quads:
            quad1 = Shape(quad[0], closed=True)
            quad2 = Shape(quad[1], closed=True)
            draw(self, quad1, **style_kwargs)
            draw(self, quad2, **style_kwargs)


def plait_diamond(self: Canvas, lace: Lace, **kwargs: object) -> None:
    """Draw lace plaits using a diamond/offset-quad fill style.

    Args:
        lace: Lace object whose plaits are drawn.
        **kwargs: Style overrides such as ``fill_color``.

    Returns:
        None

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> lace = sg.Lace([(0, 0), (20, 0), (40, 40)], closed=True)  # doctest: +SKIP
        >>> canvas.plait_diamond(lace)  # doctest: +SKIP
    """
    lace._set_plait_ends()
    quad_pairs = []
    end_quads = []
    inner_loops = []
    if "fill_color" not in kwargs:
        kwargs["fill_color"] = _resolved_plait_fill_color(lace, kwargs)
    for plait in lace.plaits:
        vertices = list(plait.vertices)
        e1, e2 = plait.ends
        ends = [e1 + 1, e2 + 1]

        count = 0
        for j, vert in enumerate(vertices):
            if j in ends:
                break
            count += 1

        if count:
            vertices = vertices[count:] + vertices[:count]

        n = len(vertices)
        inner_loop = Shape(offset_polygon(vertices, -5), closed=True)
        inner_loops.append(inner_loop)
        # self.draw(plait, fill=False,**kwargs)
        # self.draw(inner_loop, fill=False)
        quads = []

        for i, vert in enumerate(vertices):
            quad = (
                vert,
                vertices[(i + 1) % n],
                inner_loop[(i + 1) % n],
                inner_loop[i],
            )
            shp = Shape(quad, closed=True)
            quads.append(shp)
        n2 = int(len(quads) / 2) - 1
        style_kwargs = _lace_style_kwargs(kwargs)
        for i in range(n2):
            draw(self, quads[i], **style_kwargs)
            draw(self, quads[-(i + 2)], **style_kwargs)
            quad_pairs.append((quads[i], quads[-(i + 2)]))
        end_quads.extend([quads[-1], quads[n2]])

    dist = lace.width * 3
    cx, cy = lace.midpoint
    far_point = cx - dist, cy + dist
    plait_fill_color = _resolved_plait_fill_color(lace, kwargs)
    style_kwargs = _lace_style_kwargs(kwargs)

    if _resolved_shade_plaits(kwargs):
        color = plait_fill_color
        for quad1, quad2 in quad_pairs:
            angle = inclination_angle(*quad1[:2])
            shade_angle = abs(radians(135) - angle)
            shade_factor = shade_value(shade_angle)

            mid1 = midpoint(*quad1[:2])
            line1 = (far_point, mid1)
            line2 = quad1[:2]
            line3 = quad2[:2]

            x1, _ = intersect(line1, line2)
            x2, _ = intersect(line1, line3)
            shade_step = 0.2

            if x2 < x1:
                color1 = change_lightness(color, shade_factor * -shade_step)
                color2 = change_lightness(color, shade_factor * shade_step)
            elif x1 < x2:
                color1 = change_lightness(color, shade_factor * shade_step)
                color2 = change_lightness(color, shade_factor * -shade_step)
            else:
                color1 = color2 = color

            quad1_kwargs = dict(style_kwargs)
            quad1_kwargs["fill_color"] = color1
            quad2_kwargs = dict(style_kwargs)
            quad2_kwargs["fill_color"] = color2
            draw(self, quad1, **quad1_kwargs)
            draw(self, quad2, **quad2_kwargs)

        color = change_lightness(plait_fill_color, -0.1)
        for quad in end_quads:
            end_kwargs = dict(style_kwargs)
            end_kwargs["fill_color"] = color
            draw(self, quad, **end_kwargs)

        color = change_lightness(plait_fill_color, 0.1)
        for loop in inner_loops:
            loop_kwargs = dict(style_kwargs)
            loop_kwargs["fill_color"] = color
            draw(self, loop, **loop_kwargs)
        return

    for quad in end_quads:
        draw(self, quad, **style_kwargs)
    for loop in inner_loops:
        draw(self, loop, **style_kwargs)


def draw_lace_with_fillets(self: Canvas, lace: Lace, **kwargs: object) -> None:
    """Draw a lace with filleted geometry, then apply draw_lace options.

    Args:
        lace: Lace object to draw.
        **kwargs: Must include ``fillet_radii``. Other names are forwarded
            to ``draw_lace``.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> lace = sg.Lace([(0, 0), (20, 0), (40, 40)], closed=True)  # doctest: +SKIP
        >>> canvas.draw_lace_with_fillets(lace, fillet_radii=(40, 80))  # doctest: +SKIP
    """
    fillet_radii = kwargs["fillet_radii"]
    remaining = {
        name: value for name, value in kwargs.items() if name != "fillet_radii"
    }
    draw_lace(self, lace, fillet_radii=fillet_radii, **remaining)


def draw_plaits(
    self: Canvas, lace: Lace | None = None, **kwargs: object
) -> None:
    """Draw lace plaits, optionally using a plait style handler.

    Args:
        lace (optional): Lace object providing ``plaits``. If omitted,
            ``kwargs['plaits']`` is used.
        **kwargs: Style overrides such as ``plait_color`` and
            ``plait_style``.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> from simetri.render.draw import draw_plaits
        >>> plait = sg.Shape([(0, 0), (40, 0), (40, 20)], closed=True)
        >>> draw_plaits(canvas, plaits=[plait], plait_fill_color=sg.red)
        >>> len(canvas.active_page.sketches) > 0
        True
    """
    if "plaits" in kwargs:
        plaits = kwargs["plaits"]
    elif lace is not None:
        plaits = lace.plaits
    else:
        raise KeyError("plaits")

    if "plait_fill_color" not in kwargs:
        kwargs["plait_fill_color"] = _resolved_plait_fill_color(lace, kwargs)
    if "plait_style" in kwargs and kwargs["plait_style"] is not None:
        _handle_plait_style(self, lace, kwargs)
    else:
        _draw_default_plaits(self, lace, kwargs)


def draw_fragments(
    self: Canvas,
    lace: Lace | None = None,
    palette: Sequence[Sequence[float]] | None = None,
    **kwargs: object,
) -> None:
    """Draw lace fragments colored by area or radius bins from a palette.

    Args:
        lace (optional): Lace object providing ``fragments`` and, for
            ``FragmentColoring.RADIUS``, the lace center.
        palette (optional): Color palette; ``swatch`` in ``kwargs``
            overrides it.
        **kwargs: May include ``fragments``, ``fragment_coloring``, and
            style overrides forwarded to ``draw``.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> from simetri.render.draw import draw_fragments
        >>> fragment = sg.Shape([(0, 0), (20, 0), (20, 20)], closed=True)
        >>> draw_fragments(canvas, fragments=[fragment])
        >>> len(canvas.active_page.sketches) > 0
        True
    """
    if "fragments" in kwargs:
        fragments = kwargs["fragments"]
    elif lace is not None:
        fragments = lace.fragments
    else:
        raise KeyError("fragments")

    fragments_by_id = {fragment.id: fragment for fragment in fragments}
    if "fragment_coloring" in kwargs:
        fragment_coloring = kwargs["fragment_coloring"]
    else:
        fragment_coloring = runtime_defaults["fragment_coloring"]
    if fragment_coloring == FragmentColoring.AREA:
        items = [(fragment.area, fragment.id) for fragment in fragments]
        if lace is not None:
            threshold = lace.area_threshold
        else:
            threshold = runtime_defaults["area_threshold"]
    elif fragment_coloring == FragmentColoring.RADIUS:
        if lace is None:
            raise ValueError("FragmentColoring.RADIUS requires a Lace object.")
        center = lace.center
        items = [
            (distance(center, fragment.CG), fragment.id)
            for fragment in fragments
        ]
        threshold = lace.radius_threshold
    else:
        raise ValueError(f"Unknown fragment_coloring: {fragment_coloring!r}")

    bins = group_into_bins(items, threshold)

    if "swatch" in kwargs:
        palette = kwargs["swatch"]
    elif palette is None:
        palette = runtime_defaults["swatch"]

    n_palette = len(palette)
    n_bins = len(bins)
    palette = palette[:: n_palette // n_bins]
    palette = [Color(*c) for c in palette]
    n = len(palette)
    style_kwargs = _lace_style_kwargs(kwargs)
    for i, bin_ in enumerate(bins):
        color = palette[i % n]
        for _, fragment_id in bin_:
            fragment = fragments_by_id[fragment_id]
            draw_kwargs = dict(style_kwargs)
            draw_kwargs["fill_color"] = color
            draw(self, fragment, **draw_kwargs)


def _handle_plait_innerlines(
    canvas: Canvas, lace: Lace, **kwargs: object
) -> None:
    """Handle INNERLINES plait style."""
    style_kwargs = _lace_style_kwargs(kwargs)
    for plait in lace.plaits:
        extend_vertices(canvas, plait)
        canvas.active_page.sketches.append(
            create_sketch(plait, canvas, **style_kwargs)
        )

    if not lace.plaits[0].lerp_points:
        if "percent_offsets" in kwargs:
            offsets = kwargs["percent_offsets"]
        else:
            offsets = runtime_defaults["percent_offsets"]
        if "line_widths" in kwargs:
            widths = kwargs["line_widths"]
        else:
            widths = runtime_defaults["line_widths"]

        lace._set_plait_inner_lines(offsets, widths)

        if "line_widths" in kwargs and len(kwargs["line_widths"]) == len(
            lace.plaits[0].lerp_points[0]
        ):
            widths = kwargs["line_widths"]
        else:
            widths = False

        for plait in lace.plaits:
            for i in range(len(plait.lerp_points[0])):
                points = [pnts[i] for pnts in plait.lerp_points]
                shape = Shape(points)
                extend_vertices(canvas, shape)
                sketch_kwargs = dict(style_kwargs)
                if widths:
                    sketch_kwargs["line_width"] = plait.line_widths[i]
                else:
                    sketch_kwargs["line_width"] = runtime_defaults["line_width"]
                canvas.active_page.sketches.append(
                    create_sketch(shape, canvas, **sketch_kwargs)
                )


def _handle_plait_double_lines(
    canvas: Canvas, lace: Lace, kwargs: DrawStyleKwargs
) -> None:
    """Draw plaits with ``draw_double=True``."""
    if "plaits" in kwargs:
        plaits = kwargs["plaits"]
    else:
        plaits = lace.plaits
    style_kwargs = _lace_style_kwargs(kwargs)
    fill_color = _resolved_plait_fill_color(lace, kwargs)
    for plait in plaits:
        draw_kwargs = dict(style_kwargs)
        draw_kwargs["fill_color"] = fill_color
        draw_kwargs["draw_double"] = True
        draw(canvas, plait, **draw_kwargs)


def _handle_plait_style(
    canvas: Canvas, lace: Lace | None, kwargs: DrawStyleKwargs
) -> None:
    """Handle different plait styles."""
    p_style = kwargs["plait_style"]

    if p_style == PlaitStyle.INNERLINES:
        _handle_plait_innerlines(canvas, lace, **kwargs)
    elif p_style == PlaitStyle.INNERLOOPS:
        raise ValueError("PlaitStyle.INNERLOOPS is not implemented.")
    elif p_style == PlaitStyle.DIAMOND:
        plait_diamond(canvas, lace, **kwargs)
    elif p_style == PlaitStyle.EMBOSS1:
        plait_emboss1(canvas, lace, **kwargs)
    elif p_style == PlaitStyle.EMBOSS2:
        plait_emboss2(canvas, lace, **kwargs)
    elif p_style == PlaitStyle.DOUBLE_LINES:
        _handle_plait_double_lines(canvas, lace, kwargs)
    else:
        raise ValueError(f"Unknown plait_style: {p_style!r}")


def _draw_default_plaits(
    canvas: Canvas, lace: Lace, kwargs: DrawStyleKwargs
) -> None:
    """Draw plaits with default style."""
    if "plaits" in kwargs:
        plaits = kwargs["plaits"]
    else:
        plaits = lace.plaits
    style_kwargs = _lace_style_kwargs(kwargs)
    fill_color = _resolved_plait_fill_color(lace, kwargs)
    for plait in plaits:
        draw_kwargs = dict(style_kwargs)
        draw_kwargs["fill_color"] = fill_color
        extend_vertices(canvas, plait)
        canvas.active_page.sketches.append(
            create_sketch(plait, canvas, **draw_kwargs)
        )


_draw_fragments = draw_fragments
_draw_plaits = draw_plaits


def draw_lace(
    self: Canvas,
    lace: Lace,
    *,
    fragment_coloring: FragmentColoring | None = None,
    plait_style: PlaitStyle | None = None,
    shade_plaits: bool | None = None,
    fillet_radii: tuple[float, float] | None = None,
    palette: Sequence[Sequence[float]] | None = None,
    swatch: Sequence[Sequence[float]] | None = None,
    plait_color: Color | str | Sequence[float] | Sequence[int] | None = None,
    draw_fragments: bool | None = None,
    draw_plaits: bool | None = None,
    percent_offsets: Sequence[float] | None = None,
    line_widths: Sequence[float] | bool | None = None,
    **kwargs: object,
) -> Self:
    """Draw the lace object.

    Lace-specific options belong here, not on ``canvas.draw``. Generic
    shape styles in ``kwargs`` are forwarded to fragment and plait draws.

    Args:
        lace: Lace object to be drawn.
        fragment_coloring: Color equivalent fragments by area, or by
            distance from the lace center.
        plait_style: Plait drawing style. ``None`` uses filled plaits.
        shade_plaits: Shade embossed or diamond plaits.
        fillet_radii: ``(inner, outer)`` fillet radii. ``None`` does not
            fillet.
        palette: Fragment color palette.
        swatch: Overrides ``palette`` when given.
        plait_color: Fill color for plaits.
        draw_fragments: Draw fragment regions.
        draw_plaits: Draw plaits.
        percent_offsets: Inner-line positions for ``PlaitStyle.INNERLINES``.
        line_widths: Inner-line widths for ``PlaitStyle.INNERLINES``.
        **kwargs: Generic shape style overrides.

    Returns:
        Self: The canvas object.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> lace = sg.Lace([(0, 0), (20, 0), (40, 40)], closed=True)  # doctest: +SKIP
        >>> canvas.draw_lace(lace, draw_fragments=False, draw_plaits=False) is canvas  # doctest: +SKIP
        True
    """
    if fragment_coloring is None:
        fragment_coloring = runtime_defaults["fragment_coloring"]
    if plait_style is None:
        plait_style = runtime_defaults["lace_plait_style"]
    if shade_plaits is None:
        shade_plaits = runtime_defaults["shade_plaits"]
    if fillet_radii is None:
        fillet_radii = runtime_defaults["fillet_radii"]
    if swatch is not None:
        palette = swatch
    elif palette is None:
        if lace.palette is not None:
            palette = lace.palette
        elif lace.swatch is not None:
            palette = lace.swatch
        else:
            palette = runtime_defaults["swatch"]
    if plait_color is None:
        plait_color = _resolved_plait_fill_color(lace, kwargs)
    if draw_fragments is None:
        draw_fragments = runtime_defaults["draw_fragments"]
    if draw_plaits is None:
        draw_plaits = runtime_defaults["draw_plaits"]
    if percent_offsets is None:
        percent_offsets = runtime_defaults["percent_offsets"]
    if line_widths is None:
        line_widths = runtime_defaults["line_widths"]

    style_kwargs = _lace_style_kwargs(kwargs)
    fragments = lace.fragments
    plaits = lace.plaits
    if fillet_radii is not None:
        inner_radius, outer_radius = fillet_radii
        fragments = lace._fillet_fragments(inner_radius, outer_radius)
        plaits = lace._fillet_plaits(inner_radius, outer_radius)

    if draw_fragments:
        fragment_kwargs = dict(style_kwargs)
        fragment_kwargs["fragments"] = fragments
        fragment_kwargs["fragment_coloring"] = fragment_coloring
        _draw_fragments(self, lace, palette=palette, **fragment_kwargs)

    if draw_plaits:
        plait_kwargs = dict(style_kwargs)
        plait_kwargs["plaits"] = plaits
        plait_kwargs["plait_color"] = plait_color
        plait_kwargs["shade_plaits"] = shade_plaits
        plait_kwargs["percent_offsets"] = percent_offsets
        plait_kwargs["line_widths"] = line_widths
        if plait_style is not None:
            plait_kwargs["plait_style"] = plait_style
        _draw_plaits(self, lace, **plait_kwargs)

    return self


def draw_lines(
    self: Canvas, lines: Sequence[Sequence[PointType]], **kwargs: object
) -> Self:
    """Draw a collection of line segments onto the canvas.

    Args:
        lines: Sequence of line segments ``((x1, y1), (x2, y2))``.
        **kwargs: Style overrides for the line sketches.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.draw_lines([((0, 0), (40, 0))]) is canvas
        True
        >>> len(canvas.active_page.sketches)
        1
    """
    self._sketch_xform_matrix = self.xform_matrix
    for segment in lines:
        start = segment[0]
        end = segment[1]
        line_shape = Shape([start, end], closed=False, **kwargs)
        extend_vertices(self, line_shape)
        line_sketch = create_sketch(line_shape, self, **kwargs)
        self.active_page.sketches.append(line_sketch)
    self._sketch_xform_matrix = identity_matrix()
    return self


_MARKER_DEFAULT_KEYS = (
    "marker_alpha",
    "marker_color",
    "marker_line_style",
    "marker_line_width",
    "marker_radius",
    "marker_shape",
    "marker_size",
    "marker_type",
)


def _marker_style_as_draw_kwargs(marker_style: object) -> dict[str, object]:
    """Return draw-alias kwargs from a MarkerStyle, Style, or style dict."""
    if isinstance(marker_style, MarkerStyle):
        return {
            "marker_alpha": marker_style.alpha,
            "marker_color": marker_style.color,
            "marker_radius": marker_style.radius,
            "marker_shape": marker_style.shape,
            "marker_size": marker_style.size,
            "marker_type": marker_style.marker_type,
        }
    return coerce_style_overlay(marker_style)


def draw_points(
    self: Canvas,
    points: Sequence[PointType],
    marker_style: object | None = None,
    **kwargs: object,
) -> Self:
    """Draw markers at the given points.

    If ``marker_style`` is omitted, default marker values are used.
    A ``MarkerStyle``, style dict, or ``marker_*`` keyword arguments
    override those defaults.

    Args:
        points: Sequence of ``(x, y)`` positions.
        marker_style: Optional marker style. Defaults to None (library
            marker defaults).
        **kwargs: Extra style forwarded to ``draw``.

    Returns:
        Self: The canvas.

    Raises:
        ValueError: If ``points`` is empty.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.draw_points([(0, 0), (40, 0)]) is canvas
        True
        >>> sketch = canvas.active_page.sketches[0]
        >>> sketch.draw_markers
        True
        >>> sketch.markers_only
        True
        >>> sketch.marker_type == sg.defaults["marker_type"]
        True
        >>> sketch.marker_size == sg.defaults["marker_size"]
        True
        >>> tuple(sketch.vertices)
        ((0.0, 0.0), (40.0, 0.0))
        >>> canvas.draw_points([(0, 0)], marker_size=5) is canvas
        True
        >>> canvas.active_page.sketches[1].marker_size
        5
    """
    point_list = list(points)
    if not point_list:
        raise ValueError("points must be a non-empty sequence")
    draw_kwargs: dict[str, object] = {
        "draw_markers": True,
        "markers_only": True,
    }
    for name in _MARKER_DEFAULT_KEYS:
        draw_kwargs[name] = runtime_defaults[name]
    if marker_style is not None:
        overlay = _marker_style_as_draw_kwargs(marker_style)
        for name in overlay:
            draw_kwargs[name] = overlay[name]
    for name in kwargs:
        draw_kwargs[name] = kwargs[name]
    draw(self, Shape(point_list), **draw_kwargs)
    return self


def draw_image(
    self: Canvas,
    image: Image,
    position: PointType | None = None,
    scale: tuple[float, float] | float | None = None,
    **kwargs: object,
) -> Self:
    """Draw the image object.

    Args:
        image: Image object to be drawn.
        position: Position to draw the image at.
        scale: Extra scale applied on top of ``image.size``.
            ``None`` uses ``(1, 1)``.
        **kwargs: Additional keyword arguments.

    Returns:
        Self: The canvas object.

    Examples:
        >>> from PIL import Image as PILImage
        >>> import simetri.graphics as sg
        >>> from simetri.images.image import Image
        >>> image = sg.Image(PILImage.new('RGB', (2, 2)))
        >>> from simetri.render.draw import draw_image
        >>> canvas = sg.Canvas()
        >>> draw_image(canvas, image) is canvas
        True
    """
    if not image.visible:
        return self

    _translation, rotation, _ = decompose_transformations(image.xform_matrix)
    if position is None:
        x, y = image.pos[:2]
    else:
        x, y = position[:2]
    pos = [x, y]

    if scale is None:
        scale = (1, 1)

    owns_matrix = (self._sketch_xform_matrix == identity_matrix()).all()
    if owns_matrix:
        self._sketch_xform_matrix = self.xform_matrix
    extend_vertices(self, image)
    sketch = ImageSketch(
        image,
        pos=pos,
        angle=rotation,
        scale=scale,
        size=image.size,
        file_path=image.file_path,
        anchor=image.anchor,
        xform_matrix=self._sketch_xform_matrix,
    )
    if owns_matrix:
        self._sketch_xform_matrix = identity_matrix()
    for attrib_name in shape_style_map:
        attrib_value = self.resolve_property(image, attrib_name)
        setattr(sketch, attrib_name, attrib_value)
    for attrib_name, attrib_value in kwargs.items():
        setattr(sketch, attrib_name, attrib_value)
    self.active_page.sketches.append(sketch)

    return self


def draw_pdf(
    self: Canvas,
    pdf: str | PDF,
    pos: PointType | None = None,
    size: tuple[float, float] | None = None,
    scale: float | None = None,
    angle: float | None = 0,
    **kwargs: object,
) -> Self:
    """Draw a PDF file on the canvas.

    Args:
        pdf: PDF object, or a file path.
        pos: Position of the PDF. None uses the object's position, or
            ``(0, 0)`` when ``pdf`` is a path.
        size: Drawn size. None uses the object's size.
        scale: Scale factor. None uses the object's scale.
        angle: Rotation in radians. Defaults to 0.
        **kwargs: Style overrides for the PDF sketch.

    Returns:
        Self: The canvas.

    Examples:
        >>> import os
        >>> import tempfile
        >>> import simetri.graphics as sg
        >>> fd, path = tempfile.mkstemp(suffix='.pdf')
        >>> _ = os.write(fd, b'%PDF-1.0\\n%%EOF\\n')
        >>> os.close(fd)
        >>> from simetri.render.draw import draw_pdf
        >>> canvas = sg.Canvas()
        >>> draw_pdf(canvas, path) is canvas
        True
        >>> os.unlink(path)
    """
    if not isinstance(pdf, str):
        if not pdf.visible:
            return self
        translation, rotation, decomposed_scale = decompose_transformations(
            pdf.xform_matrix
        )
        if pos is None:
            x, y = pdf.pos[:2]
        else:
            x, y = pos[:2]
        dx, dy = translation
        pos = [x + dx, y + dy]

        if scale is None:
            scale = pdf.scale if pdf.scale is not None else decomposed_scale

        if angle is None:
            angle = rotation
        file_path = pdf.file_path
        if size is None:
            size = pdf.size
    else:
        # If pdf is a file path, we assume it is a PDF object
        pos = pos[:2] if pos else (0, 0)
        scale = scale if scale is not None else 1.0
        file_path = pdf
    if size is None:
        placed = [pos]
    else:
        width, height = size
        if isinstance(scale, (int, float)):
            sx = sy = float(scale)
        else:
            sx, sy = float(scale[0]), float(scale[1])
        half_w = width * sx / 2
        half_h = height * sy / 2
        x, y = pos[:2]
        placed = [
            (x - half_w, y - half_h),
            (x + half_w, y - half_h),
            (x + half_w, y + half_h),
            (x - half_w, y + half_h),
        ]
        if angle:
            placed = (
                homogenize(placed) @ rotation_matrix(angle, (x, y))
            ).tolist()
    self._sketch_xform_matrix = self.xform_matrix
    _extend_canvas_space_points(self, placed)
    sketch = PDFSketch(
        file_path,
        pos=pos,
        scale=scale,
        angle=angle if angle is not None else 0,
        size=size,
        xform_matrix=self._sketch_xform_matrix,
    )
    self._sketch_xform_matrix = identity_matrix()
    for attrib_name, attrib_value in kwargs.items():
        setattr(sketch, attrib_name, attrib_value)
    self.active_page.sketches.append(sketch)

    return self


def _pop_prefixed(draw_kwargs: dict[str, object], prefix: str) -> dict[str, object]:
    """Remove ``prefix`` keys and return them without the prefix."""
    part: dict[str, object] = {}
    for key in list(draw_kwargs):
        if key.startswith(prefix):
            part[key[len(prefix) :]] = draw_kwargs[key]
            del draw_kwargs[key]
    return part


def _stated(part: dict[str, object], name: str) -> tuple[bool, object]:
    """Return whether ``name`` was passed and is not the default marker."""
    if name not in part or part[name] is None:
        return False, None
    return True, part[name]


def _line_part_style(
    part: dict[str, object],
    color: object,
    alpha: object,
) -> dict[str, object]:
    """Line style for an extension line, shaft, or dimension mid-line."""
    style: dict[str, object] = {}
    stated, value = _stated(part, "line_color")
    if stated:
        style["line_color"] = value
    elif color is not None:
        style["line_color"] = color
    else:
        style["line_color"] = runtime_defaults["dim_color"]
    stated, value = _stated(part, "line_width")
    if stated:
        style["line_width"] = value
    else:
        style["line_width"] = runtime_defaults["dim_line_width"]
    stated, value = _stated(part, "line_dash_array")
    if stated:
        style["line_dash_array"] = value
    stated, value = _stated(part, "line_alpha")
    if stated:
        style["line_alpha"] = value
    elif alpha is not None:
        style["line_alpha"] = alpha
    return style


def _head_part_style(
    part: dict[str, object],
    color: object,
    alpha: object,
) -> dict[str, object]:
    """Head style shared by every arrow head on a dimension."""
    style: dict[str, object] = {}
    stated, value = _stated(part, "fill_color")
    if stated:
        style["fill_color"] = value
    elif color is not None:
        style["fill_color"] = color
    else:
        style["fill_color"] = runtime_defaults["dim_color"]
    stated, value = _stated(part, "line_color")
    if stated:
        style["line_color"] = value
    elif color is not None:
        style["line_color"] = color
    else:
        style["line_color"] = runtime_defaults["dim_color"]
    stated, value = _stated(part, "line_width")
    if stated:
        style["line_width"] = value
    stated, value = _stated(part, "fill_alpha")
    if stated:
        style["fill_alpha"] = value
    elif alpha is not None:
        style["fill_alpha"] = alpha
    stated, value = _stated(part, "line_alpha")
    if stated:
        style["line_alpha"] = value
    elif alpha is not None:
        style["line_alpha"] = alpha
    return style


def _apply_part_style(shape: Drawable | None, style: dict[str, object]) -> None:
    """Set resolved style attributes on one dimension part."""
    if shape is None:
        return
    for name, value in style.items():
        setattr(shape, name, value)


def _apply_arrow_style(
    arrow: object,
    shaft_style: dict[str, object],
    head_style: dict[str, object],
) -> None:
    """Style one dimension arrow. ``arrow1`` and ``arrow2`` use this same pair."""
    if arrow is None:
        return
    _apply_part_style(arrow.line, shaft_style)
    for head in arrow.heads:
        _apply_part_style(head, head_style)


def draw_dimension(self: Canvas, item: Dimension, **kwargs: object) -> Self:
    """Draw the dimension object.

    Args:
        item: Dimension object to be drawn.
        **kwargs: ``ext_line_alpha``, ``ext_line_color``,
            ``ext_line_dash_array``, and ``ext_line_width`` style both
            extension lines. ``shaft_line_alpha``, ``shaft_line_color``,
            ``shaft_line_dash_array``, and ``shaft_line_width`` style
            the dimension line and, when the label is at the side, both
            arrow shafts and the line between them. ``head_fill_alpha``,
            ``head_fill_color``, ``head_line_alpha``,
            ``head_line_color``, and ``head_line_width`` style every
            arrow head. ``tag_bold``, ``tag_fill``,
            ``tag_fill_color``, ``tag_font_alpha``, ``tag_font_color``,
            ``tag_font_family``, ``tag_font_size``, ``tag_line_color``,
            ``tag_line_width``, and ``tag_stroke`` style the label.
            ``tag_stroke`` defaults to False. ``color`` and ``alpha``
            set every part; a prefixed name wins.

    Returns:
        Self: The canvas object.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> dim = sg.Dimension((0, 0), (40, 0), "up", 5)
        >>> canvas.draw_dimension(dim) is canvas
        True
        >>> dim.ext1.line_width
        0.75
        >>> dim.dim_line.line.line_width
        0.75
        >>> dim.ext1.line_color == sg.colors.dark_gray
        True
        >>> dim.dim_line.head.fill_color == sg.colors.dark_gray
        True
    """
    for shape in item.all_shapes:
        _extend_canvas_space_points(self, shape.corners)

    def _add_sketch(sketch: Sketch | list[Sketch] | None) -> None:
        if sketch is None:
            return
        if isinstance(sketch, list):
            self.active_page.sketches.extend(sketch)
        else:
            self.active_page.sketches.append(sketch)

    draw_kwargs = dict(kwargs)
    ext_style = _line_part_style(
        _pop_prefixed(draw_kwargs, "ext_"),
        draw_kwargs["color"] if "color" in draw_kwargs else None,
        draw_kwargs["alpha"] if "alpha" in draw_kwargs else None,
    )
    common_color = draw_kwargs["color"] if "color" in draw_kwargs else None
    common_alpha = draw_kwargs["alpha"] if "alpha" in draw_kwargs else None
    if "color" in draw_kwargs:
        del draw_kwargs["color"]
    if "alpha" in draw_kwargs:
        del draw_kwargs["alpha"]
    shaft_style = _line_part_style(
        _pop_prefixed(draw_kwargs, "shaft_"), common_color, common_alpha
    )
    head_style = _head_part_style(
        _pop_prefixed(draw_kwargs, "head_"), common_color, common_alpha
    )
    tag_part = _pop_prefixed(draw_kwargs, "tag_")

    for ext in [item.ext1, item.ext2, item.ext3]:
        if ext:
            _apply_part_style(ext, ext_style)
            _add_sketch(create_sketch(ext, self, **draw_kwargs))
    _apply_arrow_style(item.dim_line, shaft_style, head_style)
    _apply_arrow_style(item.arrow1, shaft_style, head_style)
    _apply_arrow_style(item.arrow2, shaft_style, head_style)
    if item.midline is not None and item.show_midline:
        _apply_part_style(item.midline, shaft_style)
    if item.dim_line:
        _add_sketch(create_sketch(item.dim_line, self, **draw_kwargs))
    if item.arrow1:
        _add_sketch(create_sketch(item.arrow1, self, **draw_kwargs))
        if item.midline is not None and item.show_midline:
            _add_sketch(create_sketch(item.midline, self, **draw_kwargs))
    if item.arrow2:
        _add_sketch(create_sketch(item.arrow2, self, **draw_kwargs))
    x, y = item.text_pos[:2]
    tag_kwargs: dict[str, object] = {}
    if "font_size" in tag_part and tag_part["font_size"] is not None:
        tag_kwargs["font_size"] = tag_part["font_size"]
    elif "font_size" in draw_kwargs:
        tag_kwargs["font_size"] = draw_kwargs["font_size"]
    else:
        tag_kwargs["font_size"] = item.font_size
    if "anchor" in draw_kwargs:
        tag_kwargs["anchor"] = draw_kwargs["anchor"]
    else:
        tag_kwargs["anchor"] = item.text_anchor
    if "align" in draw_kwargs:
        tag_kwargs["align"] = draw_kwargs["align"]
    else:
        tag_kwargs["align"] = item.text_align
    if "fill" in tag_part and tag_part["fill"] is not None:
        tag_kwargs["fill"] = tag_part["fill"]
    elif "fill" in draw_kwargs:
        tag_kwargs["fill"] = draw_kwargs["fill"]
    else:
        tag_kwargs["fill"] = runtime_defaults["fill"]
    if "stroke" in tag_part and tag_part["stroke"] is not None:
        tag_kwargs["stroke"] = tag_part["stroke"]
    else:
        tag_kwargs["stroke"] = False
    stated, value = _stated(tag_part, "font_color")
    if stated:
        tag_kwargs["font_color"] = value
    elif common_color is not None:
        tag_kwargs["font_color"] = common_color
    else:
        tag_kwargs["font_color"] = runtime_defaults["dim_color"]
    stated, value = _stated(tag_part, "font_family")
    if stated:
        tag_kwargs["font_family"] = value
    font_alpha = None
    stated, value = _stated(tag_part, "font_alpha")
    if stated:
        font_alpha = value
    elif common_alpha is not None:
        font_alpha = common_alpha
    stated, value = _stated(tag_part, "bold")
    if stated:
        tag_kwargs["bold"] = value
    stated, value = _stated(tag_part, "fill_color")
    if stated:
        tag_kwargs["fill_color"] = value
    stated, value = _stated(tag_part, "line_color")
    if stated:
        tag_kwargs["line_color"] = value
    elif common_color is not None:
        tag_kwargs["line_color"] = common_color
    else:
        tag_kwargs["line_color"] = runtime_defaults["dim_color"]
    stated, value = _stated(tag_part, "line_width")
    if stated:
        tag_kwargs["line_width"] = value
    tag = Tag(item.text, (x, y), **tag_kwargs)
    if font_alpha is not None:
        tag.font_alpha = font_alpha
    extend_vertices(self, tag)
    self.active_page.sketches.append(create_sketch(tag, self))

    return self


def grid(
    self: Canvas,
    pos: PointType = (0, 0),
    width: float | None = None,
    height: float | None = None,
    step_size: float | None = None,
    **kwargs: object,
) -> Self:
    """Draw a rectangular grid.

    Args:
        pos: Lower-left corner of the grid.
        width: Length along the x-axis. None uses ``runtime_defaults["grid_size"]``.
        height: Length along the y-axis. None uses ``runtime_defaults["grid_size"]``
            when ``width`` is also None.
        step_size: Distance between grid lines.
        **kwargs: Style overrides for the grid lines.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.grid((0, 0), 20, 20, 10) is canvas
        True
        >>> len(canvas.active_page.sketches) > 0
        True
    """
    x, y = pos[:2]
    if width is None:
        width = runtime_defaults["grid_size"]
        height = runtime_defaults["grid_size"]
    if "line_width" not in kwargs:
        kwargs["line_width"] = runtime_defaults["grid_line_width"]
    if "line_color" not in kwargs:
        kwargs["line_color"] = runtime_defaults["grid_line_color"]
    if "line_dash_array" not in kwargs:
        kwargs["line_dash_array"] = runtime_defaults["grid_line_dash_array"]

    line_y = Shape([(x, y), (x + width, y)], **kwargs)
    line_x = Shape([(x, y), (x, y + height)], **kwargs)
    lines_x = line_y.translate(0, step_size, reps=int(height / step_size))
    lines_y = line_x.translate(step_size, 0, reps=int(width / step_size))
    self.draw(lines_x)
    self.draw(lines_y)
    return self


regular_sketch_types = [
    Types.ARC,
    Types.ARC_ARROW,
    Types.GROUP,
    Types.BEZIER,
    Types.CIRCLE,
    Types.CIRCULAR_GRID,
    Types.DCEL,
    Types.DIVISION,
    Types.DOT,
    Types.DOTS,
    Types.EDGE,
    Types.ELLIPSE,
    Types.FACE,
    Types.FRAGMENT,
    Types.HALF_EDGE,
    Types.HEX_GRID,
    Types.LINE,
    Types.MIXED_GRID,
    Types.OUTLINE,
    Types.OVERLAP,
    Types.PARALLEL_POLYLINE,
    Types.PATH2D,
    Types.PLAIT,
    Types.POLYLINE,
    Types.Q_BEZIER,
    Types.RADIAL_DIMENSION,
    Types.RECTANGLE,
    Types.SECTION,
    Types.SEGMENT,
    Types.SHAPE,
    Types.SINE_WAVE,
    Types.SQUARE,
    Types.SQUARE_GRID,
    Types.STAR,
    Types.TABLE,
    Types.TAG,
    Types.TEXT_PATH,
    Types.VERTEX,
]


def _canvas_space_points(
    canvas: Canvas, points: Sequence[PointType]
) -> list[tuple[float, float]]:
    """Map drawable points into current canvas sketch space."""
    if points is None or len(points) == 0:
        return []
    return [x[:2] for x in homogenize(points) @ canvas._sketch_xform_matrix]


def _extend_canvas_space_points(
    canvas: Canvas, points: Sequence[PointType]
) -> None:
    """Append ``points`` transformed by ``canvas._sketch_xform_matrix``."""
    canvas._all_vertices.extend(_canvas_space_points(canvas, points))


def extend_vertices(canvas: Canvas, item: Drawable | BoundingBox) -> None:
    """Append the item's vertices to the canvas vertex list.

    Args:
        canvas: Canvas whose ``_all_vertices`` list is extended (mutated).
        item: Item whose vertices are copied onto the canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> extend_vertices(canvas, sg.Shape([(0, 0), (40, 0), (0, 40)]))
        >>> len(canvas._all_vertices) > 0
        True
    """
    all_vertices = canvas._all_vertices
    if item.subtype == Types.DOTS:
        vertices = [x.pos for x in item.all_shapes]
        vertices = [
            x[:2] for x in homogenize(vertices) @ canvas._sketch_xform_matrix
        ]
        all_vertices.extend(vertices)
    elif item.subtype == Types.DOT:
        vertices = [item.pos]
        vertices = [
            x[:2] for x in homogenize(vertices) @ canvas._sketch_xform_matrix
        ]
        all_vertices.extend(vertices)
    elif item.subtype == Types.TABLE:
        vertices = [
            x[:2]
            for x in homogenize(item.all_vertices) @ canvas._sketch_xform_matrix
        ]
        all_vertices.extend(vertices)
    elif item.subtype == Types.TAG:
        # Tag objects have all_vertices property that includes text bounding box
        vertices = [
            x[:2]
            for x in homogenize(item.all_vertices) @ canvas._sketch_xform_matrix
        ]
        all_vertices.extend(vertices)
    elif item.subtype == Types.TEXT_PATH:
        vertices = [
            x[:2]
            for x in homogenize(item.all_vertices) @ canvas._sketch_xform_matrix
        ]
        all_vertices.extend(vertices)
    elif item.subtype == Types.ARROW:
        for shape in item.all_shapes:
            all_vertices.extend(_canvas_space_points(canvas, shape.corners))
    elif item.subtype == Types.LACE:
        for plait in item.plaits:
            all_vertices.extend(_canvas_space_points(canvas, plait.corners))
        for fragment in item.fragments:
            all_vertices.extend(_canvas_space_points(canvas, fragment.corners))
    elif item.subtype == Types.PATH2D:
        vertices = [
            x[:2]
            for x in homogenize(item.all_vertices) @ canvas._sketch_xform_matrix
        ]
        all_vertices.extend(vertices)
    elif item.subtype == Types.PATTERN:
        all_vertices.extend(_canvas_space_points(canvas, item.all_vertices))
    elif item.subtype in (Types.ANNOTATION, Types.DCEL, Types.GROUP):
        for element in item:
            extend_vertices(canvas, element)
    elif item.subtype == Types.FIGURE:
        if getattr(item, "draw_geometry", True) and item.geometry is not None:
            extend_vertices(canvas, item.geometry)
        if getattr(item, "draw_skin", True) and item.skin is not None:
            extend_vertices(canvas, item.skin)
    else:
        corners = [
            x[:2]
            for x in homogenize(item.corners) @ canvas._sketch_xform_matrix
        ]
        all_vertices.extend(corners)


def draw(
    self: Canvas, item: Drawable | BoundingBox | Clipping, **kwargs: object
) -> Self:
    """Draw an item on the canvas.

    Args:
        item: Shape, group, or other drawable.
        **kwargs: Style overrides applied while drawing.

    Returns:
        Self: The canvas.

    Raises:
        TypeError: If ``item`` is a ``Lattice``. Draw ``lattice.pattern``
            after ``expand`` or ``populate_unit``.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.draw(sg.Shape([(0, 0), (40, 0), (40, 40)])) is canvas
        True
        >>> len(canvas.active_page.sketches)
        1
        >>> lattice = sg.lattice_p1(40, 40)
        >>> _ = lattice.expand(sg.letter_F(), 1)
        >>> canvas.draw(lattice)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        TypeError: Cannot draw a Lattice. Draw lattice.pattern instead.
        >>> mesh = sg.DCEL()
        >>> _ = mesh.build_from_polygons([[(0, 0), (40, 0), (40, 40), (0, 40)]])
        >>> dcel_canvas = sg.Canvas()
        >>> dcel_canvas.draw(mesh) is dcel_canvas
        True
        >>> len(dcel_canvas.active_page.sketches)
        1
        >>> labeled = sg.Canvas()
        >>> labeled.draw(mesh, face_indices=True) is labeled
        True
        >>> len(labeled.active_page.sketches)
        2
    """
    try:
        draw_list = item.draw_list
    except AttributeError:
        draw_list = None
    if draw_list is not None:
        if callable(draw_list):
            raise TypeError(
                "item.draw_list must be a list of drawables and/or functions."
            )
        draw_widget(self, item, **kwargs)
        return self

    # check if the item has any points
    if not item:
        return self

    if item.type is Types.LATTICE:
        raise TypeError("Cannot draw a Lattice. Draw lattice.pattern instead.")

    active_sketches = self.active_page.sketches
    subtype = item.subtype
    if item.type is not Types.CLIPPING:
        extend_vertices(self, item)

    if subtype in _GEOMETRIC_GRID_TYPES:
        draw_geometric_grid(self, item, **kwargs)
        return self

    if subtype == Types.PATH2D and kwargs.get("handles", False):
        handle_size = runtime_defaults["handle_marker_size"]
        half_size = handle_size / 2
        for handle in item.handles:
            if not handle:
                continue
            _extend_canvas_space_points(self, handle)

            # square handle markers at segment endpoints (3x3)
            for x, y in (handle[0], handle[-1]):
                _extend_canvas_space_points(
                    self,
                    [
                        (x - half_size, y - half_size),
                        (x + half_size, y - half_size),
                        (x + half_size, y + half_size),
                        (x - half_size, y + half_size),
                    ],
                )

    if subtype == Types.ANNOTATION:
        kwargs = _style_annotation(item, kwargs)
    if subtype in (Types.GROUP, Types.STAR, Types.ANNOTATION, Types.DCEL):
        group_kwargs = dict(kwargs)
        if kwargs.get("vertex_on_hull") and "_group_hull_points" not in kwargs:
            group_kwargs["_group_hull_points"] = convex_hull(
                item.all_vertices, on_edge=True
            )
        index_requested = False
        face_index_requested = False
        if subtype == Types.DCEL:
            if "indices" in group_kwargs:
                index_requested = bool(group_kwargs["indices"])
                del group_kwargs["indices"]
            if "face_indices" in group_kwargs:
                face_index_requested = bool(group_kwargs["face_indices"])
                del group_kwargs["face_indices"]
        if subtype == Types.GROUP and group_kwargs.get("non_zero"):
            compound_path = group_to_nonzero_path(item)
            active_sketches.extend(
                get_sketches(compound_path, self, **group_kwargs)
            )
        else:
            for group_item in item:
                if subtype == Types.DCEL and group_item is item.outer_face:
                    continue
                draw(self, group_item, **group_kwargs)
            if subtype == Types.DCEL:
                for edge in item.edges:
                    if edge.half_edge is None:
                        continue
                    left, right = edge.adjacent_faces()
                    if left is not right:
                        continue
                    draw(self, edge, **group_kwargs)
                if index_requested:
                    points = [vertex.point for vertex in item.vertices]
                    index_style: dict[str, object] = {
                        "fill": False,
                        "stroke": False,
                        "indices": True,
                    }
                    for key in kwargs:
                        if key == "indices" or key.startswith("index_"):
                            index_style[key] = kwargs[key]
                    if points:
                        draw(self, Shape(points), **index_style)
                if face_index_requested:
                    tag_kwargs: dict[str, object] = {}
                    if "index_font_size" in kwargs:
                        tag_kwargs["font_size"] = kwargs["index_font_size"]
                    if "index_font_color" in kwargs:
                        tag_kwargs["font_color"] = kwargs["index_font_color"]
                    if "index_font_family" in kwargs:
                        tag_kwargs["font_family"] = kwargs["index_font_family"]
                    for index, face in enumerate(item.faces):
                        if face is item.outer_face:
                            continue
                        tag_obj = Tag(str(index), face.midpoint, **tag_kwargs)
                        tag_obj.draw_frame = False
                        draw(self, tag_obj)
    elif subtype == Types.FACE and item.inner_boundaries:
        # Holes belong to the face; same path as canvas.draw(group, non_zero=True).
        if item.mesh is None:
            raise ValueError("face with holes is not linked to a DCEL")
        outer_points = [vertex.point for vertex in item.boundary_vertices()]
        rings = [Shape(outer_points, closed=True)]
        outer_area = polygon_area(outer_points)
        for cycle in item.mesh.holes(item):
            points = [vertex.point for vertex in cycle]
            # group_to_nonzero_path reverses non-outer rings; feed matching
            # winding (DCEL hole cycles are opposite the outer).
            if polygon_area(points) * outer_area < 0:
                points = list(reversed(points))
            rings.append(Shape(points, closed=True))
        hole_kwargs = dict(kwargs)
        hole_kwargs["non_zero"] = True
        draw(self, Group(rings), **hole_kwargs)
    elif subtype in regular_sketch_types:
        sketches = get_sketches(item, self, **kwargs)
        if sketches:
            active_sketches.extend(sketches)
    elif subtype == Types.IMAGE:
        draw_image(self, item, **kwargs)
    elif subtype == Types.PATTERN:
        draw_pattern(self, item, **kwargs)
    elif subtype in (
        Types.ALIGNED_DIMENSION,
        Types.ANGULAR_DIMENSION,
        Types.DIMENSION,
    ):
        draw_dimension(self, item, **kwargs)
    elif subtype == Types.ARROW:
        active_sketches.append(create_sketch(item.line, self, **kwargs))
        for head in item.heads:
            active_sketches.append(create_sketch(head, self, **kwargs))
    elif subtype == Types.LACE:
        draw_lace(self, item, **kwargs)
    elif subtype == Types.FIGURE:
        if getattr(item, "draw_geometry", True) and item.geometry is not None:
            draw(self, item.geometry, **kwargs)
        if getattr(item, "draw_skin", True) and item.skin is not None:
            draw(self, item.skin)
    elif subtype == Types.BOUNDING_BOX:
        draw_bbox(self, item, **kwargs)
    elif subtype == Types.CLIPPING:
        target = item.target
        clipper = item.clipper
        self._sketch_xform_matrix = (
            self._sketch_xform_matrix @ self._xform_matrix
        )
        self.active_page.sketches.append(
            get_clipped_sketch(target, clipper, self, **kwargs)
        )
        extend_vertices(self, clipper)
        self._sketch_xform_matrix = identity_matrix()

        return self
    return self


def draw_all_segments(
    self: Canvas,
    item: Shape | Group,
    vert_indices: bool = False,
    **kwargs: object,
) -> Self:
    """Split edges into segments and draw them with index labels.

    Used with loop-finding workflows. When ``vert_indices`` is true,
    vertex indices are drawn at endpoints instead of edge indices at
    midpoints.

    Args:
        item: Shape or group whose edges are split and drawn.
        vert_indices: Label vertices instead of edges.
        **kwargs: Style overrides forwarded to ``draw`` and ``text``.

    Returns:
        Self: The canvas.

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> shape = sg.Shape([(0, 0), (40, 0), (40, 40), (0, 40)])
        >>> canvas.draw_all_segments(shape) is canvas
        True
    """
    segments = all_segments(item)
    count = 0
    for i, edge in enumerate(segments):
        draw(self, Shape(edge), **kwargs)
        if vert_indices:
            p1, p2 = edge
            text(self, f"{count}", p1, **kwargs)
            count += 1
            text(self, f"{count}", p2, **kwargs)
            count += 1
        else:
            text(self, f"{i}", midpoint(*edge), **kwargs)

    return self


def get_clipped_sketch(
    target: Drawable,
    clipper: Drawable,
    canvas: Canvas,
    **kwargs: object,
) -> ClippedSketch:
    """Build a ``ClippedSketch`` for a target clipped by ``clipper``.

    Args:
        target: Drawable content to clip (shape or group).
        clipper: Drawable used as the clipping path.
        canvas: Canvas providing transform context.
        **kwargs: Style overrides for target sketches.

    Returns:
        ClippedSketch: Composite sketch with clipper attached.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.draw import get_clipped_sketch
        >>> canvas = sg.Canvas()
        >>> target = sg.Shape([(0, 0), (40, 0), (40, 40)])
        >>> clipper = sg.Shape([(0, 0), (20, 0), (20, 20)])
        >>> clipped = get_clipped_sketch(target, clipper, canvas)
        >>> clipped.subtype.name
        'CLIPPED_SKETCH'
    """
    if target.type == Types.GROUP:
        sketches = [get_sketches(item, canvas, **kwargs) for item in target]
    else:
        sketches = [get_sketches(target, canvas, **kwargs)]
    clipper = get_sketches(clipper, canvas)
    return ClippedSketch(sketches=sketches, clipper=clipper)


def get_sketches(
    item: Drawable,
    canvas: Canvas | None = None,
    **kwargs: object,
) -> list[Sketch]:
    """Create sketches from the given item and return them as a list.

    Args:
        item: Item to be sketched.
        canvas: Canvas object. Defaults to None.
        **kwargs: Style overrides forwarded to ``create_sketch``.

    Returns:
        list[Sketch]: Sketches for the item, or an empty list when hidden.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.draw import get_sketches
        >>> canvas = sg.Canvas()
        >>> shape = sg.Shape([(0, 0), (40, 0), (40, 40)])
        >>> len(get_sketches(shape, canvas))
        1
    """
    if not (item.visible):
        res = []
    elif item.subtype in drawable_types:
        sketches = create_sketch(item, canvas, **kwargs)
        if isinstance(sketches, list):
            res = sketches
        elif sketches is not None:
            res = [sketches]
        else:
            res = []
    else:
        res = []
    return res


_PRESEDENCE_KEYS = frozenset(
    (
        "color",
        "line_color",
        "fill_color",
        "alpha",
        "line_alpha",
        "fill_alpha",
    )
)

_NON_STYLE_KEYS = frozenset(
    (
        "_mask_context_id",
        "_style_id",
        "_tikz_style_id",
        "vertices",
        "index_font_size",
        "vertex_font_size",
        "index_offset",
        "vertex_offset",
        "index_font_color",
        "vertex_font_color",
        "index_font_family",
        "vertex_font_family",
        "debug",
        "vertex_on_hull",
        "_group_hull_points",
        "non_zero",
    )
)


def set_shape_sketch_style(
    sketch: Sketch,
    item: Drawable,
    canvas: Canvas,
    linear: bool = False,
    **kwargs: object,
) -> None:
    """Set the style properties of the sketch.

    Args:
        sketch: Sketch object (mutated).
        item: Item whose style properties are to be set.
        canvas: Canvas object.
        linear (bool, optional): Whether the style is linear. Defaults to False.
        **kwargs: Additional keyword arguments.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.draw import create_sketch, set_shape_sketch_style
        >>> canvas = sg.Canvas()
        >>> shape = sg.Shape([(0, 0), (40, 0), (40, 40)])
        >>> sketch = create_sketch(shape, canvas)
        >>> set_shape_sketch_style(sketch, shape, canvas)
        >>> sketch.visible
        True
    """
    if linear:
        style_map = line_style_map
    else:
        style_map = shape_style_map

    resolved_style = canvas.resolve_style_properties(item, style_map, **kwargs)

    for attrib_name, attrib_value in resolved_style.items():
        setattr(sketch, attrib_name, attrib_value)

    sketch.visible = item.visible
    sketch.closed = item.closed
    # fill and stroke are resolved through resolve_property in the loop above

    # Copy tile_svg (direct shape property, not style attribute)
    if hasattr(item, "tile_svg"):
        sketch.tile_svg = item.tile_svg

    # Copy clip and mask for clipping support
    if hasattr(item, "clip"):
        sketch.clip = item.clip
    if hasattr(item, "mask"):
        sketch.mask = item.mask
    if "_mask_opacity" in item.__dict__:
        sketch._mask_opacity = item._mask_opacity
    if "_mask_stops" in item.__dict__:
        sketch._mask_stops = item._mask_stops
    if "_mask_axis" in item.__dict__:
        sketch._mask_axis = item._mask_axis

    if hasattr(item, "even_odd") and item.even_odd is not None:
        sketch.even_odd = item.even_odd
    elif "non_zero" in kwargs and kwargs["non_zero"] is not None:
        use_nonzero = bool(kwargs["non_zero"])
        sketch.even_odd = not use_nonzero
        sketch.fill_mode = FillMode.NONZERO if use_nonzero else FillMode.EVENODD

    for k, v in kwargs.items():
        if k in _PRESEDENCE_KEYS or k in _NON_STYLE_KEYS:
            continue

        setattr(sketch, k, v)

    if sketch.draw_markers:
        for name in (
            "marker_alpha",
            "marker_color",
            "marker_line_style",
            "marker_line_width",
            "marker_radius",
            "marker_size",
            "marker_type",
        ):
            if name not in sketch.__dict__ or sketch.__dict__[name] is None:
                setattr(sketch, name, runtime_defaults[name])

    if "_group_hull_points" in kwargs:
        sketch._group_hull_points = kwargs["_group_hull_points"]

    if "indices" in kwargs:
        sketch.indices = kwargs["indices"]
    elif "indices" in item.__dict__:
        sketch.indices = item.indices

    if "index_font_size" in kwargs:
        sketch.index_font_size = kwargs["index_font_size"]
    elif "index_font_size" in item.__dict__:
        sketch.index_font_size = item.index_font_size

    if (
        kwargs.get("vertices")
        or kwargs.get("vertex_on_hull")
        or getattr(item, "vertex_on_hull", False)
    ):
        sketch.show_vertex_coords = True
    elif "show_vertex_coords" in item.__dict__:
        sketch.show_vertex_coords = item.show_vertex_coords

    if "vertex_on_hull" in kwargs:
        sketch.vertex_on_hull = kwargs["vertex_on_hull"]
    elif "vertex_on_hull" in item.__dict__:
        sketch.vertex_on_hull = item.vertex_on_hull

    if "vertex_font_size" in kwargs:
        sketch.vertex_font_size = kwargs["vertex_font_size"]
    elif "vertex_font_size" in item.__dict__:
        sketch.vertex_font_size = item.vertex_font_size

    if "index_offset" in kwargs:
        sketch.index_offset = kwargs["index_offset"]
    elif "index_offset" in item.__dict__:
        sketch.index_offset = item.index_offset

    if "vertex_offset" in kwargs:
        sketch.vertex_offset = kwargs["vertex_offset"]
    elif "vertex_offset" in item.__dict__:
        sketch.vertex_offset = item.vertex_offset

    if "index_font_color" in kwargs:
        sketch.index_font_color = kwargs["index_font_color"]
    elif "index_font_color" in item.__dict__:
        sketch.index_font_color = item.index_font_color

    if "vertex_font_color" in kwargs:
        sketch.vertex_font_color = kwargs["vertex_font_color"]
    elif "vertex_font_color" in item.__dict__:
        sketch.vertex_font_color = item.vertex_font_color

    if "index_font_family" in kwargs:
        sketch.index_font_family = kwargs["index_font_family"]
    elif "index_font_family" in item.__dict__:
        sketch.index_font_family = item.index_font_family

    if "vertex_font_family" in kwargs:
        sketch.vertex_font_family = kwargs["vertex_font_family"]
    elif "vertex_font_family" in item.__dict__:
        sketch.vertex_font_family = item.vertex_font_family

    if "debug" in kwargs:
        sketch.debug = kwargs["debug"]

    if (
        sketch.marker_type == MarkerType.INDICES
        and "indices" not in kwargs
        and "indices" not in item.__dict__
    ):
        sketch.indices = True


def get_verts_in_new_pos(item: Shape, **kwargs: object) -> list[PointType]:
    """Return the item's vertices, translated when ``pos`` is given.

    Args:
        item: Item whose vertices are read.
        **kwargs: ``pos`` is the new midpoint. Other keys are ignored
            by this function.

    Returns:
        list: Vertices in the new position, or the item's current
        vertices when ``pos`` is not given.

    Examples:
        >>> import simetri.graphics as sg
        >>> shape = sg.Shape([(0, 0), (80, 0), (80, 80)])
        >>> get_verts_in_new_pos(shape, pos=(60, 40))[0]
        [20.0, 0.0]
    """
    if "pos" in kwargs:
        x, y = item.midpoint[:2]
        x1, y1 = kwargs["pos"][:2]
        dx = x1 - x
        dy = y1 - y
        trans_mat = translation_matrix(dx, dy)
        vertices = item.primary_points.homogen_coords @ trans_mat
        vertices = vertices[:, :2].tolist()
    else:
        vertices = item.vertices

    return vertices


##################################


def _get_tag_sketch(
    item: Drawable, canvas: Canvas, **kwargs: object
) -> TagSketch:
    """Create a TagSketch from the given item.

    Args:
        item: Item to be sketched.
        canvas: Canvas object.
        **kwargs: Additional keyword arguments.

    Returns:
        TagSketch: Created TagSketch.
    """
    pos = kwargs.get("pos", item.pos)
    _, rotation, _ = decompose_transformations(item.xform_matrix)

    sketch = TagSketch(
        text=item.text,
        pos=pos,
        anchor=item.anchor,
        angle=rotation,
        xform_matrix=canvas._sketch_xform_matrix,
    )
    for attrib_name in tag_style_map:
        if attrib_name == "color":
            continue
        if attrib_name == "fill_color":
            fill_color = canvas.resolve_property(item, "fill_color")
            if fill_color == colors.black:
                sketch.frame_back_color = runtime_defaults["frame_back_color"]
            else:
                sketch.frame_back_color = fill_color
            continue
        attrib_value = canvas.resolve_property(item, attrib_name)
        setattr(sketch, attrib_name, attrib_value)
    sketch.text_width = item.text_width
    sketch.visible = item.visible

    for k, v in kwargs.items():
        if k in _PRECEDENCE_KEYS:
            continue
        setattr(sketch, k, v)
    _, rotation, scale = decompose_transformations(canvas._sketch_xform_matrix)
    if rotation != 0:
        sketch.angle = float(sketch.angle) + float(rotation)
    scale_x = float(scale[0])
    if scale_x != 1.0 and isinstance(sketch.font_size, (int, float)):
        sketch.font_size = float(sketch.font_size) * scale_x
    return sketch


def _get_text_path_sketch(
    item: Drawable, canvas: Canvas, **kwargs: object
) -> TextPathSketch:
    """Create a TextPathSketch from the given item."""
    transformed_path = item.path.copy()
    transformed_path._update(item.xform_matrix)
    transformed_path._update(canvas._sketch_xform_matrix)
    sketch = TextPathSketch(
        text=item.text,
        path_data=path2d_to_svg_path(transformed_path),
        draw_path=item.draw_path,
        vertices=list(transformed_path.all_vertices),
        xform_matrix=canvas._sketch_xform_matrix,
    )
    sketch.visible = item.visible
    sketch.font_family = canvas.resolve_property(item, "font_family")
    sketch.font_size = canvas.resolve_property(item, "font_size")
    sketch.font_color = canvas.resolve_property(item, "font_color")
    sketch.bold = item.bold
    sketch.italic = item.italic
    if item.draw_path:
        set_shape_sketch_style(sketch, item, canvas, **kwargs)
        sketch.stroke = True
        sketch.fill = False
    for key, value in kwargs.items():
        if key in _PRECEDENCE_KEYS:
            continue
        setattr(sketch, key, value)
    return sketch


def _get_ellipse_sketch(
    item: Drawable, canvas: Canvas, **kwargs: object
) -> EllipseSketch:
    """Create an EllipseSketch from the given item.

    Args:
        item: Item to be sketched.
        canvas: Canvas object.
        **kwargs: Additional keyword arguments.

    Returns:
        EllipseSketch: Created EllipseSketch.
    """
    sketch = EllipseSketch(
        item.center,
        item.a,
        item.b,
        item.angle,
        xform_matrix=canvas._sketch_xform_matrix,
        **kwargs,
    )
    set_shape_sketch_style(sketch, item, canvas, **kwargs)

    return sketch


def _get_pattern_sketch(
    item: Drawable, canvas: Canvas, **kwargs: object
) -> PatternSketch:
    """Create a PatternSketch from the given item.

    Args:
        item: Item to be sketched.
        canvas: Canvas object.
        **kwargs: Additional keyword arguments.

    Returns:
        PatternSketch: Created PatternSketch.
    """
    sketch = PatternSketch(item, xform_matrix=canvas._sketch_xform_matrix)
    set_shape_sketch_style(sketch, item, canvas, **kwargs)

    return sketch


def _get_circle_sketch(
    item: Drawable, canvas: Canvas, **kwargs: object
) -> CircleSketch:
    """Create a CircleSketch from the given item.

    Args:
        item: Item to be sketched.
        canvas: Canvas object.
        **kwargs: Additional keyword arguments.

    Returns:
        CircleSketch: Created CircleSketch.
    """
    sketch = _circle_sketch(
        item.center,
        item.radius,
        canvas._sketch_xform_matrix,
    )
    set_shape_sketch_style(sketch, item, canvas, **kwargs)

    return sketch


def _get_dots_sketch(
    item: Drawable, canvas: Canvas, **kwargs: object
) -> list[Sketch]:
    """Create sketches for dots from the given item.

    Args:
        item: Item to be sketched.
        canvas: Canvas object.
        **kwargs: Additional keyword arguments.

    Returns:
        list: List of created sketches.
    """
    vertices = [x.pos for x in item.all_shapes]
    fill_color = item[0].fill_color
    radius = item[0].radius
    marker_size = item[0].marker_size
    marker_type = item[0].marker_type
    item = Shape(
        vertices,
        fill_color=fill_color,
        markers_only=True,
        draw_markers=True,
        marker_size=marker_size,
        marker_radius=radius,
        marker_type=marker_type,
    )
    sketches = get_sketches(item, canvas, **kwargs)

    return sketches


def _get_arc_sketch(
    item: Drawable, canvas: Canvas, **kwargs: object
) -> ArcSketch:
    """Create an ArcSketch from the given item.

    Args:
        item: Item to be sketched.
        canvas: Canvas object.
        **kwargs: Additional keyword arguments.

    Returns:
        ArcSketch: Created ArcSketch.
    """

    # vertices = get_verts_in_new_pos(item, **kwargs)
    sketch = ArcSketch(item.vertices, xform_matrix=canvas._sketch_xform_matrix)
    set_shape_sketch_style(sketch, item, canvas, **kwargs)

    return sketch


def _get_lace_sketch(
    item: Drawable, canvas: Canvas, **kwargs: object
) -> list[Sketch]:
    """Create sketches for lace from the given item.

    Args:
        item: Item to be sketched.
        canvas: Canvas object.
        **kwargs: Additional keyword arguments.

    Returns:
        list: List of created sketches.
    """
    sketches = [get_sketch(frag, canvas, **kwargs) for frag in item.fragments]
    sketches.extend(
        [get_sketch(plait, canvas, **kwargs) for plait in item.plaits]
    )
    return sketches


def _get_composite_sketch(
    items: Sequence[Drawable],
    canvas: Canvas,
    **kwargs: object,
) -> list[Sketch | None]:
    """Create sketches for composite drawables (arrows, grids, and similar).

    Args:
        items: Components to sketch in order.
        canvas: Canvas providing transform context.
        **kwargs: Style overrides forwarded to ``create_sketch``.

    Returns:
        list: One sketch per component (entries may be ``None``).
    """
    sketches = []
    for component in items:
        sketch = create_sketch(component, canvas, **kwargs)
        sketches.append(sketch)
    return sketches


def _sync_path_sketch_fill_rule(
    path_sketch: PathSketch,
    item: Drawable,
    **kwargs: object,
) -> None:
    """Set ``even_odd`` on a path sketch from ``even_odd``, ``non_zero``, or ``fill_mode``."""
    if hasattr(item, "even_odd") and item.even_odd is not None:
        path_sketch.even_odd = item.even_odd
        return
    if "non_zero" in kwargs and kwargs["non_zero"] is not None:
        use_nonzero = bool(kwargs["non_zero"])
        path_sketch.even_odd = not use_nonzero
        path_sketch.fill_mode = (
            FillMode.NONZERO if use_nonzero else FillMode.EVENODD
        )
        return
    fill_mode = getattr(path_sketch, "fill_mode", None)
    if fill_mode is None:
        return
    mode = get_enum_value(FillMode, fill_mode)
    path_sketch.even_odd = mode == FillMode.EVENODD.value


def _get_path_sketch(
    item: Drawable, canvas: Canvas, **kwargs: object
) -> PathSketch | list[Sketch]:
    """Create sketches for a path from the given item.

    Args:
        item: Item to be sketched.
        canvas: Canvas object.
        **kwargs: Additional keyword arguments.

    Returns:
        list: List of created sketches.
    """
    transformed_path = item.copy()
    transformed_path._update(canvas._sketch_xform_matrix)

    path_sketch = PathSketch([], canvas._sketch_xform_matrix)
    path_sketch.path_data = path2d_to_svg_path(transformed_path)
    path_sketch.visible = item.visible
    path_sketch.closed = item.closed
    path_sketch.vertices = list(transformed_path._label_vertices())
    set_shape_sketch_style(path_sketch, item, canvas, **kwargs)
    _sync_path_sketch_fill_rule(path_sketch, item, **kwargs)

    handle_sketches = []
    if kwargs.get("handles"):
        del kwargs["handles"]
        for handle in item.handles:
            shape = Shape(handle)
            shape.subtype = Types.HANDLE
            sketches = create_sketch(shape, canvas, **kwargs)
            handle_sketches.extend(sketches)

    if handle_sketches:
        return [path_sketch, *handle_sketches]

    return path_sketch


def _get_bbox_sketch(
    item: BoundingBox, canvas: Canvas, **kwargs: object
) -> ShapeSketch | None:
    """Create a bounding box sketch from the given item.

    Args:
        item: Item to be sketched.
        canvas: Canvas object.
        **kwargs: Additional keyword arguments.

    Returns:
        ShapeSketch: Created bounding box sketch.
    """
    nround = runtime_defaults["tikz_nround"]
    vertices = [
        (round(x[0], nround), round(x[1], nround)) for x in item.corners
    ]
    if not vertices:
        return None
    sketch = ShapeSketch(vertices, canvas._sketch_xform_matrix)
    sketch.subtype = Types.BBOX_SKETCH
    sketch.exclusive = item.exclusive
    sketch.visible = True
    sketch.closed = True
    bbox_style_names = (
        "fill",
        "stroke",
        "line_color",
        "line_width",
        "line_dash_array",
        "draw_markers",
    )
    for name in bbox_style_names:
        if name in kwargs:
            setattr(sketch, name, kwargs[name])
        else:
            setattr(sketch, name, runtime_defaults[f"bbox_{name}"])
    for name in shape_style_map:
        if name in kwargs and name not in bbox_style_names:
            setattr(sketch, name, kwargs[name])
    return sketch


def _get_handle_sketch(
    item: Drawable, canvas: Canvas, **kwargs: object
) -> list[Sketch] | None:
    """Create handle sketches from the given item.

    Args:
        item: Item to be sketched.
        canvas: Canvas object.
        **kwargs: Additional keyword arguments.

    Returns:
        list: List of created handle sketches.
    """
    nround = runtime_defaults["tikz_nround"]
    vertices = [
        (round(x[0], nround), round(x[1], nround)) for x in item.vertices
    ]
    if not vertices:
        return None
    if "pos" in kwargs:
        x, y = item.midpoint[:2]
        x1, y1 = kwargs["pos"][:2]
        dx = x1 - x
        dy = y1 - y
        vertices = [(x + dx, y + dy) for x, y in vertices]
    sketches = []
    sketch = ShapeSketch(vertices, canvas._sketch_xform_matrix)
    sketch.subtype = Types.HANDLE
    sketch.closed = False
    set_shape_sketch_style(sketch, item, canvas, **kwargs)
    sketches.append(sketch)
    temp_item = Shape()
    temp_item.closed = True
    handle_size = runtime_defaults["handle_marker_size"]
    handle1 = RectSketch(
        item.vertices[0],
        handle_size,
        handle_size,
        canvas._sketch_xform_matrix,
    )
    set_shape_sketch_style(handle1, temp_item, canvas, **kwargs)
    handle2 = RectSketch(
        item.vertices[-1],
        handle_size,
        handle_size,
        canvas._sketch_xform_matrix,
    )
    set_shape_sketch_style(handle2, temp_item, canvas, **kwargs)
    sketches.extend([handle1, handle2])

    return sketches


def _get_sketch(
    item: Drawable, canvas: Canvas, **kwargs: object
) -> ShapeSketch | None:
    """Create a sketch from the given item.

    Args:
        item: Item to be sketched.
        canvas: Canvas object.
        **kwargs: Additional keyword arguments.

    Returns:
        ShapeSketch: Created sketch.
    """
    if not item.vertices:
        return None

    nround = runtime_defaults["tikz_nround"]
    vertices = [
        (round(x[0], nround), round(x[1], nround)) for x in item.vertices
    ]

    sketch = ShapeSketch(vertices, canvas._sketch_xform_matrix)
    set_shape_sketch_style(sketch, item, canvas, **kwargs)

    return sketch


def _get_line_sketch(
    item: Drawable, canvas: Canvas, **kwargs: object
) -> LineSketch | None:
    """Create a line sketch from the given item.

    Args:
        item: Line drawable with ``vertices``.
        canvas: Canvas providing transform context.
        **kwargs: Style overrides forwarded to ``set_shape_sketch_style``.

    Returns:
        LineSketch or ``None`` when there are no vertices.
    """
    if not item.vertices:
        return None

    nround = runtime_defaults["tikz_nround"]
    vertices = [
        (round(x[0], nround), round(x[1], nround)) for x in item.vertices
    ]
    sketch = LineSketch(vertices, canvas._sketch_xform_matrix)
    set_shape_sketch_style(sketch, item, canvas, **kwargs)
    sketch.extent = item.extent

    return sketch


def _get_image_sketch(
    item: Drawable, canvas: Canvas, **kwargs: object
) -> ImageSketch:
    """Create an ImageSketch from the given item.

    Args:
        item: Item to be sketched.
        canvas: Canvas object.
        **kwargs: Additional keyword arguments.

    Returns:
        ImageSketch: Created ImageSketch.
    """
    _, rotation, _ = decompose_transformations(item.xform_matrix)
    sketch = ImageSketch(
        item,
        pos=item.pos,
        angle=rotation,
        scale=(1, 1),
        anchor=item.anchor,
        size=item.size,
        file_path=item.file_path,
        xform_matrix=canvas._sketch_xform_matrix,
        **kwargs,
    )
    set_shape_sketch_style(sketch, item, canvas, **kwargs)

    return sketch


def _get_table_sketch(
    item: Drawable, canvas: Canvas, **kwargs: object
) -> Sketch:
    """Create a table sketch from a ``Table`` drawable.

    Args:
        item: Table drawable.
        canvas: Canvas providing transform context.
        **kwargs: Style overrides forwarded to ``build_table_sketch``.

    Returns:
        Sketch: Table sketch for rendering.
    """
    from ..extensions.table import build_table_sketch

    return build_table_sketch(item, canvas, **kwargs)


_d_subtype_sketch = {
    Types.ANNOTATION: _get_composite_sketch,
    Types.ARC: _get_arc_sketch,
    Types.ARC_ARROW: _get_composite_sketch,
    Types.ARROW: _get_composite_sketch,
    Types.ARROW_HEAD: _get_sketch,
    Types.GROUP: _get_composite_sketch,
    Types.BEZIER: _get_sketch,
    Types.BOUNDING_BOX: _get_bbox_sketch,
    Types.CIRCLE: _get_circle_sketch,
    Types.CIRCULAR_GRID: _get_composite_sketch,
    Types.DCEL: _get_composite_sketch,
    Types.DIVISION: _get_sketch,
    Types.DOT: _get_circle_sketch,
    Types.DOTS: _get_dots_sketch,
    Types.EDGE: _get_sketch,
    Types.ELLIPSE: _get_sketch,
    Types.FACE: _get_sketch,
    Types.FRAGMENT: _get_sketch,
    Types.HALF_EDGE: _get_sketch,
    Types.HANDLE: _get_handle_sketch,
    Types.HEX_GRID: _get_composite_sketch,
    Types.IMAGE: _get_image_sketch,
    Types.LACE: _get_lace_sketch,
    Types.LINE: _get_line_sketch,
    Types.MIXED_GRID: _get_composite_sketch,
    Types.MASK: _get_sketch,
    Types.OVERLAP: _get_composite_sketch,
    Types.PARALLEL_POLYLINE: _get_composite_sketch,
    Types.PATH2D: _get_path_sketch,
    Types.PATTERN: _get_pattern_sketch,
    Types.PLAIT: _get_sketch,
    Types.POLYLINE: _get_sketch,
    Types.RADIAL_DIMENSION: _get_composite_sketch,
    Types.Q_BEZIER: _get_sketch,
    Types.RECTANGLE: _get_sketch,
    Types.SECTION: _get_sketch,
    Types.SEGMENT: _get_sketch,
    Types.SHAPE: _get_sketch,
    Types.SINE_WAVE: _get_sketch,
    Types.SQUARE: _get_sketch,
    Types.SQUARE_GRID: _get_composite_sketch,
    Types.STAR: _get_composite_sketch,
    Types.TABLE: _get_table_sketch,
    Types.TAG: _get_tag_sketch,
    Types.TEXT_PATH: _get_text_path_sketch,
    Types.VERTEX: _get_sketch,
}


def create_sketch(
    item: Drawable,
    canvas: Canvas,
    **kwargs: object,
) -> Sketch | list[Sketch | None] | None:
    """Create a sketch from the given item.

    Args:
        item: Item to be sketched.
        canvas: Canvas object.
        **kwargs: Style overrides forwarded to subtype sketch builders.

    Returns:
        A single sketch, a list of sketches, or ``None`` when invisible.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.draw import create_sketch
        >>> canvas = sg.Canvas()
        >>> shape = sg.Shape([(0, 0), (40, 0), (40, 40)])
        >>> create_sketch(shape, canvas) is not None
        True
    """
    if not (item.visible):
        return None

    return _d_subtype_sketch[item.subtype](item, canvas, **kwargs)

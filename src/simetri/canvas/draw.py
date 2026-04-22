"""Canvas object uses these methods to draw shapes and text."""

from math import cos, sin, pi, radians
from typing_extensions import Self, Sequence, Union

from ..geometry.geometry import (
    homogenize,
    close_points2,
    midpoint,
    intersect,
    inclination_angle,
    intersection,
    offset_polygon,
)
from ..graphics.all_enums import (
    Anchor,
    BackStyle,
    Drawable,
    FrameShape,
    PathOperation,
    Types,
    drawable_types,
    TexLoc,
    PlaitStyle,
    FragmentStyle,
    Connection,
)
from ..colors import colors
from ..colors.palettes import d_n_palette, d_name_palette
from ..colors.colors import Color, change_lightness, get_lightest
from ..graphics.sketch import (
    ArcSketch,
    BatchSketch,
    BezierSketch,
    CircleSketch,
    EllipseSketch,
    GroupSketch,
    HelpLinesSketch,
    LineSketch,
    PatternSketch,
    RectSketch,
    ShapeSketch,
    TagSketch,
    ImageSketch,
    PDFSketch,
    LatexSketch,
)
from ..tikz.tikz_sketch import TexSketch
from ..graphics.bbox import bounding_box
from ..settings.settings import defaults
from ..canvas.style_map import (
    line_style_map,
    shape_style_map,
    tag_style_map,
    image_style_map,
    group_args,
    StyleObj,
)
from ..helpers.illustration import Tag
from ..helpers.utilities import (
    decompose_transformations,
    group_into_bins,
    round_symmetric,
)
from ..graphics.affine import identity_matrix
from ..graphics.shape import Shape, all_segments
from ..graphics.shapes import fillet_shape_corners
from ..graphics.batch import Batch
from ..graphics.common import PointType, d_id_obj
from ..geometry.bezier import bezier_points
from ..geometry.ellipse import elliptic_arc_points
from ..graphics.affine import rotation_matrix, translation_matrix


def help_lines(
    self,
    pos: PointType = None,
    width: float = None,
    height: float = None,
    spacing=None,
    cs_size: float = None,
    deferred: bool = False,
    **kwargs,
):
    """
    Draw a square grid with the given size.

    Args:
        pos (PointType, optional): Position of the grid. Defaults to None.
        width (float, optional): Length of the grid along the x-axis. Defaults to None.
        height (float, optional): Length of the grid along the y-axis. Defaults to None.
        spacing (optional): Step size for the grid. Defaults to None.
        cs_size (float, optional): Size of the coordinate system. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Self: The canvas object.
    """
    if deferred:
        sketch = HelpLinesSketch(spacing, cs_size)
        sketch.kwargs = kwargs
        self.active_page.sketches.append(sketch)
    else:
        self.grid(pos, width, height, spacing, **kwargs)
        if cs_size > 0:
            self.draw_CS(cs_size, **kwargs)
    return self


def arc(
    self,
    center: PointType,
    radius_x: float,
    radius_y: float,
    start_angle: float,
    span_angle: float,
    rot_angle: float,
    n_points: int = None,
    **kwargs,
) -> None:
    """
    Draw an arc with the given center, radius, start and end angles in radians.
    Arc is drawn in counterclockwise direction from start to end.

    Args:
        center (PointType): Center of the arc.
        radius_x (float): Radius of the arc.
        radius_y (float): Second radius of the arc.
        start_angle (float): Start angle of the arc in radians.
        end_angle (float): End angle of the arc in radians.

        rot_angle (float): Rotation angle of the arc.
        **kwargs: Additional keyword arguments.
    """
    if radius_y is None:
        radius_y = radius_x
    vertices = elliptic_arc_points(
        center, radius_x, radius_y, start_angle, span_angle, n_points
    )
    if rot_angle != 0:
        vertices = homogenize(vertices) @ rotation_matrix(rot_angle, center)
    self._all_vertices.extend(vertices.tolist() + [center])

    sketch = ArcSketch(vertices=vertices, xform_matrix=self.xform_matrix)
    resolved = self.resolve_style_properties(sketch, shape_style_map, **kwargs)
    for attrib_name, attrib_value in resolved.items():
        setattr(sketch, attrib_name, attrib_value)
    precedence_keys = {
        "color",
        "line_color",
        "fill_color",
        "alpha",
        "line_alpha",
        "fill_alpha",
    }
    for k, v in kwargs.items():
        if k not in precedence_keys:
            setattr(sketch, k, v)
    self.active_page.sketches.append(sketch)

    return self


def bezier(self, control_points, **kwargs):
    """
    Draw a Bezier curve with the given control points.

    Args:
        control_points: Control points for the Bezier curve.
        **kwargs: Additional keyword arguments.

    Returns:
        Self: The canvas object.
    """
    self._all_vertices.extend(control_points)
    sketch = BezierSketch(control_points, self.xform_matrix)
    resolved = self.resolve_style_properties(sketch, shape_style_map, **kwargs)
    for attrib_name, attrib_value in resolved.items():
        setattr(sketch, attrib_name, attrib_value)
    precedence_keys = {
        "color",
        "line_color",
        "fill_color",
        "alpha",
        "line_alpha",
        "fill_alpha",
    }
    for k, v in kwargs.items():
        if k not in precedence_keys:
            setattr(sketch, k, v)
    self.active_page.sketches.append(sketch)
    return self


def circle(self, radius: float, center: PointType, **kwargs) -> None:
    """
    Draw a circle with the given center and radius.

    Args:
        radius (float): Radius of the circle.
        center (PointType): Center of the circle.
        **kwargs: Additional keyword arguments.
    """
    x, y = center[:2]
    p1 = x - radius, y - radius
    p2 = x + radius, y + radius
    p3 = x - radius, y + radius
    p4 = x + radius, y - radius
    self._all_vertices.extend([p1, p2, p3, p4])
    sketch = CircleSketch(center, radius, self.xform_matrix)
    resolved = self.resolve_style_properties(sketch, shape_style_map, **kwargs)
    for attrib_name, attrib_value in resolved.items():
        setattr(sketch, attrib_name, attrib_value)
    precedence_keys = {
        "color",
        "line_color",
        "fill_color",
        "alpha",
        "line_alpha",
        "fill_alpha",
    }
    for k, v in kwargs.items():
        if k not in precedence_keys:
            setattr(sketch, k, v)
    self.active_page.sketches.append(sketch)

    return self


def ellipse(
    self, center: PointType, width: float, height, angle, **kwargs
) -> None:
    """
    Draw an ellipse with the given center and x_radius and y_radius.

    Args:
        center (PointType): Center of the ellipse.
        width (float): Width of the ellipse.
        height: Height of the ellipse.
        angle: Angle of the ellipse.
        **kwargs: Additional keyword arguments.
    """
    x, y = center[:2]
    x_radius = width / 2
    y_radius = height / 2
    p1 = x - x_radius, y - y_radius
    p2 = x + x_radius, y + y_radius
    p3 = x - x_radius, y + y_radius
    p4 = x + x_radius, y - y_radius
    self._all_vertices.extend([p1, p2, p3, p4])
    sketch = EllipseSketch(center, x_radius, y_radius, angle, self.xform_matrix)
    resolved = self.resolve_style_properties(sketch, shape_style_map, **kwargs)
    for attrib_name, attrib_value in resolved.items():
        setattr(sketch, attrib_name, attrib_value)
    precedence_keys = {
        "color",
        "line_color",
        "fill_color",
        "alpha",
        "line_alpha",
        "fill_alpha",
    }
    for k, v in kwargs.items():
        if k not in precedence_keys:
            setattr(sketch, k, v)
    self.active_page.sketches.append(sketch)

    return self


def text(
    self,
    txt: str,
    pos: PointType,
    font_family: str = None,
    font_size: int = None,
    font_color: Color = None,
    anchor: Anchor = None,
    align: str = None,
    **kwargs,
) -> None:
    """
    Draw the given text at the given position.

    Args:
        txt (str): Text to be drawn.
        pos (PointType): Position of the text.
        font_family (str, optional): Font family of the text. Defaults to None.
        font_size (int, optional): Font size of the text. Defaults to None.
        font_color (Color, optional): Font color of the text. Defaults to None.
        anchor (Anchor, optional): Anchor of the text. Defaults to None.
        **kwargs: Additional keyword arguments.
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
    self._sketch_xform_matrix = identity_matrix()
    # then call get_tag_sketch to create a TagSketch object
    sketch = create_sketch(tag_obj, self, **kwargs)
    self.active_page.sketches.append(sketch)

    return self


def line(self, start, end, **kwargs):
    """
    Draw a line segment from start to end.

    Args:
        start: Starting point of the line.
        end: Ending point of the line.
        **kwargs: Additional keyword arguments.

    Returns:
        Self: The canvas object.
    """
    self._sketch_xform_matrix = self.xform_matrix
    line_shape = Shape([start, end], closed=False, **kwargs)
    line_sketch = create_sketch(line_shape, self, **kwargs)
    self.active_page.sketches.append(line_sketch)
    self._sketch_xform_matrix = identity_matrix()
    return self


def rectangle(
    self, center: PointType, width: float, height: float, angle: float, **kwargs
):
    """
    Draw a rectangle with the given center, width, height and angle.

    Args:
        center (PointType): Center of the rectangle.
        width (float): Width of the rectangle.
        height (float): Height of the rectangle.
        angle (float): Angle of the rectangle.
        **kwargs: Additional keyword arguments.

    Returns:
        Self: The canvas object.
    """
    w2 = width / 2
    h2 = height / 2
    p1 = center[0] - w2, center[1] + h2
    p2 = center[0] - w2, center[1] - h2
    p3 = center[0] + w2, center[1] - h2
    p4 = center[0] + w2, center[1] + h2
    points = homogenize([p1, p2, p3, p4]) @ rotation_matrix(angle, center)
    rect_shape = Shape(points.tolist(), closed=True, **kwargs)
    rect_sketch = create_sketch(rect_shape, self, **kwargs)
    self.active_page.sketches.append(rect_sketch)

    return self


def draw_CS(self, size: float = None, **kwargs):
    """
    Draw a coordinate system with the given size.

    Args:
        size (float, optional): Size of the coordinate system. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Self: The canvas object.
    """
    if size is None:
        size = defaults["CS_size"]
    if "colors" in kwargs:
        x_color, y_color = kwargs["colors"]
        del kwargs["colors"]
    else:
        x_color = defaults["CS_x_color"]
        y_color = defaults["CS_y_color"]
    if "line_width" not in kwargs:
        kwargs["line_width"] = defaults["CS_line_width"]
    self.line((0, 0), (size, 0), line_color=x_color, **kwargs)
    self.line((0, 0), (0, size), line_color=y_color, **kwargs)
    if "line_color" not in kwargs:
        kwargs["line_color"] = defaults["CS_origin_color"]
    self.circle(radius=defaults["CS_origin_size"], **kwargs)

    return self


def lines(self, points, **kwargs):
    """
    Draw connected line segments.

    Args:
        points: Points to be connected.
        **kwargs: Additional keyword arguments.

    Returns:
        Self: The canvas object.
    """
    self._all_vertices.extend(points)
    sketch = LineSketch(points, self.xform_matrix, **kwargs)
    for attrib_name in line_style_map:
        attrib_value = self.resolve_property(sketch, attrib_name)
        setattr(sketch, attrib_name, attrib_value)
    self.active_page.sketches.append(sketch)

    return self


def _measure_latex_formula(formula, font_size, font_family, bold):
    """Render the formula to SVG and return its (width, height) in points."""
    import io
    import re
    import matplotlib
    import matplotlib.pyplot as plt

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
    self,
    formula: str,
    pos: PointType,
    font_size: int = 14,
    font_family: str = None,
    font_color=None,
    bold: bool = False,
    anchor: Anchor = None,
    **kwargs,
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
        **kwargs: Additional keyword arguments (passed through for future use).

    Returns:
        Self: The canvas object.
    """
    sketch = LatexSketch(
        formula=formula,
        pos=pos,
        font_size=font_size,
        font_family=font_family,
        font_color=font_color,
        bold=bold,
        anchor=anchor,
        xform_matrix=self.xform_matrix,
    )
    # Measure the formula's rendered bounding box so we can register the
    # correct canvas extents in _all_vertices.
    W, H = _measure_latex_formula(formula, font_size, font_family, bold)
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


def insert_code(self, code: str, location: TexLoc = TexLoc.NONE) -> Self:
    """
    Insert code into the canvas.

    Args:
        code (str): The code to insert.

    Returns:
        Self: The canvas object.
    """
    active_sketches = self.active_page.sketches
    sketch = TexSketch(code, location=location)
    active_sketches.append(sketch)

    return self


def draw_bbox(self, bbox, **kwargs):
    """
    Draw the bounding box object.

    Args:
        bbox: Bounding box to be drawn.
        **kwargs: Additional keyword arguments.

    Returns:
        Self: The canvas object.
    """
    sketch = create_sketch(bbox, self, **kwargs)
    self.active_page.sketches.append(sketch)

    return self


def draw_pattern(self, pattern, **kwargs):
    """
    Draw the pattern object.

    Expands the pattern into individual shapes via get_shapes() and
    creates a sketch for each shape.

    Args:
        pattern: Pattern object to be drawn.
        **kwargs: Additional keyword arguments.

    Returns:
        Self: The canvas object.
    """
    active_sketches = self.active_page.sketches
    shapes = pattern.get_shapes()
    for shape in shapes:
        sketches = get_sketches(shape, self, **kwargs)
        if sketches:
            active_sketches.extend(sketches)

    return self


def draw_group(self, group, **kwargs):
    """
    Draw the group object.

    Args:
        pattern: Pattern object to be drawn.
        **kwargs: Additional keyword arguments.

    Returns:
        Self: The canvas object.
    """
    sketch = create_sketch(group, self, **kwargs)
    self.active_page.sketches.append(sketch)

    return self


def draw_hobby(
    self,
    points: Sequence[PointType],
    controls: Sequence[PointType],
    cyclic: bool = False,
    **kwargs,
):
    """Draw a Hobby curve through the given points using the control points.

    Args:
        points (Sequence[PointType]): Points through which the curve passes.
        controls (Sequence[PointType]): Control points for the curve.
        cyclic (bool, optional): Whether the curve is cyclic. Defaults to False.
        **kwargs: Additional keyword arguments.
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


def shade_value(angle):
    """
    Returns a weight value (between 0 and 1) based on angle input.

    Args:
        angle (float): Angle in radians between 0 and pi

    Returns:
        float: Weight value where:
               - angle = pi/2 -> returns 1.0
               - angle = 0 or pi -> returns 0.0

    The function follows a sine curve: sin(angle)
    """
    if not 0 <= angle <= 2 * pi:
        raise ValueError("Angle must be between 0 and 2 pi radians.")

    return sin(angle)


def plait_emboss1(self, lace, **kwargs):
    if "fill_color" not in kwargs:
        if lace.plait_color is None:
            lace.plait_color = defaults["plait_color"]
        kwargs["fill_color"] = lace.plait_color
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
            ind2 = ind1 + 1
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
        (p5, ind5), (p5, ind5), mp3 = ep[2]
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

    if lace.shade_plaits:
        kwargs.pop("fill_color")
        dist = lace.width * 3
        cx, cy = lace.midpoint
        far_point = cx - dist, cy + dist
        if lace.plait_color is None:
            color = defaults["plait_color"]
        else:
            color = lace.plait_color
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
            draw(self, quad1, fill_color=color1, **kwargs)
            draw(self, quad2, fill_color=color2, **kwargs)
    else:
        for quad in all_quads:
            draw(self, quad, **kwargs)


def plait_emboss2(self, lace, **kwargs):
    if "fill_color" not in kwargs:
        if lace.plait_color is None:
            lace.plait_color = defaults["plait_color"]
        kwargs["fill_color"] = lace.plait_color
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
    if lace.shade_plaits:
        kwargs.pop("fill_color")
        dist = lace.width * 3
        cx, cy = lace.midpoint
        far_point = cx - dist, cy + dist

        color = lace.plait_color
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

            x1, y1 = intersect(line1, line2)
            x2, y2 = intersect(line1, line3)
            shade_step = 0.2

            if x2 < x1:
                color1 = change_lightness(color, shade_factor * -shade_step)
                color2 = change_lightness(color, shade_factor * shade_step)
            elif x1 < x2:
                color1 = change_lightness(color, shade_factor * shade_step)
                color2 = change_lightness(color, shade_factor * -shade_step)
            else:
                color1 = color2 = color
            draw(self, quad1, fill_color=color1, **kwargs)
            draw(self, quad2, fill_color=color2, **kwargs)
    else:
        for quad in quads:
            draw(self, quad, **kwargs)


def plait_diamond(self, lace, **kwargs):
    lace._set_plait_ends()
    quad_pairs = []
    end_quads = []
    inner_loops = []
    if "fill_color" not in kwargs:
        if lace.plait_color is None:
            lace.plait_color = defaults["plait_color"]
        kwargs["fill_color"] = lace.plait_color
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
        for i in range(n2):
            draw(self, quads[i], **kwargs)
            draw(self, quads[-(i + 2)], **kwargs)
            quad_pairs.append((quads[i], quads[-(i + 2)]))
        end_quads.extend([quads[-1], quads[n2]])

    dist = lace.width * 3
    cx, cy = lace.midpoint
    far_point = cx - dist, cy + dist

    kwargs.pop("fill_color")
    color = lace.plait_color
    for quad1, quad2 in quad_pairs:
        angle = inclination_angle(*quad1[:2])
        shade_angle = abs(radians(135) - angle)
        shade_factor = shade_value(shade_angle)

        mid1 = midpoint(*quad1[:2])
        line1 = (far_point, mid1)
        line2 = quad1[:2]
        line3 = quad2[:2]

        x1, y1 = intersect(line1, line2)
        x2, y2 = intersect(line1, line3)
        shade_step = 0.2

        if x2 < x1:
            color1 = change_lightness(color, shade_factor * -shade_step)
            color2 = change_lightness(color, shade_factor * shade_step)
        elif x1 < x2:
            color1 = change_lightness(color, shade_factor * shade_step)
            color2 = change_lightness(color, shade_factor * -shade_step)
        else:
            color1 = color2 = color

        draw(self, quad1, fill_color=color1, **kwargs)
        draw(self, quad2, fill_color=color2, **kwargs)

    color = change_lightness(lace.plait_color, -0.1)
    for quad in end_quads:
        draw(self, quad, fill_color=color, **kwargs)

    color = change_lightness(lace.plait_color, 0.1)
    for loop in inner_loops:
        draw(self, loop, fill_color=color, **kwargs)


def draw_lace_with_fillets(self, lace, **kwargs):
    r1, r2 = kwargs["fillet_radii"]
    rounded_fragments = lace._fillet_fragments(r1, r2)
    palette = kwargs.get('palette')
    self.draw_fragments(palette=palette, fragments=rounded_fragments, **kwargs)

    rounded_plaits = lace._fillet_plaits(r1, r2)
    # We do not handle other plait options for rounded plaits yet!
    if "plait_fill_color" not in kwargs:
        fill_color = defaults["plait_color"]
    else:
        fill_color = kwargs['plait_fill_color']
    for plait in rounded_plaits:
        draw(self, plait, fill_color=fill_color, **kwargs)


def draw_plaits(self, lace=None, **kwargs):
    if lace is None:
        plaits = kwargs["plaits"]
    else:
        plaits = lace.plaits

    if "plait_fill_color" not in kwargs:
        color = defaults["plait_color"]
        if color is not None:
            kwargs["plait_fill_color"] = color
    for plait in plaits:
        self._all_vertices.extend(plait.corners) # This may be redundant!!!

    if "plait_style" in kwargs:
        _handle_plait_style(self, lace, kwargs)
    else:
        _draw_default_plaits(self, lace, kwargs)



def draw_fragments(self, lace=None, palette=None, **kwargs):
    areas = []
    if lace is None:
        fragments = kwargs["fragments"]
    else:
        fragments = lace.fragments

    for fragment in fragments:
        areas.append((fragment.area, fragment.id))
    bins = group_into_bins(areas, 2)

    if "swatch" in kwargs:
        palette = kwargs["swatch"]
    else:
        if palette is None:
            palette = d_name_palette["div_ROMA_256"]

    n_palette = len(palette)
    n_bins = len(bins)
    palette = palette[:: n_palette // n_bins]
    palette = [Color(*c) for c in palette]
    n = len(palette)
    for i, bin_ in enumerate(bins):
        color = palette[i % n]
        for _, fragment_id in bin_:
            fragment = d_id_obj[fragment_id]
            draw_kwargs = dict(kwargs)
            draw_kwargs["fill_color"] = color
            draw(self, fragment, **draw_kwargs)


def _handle_plait_innerlines(canvas, lace, **kwargs):
    """Handle INNERLINES plait style."""
    for plait in lace.plaits:
        canvas.active_page.sketches.append(
            create_sketch(plait, canvas, **kwargs)
        )

    if not lace.plaits[0].lerp_points:
        offsets = kwargs.get("percent_offsets", [0.5])
        widths = kwargs.get("line_widths", [1])

        lace._set_plait_inner_lines(offsets, widths)

        # Check if line widths match the number of lerp points
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
                line_width = plait.line_widths[i] if widths else 1
                canvas.active_page.sketches.append(
                    create_sketch(shape, canvas, line_width=line_width, **kwargs)
                )


def _handle_plait_style(canvas, lace, kwargs):
    """Handle different plait styles."""
    p_style = kwargs["plait_style"]

    if p_style == PlaitStyle.INNERLINES:
        _handle_plait_innerlines(canvas, lace, **kwargs)
    elif p_style == PlaitStyle.INNERLOOPS:
        pass  # No implementation yet
    elif p_style == PlaitStyle.DIAMOND:
        plait_diamond(canvas, lace, **kwargs)
    elif p_style == PlaitStyle.EMBOSS1:
        plait_emboss1(canvas, lace, **kwargs)
    elif p_style == PlaitStyle.EMBOSS2:
        plait_emboss2(canvas, lace, **kwargs)


def _draw_default_plaits(canvas, lace, kwargs):
    """Draw plaits with default style."""
    for plait in lace.plaits:
        canvas.active_page.sketches.append(
            create_sketch(plait, canvas, **kwargs)
        )


def draw_lace(self, lace, **kwargs):
    """Draw the lace object.

    Args:
        lace: Lace object to be drawn.
        **kwargs: Additional keyword arguments.

    Returns:
        Self: The canvas object.
    """
    # keys = list(lace.fragment_groups.keys())
    # keys.sort()
    # if lace.swatch is not None:
    #     n_colors = len(lace.swatch)
    # for i, key in enumerate(keys):
    #     if lace.swatch is not None:
    #         fill_color = colors.Color(*lace.swatch[i % n_colors])
    #         kwargs["fill_color"] = fill_color
    #     for fragment in lace.fragment_groups[key]:
    #         self.active_page.sketches.append(create_sketch(fragment, self, **kwargs))
    if "fillet_radii" in kwargs:
        self.draw_lace_with_fillets(lace, **kwargs)
        return self

    palette = kwargs.get('palette', None)

    self.draw_fragments(lace, palette=palette, **kwargs)

    self.draw_plaits(lace, **kwargs)

    return self


def draw_image(self, image, position=None, scale=None, **kwargs):
    """Draw the image object.

    Args:
        image: Image object to be drawn.
        position: Position to draw the image at.
        scale: Scale to draw the image at.
        **kwargs: Additional keyword arguments.

    Returns:
        Self: The canvas object.
    """
    if not image.visible:
        return self
    if not image.active:
        return self
    translation, rotation, decomposed_scale = decompose_transformations(
        image.xform_matrix
    )
    if position is None:
        x, y = image.pos[:2]
    else:
        x, y = position[:2]
    dx, dy = translation
    pos = [x + dx, y + dy]

    if scale is None:
        scale = decomposed_scale

    sketch = ImageSketch(
        image,
        pos=pos,
        angle=rotation,
        scale=scale,
        size=image.size,
        file_path=image.file_path,
        anchor=image.anchor,
        xform_matrix=self.xform_matrix,
    )
    for attrib_name in shape_style_map:
        attrib_value = self.resolve_property(image, attrib_name)
        setattr(sketch, attrib_name, attrib_value)
    for attrib_name, attrib_value in kwargs.items():
        setattr(sketch, attrib_name, attrib_value)
    self.active_page.sketches.append(sketch)

    return self


def draw_pdf(self, pdf, pos=None, size=None, scale=None, angle=0, **kwargs):
    """Draw a PDF file on the canvas.

    Args:
        pdf: PDF object or file path.
        position (PointType): Upper-left position to draw the PDF at.
        size (tuple, optional): Size to draw the PDF at. Defaults to None.
        scale (float, optional): Scale factor for the PDF. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Self: The canvas object.
    """
    if not isinstance(pdf, str):
        if not pdf.visible:
            return self
        if not pdf.active:
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
    sketch = PDFSketch(
        file_path,
        pos=pos,
        scale=scale,
        angle=angle,
        size=size,
    )
    for attrib_name, attrib_value in kwargs.items():
        setattr(sketch, attrib_name, attrib_value)
    self.active_page.sketches.append(sketch)

    return self


def draw_dimension(self, item, **kwargs):
    """Draw the dimension object.

    Args:
        item: Dimension object to be drawn.
        **kwargs: Additional keyword arguments.

    Returns:
        Self: The canvas object.
    """
    for shape in item.all_shapes:
        self._all_vertices.extend(shape.corners)

    def _add_sketch(sketch):
        if sketch is None:
            return
        if isinstance(sketch, list):
            self.active_page.sketches.extend(sketch)
        else:
            self.active_page.sketches.append(sketch)

    for ext in [item.ext1, item.ext2, item.ext3]:
        if ext:
            _add_sketch(create_sketch(ext, self, **kwargs))
    if item.dim_line:
        _add_sketch(create_sketch(item.dim_line, self, **kwargs))
    if item.arrow1:
        _add_sketch(create_sketch(item.arrow1, self, **kwargs))
        _add_sketch(create_sketch(item.mid_line, self))
    if item.arrow2:
        _add_sketch(create_sketch(item.arrow2, self, **kwargs))
    x, y = item.text_pos[:2]

    tag = Tag(item.text, (x, y), font_size=item.font_size, **kwargs)
    # extend vertices with the Tag's bounding box
    extend_vertices(self, tag)
    tag_sketch = create_sketch(tag, self, **kwargs)
    tag_sketch.draw_frame = False
    tag_sketch.frame_shape = FrameShape.CIRCLE
    tag_sketch.fill = True
    tag_sketch.font_color = colors.black
    tag_sketch.back_style = BackStyle.COLOR
    tag_sketch.frame_back_color = colors.white
    tag_sketch.back_color = colors.white
    tag_sketch.stroke = False
    self.active_page.sketches.append(tag_sketch)

    return self


def grid(
    self,
    pos=(0, 0),
    width: float = None,
    height: float = None,
    step_size=None,
    **kwargs,
):
    """Draw a square grid with the given size.

    Args:
        pos (tuple, optional): Position of the grid. Defaults to (0, 0).
        width (float, optional): Length of the grid along the x-axis. Defaults to None.
        height (float, optional): Length of the grid along the y-axis. Defaults to None.
        step_size (optional): Step size for the grid. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Self: The canvas object.
    """
    x, y = pos[:2]
    if width is None:
        width = defaults["grid_size"]
        height = defaults["grid_size"]
    if "line_width" not in kwargs:
        kwargs["line_width"] = defaults["grid_line_width"]
    if "line_color" not in kwargs:
        kwargs["line_color"] = defaults["grid_line_color"]
    if "line_dash_array" not in kwargs:
        kwargs["line_dash_array"] = defaults["grid_line_dash_array"]

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
    Types.BATCH,
    Types.BEZIER,
    Types.CIRCLE,
    Types.CIRCULAR_GRID,
    Types.DIVISION,
    Types.DOT,
    Types.DOTS,
    Types.ELLIPSE,
    Types.FRAGMENT,
    Types.HEX_GRID,
    Types.LINE,
    Types.LINPATH,
    Types.MIXED_GRID,
    Types.OUTLINE,
    Types.OVERLAP,
    Types.PARALLEL_POLYLINE,
    Types.PLAIT,
    Types.POLYLINE,
    Types.Q_BEZIER,
    Types.RADIAL_DIMENSION,
    Types.RECTANGLE,
    Types.SECTION,
    Types.SEGMENT,
    Types.SHAPE,
    Types.SINE_WAVE,
    Types.SQUARE_GRID,
    Types.STAR,
    Types.TAG,
]


def extend_vertices(canvas, item):
    """Extend the list of all vertices with the vertices of the given item.

    Args:
        canvas: Canvas object.
        item: Item whose vertices are to be extended.
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
    elif item.subtype == Types.TAG:
        # Tag objects have all_vertices property that includes text bounding box
        vertices = [
            x[:2]
            for x in homogenize(item.all_vertices) @ canvas._sketch_xform_matrix
        ]
        all_vertices.extend(vertices)
    elif item.subtype == Types.ARROW:
        for shape in item.all_shapes:
            all_vertices.extend(shape.corners)
    elif item.subtype == Types.LACE:
        for plait in item.plaits:
            all_vertices.extend(plait.corners)
        for fragment in item.fragments:
            all_vertices.extend(fragment.corners)
    elif item.subtype == Types.LINPATH:
        vertices = [
            x[:2]
            for x in homogenize(item.all_vertices) @ canvas._sketch_xform_matrix
        ]
        all_vertices.extend(vertices)
    elif item.subtype == Types.PATTERN:
        all_vertices.extend(item.all_vertices)
    elif item.subtype == Types.BATCH:
        for element in item:
            extend_vertices(canvas, element)
    else:
        corners = [
            x[:2]
            for x in homogenize(item.corners) @ canvas._sketch_xform_matrix
        ]
        all_vertices.extend(corners)


def draw(self, item: Union[Shape, Batch], **kwargs) -> Self:
    """The item is drawn on the canvas with the given style properties.

    Args:
        item (Drawable): Item to be drawn.
        **kwargs: Additional keyword arguments.

    Returns:
        Self: The canvas object.
    """
    # check if the item has any points
    if not item:
        return self

    active_sketches = self.active_page.sketches
    subtype = item.subtype
    extend_vertices(self, item)
    if subtype == Types.LINPATH and kwargs.get("handles", False):
        handle_size = defaults["handle_marker_size"]
        half_size = handle_size / 2
        for handle in item.handles:
            if not handle:
                continue
            # handle segment itself
            self._all_vertices.extend(handle)

            # square handle markers at segment endpoints (3x3)
            for x, y in (handle[0], handle[-1]):
                self._all_vertices.extend(
                    [
                        (x - half_size, y - half_size),
                        (x + half_size, y - half_size),
                        (x + half_size, y + half_size),
                        (x - half_size, y + half_size),
                    ]
                )

    if subtype in regular_sketch_types:
        sketches = get_sketches(item, self, **kwargs)
        if sketches:
            active_sketches.extend(sketches)
    elif subtype == Types.IMAGE:
        draw_image(self, item, **kwargs)
    elif subtype == Types.PATTERN:
        draw_pattern(self, item, **kwargs)
    elif subtype == Types.GROUP:
        draw_group(self, item, **kwargs)
    elif subtype == Types.DIMENSION:
        self.draw_dimension(item, **kwargs)
    elif subtype == Types.ARROW:
        for head in item.heads:
            active_sketches.append(create_sketch(head, self, **kwargs))
        active_sketches.append(create_sketch(item.line, self, **kwargs))
    elif subtype == Types.LACE:
        self.draw_lace(item, **kwargs)
    elif subtype == Types.BOUNDING_BOX:
        draw_bbox(self, item, **kwargs)
    return self


def draw_all_segments(
    self, item: Union[Shape, Batch], vert_indices=False, **kwargs
) -> Self:
    """
    Using intersections, splits edges of the item into separate segments and
    draws them with their indices. This is usually used for the "get_loop"
    function.
    If vert_indices is True, then vertex indices are shown instead of edge
    indices.

    Args:
        item: A shape or a batch.
        vert_indices: If vert_indices is True, then vertex indices are
        shown instead of edge indices.

    Returns:
        The canvas object.
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


def get_sketches(
    item: Drawable, canvas: "Canvas" = None, **kwargs
) -> list["Sketch"]:
    """Create sketches from the given item and return them as a list.

    Args:
        item (Drawable): Item to be sketched.
        canvas (Canvas, optional): Canvas object. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        list[Sketch]: List of sketches.
    """
    if not (item.visible and item.active):
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


def set_shape_sketch_style(sketch, item, canvas, linear=False, **kwargs):
    """Set the style properties of the sketch.

    Args:
        sketch: Sketch object.
        item: Item whose style properties are to be set.
        canvas: Canvas object.
        linear (bool, optional): Whether the style is linear. Defaults to False.
        **kwargs: Additional keyword arguments.
    """
    if linear:
        style_map = line_style_map
    else:
        style_map = shape_style_map

    resolved_style = canvas.resolve_style_properties(item, style_map, **kwargs)

    for attrib_name, attrib_value in resolved_style.items():
        setattr(sketch, attrib_name, attrib_value)

    sketch.visible = item.visible
    sketch.active = item.active
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
    if "_mask_context_id" in item.__dict__:
        sketch._mask_context_id = item._mask_context_id
    if "_mask_opacity" in item.__dict__:
        sketch._mask_opacity = item._mask_opacity
    if "_mask_stops" in item.__dict__:
        sketch._mask_stops = item._mask_stops
    if "_mask_axis" in item.__dict__:
        sketch._mask_axis = item._mask_axis

    if "even_odd" in item.__dict__:
        sketch.even_odd = item.even_odd

    precedence_keys = {
        "color",
        "line_color",
        "fill_color",
        "alpha",
        "line_alpha",
        "fill_alpha",
    }
    for k, v in kwargs.items():
        if k in precedence_keys:
            continue
        setattr(sketch, k, v)


def get_verts_in_new_pos(item, **kwargs):
    """
    Get the vertices of the item in a new position.

    Args:
        item: Item whose vertices are to be obtained.
        **kwargs: Additional keyword arguments.

    Returns:
        list: List of vertices in the new position.
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


def create_sketch(item, canvas, **kwargs):
    """Create a sketch from the given item.

    Args:
        item: Item to be sketched.
        canvas: Canvas object.
        **kwargs: Additional keyword arguments.

    Returns:
        Sketch: Created sketch.
    """
    if not (item.visible and item.active):
        return None

    def get_tag_sketch(item, canvas, **kwargs):
        """Create a TagSketch from the given item.

        Args:
            item: Item to be sketched.
            canvas: Canvas object.
            **kwargs: Additional keyword arguments.

        Returns:
            TagSketch: Created TagSketch.
        """
        if "pos" in kwargs:
            pos = kwargs["pos"]
        else:
            pos = item.pos

        sketch = TagSketch(
            text=item.text,
            pos=pos,
            anchor=item.anchor,
            xform_matrix=canvas.xform_matrix,
        )
        for attrib_name in item._style_map:
            if attrib_name == "fill_color":
                if item.fill_color in [None, colors.black]:
                    setattr(
                        sketch, "frame_back_color", defaults["frame_back_color"]
                    )
                else:
                    setattr(sketch, "frame_back_color", item.fill_color)
                continue
            attrib_value = canvas.resolve_property(item, attrib_name)
            setattr(sketch, attrib_name, attrib_value)
        sketch.text_width = item.text_width
        sketch.visible = item.visible
        sketch.active = item.active
        for k, v in kwargs.items():
            setattr(sketch, k, v)
        return sketch

    def get_ellipse_sketch(item, canvas, **kwargs):
        """Create an EllipseSketch from the given item.

        Args:
            item: Item to be sketched.
            canvas: Canvas object.
            **kwargs: Additional keyword arguments.

        Returns:
            EllipseSketch: Created EllipseSketch.
        """
        if "pos" in kwargs:
            center = kwargs["pos"]
        else:
            center = item.center

        sketch = EllipseSketch(
            center,
            item.a,
            item.b,
            item.angle,
            xform_matrix=canvas.xform_matrix,
            **kwargs,
        )
        set_shape_sketch_style(sketch, item, canvas, **kwargs)

        return sketch

    def get_pattern_sketch(item, canvas, **kwargs):
        """Create a PatternSketch from the given item.

        Args:
            item: Item to be sketched.
            canvas: Canvas object.
            **kwargs: Additional keyword arguments.

        Returns:
            PatternSketch: Created PatternSketch.
        """
        sketch = PatternSketch(item, xform_matrix=canvas.xform_matrix)
        set_shape_sketch_style(sketch, item, canvas, **kwargs)

        return sketch

    def get_group_sketch(item, canvas, **kwargs):
        """Create a GroupSketch from the given item.

        Args:
            item: Item to be sketched.
            canvas: Canvas object.
            **kwargs: Additional keyword arguments.

        Returns:
            GroupSketch: Created GroupSketch.
        """
        sketch = GroupSketch(item, xform_matrix=canvas.xform_matrix)
        set_shape_sketch_style(sketch, item, canvas, **kwargs)

        return sketch

    def get_circle_sketch(item, canvas, **kwargs):
        """Create a CircleSketch from the given item.

        Args:
            item: Item to be sketched.
            canvas: Canvas object.
            **kwargs: Additional keyword arguments.

        Returns:
            CircleSketch: Created CircleSketch.
        """
        if "pos" in kwargs:
            center = kwargs["pos"]
        else:
            center = item.center
        sketch = CircleSketch(
            center, item.radius, xform_matrix=canvas.xform_matrix
        )
        set_shape_sketch_style(sketch, item, canvas, **kwargs)

        return sketch

    def get_dots_sketch(item, canvas, **kwargs):
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

    def get_arc_sketch(item, canvas, **kwargs):
        """Create an ArcSketch from the given item.

        Args:
            item: Item to be sketched.
            canvas: Canvas object.
            **kwargs: Additional keyword arguments.

        Returns:
            ArcSketch: Created ArcSketch.
        """

        # vertices = get_verts_in_new_pos(item, **kwargs)
        sketch = ArcSketch(
            item.vertices, xform_matrix=canvas._sketch_xform_matrix
        )
        set_shape_sketch_style(sketch, item, canvas, **kwargs)

        return sketch

    def get_lace_sketch(item, canvas, **kwargs):
        """Create sketches for lace from the given item.

        Args:
            item: Item to be sketched.
            canvas: Canvas object.
            **kwargs: Additional keyword arguments.

        Returns:
            list: List of created sketches.
        """
        sketches = [
            get_sketch(frag, canvas, **kwargs) for frag in item.fragments
        ]
        sketches.extend(
            [get_sketch(plait, canvas, **kwargs) for plait in item.plaits]
        )
        return sketches

    def get_batch_sketch(item, canvas, **kwargs):
        """Create a BatchSketch from the given item.

        Args:
            item: Item to be sketched.
            canvas: Canvas object.
            **kwargs: Additional keyword arguments.

        Returns:
            BatchSketch: Created BatchSketch.
        """
        swatch = kwargs.get("swatch", None)
        batch_clip = kwargs.get("clip", None)

        raw_mask = kwargs.get("mask", None)
        if raw_mask is not None and hasattr(raw_mask, "shape"):
            batch_mask = raw_mask.shape
            mask_payload = {
                "_mask_opacity": getattr(raw_mask, "opacity", 1.0),
                "_mask_stops": getattr(raw_mask, "stops", None),
                "_mask_axis": getattr(
                    raw_mask, "axis", ((0.0, 0.0), (1.0, 0.0))
                ),
            }
        else:
            batch_mask = raw_mask
            mask_payload = {
                "_mask_opacity": kwargs.get("_mask_opacity", 1.0),
                "_mask_stops": kwargs.get("_mask_stops", None),
                "_mask_axis": kwargs.get(
                    "_mask_axis", ((0.0, 0.0), (1.0, 0.0))
                ),
            }

        def _apply_mask_kwargs(target_kwargs):
            if "mask" not in target_kwargs and batch_mask is not None:
                target_kwargs["mask"] = batch_mask
            if "clip" not in target_kwargs and batch_clip is not None:
                target_kwargs["clip"] = batch_clip
            if mask_payload["_mask_opacity"] != 1.0:
                target_kwargs["_mask_opacity"] = mask_payload["_mask_opacity"]
            if mask_payload["_mask_stops"] is not None:
                target_kwargs["_mask_stops"] = mask_payload["_mask_stops"]
                target_kwargs["_mask_axis"] = mask_payload["_mask_axis"]

        batch_mask_context_id = None
        batch_mask_context_bbox = None
        if batch_mask is not None and not batch_clip:
            batch_mask_context_id = f"mask_batch_{id(item)}"
            try:
                bbox = item.b_box
                batch_mask_context_bbox = (
                    bbox.southwest[0],
                    bbox.southwest[1],
                    bbox.width,
                    bbox.height,
                )
            except Exception:
                batch_mask_context_bbox = None
        if swatch:
            n_swatch = len(swatch)
        sketches = []
        count = 0
        for element in item.elements:
            if element.visible and element.active:
                element_kwargs = dict(kwargs)
                if swatch:
                    element_kwargs["fill_color"] = Color(
                        *swatch[count % n_swatch]
                    )
                _apply_mask_kwargs(element_kwargs)
                if batch_mask_context_id is not None:
                    element_kwargs["_mask_context_id"] = (
                        batch_mask_context_id
                    )
                if batch_mask_context_bbox is not None:
                    element_kwargs["_mask_context_bbox"] = (
                        batch_mask_context_bbox
                    )
                sketches.extend(get_sketches(element, canvas, **element_kwargs))
                count += 1

        sketch = BatchSketch(sketches=sketches)
        for arg in group_args:
            setattr(sketch, arg, getattr(item, arg, None))
        if batch_clip is not None:
            sketch.clip = batch_clip
        if batch_mask is not None:
            sketch.mask = batch_mask
        if batch_mask_context_id is not None:
            sketch._mask_context_id = batch_mask_context_id
        if batch_mask_context_bbox is not None:
            sketch._mask_context_bbox = batch_mask_context_bbox
        if mask_payload["_mask_opacity"] != 1.0:
            sketch._mask_opacity = mask_payload["_mask_opacity"]
        if mask_payload["_mask_stops"] is not None:
            sketch._mask_stops = mask_payload["_mask_stops"]
            sketch._mask_axis = mask_payload["_mask_axis"]

        return sketch

    def get_path_sketch(item, canvas, **kwargs):
        """Create sketches for a path from the given item.

        Args:
            item: Item to be sketched.
            canvas: Canvas object.
            **kwargs: Additional keyword arguments.

        Returns:
            list: List of created sketches.
        """

        def extend_verts(obj, vertices):
            obj_vertices = obj.vertices
            if obj_vertices:
                if vertices and close_points2(vertices[-1], obj_vertices[0]):
                    obj_vertices = obj_vertices[1:]
                vertices.extend(obj_vertices)

        path_op = PathOperation
        linears = [
            path_op.ARC,
            path_op.ARC_TO,
            path_op.BLEND_ARC,
            path_op.BLEND_CUBIC,
            path_op.BLEND_QUAD,
            path_op.BLEND_SINE,
            path_op.CUBIC_TO,
            path_op.FORWARD,
            path_op.H_LINE_TO,
            path_op.R_H_LINE,
            path_op.HOBBY_TO,
            path_op.QUAD_TO,
            path_op.R_LINE,
            path_op.SEGMENTS,
            path_op.SINE,
            path_op.V_LINE_TO,
            path_op.R_V_LINE,
            path_op.LINE_TO,
        ]
        sketches = []
        vertices = []
        for i, op in enumerate(item.operations):
            if op.subtype in linears:
                obj = item.objects[i]
                extend_verts(obj, vertices)
            elif op.subtype in [path_op.MOVE_TO, path_op.R_MOVE]:
                if i == 0:
                    continue
                shape = Shape(vertices)
                sketch = create_sketch(shape, canvas, **kwargs)
                if sketch:
                    sketch.visible = item.visible
                    sketch.active = item.active
                    sketches.append(sketch)
                vertices = []
            elif op.subtype == path_op.CLOSE:
                shape = Shape(vertices, closed=True)
                sketch = create_sketch(shape, canvas, **kwargs)
                if sketch:
                    sketch.visible = item.visible
                    sketch.active = item.active
                    sketches.append(sketch)
                vertices = []
        if vertices:
            shape = Shape(vertices)
            sketch = create_sketch(shape, canvas, **kwargs)
            sketches.append(sketch)

        if "handles" in kwargs and kwargs["handles"]:
            handles = kwargs["handles"]
            del kwargs["handles"]
            for handle in item.handles:
                shape = Shape(handle)
                shape.subtype = Types.HANDLE
                handle_sketches = create_sketch(shape, canvas, **kwargs)
                sketches.extend(handle_sketches)

        for sketch in sketches:
            item.closed = sketch.closed
            set_shape_sketch_style(sketch, item, canvas, **kwargs)

        return sketches

    def get_bbox_sketch(item, canvas, **kwargs):
        """Create a bounding box sketch from the given item.

        Args:
            item: Item to be sketched.
            canvas: Canvas object.
            **kwargs: Additional keyword arguments.

        Returns:
            ShapeSketch: Created bounding box sketch.
        """
        nround = defaults["tikz_nround"]
        vertices = [
            (round(x[0], nround), round(x[1], nround)) for x in item.corners
        ]
        if not vertices:
            return None

        sketch = ShapeSketch(vertices, canvas._sketch_xform_matrix)
        sketch.subtype = Types.BBOX_SKETCH
        sketch.visible = True
        sketch.active = True
        sketch.closed = True
        sketch.fill = False
        sketch.stroke = True
        sketch.line_color = colors.gray
        sketch.line_width = 1
        sketch.line_dash_array = [3, 3]
        sketch.draw_markers = False
        return sketch

    def get_handle_sketch(item, canvas, **kwargs):
        """Create handle sketches from the given item.

        Args:
            item: Item to be sketched.
            canvas: Canvas object.
            **kwargs: Additional keyword arguments.

        Returns:
            list: List of created handle sketches.
        """
        nround = defaults["tikz_nround"]
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
        handle_size = defaults["handle_marker_size"]
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

    def get_sketch(item, canvas, **kwargs):
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

        # vertices = get_verts_in_new_pos(item, **kwargs)
        nround = defaults["tikz_nround"]
        vertices = [
            (round(x[0], nround), round(x[1], nround)) for x in item.vertices
        ]

        sketch = ShapeSketch(vertices, canvas._sketch_xform_matrix)
        set_shape_sketch_style(sketch, item, canvas, **kwargs)

        return sketch

    def get_line_sketch(item, canvas, **kwargs):
        """Create a line sketch from the given item."""
        if not item.vertices:
            return None

        nround = defaults["tikz_nround"]
        vertices = [
            (round(x[0], nround), round(x[1], nround)) for x in item.vertices
        ]
        sketch = LineSketch(vertices, canvas._sketch_xform_matrix)
        set_shape_sketch_style(sketch, item, canvas, **kwargs)
        sketch.alpha = canvas.resolve_property(item, "alpha")
        sketch.extent = item.extent

        return sketch

    def get_image_sketch(item, canvas, **kwargs):
        """Create an ImageSketch from the given item.

        Args:
            item: Item to be sketched.
            canvas: Canvas object.
            **kwargs: Additional keyword arguments.

        Returns:
            ImageSketch: Created ImageSketch.
        """
        _, rotation, scale = decompose_transformations(item.xform_matrix)
        sketch = ImageSketch(
            item,
            pos=item.pos,
            angle=rotation,
            scale=scale,
            anchor=item.anchor,
            size=item.size,
            file_path=item.file_path,
            xform_matrix=canvas.xform_matrix,
            **kwargs,
        )
        set_shape_sketch_style(sketch, item, canvas, **kwargs)

        return sketch

    d_subtype_sketch = {
        Types.ARC: get_arc_sketch,
        Types.ARC_ARROW: get_batch_sketch,
        Types.ARROW: get_batch_sketch,
        Types.ARROW_HEAD: get_sketch,
        Types.BATCH: get_batch_sketch,
        Types.BEZIER: get_sketch,
        Types.BOUNDING_BOX: get_bbox_sketch,
        Types.CIRCLE: get_circle_sketch,
        Types.CIRCULAR_GRID: get_batch_sketch,
        Types.DIVISION: get_sketch,
        Types.DOT: get_circle_sketch,
        Types.DOTS: get_dots_sketch,
        Types.ELLIPSE: get_sketch,
        Types.FRAGMENT: get_sketch,
        Types.GROUP: get_group_sketch,
        Types.HANDLE: get_handle_sketch,
        Types.HEX_GRID: get_batch_sketch,
        Types.IMAGE: get_image_sketch,
        Types.LACE: get_lace_sketch,
        Types.LINE: get_line_sketch,
        Types.LINPATH: get_path_sketch,
        Types.MIXED_GRID: get_batch_sketch,
        Types.MASK: get_sketch,
        Types.OVERLAP: get_batch_sketch,
        Types.PARALLEL_POLYLINE: get_batch_sketch,
        Types.PATTERN: get_pattern_sketch,
        Types.PLAIT: get_sketch,
        Types.POLYLINE: get_sketch,
        Types.RADIAL_DIMENSION: get_batch_sketch,
        Types.Q_BEZIER: get_sketch,
        Types.RECTANGLE: get_sketch,
        Types.SECTION: get_sketch,
        Types.SEGMENT: get_sketch,
        Types.SHAPE: get_sketch,
        Types.SINE_WAVE: get_sketch,
        Types.SQUARE_GRID: get_batch_sketch,
        Types.STAR: get_batch_sketch,
        Types.TAG: get_tag_sketch,
    }

    return d_subtype_sketch[item.subtype](item, canvas, **kwargs)

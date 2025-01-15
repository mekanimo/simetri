""" Canvas object uses these methods to draw shapes and text. """

from math import cos, sin
from typing_extensions import Self

from ..helpers.geometry import homogenize
from ..graphics.all_enums import (
    Types,
    Drawable,
    drawable_types,
    BackStyle,
    FrameShape,
    Anchor,
)
from ..colors import colors
from ..tikz.tikz import scope_code_required
from ..graphics.sketch import (
    TagSketch,
    LineSketch,
    CircleSketch,
    ArcSketch,
    EllipseSketch,
    ShapeSketch,
    BatchSketch,
)
from ..settings.settings import defaults
from ..canvas.style_map import line_style_map, shape_style_map, group_args
from ..helpers.illustration import Tag
from ..graphics.shape import Shape
from ..graphics.common import Point


Color = colors.Color


def arc(
    self, center: Point, radius: float, start_angle: float, end_angle: float, **kwargs
) -> None:
    """Draw an arc with the given center, radius, start and end
    angles in radians.
    Arc is drawn in counterclockwise direction from start to end.
    """
    cx, cy = center[:2]
    x = cx + radius * cos(start_angle)
    y = cy + radius * sin(start_angle)
    end_point = cx + radius * cos(end_angle), cy + radius * sin(end_angle)
    self._all_vertices.extend([(x, y), end_point, center])

    x, y = center[:2]
    p1 = x - radius, y - radius
    p2 = x + radius, y + radius
    p3 = x - radius, y + radius
    p4 = x + radius, y - radius
    self._all_vertices.extend([p1, p2, p3, p4])
    sketch = ArcSketch(
        center=center,
        radius=radius,
        start_angle=start_angle,
        end_angle=end_angle,
        xform_matrix=self.xform_matrix,
    )
    # for attrib_name in line_style_map:
    #     setattr(sketch, attrib_name, defaults[attrib_name])
    for attrib_name in shape_style_map:
        if hasattr(sketch, attrib_name):
            attrib_value = self.resolve_property(sketch, attrib_name)
        else:
            attrib_value = defaults[attrib_name]
        setattr(sketch, attrib_name, attrib_value)
    for k, v in kwargs.items():
        setattr(sketch, k, v)
    self.active_page.sketches.append(sketch)

    return self


def circle(self, center: Point, radius: float, **kwargs) -> None:
    """Draw a circle with the given center and radius."""
    x, y = center[:2]
    p1 = x - radius, y - radius
    p2 = x + radius, y + radius
    p3 = x - radius, y + radius
    p4 = x + radius, y - radius
    self._all_vertices.extend([p1, p2, p3, p4])
    sketch = CircleSketch(center, radius, self.xform_matrix)
    for attrib_name in shape_style_map:
        if hasattr(sketch, attrib_name):
            attrib_value = self.resolve_property(sketch, attrib_name)
        else:
            attrib_value = defaults[attrib_name]
        setattr(sketch, attrib_name, attrib_value)
    for k, v in kwargs.items():
        setattr(sketch, k, v)
    self.active_page.sketches.append(sketch)

    return self


def text(
    self,
    txt: str,
    pos: Point,
    font_name: str = None,
    font_size: int = None,
    font_color: Color = None,
    anchor: Anchor = None,
    **kwargs
) -> None:
    """Draw the given text at the given position."""
    # first create a Tag object
    tag_obj = Tag(
        txt,
        pos,
        font_name=font_name,
        font_size=font_size,
        font_color=font_color,
        anchor=anchor,
        **kwargs
    )
    tag_obj.draw_frame = False
    # then call get_tag_sketch to create a TagSketch object
    sketch = create_sketch(tag_obj, self, **kwargs)
    self.active_page.sketches.append(sketch)

    return self


def line(self, start, end, **kwargs):
    """Draw a line segment from start to end."""

    line_shape = Shape([start, end], closed=False, **kwargs)
    line_sketch = create_sketch(line_shape, self, **kwargs)
    self.active_page.sketches.append(line_sketch)

    return self


def draw_CS(self, size: float = None, **kwargs):
    """Draw a coordinate system with the given size."""
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
    self.circle((0, 0), radius=defaults["CS_origin_size"], **kwargs)

    return self


def lines(self, points, **kwargs):
    """Draw connected line segments."""
    self._all_vertices.extend(points)
    sketch = LineSketch(points, self.xform_matrix, **kwargs)
    for attrib_name in line_style_map:
        attrib_value = self._resolve_property(sketch, attrib_name)
        setattr(sketch, attrib_name, attrib_value)
    self.active_page.sketches.append(sketch)

    return self


def draw_lace(self, lace, **kwargs):
    """Draw the lace object."""
    keys = list(lace.fragment_groups.keys())
    keys.sort()
    if lace.swatch is not None:
        n_colors = len(lace.swatch)
    for i, key in enumerate(keys):
        if lace.swatch is not None:

            fill_color = colors.Color(*lace.swatch[i % n_colors])
            kwargs["fill_color"] = fill_color
        for fragment in lace.fragment_groups[key]:
            self.active_page.sketches.append(create_sketch(fragment, self, **kwargs))
    for plait in lace.plaits:
        if lace.swatch is not None:
            fill_color = colors.white
            kwargs["fill_color"] = fill_color
        else:
            kwargs["fill_color"] = None
        self.active_page.sketches.append(create_sketch(plait, self, **kwargs))
        self._all_vertices.extend(plait.corners)

    return self


def draw_dimension(self, item, **kwargs):
    """Draw the dimension object."""
    for shape in item.all_shapes:
        self._all_vertices.extend(shape.corners)
    for ext in [item.ext1, item.ext2, item.ext3]:
        if ext:
            ext_sketch = create_sketch(ext, self, **kwargs)
            self.active_page.sketches.append(ext_sketch)
    if item.dim_line:
        dim_sketch = create_sketch(item.dim_line, self, **kwargs)
        self.active_page.sketches.extend(dim_sketch)
    if item.arrow1:
        arrow_sketch = create_sketch(item.arrow1, self, **kwargs)
        self.active_page.sketches.extend(arrow_sketch)
        self.active_page.sketches.append(create_sketch(item.mid_line, self))
    if item.arrow2:
        arrow_sketch = create_sketch(item.arrow2, self, **kwargs)
        self.active_page.sketches.extend(arrow_sketch)
    x, y = item.text_pos[:2]
    tag_sketch = TagSketch(item.text, (x, y), font_size=item.font_size, **kwargs)
    tag_sketch.draw_frame = True
    tag_sketch.frame_shape = FrameShape.CIRCLE
    tag_sketch.fill = True
    tag_sketch.font_color = colors.black
    tag_sketch.frame_back_style = BackStyle.COLOR
    tag_sketch.back_style = BackStyle.COLOR
    tag_sketch.frame_back_color = colors.white
    tag_sketch.back_color = colors.white
    tag_sketch.stroke = False
    self.active_page.sketches.append(tag_sketch)

    return self


def grid(
    self, pos=(0, 0), x_len: float = None, y_len: float = None, step_size=None, **kwargs
):
    """Draw a square grid with the given size."""
    x, y = pos
    if x_len is None:
        x_len = defaults["grid_size"]
        y_len = defaults["grid_size"]
    if "line_width" not in kwargs:
        kwargs["line_width"] = defaults["grid_line_width"]
    if "line_color" not in kwargs:
        kwargs["line_color"] = defaults["grid_line_color"]
    if "line_dash_array" not in kwargs:
        kwargs["line_dash_array"] = defaults["grid_line_dash_array"]
    # draw x-axis
    # self.line((-size, 0), (size, 0), **kwargs)
    line_y = Shape([(x, y), (x + x_len, y)], **kwargs)
    line_x = Shape([(x, y), (x, y + y_len)], **kwargs)
    lines_x = line_y.translate(0, step_size, reps=int(y_len / step_size))
    lines_y = line_x.translate(step_size, 0, reps=int(x_len / step_size))
    # self.line((x, 0), (x+size, 0), **kwargs)
    # # draw y-axis
    # self.line((0, y), (0, y+size), **kwargs)
    # for i in arange(step_size, size + 1, step_size):
    #     self.line((i, -size), (i, size), **kwargs)
    #     self.line((-size, i), (size, i), **kwargs)
    #     self.line((-i, -size), (-i, size), **kwargs)
    #     self.line((-size, -i), (size, -i), **kwargs)
    self.draw(lines_x)
    self.draw(lines_y)
    return self


regular_sketch_types = [
    Types.ARC,
    Types.ARCARROW,
    Types.BATCH,
    Types.CIRCLE,
    Types.DIVISION,
    Types.DOT,
    Types.DOTS,
    Types.ELLIPSE,
    Types.FRAGMENT,
    Types.OUTLINE,
    Types.OVERLAP,
    Types.PARALLELPOLYLINE,
    Types.PLAIT,
    Types.POLYLINE,
    Types.RECTANGLE,
    Types.SECTION,
    Types.SEGMENT,
    Types.SHAPE,
    Types.STAR,
    Types.TAG,
]


def extend_vertices(canvas, item):
    """Extend the list of all vertices with the vertices of the given item."""
    all_vertices = canvas._all_vertices
    if item.subtype == Types.DOTS:
        vertices = [x.pos for x in item.all_shapes]
        vertices = [x[:2] for x in homogenize(vertices) @ canvas._xform_matrix]
        all_vertices.extend(vertices)
    elif item.subtype == Types.DOT:
        vertices = [item.pos]
        vertices = [x[:2] for x in homogenize(vertices) @ canvas._xform_matrix]
        all_vertices.extend(vertices)
    elif item.subtype == Types.ARROW:
        for shape in item.all_shapes:
            all_vertices.extend(shape.corners)
    elif item.subtype == Types.LACE:
        for plait in item.plaits:
            all_vertices.extend(plait.corners)
        for fragment in item.fragments:
            all_vertices.extend(fragment.corners)
    else:
        corners = [x[:2] for x in homogenize(item.corners) @ canvas._xform_matrix]
        all_vertices.extend(corners)


def draw(self, item: Drawable, **kwargs) -> Self:
    """The item is drawn on the canvas with the given style properties."""
    # check if the item has any points
    if not item:
        return self
    active_sketches = self.active_page.sketches
    subtype = item.subtype
    extend_vertices(self, item)
    if subtype in regular_sketch_types:
        sketches = get_sketches(item, self, **kwargs)
        if sketches:
            active_sketches.extend(sketches)
    elif subtype == Types.DIMENSION:
        self.draw_dimension(item, **kwargs)
    elif subtype == Types.ARROW:
        for head in item.heads:
            active_sketches.append(create_sketch(head, self, **kwargs))
        active_sketches.append(create_sketch(item.line, self, **kwargs))
    elif subtype == Types.LACE:
        self.draw_lace(item, **kwargs)

    return self


def get_sketches(item: Drawable, canvas: "Canvas" = None, **kwargs) -> list["Sketch"]:
    """Create sketches from the given item and return them as a list."""
    if not (item.visible and item.active):
        res = []
    elif item.subtype in drawable_types:
        sketches = create_sketch(item, canvas, **kwargs)
        if isinstance(sketches, list):
            res = sketches
        else:
            res = [sketches]
    else:
        res = []
    return res


def create_sketch(item, canvas, **kwargs):
    """Sketch is a neutral format for drawing.
    It contains geometry and style properties.
    They are not meant to be transformed, only to be drawn.
    Sketches have no methods, only data.
    Like all objects, they have a unique ID, visible and active properties.
    They do not check anything, they just store data.
    You should make sure the data is correct before creating a sketch.
    """
    if not (item.visible and item.active):
        return None

    def get_tag_sketch(item, canvas, **kwargs):
        pos = item.pos
        # pos = [(round(pos[0], nround), round(pos[1], nround))]
        sketch = TagSketch(text=item.text, pos=pos, anchor=item.anchor)
        for attrib_name in item._style_map:
            if attrib_name == "fill_color":
                if item.fill_color in [None, colors.black]:
                    setattr(sketch, "frame_back_color", defaults["frame_back_color"])
                else:
                    setattr(sketch, "frame_back_color", item.fill_color)
                continue
            attrib_value = canvas._resolve_property(item, attrib_name)
            setattr(sketch, attrib_name, attrib_value)

        sketch.visible = item.visible
        sketch.active = item.active
        for k, v in kwargs.items():
            setattr(sketch, k, v)
        return sketch

    def get_ellipse_sketch(item, canvas, **kwargs):
        sketch = EllipseSketch(
            item.center, item.width, item.height, xform_matrix=canvas.xform_matrix
        )
        sketch.visible = item.visible
        sketch.active = item.active
        for attrib_name in shape_style_map:
            attrib_value = canvas._resolve_property(item, attrib_name)
            setattr(sketch, attrib_name, attrib_value)

        for k, v in kwargs.items():
            setattr(sketch, k, v)

        return sketch

    def get_circle_sketch(item, canvas, **kwargs):
        sketch = CircleSketch(
            item.center, item.radius, xform_matrix=canvas.xform_matrix
        )
        sketch.visible = item.visible
        sketch.active = item.active
        for attrib_name in shape_style_map:
            attrib_value = canvas._resolve_property(item, attrib_name)
            setattr(sketch, attrib_name, attrib_value)

        for k, v in kwargs.items():
            setattr(sketch, k, v)

        return sketch

    def get_dots_sketch(item, canvas, **kwargs):
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
        sketch = ArcSketch(
            center=item.center,
            radius=item.radius,
            start_angle=item.start_angle,
            end_angle=item.end_angle,
            xform_matrix=canvas.xform_matrix,
        )
        sketch.visible = item.visible
        sketch.active = item.active
        for attrib_name in line_style_map:
            attrib_value = canvas._resolve_property(item, attrib_name)
            setattr(sketch, attrib_name, attrib_value)

        for k, v in kwargs.items():
            setattr(sketch, k, v)

        return sketch

    def get_lace_sketch(item, canvas, **kwargs):
        sketches = [get_sketch(frag, canvas, **kwargs) for frag in item.fragments]
        sketches.extend([get_sketch(plait, canvas, **kwargs) for plait in item.plaits])
        return sketches

    def get_batch_sketch(item, canvas, **kwargs):
        if scope_code_required(item):
            sketches = []
            for element in item.elements:
                if element.visible and element.active:
                    sketches.extend(get_sketches(element, canvas, **kwargs))

            sketch = BatchSketch(sketches=sketches)
            for arg in group_args:
                setattr(sketch, arg, getattr(item, arg))

            res = sketch
        else:
            sketches = []
            for element in item.elements:
                if element.visible and element.active:
                    sketches.extend(get_sketches(element, canvas, **kwargs))

            res = sketches

        return res

    def get_sketch(item, canvas, **kwargs):
        nround = defaults["tikz_nround"]
        vertices = [
            (round(x[0], nround), round(x[1], nround)) for x in item.final_coords
        ]
        sketch = ShapeSketch(vertices, canvas._xform_matrix)
        for attrib_name in item._style_map:
            attrib_value = canvas._resolve_property(item, attrib_name)
            setattr(sketch, attrib_name, attrib_value)
        sketch.visible = item.visible
        sketch.active = item.active
        sketch.closed = item.closed
        for k, v in kwargs.items():
            setattr(sketch, k, v)

        return sketch

    d_subtype_sketch = {
        Types.ARC: get_arc_sketch,
        Types.ARCARROW: get_batch_sketch,
        Types.ARROW: get_batch_sketch,
        Types.BATCH: get_batch_sketch,
        Types.CIRCLE: get_circle_sketch,
        Types.DIVISION: get_sketch,
        Types.DOT: get_circle_sketch,
        Types.DOTS: get_dots_sketch,
        Types.ELLIPSE: get_ellipse_sketch,
        Types.FRAGMENT: get_sketch,
        Types.LACE: get_lace_sketch,
        Types.OVERLAP: get_batch_sketch,
        Types.PARALLELPOLYLINE: get_batch_sketch,
        Types.PLAIT: get_sketch,
        Types.POLYLINE: get_sketch,
        Types.RECTANGLE: get_sketch,
        Types.SECTION: get_sketch,
        Types.SEGMENT: get_sketch,
        Types.SHAPE: get_sketch,
        Types.STAR: get_batch_sketch,
        Types.TAG: get_tag_sketch,
        Types.ARROWHEAD: get_sketch,
    }

    return d_subtype_sketch[item.subtype](item, canvas, **kwargs)

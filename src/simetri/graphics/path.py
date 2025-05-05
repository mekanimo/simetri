"""Path module for graphics package."""

from dataclasses import dataclass
from math import sin, cos, pi
from collections import deque
from typing_extensions import Self

import numpy as np

from .core import StyleMixin
from .batch import Batch
from .shape import Shape
from .common import Point, common_properties
from ..helpers.validation import validate_args
from .all_enums import PathOperation as PathOps
from .all_enums import Types
from ..canvas.style_map import shape_style_map, ShapeStyle, shape_args
from ..geometry.bezier import Bezier
from ..geometry.hobby import hobby_shape
from ..geometry.geometry import (
    homogenize,
    positive_angle,
    polar_to_cartesian,
    sine_points,
    close_points2
)
from ..geometry.ellipse import (
    ellipse_point,
    ellipse_tangent,
    elliptic_arc_points,
)
from ..geometry.geometry import extended_line, line_angle, line_by_point_angle_length
from .affine import translation_matrix, rotation_matrix
from ..settings.settings import defaults

array = np.array


@dataclass
class Operation:
    """An operation for a Path object.

    Attributes:
        subtype (Types): The subtype of the operation.
        data (tuple): The data associated with the operation.
        name (str): The name of the operation.
    """

    subtype: Types
    data: tuple
    name: str = ""

    def __post_init__(self):
        """Post-initialization to set the type and common properties."""
        self.type = Types.PATH_OPERATION
        common_properties(self, False)


class LinPath(Batch, StyleMixin):
    """LinerPath.
    A LinPath object is a container for various linear elements.
    Path objects can be transformed like other Shape and Batch objects.
    """

    def __init__(self, start: Point = (0, 0), angle: float = pi/2, **kwargs):
        """Initialize a Path object.

        Args:
            start (Point, optional): The starting point of the path. Defaults to (0, 0).
            angle (float, optional): The heading angle of the path. Defaults to pi/2.
            **kwargs: Additional keyword arguments. Common properties are line_width,
            line_color, stroke, etc.
        """
        if "style" in kwargs:
            self.__dict__["style"] = kwargs["style"]
            del kwargs["style"]
        else:
            self.__dict__["style"] = ShapeStyle()
        self.__dict__["_style_map"] = shape_style_map
        self._set_aliases()
        valid_args = shape_args
        validate_args(kwargs, valid_args)
        self.pos = start
        self.start = start
        self.angle = angle  # heading angle
        self.operations = []
        self.objects = []
        self.even_odd = True  # False is non-zero winding rule
        super().__init__(**kwargs)
        self.subtype = Types.LINPATH
        self.cur_shape = Shape([start])
        self.append(self.cur_shape)
        self.rc = self.r_coord  # alias for r_coord
        self.rp = self.r_polar  # alias for rel_polar
        self.handles = []
        self.stack = deque()
        for key, value in kwargs.items():
            setattr(self, key, value)
        common_properties(self)
        self.closed = False

    def __getattr__(self, name):
        """Retrieve an attribute of the shape.

        Args:
            name (str): The attribute name to return.

        Returns:
            Any: The value of the attribute.

        Raises:
            AttributeError: If the attribute cannot be found.
        """
        try:
            res = super().__getattr__(name)
        except AttributeError:
            res = self.__dict__[name]
        return res

    def __bool__(self):
        """Return True if the path has operations.
        Batch may have no elements yet still be True.

        Returns:
            bool: True if the path has operations.
        """
        return bool(self.operations)

    def _create_object(self):
        """Create an object using the last operation."""
        PO = PathOps
        op = self.operations[-1]
        op_type = op.subtype
        data = op.data
        if op_type in [PO.MOVE_TO, PO.R_MOVE]:
            self.cur_shape = Shape([data])
            self.append(self.cur_shape)
            self.objects.append(None)
        elif op_type in [PO.LINE_TO, PO.R_LINE, PO.H_LINE, PO.V_LINE, PO.FORWARD]:
            self.objects.append(Shape(data))
            self.cur_shape.append(data[1])
        elif op_type in [PO.SEGMENTS]:
            self.objects.append(Shape(data[1]))
            self.cur_shape.extend(data[1])
        elif op_type in [PO.SINE, PO.BLEND_SINE]:
            self.objects.append(Shape(data[0]))
            self.cur_shape.extend(data[0])
        elif op_type in [PO.CUBIC_TO, PO.QUAD_TO]:
            n_points = defaults["n_bezier_points"]
            curve = Bezier(data, n_points=n_points)
            self.objects.append(curve)
            self.cur_shape.extend(curve.vertices[1:])
            if op_type == PO.CUBIC_TO:
                self.handles.extend([(data[0], data[1]), (data[2], data[3])])
            else:
                self.handles.append((data[0], data[1]))
                self.handles.append((data[1], data[2]))
        elif op_type in [PO.HOBBY_TO]:
            n_points = defaults['n_hobby_points']
            curve = hobby_shape(data[1], n_points=n_points)
            self.objects.append(Shape(curve.vertices))
        elif op_type in [PO.ARC, PO.BLEND_ARC]:
            self.objects.append(Shape(data[-1]))
            self.cur_shape.extend(data[-1][1:])
        elif op_type in [PO.CLOSE]:
            self.cur_shape.closed = True
            self.cur_shape = Shape([self.pos])
            self.objects.append(None)
            self.append(self.cur_shape)
        else:
            raise ValueError(f"Invalid operation type: {op_type}")

    def copy(self) -> "LinPath":
        """Return a copy of the path.

        Returns:
            LinPath: The copied path object.
        """

        new_path = LinPath(start=self.start)
        new_path.pos = self.pos
        new_path.angle = self.angle
        new_path.operations = self.operations.copy()
        new_path.objects = []
        for obj in self.objects:
            if obj is not None:
                new_path.objects.append(obj.copy())
        new_path.even_odd = self.even_odd
        new_path.cur_shape = self.cur_shape.copy()
        new_path.handles = self.handles.copy()
        new_path.stack = deque(self.stack)
        for attrib in shape_style_map:
            setattr(new_path, attrib, getattr(self, attrib))

        return new_path

    def _add(self, pos, op, data, pnt2=None, **kwargs):
        """Add an operation to the path.

        Args:
            pos (Point): The position of the operation.
            op (PathOps): The operation type.
            data (tuple): The data for the operation.
            pnt2 (Point, optional): An optional second point for the operation. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.operations.append(Operation(op, data))
        if op in [PathOps.ARC, PathOps.BLEND_ARC, PathOps.SINE, PathOps.BLEND_SINE]:
            self.angle = data[1]
        else:
            if pnt2 is not None:
                self.angle = line_angle(pnt2, pos)
            else:
                self.angle = line_angle(self.pos, pos)
        self._create_object()
        if "name" in kwargs:
            setattr(self, kwargs["name"], self.operations[-1])
        list(pos)[:2]
        self.pos = pos

    def push(self):
        """Push the current position onto the stack."""
        self.stack.append((self.pos, self.angle))

    def pop(self):
        """Pop the last position from the stack."""
        if self.stack:
            self.pos, self.angle = self.stack.pop()

    def r_coord(self, dx: float, dy: float) -> Point:
        """Return the relative coordinates of a point in a
        coordinate system with the path's midpoint and y-axis aligned
        with the path.angle.

        Args:
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            tuple: The relative coordinates.
        """
        x, y = self.pos[:2]
        theta = self.angle - pi / 2
        x1 = dx * cos(theta) - dy * sin(theta) + x
        y1 = dx * sin(theta) + dy * cos(theta) + y

        return x1, y1

    def r_polar(self, r: float, angle: float) -> Point:
        """Return the relative coordinates of a point in a polar
        coordinate system with the path's midpoint and 0 degree axis aligned
        with the path.angle.

        Args:
            r (float): The radius.
            angle (float): The angle in radians.

        Returns:
            tuple: The relative coordinates.
        """
        x, y = polar_to_cartesian(r, angle + self.angle - pi / 2)[:2]
        x1, y1 = self.pos[:2]

        return x1 + x, y1 + y

    def line_to(self, point: Point, **kwargs) -> Self:
        """Add a line to the path.

        Args:
            point (Point): The end point of the line.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        self._add(point, PathOps.LINE_TO, (self.pos, point))

        return self

    def forward(self, length: float, **kwargs) -> Self:
        """Extend the path by the given length.

        Args:
            length (float): The length to extend.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.

        Raises:
            ValueError: If the path angle is not set.
        """
        if self.angle is None:
            raise ValueError("Path angle is not set.")
        else:
            x, y = line_by_point_angle_length(self.pos, self.angle, length)[1][:2]
        self._add((x, y), PathOps.FORWARD, (self.pos, (x, y)))

        return self

    def move_to(self, point: Point, **kwargs) -> Self:
        """Move the path to a new point.

        Args:
            point (Point): The new point.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        self._add(point, PathOps.MOVE_TO, point)

        return self

    def r_line(self, dx: float, dy: float, **kwargs) -> Self:
        """Add a relative line to the path.

        Args:
            dx (float): The x offset.
            dy (float): The y offset.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        point = self.pos[0] + dx, self.pos[1] + dy
        self._add(point, PathOps.R_LINE, (self.pos, point))

        return self

    def r_move(self, dx: float = 0, dy: float = 0, **kwargs) -> Self:
        """Move the path to a new relative point.

        Args:
            dx (float): The x offset.
            dy (float): The y offset.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        x, y = self.pos[:2]
        point = (x + dx, y + dy)
        self._add(point, PathOps.R_MOVE, point)
        return self

    def h_line(self, length: float, **kwargs) -> Self:
        """Add a horizontal line to the path.

        Args:
            length (float): The length of the line.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        x, y = self.pos[0] + length, self.pos[1]
        self._add((x, y), PathOps.H_LINE, (self.pos, (x, y)))
        return self

    def v_line(self, length: float, **kwargs) -> Self:
        """Add a vertical line to the path.

        Args:
            length (float): The length of the line.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        x, y = self.pos[0], self.pos[1] + length
        self._add((x, y), PathOps.V_LINE, (self.pos, (x, y)))
        return self

    def segments(self, points, **kwargs) -> Self:
        """Add a series of line segments to the path.

        Args:
            points (list): The points of the segments.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """

        self._add(points[-1], PathOps.SEGMENTS, (self.pos, points), pnt2=points[-2], **kwargs)
        return self

    def cubic_to(self, control1: Point, control2: Point, end: Point, *args, **kwargs) -> Self:
        """Add a Bézier curve with two control points to the path. Multiple blended curves can be added
        by providing additional arguments.

        Args:
            control1 (Point): The first control point.
            control2 (Point): The second control point.
            end (Point): The end point of the curve.
            *args: Additional arguments for blended curves.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        self._add(
            end,
            PathOps.CUBIC_TO,
            (self.pos, control1, control2, end),
            pnt2=control2,
            **kwargs,
        )
        return self

    def hobby_to(self, points, **kwargs) -> Self:
        """Add a Hobby curve to the path.

        Args:
            points (list): The points of the Hobby curve.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        self._add(points[-1], PathOps.HOBBY_TO, (self.pos, points))
        return self


    def quad_to(self, control: Point, end: Point, *args, **kwargs) -> Self:
        """Add a quadratic Bézier curve to the path. Multiple blended curves can be added by providing
        additional arguments.

        Args:
            control (Point): The control point.
            end (Point): The end point of the curve.
            *args: Additional arguments for blended curves.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.

        Raises:
            ValueError: If an argument does not have exactly two elements.
        """
        self._add(
            end, PathOps.QUAD_TO, (self.pos[:2], control, end[:2]), pnt2=control, **kwargs
        )
        pos = end
        for arg in args:
            if len(arg) != 2:
                raise ValueError("Invalid number of arguments for curve.")
            if isinstance(arg[0], (int, float)):
                # (length, end)
                length = arg[0]
                control = extended_line(length, control, pos)
                end = arg[1]
                self._add(end, PathOps.QUAD_TO, (pos, control, end), pnt2=control)
                pos = end
            elif isinstance(arg[0], (list, tuple)):
                # (control, end)
                control = arg[0]
                end = arg[1]
                self._add(end, PathOps.QUAD_TO, (pos, control, end), pnt2=control)
                pos = end
        return self

    def blend_cubic(self, control1_length, control2: Point, end: Point, **kwargs) -> Self:
        """Add a cubic Bézier curve to the path where the first control point is computed based on a length.

        Args:
            control1_length (float): The length to the first control point.
            control2 (Point): The second control point.
            end (Point): The end point of the curve.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        c1 = line_by_point_angle_length(self.pos, self.angle, control1_length)[1]
        self._add(
            end,
            PathOps.CUBIC_TO,
            (self.pos, c1, control2, end),
            pnt2=control2,
            **kwargs,
        )
        return self

    def blend_quad(self, control_length, end: Point, **kwargs) -> Self:
        """Add a quadratic Bézier curve to the path where the control point is computed based on a length.

        Args:
            control_length (float): The length to the control point.
            end (Point): The end point of the curve.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        pos = list(self.pos[:2])
        c1 = line_by_point_angle_length(pos, self.angle, control_length)[1]
        self._add(end, PathOps.QUAD_TO, (pos, c1, end), pnt2=c1, **kwargs)
        return self

    def arc(
        self,
        radius_x: float,
        radius_y: float,
        start_angle: float,
        span_angle: float,
        rot_angle: float = 0,
        n_points=None,
        **kwargs,
    ) -> Self:
        """Add an arc to the path. The arc is defined by an ellipse (with rx as half-width and ry as half-height).
        The sign of the span angle determines the drawing direction.

        Args:
            radius_x (float): The x radius of the arc.
            radius_y (float): The y radius of the arc.
            start_angle (float): The starting angle of the arc.
            span_angle (float): The span angle of the arc.
            rot_angle (float, optional): The rotation angle of the arc. Defaults to 0.
            n_points (int, optional): The number of points to use for the arc. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        rx = radius_x
        ry = radius_y
        start_angle = positive_angle(start_angle)
        clockwise = span_angle < 0
        if n_points is None:
            n_points = defaults["n_arc_points"]
        points = elliptic_arc_points((0, 0), rx, ry, start_angle, span_angle, n_points)
        start = points[0]
        end = points[-1]
        # Translate the start to the current position and rotate by the rotation angle.
        dx = self.pos[0] - start[0]
        dy = self.pos[1] - start[1]
        rotocenter = start
        if rot_angle != 0:
            points = (
                homogenize(points)
                @ rotation_matrix(rot_angle, rotocenter)
                @ translation_matrix(dx, dy)
            )
        else:
            points = homogenize(points) @ translation_matrix(dx, dy)
        tangent_angle = ellipse_tangent(rx, ry, *end) + rot_angle
        if clockwise:
            tangent_angle += pi
        pos = points[-1]
        self._add(
            pos,
            PathOps.ARC,
            (pos, tangent_angle, rx, ry, start_angle, span_angle, rot_angle, points),
        )
        return self

    def blend_arc(
        self,
        radius_x: float,
        radius_y: float,
        start_angle: float,
        span_angle: float,
        sharp=False,
        n_points=None,
        **kwargs,
    ) -> Self:
        """Add a blended elliptic arc to the path.

        Args:
            radius_x (float): The x radius of the arc.
            radius_y (float): The y radius of the arc.
            start_angle (float): The starting angle of the arc.
            span_angle (float): The span angle of the arc.
            sharp (bool, optional): Whether the arc is sharp. Defaults to False.
            n_points (int, optional): The number of points to use for the arc. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        rx = radius_x
        ry = radius_y
        start_angle = positive_angle(start_angle)
        clockwise = span_angle < 0
        if n_points is None:
            n_points = defaults["n_arc_points"]
        points = elliptic_arc_points((0, 0), rx, ry, start_angle, span_angle, n_points)
        start = points[0]
        end = points[-1]
        # Translate the start to the current position and rotate by the computed rotation angle.
        dx = self.pos[0] - start[0]
        dy = self.pos[1] - start[1]
        rotocenter = start
        tangent = ellipse_tangent(rx, ry, *start)
        rot_angle = self.angle - tangent
        if clockwise:
            rot_angle += pi
        if sharp:
            rot_angle += pi
        points = (
            homogenize(points)
            @ rotation_matrix(rot_angle, rotocenter)
            @ translation_matrix(dx, dy)
        )
        tangent_angle = ellipse_tangent(rx, ry, *end) + rot_angle
        if clockwise:
            tangent_angle += pi
        pos = points[-1][:2]
        self._add(
            pos,
            PathOps.ARC,
            (pos, tangent_angle, rx, ry, start_angle, span_angle, rot_angle, points),
        )
        return self

    def sine(
        self,
        period: float = 40,
        amplitude: float = 20,
        duration: float = 40,
        phase_angle: float = 0,
        rot_angle: float = 0,
        damping: float = 0,
        n_points: int = 100,
        **kwargs,
    ) -> Self:
        """Add a sine wave to the path.

        Args:
            period (float, optional): _description_. Defaults to 40.
            amplitude (float, optional): _description_. Defaults to 20.
            duration (float, optional): _description_. Defaults to 1.
            n_points (int, optional): _description_. Defaults to 100.
            phase_angle (float, optional): _description_. Defaults to 0.
            damping (float, optional): _description_. Defaults to 0.
            rot_angle (float, optional): _description_. Defaults to 0.

        Returns:
            Path: The path object.
        """

        points = sine_points(
            period, amplitude, duration, n_points, phase_angle, damping
        )
        if rot_angle != 0:
            points = homogenize(points) @ rotation_matrix(rot_angle, points[0])
        points = homogenize(points) @ translation_matrix(*self.pos[:2])
        angle = line_angle(points[-2], points[-1])
        self._add(points[-1], PathOps.SINE, (points, angle))
        return self

    def blend_sine(
        self,
        period: float = 40,
        amplitude: float = 20,
        duration: float = 40,
        phase_angle: float = 0,
        damping: float = 0,
        n_points: int = 100,
        **kwargs,
    ) -> Self:
        """Add a blended sine wave to the path.

        Args:
            amplitude (float): The amplitude of the wave.
            frequency (float): The frequency of the wave.
            length (float): The length of the wave.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """

        points = sine_points(
            period, amplitude, duration, n_points, phase_angle, damping
        )
        start_angle = line_angle(points[0], points[1])
        rot_angle = self.angle - start_angle
        points = homogenize(points) @ rotation_matrix(rot_angle, points[0])
        points = homogenize(points) @ translation_matrix(*self.pos[:2])
        angle = line_angle(points[-2], points[-1])
        self._add(points[-1], PathOps.SINE, (points, angle))
        return self

    def close(self, **kwargs) -> Self:
        """Close the path.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        self._add(self.pos, PathOps.CLOSE, None, **kwargs)
        return self

    @property
    def vertices(self):
        """Return the vertices of the path.

        Returns:
            list: The vertices of the path.
        """
        vertices = []
        last_vert = None
        dist_tol2 = defaults["dist_tol"] ** 2
        for obj in self.objects:
            if obj is not None and obj.vertices:
                obj_verts = obj.vertices
                if last_vert:
                    if close_points2(last_vert,  obj_verts[0], dist_tol2):
                        vertices.extend(obj_verts[1:])
                    else:
                        vertices.extend(obj_verts)
                else:
                    vertices.extend(obj_verts)
                last_vert = obj_verts[-1]

        return vertices

    def set_style(self, name, value, **kwargs) -> Self:
        """Set the style of the path.

        Args:
            name (str): The name of the style.
            value (Any): The value of the style.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        self.operations.append((PathOps.STYLE, (name, value, kwargs)))
        return self

    def _update(self, xform_matrix: array, reps: int = 0, merge: bool = False) -> Batch:
        """Used internally. Update the shape with a transformation matrix.

        Args:
            xform_matrix (array): The transformation matrix.
            reps (int, optional): The number of repetitions, defaults to 0.

        Returns:
            Batch: The updated shape or a batch of shapes.
        """
        if reps == 0:
            for obj in self.objects:
                if obj is not None:
                    obj._update(xform_matrix)
            res = self
        else:
            paths = [self]
            path = self
            for _ in range(reps):
                path = path.copy()
                path._update(xform_matrix)
                paths.append(path)
            res = Batch(paths)
        if merge and reps > 0:
            res = res.merge_shapes()
        return res

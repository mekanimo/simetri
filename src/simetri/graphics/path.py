"""Path module for graphics package."""

from dataclasses import dataclass
from math import sin, cos, pi, degrees, radians, sqrt, acos, atan2, ceil
from collections import deque
import re
from typing_extensions import Self

import numpy as np

from .core import StyleMixin
from .batch import Batch
from .shape import Shape
from .bbox import bounding_box
from .common import PointType, common_properties
from ..helpers.validation import validate_args
from .all_enums import PathOperation as PathOps
from .all_enums import Types, TransformationType
from ..canvas.style_map import shape_style_map, ShapeStyle, shape_args
from ..geometry.bezier import Bezier
from ..geometry.hobby import hobby_shape
from ..geometry.geometry import (
    homogenize,
    positive_angle,
    polar_to_cartesian,
    sine_points,
    close_points2,
)
from ..geometry.ellipse import (
    ellipse_point,
    ellipse_tangent,
    elliptic_arc_points,
)
from ..geometry.geometry import (
    extended_line,
    line_angle,
    line_by_point_angle_length,
)
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

    def __init__(
        self, start: PointType = (0, 0), angle: float = pi / 2, **kwargs
    ):
        """Initialize a Path object.

        Args:
            start (PointType, optional): The starting point of the path. Defaults to (0, 0).
            angle (float, optional): The heading angle of the path. Defaults to pi/2.
            **kwargs: Additional keyword arguments. Common properties are line_width,
            line_color, stroke, etc.
        """
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
        # First try StyleMixin for style attributes (which handles aliases)
        if hasattr(self, "_aliases") and name in self._aliases:
            return StyleMixin.__getattr__(self, name)

        try:
            res = super().__getattr__(name)
        except AttributeError:
            try:
                res = self.__dict__[name]
            except KeyError:
                raise AttributeError(name)
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
        elif op_type in [
            PO.LINE_TO,
            PO.R_LINE,
            PO.H_LINE_TO,
            PO.V_LINE_TO,
            PO.FORWARD,
        ]:
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
            n_points = defaults["n_hobby_points"]
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
            else:
                new_path.objects.append(None)
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
            pos (PointType): The position of the operation.
            op (PathOps): The operation type.
            data (tuple): The data for the operation.
            pnt2 (PointType, optional): An optional second point for the operation. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        self.operations.append(Operation(op, data))
        if op in [
            PathOps.ARC,
            PathOps.BLEND_ARC,
            PathOps.SINE,
            PathOps.BLEND_SINE,
        ]:
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

    @property
    def all_vertices(self):
        """Return all vertices of the path.

        Returns:
            list: All vertices of the path.
        """
        all_vertices = []
        for obj in self.objects:
            if obj is not None:
                all_vertices.extend(obj.vertices)

        return all_vertices

    @property
    def all_elements(self):
        """Return the transformed geometric objects that make up the path."""
        return [obj for obj in self.objects if obj is not None]

    @property
    def b_box(self):
        """Return the bounding box of the path.

        Returns:
            The bounding box of the path.
        """

        return bounding_box(self.all_vertices)

    def push(self):
        """Push the current position onto the stack."""
        self.stack.append((self.pos, self.angle))

    def pop(self):
        """Pop the last position from the stack."""
        if self.stack:
            self.pos, self.angle = self.stack.pop()

    def r_coord(self, dx: float, dy: float) -> PointType:
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

    def r_polar(self, r: float, angle: float) -> PointType:
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

    def line_to(self, point: PointType, **kwargs) -> Self:
        """Add a line to the path.

        Args:
            point (PointType): The end point of the line.
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


        """

        x, y = line_by_point_angle_length(self.pos, self.angle, length)[1][:2]
        self._add((x, y), PathOps.FORWARD, (self.pos, (x, y)))

        return self

    def orient(self, angle: float) -> Self:
        """Set the path angle.

        Args:
            angle (float): The angle in radians.

        Returns: self
        """
        self.angle = angle

        return self

    def turn(self, angle: float, distance: float = 0) -> Self:
        """Turn by the given angle and forward by the given distance.

        Args:
            angle (float): The angle increment.
            distance (float): The forward distance.

        Returns:
            self: The path object.
        """

        self.angle += angle
        if distance != 0:
            self.forward(distance)

        return self

    def move_to(self, point: PointType, **kwargs) -> Self:
        """Move the path to a new point.

        Args:
            point (PointType): The new point.
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

    def h_line_to(self, x: float, **kwargs) -> Self:
        """Add a horizontal line to the path.

        Args:
            x (float): The x coordinate of the line end.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        y = self.pos[1]
        self._add((x, y), PathOps.H_LINE_TO, (self.pos, (x, y)))
        return self

    def r_h_line(self, length: float, **kwargs) -> Self:
        """Add a horizontal line to the path.

        Args:
            length (float): The length of the line.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        x, y = self.pos[0] + length, self.pos[1]
        self._add((x, y), PathOps.R_H_LINE, (self.pos, (x, y)))
        return self

    def v_line_to(self, y: float, **kwargs) -> Self:
        """Add a vertical line to the path.

        Args:
            y (float): The y coordinate of the line end.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        x = self.pos[0]
        self._add((x, y), PathOps.V_LINE_TO, (self.pos, (x, y)))
        return self

    def r_v_line(self, length: float, **kwargs) -> Self:
        """Add a vertical line to the path.

        Args:
            length (float): The length of the line.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        x, y = self.pos[0], self.pos[1] + length
        self._add((x, y), PathOps.R_V_LINE, (self.pos, (x, y)))
        return self

    def segments(self, points, **kwargs) -> Self:
        """Add a series of line segments to the path.

        Args:
            points (list): The points of the segments.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """

        self._add(
            points[-1],
            PathOps.SEGMENTS,
            (self.pos, points),
            pnt2=points[-2],
            **kwargs,
        )
        return self

    def cubic_to(
        self,
        control1: PointType,
        control2: PointType,
        end: PointType,
        *args,
        **kwargs,
    ) -> Self:
        """Add a Bézier curve with two control points to the path. Multiple blended curves can be added
        by providing additional arguments.

        Args:
            control1 (PointType): The first control point.
            control2 (PointType): The second control point.
            end (PointType): The end point of the curve.
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

    def r_segments(self, r_points, **kwargs) -> Self:
        """Add a series of line segments to the path.
           Instead of absolute coodinates uses (dx, dy) offsets.
        Args:
            r_points (list): The (dx, dy) values of the segments.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        # Convert relative offsets to absolute points
        points = []
        current_x, current_y = self.pos
        for dx, dy in r_points:
            current_x += dx
            current_y += dy
            points.append((current_x, current_y))

        self._add(
            points[-1],
            PathOps.SEGMENTS,
            (self.pos, points),
            pnt2=points[-2],
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

    def quad_to(
        self, control: PointType, end: PointType, *args, **kwargs
    ) -> Self:
        """Add a quadratic Bézier curve to the path. Multiple blended curves can be added by providing
        additional arguments.

        Args:
            control (PointType): The control point.
            end (PointType): The end point of the curve.
            *args: Additional arguments for blended curves.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.

        Raises:
            ValueError: If an argument does not have exactly two elements.
        """
        self._add(
            end,
            PathOps.QUAD_TO,
            (self.pos[:2], control, end[:2]),
            pnt2=control,
            **kwargs,
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
                self._add(
                    end, PathOps.QUAD_TO, (pos, control, end), pnt2=control
                )
                pos = end
            elif isinstance(arg[0], (list, tuple)):
                # (control, end)
                control = arg[0]
                end = arg[1]
                self._add(
                    end, PathOps.QUAD_TO, (pos, control, end), pnt2=control
                )
                pos = end
        return self

    def mirror_cubic_to(
        self, control2: PointType, end: PointType, **kwargs
    ) -> Self:
        """Same as SVG S (smooth cubic Bezier).

        Mirrors the previous second control point across the current position
        to get the first control point, then uses control2 as the second control point.

        Args:
            control2 (PointType): The second control point.
            end (PointType): The end point of the curve.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        # Get previous control point from last operation if it was a cubic
        prev_c2 = self.pos
        last_op = self.operations[-1] if self.operations else None
        if last_op and last_op.subtype in [
            PathOps.CUBIC_TO,
            PathOps.BLEND_CUBIC,
        ]:
            # data: (start, c1, c2, end)
            prev_c2 = last_op.data[2]

        # Mirror prev_c2 across current position
        cur_x, cur_y = self.pos
        c1_x = 2 * cur_x - prev_c2[0]
        c1_y = 2 * cur_y - prev_c2[1]
        control1 = (c1_x, c1_y)

        self._add(
            end,
            PathOps.CUBIC_TO,
            (self.pos, control1, control2, end),
            pnt2=control2,
            **kwargs,
        )
        return self

    def r_mirror_cubic_to(
        self, r_control2: PointType, r_end: PointType, **kwargs
    ) -> Self:
        """Same as SVG s (relative smooth cubic Bezier).

        Args:
            r_control2 (PointType): The relative second control point (dx, dy).
            r_end (PointType): The relative end point (dx, dy).
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        cur_x, cur_y = self.pos
        control2 = (cur_x + r_control2[0], cur_y + r_control2[1])
        end = (cur_x + r_end[0], cur_y + r_end[1])
        return self.mirror_cubic_to(control2, end, **kwargs)

    def mirror_quad_to(self, end: PointType, **kwargs) -> Self:
        """Same as SVG T (smooth quadratic Bezier).

        Mirrors the previous control point across the current position
        to get the new control point.

        Args:
            end (PointType): The end point of the curve.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        # Get previous control point from last operation if it was a quad
        prev_c1 = self.pos
        last_op = self.operations[-1] if self.operations else None
        if last_op and last_op.subtype in [PathOps.QUAD_TO, PathOps.BLEND_QUAD]:
            # data: (start, c1, end)
            prev_c1 = last_op.data[1]

        # Mirror prev_c1 across current position
        cur_x, cur_y = self.pos
        c1_x = 2 * cur_x - prev_c1[0]
        c1_y = 2 * cur_y - prev_c1[1]
        control = (c1_x, c1_y)

        self._add(
            end,
            PathOps.QUAD_TO,
            (self.pos[:2], control, end[:2]),
            pnt2=control,
            **kwargs,
        )
        return self

    def r_mirror_quad_to(self, r_end: PointType, **kwargs) -> Self:
        """Same as SVG t (relative smooth quadratic Bezier).

        Args:
            r_end (PointType): The relative end point (dx, dy).
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        cur_x, cur_y = self.pos
        end = (cur_x + r_end[0], cur_y + r_end[1])
        return self.mirror_quad_to(end, **kwargs)

    def blend_cubic(
        self, control1_length, control2: PointType, end: PointType, **kwargs
    ) -> Self:
        """Add a cubic Bézier curve to the path where the first control point is computed based on a length.

        Args:
            control1_length (float): The length to the first control point.
            control2 (PointType): The second control point.
            end (PointType): The end point of the curve.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        c1 = line_by_point_angle_length(self.pos, self.angle, control1_length)[
            1
        ]
        self._add(
            end,
            PathOps.CUBIC_TO,
            (self.pos, c1, control2, end),
            pnt2=control2,
            **kwargs,
        )
        return self

    def blend_quad(self, control_length, end: PointType, **kwargs) -> Self:
        """Add a quadratic Bézier curve to the path where the control point is computed based on a length.

        Args:
            control_length (float): The length to the control point.
            end (PointType): The end point of the curve.
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
        points = elliptic_arc_points(
            (0, 0), rx, ry, start_angle, span_angle, n_points
        )
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
            (
                pos,
                tangent_angle,
                rx,
                ry,
                start_angle,
                span_angle,
                rot_angle,
                points,
            ),
        )
        return self

    def arc_to(
        self,
        rx: float,
        ry: float,
        angle: float,
        large_arc_flag: bool,
        sweep_flag: bool,
        end: PointType,
        **kwargs,
    ) -> Self:
        """Same as SVG A (elliptical arc).

        Args:
            rx (float): The x radius of the ellipse.
            ry (float): The y radius of the ellipse.
            angle (float): The rotation angle of the ellipse in degrees.
            large_arc_flag (bool): Whether to use the large arc (True) or small arc (False).
            sweep_flag (bool): The sweep direction (True for clockwise in SVG coords).
            end (PointType): The end point of the arc.
            **kwargs: Additional keyword arguments.

        Returns:
            Path: The path object.
        """
        params = _get_svg_arc_params(
            self.pos, rx, ry, angle, large_arc_flag, sweep_flag, end
        )

        if params["type"] == "line":
            self.line_to(params["end"], **kwargs)
        elif params["type"] == "arc":
            self.arc(
                params["rx"],
                params["ry"],
                params["start_angle"],
                params["span_angle"],
                rot_angle=params["rot_angle"],
                **kwargs,
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
        points = elliptic_arc_points(
            (0, 0), rx, ry, start_angle, span_angle, n_points
        )
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
            (
                pos,
                tangent_angle,
                rx,
                ry,
                start_angle,
                span_angle,
                rot_angle,
                points,
            ),
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
                    if close_points2(last_vert, obj_verts[0], dist_tol2):
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

    def _update(
        self,
        xform_matrix: array,
        reps: int = 0,
        take: slice = None,
        incr: float = None,
        merge: bool = False,
        xform_type: TransformationType = None,
    ) -> Batch:
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


def lin_path_svg(LinPath):
    """Given a LinPath object returns the equivalent svg path."""

    def fmt(val):
        """Format a float to a string with 3 decimal places."""
        return f"{val:.3f}".rstrip("0").rstrip(".")

    parts = [f"M {fmt(linpath.start[0])},{fmt(linpath.start[1])}"]

    # We iterate operations. We align with objects by filtering out Style ops
    # which don't produce objects?
    # Wait, set_style appends to operations but NOT to objects?
    # Let's check set_style.
    # It appends to operations. It does NOT call _create_object or append to objects.
    # So objects list only corresponds to geometric operations.
    # I need to maintain an index for objects.

    obj_idx = 0
    PO = PathOps

    for op in linpath.operations:
        if isinstance(op, tuple):
            # Style operation
            continue

        st = op.subtype
        data = op.data

        # Current geometry object (if applicable)
        # Some ops like MOVE_TO append None to objects.
        # CLOSE appends None.
        current_obj = (
            linpath.objects[obj_idx] if obj_idx < len(linpath.objects) else None
        )

        if st in [PO.MOVE_TO, PO.R_MOVE]:
            # data is point (x,y)
            parts.append(f"M {fmt(data[0])},{fmt(data[1])}")

        elif st in [
            PO.LINE_TO,
            PO.R_LINE,
            PO.H_LINE_TO,
            PO.V_LINE_TO,
            PO.FORWARD,
        ]:
            # data is (start, end)
            end = data[1]
            parts.append(f"L {fmt(end[0])},{fmt(end[1])}")

        elif st == PO.SEGMENTS:
            # data is (start, points_list)
            for p in data[1]:
                parts.append(f"L {fmt(p[0])},{fmt(p[1])}")

        elif st in [PO.CUBIC_TO, PO.BLEND_CUBIC]:
            # data: (start, c1, c2, end)
            c1, c2, end = data[1], data[2], data[3]
            parts.append(
                f"C {fmt(c1[0])},{fmt(c1[1])} {fmt(c2[0])},{fmt(c2[1])} {fmt(end[0])},{fmt(end[1])}"
            )

        elif st in [PO.QUAD_TO, PO.BLEND_QUAD]:
            # data: (start, c1, end)
            c1, end = data[1], data[2]
            parts.append(
                f"Q {fmt(c1[0])},{fmt(c1[1])} {fmt(end[0])},{fmt(end[1])}"
            )

        elif st in [PO.ARC, PO.BLEND_ARC]:
            # data: (pos, tangent_angle, rx, ry, start_angle, span_angle, rot_angle, points)
            rx, ry = data[2], data[3]
            span = data[5]
            rot = degrees(data[6])
            points = data[7]
            end = points[-1]
            large_arc = 1 if abs(span) > pi else 0
            # Simetri convention: span > 0 is CCW.
            # SVG sweep-flag: 1 is positive-angle direction (CW in y-down).
            # If we assume Simetri coords map directly to SVG coords:
            # CCW span (>0) -> Positive SVG angle change -> sweep 1.
            # CW span (<0) -> Negative SVG angle change -> sweep 0.
            sweep = 1 if span > 0 else 0
            parts.append(
                f"A {fmt(rx)} {fmt(ry)} {fmt(rot)} {large_arc} {sweep} {fmt(end[0])},{fmt(end[1])}"
            )

        elif st == PO.CLOSE:
            parts.append("Z")

        elif st in [PO.SINE, PO.BLEND_SINE]:
            # data[0] is points
            for p in data[0]:
                parts.append(f"L {fmt(p[0])},{fmt(p[1])}")

        elif st == PO.HOBBY_TO:
            # Use the resolved shape vertices from objects
            if current_obj:
                # Skip the first point since it should match current pos
                verts = current_obj.vertices
                # HOBBY curve usually connects smoothly.
                # current_obj is a Shape.
                for p in verts[1:]:
                    parts.append(f"L {fmt(p[0])},{fmt(p[1])}")

        obj_idx += 1

    return " ".join(parts)


def svg_path_to_linpath(svg_path: str) -> LinPath:
    """Converts an svg path string to LinPath object."""
    if not svg_path:
        return LinPath()

    # Tokenizer
    tokens = re.findall(
        r"[A-Za-z]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", svg_path
    )

    start_point = (0.0, 0.0)
    idx = 0

    # Handle optional start M strictly for initialization
    if idx < len(tokens) and tokens[idx].lower() == "m":
        try:
            x = float(tokens[idx + 1])
            y = float(tokens[idx + 2])
            start_point = (x, y)
            # We consume M x y by setting start
            # But we need to be careful if M has multiple points (implicit L)
            # or if we want to run the loop cleanly.
            # Easier: Just start at (0,0) and let the first Move set the pos.
            # But LinPath always starts with a point.
            # So initialization with correct start is better.
            idx = 3
        except IndexError:
            pass

    lp = LinPath(start=start_point)

    # If we consumed the first coordinate, we must handle implicit L if any.
    # The loop below handles commands.
    # If we skipped M x y, the next token might be a number (implicit L) or a Command.
    # We need to prime 'current_cmd' if we skipped.

    current_cmd = ""
    i = 0

    if idx == 3:
        i = 3
        # If the first command was 'm' (relative), and we treated it as absolute for start,
        # subsequent implicit linetos are relative.
        # If 'M', absolute.
        first_cmd = tokens[0]
        if first_cmd == "M":
            current_cmd = "L"
        if first_cmd == "m":
            current_cmd = "l"

    while i < len(tokens):
        t = tokens[i]

        if t.isalpha():
            current_cmd = t
            i += 1
        else:
            if not current_cmd:
                # Should not happen if path is valid
                i += 1
                continue
            # Implicit command logic
            # If M/m -> L/l
            if current_cmd == "M":
                current_cmd = "L"
            elif current_cmd == "m":
                current_cmd = "l"
            # L, l, H, h, V, v, C, c, S, s, Q, q, T, t, A, a remain same

        cmd_lower = current_cmd.lower()
        is_rel = current_cmd == cmd_lower

        def get_nums(count):
            nonlocal i
            nums = []
            for _ in range(count):
                if i < len(tokens):
                    try:
                        nums.append(float(tokens[i]))
                    except ValueError:
                        break  # Hits a command?
                    i += 1
                else:
                    break
            if len(nums) < count:
                return None
            return nums

        if cmd_lower == "z":
            lp.close()

        elif cmd_lower == "m":
            coords = get_nums(2)
            if coords:
                if is_rel:
                    lp.r_move(*coords)
                else:
                    lp.move_to(coords)

        elif cmd_lower == "l":
            coords = get_nums(2)
            if coords:
                if is_rel:
                    lp.r_line(*coords)
                else:
                    lp.line_to(coords)

        elif cmd_lower == "h":
            coords = get_nums(1)
            if coords:
                val = coords[0]
                if is_rel:
                    lp.r_h_line(val)
                else:
                    lp.h_line_to(val)

        elif cmd_lower == "v":
            coords = get_nums(1)
            if coords:
                val = coords[0]
                if is_rel:
                    lp.r_v_line(val)
                else:
                    lp.v_line_to(val)

        elif cmd_lower == "c":
            coords = get_nums(6)
            if coords:
                c1 = (coords[0], coords[1])
                c2 = (coords[2], coords[3])
                end = (coords[4], coords[5])

                if is_rel:
                    cur_x, cur_y = lp.pos
                    c1 = (cur_x + c1[0], cur_y + c1[1])
                    c2 = (cur_x + c2[0], cur_y + c2[1])
                    end = (cur_x + end[0], cur_y + end[1])

                lp.cubic_to(c1, c2, end)

        elif cmd_lower == "s":
            coords = get_nums(4)
            if coords:
                c2 = (coords[0], coords[1])
                end = (coords[2], coords[3])

                prev_c2 = lp.pos
                last_op = lp.operations[-1] if lp.operations else None
                # Check for Cubic/BlendCubic
                if last_op and last_op.subtype in [
                    PathOps.CUBIC_TO,
                    PathOps.BLEND_CUBIC,
                ]:
                    # data: (start, c1, c2, end)
                    prev_c2 = last_op.data[2]

                cur_x, cur_y = lp.pos
                ref_x = 2 * cur_x - prev_c2[0]
                ref_y = 2 * cur_y - prev_c2[1]
                c1 = (ref_x, ref_y)

                if is_rel:
                    cur_x, cur_y = lp.pos
                    c2 = (cur_x + c2[0], cur_y + c2[1])
                    end = (cur_x + end[0], cur_y + end[1])

                lp.cubic_to(c1, c2, end)

        elif cmd_lower == "q":
            coords = get_nums(4)
            if coords:
                c1 = (coords[0], coords[1])
                end = (coords[2], coords[3])
                if is_rel:
                    cur_x, cur_y = lp.pos
                    c1 = (cur_x + c1[0], cur_y + c1[1])
                    end = (cur_x + end[0], cur_y + end[1])
                lp.quad_to(c1, end)

        elif cmd_lower == "t":
            coords = get_nums(2)
            if coords:
                end = (coords[0], coords[1])

                prev_c1 = lp.pos
                last_op = lp.operations[-1] if lp.operations else None
                if last_op and last_op.subtype in [
                    PathOps.QUAD_TO,
                    PathOps.BLEND_QUAD,
                ]:
                    # data: (start, c1, end) or similar?
                    # quad_to adds: PathOps.QUAD_TO, (pos, c1, end)
                    prev_c1 = last_op.data[1]

                cur_x, cur_y = lp.pos
                ref_x = 2 * cur_x - prev_c1[0]
                ref_y = 2 * cur_y - prev_c1[1]
                c1 = (ref_x, ref_y)

                if is_rel:
                    end = (cur_x + end[0], cur_y + end[1])

                lp.quad_to(c1, end)

        elif cmd_lower == "a":
            coords = get_nums(7)
            if coords:
                rx, ry = coords[0], coords[1]
                rot_deg = coords[2]
                large_arc = bool(coords[3])
                sweep = bool(coords[4])
                end = (coords[5], coords[6])

                if is_rel:
                    cur_x, cur_y = lp.pos
                    end = (cur_x + end[0], cur_y + end[1])

                params = _get_svg_arc_params(
                    lp.pos, rx, ry, rot_deg, large_arc, sweep, end
                )

                if params["type"] == "line":
                    lp.line_to(params["end"])
                elif params["type"] == "arc":
                    lp.arc(
                        params["rx"],
                        params["ry"],
                        params["start_angle"],
                        params["span_angle"],
                        rot_angle=params["rot_angle"],
                    )

    return lp


def _get_svg_arc_params(start, rx, ry, phi_deg, fA, fs, end):
    x1, y1 = start[:2]
    x2, y2 = end[:2]

    rx = abs(rx)
    ry = abs(ry)
    phi = radians(phi_deg)

    if rx == 0 or ry == 0:
        return {"type": "line", "end": end}

    if x1 == x2 and y1 == y2:
        return {"type": "none"}

    # Matrix for rotation
    cos_phi = cos(phi)
    sin_phi = sin(phi)

    # Step 1: Prime coords
    dx = (x1 - x2) / 2
    dy = (y1 - y2) / 2
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy

    # Radii check
    lamb = (x1p**2) / (rx**2) + (y1p**2) / (ry**2)
    if lamb > 1:
        s = sqrt(lamb)
        rx *= s
        ry *= s

    # Step 2: Center prime
    sign = -1 if fA == fs else 1
    num = (rx**2 * ry**2) - (rx**2 * y1p**2) - (ry**2 * x1p**2)
    den = (rx**2 * y1p**2) + (ry**2 * x1p**2)
    # precision check
    if abs(num) < 1e-9:
        num = 0
    if abs(den) < 1e-9:
        coef = 0
    else:
        coef = sign * sqrt(max(0, num / den))

    cxp = coef * (rx * y1p / ry)
    cyp = coef * (-ry * x1p / rx)

    # Step 3: Center (not strictly needed for params unless we want to debug,
    # but the angles are calculated relative to prime center)

    # Step 4: Angles
    def vector_angle(ux, uy, vx, vy):
        sign = 1 if (ux * vy - uy * vx) >= 0 else -1
        dot = ux * vx + uy * vy
        mag = sqrt(ux**2 + uy**2) * sqrt(vx**2 + vy**2)
        if mag == 0:
            return 0
        val = max(-1, min(1, dot / mag))
        return sign * acos(val)

    # start vector
    ux = (x1p - cxp) / rx
    uy = (y1p - cyp) / ry
    theta1 = vector_angle(1, 0, ux, uy)

    # dtheta
    vx = (-x1p - cxp) / rx
    vy = (-y1p - cyp) / ry
    dtheta = vector_angle(ux, uy, vx, vy)

    if not fs and dtheta > 0:
        dtheta -= 2 * pi
    elif fs and dtheta < 0:
        dtheta += 2 * pi

    return {
        "type": "arc",
        "rx": rx,
        "ry": ry,
        "start_angle": theta1,
        "span_angle": dtheta,
        "rot_angle": phi,
    }


def shape_to_path(shape: Shape) -> LinPath:
    """Given a Shape instance returns the equivalent LinPath object."""
    path = LinPath()
    path.move_to(shape[0])
    for vert in shape.vertices[1:]:
        path.line_to(vert)

    if shape.closed:
        path.close()

    return path


def batch_to_path(batch: Batch) -> LinPath:
    """Given a Batch instance returns the equivalent LinPath object."""
    shapes = batch.all_shapes
    path = LinPath()
    path.move_to(shapes[0][0])
    for i, shape in enumerate(shapes):
        if i > 0:
            path.move_to(shape[0])
        for vert in shape.vertices[1:]:
            path.line_to(vert)
        if shape.closed:
            path.line_to(shape[0])

    return path

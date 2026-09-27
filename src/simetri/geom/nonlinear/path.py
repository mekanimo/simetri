"""Path builders for polylines, Beziers, arcs, and related operations.

``Path2D`` (alias ``LinPath``) is a ``Group`` that accumulates drawing
operations from a turtle-like pen state.

**Examples**

```python
import simetri.graphics as sg
p = sg.Path2D(start=(0, 0), angle=0)
_ = p.line_to((10, 0)).turn(sg.pi / 2).forward(10).close()
```
"""

import re
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from math import acos, atan2, cos, degrees, isclose, pi, radians, sin, sqrt
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from ...base.all_enums import (
    Anchor,
    FillMode,
    InPlace,
    LineCap,
    LineJoin,
    TransformationType,
    Types,
    get_enum_value,
)
from ...base.all_enums import PathOperation as PathOps
from ...base.common import PointType
from ...base.common_style import CommonStyle
from ...base.core import _next_xform_matrix, _Targets
from ...coloring.colors import Color
from ...config.settings import runtime_defaults
from ...group.batch import Group
from ...shapes.shape import Shape
from ..affine import rotation_matrix, translation_matrix
from ..bbox import BoundingBox, bounding_box
from ..geom_utils import close_points_square
from ..geometry import (
    normalize_angle,
    polar_to_cartesian,
    positive_angle,
)
from ..points.point_utils import distance
from ..homogenize import homogenize
from ..polygons.polygon import polygon_area
from ..segments.line_utils import (
    extended_line,
    line_angle,
    line_by_point_angle_length,
)
from .bezier import Bezier
from .ellipse import (
    ellipse_tangent,
    elliptic_arc_points,
)
from .hobby import hobby_shape
from .sine import sine_points

# Path operations whose objects are dense samples; labels use endpoints only.
_CURVE_PATH_OPS = frozenset(
    (
        PathOps.ARC,
        PathOps.ARC_TO,
        PathOps.BLEND_ARC,
        PathOps.BLEND_CUBIC,
        PathOps.BLEND_QUAD,
        PathOps.BLEND_SINE,
        PathOps.CUBIC_TO,
        PathOps.HOBBY_TO,
        PathOps.QUAD_TO,
        PathOps.SINE,
    )
)
_ARC_CODE_PATH_OPS = frozenset((PathOps.ARC, PathOps.ARC_TO, PathOps.BLEND_ARC))
_ARC_PATH_OPS = frozenset((PathOps.ARC, PathOps.BLEND_ARC))
_BEZIER_PATH_OPS = frozenset((PathOps.CUBIC_TO, PathOps.QUAD_TO))
_CUBIC_PATH_OPS = frozenset((PathOps.BLEND_CUBIC, PathOps.CUBIC_TO))
_LINE_PATH_OPS = frozenset(
    (
        PathOps.FORWARD,
        PathOps.H_LINE_TO,
        PathOps.LINE_TO,
        PathOps.R_H_LINE,
        PathOps.R_LINE,
        PathOps.R_V_LINE,
        PathOps.V_LINE_TO,
    )
)
_LINE_TO_FORWARD_OPS = frozenset((PathOps.FORWARD, PathOps.LINE_TO))
_MOVE_PATH_OPS = frozenset((PathOps.MOVE_TO, PathOps.R_MOVE))
_POLYLINE_PATH_OPS = frozenset((PathOps.HOBBY_TO, PathOps.SEGMENTS))
_QUAD_PATH_OPS = frozenset((PathOps.BLEND_QUAD, PathOps.QUAD_TO))
_SINE_PATH_OPS = frozenset((PathOps.BLEND_SINE, PathOps.SINE))
_BEZIER_BLEND_PATH_OPS = _CUBIC_PATH_OPS | _QUAD_PATH_OPS
_HEADING_FROM_DATA_OPS = _ARC_PATH_OPS | _SINE_PATH_OPS
_OPEN_SUBPATH_OPS = (
    _ARC_PATH_OPS
    | _BEZIER_PATH_OPS
    | _LINE_PATH_OPS
    | _SINE_PATH_OPS
    | frozenset((PathOps.SEGMENTS,))
)

array = np.array


@dataclass
class Operation:
    """One recorded path operation (move, line, curve, and so on).

    Attributes:
        subtype: Path operation kind (from ``PathOperation`` / ``Types``).
        data: Payload for the operation (points, radii, and so on).
        name: Optional label.
        type: Always ``Types.PATH_OPERATION`` after init.

    Examples:
        >>> import simetri.graphics as sg
        >>> op = sg.Operation(sg.PathOps.LINE_TO, ((0, 0), (10, 0)))
        >>> op.type.name
        'PATH_OPERATION'
"""

    subtype: Types
    data: tuple
    name: str = ""

    def __post_init__(self) -> None:
        """Set ``type`` to ``Types.PATH_OPERATION``."""
        self.type = Types.PATH_OPERATION


class Path2D(Group, CommonStyle):
    """Constructive 2D path with a pen position and heading.

    Operations such as ``line_to``, ``cubic_to``, and ``arc`` append geometry
    while updating ``pos`` and ``angle``. The path is also a ``Group`` of
    ``Shape`` segments for transforms and drawing.

    ``color`` sets both ``line_color`` and ``fill_color``. ``alpha`` sets both
    ``line_alpha`` and ``fill_alpha``. Reading an unset ``line_color`` /
    ``fill_color`` / ``line_alpha`` / ``fill_alpha`` returns the matching
    configured default. Style color/alpha and ``copy_style`` come from
    ``CommonStyle``.

    Attributes:
        pos: Current pen position.
        start: Path start point.
        angle: Current heading in radians.
        operations: List of ``Operation`` records.
        subtype: ``Types.PATH2D``.

    Examples:
        >>> import simetri.graphics as sg
        >>> path = sg.Path2D((0, 0), angle=0)
        >>> _ = path.forward(10).turn(sg.pi / 2).forward(5)
        >>> round(path.pos[0], 6), round(path.pos[1], 6)
        (10.0, 5.0)
"""

    # Group.__setattr__ adds a frame above the color/alpha property setters.
    _style_warning_stacklevel: int = 4

    def __init__(
        self,
        start: PointType = (0, 0),
        angle: float = pi / 2,
        fill: bool = True,
        stroke: bool = True,
        alpha: float | None = None,
        color: Color | None = None,
        draw_double: bool = False,
        draw_fillets: bool = False,
        draw_markers: bool = False,
        even_odd: bool | None = None,
        back_style: Any = None,
        double_distance: float | None = None,
        double_color: Color | None = None,
        fill_alpha: float | None = None,
        fill_color: Color | None = None,
        fill_mode: FillMode = FillMode.NONZERO,
        fillet_radius: float | None = None,
        gradient: Any = None,
        line_alpha: float | None = None,
        line_cap: LineCap = LineCap.BUTT,
        line_color: Color | None = None,
        line_dash_array: Any = None,
        line_dash_phase: float | None = None,
        line_join: LineJoin = LineJoin.MITER,
        line_miter_limit: float | None = None,
        line_width: float | None = None,
        marker_alpha: float | None = None,
        marker_color: Color | None = None,
        marker_radius: float | None = None,
        marker_shape: Any = None,
        marker_size: float | None = None,
        marker_type: Any = None,
        markers_only: bool | None = None,
        smooth: bool | None = None,
    ) -> None:
        """Initialize a Path2D.

        Args:
            start: Starting pen position. Defaults to ``(0, 0)``.
            angle: Initial heading in radians. Defaults to upward (``pi/2``).
            fill: Whether to fill when drawn. Defaults to True.
            stroke: Whether to stroke when drawn. Defaults to True.
            alpha: Overall opacity override.
            color: Convenience color applied to stroke/fill when set.
            draw_double: Draw a double stroke if True.
            draw_fillets: Draw fillets at corners if True.
            draw_markers: Draw markers along the path if True.
            even_odd: If True, use even-odd fill rule; ``None`` uses
                ``runtime_defaults["even_odd"]`` at draw time.
            back_style: Background style.
            double_distance: Spacing for double stroke.
            double_color: Color for the second stroke.
            fill_alpha: Fill opacity. Unset values use ``runtime_defaults['fill_alpha']``.
            fill_color: Fill color. Unset values use ``runtime_defaults['fill_color']``.
            fill_mode: Fill rule (``FillMode``). Defaults to non-zero winding.
            fillet_radius: Fillet radius when fillets are enabled.
            gradient: Optional fill gradient.
            line_alpha: Stroke opacity. Unset values use ``runtime_defaults['line_alpha']``.
            line_cap: Stroke line cap. Defaults to butt.
            line_color: Stroke color. Unset values use ``runtime_defaults['line_color']``.
            line_dash_array: Dash pattern.
            line_dash_phase: Dash phase offset.
            line_join: Stroke line join. Defaults to miter.
            line_miter_limit: Miter limit for joins.
            line_width: Stroke width. Unset values use ``runtime_defaults['line_width']``.
            marker_*: Marker drawing options (same as Shape).
            markers_only: If True, draw markers without the path.
            smooth: Prefer smooth curve rendering when applicable.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0), angle=sg.pi / 2, line_width=2)
            >>> p.angle == sg.pi / 2
            True
"""

        self.pos = start
        self.start = start
        self.angle = angle  # heading angle
        self.operations = []
        self.objects = []
        super().__init__()
        self.subtype = Types.PATH2D
        self.cur_shape = Shape([start])
        self.append(self.cur_shape)
        self.rc = self.r_coord  # alias for r_coord
        self.rp = self.r_polar  # alias for rel_polar
        self.handles = []
        self.stack = deque()

        self.closed = False
        self._init_from_style_kwargs(
            {
                "color": color,
                "alpha": alpha,
                "line_color": line_color,
                "fill_color": fill_color,
                "line_alpha": line_alpha,
                "fill_alpha": fill_alpha,
                "line_width": line_width,
                "fill": fill,
                "stroke": stroke,
                "line_dash_array": line_dash_array,
                "line_dash_phase": line_dash_phase,
                "line_cap": line_cap,
                "line_join": line_join,
                "line_miter_limit": line_miter_limit,
                "smooth": smooth,
                "back_style": back_style,
                "draw_double": draw_double,
                "draw_fillets": draw_fillets,
                "double_distance": double_distance,
                "double_color": double_color,
                "fill_mode": fill_mode,
                "fillet_radius": fillet_radius,
                "gradient": gradient,
                "draw_markers": draw_markers,
                "even_odd": even_odd,
                "marker_type": marker_type,
                "marker_size": marker_size,
                "marker_radius": marker_radius,
                "marker_alpha": marker_alpha,
                "marker_color": marker_color,
                "marker_shape": marker_shape,
                "markers_only": markers_only,
            }
        )
        self.visible = True

    def __bool__(self) -> bool:
        """Return True if the path has recorded operations.

        A path may still be truthy when the group has no drawable elements yet.

        Returns:
            True if ``operations`` is non-empty.

        Examples:
            >>> import simetri.graphics as sg
            >>> bool(sg.Path2D((0, 0)))
            False
            >>> bool(sg.Path2D((0, 0)).line_to((1, 0)))
            True
"""
        return bool(self.operations)

    def _create_object(self) -> None:
        """Build geometry for the most recently appended operation."""
        PO = PathOps
        op = self.operations[-1]
        op_type = op.subtype
        data = op.data
        if op_type in _OPEN_SUBPATH_OPS and self.cur_shape.closed:
            self.cur_shape = Shape([self.pos])
            self.append(self.cur_shape)
        if op_type in _MOVE_PATH_OPS:
            self.cur_shape = Shape([data])
            self.append(self.cur_shape)
            self.objects.append(None)
        elif op_type in _LINE_PATH_OPS:
            self.objects.append(Shape(data))
            self.cur_shape.append(data[1])
        elif op_type == PO.SEGMENTS:
            self.objects.append(Shape(data[1]))
            self.cur_shape.extend(data[1])
        elif op_type in _SINE_PATH_OPS:
            self.objects.append(Shape(data[0]))
            self.cur_shape.extend(data[0])
        elif op_type in _BEZIER_PATH_OPS:
            n_points = runtime_defaults["n_bezier_points"]
            curve = Bezier(data, n_points=n_points)
            self.objects.append(curve)
            self.cur_shape.extend(curve.vertices[1:])
            if op_type == PO.CUBIC_TO:
                self.handles.extend([(data[0], data[1]), (data[2], data[3])])
            else:
                self.handles.append((data[0], data[1]))
                self.handles.append((data[1], data[2]))
        elif op_type == PO.HOBBY_TO:
            n_points = runtime_defaults["n_hobby_points"]
            curve = hobby_shape(data[1], n_points=n_points)
            self.objects.append(Shape(curve.vertices))
        elif op_type in _ARC_PATH_OPS:
            self.objects.append(Shape(data[-1]))
            self.cur_shape.extend(data[-1][1:])
        elif op_type == PO.CLOSE:
            self.cur_shape.closed = True
            self.objects.append(None)
        else:
            raise ValueError(f"Invalid operation type: {op_type}")

    def copy(self) -> "Path2D":
        """Return a deep-enough copy of the path for independent editing.

        Returns:
            Path2D: Copied path with cloned elements and operations.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).line_to((10, 0))
            >>> q = p.copy()
            >>> _ = q.line_to((10, 5))
            >>> len(p.operations), len(q.operations)
            (1, 2)
"""

        new_path = Path2D(start=self.start)
        cur_shape_index = None
        for index, element in enumerate(self.elements):
            if element is self.cur_shape:
                cur_shape_index = index
                break

        new_path.pos = self.pos
        new_path.angle = self.angle
        new_path.operations = (
            self.operations.copy()
        )  # shallow-copy of list is fine if Operations are treated as immutable
        new_path.elements = [element.copy() for element in self.elements]
        new_path.objects = []
        for obj in self.objects:
            if obj is not None:
                new_path.objects.append(obj.copy())
            else:
                new_path.objects.append(None)
        new_path.even_odd = self.even_odd
        new_path.closed = self.closed
        if cur_shape_index is not None:
            new_path.cur_shape = new_path.elements[cur_shape_index]
        else:
            new_path.cur_shape = self.cur_shape.copy()
        new_path.handles = list(self.handles)
        new_path.stack = deque(self.stack)
        new_path.copy_style(self)
        return new_path

    def _add(
        self,
        pos: PointType,
        op: PathOps,
        data: object,
        pnt2: PointType | None = None,
        **kwargs: object,
    ) -> None:
        """Add an operation and update pen state.

        Args:
            pos: New pen position after the operation.
            op: Path operation kind.
            data: Operation payload.
            pnt2: Optional point used to update heading. Defaults to None.
            **kwargs: ``name`` labels the operation. Other keys are style
                overrides applied to the segment.
        """
        self.operations.append(Operation(op, data))
        if op in _HEADING_FROM_DATA_OPS:
            self.angle = data[1]
        else:
            if pnt2 is not None:
                self.angle = line_angle(pnt2, pos)
            else:
                self.angle = line_angle(self.pos, pos)
        self._create_object()
        style = dict(kwargs)
        if "name" in style:
            setattr(self, style["name"], self.operations[-1])
            del style["name"]
        targets = []
        segment = self.objects[-1]
        if segment is not None:
            targets.append(segment)
        if self.cur_shape is not None and self.cur_shape is not segment:
            targets.append(self.cur_shape)
        for target in targets:
            for key, value in style.items():
                setattr(target, key, value)
        list(pos)[:2]
        self.pos = pos

    @property
    def all_vertices(self) -> list:
        """Return all vertices of the path after applying transforms.

        Returns:
            list: Flattened vertex list.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).line_to((10, 0)).line_to((10, 5))
            >>> [[round(c, 6) for c in q[:2]] for q in p.all_vertices]
            [[0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 5.0]]
"""
        all_vertices = []
        for obj in self.objects:
            if obj is not None:
                all_vertices.extend(obj.vertices)

        return all_vertices

    @property
    def all_elements(self) -> list:
        """Return the transformed geometric objects that make up the path.

        Returns:
            list: Drawable elements in the path group.
        """
        return [obj for obj in self.objects if obj is not None]

    @property
    def b_box(self) -> BoundingBox:
        """Return the axis-aligned bounding box of the path.

        Returns:
            Bounding box of all path vertices.

        Examples:
            >>> import simetri.graphics as sg
            >>> box = sg.Path2D((0, 0)).line_to((10, 5)).b_box
            >>> box.width, box.height
            (10.0, 5.0)
"""

        return bounding_box(self.all_vertices)

    def push(self) -> None:
        """Push the current pen position and heading onto the stack.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0), angle=0)
            >>> p.push()
            >>> _ = p.forward(10)
            >>> p.pop()
            >>> p.pos
            (0, 0)
"""
        self.stack.append((self.pos, self.angle))

    def pop(self) -> None:
        """Restore the pen position and heading from the stack.

        Starts a new subpath at the restored position so later drawing is
        not connected to the segment that was drawn after ``push``. Does
        nothing if the stack is empty.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0), angle=0)
            >>> p.push()
            >>> _ = p.forward(5).turn(sg.pi / 2)
            >>> p.pop()
            >>> p.angle
            0
            >>> p.pos
            (0, 0)
"""
        if not self.stack:
            return
        pos, angle = self.stack.pop()
        self.operations.append(Operation(PathOps.MOVE_TO, pos))
        self._create_object()
        self.pos = pos
        self.angle = angle

    def r_coord(self, dx: float, dy: float) -> PointType:
        """Map local offsets into world coordinates relative to the pen.

        The local y-axis is aligned with the current heading.

        Args:
            dx: Offset along the local x-axis.
            dy: Offset along the local y-axis (heading direction).

        Returns:
            PointType: Absolute point.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0), angle=sg.pi / 2)
            >>> x, y = p.r_coord(0, 10)
            >>> round(x, 6), round(y, 6)
            (0.0, 10.0)
"""
        x, y = self.pos[:2]
        theta = self.angle - pi / 2
        x1 = dx * cos(theta) - dy * sin(theta) + x
        y1 = dx * sin(theta) + dy * cos(theta) + y

        return x1, y1

    def r_polar(self, r: float, angle: float) -> PointType:
        """Return an absolute point from polar offsets relative to the pen.

        Angle ``0`` is aligned with the current heading.

        Args:
            r: Radius.
            angle: Angle in radians relative to the heading.

        Returns:
            PointType: Absolute point.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0), angle=sg.pi / 2)
            >>> x, y = p.r_polar(10, 0)
            >>> round(x, 6), round(y, 6)
            (10.0, 0.0)
"""
        x, y = polar_to_cartesian(r, angle + self.angle - pi / 2)[:2]
        x1, y1 = self.pos[:2]

        return x1 + x, y1 + y

    def line_to(self, point: PointType, **kwargs: object) -> Self:
        """Draw a straight segment from the current position to ``point``.

        Args:
            point: Absolute end point.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path (for chaining).

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).line_to((10, 0))
            >>> p.pos
            (10, 0)
"""
        self._add(point, PathOps.LINE_TO, (self.pos, point), **kwargs)

        return self

    def forward(self, length: float, **kwargs: object) -> Self:
        """Advance the pen ``length`` units along the current heading.

        Args:
            length: Distance to travel.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path (for chaining).

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0), angle=0).forward(10)
            >>> p.pos
            (10.0, 0.0)
"""

        x, y = line_by_point_angle_length(self.pos, self.angle, length)[1][:2]
        self._add((x, y), PathOps.FORWARD, (self.pos, (x, y)), **kwargs)

        return self

    def orient(self, angle: float) -> Self:
        """Set the absolute heading angle.

        Args:
            angle: New heading in radians.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).orient(sg.pi)
            >>> p.angle == sg.pi
            True
"""
        self.angle = angle

        return self

    def turn(self, angle: float, distance: float = 0) -> Self:
        """Turn by ``angle`` radians, optionally moving ``distance`` forward.

        Args:
            angle: Heading increment in radians.
            distance: Optional forward distance after the turn. Defaults to 0.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0), angle=0).turn(sg.pi / 2, distance=10)
            >>> round(p.pos[0], 6), round(p.pos[1], 6)
            (0.0, 10.0)
"""

        self.angle += angle
        if distance != 0:
            self.forward(distance)

        return self

    def move(
        self,
        pos: PointType,
        anchor: Anchor = Anchor.CENTER,
        **kwargs: object,
    ) -> Self:
        """Translate the whole path so a bbox anchor sits at ``pos``.

        Args:
            pos: Target location for the chosen anchor.
            anchor: Bounding-box anchor (default ``Anchor.CENTER``).
            **kwargs: Extra attributes set on the path before transforming.

        Returns:
            Self or Group: Transformed path (or group when repetitions apply).

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).line_to((10, 0))
            >>> _ = p.move((50, 50))
            >>> abs(p.b_box.center[0] - 50) < 1e-6
            True
"""
        x, y = pos[:2]
        anchor = get_enum_value(Anchor, anchor)
        x1, y1 = getattr(self.b_box, anchor)
        transform = translation_matrix(x - x1, y - y1)
        for k, v in kwargs.items():
            setattr(self, k, v)
        res = self._update(transform, reps=0)

        return res

    def move_to(self, point: PointType, **kwargs: object) -> Self:
        """Lift the pen and start a new subpath at ``point``.

        Args:
            point: Absolute destination.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).line_to((5, 0)).move_to((0, 5))
            >>> p.pos
            (0, 5)
"""
        self._add(point, PathOps.MOVE_TO, point)

        return self

    def r_line(self, dx: float, dy: float, **kwargs: object) -> Self:
        """Draw a line using relative offsets from the current position.

        Args:
            dx: X offset.
            dy: Y offset.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((1, 1)).r_line(3, 4)
            >>> p.pos
            (4, 5)
"""
        point = self.pos[0] + dx, self.pos[1] + dy
        self._add(point, PathOps.R_LINE, (self.pos, point), **kwargs)

        return self

    def r_move(self, dx: float = 0, dy: float = 0, **kwargs: object) -> Self:
        """Move the pen by relative offsets without drawing.

        Args:
            dx: X offset. Defaults to 0.
            dy: Y offset. Defaults to 0.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((1, 1)).r_move(2, 3)
            >>> p.pos
            (3, 4)
"""
        x, y = self.pos[:2]
        point = (x + dx, y + dy)
        self._add(point, PathOps.R_MOVE, point, **kwargs)
        return self

    def h_line_to(self, x: float, **kwargs: object) -> Self:
        """Draw a horizontal line to absolute x, keeping the current y.

        Args:
            x: Absolute x coordinate of the end point.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 2)).h_line_to(8)
            >>> p.pos
            (8, 2)
"""
        y = self.pos[1]
        self._add((x, y), PathOps.H_LINE_TO, (self.pos, (x, y)), **kwargs)
        return self

    def r_h_line(self, length: float, **kwargs: object) -> Self:
        """Draw a horizontal line of the given length.

        Args:
            length: Signed horizontal distance.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).r_h_line(5)  # doctest: +SKIP
            >>> p.pos  # doctest: +SKIP
            (5, 0)
"""
        x, y = self.pos[0] + length, self.pos[1]
        self._add((x, y), PathOps.R_H_LINE, (self.pos, (x, y)), **kwargs)
        return self

    def v_line_to(self, y: float, **kwargs: object) -> Self:
        """Draw a vertical line to absolute y, keeping the current x.

        Args:
            y: Absolute y coordinate of the end point.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((3, 0)).v_line_to(7)
            >>> p.pos
            (3, 7)
"""
        x = self.pos[0]
        self._add((x, y), PathOps.V_LINE_TO, (self.pos, (x, y)), **kwargs)
        return self

    def r_v_line(self, length: float, **kwargs: object) -> Self:
        """Draw a vertical line of the given length.

        Args:
            length: Signed vertical distance.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).r_v_line(4)  # doctest: +SKIP
            >>> p.pos  # doctest: +SKIP
            (0, 4)
"""
        x, y = self.pos[0], self.pos[1] + length
        self._add((x, y), PathOps.R_V_LINE, (self.pos, (x, y)), **kwargs)
        return self

    def segments(
        self, points: Sequence[PointType], **kwargs: object
    ) -> Self:
        """Append polyline segments through absolute ``points``.

        Args:
            points: Sequence of absolute points.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).segments([(5, 0), (5, 5)])
            >>> p.pos
            (5, 5)
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
        **kwargs: object,
    ) -> Self:
        """Append a cubic Bézier from the current position to ``end``.

        Args:
            control1: First control point.
            control2: Second control point.
            end: Curve end point.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).cubic_to((1, 2), (3, 2), (4, 0))
            >>> p.pos
            (4, 0)
"""
        self._add(
            end,
            PathOps.CUBIC_TO,
            (self.pos, control1, control2, end),
            pnt2=control2,
            **kwargs,
        )
        return self

    def r_cubic_to(
        self,
        r_control1: PointType,
        r_control2: PointType,
        r_end: PointType,
        **kwargs: object,
    ) -> Self:
        """Append a cubic Bézier with relative control points and end (SVG ``c``).

        Args:
            r_control1: First control offset ``(dx, dy)`` from the pen.
            r_control2: Second control offset ``(dx, dy)`` from the pen.
            r_end: End offset ``(dx, dy)`` from the pen.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).r_cubic_to((1, 2), (3, 2), (4, 0))
            >>> p.pos
            (4, 0)
"""
        cur_x, cur_y = self.pos
        control1 = (cur_x + r_control1[0], cur_y + r_control1[1])
        control2 = (cur_x + r_control2[0], cur_y + r_control2[1])
        end = (cur_x + r_end[0], cur_y + r_end[1])
        return self.cubic_to(control1, control2, end, **kwargs)

    def r_segments(
        self, r_points: Sequence[PointType], **kwargs: object
    ) -> Self:
        """Append polyline segments using successive ``(dx, dy)`` offsets.

        Args:
            r_points: Sequence of relative offsets.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).r_segments([(5, 0), (0, 5)])
            >>> p.pos
            (5, 5)
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

    def hobby_to(
        self, points: Sequence[PointType], **kwargs: object
    ) -> Self:
        """Append a Hobby smooth curve through ``points``.

        Args:
            points: Curve points after the current position.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).hobby_to([(10, 5), (20, 0)])  # doctest: +SKIP
            >>> p.pos  # doctest: +SKIP
            (20, 0)
"""
        self._add(points[-1], PathOps.HOBBY_TO, (self.pos, points), **kwargs)
        return self

    def quad_to(
        self,
        control: PointType,
        end: PointType,
        *args: object,
        **kwargs: object,
    ) -> Self:
        """Append a quadratic Bézier to ``end`` with control point ``control``.

        Additional ``*args`` entries can continue the curve as blended quads.

        Args:
            control: Control point.
            end: Curve end point.
            *args: Either ``(length, end)`` or ``(control, end)`` pairs.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Raises:
            ValueError: If an ``*args`` entry does not have exactly two items.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).quad_to((5, 5), (10, 0))  # doctest: +SKIP
            >>> p.pos  # doctest: +SKIP
            (10, 0)
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

    def r_quad_to(
        self,
        r_control: PointType,
        r_end: PointType,
        **kwargs: object,
    ) -> Self:
        """Append a quadratic Bézier with relative control and end (SVG ``q``).

        Args:
            r_control: Control offset ``(dx, dy)`` from the pen.
            r_end: End offset ``(dx, dy)`` from the pen.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).r_quad_to((5, 5), (10, 0))
            >>> p.pos
            (10, 0)
"""
        cur_x, cur_y = self.pos
        control = (cur_x + r_control[0], cur_y + r_control[1])
        end = (cur_x + r_end[0], cur_y + r_end[1])
        return self.quad_to(control, end, **kwargs)

    def mirror_cubic_to(
        self, control2: PointType, end: PointType, **kwargs: object
    ) -> Self:
        """Append a smooth cubic Bézier (SVG ``S``).

        Mirrors the previous second control point across the current position
        to obtain the first control point.

        Args:
            control2: Second control point.
            end: Curve end point.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = (
            ... sg.Path2D((0, 0))
            ... .cubic_to((1, 2), (3, 2), (4, 0))
            ... .mirror_cubic_to((5, -2), (8, 0))
            ... )
            >>> p.pos
            (8, 0)
"""
        # Get previous control point from last operation if it was a cubic
        prev_c2 = self.pos
        last_op = self.operations[-1] if self.operations else None
        if last_op and last_op.subtype in _CUBIC_PATH_OPS:
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
        self, r_control2: PointType, r_end: PointType, **kwargs: object
    ) -> Self:
        """Append a relative smooth cubic Bézier (SVG ``s``).

        Args:
            r_control2: Relative second control point ``(dx, dy)``.
            r_end: Relative end point ``(dx, dy)``.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = (
            ... sg.Path2D((0, 0))
            ... .cubic_to((1, 2), (3, 2), (4, 0))
            ... .r_mirror_cubic_to((1, -2), (4, 0))
            ... )
            >>> p.pos
            (8, 0)
"""
        cur_x, cur_y = self.pos
        control2 = (cur_x + r_control2[0], cur_y + r_control2[1])
        end = (cur_x + r_end[0], cur_y + r_end[1])
        return self.mirror_cubic_to(control2, end, **kwargs)

    def mirror_quad_to(self, end: PointType, **kwargs: object) -> Self:
        """Append a smooth quadratic Bézier (SVG ``T``).

        Args:
            end: Curve end point.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).quad_to((5, 5), (10, 0)).mirror_quad_to((20, 0))  # doctest: +SKIP
            >>> p.pos  # doctest: +SKIP
            (20, 0)
"""
        # Get previous control point from last operation if it was a quad
        prev_c1 = self.pos
        last_op = self.operations[-1] if self.operations else None
        if last_op and last_op.subtype in _QUAD_PATH_OPS:
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

    def r_mirror_quad_to(self, r_end: PointType, **kwargs: object) -> Self:
        """Append a relative smooth quadratic Bézier (SVG ``t``).

        Args:
            r_end: Relative end point ``(dx, dy)``.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).quad_to((5, 5), (10, 0)).r_mirror_quad_to((10, 0))  # doctest: +SKIP
            >>> p.pos  # doctest: +SKIP
            (20, 0)
"""
        cur_x, cur_y = self.pos
        end = (cur_x + r_end[0], cur_y + r_end[1])
        return self.mirror_quad_to(end, **kwargs)

    def blend_cubic(
        self,
        control1_length: float,
        control2: PointType,
        end: PointType,
        **kwargs: object,
    ) -> Self:
        """Append a cubic Bézier whose first control lies along the heading.

        Args:
            control1_length: Distance from the pen to the first control point.
            control2: Second control point.
            end: Curve end point.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0), angle=0).blend_cubic(5, (8, 4), (12, 0))
            >>> p.pos
            (12, 0)
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

    def blend_quad(
        self, control_length: float, end: PointType, **kwargs: object
    ) -> Self:
        """Append a quadratic Bézier whose control lies along the heading.

        Args:
            control_length: Distance from the pen to the control point.
            end: Curve end point.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0), angle=0).blend_quad(5, (10, 0))  # doctest: +SKIP
            >>> p.pos  # doctest: +SKIP
            (10, 0)
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
        n_points: int | None = None,
        **kwargs: object,
    ) -> Self:
        """Append an elliptic arc starting at the current pen position.

        The sign of ``span_angle`` selects the drawing direction.

        Args:
            radius_x: Ellipse half-width.
            radius_y: Ellipse half-height.
            start_angle: Arc start angle in radians.
            span_angle: Signed sweep in radians.
            rot_angle: Ellipse rotation in radians. Defaults to 0.
            n_points: Sample count; defaults to ``runtime_defaults['n_arc_points']``.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((10, 0), angle=0).arc(10, 10, 0, sg.pi / 2)
            >>> round(float(p.pos[0]), 5), round(float(p.pos[1]), 5)
            (0.0, 10.0)
"""
        rx = radius_x
        ry = radius_y
        start_angle = positive_angle(start_angle)
        clockwise = span_angle < 0
        if n_points is None:
            n_points = runtime_defaults["n_arc_points"]
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
            **kwargs,
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
        **kwargs: object,
    ) -> Self:
        """Append an SVG-style elliptical arc to ``end`` (SVG ``A``).

        Args:
            rx: Ellipse x-radius.
            ry: Ellipse y-radius.
            angle: Ellipse rotation in degrees.
            large_arc_flag: Use the large arc if True.
            sweep_flag: SVG sweep flag.
            end: Absolute end point.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).arc_to(10, 10, 0, False, True, (10, 10))
            >>> [round(float(c), 5) for c in p.pos[:2]]
            [10.0, 10.0]
            >>> [round(float(c), 5) for c in p.vertices[0][:2]], [round(float(c), 5) for c in p.vertices[-1][:2]]
            ([0.0, 0.0], [10.0, 10.0])
"""
        params = _get_svg_arc_params(
            self.pos, rx, ry, angle, large_arc_flag, sweep_flag, end
        )

        if params is not None and params[0] is PathOps.LINE_TO:
            self.line_to(params[1], **kwargs)
        elif params is not None and params[0] is PathOps.ARC:
            (
                _op,
                radius_x,
                radius_y,
                start_angle,
                span_angle,
                rot_angle,
            ) = params
            self.arc(
                radius_x,
                radius_y,
                start_angle,
                span_angle,
                rot_angle=rot_angle,
                **kwargs,
            )
        return self

    def r_arc_to(
        self,
        rx: float,
        ry: float,
        angle: float,
        large_arc_flag: bool,
        sweep_flag: bool,
        r_end: PointType,
        **kwargs: object,
    ) -> Self:
        """Append an SVG-style elliptical arc with a relative end (SVG ``a``).

        Radii, rotation, and flags match SVG ``a``; only ``r_end`` is an offset
        from the current pen.

        Args:
            rx: Ellipse x-radius.
            ry: Ellipse y-radius.
            angle: Ellipse rotation in degrees.
            large_arc_flag: Use the large arc if True.
            sweep_flag: SVG sweep flag.
            r_end: End offset ``(dx, dy)`` from the pen.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).r_arc_to(10, 10, 0, False, True, (10, 10))
            >>> round(float(p.pos[0]), 5), round(float(p.pos[1]), 5)
            (10.0, 10.0)
"""
        cur_x, cur_y = self.pos[:2]
        end = (cur_x + r_end[0], cur_y + r_end[1])
        return self.arc_to(
            rx, ry, angle, large_arc_flag, sweep_flag, end, **kwargs
        )

    def blend_arc(
        self,
        radius_x: float,
        radius_y: float,
        start_angle: float,
        span_angle: float,
        sharp: bool = False,
        n_points: int | None = None,
        **kwargs: object,
    ) -> Self:
        """Append an elliptic arc blended to the current heading.

        Args:
            radius_x: Ellipse half-width.
            radius_y: Ellipse half-height.
            start_angle: Arc start angle in radians.
            span_angle: Signed sweep in radians.
            sharp: Flip the blend orientation if True. Defaults to False.
            n_points: Sample count; defaults to ``runtime_defaults['n_arc_points']``.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0), angle=0).blend_arc(10, 10, 0, sg.pi / 2)
            >>> [round(float(c), 5) for c in p.pos[:2]]
            [10.0, 10.0]
            >>> [round(float(c), 5) for c in p.vertices[0][:2]], [round(float(c), 5) for c in p.vertices[-1][:2]]
            ([0.0, 0.0], [10.0, 10.0])
"""
        rx = radius_x
        ry = radius_y
        start_angle = positive_angle(start_angle)
        clockwise = span_angle < 0
        if n_points is None:
            n_points = runtime_defaults["n_arc_points"]
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
            kwargs,
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
        **kwargs: object,
    ) -> Self:
        """Append a sine-wave polyline from the current pen position.

        Args:
            period: Wavelength. Defaults to 40.
            amplitude: Wave amplitude. Defaults to 20.
            duration: Horizontal length of the wave. Defaults to 40.
            phase_angle: Phase offset in radians. Defaults to 0.
            rot_angle: Rotation of the wave in radians. Defaults to 0.
            damping: Exponential damping factor. Defaults to 0.
            n_points: Number of samples. Defaults to 100.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).sine(period=20, amplitude=5, duration=20, n_points=4)
            >>> [round(float(c), 6) or 0.0 for c in p.pos[:2]]
            [20.0, 0.0]
            >>> [[round(float(c), 6) or 0.0 for c in q[:2]] for q in p.vertices]
            [[0.0, 0.0], [6.666667, 4.330127], [13.333333, -4.330127], [20.0, 0.0]]
"""

        points = sine_points(
            period, amplitude, duration, n_points, phase_angle, damping
        )
        if rot_angle != 0:
            points = homogenize(points) @ rotation_matrix(rot_angle, points[0])
        points = homogenize(points) @ translation_matrix(*self.pos[:2])
        angle = line_angle(points[-2], points[-1])
        self._add(points[-1], PathOps.SINE, (points, angle), **kwargs)
        return self

    def blend_sine(
        self,
        period: float = 40,
        amplitude: float = 20,
        duration: float = 40,
        phase_angle: float = 0,
        damping: float = 0,
        n_points: int = 100,
        **kwargs: object,
    ) -> Self:
        """Append a sine wave rotated to match the current heading.

        Args:
            period: Wavelength. Defaults to 40.
            amplitude: Wave amplitude. Defaults to 20.
            duration: Horizontal length of the wave. Defaults to 40.
            phase_angle: Phase offset in radians. Defaults to 0.
            damping: Exponential damping factor. Defaults to 0.
            n_points: Number of samples. Defaults to 100.
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0), angle=sg.pi / 4).blend_sine(duration=20, n_points=4)
            >>> [round(float(c), 5) for c in p.pos[:2]]
            [14.14214, 14.14214]
            >>> [[round(float(c), 6) for c in q[:2]] for q in p.vertices]
            [[0.0, 0.0], [14.142136, 14.142136]]
"""

        points = sine_points(
            period, amplitude, duration, n_points, phase_angle, damping
        )
        start_angle = line_angle(points[0], points[1])
        rot_angle = self.angle - start_angle
        points = homogenize(points) @ rotation_matrix(rot_angle, points[0])
        points = homogenize(points) @ translation_matrix(*self.pos[:2])
        angle = line_angle(points[-2], points[-1])
        self._add(points[-1], PathOps.SINE, (points, angle), **kwargs)
        return self

    def close(self, **kwargs: object) -> Self:
        """Close the current subpath back to its start.

        Args:
            **kwargs: Style overrides applied to the segment. ``name`` labels the operation.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).line_to((10, 0)).line_to((10, 10)).close()
            >>> p.closed
            True
            >>> [[round(c, 6) for c in q[:2]] for q in p[-1].vertices]
            [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
"""
        self.closed = True
        self._add(self.pos, PathOps.CLOSE, None, **kwargs)
        return self

    @property
    def vertices(self) -> list[PointType]:
        """Return deduplicated vertices from the path objects.

        Returns:
            list: Path vertices in drawing order.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).line_to((5, 0)).line_to((5, 5))
            >>> [[round(c, 6) for c in q[:2]] for q in p.vertices]
            [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]
"""
        vertices = []
        last_vert = None
        abs_tol2 = runtime_defaults["abs_tol"] ** 2
        for obj in self.objects:
            if obj is not None and obj.vertices:
                obj_verts = obj.vertices
                if last_vert:
                    if close_points_square(last_vert, obj_verts[0], abs_tol2):
                        vertices.extend(obj_verts[1:])
                    else:
                        vertices.extend(obj_verts)
                else:
                    vertices.extend(obj_verts)
                last_vert = obj_verts[-1]

        return vertices

    def _label_vertices(self) -> list[PointType]:
        """Return vertices suitable for index / coordinate labels.

        Line and polyline segments keep all corners. Sampled curves (arcs,
        Beziers, Hobby, sine) contribute only their first and last points so
        dense ellipse samples are not labeled.

        Returns:
            list: Deduplicated landmark vertices in drawing order.
        """
        if len(self.objects) != len(self.operations):
            raise ValueError(
                "Path2D.objects and Path2D.operations length mismatch"
            )

        vertices = []
        last_vert = None
        abs_tol2 = runtime_defaults["abs_tol"] ** 2
        for obj, operation in zip(self.objects, self.operations):
            if obj is None or not obj.vertices:
                continue
            obj_verts = obj.vertices
            if operation.subtype in _CURVE_PATH_OPS and len(obj_verts) > 2:
                obj_verts = [obj_verts[0], obj_verts[-1]]
            if last_vert:
                if close_points_square(last_vert, obj_verts[0], abs_tol2):
                    vertices.extend(obj_verts[1:])
                else:
                    vertices.extend(obj_verts)
            else:
                vertices.extend(obj_verts)
            last_vert = obj_verts[-1]

        return vertices

    def set_style(
        self, name: str, value: object, **kwargs: object
    ) -> Self:
        """Record a style change in the path operation list.

        Args:
            name: Style attribute name.
            value: Style value.
            **kwargs: Extra style metadata.

        Returns:
            Self: This path.

        Examples:
            >>> import simetri.graphics as sg
            >>> p = sg.Path2D((0, 0)).set_style("line_width", 2)  # doctest: +SKIP
            >>> p.operations[-1][1][0]  # doctest: +SKIP
            'line_width'
"""
        self.operations.append((PathOps.STYLE, (name, value, kwargs)))
        return self

    def _update(
        self,
        xform_matrix: NDArray,
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
        xform_type: TransformationType = None,
    ) -> Group:
        """Apply a transform to this path, optionally repeating it.

        Args:
            xform_matrix: 3x3 affine matrix.
            reps: Extra copies to generate. Defaults to 0.
            take: Not supported; must be ``None``.
            incr: Increment applied between repetitions when ``reps > 0``.
            dyn_ref: Matrix factory built by the transform method when
                dynamic references are in use. Defaults to None.
            merge: If True and ``reps > 0``, merge resulting shapes.
            xform_type: Transform kind used with ``incr``.

        Returns:
            Group: ``self`` when ``reps == 0``, otherwise a group of copies.

        Raises:
            ValueError: If ``take`` is set.
        """
        if take is not None:
            raise ValueError(
                "Path2D._update does not support take=; transform the whole path."
            )
        if reps == 0:
            original_pos = self.pos
            direction_point = (
                original_pos[0] + cos(self.angle),
                original_pos[1] + sin(self.angle),
            )
            self.start = _transform_path_point(self.start, xform_matrix)
            self.pos = _transform_path_point(self.pos, xform_matrix)
            self.angle = line_angle(
                _transform_path_point(original_pos, xform_matrix),
                _transform_path_point(direction_point, xform_matrix),
            )
            self.operations = [
                _transform_path_operation(operation, xform_matrix)
                for operation in self.operations
            ]
            self.handles = [
                _transform_path_points(handle, xform_matrix)
                for handle in self.handles
            ]
            for obj in self.objects:
                if obj is not None:
                    obj._update(xform_matrix)

            # Check this!!!!
            for element in self.elements:
                if element is not None and element.type == Types.SHAPE:
                    element._update(xform_matrix)
            res = self
        else:
            paths = [self]
            path = self
            if dyn_ref:
                pattern = Group()
                pattern.elements = paths
                targets = _Targets(self, pattern)
            else:
                targets = None
            for i in range(reps):
                path = path.copy()
                if targets is not None:
                    targets.active = path
                xform_matrix = _next_xform_matrix(
                    xform_matrix, xform_type, incr, dyn_ref, targets, i
                )
                path._update(xform_matrix)
                paths.append(path)
            res = Group(paths)
        if merge and reps > 0:
            res = res.merge_shapes()
        return res


def _transform_path_point(point: PointType, xform_matrix: NDArray) -> PointType:
    """Return ``point`` transformed by ``xform_matrix``."""
    transformed_point = homogenize([point]) @ xform_matrix
    return tuple(transformed_point[0][:2])


def _transform_path_points(
    points: Sequence[PointType], xform_matrix: NDArray
) -> list[PointType]:
    """Return ``points`` transformed by ``xform_matrix``."""
    transformed_points = homogenize(points) @ xform_matrix
    return [tuple(point[:2]) for point in transformed_points.tolist()]


def _transform_arc_data(data: tuple, xform_matrix: NDArray) -> tuple:
    """Return arc operation data transformed by ``xform_matrix``."""
    _, _, rx, ry, start_angle, span_angle, rot_angle, points = data
    transformed_points = _transform_path_points(points, xform_matrix)
    end_point = tuple(transformed_points[-1])

    linear_part = np.array(
        [
            [xform_matrix[0, 0], xform_matrix[1, 0]],
            [xform_matrix[0, 1], xform_matrix[1, 1]],
        ],
        dtype=float,
    )
    rotation = np.array(
        [
            [cos(rot_angle), -sin(rot_angle)],
            [sin(rot_angle), cos(rot_angle)],
        ],
        dtype=float,
    )
    ellipse_matrix = linear_part @ rotation @ np.diag([rx, ry])
    vectors, singular_values, _ = np.linalg.svd(ellipse_matrix)
    if np.linalg.det(vectors) < 0:
        vectors[:, -1] *= -1

    new_rot_angle = atan2(vectors[1, 0], vectors[0, 0])
    new_rx, new_ry = singular_values[:2]
    new_span_angle = span_angle
    if np.linalg.det(linear_part) < 0:
        new_span_angle = -new_span_angle
    tangent_angle = line_angle(transformed_points[-2], transformed_points[-1])

    return (
        end_point,
        tangent_angle,
        new_rx,
        new_ry,
        start_angle,
        new_span_angle,
        new_rot_angle,
        transformed_points,
    )


def _transform_path_operation(
    operation: Operation | tuple, xform_matrix: NDArray
) -> Operation | tuple:
    """Return a path operation with geometry transformed by ``xform_matrix``."""
    if isinstance(operation, tuple):
        return operation

    subtype = operation.subtype
    data = operation.data
    transformed_data = data

    if subtype in _MOVE_PATH_OPS:
        transformed_data = _transform_path_point(data, xform_matrix)
    elif subtype in _LINE_PATH_OPS:
        transformed_data = tuple(
            _transform_path_point(point, xform_matrix) for point in data
        )
    elif subtype in _POLYLINE_PATH_OPS:
        transformed_data = (
            _transform_path_point(data[0], xform_matrix),
            _transform_path_points(data[1], xform_matrix),
        )
    elif subtype in _BEZIER_BLEND_PATH_OPS:
        transformed_data = tuple(
            _transform_path_point(point, xform_matrix) for point in data
        )
    elif subtype in _ARC_PATH_OPS:
        transformed_data = _transform_arc_data(data, xform_matrix)
    elif subtype in _SINE_PATH_OPS:
        transformed_points = _transform_path_points(data[0], xform_matrix)
        transformed_data = (
            transformed_points,
            line_angle(transformed_points[-2], transformed_points[-1]),
        )

    return Operation(subtype, transformed_data, operation.name)


def path2d_to_svg_path(path2d: Path2D) -> str:
    """Return the SVG path ``d`` string for a ``Path2D``.

    Args:
        path2d: Path to convert.

    Returns:
        str: SVG path data.

    Examples:
        >>> d = path2d_to_svg_path(sg.Path2D((0, 0)).line_to((10, 0)))
        >>> d.startswith("M")
        True
    """

    def fmt(val: float) -> str:
        """Format a float to a string with 3 decimal places."""
        return f"{val:.3f}".rstrip("0").rstrip(".")

    parts = [f"M {fmt(path2d.start[0])},{fmt(path2d.start[1])}"]

    obj_idx = 0
    PO = PathOps

    for op in path2d.operations:
        if isinstance(op, tuple):
            continue

        st = op.subtype
        data = op.data

        current_obj = (
            path2d.objects[obj_idx]
            if obj_idx < len(path2d.objects)
            else None
        )

        if st in _MOVE_PATH_OPS:
            parts.append(f"M {fmt(data[0])},{fmt(data[1])}")

        elif st in _LINE_PATH_OPS:
            end = data[1]
            parts.append(f"L {fmt(end[0])},{fmt(end[1])}")

        elif st == PO.SEGMENTS:
            parts.extend(f"L {fmt(p[0])},{fmt(p[1])}" for p in data[1])

        elif st in _CUBIC_PATH_OPS:
            c1, c2, end = data[1], data[2], data[3]
            parts.append(
                f"C {fmt(c1[0])},{fmt(c1[1])} {fmt(c2[0])},{fmt(c2[1])} {fmt(end[0])},{fmt(end[1])}"
            )

        elif st in _QUAD_PATH_OPS:
            c1, end = data[1], data[2]
            parts.append(
                f"Q {fmt(c1[0])},{fmt(c1[1])} {fmt(end[0])},{fmt(end[1])}"
            )

        elif st in _ARC_PATH_OPS:
            rx, ry = data[2], data[3]
            span = data[5]
            rot = degrees(data[6])
            points = data[7]
            end = points[-1]
            large_arc = 1 if abs(span) > pi else 0
            sweep = 1 if span > 0 else 0
            parts.append(
                f"A {fmt(rx)} {fmt(ry)} {fmt(rot)} {large_arc} {sweep} {fmt(end[0])},{fmt(end[1])}"
            )

        elif st == PO.CLOSE:
            parts.append("Z")

        elif st in _SINE_PATH_OPS:
            parts.extend(f"L {fmt(p[0])},{fmt(p[1])}" for p in data[0])

        elif st == PO.HOBBY_TO and current_obj:
            verts = current_obj.vertices
            parts.extend(f"L {fmt(p[0])},{fmt(p[1])}" for p in verts[1:])

        obj_idx += 1

    return " ".join(parts)


lin_path_svg = path2d_to_svg_path
path2d_svg = path2d_to_svg_path


def _format_path_code_number(value: Any, n_round: int | None = None) -> str:
    """Return a Python numeric literal for path-code generation."""
    if n_round is None:
        if isinstance(value, (int, np.integer)):
            return repr(int(value))
        number = float(value)
    else:
        number = round(float(value), n_round)
    if number.is_integer():
        return repr(int(number))
    return repr(number)


def _format_path_code_point(point: PointType, n_round: int) -> str:
    """Return a ``(x, y)`` literal. ``point`` may be length 2 or 3."""
    x, y = point[:2]
    return (
        f"({_format_path_code_number(x, n_round)}, "
        f"{_format_path_code_number(y, n_round)})"
    )


def _format_path_code_points(points: Sequence, n_round: int) -> str:
    """Return a ``[(x, y), ...]`` literal."""
    items = ", ".join(
        _format_path_code_point(point, n_round) for point in points
    )
    return f"[{items}]"


def _format_path_code_value(value: Any) -> str:
    """Return a Python literal for a style or scalar path-code value."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, Enum):
        return f"sg.{type(value).__name__}.{value.name}"
    if isinstance(value, Color):
        red = _format_path_code_number(value.red)
        green = _format_path_code_number(value.green)
        blue = _format_path_code_number(value.blue)
        if value.alpha == 1:
            return f"sg.Color({red}, {green}, {blue})"
        alpha = _format_path_code_number(value.alpha)
        return f"sg.Color({red}, {green}, {blue}, {alpha})"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _format_path_code_number(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = ", ".join(_format_path_code_value(item) for item in value)
        return f"[{items}]"
    raise TypeError(
        f"Cannot generate path code for value of type {type(value).__name__}: "
        f"{value!r}"
    )


def _format_path_code_call(method_name: str, *args: str) -> str:
    """Return ``path.method(arg, ...)`` from already-formatted argument strings."""
    return f"path.{method_name}({', '.join(args)})"


def path_code(path2d: Path2D, n_round: int | None = None) -> str:
    """Return Python source that reconstructs ``path2d``.

    The snippet assumes ``import simetri.graphics as sg``. Geometry comes from
    ``path2d.start``, ``path2d.angle``, and ``path2d.operations``. ``turn``,
    ``orient``, ``push``, and ``pop`` are not stored; ``forward`` is emitted as
    ``line_to`` so the replay does not depend on a lost heading.

    Args:
        path2d: Path to serialize.
        n_round: Decimal places for coordinates and lengths derived from them.
            ``None`` uses ``runtime_defaults['n_round']``.

    Returns:
        Python source that builds an equivalent ``Path2D``.

    Raises:
        ValueError: If an operation subtype has no code mapping, or
            ``n_round`` is negative.
        TypeError: If a style value cannot be serialized.

    Examples:
        >>> import simetri.graphics as sg
        >>> sample = sg.Path2D((0, 0), angle=0).line_to((10, 0)).close()
        >>> print(sg.path_code(sample))
        path = sg.Path2D(start=(0, 0), angle=0)
        path.line_to((10, 0))
        path.close()
        >>> messy = sg.Path2D((0, 0), angle=0).line_to((10.126, 0))
        >>> print(sg.path_code(messy, n_round=2))
        path = sg.Path2D(start=(0, 0), angle=0)
        path.line_to((10.13, 0))
    """
    if n_round is None:
        n_round = runtime_defaults["n_round"]
    if n_round < 0:
        raise ValueError("n_round must be a nonnegative integer.")
    start_x, start_y = path2d.start[:2]
    lines = [
        (
            "path = sg.Path2D("
            f"start={_format_path_code_point((start_x, start_y), n_round)}, "
            f"angle={_format_path_code_number(path2d.angle)}"
            ")"
        )
    ]

    for operation in path2d.operations:
        if isinstance(operation, tuple):
            _opcode, payload = operation
            name, value, style_kwargs = payload
            arguments = [
                _format_path_code_value(name),
                _format_path_code_value(value),
            ]
            for key, style_value in style_kwargs.items():
                arguments.append(
                    f"{key}={_format_path_code_value(style_value)}"
                )
            lines.append(_format_path_code_call("set_style", *arguments))
            continue

        subtype = operation.subtype
        data = operation.data

        if subtype in _LINE_TO_FORWARD_OPS:
            _start, end = data
            lines.append(
                _format_path_code_call(
                    "line_to", _format_path_code_point(end, n_round)
                )
            )
        elif subtype in _MOVE_PATH_OPS:
            lines.append(
                _format_path_code_call(
                    "move_to", _format_path_code_point(data, n_round)
                )
            )
        elif subtype == PathOps.R_LINE:
            start_point, end_point = data
            start_x, start_y = start_point[:2]
            end_x, end_y = end_point[:2]
            lines.append(
                _format_path_code_call(
                    "r_line",
                    _format_path_code_number(end_x - start_x, n_round),
                    _format_path_code_number(end_y - start_y, n_round),
                )
            )
        elif subtype == PathOps.H_LINE_TO:
            _start, end_point = data
            end_x, _end_y = end_point[:2]
            lines.append(
                _format_path_code_call(
                    "h_line_to", _format_path_code_number(end_x, n_round)
                )
            )
        elif subtype == PathOps.R_H_LINE:
            start_point, end_point = data
            start_x, _start_y = start_point[:2]
            end_x, _end_y = end_point[:2]
            lines.append(
                _format_path_code_call(
                    "r_h_line",
                    _format_path_code_number(end_x - start_x, n_round),
                )
            )
        elif subtype == PathOps.V_LINE_TO:
            _start, end_point = data
            _end_x, end_y = end_point[:2]
            lines.append(
                _format_path_code_call(
                    "v_line_to", _format_path_code_number(end_y, n_round)
                )
            )
        elif subtype == PathOps.R_V_LINE:
            start_point, end_point = data
            _start_x, start_y = start_point[:2]
            _end_x, end_y = end_point[:2]
            lines.append(
                _format_path_code_call(
                    "r_v_line",
                    _format_path_code_number(end_y - start_y, n_round),
                )
            )
        elif subtype == PathOps.SEGMENTS:
            _start, points = data
            lines.append(
                _format_path_code_call(
                    "segments", _format_path_code_points(points, n_round)
                )
            )
        elif subtype in _CUBIC_PATH_OPS:
            _start, control1, control2, end = data
            lines.append(
                _format_path_code_call(
                    "cubic_to",
                    _format_path_code_point(control1, n_round),
                    _format_path_code_point(control2, n_round),
                    _format_path_code_point(end, n_round),
                )
            )
        elif subtype in _QUAD_PATH_OPS:
            _start, control, end = data
            lines.append(
                _format_path_code_call(
                    "quad_to",
                    _format_path_code_point(control, n_round),
                    _format_path_code_point(end, n_round),
                )
            )
        elif subtype == PathOps.HOBBY_TO:
            _start, points = data
            lines.append(
                _format_path_code_call(
                    "hobby_to", _format_path_code_points(points, n_round)
                )
            )
        elif subtype in _ARC_CODE_PATH_OPS:
            (
                _end,
                _tangent_angle,
                radius_x,
                radius_y,
                start_angle,
                span_angle,
                rot_angle,
                _points,
            ) = data
            arguments = [
                _format_path_code_number(radius_x, n_round),
                _format_path_code_number(radius_y, n_round),
                _format_path_code_number(start_angle),
                _format_path_code_number(span_angle),
            ]
            if rot_angle != 0:
                arguments.append(
                    f"rot_angle={_format_path_code_number(rot_angle)}"
                )
            lines.append(_format_path_code_call("arc", *arguments))
        elif subtype in _SINE_PATH_OPS:
            points, _angle = data
            remaining = list(points[1:])
            lines.append(
                _format_path_code_call(
                    "segments", _format_path_code_points(remaining, n_round)
                )
            )
        elif subtype == PathOps.CLOSE:
            lines.append("path.close()")
        else:
            raise ValueError(
                f"Cannot generate path code for operation {subtype!r}"
            )

    return "\n".join(lines)


def _get_path_nums(
    tokens: Sequence[str], index: int, count: int
) -> tuple[list[float] | None, int]:
    nums: list[float] = []
    i = index
    for _ in range(count):
        if i < len(tokens):
            try:
                nums.append(float(tokens[i]))
            except ValueError:
                break
            i += 1
        else:
            break
    if len(nums) < count:
        return None, i
    return nums, i


def svg_path_to_path2d(svg_path: str) -> Path2D:
    """Parse an SVG path ``d`` string into a ``Path2D``.

        Args:
        svg_path: SVG path data string.

        Returns:
        Path2D: Equivalent path.

    Examples:
        >>> import simetri.graphics as sg
        >>> p = sg.svg_path_to_path2d("M 0 0 L 10 0 L 10 10 Z")
        >>> p.closed
        True
"""
    if not svg_path:
        return Path2D()

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
            # But Path2D always starts with a point.
            # So initialization with correct start is better.
            idx = 3
        except IndexError:
            pass

    lp = Path2D(start=start_point)

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

        if cmd_lower == "z":
            lp.close()

        elif cmd_lower == "m":
            coords, i = _get_path_nums(tokens, i, 2)
            if coords:
                if is_rel:
                    lp.r_move(*coords)
                else:
                    lp.move_to(coords)

        elif cmd_lower == "l":
            coords, i = _get_path_nums(tokens, i, 2)
            if coords:
                if is_rel:
                    lp.r_line(*coords)
                else:
                    lp.line_to(coords)

        elif cmd_lower == "h":
            coords, i = _get_path_nums(tokens, i, 1)
            if coords:
                val = coords[0]
                if is_rel:
                    lp.r_h_line(val)
                else:
                    lp.h_line_to(val)

        elif cmd_lower == "v":
            coords, i = _get_path_nums(tokens, i, 1)
            if coords:
                val = coords[0]
                if is_rel:
                    lp.r_v_line(val)
                else:
                    lp.v_line_to(val)

        elif cmd_lower == "c":
            coords, i = _get_path_nums(tokens, i, 6)
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
            coords, i = _get_path_nums(tokens, i, 4)
            if coords:
                c2 = (coords[0], coords[1])
                end = (coords[2], coords[3])

                prev_c2 = lp.pos
                last_op = lp.operations[-1] if lp.operations else None
                # Check for Cubic/BlendCubic
                if last_op and last_op.subtype in _CUBIC_PATH_OPS:
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
            coords, i = _get_path_nums(tokens, i, 4)
            if coords:
                c1 = (coords[0], coords[1])
                end = (coords[2], coords[3])
                if is_rel:
                    cur_x, cur_y = lp.pos
                    c1 = (cur_x + c1[0], cur_y + c1[1])
                    end = (cur_x + end[0], cur_y + end[1])
                lp.quad_to(c1, end)

        elif cmd_lower == "t":
            coords, i = _get_path_nums(tokens, i, 2)
            if coords:
                end = (coords[0], coords[1])

                prev_c1 = lp.pos
                last_op = lp.operations[-1] if lp.operations else None
                if last_op and last_op.subtype in _QUAD_PATH_OPS:
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
            coords, i = _get_path_nums(tokens, i, 7)
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

                if params is not None and params[0] is PathOps.LINE_TO:
                    lp.line_to(params[1])
                elif params is not None and params[0] is PathOps.ARC:
                    (
                        _op,
                        radius_x,
                        radius_y,
                        start_angle,
                        span_angle,
                        rot_angle,
                    ) = params
                    lp.arc(
                        radius_x,
                        radius_y,
                        start_angle,
                        span_angle,
                        rot_angle=rot_angle,
                    )

    return lp


def _get_svg_arc_params(
    start: PointType,
    rx: float,
    ry: float,
    phi_deg: float,
    fA: float,
    fs: float,
    end: PointType,
) -> tuple | None:
    """Convert SVG arc parameters to ``Path2D.arc`` arguments.

    Args:
        start: Arc start point.
        rx: Ellipse x-radius.
        ry: Ellipse y-radius.
        phi_deg: Ellipse rotation in degrees.
        fA: Large-arc flag.
        fs: Sweep flag.
        end: Arc end point.

    Returns:
        ``(PathOps.LINE_TO, end)`` if the arc degenerates to a line,
        ``None`` if start and end coincide, otherwise
        ``(PathOps.ARC, rx, ry, start_angle, span_angle, rot_angle)``.
    """
    x1, y1 = start[:2]
    x2, y2 = end[:2]

    rx = abs(rx)
    ry = abs(ry)
    phi = radians(phi_deg)

    if rx == 0 or ry == 0:
        return (PathOps.LINE_TO, end)

    if x1 == x2 and y1 == y2:
        return None

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
    def vector_angle(ux: float, uy: float, vx: float, vy: float) -> float:
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

    return (PathOps.ARC, rx, ry, theta1, dtheta, phi)


def _shape_edge_turn(path: Path2D, start: PointType, end: PointType) -> None:
    """Append one shape edge using ``turn`` and ``forward``."""
    edge_angle = line_angle(start, end)
    turn_angle = normalize_angle(edge_angle - path.angle)
    length = distance(start, end)
    if length == 0:
        return
    path.turn(turn_angle, distance=length)


def _shape_edge_relative(path: Path2D, start: PointType, end: PointType) -> None:
    """Append one shape edge with relative line ops when axis-aligned."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if isclose(dx, 0.0) and isclose(dy, 0.0):
        return
    if isclose(dy, 0.0):
        path.r_h_line(dx)
    elif isclose(dx, 0.0):
        path.r_v_line(dy)
    else:
        path.r_line(dx, dy)


def shape_to_path2d(
    shape: Shape,
    *,
    turns: bool = False,
    relative: bool = False,
) -> Path2D:
    """Convert a ``Shape`` into an equivalent ``Path2D``.

    By default the pen ``move_to``s the first vertex, ``line_to``s each
    remaining vertex, and ``close()`` when the shape is closed.

    With ``turns=True``, the pen starts at the first vertex (no initial
    ``move_to``) and each edge is recorded as ``turn(angle, distance)``.

    With ``relative=True`` and ``turns=False``, axis-aligned edges use
    ``r_h_line`` / ``r_v_line``; other edges use ``r_line``.

    Args:
        shape: Source shape.
        turns: Use turtle-style ``turn`` / ``forward`` for edges.
        relative: Use relative line ops when ``turns`` is False.

    Returns:
        Path2D: Path following the shape vertices.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.base.all_enums import PathOperation as PO
        >>> from simetri.geom.nonlinear.path import shape_to_path2d
        >>> rect = sg.Shape([(0, 0), (10, 0), (10, 10)], closed=True)
        >>> shape_to_path2d(rect).closed
        True
        >>> any(op.subtype == PO.FORWARD for op in shape_to_path2d(rect, turns=True).operations)
        True
        >>> ops = {op.subtype for op in shape_to_path2d(rect, relative=True).operations}
        >>> PO.R_H_LINE in ops and PO.R_V_LINE in ops
        True
"""
    vertices = shape.vertices
    if not vertices:
        return Path2D()

    if turns:
        path = Path2D(vertices[0])
        start = vertices[0]
        for vert in vertices[1:]:
            _shape_edge_turn(path, start, vert)
            start = vert
        if shape.closed:
            path.close()
        return path

    path = Path2D()
    path.move_to(vertices[0])
    start = vertices[0]
    for vert in vertices[1:]:
        if relative:
            _shape_edge_relative(path, start, vert)
        else:
            path.line_to(vert)
        start = vert
    if shape.closed:
        path.close()
    return path


def shape_to_path(
    shape: Shape,
    *,
    turns: bool = False,
    relative: bool = False,
) -> Path2D:
    """Convert a ``Shape`` into a ``Path2D`` (alias of ``shape_to_path2d``).

    Args:
        shape: Source shape.
        turns: Passed to ``shape_to_path2d``.
        relative: Passed to ``shape_to_path2d``.

    Returns:
        Path2D: Path following the shape vertices.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.geom.nonlinear.path import shape_to_path
        >>> p = shape_to_path(sg.Shape([(0, 0), (10, 0), (10, 10)], closed=True))
        >>> p.closed
        True
"""
    return shape_to_path2d(shape, turns=turns, relative=relative)


def group_to_path(group: Group) -> Path2D:
    """Convert a ``Group`` of shapes into a single ``Path2D``.

    Args:
    group: Source group.

    Returns:
    Path2D: Path that visits each shape as a subpath.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.geom.nonlinear.path import group_to_path
        >>> g = sg.Group([sg.Shape([(0, 0), (1, 0)]), sg.Shape([(2, 0), (3, 0)])])
        >>> p = group_to_path(g)
        >>> [op.subtype.name for op in p.operations]
        ['MOVE_TO', 'LINE_TO', 'MOVE_TO', 'LINE_TO']
        >>> [[round(c, 6) for c in q[:2]] for q in p.vertices]
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
"""
    shapes = group.all_shapes
    path = Path2D()
    path.move_to(shapes[0][0])
    for i, shape in enumerate(shapes):
        if i > 0:
            path.move_to(shape[0])
        for vert in shape.vertices[1:]:
            path.line_to(vert)
        if shape.closed:
            path.line_to(shape[0])

    return path


def group_to_nonzero_path(group: Group) -> Path2D:
    """Build one ``Path2D`` from a group's shapes for non-zero compound fill.

    The largest-area closed contour keeps its vertex order; every other
    closed contour is reversed so it counts as a hole under the non-zero rule.

    Args:
        group: Source group.

    Returns:
        Path2D: Combined subpaths with ``fill_mode`` ``NONZERO``.

    Raises:
        ValueError: If the group has no shape geometry.
    """
    shapes = group.all_shapes
    if not shapes:
        raise ValueError("group_to_nonzero_path requires at least one shape.")
    abs_areas = [abs(polygon_area(shape.vertices)) for shape in shapes]
    outer_index = abs_areas.index(max(abs_areas))

    def ring_vertices(shape_index: int) -> list[PointType]:
        shape = shapes[shape_index]
        vertices = list(shape.vertices)
        if shape.closed and shape_index != outer_index:
            vertices = list(reversed(vertices))
        return vertices

    first_ring = ring_vertices(0)
    path = Path2D(start=first_ring[0], fill_mode=FillMode.NONZERO)
    for vertex in first_ring[1:]:
        path.line_to(vertex)
    if shapes[0].closed:
        path.close()
    for shape_index in range(1, len(shapes)):
        ring = ring_vertices(shape_index)
        path.move_to(ring[0])
        for vertex in ring[1:]:
            path.line_to(vertex)
        if shapes[shape_index].closed:
            path.close()
    return path

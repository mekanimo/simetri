"""Shape objects are the main geometric entities in Simetri.
They are created by providing a sequence of points (a list of (x, y) coordinates).
If a style argument (a ShapeStyle object) is provided, then the style attributes
of this ShapeStyle object will superseed the style attributes of the Shape object.
"""

__all__ = [
    "Clipping",
    "Shape",
    "all_segments",
    "clip",
    "custom_attributes",
    "get_loop",
    "get_partition",
    "polygon_diff",
    "polygon_difference",
    "polygon_intersection",
    "polygon_union",
    "polygon_xor",
    "trim_margins",
]

from typing import Sequence, Union, List, Tuple, Any
from math import pi, isclose, floor
from copy import deepcopy
from dataclasses import dataclass

import json
import numpy as np
from numpy import around, array, allclose
from numpy.linalg import inv
import networkx as nx
from typing_extensions import Self

from .affine import identity_matrix
from .all_enums import (
    Types,
    InPlace,
    TransformationType,
    shape_attributes,
    FillMode,
    LineCap,
    LineJoin,
)
from .bbox import BoundingBox
from .common import PointType, LineType, get_defaults, get_unique_id
from ..canvas.style_map import shape_style_map
from ..settings.settings import defaults
from ..helpers.utilities import (
    get_transform,
    is_nested_sequence,
    decompose_transformations,
)
from ..geometry.geometry import (
    homogenize,
    right_handed,
    all_intersections,
    polygon_area,
    polyline_length,
    close_points2,
    connected_pairs,
    distance,
    remove_duplicate_points,
    multi_split_segment,
    in_polygon,
    midpoint,
    angle_between_lines2,
    positive_angle,
    lerp_point,
    polar_to_cartesian,
)
from .core import Base, _update_inplace
from .bbox import bounding_box
from .points import Points
from .batch import Group
from ..colors.colors import Color, black


class Shape(Base):
    """The main class for all geometric entities in Simetri.

     A Shape is created by providing a sequence of points (a sequence of (x, y) coordinates).
    If a style argument (a ShapeStyle object) is provided, then its style attributes override
    the default values the Shape object would assign. Additional attributes (e.g. line_width, fill_color, line_style)
    may be provided.

    """

    __slots__ = [
        "alpha",
        "back_style",
        "closed",
        "color",
        "double_distance",
        "double_color",
        "draw_double",
        "draw_fillets",
        "draw_markers",
        "fill",
        "fill_alpha",
        "fill_color",
        "fill_mode",
        "fillet_radius",
        "gradient",
        "line_alpha",
        "line_cap",
        "line_color",
        "line_dash_array",
        "line_dash_phase",
        "line_join",
        "line_miter_limit",
        "line_width",
        "marker_alpha",
        "marker_color",
        "marker_radius",
        "marker_shape",
        "marker_size",
        "marker_type",
        "markers_only",
        "points",
        "smooth",
        "stroke",
        "subtype",
        "type",
        "visible",
        "xform_matrix",
    ]

    def __init__(
        self,
        points: Union[Sequence[PointType], None] = None,
        closed: bool = False,
        fill: bool = True,
        stroke: bool = True,
        alpha: Union[float, None] = None,
        color: Union[Color, None] = None,
        draw_double: bool = False,
        draw_fillets: bool = False,
        draw_markers: bool = False,
        back_style: Any = None,
        double_distance: Union[float, None] = None,
        double_color: Union[Color, None] = None,
        fill_alpha: float = 1,
        fill_color: Color = black,
        fill_mode: FillMode = FillMode.EVENODD,
        fillet_radius: Union[float, None] = None,
        gradient: Any = None,
        line_alpha: float = 1,
        line_cap: LineCap = LineCap.BUTT,
        line_color: Color = black,
        line_dash_array: Any = None,
        line_dash_phase: Union[float, None] = None,
        line_join: LineJoin = LineJoin.MITER,
        line_miter_limit: Union[float, None] = None,
        line_width: float = 1,
        marker_alpha: float = 1,
        marker_color: Union[Color, None] = None,
        marker_radius: Union[float, None] = None,
        marker_shape: Any = None,
        marker_size: Union[float, None] = None,
        marker_type: Any = None,
        markers_only: bool = False,
        smooth: bool = False,
        subtype: Types = Types.SHAPE,
        xform_matrix: Union[np.ndarray, None] = None,
    ) -> None:
        """Initialize a Shape object.

        Args:
            points (Sequence[PointType], optional): The points that make up the shape.
            closed (bool, optional): Whether the shape is closed. Defaults to False.
            xform_matrix (np.array, optional): The transformation matrix. Defaults to None.
            **kwargs (dict): Additional attributes for the shape.

        Raises:
            ValueError: If the provided subtype is not valid.
        """

        self.id = get_unique_id(self)
        self._external = False

        if points is None:
            self.primary_points = Points()
            self.closed = False
        else:
            self.closed, points = self._get_closed(points, closed)
            self.primary_points = Points(points)
            self.primary_points.nd_array_changed = True
        self.xform_matrix = get_transform(xform_matrix)
        self.type = Types.SHAPE
        self.alpha = alpha
        self.color = color
        self.draw_double = draw_double
        self.draw_fillets = draw_fillets
        self.draw_markers = draw_markers
        self.back_style = back_style
        self.double_distance = double_distance
        self.double_color = double_color
        self.fill = fill
        self.fill_alpha = fill_alpha
        self.fill_color = fill_color
        self.fill_mode = fill_mode
        self.fillet_radius = fillet_radius
        self.gradient = gradient
        self.line_alpha = line_alpha
        self.line_cap = line_cap
        self.line_color = line_color
        self.line_dash_array = line_dash_array
        self.line_dash_phase = line_dash_phase
        self.line_join = line_join
        self.line_miter_limit = line_miter_limit
        self.line_width = line_width
        self.marker_alpha = marker_alpha
        self.marker_color = marker_color
        self.marker_radius = marker_radius
        self.marker_shape = marker_shape
        self.marker_size = marker_size
        self.marker_type = marker_type
        self.markers_only = markers_only
        self.smooth = smooth
        self.stroke = stroke
        self.subtype = subtype
        self.visible = True

        self._b_box = None

    def _get_closed(self, points: Sequence[PointType], closed: bool):
        """Determine whether the shape should be considered closed.

        Args:
            points (Sequence[PointType]): The points that define the shape.
            closed (bool): The user-specified closed flag.

        Returns:
            tuple: A tuple consisting of:
                - bool: True if the shape is closed, False otherwise.
                - list: The (possibly modified) list of points.
        """

        n = len(points)
        if n < 3:
            res = False
        else:
            points = [tuple(x[:2]) for x in points]
            polygon = self._is_polygon(points)
            res = bool(closed) or polygon
            if polygon:
                points.pop()
        return res, points

    def __len__(self):
        """Return the number of points in the shape.

        Returns:
            int: The number of primary points.
        """
        return len(self.primary_points)

    def __str__(self):
        """Return a string representation of the shape.

        Returns:
            str: A string representation of the shape.
        """
        if len(self.primary_points) == 0:
            res = "Shape()"
        elif len(self.primary_points) < 4:
            res = f"Shape({self.vertices})"
        else:
            res = f"Shape([{self.vertices[0]}, ..., {self.vertices[-1]}])"
        return res

    def __repr__(self):
        """Return a string representation of the shape.

        Returns:
            str: A string representation of the shape.
        """
        return self.__str__()

    def __getitem__(self, subscript: Union[int, float, slice]):
        """Retrieve point(s) from the shape by index or slice.

        Args:
            subscript (int, float or slice): The index or slice specifying the point(s) to retrieve. If float then the point is compute by interpolation/extrapolation.

        Returns:
            PointType or list[PointType]: The requested point or list of points (after applying the transformation).

        Raises:
            TypeError: If the subscript type is invalid.
        """
        # Use cached final_coords instead of recalculating matrix multiplication
        final_coords = self.final_coords

        if isinstance(subscript, slice):
            res = [tuple(coord[:2]) for coord in final_coords[subscript]]
        elif isinstance(subscript, float):
            if subscript.is_integer():
                subscript = int(subscript)
                coord = final_coords[subscript]
                res = (coord[0], coord[1])
            else:
                n = len(final_coords)
                index = int(floor(subscript))
                if self.closed:
                    next_index = (index + 1) % n
                else:
                    if subscript >= n - 1:
                        raise ValueError("Invalid index!")
                    else:
                        next_index = index + 1
                vertex = final_coords[index]
                next_vertex = final_coords[next_index]
                t = subscript - index
                res = lerp_point(vertex, next_vertex, t)
        else:
            coord = final_coords[subscript]
            res = (coord[0], coord[1])
        return res

    def __setitem__(self, subscript, value):
        """Set the point(s) at the given subscript.

        Args:
            subscript (int or slice): The subscript to set the point(s) at.
            value (PointType or list[PointType]): The value to set the point(s) to.

        Raises:
            TypeError: If the subscript type is invalid.
        """
        if isinstance(subscript, slice):
            if is_nested_sequence(value):
                value = homogenize(value) @ inv(self.xform_matrix)
            else:
                value = homogenize([value]) @ inv(self.xform_matrix)
            self.primary_points[
                subscript.start : subscript.stop : subscript.step
            ] = [tuple(x[:2]) for x in value]
            self.primary_points.nd_array_changed = True
        elif isinstance(subscript, int):
            value = homogenize([value]) @ inv(self.xform_matrix)
            self.primary_points[subscript] = tuple(value[0][:2])
            self.primary_points.nd_array_changed = True
        else:
            raise TypeError("Invalid subscript type")

    def __delitem__(self, subscript) -> Self:
        """Delete the point(s) at the given subscript.

        Args:
            subscript (int or slice): The subscript to delete the point(s) from.
        """
        del self.primary_points[subscript]

    def index(self, point: PointType, abs_tol=None) -> int:
        """Return the index of the given point.

        Args:
            point (PointType): The point to find the index of.
            abs_tol (float, optional): Absolute tolerance for comparison. Defaults to None.

        Returns:
            int: The index of the point.
        """
        point = tuple(point[:2])

        if abs_tol is None:
            abs_tol = defaults["abs_tol"]
        ind = np.where(
            (np.isclose(self.vertices, point, atol=abs_tol)).all(axis=1)
        )[0][0]

        return ind

    def remove(self, point: PointType) -> Self:
        """Remove a point from the shape.

        Args:
            point (PointType): The point to remove.
        """
        ind = self.vertices.index(point)
        self.primary_points.pop(ind)

        return self

    def append(self, point: PointType) -> Self:
        """Append a point to the shape.

        Args:
            point (PointType): The point to append.
        """
        point = homogenize([point]) @ inv(self.xform_matrix)
        self.primary_points.append(tuple(point[0][:2]))

    def insert(self, index: int, point: PointType) -> Self:
        """Insert a point at a given index.

        Args:
            index (int): The index to insert the point at.
            point (PointType): The point to insert.
        """
        point = homogenize([point]) @ inv(self.xform_matrix)
        self.primary_points.insert(index, tuple(point[0][:2]))

        return self

    def extend(self, points: Sequence[PointType]) -> Self:
        """Extend the shape with a list of points.

        Args:
            values (list[PointType]): The points to extend the shape with.
        """
        homogenized = homogenize(points) @ inv(self.xform_matrix)
        self.primary_points.extend([tuple(x[:2]) for x in homogenized])

        return self

    def pop(self, index: int = -1) -> PointType:
        """Pop a point from the shape.

        Args:
            index (int, optional): The index to pop the point from, defaults to -1.

        Returns:
            PointType: The popped point.
        """
        point = self.vertices[index]
        self.primary_points.pop(index)

        return point

    def __iter__(self):
        """Return an iterator over the vertices of the shape.

        Returns:
            Iterator[PointType]: An iterator over the vertices of the shape.
        """
        return iter(self.vertices)

    def _update(
        self,
        xform_matrix: array,
        reps: int = 0,
        take: slice | None = None,
        incr: float
        | tuple[float, float]
        | tuple[callable, Any]
        | tuple[InPlace, Any]
        | None = None,
        merge: bool = False,
        xform_type: TransformationType = None,
    ) -> Union["Shape", Group]:
        """Used internally. Update the shape with a transformation matrix.

        Args:
            xform_matrix (array): The transformation matrix.
            reps (int, optional): The number of repetitions, defaults to 0.

        Returns:
            Shape or Group: The updated shape or a group of shapes.
        """
        if reps == 0:
            fillet_radius = getattr(self, "fillet_radius", None)
            if fillet_radius:
                scale = max(decompose_transformations(xform_matrix)[2])
                self.fillet_radius = fillet_radius * scale

            self.xform_matrix = self.xform_matrix @ xform_matrix
            # Invalidate coordinate caches when transformation changes
            if "_final_coords" in self.__dict__:
                delattr(self, "_final_coords")
            if "_vertices" in self.__dict__:
                delattr(self, "_vertices")
            res = self
        else:
            shapes = [self]
            shape = self
            for i in range(reps):
                shape = shape.copy()
                if incr is not None and i > 0:
                    xform_matrix = _update_inplace(
                        xform_matrix, xform_type, incr
                    )

                shape._update(xform_matrix)
                shapes.append(shape)
            res = Group(shapes)

        if merge and reps > 0:
            return res.merge_shapes()

        return res

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        """Check if the shape is equal to another shape.

        Args:
            other (Shape): The other shape to compare to.

        Returns:
            bool: True if the shapes are equal, False otherwise.
        """
        if not hasattr(other, "type"):
            return False
        if other.type != Types.SHAPE:
            return False

        len1 = len(self)
        len2 = len(other)
        if len1 == 0 and len2 == 0:
            res = True
        elif len1 == 0 or len2 == 0:
            res = False
        elif isinstance(other, Shape) and len1 == len2:
            res = allclose(
                self.xform_matrix,
                other.xform_matrix,
                rtol=defaults["rel_tol"],
                atol=defaults["abs_tol"],
            ) and allclose(
                self.primary_points.nd_array,
                other.primary_points.nd_array,
                rtol=defaults["rel_tol"],
                atol=defaults["abs_tol"],
            )
        else:
            res = False

        return res

    def __bool__(self):
        """Return whether the shape has any points.

        Returns:
            bool: True if the shape has points, False otherwise.
        """
        return len(self.primary_points) > 0

    def to_json(self) -> str:
        """Serialize the Shape into a JSON string.

        The payload includes:
          - type, subtype (as strings)
          - closed (bool)
          - points (original primary points, not transformed)
          - xform_matrix (3x3)
          - style (resolved style attributes present on the shape)
        """

        def _to_jsonable(obj):
            # Enums (StrEnum) -> value
            if hasattr(obj, "value"):
                return obj.value
            # Native primitives
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            # Numpy arrays -> lists
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Sequences -> list
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(x) for x in obj]
            # Dict -> dict
            if isinstance(obj, dict):
                return {k: _to_jsonable(v) for k, v in obj.items()}
            # Fallbacks
            try:
                return float(obj)
            except Exception:
                try:
                    return str(obj)
                except Exception:
                    return None

        # Original points (primary points before transform)
        try:
            prim_points = (
                list(self.primary_points) if self.primary_points else []
            )
        except Exception:
            prim_points = []

        data = {
            "type": getattr(self.type, "value", str(self.type)),
            "subtype": getattr(self.subtype, "value", str(self.subtype)),
            "closed": bool(self.closed),
            "points": [(float(p[0]), float(p[1])) for p in prim_points],
            "xform_matrix": _to_jsonable(self.xform_matrix),
            "style": {},
        }

        # Include style attributes that are set on the shape
        for attrib in shape_style_map:
            val = getattr(self, attrib, None)
            if val is not None:
                data["style"][attrib] = _to_jsonable(val)

        return json.dumps(data, ensure_ascii=False)

    def copy_style(self, other):
        """Copies the other shape's style."""
        self.alpha = other.alpha
        self.color = other.color

        if other.color is not None:
            self.line_color = other.color
            self.fill_color = other.color
        else:
            self.line_color = other.line_color
            self.fill_color = other.fill_color

        if other.alpha is not None:
            self.line_alpha = other.alpha
            self.fill_alpha = other.alpha
        else:
            self.line_alpha = other.line_alpha
            self.fill_alpha = other.fill_alpha

        self.line_width = other.line_width
        self.fill = other.fill
        self.stroke = other.stroke
        self.line_dash_array = other.line_dash_array
        self.line_dash_phase = other.line_dash_phase
        self.line_cap = other.line_cap
        self.line_join = other.line_join
        self.smooth = other.smooth
        self.back_style = other.back_style
        self.draw_markers = other.draw_markers
        self.marker_type = other.marker_type
        self.marker_size = other.marker_size
        self.marker_radius = other.marker_radius
        self.markers_only = other.markers_only

        return self

    def is_clockwise(self) -> bool:
        """Check if the shape is oriented clockwise.

        Returns:
            bool: True if the shape is oriented clockwise, False otherwise.
        """
        if not self.closed:
            raise ValueError("Shape must be closed to check orientation")
        vertices = self.vertices
        area = polygon_area(vertices)
        return area < 0

    def reordered(self, index) -> Self:
        """Return a copy of the shape starting from a point
        at the given index.

        Args:
            point (PointType): The point to start from.

        Returns:
            Shape: The shape with the starting point set.
        """
        if not isinstance(index, int):
            raise TypeError("Index must be an integer")
        if not self.closed:
            raise ValueError("Shape must be closed to start from a point")

        if index == 0:
            res = self.copy()
        else:
            shape = self.copy()
            vertices = shape.vertices
            shape[:] = vertices[index:] + vertices[:index]
            res = shape

        return res

    def lerp(self, edge: int, t: float) -> PointType:
        """Given an edge index and t value (between 0 and 1)
        returns the corresponding interpolated point.
        """
        return lerp_point(*self.edges[edge], t)

    def merge_collinears(self):
        """Merges collinear edges."""
        return Group([self]).merge_shapes()[0]

    def merge(self, other, dist_tol: float = None) -> Union[Self, None]:
        """Merge two shapes if they are connected. Does not work for polygons.
        Only polyline shapes can be merged together.

        Args:
            other (Shape): The other shape to merge with.
            dist_tol (float, optional): The distance tolerance for merging, defaults to None.

        Returns:
            Shape or None: The merged shape or None if the shapes cannot be merged.
        """
        if dist_tol is None:
            dist_tol = defaults["dist_tol"]

        if self.closed or other.closed or self.is_polygon or other.is_polygon:
            res = None
        else:
            vertices = self._chain_vertices(
                self.as_list(), other.as_list(), dist_tol=dist_tol
            )
            if vertices:
                closed = close_points2(
                    vertices[0], vertices[-1], dist2=defaults["dist_tol2"]
                )
                res = Shape(vertices, closed=closed)
            else:
                res = None

        return res

    def connect(self, other) -> Self:
        """Connect two shapes by adding the other shape's vertices to self.

        Args:
            other (Shape): The other shape to connect.
        """
        self.extend(other.vertices)

        return self

    def _chain_vertices(
        self,
        verts1: Sequence[PointType],
        verts2: Sequence[PointType],
        dist_tol: float = None,
    ) -> Union[List[PointType], None]:
        """Chain two sets of vertices if they are connected.

        Args:
            verts1 (list[PointType]): The first set of vertices.
            verts2 (list[PointType]): The second set of vertices.
            dist_tol (float, optional): The distance tolerance for chaining, defaults to None.

        Returns:
            list[PointType] or None: The chained vertices or None if the vertices cannot be chained.
        """
        dist_tol2 = dist_tol * dist_tol
        start1, end1 = verts1[0], verts1[-1]
        start2, end2 = verts2[0], verts2[-1]
        same_starts = close_points2(start1, start2, dist2=dist_tol2)
        same_ends = close_points2(end1, end2, dist2=dist_tol2)
        if same_starts and same_ends:
            res = verts1
        elif close_points2(end1, start2, dist2=dist_tol2):
            verts2.pop(0)
        elif close_points2(start1, end2, dist2=dist_tol2):
            verts2.reverse()
            verts1.reverse()
            verts2.pop(0)
        elif same_starts:
            verts2.reverse()
            verts2.pop(-1)
            start = verts2[:]
            end = verts1[:]
            verts1 = start
            verts2 = end
        elif same_ends:
            verts2.reverse()
            verts2.pop(0)
        else:
            return None
        if same_starts and same_ends:
            all_verts = verts1 + verts2
            if not right_handed(all_verts):
                all_verts.reverse()
            res = all_verts
        else:
            res = verts1 + verts2

        return res

    def _is_polygon(self, vertices: Sequence[PointType]) -> bool:
        """Return True if the vertices form a polygon.

        Args:
            vertices (list[PointType]): The vertices to check.

        Returns:
            bool: True if the vertices form a polygon, False otherwise.
        """
        return close_points2(
            vertices[0][:2], vertices[-1][:2], dist2=defaults["dist_tol2"]
        )

    def as_array(self, homogeneous=False) -> np.ndarray:
        """Return the vertices as an array.

        Args:
            homogeneous (bool, optional): Whether to return homogeneous coordinates, defaults to False.

        Returns:
            ndarray: The vertices as an array.
        """
        if homogeneous:
            # Use cached final_coords to avoid redundant matrix multiplication
            res = self.final_coords
        else:
            res = array(self.vertices)
        return res

    def as_list(self) -> List[PointType]:
        """Return the vertices as a list of tuples.

        Returns:
            list[tuple]: The vertices as a list of tuples.
        """
        return list(self.vertices)

    @property
    def final_coords(self) -> np.ndarray:
        """The final coordinates of the shape. primary_points @ xform_matrix.

        Returns:
            ndarray: The final coordinates of the shape.
        """
        if self.primary_points:
            # Cache the expensive matrix multiplication
            if (
                "_final_coords" not in self.__dict__
                or self.primary_points.nd_array_changed
            ):
                self._final_coords = (
                    self.primary_points.homogen_coords @ self.xform_matrix
                )
            res = self._final_coords
        else:
            res = array([])

        return res

    @property
    def angle(self):
        """Orientation angle of the shape."""
        res = decompose_transformations(self.xform_matrix)[1]
        return positive_angle(res)

    @property
    def vertices(self) -> tuple[PointType]:
        """The final coordinates of the shape.

        Returns:
            tuple: The final coordinates of the shape.
        """

        if self.primary_points:
            # Cache vertices computation and only recompute when data changes
            if (
                "_vertices" not in self.__dict__
                or self.primary_points.nd_array_changed
            ):
                res = tuple((x[0], x[1]) for x in (self.final_coords[:, :2]))
                self._vertices = res
                self.primary_points.nd_array_changed = False
            else:
                res = self._vertices
        else:
            res = ()

        return res

    @property
    def vertex_pairs(self) -> List[Tuple[PointType, PointType]]:
        """Return a list of connected pairs of vertices.

        Returns:
            list[tuple[PointType, PointType]]: A list of connected pairs of vertices.
        """
        vertices = list(self.vertices)
        if self.closed:
            vertices.append(vertices[0])
        return connected_pairs(vertices)

    @property
    def orig_coords(self) -> np.ndarray:
        """The primary points in homogeneous coordinates.

        Returns:
            ndarray: The primary points in homogeneous coordinates.
        """
        return self.primary_points.homogen_coords

    @property
    def b_box(self) -> BoundingBox:
        """Return the bounding box of the shape.

        Returns:
            BoundingBox: The bounding box of the shape.
        """
        if self.primary_points:
            self._b_box = bounding_box(self.final_coords)
        else:
            self._b_box = bounding_box([(0, 0)])
        return self._b_box

    @property
    def area(self) -> float:
        """Return the area of the shape.

        Returns:
            float: The area of the shape.
        """
        if self.closed:
            vertices = self.vertices[:]
            if not close_points2(
                vertices[0], vertices[-1], dist2=defaults["dist_tol2"]
            ):
                vertices = list(vertices) + [vertices[0]]
            res = polygon_area(vertices)
        else:
            res = 0

        return res

    @property
    def total_length(self) -> float:
        """Return the total length of the shape.

        Returns:
            float: The total length of the shape.
        """
        return polyline_length(self.vertices[:-1], self.closed)

    @property
    def is_polygon(self) -> bool:
        """Return True if 'closed'.

        Returns:
            bool: True if the shape is closed, False otherwise.
        """
        return self.closed

    def clear(self) -> Self:
        """Clear all points and reset the style attributes.

        Returns:
            None
        """
        self.primary_points = Points()
        self.xform_matrix = identity_matrix()
        self._b_box = None
        # Clear coordinate caches
        if "_final_coords" in self.__dict__:
            delattr(self, "_final_coords")
        if "_vertices" in self.__dict__:
            delattr(self, "_vertices")

        return self

    def count(self, point: PointType) -> int:
        """Return the number of times the point is found in the shape.

        Args:
            point (PointType): The point to count.

        Returns:
            int: The number of times the point is found in the shape.
        """
        verts = self.orig_coords @ self.xform_matrix
        verts = verts[:, :2]
        n = verts.shape[0]
        point = array(point[:2])
        values = np.tile(point, (n, 1))
        col1 = (verts[:, 0] - values[:, 0]) ** 2
        col2 = (verts[:, 1] - values[:, 1]) ** 2
        distances = col1 + col2

        return np.count_nonzero(distances <= defaults["dist_tol2"])

    def copy(self) -> "Shape":
        """Return a copy of the shape.

        Returns:
            Shape: A copy of the shape.
        """
        self._b_box = None
        return deepcopy(self)

    def segment(
        self, i: int, j: int, midpoints: bool = False
    ) -> tuple[PointType]:
        """Returns a line segment with shape[i] and shape[j] endpoints.
        If midpoints is True then returns a line segment between midpoints
        of the shape.edges[i] and shape.edges[j]
        """
        if midpoints:
            res = (self.edge_midpoint(i), self.edge_midpoint(j))
        else:
            res = (self[i], self[j])

        return res

    def edge_midpoint(self, i: int) -> PointType:
        """Return the midpoint of shape.edges[i]."""

        n = len(self)
        edge = (self[i], self[(i + 1) % n])

        return midpoint(*edge)

    @property
    def edge_midpoints(self) -> List[PointType]:
        """Return a list of the edge midpoints."""
        edges = self.edges

        return [midpoint(*edge) for edge in edges]

    @property
    def edges(self) -> List[LineType]:
        """Return a list of the edges of the shape.

        Edges are represented as tuples of points:
        edge: ((x1, y1), (x2, y2))
        edges: [((x1, y1), (x2, y2)), ((x2, y2), (x3, y3)), ...]

        Returns:
            list[tuple[PointType, PointType]]: A list of edges.
        """
        vertices = list(self.vertices[:])
        if self.closed:
            vertices.append(vertices[0])

        return tuple(connected_pairs(vertices))

    @property
    def midpoints(self) -> List[PointType]:
        """Returns a list of the midpoints of the edges."""
        return [midpoint(*edge) for edge in self.edges]

    @property
    def segments(self) -> List[LineType]:
        """Return a list of edges.

        Edges are represented as tuples of points:
        edge: ((x1, y1), (x2, y2))
        edges: [((x1, y1), (x2, y2)), ((x2, y2), (x3, y3)), ...]

        Returns:
            list[tuple[PointType, PointType]]: A list of edges.
        """

        return self.edges

    def reverse(self) -> Self:
        """Reverse the order of the vertices.

        Returns:
            None
        """
        self.primary_points.reverse()

        return self

    ##################################################################

    @property
    def left(self):
        """
        Return the left edge.

        Returns:
            tuple: The left edge.
        """
        return self.b_box.left

    @property
    def right(self):
        """
        Return the right edge.

        Returns:
            tuple: The right edge.
        """
        return self.b_box.right

    @property
    def top(self):
        """
        Return the top edge.

        Returns:
            tuple: The top edge.
        """
        return self.b_box.top

    @property
    def bottom(self):
        """
        Return the bottom edge.

        Returns:
            tuple: The bottom edge.
        """
        return self.b_box.bottom

    @property
    def vert_centerline(self):
        """
        Return the vertical centerline.

        Returns:
            tuple: The vertical centerline.
        """
        return (self.b_box.north, self.b_box.south)

    @property
    def horiz_centerline(self):
        """
        Return the horizontal centerline.

        Returns:
            tuple: The horizontal centerline.
        """
        return (self.b_box.west, self.b_box.east)

    @property
    def midpoint(self):
        """
        Return the center of the bounding box.

        Returns:
            tuple: The center of the bounding box.
        """
        x1, y1 = self.southwest
        x2, y2 = self.northeast

        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2

        return (xc, yc)

    @property
    def corners(self):
        """
        Return the four corners of the bounding box.

        Returns:
            tuple: The four corners of the bounding box.
        """
        return (self.northwest, self.southwest, self.southeast, self.northeast)

    @property
    def diamond(self):
        """
        Return the four center points of the bounding box in a diamond shape.

        Returns:
            tuple: The four center points of the bounding box in a diamond shape.
        """
        return (self.north, self.west, self.south, self.east)

    @property
    def all_anchors(self):
        """
        Return all anchors of the bounding box.

        Returns:
            tuple: All anchors of the bounding box.
        """
        return (
            self.west,
            self.southwest,
            self.south,
            self.northeast,
            self.east,
            self.northeast,
            self.north,
            self.northwest,
            self.midpoint,
        )

    @property
    def all_lines(self):
        """
        Return all lines of the bounding box.

        Returns:
            tuple: All lines of the bounding box.
        """

        return (
            self.left,
            self.bottom,
            self.right,
            self.top,
            self.horiz_centerline,
            self.vert_centerline,
            self.diagonal1,
            self.diagonal2,
        )

    @property
    def width(self):
        """
        Return the width of the bounding box.

        Returns:
            float: The width of the bounding box.
        """
        return distance(self.northwest, self.northeast)

    @property
    def height(self):
        """
        Return the height of the bounding box.

        Returns:
            float: The height of the bounding box.
        """
        return distance(self.northwest, self.southwest)

    @property
    def size(self):
        """
        Return the size of the bounding box.

        Returns:
            tuple: The size of the bounding box.
        """
        return (self.width, self.height)

    @property
    def west(self):
        """
        Return the left edge midpoint.

        Returns:
            tuple: The left edge midpoint.
        """
        return midpoint(*self.left)

    @property
    def south(self):
        """
        Return the bottom edge midpoint.

        Returns:
            tuple: The bottom edge midpoint.
        """
        return midpoint(*self.bottom)

    @property
    def east(self):
        """
        Return the right edge midpoint.

        Returns:
            tuple: The right edge midpoint.
        """
        return midpoint(*self.right)

    @property
    def north(self):
        """
        Return the top edge midpoint.

        Returns:
            tuple: The top edge midpoint.
        """
        return midpoint(*self.top)

    @property
    def northwest(self):
        """
        Return the top left corner.

        Returns:
            tuple: The top left corner.
        """
        return self.b_box.northwest

    @property
    def northeast(self):
        """
        Return the top right corner.

        Returns:
            tuple: The top right corner.
        """
        return self.b_box.northeast

    @property
    def southwest(self):
        """
        Return the bottom left corner.

        Returns:
            tuple: The bottom left corner.
        """
        return self.b_box.southwest

    @property
    def southeast(self):
        """
        Return the bottom right corner.

        Returns:
            tuple: The bottom right corner.
        """
        return self.b_box.southeast

    @property
    def diagonal1(self):
        """
        Return the first diagonal. From the top left to the bottom right.

        Returns:
            tuple: The first diagonal.
        """
        return (self.southwest, self.northeast)

    @property
    def diagonal2(self):
        """
        Return the second diagonal. From the top right to the bottom left.

        Returns:
            tuple: The second diagonal.
        """
        return (self.southeast, self.northwest)

    def get_inflated_b_box(
        self,
        left_margin=None,
        bottom_margin=None,
        right_margin=None,
        top_margin=None,
    ):
        """
        Return a bounding box with offset edges.

        Args:
            left_margin (float, optional): The left margin.
            bottom_margin (float, optional): The bottom margin.
            right_margin (float, optional): The right margin.
            top_margin (float, optional): The top margin.

        Returns:
            BoundingBox: The inflated bounding box.
        """

        if bottom_margin is None:
            bottom_margin = left_margin
        if right_margin is None:
            right_margin = left_margin
        if top_margin is None:
            top_margin = bottom_margin

        x, y = self.southwest[:2]
        southwest = (x - left_margin, y - bottom_margin)

        x, y = self.northeast[:2]
        northeast = (x + right_margin, y + top_margin)

        return BoundingBox(southwest, northeast)

    def offset_line(self, side, offset):
        """
        Offset is applied outwards. Use negative values for inward offset.

        Args:
            side (Side): The side to offset.
            offset (float): The offset distance.

        Returns:
            tuple: The offset line.
        """
        return self.b_box.offset_line(side, offset)

    def offset_point(self, anchor, dx, dy):
        """
        Return an offset point from the given corner.

        Args:
            anchor (Anchor): The anchor point.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            list: The offset point.
        """
        return self.b_box.offset_point(anchor, dx, dy)

    def centered(
        self, item: "Union[Shape, Group]", dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the center of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.midpoint of the reference item's bounding-box.
        """

        x, y = item.midpoint[:2]
        x += dx
        y += dy
        return x, y

    def left_of(
        self, item: "Union[Shape, Group]", dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the item.west of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.west of the reference item's bounding-box.
        """
        x, y = item.west[:2]
        w2 = self.width / 2
        x += dx - w2
        y += dy
        return x, y

    def right_of(
        self, item: "Union[Shape, Group]", dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the item.east of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.east of the reference item's bounding-box.
        """
        x, y = item.east[:2]
        w2 = self.width / 2
        x += dx + w2
        y += dy
        return x, y

    def above(
        self, item: "Union[Shape, Group]", dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the item.north of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.north of the reference item's bounding-box.
        """
        x, y = item.north[:2]
        h2 = self.height / 2
        x += dx
        y += dy + h2
        return x, y

    def below(
        self, item: "Union[Shape, Group]", dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the item.south of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.south of the reference item's bounding-box.
        """
        x, y = item.south[:2]
        h2 = self.height / 2
        x += dx
        y += dy - h2
        return x, y

    def above_left(
        self, item: "Union[Shape, Group]", dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the item.northwest of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.northwest of the reference item's bounding-box.
        """
        x, y = item.northwest[:]
        w2 = self.width / 2
        h2 = self.height / 2
        x += dx - w2
        y += dy + h2

        return x, y

    def above_right(
        self, item: "Union[Shape, Group]", dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the item.northeast of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.northeast of the reference item's bounding-box.
        """
        x, y = item.northeast[:2]
        w2 = self.width / 2
        h2 = self.height / 2
        x += dx + w2
        y += dy + h2

        return x, y

    def below_left(
        self, item: "Union[Shape, Group]", dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the item.southwest of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.southwest of the reference item's bounding-box.
        """
        x, y = item.southwest[:2]
        w2 = self.width / 2
        h2 = self.height / 2
        x += dx - w2
        y += dy - h2

        return x, y

    def below_right(
        self, item: "Union[Shape, Group]", dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the item.southeast of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.southeast of the reference item's bounding-box.
        """
        x, y = item.southeast[:2]
        w2 = self.width / 2
        h2 = self.height / 2
        x += dx + w2
        y += dy - h2

        return x, y

    def polar_pos(
        self, item: "Union[Shape, Group]", angle: float, radius: float
    ) -> PointType:
        """
        Get the polar position of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            theta (float): The angle in radians.
            radius (float): The radius.

        Returns:
            PointType: The polar position of the reference item.
        """

        x, y = item.midpoint[:2]

        x1, y1 = polar_to_cartesian(radius, angle)
        x += x1
        y += y1

        return x, y

    ##################################################################

    def reorder_vertices(
        self, value: PointType, index: int = 0, tol: float = None
    ) -> Union["Shape", None]:
        """If index is not given, the vertex with the given value will be
        the first index.
        If index is given, the vertex with the given value will be
        at the given index.
        The rest of the indices will be shifted accordingly.

        Shape must be closed.

        Args:
            index (int): The target index.
            value (PointType): The vertex to relocate at the given index.

        Returns:
            Shape: A new shape with the adjusted vertices.
        """
        if not isinstance(index, int):
            raise TypeError("Index must be an integer")

        if not isinstance(value, Sequence) or len(value) < 2:
            raise TypeError("Value must be a [x, y] sequence")

        if self.closed:
            vertices = list(self.vertices)
            if value in vertices:
                cur_index = vertices.index(value)
            else:
                if tol is None:
                    tol = defaults["dist_tol"]
                dist, ind = min(
                    [(distance(value, v), i) for i, v in enumerate(vertices)],
                    key=lambda x: x[0],
                )
                if dist < tol:
                    cur_index = ind
                else:
                    return None

            shift = index - cur_index
            if shift == 0:
                return None

            if value in vertices:
                new_vertices = vertices[cur_index:] + vertices[:cur_index]
            else:
                if tol is None:
                    tol = defaults["dist_tol"]
                if distance(value, vertices[cur_index]) < tol:
                    new_vertices = vertices[cur_index:] + vertices[:cur_index]
                else:
                    new_vertices = None
            if new_vertices is not None:
                res = self.copy()
                res[:] = new_vertices
            else:
                res = None
        else:
            res = None

        return res


def trim_margins(
    item: Shape | Group,
    left: float = 0,
    bottom: float = 0,
    right: float = 0,
    top: float = 0,
) -> Shape | Group:
    """Trim the margins of a Shape or Group.

    Args:
        item (Union[Shape, Group]): The Shape or Group to trim.
        left (float, optional): The left margin to trim. Defaults to 0.
        bottom (float, optional): The bottom margin to trim. Defaults to 0.
        right (float, optional): The right margin to trim. Defaults to 0.
        top (float, optional): The top margin to trim. Defaults to 0.

    Returns:
        Union[Shape, Group]: The trimmed Shape or Group.
    """
    corners = item.b_box.get_inflated_b_box(
        -left, -bottom, -right, -top
    ).corners
    clipper = Shape(corners, closed=True)

    return clip(item, clipper, exclude_clipper=True)


def clip(
    item: Shape | Group,
    clipper: Shape,
    exclude_clipper: bool = False,
    rel_tol: float = None,
    abs_tol: float = None,
    merge: bool = True,
):
    if isinstance(item, Group):
        return _clip_group(item, clipper, exclude_clipper, rel_tol, abs_tol)
    elif isinstance(item, Shape):
        clipped_item = _clip_shape(
            item,
            clipper,
            exclude_clipper,
            rel_tol,
            abs_tol,
        )
        if clipped_item.type == Types.GROUP:
            for clipped_shape in clipped_item:
                clipped_shape.copy_style(item)
        else:
            clipped_item.copy_style(item)

        if merge:
            clipped_item = clipped_item.merge_shapes()

        return clipped_item
    else:
        raise TypeError("Invalid item type")


def _clip_group(
    group: Group,
    clipper: Shape,
    exclude_clipper: bool = True,
    rel_tol: float = None,
    abs_tol: float = None,
):
    """
    group Group: group to be clipped
    clipper Shape: clipping region
    exclude_clipper bool: If True, clipper's edges are excluded.
    """
    res = Group()

    for item in group.elements:
        if item.type == Types.GROUP:
            clipped_item = _clip_group(
                item,
                clipper,
                exclude_clipper,
                rel_tol,
                abs_tol,
            )
        elif item.type == Types.SHAPE:
            clipped_item = _clip_shape(
                item,
                clipper,
                exclude_clipper,
                rel_tol,
                abs_tol,
            )
            if clipped_item.type == Types.GROUP:
                for clipped_shape in clipped_item:
                    clipped_shape.copy_style(item)
            else:
                clipped_item.copy_style(item)
        else:
            raise TypeError("Invalid item type")

        if clipped_item.type == Types.GROUP:
            if len(clipped_item) > 0:
                res.append(clipped_item)
        else:
            res.append(clipped_item)

    return res


def _clip_shape(
    shape: "Shape",
    clipper: "Shape",
    exclude_clipper: bool = False,
    rel_tol: float = None,
    abs_tol: float = None,
):
    """
    shape Shape: shape to be clipped
    clipper Shape: clipping region
    exclude_clipper bool: If True, clipper's edges are excluded.
    """
    if not clipper.closed:
        raise ValueError("Clipper shape is not closed")
    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    n_shape = len(shape)
    segments = [[p1[:2], p2[:2]] for (p1, p2) in shape.edges] + [
        [p1[:2], p2[:2]] for (p1, p2) in clipper.edges
    ]
    intersections = all_intersections(segments)

    split_points_by_index = {}
    for key, value in intersections[0].items():
        points = [point_data[0] for point_data in value]
        split_points_by_index[key] = remove_duplicate_points(points)

    def split_segment(segment_index: int):
        points = split_points_by_index.get(segment_index, [])
        if points:
            return multi_split_segment(segments[segment_index], points)
        return [segments[segment_index]]

    clipped = Group()
    shape_vertices = shape.vertices
    clipper_vertices = clipper.vertices
    if shape.closed:
        shape_segment_count = n_shape
    else:
        shape_segment_count = n_shape - 1

    for segment_index in range(shape_segment_count):
        for seg in split_segment(segment_index):
            if not isclose(distance(*seg), 0, rel_tol=rel_tol, abs_tol=abs_tol):
                if in_polygon(
                    midpoint(*seg), clipper_vertices, exclude_clipper
                ):
                    clipped.append(Shape(seg))

    if shape.closed and not exclude_clipper:
        for segment_index in range(shape_segment_count, len(segments)):
            for seg in split_segment(segment_index):
                if not isclose(
                    distance(*seg),
                    0,
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                ) and in_polygon(
                    midpoint(*seg),
                    shape_vertices,
                    exclude_clipper,
                ):
                    clipped.append(Shape(seg))

    if len(clipped) == 1:
        return clipped[0]

    return clipped


def custom_attributes(item: Shape) -> List[str]:
    """Return a list of custom attributes of a Shape or Group instance.

    Args:
        item (Shape): The Shape or Group instanc
    Returns:
        list[str]: A list of custom attribute names.

    Raises:
        TypeError: If the item is not a Shape instance.
    """
    if isinstance(item, Shape):
        dummy = Shape([(0, 0), (1, 0)])
    else:
        raise TypeError("Invalid item type")
    native_attribs = set(dir(dummy))
    known_shape_attribs = set(shape_attributes)
    custom_attribs = set(dir(item)) - native_attribs - known_shape_attribs

    if hasattr(item, "_aliases") and isinstance(item._aliases, dict):
        custom_attribs = custom_attribs.difference(set(item._aliases.keys()))

    return sorted(custom_attribs)


@dataclass
class Clipping:
    target: Union[Shape, Group]
    clipper: Shape

    def __post_init__(self):
        self.type = Types.CLIPPING
        self.subtype = Types.CLIPPING


def polygon_union(shape1: "Shape", shape2: "Shape", merge: bool = True):
    """
    shape1 Shape: shape to be clipped
    shape2 Shape: clipping region
    """
    if not (shape1.closed and shape2.closed):
        raise Warning("Both shapes must be closed")

    segments = [[p1[:2], p2[:2]] for (p1, p2) in shape1.edges] + [
        [p1[:2], p2[:2]] for (p1, p2) in shape2.edges
    ]
    intersections = all_intersections(segments)

    all_segments_ = []
    for key, value in intersections[0].items():
        segment = segments[key]
        points = [x[0] for x in value]
        points = remove_duplicate_points(points)
        all_segments_.append(multi_split_segment(segment, points))

    union_ = Group()
    shape_vertices = shape1.vertices
    shape2_vertices = shape2.vertices
    for segs in all_segments_:
        for seg in segs:
            if distance(*seg) < 0.001:
                continue
            in1 = in_polygon(midpoint(*seg), shape_vertices)
            in2 = in_polygon(midpoint(*seg), shape2_vertices)
            if in1 ^ in2:  # only one can be True
                union_.append(Shape(seg))
    if merge:
        union_ = union_.merge_shapes()

    return union_


def polygon_diff(
    shape1: "Shape",
    shape2: "Shape",
    dist_tol: float = 0.01,
    merge: bool = True,
):
    """
    shape1 Shape: shape to be clipped
    shape2 Shape: clipping region
    """
    exclude_clipper = False
    if not (shape1.closed and shape2.closed):
        raise Warning("Both shapes must be closed")

    segments = [[p1[:2], p2[:2]] for (p1, p2) in shape1.edges] + [
        [p1[:2], p2[:2]] for (p1, p2) in shape2.edges
    ]
    intersections = all_intersections(segments, rel_tol=0, abs_tol=dist_tol)

    all_segments_ = []
    for key, value in intersections[0].items():
        segment = segments[key]
        points = [x[0] for x in value]
        points = remove_duplicate_points(points)
        all_segments_.append(multi_split_segment(segment, points))

    diff_ = Group()
    shape_vertices = shape1.vertices
    shape2_vertices = shape2.vertices
    for segs in all_segments_:
        for seg in segs:
            in1 = in_polygon(midpoint(*seg), shape_vertices)
            in2 = in_polygon(
                midpoint(*seg), shape2_vertices, not exclude_clipper
            )
            if in1 and not in2:
                diff_.append(Shape(seg))

    if merge:
        diff_ = diff_.merge_shapes()

    return diff_


def polygon_difference(
    shape1: "Shape",
    shape2: "Shape",
    dist_tol: float = 0.01,
    merge: bool = True,
):
    return polygon_diff(shape1, shape2, exclude_clipper=False)


def polygon_intersection(shape1: "Shape", shape2: "Shape", merge: bool = True):
    """Returns the intersection of two polygons."""
    if not (shape1.closed and shape2.closed):
        raise ValueError("Invalid input: shape1 and shape2 must be closed!")
    return clip(shape1, shape2, merge=merge)


def polygon_xor(
    shape1: "Shape",
    shape2: "Shape",
    dist_tol: float = 0.01,
    merge: bool = True,
):
    """
    shape1 Shape: shape to be clipped
    shape2 Shape: clipping region
    """
    res1 = polygon_diff(shape1, shape2)
    res2 = polygon_diff(shape2, shape1)

    res = Group([res1, res2])

    if merge:
        res = res.merge_shapes()

    return res


def all_segments(
    item: Union[Shape, Group],
    n_round: int = 1,
    rel_tol: float = None,
    abs_tol: float = None,
):
    """
    Get all line segments from a Shape or Group instance.
    Args:
        item (Union[Shape, Group]): The input shape or group.
        n_round (int): The number of decimal places to round segment coordinates.
        rel_tol (float): The relative tolerance for segment comparison.
        abs_tol (float): The absolute tolerance for segment comparison.
    Returns:
        List[LineType]: A list of line segments.
    """

    rel_tol, abs_tol = get_defaults(["rel_tol", "abs_tol"], [rel_tol, abs_tol])
    if isinstance(item, Group):
        shapes = item.all_shapes
    else:
        shapes = [item]
    edges = []
    for shp in shapes:
        edges.extend(shp.edges)
    segments = [[p1[:2], p2[:2]] for (p1, p2) in edges]
    intersections = all_intersections(segments)

    all_segments_ = []
    for key, value in intersections[0].items():
        segment = segments[key]
        points = [x[0] for x in value]
        points = remove_duplicate_points(points)
        all_segments_.append(multi_split_segment(segment, points))

    edges = []
    for segs in all_segments_:
        for seg in segs:
            if distance(*seg) < 0.1:
                continue
            seg = around((seg), n_round)
            seg = (tuple(seg[0]), tuple(seg[1]))
            edges.append(seg)

    return edges


def get_loop(edges: Sequence[LineType], start_edge: LineType, ccw: bool = True):
    """
    Find a loop in a set of edges starting from a given edge.
        Args:
            edges (Sequence[LineType]): The set of edges to search.
            start_edge (LineType): The edge to start the search from.
        Returns:
            Shape: A shape representing the found loop, or an empty shape if no loop is found.
    """
    G = nx.Graph()
    G.add_edges_from(edges)
    if not ccw:
        start_edge = (start_edge[1], start_edge[0])

    res = [*start_edge]
    start_node = start_edge[0]
    cur_node = start_edge[1]
    cur_edge = start_edge
    open_ = True
    while open_:
        edges_cur_node = set(G.edges(cur_node))
        angles = []
        for edge in edges_cur_node:
            if (edge[1], edge[0]) == cur_edge:
                continue
            if edge[1] == start_node:
                open_ = False
                break
            angle = angle_between_lines2(*cur_edge, edge[1])
            angle = positive_angle(angle)
            pi_ = round(pi, 2)
            if round(angle, 2) not in [0, -pi_, pi_, 2 * pi_]:
                angles.append((angle, edge))
        if open_:
            angles.sort()
            if not angles:
                break
            cur_edge = angles[0][1]
            cur_node = cur_edge[1]
            res.append(cur_node)

    return Shape(res, closed=not (open_))


def get_partition(
    item: Union[Shape, Group], edge_index: int, ccw: bool = True
) -> Shape:
    """
    Get a sub-region from a shape or group object.
    Draw the segments by using canvas.draw_all_segments first to get the indices.
    Args:
        item Union[Shape, Group]: A shape or a group object.
        edge_index int: Index of the starting edge of the partition.
        ccw bool: If True, the region is formed by looping in
        counterclockwise direction, clockwise otherwise.

    Returns:
        The resulting shape object.
    """

    edges = all_segments(item)

    return get_loop(edges, edges[edge_index], ccw)

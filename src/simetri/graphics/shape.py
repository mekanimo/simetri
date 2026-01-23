"""Shape objects are the main geometric entities in Simetri.
They are created by providing a sequence of points (a list of (x, y) coordinates).
If a style argument (a ShapeStyle object) is provided, then the style attributes
of this ShapeStyle object will superseed the style attributes of the Shape object.
"""

__all__ = ["Shape", "custom_attributes", "clip", "trim_margins",
           "all_segments", "get_loop", "get_partition", "union", "diff", "xor"]

from typing import Sequence, Union, List, Tuple
from math import pi, isclose

import json
import numpy as np
from numpy import around, array, allclose
from numpy.linalg import inv
import networkx as nx
from typing_extensions import Self

from .affine import identity_matrix
from .all_enums import Types, Topology, shape_types
from .bbox import BoundingBox
from ..canvas.style_map import ShapeStyle, shape_style_map, shape_args
from ..helpers.validation import validate_args
from .common import Point, common_properties, Line, get_defaults
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
)
from ..helpers.graph import Node, Graph, GraphEdge
from .core import Base, StyleMixin
from .bbox import bounding_box
from .points import Points
from .batch import Batch


class Shape(Base, StyleMixin):
    """The main class for all geometric entities in Simetri.

     A Shape is created by providing a sequence of points (a sequence of (x, y) coordinates).
    If a style argument (a ShapeStyle object) is provided, then its style attributes override
    the default values the Shape object would assign. Additional attributes (e.g. line_width, fill_color, line_style)
    may be provided.

    """

    def __init__(
        self,
        points: Sequence[Point] = None,
        closed: bool = False,
        xform_matrix: np.array = None,
        **kwargs,
    ) -> None:
        """Initialize a Shape object.

        Args:
            points (Sequence[Point], optional): The points that make up the shape.
            closed (bool, optional): Whether the shape is closed. Defaults to False.
            xform_matrix (np.array, optional): The transformation matrix. Defaults to None.
            **kwargs (dict): Additional attributes for the shape.

        Raises:
            ValueError: If the provided subtype is not valid.
        """

        self.__dict__["style"] = ShapeStyle()
        self.__dict__["_style_map"] = shape_style_map
        self._set_aliases()
        valid_args = shape_args
        validate_args(kwargs, valid_args)
        self._external = False
        if "subtype" in kwargs:
            if kwargs["subtype"] not in shape_types:
                self._external = True
            self.subtype = kwargs["subtype"]
            kwargs.pop("subtype")
        else:
            self.subtype = Types.SHAPE

        if "dist_tol" in kwargs:
            self.dist_tol = kwargs["dist_tol"]
            self.dist_tol2 = self.dist_tol**2
            kwargs.pop("dist_tol")
        else:
            self.dist_tol = defaults["dist_tol"]
            self.dist_tol2 = self.dist_tol**2

        if points is None:
            self.primary_points = Points()
            self.closed = False
        else:
            self.closed, points = self._get_closed(points, closed)
            self.primary_points = Points(points)
            self.primary_points.nd_array_changed = True
        self.xform_matrix = get_transform(xform_matrix)
        self.type = Types.SHAPE
        for key, value in kwargs.items():
            setattr(self, key, value)

        self._b_box = None
        common_properties(self)

    def __getattr__(self, name):
        """Retrieve an attribute of the shape.

        Args:
            name (str): The attribute name to return.

        Returns:
            Any: The value of the attribute.

        Raises:
            AttributeError: If the attribute cannot be found.
        """
        # Check if it's a style attribute alias first (StyleMixin behavior)
        if hasattr(self, '_aliasses') and name in self._aliasses:
            obj, attrib = self._aliasses[name]
            return getattr(obj, attrib)

        # Try the normal attribute resolution chain
        try:
            res = super().__getattr__(name)
        except AttributeError:
            try:
                res = self.__dict__[name]
            except KeyError as exc:
                raise AttributeError(f"'Shape' object has no attribute '{name}'") from exc
        return res

    def _get_closed(self, points: Sequence[Point], closed: bool):
        """Determine whether the shape should be considered closed.

        Args:
            points (Sequence[Point]): The points that define the shape.
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

    def __getitem__(self, subscript: Union[int, slice]):
        """Retrieve point(s) from the shape by index or slice.

        Args:
            subscript (int or slice): The index or slice specifying the point(s) to retrieve.

        Returns:
            Point or list[Point]: The requested point or list of points (after applying the transformation).

        Raises:
            TypeError: If the subscript type is invalid.
        """
        # Use cached final_coords instead of recalculating matrix multiplication
        final_coords = self.final_coords

        if isinstance(subscript, slice):
            res = [tuple(coord[:2]) for coord in final_coords[subscript]]
        else:
            coord = final_coords[subscript]
            res = (coord[0], coord[1])
        return res

    def __setitem__(self, subscript, value):
        """Set the point(s) at the given subscript.

        Args:
            subscript (int or slice): The subscript to set the point(s) at.
            value (Point or list[Point]): The value to set the point(s) to.

        Raises:
            TypeError: If the subscript type is invalid.
        """
        if isinstance(subscript, slice):
            if is_nested_sequence(value):
                value = homogenize(value) @ inv(self.xform_matrix)
            else:
                value = homogenize([value]) @ inv(self.xform_matrix)
            self.primary_points[subscript.start : subscript.stop : subscript.step] = [
                tuple(x[:2]) for x in value
            ]
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

    def index(self, point: Point, abs_tol=None) -> int:
        """Return the index of the given point.

        Args:
            point (Point): The point to find the index of.
            abs_tol (float, optional): Absolute tolerance for comparison. Defaults to None.

        Returns:
            int: The index of the point.
        """
        point = tuple(point[:2])

        if abs_tol is None:
            abs_tol = defaults["abs_tol"]
        ind = np.where((np.isclose(self.vertices, point, atol=abs_tol)).all(axis=1))[0][0]

        return ind

    def remove(self, point: Point) -> Self:
        """Remove a point from the shape.

        Args:
            point (Point): The point to remove.
        """
        ind = self.vertices.index(point)
        self.primary_points.pop(ind)

        return self

    def append(self, point: Point) -> Self:
        """Append a point to the shape.

        Args:
            point (Point): The point to append.
        """
        point = homogenize([point]) @ inv(self.xform_matrix)
        self.primary_points.append(tuple(point[0][:2]))

    def insert(self, index: int, point: Point) -> Self:
        """Insert a point at a given index.

        Args:
            index (int): The index to insert the point at.
            point (Point): The point to insert.
        """
        point = homogenize([point]) @ inv(self.xform_matrix)
        self.primary_points.insert(index, tuple(point[0][:2]))

        return self

    def extend(self, points: Sequence[Point]) -> Self:
        """Extend the shape with a list of points.

        Args:
            values (list[Point]): The points to extend the shape with.
        """
        homogenized = homogenize(points) @ inv(self.xform_matrix)
        self.primary_points.extend([tuple(x[:2]) for x in homogenized])

        return self

    def pop(self, index: int = -1) -> Point:
        """Pop a point from the shape.

        Args:
            index (int, optional): The index to pop the point from, defaults to -1.

        Returns:
            Point: The popped point.
        """
        point = self.vertices[index]
        self.primary_points.pop(index)

        return point

    def __iter__(self):
        """Return an iterator over the vertices of the shape.

        Returns:
            Iterator[Point]: An iterator over the vertices of the shape.
        """
        return iter(self.vertices)

    def _update(
        self, xform_matrix: array, reps: int = 0, merge: bool = False
    ) -> Union["Shape", Batch]:
        """Used internally. Update the shape with a transformation matrix.

        Args:
            xform_matrix (array): The transformation matrix.
            reps (int, optional): The number of repetitions, defaults to 0.

        Returns:
            Shape or Batch: The updated shape or a batch of shapes.
        """
        if reps == 0:
            fillet_radius = getattr(self, 'fillet_radius', None)
            if fillet_radius:
                scale = max(decompose_transformations(xform_matrix)[2])
                self.fillet_radius = fillet_radius * scale

            self.xform_matrix = self.xform_matrix @ xform_matrix
            # Invalidate coordinate caches when transformation changes
            if '_final_coords' in self.__dict__:
                delattr(self, '_final_coords')
            if '_vertices' in self.__dict__:
                delattr(self, '_vertices')
            res = self
        else:
            shapes = [self]
            shape = self
            for _ in range(reps):
                shape = shape.copy()
                shape._update(xform_matrix)
                shapes.append(shape)
            res = Batch(shapes)

        if merge and reps > 0:
            return res.merge_shapes()

        return res

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
                atol=defaults["abs_tol"]
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
            prim_points = list(self.primary_points) if self.primary_points else []
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
            point (Point): The point to start from.

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

    def topology(self) -> Topology:
        """Return info about the topology of the shape.

        Returns:
            set: A set of topology values.
        """
        t_map = {
            "WITHIN": Topology.FOLDED,
            "CONTAINS": Topology.FOLDED,
            "COLL_CHAIN": Topology.COLLINEAR,
            "YJOINT": Topology.YJOINT,
            "CHAIN": Topology.SIMPLE,
            "CONGRUENT": Topology.CONGRUENT,
            "INTERSECT": Topology.INTERSECTING,
        }
        intersections = all_intersections(self.vertex_pairs, use_intersection3=True)
        connections = []
        for val in intersections.values():
            connections.extend([x[0].value for x in val])
        connections = set(connections)
        topology = set((t_map[x] for x in connections))

        if len(topology) > 1 and Topology.SIMPLE in topology:
            topology.discard(Topology.SIMPLE)

        return topology

    def merge_collinears(self):
        '''Merges collinear edges.'''
        return Batch([self]).merge_shapes()[0]

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
                closed = close_points2(vertices[0], vertices[-1], dist2=self.dist_tol2)
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
        self, verts1: Sequence[Point], verts2: Sequence[Point], dist_tol: float = None
    ) -> Union[List[Point], None]:
        """Chain two sets of vertices if they are connected.

        Args:
            verts1 (list[Point]): The first set of vertices.
            verts2 (list[Point]): The second set of vertices.
            dist_tol (float, optional): The distance tolerance for chaining, defaults to None.

        Returns:
            list[Point] or None: The chained vertices or None if the vertices cannot be chained.
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

    def _is_polygon(self, vertices: Sequence[Point]) -> bool:
        """Return True if the vertices form a polygon.

        Args:
            vertices (list[Point]): The vertices to check.

        Returns:
            bool: True if the vertices form a polygon, False otherwise.
        """
        return close_points2(vertices[0][:2], vertices[-1][:2], dist2=self.dist_tol2)

    def as_graph(self, directed=False, weighted=False, n_round=None, cycles=False) -> nx.Graph:
        """Return the shape as a graph object.

        Args:
            directed (bool, optional): Whether the graph is directed, defaults to False.
            weighted (bool, optional): Whether the graph is weighted, defaults to False.
            n_round (int, optional): The number of decimal places to round to, defaults to None.

            cycles (bool, optional): If True, cycles are returned.

        Returns:
            Graph: The graph object.
        """
        if n_round is None:
            n_round = defaults["n_round"]
        vertices = [(round(v[0], n_round), round(v[1], n_round)) for v in self.vertices]
        points = [Node(*n) for n in vertices]
        pairs = connected_pairs(points)
        edges = [GraphEdge(p[0], p[1]) for p in pairs]
        if self.closed:
            edges.append(GraphEdge(points[-1], points[0]))

        if directed:
            nx_graph = nx.DiGraph()
            graph_type = Types.DIRECTED
        else:
            nx_graph = nx.Graph()
            graph_type = Types.UNDIRECTED

        for point in points:
            nx_graph.add_node(point.id, point=point)

        if weighted:
            for edge in edges:
                nx_graph.add_edge(edge.start.id, edge.end.id, weight=edge.length)
            subtype = Types.WEIGHTED
        else:
            id_pairs = [(e.start.id, e.end.id) for e in edges]
            nx_graph.add_edges_from(id_pairs)
            subtype = Types.NONE
        pairs = [(e.start.id, e.end.id) for e in edges]
        try:
            cycles_ = nx.cycle_basis(nx_graph)
        except nx.exception.NetworkXNoCycle:
            cycles_ = None

        if cycles:
            n = len(cycles)
            for cycle in cycles_:
                cycle.append(cycle[0])
            if n == 1:
                cycle = cycles[0]
        else:
            cycles_ = None

        if cycles:
            return cycles_

        graph = Graph(type=graph_type, subtype=subtype, nx_graph=nx_graph)
        return graph

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

    def as_list(self) -> List[Point]:
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
            if '_final_coords' not in self.__dict__ or self.primary_points.nd_array_changed:
                self._final_coords = self.primary_points.homogen_coords @ self.xform_matrix
            res = self._final_coords
        else:
            res = array([])

        return res

    @property
    def vertices(self) -> Tuple[Point]:
        """The final coordinates of the shape.

        Returns:
            tuple: The final coordinates of the shape.
        """

        if self.primary_points:
            # Cache vertices computation and only recompute when data changes
            if '_vertices' not in self.__dict__ or self.primary_points.nd_array_changed:
                res = tuple(((x[0], x[1]) for x in (self.final_coords[:, :2])))
                self._vertices = res
                self.primary_points.nd_array_changed = False
            else:
                res = self._vertices
        else:
            res = ()

        return res

    @property
    def vertex_pairs(self) -> List[Tuple[Point, Point]]:
        """Return a list of connected pairs of vertices.

        Returns:
            list[tuple[Point, Point]]: A list of connected pairs of vertices.
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
            if not close_points2(vertices[0], vertices[-1], dist2=self.dist_tol2):
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
        # self.style = ShapeStyle()
        self._set_aliases()
        self._b_box = None
        # Clear coordinate caches
        if '_final_coords' in self.__dict__:
            delattr(self, '_final_coords')
        if '_vertices' in self.__dict__:
            delattr(self, '_vertices')

        return self

    def count(self, point: Point) -> int:
        """Return the number of times the point is found in the shape.

        Args:
            point (Point): The point to count.

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

        return np.count_nonzero(distances <= self.dist_tol2)

    def copy(self) -> "Shape":
        """Return a copy of the shape.

        Returns:
            Shape: A copy of the shape.
        """
        if self.primary_points.coords:
            points = self.primary_points.copy()
        else:
            points = []
        marker_type = getattr(self, 'marker_type', defaults.get('marker_type'))
        shape = Shape(
            points,
            xform_matrix=self.xform_matrix.copy(),
            closed=self.closed,
            marker_type=marker_type,
        )
        for attrib in shape_style_map:
            value = getattr(self, attrib, defaults.get(attrib))
            if value is not None:
                setattr(shape, attrib, value)
        shape.subtype = self.subtype
        custom_attribs = custom_attributes(self)
        for attrib in custom_attribs:
            setattr(shape, attrib, getattr(self, attrib))

        return shape

    @property
    def edges(self) -> List[Line]:
        """Return a list of edges.

        Edges are represented as tuples of points:
        edge: ((x1, y1), (x2, y2))
        edges: [((x1, y1), (x2, y2)), ((x2, y2), (x3, y3)), ...]

        Returns:
            list[tuple[Point, Point]]: A list of edges.
        """
        vertices = list(self.vertices[:])
        if self.closed:
            vertices.append(vertices[0])

        return tuple(connected_pairs(vertices))

    @property
    def segments(self) -> List[Line]:
        """Return a list of edges.

        Edges are represented as tuples of points:
        edge: ((x1, y1), (x2, y2))
        edges: [((x1, y1), (x2, y2)), ((x2, y2), (x3, y3)), ...]

        Returns:
            list[tuple[Point, Point]]: A list of edges.
        """

        return self.edges

    def reverse(self) -> Self:
        """Reverse the order of the vertices.

        Returns:
            None
        """
        self.primary_points.reverse()

        return self

    def reorder_vertices(
        self, value: Point, index: int = 0, tol: float = None
    ) -> Union["Shape", None]:
        """If index is not given, the vertex with the given value will be
        the first index.
        If index is given, the vertex with the given value will be
        at the given index.
        The rest of the indices will be shifted accordingly.

        Shape must be closed.

        Args:
            index (int): The target index.
            value (Point): The vertex to relocate at the given index.

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

def trim_margins(item: Union[Shape, Batch], left: float = 0, bottom: float = 0,
                            right: float = 0, top: float = 0) -> Union[Shape, Batch]:
    """Trim the margins of a Shape or Batch.

    Args:
        item (Union[Shape, Batch]): The Shape or Batch to trim.
        left (float, optional): The left margin to trim. Defaults to 0.
        bottom (float, optional): The bottom margin to trim. Defaults to 0.
        right (float, optional): The right margin to trim. Defaults to 0.
        top (float, optional): The top margin to trim. Defaults to 0.

    Returns:
        Union[Shape, Batch]: The trimmed Shape or Batch.
    """
    corners = item.b_box.get_inflated_b_box(-left, -bottom, -right, -top).corners
    clipper = Shape(corners, closed=True)

    return clip(item, clipper, exclude_clipper=True)

def clip(item: Union[Shape, Batch], clipper: Shape, exclude_clipper: bool=False,
                                        rel_tol:float=None, abs_tol:float=None):
    if isinstance(item, Batch):
        return _clip_batch(item, clipper, exclude_clipper, rel_tol, abs_tol)
    elif isinstance(item, Shape):
        return _clip_shape(item, clipper, exclude_clipper, rel_tol, abs_tol)
    else:
        raise TypeError("Invalid item type")

def _clip_batch(batch:Batch, clipper:Shape, exclude_clipper: bool = True,
                                    rel_tol:float=None, abs_tol:float=None):
    '''
    batch Batch: batch to be clipped
    clipper Shape: clipping region
    exclude_clipper bool: If True, clipper's edges are excluded.
    '''
    res = Batch()
    for shp in batch.all_shapes:
        res.append(_clip_shape(shp, clipper, exclude_clipper, rel_tol, abs_tol))

    return res

def _clip_shape(shape: 'Shape', clipper: 'Shape', exclude_clipper: bool = False,
                                    rel_tol:float=None, abs_tol:float=None):
    '''
    shape Shape: shape to be clipped
    clipper Shape: clipping region
    exclude_clipper bool: If True, clipper's edges are excluded.
    '''
    if not clipper.closed:
        raise Warning("Clipper shape is not closed")
    rel_tol, abs_tol = get_defaults(['rel_tol', 'abs_tol'], [rel_tol, abs_tol])
    n_shape = len(shape)
    segments = ([[p1[:2], p2[:2]] for (p1, p2) in shape.edges] +
                [[p1[:2], p2[:2]] for (p1, p2) in clipper.edges])
    intersections = all_intersections(segments)

    all_segments_ = []
    for key, value in intersections[0].items():
        segment = segments[key]
        points = [x[0] for x in value]
        points = remove_duplicate_points(points)
        all_segments_.append(multi_split_segment(segment, points))

    clipped = Batch()
    shape_vertices = shape.vertices
    polygon = clipper.vertices
    if shape.closed:
        max_i = n_shape
    else:
        max_i = n_shape - 1
    for i, segs in enumerate(all_segments_):
        if i >= max_i:
            if not shape.closed or exclude_clipper:
                break
            polygon = shape_vertices
        for seg in segs:
            if not isclose(distance(*seg), 0, rel_tol=rel_tol, abs_tol=abs_tol):
                if in_polygon(midpoint(*seg), polygon, exclude_clipper):
                    clipped.append(Shape(seg))

    clipped = clipped.merge_shapes()
    if len(clipped) == 1:
        clipped = clipped[0]

    return clipped


def custom_attributes(item: Shape) -> List[str]:
    """Return a list of custom attributes of a Shape or Batch instance.

    Args:
        item (Shape): The Shape or Batch instanc
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
    custom_attribs = set(dir(item)) - native_attribs

    return list(custom_attribs)

def union(shape1: 'Shape', shape2: 'Shape'):
    '''
    shape1 Shape: shape to be clipped
    shape2 Shape: clipping region
    '''
    if not (shape1.closed and shape2.closed):
        raise Warning("Both shapes must be closed")

    segments = ([[p1[:2], p2[:2]] for (p1, p2) in shape1.edges] +
                [[p1[:2], p2[:2]] for (p1, p2) in shape2.edges])
    intersections = all_intersections(segments)

    all_segments_ = []
    for key, value in intersections[0].items():
        segment = segments[key]
        points = [x[0] for x in value]
        points = remove_duplicate_points(points)
        all_segments_.append(multi_split_segment(segment, points))

    union_ = Batch()
    shape_vertices = shape1.vertices
    shape2_vertices = shape2.vertices
    for segs in all_segments_:
        for seg in segs:
            in1 = in_polygon(midpoint(*seg), shape_vertices)
            in2 = in_polygon(midpoint(*seg), shape2_vertices)
            if in1 ^ in2: # only one can be True
                union_.append(Shape(seg))

    return union_

def diff(shape1: 'Shape', shape2: 'Shape', exclude_clipper: bool = False,
                                                    dist_tol: float =.01):
    '''
    shape1 Shape: shape to be clipped
    shape2 Shape: clipping region
    exclude_clipper bool: If True, clipper's edges are excluded.
    '''
    if not (shape1.closed and shape2.closed):
        raise Warning("Both shapes must be closed")

    segments = ([[p1[:2], p2[:2]] for (p1, p2) in shape1.edges] +
                [[p1[:2], p2[:2]] for (p1, p2) in shape2.edges])
    intersections = all_intersections(segments)

    all_segments_ = []
    for key, value in intersections[0].items():
        segment = segments[key]
        points = [x[0] for x in value]
        points = remove_duplicate_points(points)
        all_segments_.append(multi_split_segment(segment, points))

    diff_ = Batch()
    shape_vertices = shape1.vertices
    shape2_vertices = shape2.vertices
    for segs in all_segments_:
        for seg in segs:
            in1 = in_polygon(midpoint(*seg), shape_vertices)
            in2 = in_polygon(midpoint(*seg), shape2_vertices, not exclude_clipper)
            if in1 and not in2:
                diff_.append(Shape(seg))

    # diff_ = diff_.merge_shapes()
    # if len(diff_) == 1:
    #     diff_ = diff_[0]

    return diff_

def xor(shape1: 'Shape', shape2: 'Shape', exclude_clipper: bool = False,
                                                    dist_tol: float =.01):
    '''
    shape1 Shape: shape to be clipped
    shape2 Shape: clipping region
    exclude_clipper bool: If True, clipper's edges are excluded.
    '''
    res1 = diff(shape1, shape2, exclude_clipper)
    res2 = diff(shape2, shape1, exclude_clipper)

    return Batch([res1, res2])


def all_segments(item: Union[Shape, Batch], n_round: int = 1,
                 rel_tol: float = None, abs_tol: float = None):
    '''
        Get all line segments from a Shape or Batch instance.
        Args:
            item (Union[Shape, Batch]): The input shape or batch.
            n_round (int): The number of decimal places to round segment coordinates.
            rel_tol (float): The relative tolerance for segment comparison.
            abs_tol (float): The absolute tolerance for segment comparison.
        Returns:
            List[Line]: A list of line segments.
    '''

    rel_tol, abs_tol = get_defaults(['rel_tol', 'abs_tol'], [rel_tol, abs_tol])
    if isinstance(item, Batch):
        shapes = item.all_shapes
    else:
        shapes = [item]
    edges = []
    for shp in shapes:
        edges.extend(shp.edges)
    segments = ([[p1[:2], p2[:2]] for (p1, p2) in edges])
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
            if distance(*seg) < .1:
                continue
            seg = around((seg), n_round)
            seg = (tuple(seg[0]), tuple(seg[1]))
            edges.append(seg)

    return edges

def get_loop(edges: Sequence[Line], start_edge: Line, ccw: bool=True):
    '''
    Find a loop in a set of edges starting from a given edge.
        Args:
            edges (Sequence[Line]): The set of edges to search.
            start_edge (Line): The edge to start the search from.
        Returns:
            Shape: A shape representing the found loop, or an empty shape if no loop is found.
    '''
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
            if round(angle, 2) not in [0, -pi_, pi_, 2*pi_]:
                angles.append((angle, edge))
        if open_:
            angles.sort()
            if not angles:
                break
            cur_edge = angles[0][1]
            cur_node = cur_edge[1]
            res.append(cur_node)

    return Shape(res, closed=not(open_))

def get_partition(item: Union[Shape, Batch], edge_index: int, ccw: bool=True) -> Shape:
    '''
        Get a sub-region from a shape or batch object.
        Draw the segments by using canvas.draw_all_segments first to get the indices.
        Args:
            item Union[Shape, Batch]: A shape or a batch object.
            edge_index int: Index of the starting edge of the partition.
            ccw bool: If True, the region is formed by looping in
            counterclockwise direction, clockwise otherwise.

        Returns:
            The resulting shape object.
    '''

    edges = all_segments(item)

    return get_loop(edges, edges[edge_index], ccw)

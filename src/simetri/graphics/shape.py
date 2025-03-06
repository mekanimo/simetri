"""Shape objects are the main geometric entities in Simetri. They are created by providing a sequence of points (a list of (x, y) coordinates). If a style argument (a ShapeStyle object) is provided, then the style attributes of this ShapeStyle object will superseed the style attributes of the Shape object. The dist_tol argument is the distance tolerance for checking. The xform_matrix argument is the transformation matrix. Additional attributes can be provided as keyword arguments. The line_width, fill_color, line_style, etc. for style attributes can be provided as keyword arguments. The Shape object has a subtype attribute that can be set to one of the values in the shape_types dictionary. The Shape object has a dist_tol attribute that is the distance tolerance for checking. The Shape object has a dist_tol2 attribute that is the square of the distance tolerance. The Shape object has a primary_points attribute that is a Points object. The Shape object has a closed attribute that is a boolean value. The Shape object has an xform_matrix attribute that is a transformation matrix. The Shape object has a type attribute that is a Types.SHAPE object. The Shape object has a subtype attribute that is a Types.SHAPE object. The Shape object has a dist_tol attribute that is a distance tolerance for checking. The Shape object has a dist_tol2 attribute that is the square of the distance tolerance. The Shape object has a _b_box attribute that is a bounding box. The Shape object has a area attribute that is the area of the shape. The Shape object has a total_length attribute that is the total length of the shape. The Shape object has a is_polygon attribute that is a boolean value. The Shape object has a topology attribute that is a set of topology values. The Shape object has a merge method that merges two shapes if they are connected. The Shape object has a _chain_vertices method that chains two sets of vertices if they are connected. The Shape object has a _is_polygon method that returns True if the vertices form a polygon. The Shape object has an as_graph method that returns the shape as a graph object. The Shape object has an as_array method that returns the vertices as an array. The Shape object has an as_list method that returns the vertices as a list of tuples. The Shape object has a final_coords attribute that is the final coordinates of the shape. The Shape object has a vertices attribute that is the final coordinates of the shape. The Shape object has a vertex"""

__all__ = ["Shape", "custom_attributes"]

from typing import Sequence, Union, List

import numpy as np
from numpy import array, allclose
from numpy.linalg import inv
import networkx as nx
from typing_extensions import Self

from .affine import identity_matrix
from .all_enums import *
from ..canvas.style_map import ShapeStyle, shape_style_map
from ..helpers.validation import validate_args
from .common import Point, common_properties, get_item_by_id, Line
from ..canvas.style_map import shape_args
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
)
from ..helpers.graph import Node, Graph, GraphEdge
from .core import Base
from .bbox import bounding_box
from .points import Points
from .batch import Batch


class Shape(Base):
    """The main class for all geometric entities in Simetri.

    A Shape is created by providing a sequence of points (a list of (x, y) coordinates).
    If a style argument (a ShapeStyle object) is provided, then its style attributes override
    those of the Shape object. Additional attributes (e.g. line_width, fill_color, line_style)
    may be provided.

    Attributes:
        dist_tol: Distance tolerance for checking.
        xform_matrix: Transformation matrix.
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
        if "subtype" in kwargs:
            if kwargs["subtype"] not in shape_types:
                raise ValueError(f"Invalid subtype: {kwargs['subtype']}")
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
        self.xform_matrix = get_transform(xform_matrix)
        self.type = Types.SHAPE
        for key, value in kwargs.items():
            setattr(self, key, value)

        self._b_box = None
        common_properties(self)

    def __setattr__(self, name, value):
        """Set an attribute of the shape.

        Args:
            name (str): The name of the attribute.
            value (Any): The value to set.
        """
        obj, attrib = self.__dict__["_aliasses"].get(name, (None, None))
        if obj:
            setattr(obj, attrib, value)
        else:
            self.__dict__[name] = value

    def __getattr__(self, name):
        """Retrieve an attribute of the shape.

        Args:
            name (str): The attribute name to return.

        Returns:
            Any: The value of the attribute.

        Raises:
            AttributeError: If the attribute cannot be found.
        """
        obj, attrib = self.__dict__["_aliasses"].get(name, (None, None))
        if obj:
            res = getattr(obj, attrib)
        else:
            try:
                res = super().__getattr__(name)
            except AttributeError:
                res = self.__dict__[name]
        return res

    def _set_aliases(self):
        """Set aliases for style attributes based on the style map."""
        _aliasses = {}
        for alias, path_attrib in self._style_map.items():
            style_path, attrib = path_attrib
            obj = self
            for attrib_name in style_path.split("."):
                obj = obj.__dict__[attrib_name]
            _aliasses[alias] = (obj, attrib)
        self.__dict__["_aliasses"] = _aliasses

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
        decision_table = {
            (True, True): True,
            (True, False): True,
            (False, True): True,
            (False, False): False,
        }
        n = len(points)
        if n < 3:
            res = False
        else:
            points = [tuple(x[:2]) for x in points]
            polygon = self._is_polygon(points)
            res = decision_table[(bool(closed), polygon)]
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
        if isinstance(subscript, slice):
            coords = self.primary_points.homogen_coords
            res = list(coords[subscript.start : subscript.stop : subscript.step] @ self.xform_matrix)
        else:
            res = self.primary_points.homogen_coords[subscript] @ self.xform_matrix
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
        elif isinstance(subscript, int):
            value = homogenize([value]) @ inv(self.xform_matrix)
            self.primary_points[subscript] = tuple(value[0][:2])
        else:
            raise TypeError("Invalid subscript type")

    def __delitem__(self, subscript) -> Self:
        """Delete the point(s) at the given subscript.

        Args:
            subscript (int or slice): The subscript to delete the point(s) from.
        """
        del self.primary_points[subscript]

    def remove(self, value):
        """Remove a point from the shape.

        Args:
            value (Point): The point to remove.
        """
        ind = self.vertices.index(value)
        self.primary_points.pop(ind)

    def append(self, value):
        """Append a point to the shape.

        Args:
            value (Point): The point to append.
        """
        value = homogenize([value]) @ inv(self.xform_matrix)
        self.primary_points.append(tuple(value[0][:2]))

    def insert(self, index, value):
        """Insert a point at a given index.

        Args:
            index (int): The index to insert the point at.
            value (Point): The point to insert.
        """
        value = homogenize([value]) @ inv(self.xform_matrix)
        self.primary_points.insert(index, tuple(value[0][:2]))

    def extend(self, values):
        """Extend the shape with a list of points.

        Args:
            values (list[Point]): The points to extend the shape with.
        """
        homogenized = homogenize(values) @ inv(self.xform_matrix)
        self.primary_points.extend([tuple(x[:2]) for x in homogenized])

    def pop(self, index: int = -1):
        """Pop a point from the shape.

        Args:
            index (int, optional): The index to pop the point from, defaults to -1.

        Returns:
            Point: The popped point.
        """
        value = self.vertices[index]
        self.primary_points.pop(index)
        return value

    def __iter__(self):
        """Return an iterator over the vertices of the shape.

        Returns:
            Iterator[Point]: An iterator over the vertices of the shape.
        """
        return iter(self.vertices)

    def _update(self, xform_matrix: array, reps: int = 0) -> Batch:
        """Used internally. Update the shape with a transformation matrix.

        Args:
            xform_matrix (array): The transformation matrix.
            reps (int, optional): The number of repetitions, defaults to 0.

        Returns:
            Batch: The updated shape or a batch of shapes.
        """
        if reps == 0:
            fillet_radius = self.fillet_radius
            if fillet_radius:
                scale = max(decompose_transformations(xform_matrix)[2])
                self.fillet_radius = fillet_radius * scale
            self.xform_matrix = self.xform_matrix @ xform_matrix
            res = self
        else:
            shapes = [self]
            shape = self
            for _ in range(reps):
                shape = shape.copy()
                shape._update(xform_matrix)
                shapes.append(shape)
            res = Batch(shapes)
        return res

    def __eq__(self, other):
        """Check if the shape is equal to another shape.

        Args:
            other (Shape): The other shape to compare to.

        Returns:
            bool: True if the shapes are equal, False otherwise.
        """
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
                rtol=defaults["rtol"],
                atol=defaults["atol"],
            ) and allclose(self.primary_points.nd_array, other.primary_points.nd_array)
        else:
            res = False

        return res

    def __bool__(self):
        """Return whether the shape has any points.

        Returns:
            bool: True if the shape has points, False otherwise.
        """
        return len(self.primary_points) > 0

    def topology(self):
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

    def merge(self, other, dist_tol: float = None):
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

    def connect(self, other):
        """Connect two shapes by adding the other shape's vertices to self.

        Args:
            other (Shape): The other shape to connect.
        """
        self.extend(other.vertices)

    def _chain_vertices(self, verts1, verts2, dist_tol: float = None):
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
        same_ends = close_points2(end1, end2, dist2=self.dist_tol2)
        if same_starts and same_ends:
            res = verts1
        elif close_points2(end1, start2, dist2=self.dist_tol2):
            verts2.pop(0)
        elif close_points2(start1, end2, dist2=self.dist_tol2):
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

    def _is_polygon(self, vertices):
        """Return True if the vertices form a polygon.

        Args:
            vertices (list[Point]): The vertices to check.

        Returns:
            bool: True if the vertices form a polygon, False otherwise.
        """
        return close_points2(vertices[0][:2], vertices[-1][:2], dist2=self.dist_tol2)

    def as_graph(self, directed=False, weighted=False, n_round=None):
        """Return the shape as a graph object.

        Args:
            directed (bool, optional): Whether the graph is directed, defaults to False.
            weighted (bool, optional): Whether the graph is weighted, defaults to False.
            n_round (int, optional): The number of decimal places to round to, defaults to None.

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
            cycles = nx.cycle_basis(nx_graph)
        except nx.exception.NetworkXNoCycle:
            cycles = None

        if cycles:
            n = len(cycles)
            for cycle in cycles:
                cycle.append(cycle[0])
            if n == 1:
                cycle = cycles[0]
        else:
            cycles = None
        graph = Graph(type=graph_type, subtype=subtype, nx_graph=nx_graph)
        return graph

    def as_array(self, homogeneous=False):
        """Return the vertices as an array.

        Args:
            homogeneous (bool, optional): Whether to return homogeneous coordinates, defaults to False.

        Returns:
            ndarray: The vertices as an array.
        """
        if homogeneous:
            res = self.primary_points.nd_array @ self.xform_matrix
        else:
            res = array(self.vertices)
        return res

    def as_list(self):
        """Return the vertices as a list of tuples.

        Returns:
            list[tuple]: The vertices as a list of tuples.
        """
        return list(self.vertices)

    @property
    def final_coords(self):
        """The final coordinates of the shape. primary_points @ xform_matrix.

        Returns:
            ndarray: The final coordinates of the shape.
        """
        if self.primary_points:
            res = self.primary_points.homogen_coords @ self.xform_matrix
        else:
            res = []

        return res

    @property
    def vertices(self):
        """The final coordinates of the shape.

        Returns:
            tuple: The final coordinates of the shape.
        """
        if self.primary_points:
            res = tuple(((x[0], x[1]) for x in (self.final_coords[:, :2])))
        else:
            res = []

        return res

    @property
    def vertex_pairs(self):
        """Return a list of connected pairs of vertices.

        Returns:
            list[tuple[Point, Point]]: A list of connected pairs of vertices.
        """
        vertices = list(self.vertices)
        if self.closed:
            vertices.append(vertices[0])
        return connected_pairs(vertices)

    @property
    def orig_coords(self):
        """The primary points in homogeneous coordinates.

        Returns:
            ndarray: The primary points in homogeneous coordinates.
        """
        return self.primary_points.homogen_coords

    @property
    def b_box(self):
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
    def area(self):
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
    def total_length(self):
        """Return the total length of the shape.

        Returns:
            float: The total length of the shape.
        """
        return polyline_length(self.vertices[:-1], self.closed)

    @property
    def is_polygon(self):
        """Return True if 'closed'.

        Returns:
            bool: True if the shape is closed, False otherwise.
        """
        return self.closed

    def clear(self):
        """Clear all points and reset the style attributes.

        Returns:
            None
        """
        self.primary_points = Points()
        self.xform_matrix = identity_matrix()
        self.style = ShapeStyle()
        self._set_aliases()
        self._b_box = None

    def count(self, value):
        """Return the number of times the value is found in the shape.

        Args:
            value (Point): The value to count.

        Returns:
            int: The number of times the value is found in the shape.
        """
        verts = self.orig_coords @ self.xform_matrix
        verts = verts[:, :2]
        n = verts.shape[0]
        value = array(value[:2])
        values = np.tile(value, (n, 1))
        col1 = (verts[:, 0] - values[:, 0]) ** 2
        col2 = (verts[:, 1] - values[:, 1]) ** 2
        distances = col1 + col2

        return np.count_nonzero(distances <= self.dist_tol2)

    def copy(self):
        """Return a copy of the shape.

        Returns:
            Shape: A copy of the shape.
        """
        if self.primary_points.coords:
            points = self.primary_points.copy()
        else:
            points = []
        shape = Shape(
            points,
            xform_matrix=self.xform_matrix,
            closed=self.closed,
            marker_type=self.marker_type,
        )

        for attrib in shape_style_map:
            setattr(shape, attrib, getattr(self, attrib))
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

        return connected_pairs(vertices)

    def reverse(self):
        """Reverse the order of the vertices.

        Returns:
            None
        """
        self.primary_points.reverse()


def custom_attributes(item: Shape) -> List[str]:
    """Return a list of custom attributes of a Shape or Batch instance.

    Args:
        item (Shape): The Shape or Batch instance.

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

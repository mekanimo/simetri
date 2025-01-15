"""Shape objects are the main geometric entities in Simetri. They are created by providing a sequence of points (a list of (x, y) coordinates). If a style argument (a ShapeStyle object) is provided, then the style attributes of this ShapeStyle object will superseed the style attributes of the Shape object. The dist_tol argument is the distance tolerance for checking. The xform_matrix argument is the transformation matrix. Additional attributes can be provided as keyword arguments. The line_width, fill_color, line_style, etc. for style attributes can be provided as keyword arguments. The Shape object has a subtype attribute that can be set to one of the values in the shape_types dictionary. The Shape object has a dist_tol attribute that is the distance tolerance for checking. The Shape object has a dist_tol2 attribute that is the square of the distance tolerance. The Shape object has a primary_points attribute that is a Points object. The Shape object has a closed attribute that is a boolean value. The Shape object has an xform_matrix attribute that is a transformation matrix. The Shape object has a type attribute that is a Types.SHAPE object. The Shape object has a subtype attribute that is a Types.SHAPE object. The Shape object has a dist_tol attribute that is a distance tolerance for checking. The Shape object has a dist_tol2 attribute that is the square of the distance tolerance. The Shape object has a _b_box attribute that is a bounding box. The Shape object has a area attribute that is the area of the shape. The Shape object has a total_length attribute that is the total length of the shape. The Shape object has a is_polygon attribute that is a boolean value. The Shape object has a topology attribute that is a set of topology values. The Shape object has a merge method that merges two shapes if they are connected. The Shape object has a _chain_vertices method that chains two sets of vertices if they are connected. The Shape object has a _is_polygon method that returns True if the vertices form a polygon. The Shape object has an as_graph method that returns the shape as a graph object. The Shape object has an as_array method that returns the vertices as an array. The Shape object has an as_list method that returns the vertices as a list of tuples. The Shape object has a final_coords attribute that is the final coordinates of the shape. The Shape object has a vertices attribute that is the final coordinates of the shape. The Shape object has a vertex"""

__all__=["Shape", "custom_attributes"]

from typing import Sequence, Union, List
import logging

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
from ..helpers.geometry import (
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
    Created by providing a sequence of points (a list of (x, y)
    coordinates).
    If style argument (a ShapeStyle object) is provided, then style attributes of this
    ShapeStyle object will superseed the style attributes of the Shape object.
    dist_tol: distance tolerance for checking
    xform_matrix: transformation matrix
    kwargs: additional attributes
    line_width, fill_color line_style, etc. for style attributes
    """

    def __init__(
        self,
        points: Sequence[Point] = None,
        closed: bool = False,
        xform_matrix: np.array = None,
        **kwargs,
    ) -> None:
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
        obj, attrib = self.__dict__["_aliasses"].get(name, (None, None))
        if obj:
            setattr(obj, attrib, value)
        else:
            self.__dict__[name] = value

    def __getattr__(self, name):
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
        _aliasses = {}

        for alias, path_attrib in self._style_map.items():
            style_path, attrib = path_attrib
            obj = self
            for attrib_name in style_path.split("."):
                obj = obj.__dict__[attrib_name]

            _aliasses[alias] = (obj, attrib)
        self.__dict__["_aliasses"] = _aliasses

    def _get_closed(self, points: Sequence[Point], closed: bool):
        # closed, same start and end points => self.closed
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
        return len(self.primary_points)

    def __str__(self):
        if len(self.primary_points) == 0:
            res = "Shape()"
        elif len(self.primary_points) < 4:
            res = f"Shape({self.vertices})"
        else:
            res = f"Shape([{self.vertices[0]}, ..., {self.vertices[-1]}])"
        return res

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, subscript: Union[int, slice]):
        if isinstance(subscript, slice):
            res = list(self.vertices[subscript.start : subscript.stop : subscript.step])
        else:
            res = self.vertices[subscript]

        return res

    def __setitem__(self, subscript, value):
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
        del self.primary_points[subscript]

    def remove(self, value):
        """Remove a point from the shape."""
        ind = self.vertices.index(value)
        self.primary_points.pop(ind)

    def append(self, value):
        """Append a point to the shape."""
        value = homogenize([value]) @ inv(self.xform_matrix)
        self.primary_points.append(tuple(value[0][:2]))

    def insert(self, index, value):
        """Insert a point at a given index."""
        value = homogenize([value]) @ inv(self.xform_matrix)
        self.primary_points.insert(index, tuple(value[0][:2]))

    def extend(self, values):
        """Extend the shape with a list of points."""
        for value in values:
            value = homogenize([value]) @ inv(self.xform_matrix)
            self.primary_points.append(tuple(value[0][:2]))

    def pop(self, index: int = -1):
        """Pop a point from the shape."""
        value = self.vertices[index]
        self.primary_points.pop(index)
        return value

    def __iter__(self):
        return iter(self.vertices)

    def _update(self, xform_matrix: array, reps: int = 0) -> Batch:
        """Used internally. Update the shape with a transformation matrix."""
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
        if isinstance(other, Shape) and len(self) == len(other):
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
        return len(self.primary_points) > 0

    def topology(self):
        """Return info about the topology of the shape."""
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

    def _chain_vertices(self, verts1, verts2, dist_tol: float = None):
        """Chain two sets of vertices if they are connected."""
        dist_tol2 = dist_tol * dist_tol
        start1, end1 = verts1[0], verts1[-1]
        start2, end2 = verts2[0], verts2[-1]
        same_starts = close_points2(start1, start2, dist2=dist_tol2)
        same_ends = close_points2(end1, end2, dist2=self.dist_tol2)
        if same_starts and same_ends:
            # verts1.pop(-1)
            # verts2.reverse()
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
        """Return True if the vertices form a polygon."""
        return close_points2(vertices[0][:2], vertices[-1][:2], dist2=self.dist_tol2)

    def as_graph(self, directed=False, weighted=False, n_round=None):
        """Return the shape as a graph object."""
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
                # nx_graph.add_edge(edge.start.id, edge.end.id, weight=edge.length)
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
            msg = "No cycles found in 'batch.as_graph'."
            logging.warning(msg)
            cycles = None

        if cycles:
            n = len(cycles)
            for cycle in cycles:
                cycle.append(cycle[0])
            if n == 1:
                cycle = cycles[0]
                msg = (
                    f"Shape.as_graph found one cycle:"
                    f"{[get_item_by_id(node).pos for node in cycle]}"
                )
                logging.info(msg)
            else:
                msg = f"Shape.as_graph found {n} cycles:"
                for cycle in cycles:
                    msg += f"{[get_item_by_id(node).pos for node in cycle]}"
                logging.info(msg)
        else:
            cycles = None
        graph = Graph(type=graph_type, subtype=subtype, nx_graph=nx_graph)
        return graph

    def as_array(self, homogeneous=False):
        """Return the vertices as an array."""
        if homogeneous:
            res = self.primary_points.nd_array @ self.xform_matrix
        else:
            res = array(self.vertices)
        return res

    def as_list(self):
        """Return the vertices as a list of tuples."""
        return list(self.vertices)

    @property
    def final_coords(self):
        """The final coordinates of the shape. primary_points @ xform_matrix."""
        if self.primary_points:
            res = self.primary_points.homogen_coords @ self.xform_matrix
        else:
            res = []

        return res

    @property
    def vertices(self):
        """The final coordinates of the shape."""
        if self.primary_points:
            res = tuple(
                (
                    (round(x[0], defaults["n_round"]), round(x[1], defaults["n_round"]))
                    for x in (self.final_coords[:, :2])
                )
            )
        else:
            res = []

        return res

    @property
    def vertex_pairs(self):
        """Return a list of connected pairs of vertices."""
        return connected_pairs(self.vertices)

    @property
    def orig_coords(self):
        """The primary points in homogeneous coordinates."""
        return self.primary_points.homogen_coords

    @property
    def b_box(self):
        """Return the bounding box of the shape."""
        self._b_box = bounding_box(self.final_coords)
        return self._b_box

    @property
    def area(self):
        """Return the area of the shape."""
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
        """Return the total length of the shape."""
        return polyline_length(self.vertices[:-1], self.closed)

    @property
    def is_polygon(self):
        """Return True if the last vertex is the same as the first.
        Ignores the 'closed' attribute."""
        return close_points2(self.vertices[0], self.vertices[-1], dist2=self.dist_tol2)

    def clear(self):
        """Clear all points and reset the style attributes."""
        self.primary_points = Points()
        self.xform_matrix = identity_matrix()
        self.style = ShapeStyle()
        self._b_box = None

    def count(self, value):
        """Return the number of times the value is found in the shape."""
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
        """Return a copy of the shape."""
        points = self.primary_points.copy()
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
        """
        Return a list of edges.
        edge: ((x1, y1), (x2, y2))
        edges: [((x1, y1), (x2, y2)), ((x2, y2), (x3, y3)), ...]
        """
        vertices = list(self.vertices[:])
        if self.closed:
            vertices.append(vertices[0])

        return connected_pairs(vertices)

    def reverse(self):
        """Reverse the order of the vertices."""
        self.primary_points.reverse()

def custom_attributes(item: Shape)-> List[str]:
    """
    Return a list of custom attributes of a Shape or
    Batch instance.
    """
    if isinstance(item, Shape):
        dummy = Shape([(0, 0), (1, 0)])
    else:
        raise TypeError("Invalid item type")
    native_attribs = set(dir(dummy))
    custom_attribs = set(dir(item)) - native_attribs

    return list(custom_attribs)
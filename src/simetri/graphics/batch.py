"""Batch objects are used for grouping other Shape and Batch objects.
"""

import logging
from typing import Any, Iterator, List, Sequence

from numpy import ndarray, isclose, around, array
from typing_extensions import Self
import networkx as nx


from .all_enums import Types, batch_types, drawable_types, get_enum_value
from .common import common_properties, get_defaults, _set_Nones, Point, Line
from .core import Base
from .bbox import bounding_box
from ..canvas.style_map import batch_args
from ..helpers.validation import validate_args
from ..helpers.geometry import (
    inclination_angle,
    all_intersections,
    fix_degen_points,
    get_polygons,
    all_close_points,
    mid_point,
    distance,
    connected_pairs,
)
from ..helpers.graph import is_cycle, is_open_walk, Graph
from ..settings.settings import defaults

from .merge import merge_shapes_


class Batch(Base):
    """
    A Batch object is a collection of other objects (Batch, Shape,
    and Tag objects). It can be used to apply a transformation to
    all the objects in the Batch. It is used for creating 1D and 2D
    patterns of objects. all_vertices, all_elements, etc. means a flat
    list of the specified object gathered recursively from all the
    elements in the Batch.
    """

    def __init__(
        self,
        elements: Sequence[Any] = None,
        modifiers: Sequence["Modifier"] = None,
        subtype: Types = Types.BATCH,
        **kwargs,
    ):
        validate_args(kwargs, batch_args)
        if elements and not isinstance(elements, (list, tuple, ndarray)):
            raise TypeError("elements must be a sequence of objects!")
        self.elements = elements if elements is not None else []
        self.type = Types.BATCH
        if subtype not in batch_types:
            raise ValueError(f"Invalid subtype '{subtype}' for a Batch object!")
        self.subtype = get_enum_value(Types, subtype)
        self.modifiers = modifiers
        self.blend_mode = None
        self.alpha = None
        self.line_alpha = None
        self.fill_alpha = None
        self.text_alpha = None
        self.clip = False  # if clip is True, the batch.mask is used as a clip path
        self.mask = None
        self.even_odd_rule = False
        self.blend_group = False
        self.transparency_group = False
        common_properties(self)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_attribs(self, attrib, value):
        """Sets the attribute to the given value for all elements in the batch if it is applicable."""
        for element in self.elements:
            if element.type == Types.BATCH:
                setattr(element, attrib, value)
            elif hasattr(element, attrib):
                setattr(element, attrib, value)

    def set_batch_attr(self, attrib: str, value: Any) -> Self:
        """Sets the attribute to the given value for the batch itself.
        batch.attrib = value would set the attribute to the elements
        of the batch object but not the batch itself."""

        self.__dict__[attrib] = value

    def __str__(self):
        if self.elements is None or len(self.elements) == 0:
            res = "Batch()"
        elif len(self.elements) in [1, 2]:
            res = f"Batch({self.elements})"
        else:
            res = f"Batch({self.elements[0]}...{self.elements[-1]})"
        return res

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, subscript):
        if isinstance(subscript, slice):
            res = self.elements[subscript.start : subscript.stop : subscript.step]
        else:
            res = self.elements[subscript]
        return res

    def __setitem__(self, subscript, value):
        elements = self.elements
        if isinstance(subscript, slice):
            elements[subscript.start : subscript.stop : subscript.step] = value
        elif isinstance(subscript, int):
            elements[subscript] = value
        else:
            raise TypeError("Invalid subscript type")

    def __add__(self, other: "Batch") -> "Batch":
        if other.type == Types.BATCH:
            #!!! shoulde we use self instead of self.copy()?
            batch = self.copy()
            for element in other.elements:
                batch.append(element)
            res = batch
        else:
            raise RuntimeError(
                "Invalid object. Only Batch objects can be added together!"
            )
        return res

    def __bool__(self):
        return len(self.elements) > 0

    def __iter__(self):
        return iter(self.elements)

    def _duplicates(self, elements):
        for element in elements:
            if element.type in drawable_types:
                ids = [x.id for x in self.elements]
                if element.id in ids:
                    raise ValueError("Only unique elements are allowed!")
            else:
                msg = 'Invalid object. Only "drawable" objects are valid!'
                raise ValueError(msg)

        return len(set(elements)) != len(elements)

    def proximity(self, dist_tol: float = None, n: int = 5) -> list[Point]:
        """Returns the n closest points in the batch."""
        if dist_tol is None:
            dist_tol = defaults["dist_tol"]
        vertices = self.all_vertices
        vertices = [(*v, i) for i, v in enumerate(vertices)]
        _, pairs = all_close_points(vertices, dist_tol=dist_tol, with_dist=True)
        return [pair for pair in pairs if pair[2] > 0][:n]

    def append(self, element: Any) -> Self:
        """Appends the element to the batch."""
        if element.type in drawable_types:
            if element not in self.elements:
                self.elements.append(element)
        return self

    def reverse(self) -> Self:
        """Reverses the order of the elements in the batch."""
        self.elements = self.elements[::-1]
        return self

    def insert(self, index, element: Any) -> Self:
        """Inserts the element at the given index."""
        if element.type in drawable_types:
            if element not in self.elements:
                self.elements.insert(index, element)
            else:
                logging.info(
                    f"Batch.insert: Ignoring duplicate element(id:{element.id})"
                )
        return self

    def remove(self, element: Any) -> Self:
        """Removes the element from the batch."""
        if element in self.elements:
            self.elements.remove(element)
        return self

    def pop(self, index: int) -> Any:
        """Removes the element at the given index and returns it."""
        return self.elements.pop(index)

    def clear(self) -> Self:
        """Removes all elements from the batch."""
        self.elements = []
        return self

    def extend(self, elements: Sequence[Any]) -> Self:
        """Extends the batch with the given elements."""
        for element in elements:
            if element.type in drawable_types and element not in self.elements:
                self.elements.append(element)
            else:
                logging.info(
                    "Batch.extend: Ignoring duplicate element(id:%i)", element.id
                )
        return self

    def iter_elements(self, element_type: Types = None) -> Iterator:
        """Iterate over all elements in the batch, including the elements
        in the nested batches."""
        for elem in self.elements:
            if elem.type == Types.BATCH:
                yield from elem.iter_elements(element_type)
            else:
                if element_type is None:
                    yield elem
                elif elem.type == element_type:
                    yield elem

    @property
    def all_elements(self) -> list[Any]:
        """Return a list of all elements in the batch,
        including the elements in the nested batches."""
        elements = []
        for elem in self.elements:
            if elem.type == Types.BATCH:
                elements.extend(elem.all_elements)
            else:
                elements.append(elem)
        return elements

    @property
    def all_shapes(self) -> list["Shape"]:
        """Return a list of all shapes in the batch."""
        elements = self.all_elements
        shapes = []
        for element in elements:
            if element.type == Types.SHAPE:
                shapes.append(element)
        return shapes

    @property
    def all_vertices(self) -> list[Point]:
        """Return a list of all points in the batch in their
        transformed positions."""
        elements = self.all_elements
        vertices = []
        for element in elements:
            if element.type == Types.SHAPE:
                vertices.extend(element.vertices)
            elif element.type == Types.BATCH:
                vertices.extend(element.all_vertices)
        return vertices

    @property
    def all_segments(self) -> list[Line]:
        """Return a list of all segments in the batch."""
        elements = self.all_elements
        segments = []
        for element in elements:
            if element.type == Types.SHAPE:
                segments.extend(element.vertex_pairs)
        return segments

    def _get_graph_nodes_and_edges(self, dist_tol: float = None, n_round=None):
        _set_Nones(self, ["dist_tol", "n_round"], [dist_tol, n_round])
        vertices = self.all_vertices
        shapes = self.all_shapes
        d_ind_coords = {}
        point_id = []
        rounded_vertices = []
        for i, vert in enumerate(vertices):
            coords = tuple(around(vert, n_round))
            rounded_vertices.append(coords)
            d_ind_coords[i] = coords
            point_id.append([vert[0], vert[1], i])

        _, pairs = all_close_points(point_id, dist_tol=dist_tol, with_dist=True)

        for pair in pairs:
            id1, id2, _ = pair
            average = tuple(mid_point(vertices[id1], vertices[id2]))
            # !!! add logging her for average info
            d_ind_coords[id1] = average
            d_ind_coords[id2] = average
            rounded_vertices[id1] = average
            rounded_vertices[id2] = average

        d_coords_node_id = {}
        d_node_id__rounded_coords = {}

        s_rounded_vertices = set(rounded_vertices)
        for i, vertex in enumerate(s_rounded_vertices):
            d_coords_node_id[vertex] = i
            d_node_id__rounded_coords[i] = vertex

        edges = []
        ind = 0
        for shape in shapes:
            node_ids = []
            s_vertices = shape.vertices[:]
            for vertex in s_vertices:
                node_ids.append(d_coords_node_id[rounded_vertices[ind]])
                ind += 1
            edges.extend(connected_pairs(node_ids))
            if shape.closed:
                edges.append((node_ids[-1], node_ids[0]))

        return d_node_id__rounded_coords, edges

    def as_graph(
        self,
        directed: bool = False,
        weighted: bool = False,
        dist_tol: float = None,
        atol=None,
        n_round: int = None,
    ) -> Graph:
        """Return the batch as a Graph object.
        Graph.nx is the networkx graph.
        """
        _set_Nones(self, ["dist_tol", "atol", "n_round"], [dist_tol, atol, n_round])
        d_node_id_coords, edges = self._get_graph_nodes_and_edges(dist_tol, n_round)
        if directed:
            nx_graph = nx.DiGraph()
            graph_type = Types.DIRECTED
        else:
            nx_graph = nx.Graph()
            graph_type = Types.UNDIRECTED

        for id_, coords in d_node_id_coords.items():
            nx_graph.add_node(id_, pos=coords)

        if weighted:
            for edge in edges:
                p1 = d_node_id_coords[edge[0]]
                p2 = d_node_id_coords[edge[1]]
                nx_graph.add_edge(edge[0], edge[1], weight=distance(p1, p2))
            subtype = Types.WEIGHTED
        else:
            nx_graph.update(edges)
            subtype = Types.NONE

        graph = Graph(type=graph_type, subtype=subtype, nx_graph=nx_graph)
        return graph

    def graph_summary(self, dist_tol: float = None, n_round: int = None) -> str:
        """Returns a representation of the Batch object as a graph."""
        if dist_tol is None:
            dist_tol = defaults["dist_tol"]
        if n_round is None:
            n_round = defaults["n_round"]
        all_shapes = self.all_shapes
        all_vertices = self.all_vertices
        lines = []
        lines.append("Batch summary:")
        lines.append(f"# shapes: {len(all_shapes)}")
        lines.append(f"# vertices: {len(all_vertices)}")
        for shape in self.all_shapes:
            if shape.subtype:
                s = (
                    f"# vertices in shape(id: {shape.id}, subtype: "
                    f"{shape.subtype}): {len(shape.vertices)}"
                )
            else:
                s = f"# vertices in shape(id: {shape.id}): " f"{len(shape.vertices)}"
            lines.append(s)
        graph = self.as_graph(dist_tol=dist_tol, n_round=n_round).nx_graph

        for island in nx.connected_components(graph):
            lines.append(f"Island: {island}")
            if is_cycle(graph, island):
                lines.append(f"Cycle: {len(island)} nodes")
            elif is_open_walk(graph, island):
                lines.append(f"Open Walk: {len(island)} nodes")
            else:
                degens = [node for node in island if graph.degree(node) > 2]
                degrees = f"{[(node, graph.degree(node)) for node in degens]}"
                lines.append(f"Degenerate: {len(island)} nodes")
                lines.append(f"(Node, Degree): {degrees}")
            lines.append("-" * 40)

        return "\n".join(lines)

    def _merge_collinears(
        self,
        d_node_id_coord: dict[int, Point],
        edges: List[Line],
        angle_bin_size: float = 0.1,
        tol: float = None,
        rtol: float = None,
        atol: float = None,
    ) -> List[Line]:
        """
        delta: maximum difference in slope between two edges to be
        considered different
        1- Create a list of [(line_angle, edge), ...] for all edges
        2- Sort the list by line_angle
        3- Create bins of edges with similar line_angle
        4- Merge edges in each bin
        5- Return the merged edges
        """
        tol, rtol, atol = get_defaults(["tol", "rtol", "atol"], [tol, rtol, atol])

        def merge_multiple_edges(collinear_edges):
            """
            Merge multiple collinear edges in a list of edges.
            collinear_edges: [((x1, y1), (x2, y2)), ((x3, y3), (x4, y4)),...]
            These edges are all collinear.
            """
            x_coords = []
            y_coords = []
            points = []
            for edge in collinear_edges:
                x_coords.extend([edge[0][0], edge[1][0]])
                y_coords.extend([edge[0][1], edge[1][1]])
                points.extend(edge)

            xmin = min(x_coords)
            xmax = max(x_coords)
            ymin = min(y_coords)
            ymax = max(y_coords)
            if isclose(xmin, xmax, rtol=defaults["rtol"], atol=defaults["atol"]):
                p1 = [
                    p
                    for p in points
                    if isclose(p[1], ymin, rtol=defaults["rtol"], atol=atol)
                ][0]
                p2 = [
                    p
                    for p in points
                    if isclose(p[1], ymax, rtol=defaults["rtol"], atol=atol)
                ][0]
            else:
                p1 = [
                    p
                    for p in points
                    if isclose(p[0], xmin, rtol=defaults["rtol"], atol=atol)
                ][0]
                p2 = [
                    p
                    for p in points
                    if isclose(p[0], xmax, rtol=defaults["rtol"], atol=atol)
                ][0]

            return [p1, p2]

        def process_islands(islands, res, merged):
            for island in islands:
                collinear_edges = []
                collinear_edge_indices = []
                for i in island:
                    edge1_indices = bin_[i][i_edge]
                    edge1 = [d_node_id_coord[x] for x in edge1_indices]
                    for conn_type_x_res_ind2 in d_ind1_conn_type_x_res_ind2[i]:
                        _, _, ind2 = conn_type_x_res_ind2
                    edge2_indices = bin_[ind2][i_edge]
                    edge2 = [d_node_id_coord[x] for x in edge2_indices]
                    if set((i, ind2)) in s_processed_edge_indices:
                        continue
                    collinear_edges.extend([edge1, edge2])
                    collinear_edge_indices.append(edge1_indices)
                    collinear_edge_indices.append(edge2_indices)
                    s_processed_edge_indices.add(frozenset((i, ind2)))
                if collinear_edges:
                    for edge in collinear_edge_indices:
                        merged.append(frozenset(edge))
                    res.append(merge_multiple_edges(collinear_edges))

        if len(edges) < 2:
            return edges

        angles_edges = []
        i_angle, i_edge = 0, 1
        for edge in edges:
            edge = list(edge)
            p1 = d_node_id_coord[edge[0]]
            p2 = d_node_id_coord[edge[1]]
            angle = inclination_angle(p1, p2)
            angles_edges.append((angle, edge))

        # group angles into bins
        angles_edges.sort()

        bins = []
        bin_ = [angles_edges[0]]
        for angle, edge in angles_edges[1:]:
            angle1 = bin_[0][i_angle]
            if abs(angle - angle1) <= angle_bin_size:
                bin_.append((angle, edge))
            else:
                bins.append(bin_)
                bin_ = [(angle, edge)]
        bins.append(bin_)
        merged = []
        res = []
        # x_res can be a point or segment(overlapping edges)
        i_ind2 = 2  # indices for intersection results
        for bin_ in bins:
            segments = [[d_node_id_coord[node] for node in x[i_edge]] for x in bin_]

            d_ind1_conn_type_x_res_ind2 = all_intersections(
                segments, rtol=rtol, use_intersection3=True
            )
            connections = {}
            for i in range(len(bin_)):
                connections[i] = set()
            for k, values in d_ind1_conn_type_x_res_ind2.items():
                for v in values:
                    connections[k].add(v[i_ind2])
            # create a graph of connections
            g_connections = nx.Graph()
            for k, v in connections.items():
                for x in v:
                    g_connections.add_edge(k, x)
            islands = list(nx.connected_components(g_connections))
            s_processed_edge_indices = set()
            process_islands(islands, res, merged)
        for edge in edges:
            if set(edge) not in merged:
                res.append([d_node_id_coord[x] for x in edge])

        return res

    def merge_shapes(
        self, tol: float = None, rtol: float = None, atol: float = None
    ) -> Self:
        return merge_shapes_(self, tol=tol, rtol=rtol, atol=atol)

    def all_polygons(self, dist_tol: float = None) -> list:
        """Return a list of all polygons in the batch in their
        transformed positions."""
        if dist_tol is None:
            dist_tol = defaults["dist_tol"]
        exclude = []
        include = []
        for shape in self.all_shapes:
            if len(shape.primary_points) > 2 and shape.closed:
                vertices = shape.vertices
                exclude.append(vertices)
            else:
                include.append(shape)
        polylines = []
        for element in include:
            points = element.vertices
            points = fix_degen_points(points, dist_tol=dist_tol, closed=element.closed)
            polylines.append(points)
        fixed_polylines = []
        if polylines:
            for polyline in polylines:
                fixed_polylines.append(
                    fix_degen_points(polyline, dist_tol=dist_tol, closed=True)
                )
            polygons = get_polygons(fixed_polylines, dist_tol=dist_tol)
            res = polygons + exclude
        else:
            res = exclude
        return res

    def copy(self) -> "Batch":
        """Returns a copy of the batch."""
        b = Batch(modifiers=self.modifiers)
        if self.elements:
            b.elements = [elem.copy() for elem in self.elements]
        else:
            b.elements = []
        custom_attribs = custom_batch_attributes(self)
        for attrib in custom_attribs:
            setattr(b, attrib, getattr(self, attrib))
        return b

    @property
    def b_box(self):
        """Returns the bounding box of the batch."""
        xy_list = []
        for elem in self.elements:
            xy_list.extend(
                elem.b_box.corners
            )  # To do: we should eliminate this. Just add all points.
        # To do: memoize the bounding box
        return bounding_box(array(xy_list))

    def _modify(self, modifier):
        modifier.apply()

    def _update(self, xform_matrix, reps: int = 0):
        """Updates the batch with the given transformation matrix.
        If reps is 0, the transformation is applied to all elements.
        If reps is greater than 0, the transformation creates
        new elements with the transformed xform_matrix."""
        if reps == 0:
            for element in self.elements:
                element._update(xform_matrix, reps=0)
                if self.modifiers:
                    for modifier in self.modifiers:
                        modifier.apply(element)
        else:
            elements = self.elements[:]
            new = []
            for _ in range(reps):
                for element in elements:
                    new_element = element.copy()
                    new_element._update(xform_matrix)
                    self.elements.append(new_element)
                    new.append(new_element)
                    if self.modifiers:
                        for modifier in self.modifiers:
                            modifier.apply(new_element)
                elements = new[:]
                new = []
        return self


def custom_batch_attributes(item: Batch) -> List[str]:
    """
    Return a list of custom attributes of a Shape or
    Batch instance.
    """
    from .shape import Shape

    if isinstance(item, Batch):
        dummy_shape = Shape([(0, 0), (1, 0)])
        dummy = Batch([dummy_shape])
    else:
        raise TypeError("Invalid item type")
    native_attribs = set(dir(dummy))
    custom_attribs = set(dir(item)) - native_attribs

    return list(custom_attribs)

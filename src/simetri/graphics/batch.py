"""Batch objects are used for grouping other Shape and Batch objects.
"""

from typing import Any, Iterator, List, Sequence

from numpy import around, array
from typing_extensions import Self, Dict
import networkx as nx


from .all_enums import Types, batch_types, get_enum_value
from .common import common_properties, _set_Nones, Point, Line
from .core import Base
from .bbox import bounding_box
from ..canvas.style_map import batch_args
from ..helpers.validation import validate_args
from ..geometry.geometry import(
    fix_degen_points,
    get_polygons,
    all_close_points,
    mid_point,
    distance,
    connected_pairs,
    round_segment,
    round_point
)
from ..helpers.graph import is_cycle, is_open_walk, Graph
from ..settings.settings import defaults

from .merge import _merge_shapes, _merge_collinears


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
        """
        Initialize a Batch object.

        Args:
            elements (Sequence[Any], optional): The elements to include in the batch.
            modifiers (Sequence[Modifier], optional): The modifiers to apply to the batch.
            subtype (Types, optional): The subtype of the batch.
            kwargs (dict): Additional keyword arguments.
        """
        validate_args(kwargs, batch_args)
        if elements and not isinstance(elements, (list, tuple)):
            self.elements = [elements]
        else:
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
        """
        Sets the attribute to the given value for all elements in the batch if it is applicable.

        Args:
            attrib (str): The attribute to set.
            value (Any): The value to set the attribute to.
        """
        for element in self.elements:
            if element.type == Types.BATCH:
                setattr(element, attrib, value)
            elif hasattr(element, attrib):
                setattr(element, attrib, value)

    def set_batch_attr(self, attrib: str, value: Any) -> Self:
        """
        Sets the attribute to the given value for the batch itself.
        batch.attrib = value would set the attribute to the elements
        of the batch object but not the batch itself.

        Args:
            attrib (str): The attribute to set.
            value (Any): The value to set the attribute to.

        Returns:
            Self: The batch object.
        """
        self.__dict__[attrib] = value

    def __str__(self):
        """
        Return a string representation of the batch.

        Returns:
            str: The string representation of the batch.
        """
        if self.elements is None or len(self.elements) == 0:
            res = "Batch()"
        elif len(self.elements) in [1, 2]:
            res = f"Batch({self.elements})"
        else:
            res = f"Batch({self.elements[0]}...{self.elements[-1]})"
        return res

    def __repr__(self):
        """
        Return a string representation of the batch.

        Returns:
            str: The string representation of the batch.
        """
        return self.__str__()

    def __len__(self):
        """
        Return the number of elements in the batch.

        Returns:
            int: The number of elements in the batch.
        """
        return len(self.elements)

    def __getitem__(self, subscript):
        """
        Get the element(s) at the given subscript.

        Args:
            subscript (int or slice): The subscript to get the element(s) from.

        Returns:
            Any: The element(s) at the given subscript.
        """
        if isinstance(subscript, slice):
            res = self.elements[subscript.start : subscript.stop : subscript.step]
        else:
            res = self.elements[subscript]
        return res

    def __setitem__(self, subscript, value):
        """
        Set the element(s) at the given subscript.

        Args:
            subscript (int or slice): The subscript to set the element(s) at.
            value (Any): The value to set the element(s) to.
        """
        elements = self.elements
        if isinstance(subscript, slice):
            elements[subscript.start : subscript.stop : subscript.step] = value
        elif isinstance(subscript, int):
            elements[subscript] = value
        else:
            raise TypeError("Invalid subscript type")

    def __add__(self, other: "Batch") -> "Batch":
        """
        Add another batch to this batch.

        Args:
            other (Batch): The other batch to add.

        Returns:
            Batch: The combined batch.

        Raises:
            RuntimeError: If the other object is not a batch.
        """
        if other.type == Types.BATCH:
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
        """
        Return whether the batch has any elements.

        Returns:
            bool: True if the batch has elements, False otherwise.
        """
        return len(self.elements) > 0

    def __iter__(self):
        """
        Return an iterator over the elements in the batch.

        Returns:
            Iterator: An iterator over the elements in the batch.
        """
        return iter(self.elements)

    def _duplicates(self, elements):
        """
        Check for duplicate elements in the batch.

        Args:
            elements (Sequence[Any]): The elements to check for duplicates.

        Raises:
            ValueError: If duplicate elements are found.

        Returns:
            bool: True if duplicates are found, False otherwise.
        """
        for element in elements:
            ids = [x.id for x in self.elements]
            if element.id in ids:
                raise ValueError("Only unique elements are allowed!")

        return len(set(elements)) != len(elements)

    def proximity(self, dist_tol: float = None, n: int = 5) -> list[Point]:
        """
        Returns the n closest points in the batch.

        Args:
            dist_tol (float, optional): The distance tolerance for proximity.
            n (int, optional): The number of closest points to return.

        Returns:
            list[Point]: The n closest points in the batch.
        """
        if dist_tol is None:
            dist_tol = defaults["dist_tol"]
        vertices = self.all_vertices
        vertices = [(*v, i) for i, v in enumerate(vertices)]
        _, pairs = all_close_points(vertices, dist_tol=dist_tol, with_dist=True)
        return [pair for pair in pairs if pair[2] > 0][:n]

    def append(self, element: Any) -> Self:
        """
        Appends the element to the batch.

        Args:
            element (Any): The element to append.

        Returns:
            Self: The batch object.
        """
        if element not in self.elements:
            self.elements.append(element)
        return self

    def reverse(self) -> Self:
        """
        Reverses the order of the elements in the batch.

        Returns:
            Self: The batch object.
        """
        self.elements = self.elements[::-1]
        return self

    def insert(self, index, element: Any) -> Self:
        """
        Inserts the element at the given index.

        Args:
            index (int): The index to insert the element at.
            element (Any): The element to insert.

        Returns:
            Self: The batch object.
        """
        if element not in self.elements:
            self.elements.insert(index, element)

        return self

    def remove(self, element: Any) -> Self:
        """
        Removes the element from the batch.

        Args:
            element (Any): The element to remove.

        Returns:
            Self: The batch object.
        """
        if element in self.elements:
            self.elements.remove(element)
        return self

    def pop(self, index: int) -> Any:
        """
        Removes the element at the given index and returns it.

        Args:
            index (int): The index to remove the element from.

        Returns:
            Any: The removed element.
        """
        return self.elements.pop(index)

    def clear(self) -> Self:
        """
        Removes all elements from the batch.

        Returns:
            Self: The batch object.
        """
        self.elements = []
        return self

    def extend(self, elements: Sequence[Any]) -> Self:
        """
        Extends the batch with the given elements.

        Args:
            elements (Sequence[Any]): The elements to extend the batch with.

        Returns:
            Self: The batch object.
        """
        for element in elements:
            if element not in self.elements:
                self.elements.append(element)

        return self

    def iter_elements(self, element_type: Types = None) -> Iterator:
        """Iterate over all elements in the batch, including the elements
        in the nested batches.

        Args:
            element_type (Types, optional): The type of elements to iterate over. Defaults to None.

        Returns:
            Iterator: An iterator over the elements in the batch.
        """
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
        including the elements in the nested batches.

        Returns:
            list[Any]: A list of all elements in the batch.
        """
        elements = []
        for elem in self.elements:
            if elem.type == Types.BATCH:
                elements.extend(elem.all_elements)
            else:
                elements.append(elem)
        return elements

    @property
    def all_shapes(self) -> list["Shape"]:
        """Return a list of all shapes in the batch.

        Returns:
            list[Shape]: A list of all shapes in the batch.
        """
        elements = self.all_elements
        shapes = []
        for element in elements:
            if element.type == Types.SHAPE:
                shapes.append(element)
        return shapes

    @property
    def all_vertices(self) -> list[Point]:
        """Return a list of all points in the batch in their
        transformed positions.

        Returns:
            list[Point]: A list of all points in the batch in their transformed positions.
        """
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
        """Return a list of all segments in the batch.

        Returns:
            list[Line]: A list of all segments in the batch.
        """
        elements = self.all_elements
        segments = []
        for element in elements:
            if element.type == Types.SHAPE:
                segments.extend(element.vertex_pairs)
        return segments


    def _get_graph_nodes_and_edges(self, dist_tol: float = None, n_round=None):
        """Get the graph nodes and edges for the batch.

        Args:
            dist_tol (float, optional): The distance tolerance for proximity. Defaults to None.
            n_round (int, optional): The number of decimal places to round to. Defaults to None.

        Returns:
            tuple: A tuple containing the node coordinates and edges.
        """
        if n_round is None:
            n_round = defaults["n_round"]
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

        Args:
            directed (bool, optional): Whether the graph is directed. Defaults to False.
            weighted (bool, optional): Whether the graph is weighted. Defaults to False.
            dist_tol (float, optional): The distance tolerance for proximity. Defaults to None.
            atol (optional): The absolute tolerance. Defaults to None.
            n_round (int, optional): The number of decimal places to round to. Defaults to None.

        Returns:
            Graph: The batch as a Graph object.
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
        """Returns a representation of the Batch object as a graph.

        Args:
            dist_tol (float, optional): The distance tolerance for proximity. Defaults to None.
            n_round (int, optional): The number of decimal places to round to. Defaults to None.

        Returns:
            str: A representation of the Batch object as a graph.
        """
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

    def _merge_collinears(self, edges, n_round=2):
        """Merge collinear edges in the batch.

        Args:
            d_node_id_coords (dict): The node coordinates.
            edges (list): The edges to merge.
            tol (float, optional): The tolerance for merging. Defaults to None.
            rtol (float, optional): The relative tolerance. Defaults to None.
            atol (float, optional): The absolute tolerance. Defaults to None.

        Returns:
            list: The merged edges.
        """
        return _merge_collinears(self, edges, n_round=n_round)

    def merge_shapes(
        self, tol: float = None, rtol: float = None, atol: float = None
    ) -> Self:
        """Merges the shapes in the batch if they are connected.
        Returns a new batch with the merged shapes as well as the shapes
        as well as the shapes that could not be merged.

        Args:
            tol (float, optional): The tolerance for merging shapes. Defaults to None.
            rtol (float, optional): The relative tolerance. Defaults to None.
            atol (float, optional): The absolute tolerance. Defaults to None.

        Returns:
            Self: The batch object with merged shapes.
        """
        return _merge_shapes(self, tol=tol, rtol=rtol, atol=atol)

    def _get_edges_and_segments(self, dist_tol: float = None, n_round: int = None):
        """Get the edges and segments for the batch.

        Args:
            dist_tol (float, optional): The distance tolerance for proximity. Defaults to None.
            n_round (int, optional): The number of decimal places to round to. Defaults to None.

        Returns:
            tuple: A tuple containing the edges and segments.
        """
        if dist_tol is None:
            dist_tol = defaults["dist_tol"]
        if n_round is None:
            n_round = defaults["n_round"]
        d_coord_node = self.d_coord_node
        segments = self.all_segments
        segments = [round_segment(segment, n_round) for segment in segments]
        edges = []
        for seg in segments:
            p1, p2 = seg
            id1 = d_coord_node[p1]
            id2 = d_coord_node[p2]
            edges.append((id1, id2))

        return edges, segments

    def _set_node_dictionaries(self, coords: List[Point], n_round: int=2) -> List[Dict]:
        '''Set dictionaries for nodes and coordinates.
        d_node_coord: Dictionary of node id to coordinates.
        d_coord_node: Dictionary of coordinates to node id.

        Args:
            nodes (List[Point]): List of vertices.
            n_round (int, optional): Number of rounding digits. Defaults to 2.
        '''

        coords = [tuple(round_point(coord, n_round)) for coord in coords]
        coords = list(set(coords))   # remove duplicates
        coords.sort()    # sort by x coordinates
        coords.sort(key=lambda x: x[1])  # sort by y coordinates

        d_node_coord = {}
        d_coord_node = {}

        for i, coord in enumerate(coords):
            d_node_coord[i] = coord
            d_coord_node[coord] = i

        self.d_node_coord = d_node_coord
        self.d_coord_node = d_coord_node

    def all_polygons(self, dist_tol: float = None) -> list:
        """Return a list of all polygons in the batch in their
        transformed positions.

        Args:
            dist_tol (float, optional): The distance tolerance for proximity. Defaults to None.

        Returns:
            list: A list of all polygons in the batch.
        """
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
        """Returns a copy of the batch.

        Returns:
            Batch: A copy of the batch.
        """
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
        """Returns the bounding box of the batch.

        Returns:
            BoundingBox: The bounding box of the batch.
        """
        xy_list = []
        for elem in self.elements:
            xy_list.extend(
                elem.b_box.corners
            )  # To do: we should eliminate this. Just add all points.
        # To do: memoize the bounding box
        return bounding_box(array(xy_list))

    def _modify(self, modifier):
        """Apply a modifier to the batch.

        Args:
            modifier (Modifier): The modifier to apply.
        """
        modifier.apply()

    def _update(self, xform_matrix, reps: int = 0):
        """Updates the batch with the given transformation matrix.
        If reps is 0, the transformation is applied to all elements.
        If reps is greater than 0, the transformation creates
        new elements with the transformed xform_matrix.

        Args:
            xform_matrix (ndarray): The transformation matrix.
            reps (int, optional): The number of repetitions. Defaults to 0.
        """
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

    Args:
        item (Batch): The batch object.

    Returns:
        List[str]: A list of custom attributes.
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

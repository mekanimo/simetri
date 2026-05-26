"""Group objects are used for grouping other Shape and Group objects."""

from typing import Any, Iterator, List, Sequence, Callable

from numpy import array
from typing_extensions import Self, Dict
import json


from .all_enums import (
    InPlace,
    Types,
    TransformationType,
    get_enum_value,
)
from .common import PointType, LineType, get_unique_id
from .core import Base, _update_inplace
from .bbox import bounding_box
from ..geometry.geometry import (
    fix_degen_points,
    get_polygons,
    all_close_points,
    round_segment,
    round_point,
)
from ..settings.settings import defaults, issue_warning

from .merge import _merge_shapes, _merge_collinears


class Group(Base):
    """
    A Group object is a collection of other objects (Group, Shape,
    and Tag objects). It can be used to apply a transformation to
    all the objects in the Group. It is used for creating 1D and 2D
    patterns of objects. all_vertices, all_elements, etc. means a flat
    list of the specified object gathered recursively from all the
    elements in the Group.
    """

    # __slots__ = [
    #     "elements",
    #     "type",
    #     "subtype",
    #     "modifiers",
    #     "visible",
    #     "d_node_coord",
    #     "d_coord_node",
    #     "d_rounded_coord",
    # ]

    def __init__(
        self,
        elements: Sequence[Any] = None,
        modifiers: Sequence["Modifier"] = None,
        subtype: Types = Types.GROUP,
    ):
        """
        Initialize a Group object.

        Args:
            elements (Sequence[Any], optional): The elements to include in the group.
            modifiers (Sequence[Modifier], optional): The modifiers to apply to the group.
            subtype (Types, optional): The subtype of the group.
            kwargs (dict): Additional keyword arguments.
        """

        def flatten_elements(nested_list):
            """Flatten a nested list.

            Args:
                nested_list: The nested list to flatten.

            Yields:
                The flattened elements.
            """
            for i in nested_list:
                if isinstance(i, (list, tuple)):
                    yield from flatten_elements(i)
                else:
                    yield i

        # validate_args(kwargs, group_args)
        # We need to handle this differently now!!!

        if elements is None:
            self.elements = []
        elif not isinstance(elements, (list, tuple)):
            self.elements = [elements]
        else:
            _elements = []
            for element in elements:
                if isinstance(element, (list, tuple)):
                    for elem in flatten_elements(element):
                        if elem:
                            _elements.append(elem)
                else:
                    if element:
                        _elements.append(element)
            self.elements = _elements[:]
            # self.elements = elements if elements is not None else []

        self.type = Types.GROUP
        self.subtype = get_enum_value(Types, subtype)
        if modifiers is None:
            modifiers = []
        self.modifiers = modifiers
        self.visible = True
        self.id = get_unique_id(self)

    def set_attribs(
        self, attrib: str, value: Any, key: Callable = None
    ) -> Self:
        """
        Sets the attribute to the given value for all elements in the group if it is applicable.

        Args:
            attrib (str): The attribute to set.
            value (Any): The value to set the attribute to.
            key (Callable, optional): A function to filter elements by a specific key. Defaults to None.

        Returns:
            Self: The group object.

        Example: group.set_attribs('fill_color', 'red', key=lambda x: x.type == Types.SHAPE)
        """
        for element in self.elements:
            if key is not None:
                if key(element):
                    if element.type == Types.GROUP:
                        element.set_attribs(attrib, value, key=key)
                    elif hasattr(element, attrib):
                        setattr(element, attrib, value)
            else:
                if element.type == Types.GROUP:
                    element.set_attribs(attrib, value)
                elif hasattr(element, attrib):
                    setattr(element, attrib, value)

        return self

    def __str__(self):
        """
        Return a string representation of the group.

        Returns:
            str: The string representation of the group.
        """
        if self.elements is None or len(self.elements) == 0:
            res = "Group()"
        elif len(self.elements) in [1, 2]:
            res = f"Group({self.elements})"
        else:
            res = f"Group({self.elements[0]}...{self.elements[-1]})"
        return res

    def __repr__(self):
        """
        Return a string representation of the group.

        Returns:
            str: The string representation of the group.
        """
        return self.__str__()

    def __len__(self):
        """
        Return the number of elements in the group.

        Returns:
            int: The number of elements in the group.
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
            res = self.elements[
                subscript.start : subscript.stop : subscript.step
            ]
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

    def __add__(self, other: "Group") -> "Group":
        """
        Add another group to this group.

        Args:
            other (Group): The other group to add.

        Returns:
            Group: The combined group.

        Raises:
            RuntimeError: If the other object is not a group.
        """
        if other.type == Types.GROUP:
            group = self.copy()
            for element in other.elements:
                group.append(element)
            res = group
        else:
            raise RuntimeError(
                "Invalid object. Only Group objects can be added together!"
            )
        return res

    def __bool__(self):
        """
        Return whether the group has any elements.

        Returns:
            bool: True if the group has elements, False otherwise.
        """
        return len(self.elements) > 0

    def __iter__(self):
        """
        Return an iterator over the elements in the group.

        Returns:
            Iterator: An iterator over the elements in the group.
        """
        return iter(self.elements)

    def _duplicates(self, elements):
        """
        Check for duplicate elements in the group.

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

    def to_json(self) -> str:
        """Serialize the Group into a JSON string.

        The payload includes:
          - type, subtype
          - elements (recursively serialized)
          - modifiers (stringified)
          - attributes (common group attributes incl. mask if present)
        """

        def _to_jsonable(obj):
            # Enum-like with 'value'
            if hasattr(obj, "value"):
                return obj.value
            if isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            if isinstance(obj, dict):
                return {k: _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(x) for x in obj]
            try:
                return float(obj)
            except Exception:
                try:
                    return str(obj)
                except Exception:
                    return None

        # Elements
        elems = []
        for elem in self.elements or []:
            if hasattr(elem, "to_json") and callable(elem.to_json):
                try:
                    elems.append(json.loads(elem.to_json()))
                    continue
                except Exception:
                    pass
            # Fallback summary for unknown elements
            summary = {
                "type": _to_jsonable(
                    getattr(
                        getattr(elem, "type", None),
                        "value",
                        getattr(elem, "type", "UNKNOWN"),
                    )
                ),
                "repr": str(elem),
            }
            elems.append(summary)

        # Modifiers (stringify conservatively)
        mods = None
        if self.modifiers:
            mods = []
            for m in self.modifiers:
                name = getattr(m, "name", None)
                mods.append(name if isinstance(name, str) else str(m))

        data = {
            "type": _to_jsonable(getattr(self.type, "value", self.type)),
            "subtype": _to_jsonable(self.subtype),
            "elements": elems,
            "modifiers": mods,
            "attributes": {},
        }

        return json.dumps(data, ensure_ascii=False)

    def proximity(self, dist_tol: float = None, n: int = 5) -> list[PointType]:
        """
        Returns the n closest points in the group.

        Args:
            dist_tol (float, optional): The distance tolerance for proximity.
            n (int, optional): The number of closest points to return.

        Returns:
            list[PointType]: The n closest points in the group.
        """
        if dist_tol is None:
            dist_tol = defaults["dist_tol"]
        vertices = self.all_vertices
        vertices = [(*v, i) for i, v in enumerate(vertices)]
        _, pairs = all_close_points(vertices, dist_tol=dist_tol, with_dist=True)
        return [pair for pair in pairs if pair[2] > 0][:n]

    def append(self, element: Any) -> Self:
        """
        Appends the element to the group.

        Args:
            element (Any): The element to append.

        Returns:
            Self: The group object.
        """
        if element in self.elements:
            issue_warning(
                f"Duplicate element added to Group: {element}",
                stacklevel=2,
            )
        self.elements.append(element)
        return self

    def reverse(self) -> Self:
        """
        Reverses the order of the elements in the group.

        Returns:
            Self: The group object.
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
            Self: The group object.
        """
        if element not in self.elements:
            self.elements.insert(index, element)

        return self

    def remove(self, element: Any) -> Self:
        """
        Removes the element from the group.

        Args:
            element (Any): The element to remove.

        Returns:
            Self: The group object.
        """
        if element in self.elements:
            self.elements.remove(element)
        return self

    def pop(self, index: int = -1) -> Any:
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
        Removes all elements from the group.

        Returns:
            Self: The group object.
        """
        self.elements = []
        return self

    def extend(self, elements: Sequence[Any]) -> Self:
        """
        Extends the group with the given elements.

        Args:
            elements (Sequence[Any]): The elements to extend the group with.

        Returns:
            Self: The group object.
        """
        for element in elements:
            if element not in self.elements:
                self.elements.append(element)

        return self

    def iter_elements(self, element_type: Types = None) -> Iterator:
        """Iterate over all elements in the group, including the elements
        in the nested groups.

        Args:
            element_type (Types, optional): The type of elements to iterate over. Defaults to None.

        Returns:
            Iterator: An iterator over the elements in the group.
        """
        for elem in self.elements:
            if elem.type == Types.GROUP:
                yield from elem.iter_elements(element_type)
            else:
                if element_type is None:
                    yield elem
                elif elem.type == element_type:
                    yield elem

    @property
    def all_elements(self) -> list[Any]:
        """Return a list of all elements in the group,
        including the elements in the nested groups.

        Returns:
            list[Any]: A list of all elements in the group.
        """
        elements = []
        for elem in self.elements:
            if elem.type == Types.GROUP:
                elements.extend(elem.all_elements)
            else:
                elements.append(elem)
        return elements

    @property
    def all_shapes(self) -> list["Shape"]:
        """Return a list of all shapes in the group.

        Returns:
            list[Shape]: A list of all shapes in the group.
        """
        elements = self.all_elements
        shapes = []
        for element in elements:
            if element.type == Types.SHAPE:
                shapes.append(element)
        return shapes

    @property
    def all_vertices(self) -> list[PointType]:
        """Return a list of all points in the group in their
        transformed positions.

        Returns:
            list[PointType]: A list of all points in the group in their transformed positions.
        """
        elements = self.all_elements
        vertices = []
        for element in elements:
            if element.type == Types.SHAPE:
                vertices.extend(element.vertices)
            elif element.type == Types.GROUP:
                vertices.extend(element.all_vertices)
        return vertices

    @property
    def all_segments(self) -> list[LineType]:
        """Return a list of all segments in the group.

        Returns:
            list[LineType]: A list of all segments in the group.
        """
        elements = self.all_elements
        segments = []
        for element in elements:
            if element.type == Types.SHAPE:
                segments.extend(element.vertex_pairs)
        return segments

    def merge_collinears(self, edges, rel_tol=None, abs_tol=None):
        """Merge collinear edges in the group.

        Args:
            d_node_id_coords (dict): The node coordinates.
            edges (list): The edges to merge.
            rel_tol (float, optional): The relative tolerance. Defaults to None.
            abs_tol (float, optional): The absolute tolerance. Defaults to None.

        Returns:
            list: The merged edges.
        """
        return _merge_collinears(self, edges)

    def merge_shapes(self, dist_tol: float = None, n_round: int = None) -> Self:
        """Merges the shapes in the group if they are connected.
        Returns a new group with the merged shapes as well as the shapes
        as well as the shapes that could not be merged.

        Args:
            tol (float, optional): The tolerance for merging shapes. Defaults to None.
            rel_tol (float, optional): The relative tolerance. Defaults to None.
            abs_tol (float, optional): The absolute tolerance. Defaults to None.

        Returns:
            Self: The group object with merged shapes.
        """
        return _merge_shapes(self, dist_tol=dist_tol, n_round=n_round)

    def _get_edges_and_segments(
        self, dist_tol: float = None, n_round: int = None
    ):
        """Get the edges and segments for the group.

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

    def _set_node_dictionaries(
        self, coords: List[PointType], n_round: int = 2
    ) -> List[Dict]:
        """Set dictionaries for nodes and coordinates.
        d_node_coord: Dictionary of node id to coordinates.
        d_coord_node: Dictionary of coordinates to node id.

        Args:
            nodes (List[PointType]): List of vertices.
            n_round (int, optional): Number of rounding digits. Defaults to 2.
        """

        d_rounded_coord = {}
        rounded = []
        for coord in coords:
            val = tuple(round_point(coord, n_round))
            rounded.append(val)
            d_rounded_coord[val] = coord

        coords = list(set(rounded))  # remove duplicates
        coords.sort()  # sort by x coordinates
        coords.sort(key=lambda x: x[1])  # sort by y coordinates

        d_node_coord = {}
        d_coord_node = {}

        for i, coord in enumerate(coords):
            d_node_coord[i] = coord
            d_coord_node[coord] = i

        self.d_node_coord = d_node_coord
        self.d_coord_node = d_coord_node
        self.d_rounded_coord = d_rounded_coord

    def all_polygons(self, dist_tol: float = None) -> list:
        """Return a list of all polygons in the group in their
        transformed positions.

        Args:
            dist_tol (float, optional): The distance tolerance for proximity. Defaults to None.

        Returns:
            list: A list of all polygons in the group.
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
            points = fix_degen_points(
                points, dist_tol=dist_tol, closed=element.closed
            )
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

    def copy(self) -> "Group":
        """Returns a copy of the group.

        Returns:
            Group: A copy of the group.
        """

        # return deepcopy(self)
        b = Group(modifiers=self.modifiers)
        if self.elements:
            b.elements = [elem.copy() for elem in self.elements]
        else:
            b.elements = []
        # custom_attribs = custom_group_attributes(self)
        # for attrib in custom_attribs:
        #     setattr(b, attrib, getattr(self, attrib))
        return b

    @property
    def b_box(self):
        """Returns the bounding box of the group.

        Returns:
            BoundingBox: The bounding box of the group.
        """
        # To do: memoize the bounding box
        return bounding_box(array(self.all_vertices))

    def _modify(self, modifier):
        """Apply a modifier to the group.

        Args:
            modifier (Modifier): The modifier to apply.
        """
        modifier.apply()

    def _update(
        self,
        xform_matrix: "ndarray",
        reps: int = 0,
        take: slice = None,
        incr: float
        | tuple[float, float]
        | tuple[callable, Any]
        | tuple[InPlace, Any] = None,
        merge: bool = False,
        xform_type: TransformationType = None,
    ) -> Self:
        """Updates the group with the given transformation matrix.
        If reps is 0, the transformation is applied to all elements.
        If reps is greater than 0, the transformation creates
        new elements with the transformed xform_matrix.

        Args:
            xform_matrix (ndarray): The transformation matrix.
            reps (int, optional): The number of repetitions. Defaults to 0.
            merge(bool, optional): If True, shapes are merged.
        """
        if take is None:
            elements = self.elements[:]
        else:
            elements = self.elements[take]
        if reps == 0:
            for element in elements:
                element._update(xform_matrix, reps=0)
                if self.modifiers:
                    for modifier in self.modifiers:
                        modifier.apply(element)
        else:
            new = []
            for i in range(reps):
                if incr is not None and i > 0:
                    xform_matrix = _update_inplace(
                        xform_matrix, xform_type, incr
                    )
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
        if merge and reps > 0:
            merged = self.merge_shapes()
            self[:] = merged.elements[:]

        return self

    def union(self, other: "Group") -> Self:
        """Returns the union of two groups.

        Args:
            other (Group): The other group to union with.

        Returns:
            Group: The union of the two groups.
        """
        if not isinstance(other, Group):
            raise TypeError(
                "Invalid object. Only Group objects can be unioned!"
            )

        self_ids = {item.id for item in self.elements}
        other_ids = {item.id for item in other.elements}

        union_ids = self_ids.union(other_ids)

        return Group(
            elements=[item for item in self.elements if item.id in union_ids],
            modifiers=self.modifiers,
            subtype=self.subtype,
        )

    def intersection(self, other: "Group") -> Self:
        """Returns the intersection of two groups.

        Args:
            other (Group): The other group to intersect with.

        Returns:
            Group: The intersection of the two groups.
        """
        if not isinstance(other, Group):
            raise TypeError(
                "Invalid object. Only Group objects can be intersected!"
            )

        self_ids = {item.id for item in self.elements}
        other_ids = {item.id for item in other.elements}

        intersection_ids = self_ids.intersection(other_ids)

        return Group(
            elements=[
                item for item in self.elements if item.id in intersection_ids
            ],
            modifiers=self.modifiers,
            subtype=self.subtype,
        )

    def difference(self, other: "Group") -> Self:
        """Returns the difference of two groups.

        Args:
            other (Group): The other group to subtract.

        Returns:
            Group: The difference of the two groups.
        """
        if not isinstance(other, Group):
            raise TypeError(
                "Invalid object. Only Group objects can be subtracted!"
            )

        self_ids = {item.id for item in self.elements}
        other_ids = {item.id for item in other.elements}

        difference_ids = self_ids.difference(other_ids)

        return Group(
            elements=[
                item for item in self.elements if item.id in difference_ids
            ],
            modifiers=self.modifiers,
            subtype=self.subtype,
        )

    def symmetric_difference(self, other: "Group") -> Self:
        """Returns the symmetric difference of two groups.

        Args:
            other (Group): The other group to find the symmetric difference with.

        Returns:
            Group: The symmetric difference of the two groups.
        """
        if not isinstance(other, Group):
            raise TypeError(
                "Invalid object. Only Group objects can be symmetrically differenced!"
            )

        self_ids = {item.id for item in self.elements}
        other_ids = {item.id for item in other.elements}

        symmetric_difference_ids = self_ids.symmetric_difference(other_ids)

        return Group(
            elements=[
                item
                for item in self.elements
                if item.id in symmetric_difference_ids
            ],
            modifiers=self.modifiers,
            subtype=self.subtype,
        )

    def subset(self, other: "Group") -> bool:
        """Checks if the current group is a subset of another group.

        Args:
            other (Group): The other group to check against.

        Returns:
            bool: True if the current group is a subset of the other group, False otherwise.
        """
        if not isinstance(other, Group):
            raise TypeError(
                "Invalid object. Only Group objects can be checked for subset!"
            )

        self_ids = {item.id for item in self.elements}
        other_ids = {item.id for item in other.elements}

        return self_ids.issubset(other_ids)

    def superset(self, other: "Group") -> bool:
        """Checks if the current group is a superset of another group.

        Args:
            other (Group): The other group to check against.

        Returns:
            bool: True if the current group is a superset of the other group, False otherwise.
        """
        if not isinstance(other, Group):
            raise TypeError(
                "Invalid object. Only Group objects can be checked for superset!"
            )

        self_ids = {item.id for item in self.elements}
        other_ids = {item.id for item in other.elements}

        return self_ids.issuperset(other_ids)

    def __hash__(self) -> int:
        """Return the hash of the group.

        Returns:
            int: The hash of the group.
        """
        return hash(tuple(self.ids))

    def __eq__(self, other: object) -> bool:
        """Check if two groups are equal.

        Args:
            other (object): The other group to compare.

        Returns:
            bool: True if the groups are equal, False otherwise.
        """
        if not isinstance(other, Group):
            return False

        if len(self.elements) != len(other.elements):
            return False

        return (
            self.elements == other.elements
            and self.modifiers == other.modifiers
        )


@property
def ids(self):
    """Return a list of ids of the elements in the group. If the element has an id attribute, it is used.
    Otherwise, id(element) is used.

    Returns:
        list: A list of ids of the elements in the group.
    """
    return [
        item.id if hasattr(item, "id") else id(item) for item in self.elements
    ]


@property
def all_ids(self):
    """Return a list of ids of the elements in the group. If the element has an id attribute, it is used.
    Otherwise, id(element) is used.

    Returns:
        list: A list of ids of the elements in the group.
    """
    ids = []
    for item in self.elements:
        if hasattr(item, "type") and item.type == Types.GROUP:
            ids.extend(item.all_ids)
        else:
            ids.append(item.id if hasattr(item, "id") else id(item))

    return ids


def custom_group_attributes(item: Group) -> List[str]:
    """
    Return a list of custom attributes of a Shape or
    Group instance.

    Args:
        item (Group): The group object.

    Returns:
        List[str]: A list of custom attributes.
    """
    from .shape import Shape

    if isinstance(item, Group):
        dummy_shape = Shape([(0, 0), (1, 0)])
        dummy = Group([dummy_shape])
    else:
        raise TypeError("Invalid item type")
    native_attribs = set(dir(dummy))
    custom_attribs = set(dir(item)) - native_attribs

    return list(custom_attribs)

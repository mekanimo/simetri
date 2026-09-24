"""Group containers for Shape, Group, and Tag objects.

A Group applies transforms to its members and supports set-like geometry
operations (union, intersection, …) and edge merging.

Examples:
    >>> from simetri.config.settings import set_defaults
    >>> set_defaults()
    >>> from simetri.group.batch import Group
    >>> from simetri.shapes.shape import Shape
    >>> g = Group(
    ...     [
    ...         Shape([(0, 0), (10, 0), (10, 10)], closed=True),
    ...         Shape([(20, 0), (30, 0)]),
    ...     ]
    ... )
    >>> len(g)
    2
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from itertools import combinations
from typing import TYPE_CHECKING, Any, Self

from numpy import array
from numpy.typing import NDArray

from ..base.all_enums import (
    InPlace,
    TransformationType,
    Types,
    WarningType,
    get_enum_value,
)
from ..base.common import LineType, PointType, get_unique_id
from ..base.common_style import coerce_style_overlay
from ..base.core import (
    STYLE_ATTRIBUTES,
    Base,
    _next_xform_matrix,
    _Targets,
)
from ..config.settings import defaults, issue_warning
from ..geom.bbox import BoundingBox, bounding_box
from ..geom.points.point_utils import distance, fix_degen_points, round_point
from ..geom.polygons.poly import get_polygons
from ..geom.segments.line_utils import round_segment
from ..helpers.modifiers import Modifier
from .merge import (
    _closest_angle_differences,
    _merge_collinears,
    _merge_shapes,
    _segment_angles,
)

if TYPE_CHECKING:
    from ..shapes.shape import Shape


def check_dist_tol(
    shapes_groups: Any | Sequence[Any],
    n: int,
    n_round: int | None = None,
) -> set[float]:
    """Return up to ``n`` smallest positive pairwise vertex distances.

    Args:
        shapes_groups: One shape/group or a sequence mixing shapes and groups.
        n: Number of distinct distances to return.
        n_round: Number of decimal places used to round distances. If ``None``,
            uses the configured ``n_round`` default.

    Returns:
        set[float]: Up to ``n`` smallest positive rounded distances.

    Examples:
        >>> from simetri.config.settings import set_defaults
        >>> set_defaults()
        >>> from simetri.group.batch import check_dist_tol
        >>> from simetri.shapes.shape import Shape
        >>> values = check_dist_tol(
        ...     [
        ...         Shape([(0, 0), (5e-14, 0)]),
        ...         Shape([(0, 0), (1, 0)]),
        ...     ],
        ...     2,
        ...     n_round=12,
        ... )
        >>> values == {1.0}
        True
"""
    return Group(shapes_groups).check_dist_tol(n, n_round=n_round)


def check_angle_tol(
    shapes_groups: Any | Sequence[Any],
    n: int,
    n_round: int | None = None,
) -> set[float]:
    """Return up to ``n`` smallest positive edge-angle differences.

    Args:
        shapes_groups: One shape/group or a sequence mixing shapes and groups.
        n: Number of distinct angle differences to return.
        n_round: Number of decimal places used to round angle differences. If
            ``None``, uses the configured ``n_round`` default.

    Returns:
        set[float]: Up to ``n`` smallest positive rounded angle differences.

    Examples:
        >>> from simetri.config.settings import set_defaults
        >>> set_defaults()
        >>> from simetri.group.batch import check_angle_tol
        >>> from simetri.shapes.shape import Shape
        >>> values = check_angle_tol(
        ...     [
        ...         Shape([(0, 0), (1, 0)]),
        ...         Shape([(0, 0), (0, 1)]),
        ...     ],
        ...     1,
        ...     n_round=2,
        ... )
        >>> values == {1.57}
        True
"""
    return Group(shapes_groups).check_angle_tol(n, n_round=n_round)


class Group(Base):
    """Collection of drawable elements that transform together.

    Elements may be Shape, Group, or Tag objects. Methods such as
    ``all_vertices``, ``all_edges``, ``all_segments``, and ``all_shapes``
    flatten nested groups recursively.
    Alias: ``Batch`` on the ``simetri.graphics`` namespace.

    Attributes:
        elements: Flat list of top-level members.
        type: Always ``Types.GROUP``.
        subtype: Group subtype (for example ``Types.DOTS``).
        modifiers: Optional animation/constraint modifiers.
        visible: Whether the group is drawn.
        id: Unique object id.

    Examples:
        >>> from simetri.config.settings import set_defaults
        >>> set_defaults()
        >>> from simetri.group.batch import Group
        >>> from simetri.shapes.shape import Shape
        >>> g = Group([Shape([(0, 0), (1, 0)])])
        >>> _ = g.append(Shape([(2, 0), (3, 0)]))
        >>> len(g)
        2
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
        *elements: Any,
        modifiers: Sequence[Modifier] | None = None,
        subtype: Types = Types.GROUP,
    ) -> None:
        """Initialize a Group.

        Args:
            *elements: Drawables to include. Pass them individually
                (``Group(a, b)``) or as one sequence (``Group([a, b])``).
                Nested lists/tuples are flattened. Omit for an empty group.
            modifiers: Optional modifiers applied to the group.
            subtype: Group subtype enum or name. Defaults to ``Types.GROUP``.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> len(Group([Shape([(0, 0), (1, 0)])]))
            1
            >>> len(Group())
            0
        """

        def flatten_elements(nested_list: Any) -> Iterator[Any]:
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

        if not elements:
            self.elements = []
        elif len(elements) == 1 and isinstance(elements[0], (list, tuple)):
            _elements = []
            for element in elements[0]:
                if isinstance(element, (list, tuple)):
                    _elements.extend(
                        elem for elem in flatten_elements(element) if elem
                    )
                else:
                    if element:
                        _elements.append(element)
            self.elements = _elements[:]
        elif len(elements) == 1:
            self.elements = [elements[0]]
        else:
            _elements = []
            for element in elements:
                if isinstance(element, (list, tuple)):
                    _elements.extend(
                        elem for elem in flatten_elements(element) if elem
                    )
                else:
                    if element:
                        _elements.append(element)
            self.elements = _elements[:]

        self.type = Types.GROUP
        self.subtype = get_enum_value(Types, subtype)
        if modifiers is None:
            modifiers = []
        self.modifiers = modifiers
        self.visible = True
        self.id = get_unique_id(self)

    def set_style(self, mapping: Any = None, **kwargs: object) -> Self:
        """Set style fields on every member (mutated).

        Nested groups are walked like ``set_attribs``. A missing attribute
        on a member raises.

        Args:
            mapping: A ``Style``, a dict of draw aliases, or omitted.
            **kwargs: Draw-alias fields; overwrite ``mapping`` for those keys.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (1, 0)])])
            >>> g.set_style(line_width=3) is g
            True
            >>> g[0].line_width
            3
        """
        if mapping is None and not kwargs:
            raise TypeError(
                "Group.set_style() requires a Style, a dict, or keyword arguments"
            )
        overlay = coerce_style_overlay(mapping, kwargs)
        for element in self.elements:
            if element.type == Types.GROUP:
                element.set_style(overlay)
            else:
                for key, value in overlay.items():
                    setattr(element, key, value)
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """Warn when a Shape attribute is assigned directly to a bare Group.

        Subclasses that own style on themselves (e.g. Path2D) are not warned.

        Examples:
            >>> import warnings
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (1, 0)])])
            >>> with warnings.catch_warnings(record=True):
            ...     g.line_width = 2
            >>> True
            True
        """
        if (
            type(self) is Group
            and name in STYLE_ATTRIBUTES
            and name != "subtype"
        ):
            issue_warning(
                f"'{name}' is a Shape property and has no effect on a Group. "
                f"Use group.set_attribs('{name}', value) to apply it to the "
                "shapes in the group.",
                warning_type=WarningType.group.shape_attr,
                stacklevel=3,
            )
        super().__setattr__(name, value)

    def set_attribs(
        self, attrib: str, value: Any, key: Callable | None = None
    ) -> Self:
        """
        Sets the attribute to the given value for all elements in the group if it is applicable.

        Args:
            attrib (str): The attribute to set.
            value (Any): The value to set the attribute to.
            key (Callable, optional): A function to filter elements by a specific key. Defaults to None.

        Returns:
            Self: The group object.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (1, 0)])])
            >>> _ = g.set_attribs("line_width", 2)
            >>> g[0].line_width
            2
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

    def __str__(self) -> str:
        """
        Return a string representation of the group.

        Returns:
            str: The string representation of the group.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> str(Group()).startswith("Group")
            True
        """
        if self.elements is None or len(self.elements) == 0:
            res = "Group()"
        elif len(self.elements) in [1, 2]:
            res = f"Group({self.elements})"
        else:
            res = f"Group({self.elements[0]}...{self.elements[-1]})"
        return res

    def __repr__(self) -> str:
        """
        Return a string representation of the group.

        Returns:
            str: The string representation of the group.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> repr(Group()) == str(Group())
            True
        """
        return self.__str__()

    def __len__(self) -> int:
        """
        Return the number of elements in the group.

        Returns:
            int: The number of elements in the group.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> len(Group([Shape([(0, 0), (1, 0)]), Shape([(1, 0), (2, 0)])]))
            2
        """
        return len(self.elements)

    def __getitem__(
        self, subscript: int | slice
    ) -> Any | list[Any]:
        """
        Get the element(s) at the given subscript.

        Args:
            subscript (int or slice): The subscript to get the element(s) from.

        Returns:
            Any: The element(s) at the given subscript.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (1, 0)]), Shape([(1, 0), (2, 0)])])
            >>> g[0].type.name
            'SHAPE'
            >>> len(g[0:1])
            1
        """
        if isinstance(subscript, slice):
            res = self.elements[
                subscript.start : subscript.stop : subscript.step
            ]
        else:
            res = self.elements[subscript]
        return res

    def __setitem__(self, subscript: int | slice, value: Any) -> None:
        """
        Set the element(s) at the given subscript.

        Args:
            subscript (int or slice): The subscript to set the element(s) at.
            value (Any): The value to set the element(s) to.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (1, 0)])])
            >>> g[0] = Shape([(9, 0), (10, 0)])
            >>> len(g[0])
            2
        """
        elements = self.elements
        if isinstance(subscript, slice):
            elements[subscript.start : subscript.stop : subscript.step] = value
        elif isinstance(subscript, int):
            elements[subscript] = value
        else:
            raise TypeError("Invalid subscript type")

    def __add__(self, other: Group) -> Group:
        """
        Add another group to this group.

        Args:
            other (Group): The other group to add.

        Returns:
            Group: The combined group.

        Raises:
            RuntimeError: If the other object is not a group.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g1 = Group([Shape([(0, 0), (1, 0)])])
            >>> g2 = Group([Shape([(2, 0), (3, 0)])])
            >>> len(g1 + g2)
            2
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

    def __bool__(self) -> bool:
        """
        Return whether the group has any elements.

        Returns:
            bool: True if the group has elements, False otherwise.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> bool(Group())
            False
            >>> bool(Group([Shape([(0, 0), (1, 0)])]))
            True
        """
        return len(self.elements) > 0

    def __iter__(self) -> Iterator[Any]:
        """
        Return an iterator over the elements in the group.

        Returns:
            Iterator: An iterator over the elements in the group.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (1, 0)]), Shape([(1, 0), (2, 0)])])
            >>> len(list(g))
            2
        """
        return iter(self.elements)

    def _duplicates(self, elements: Sequence[Any]) -> bool:
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

    def proximity(
        self, dist_tol: float | None = None, n: int = 5
    ) -> list[PointType]:
        """
        Returns the n closest points in the group.

        Args:
            dist_tol (float, optional): The distance tolerance for proximity.
            n (int, optional): The number of closest points to return.

        Returns:
            list[PointType]: The n closest points in the group.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (1, 0), (1, 1)], closed=True)])
            >>> isinstance(g.proximity(n=1), list)
            True
        """
        if dist_tol is None:
            dist_tol = defaults["dist_tol"]
        vertices = self.all_vertices
        vertices = [(*v, i) for i, v in enumerate(vertices)]
        from ..geom.polygons.polygon import all_close_points

        _, pairs = all_close_points(vertices, dist_tol=dist_tol, with_dist=True)
        return [pair for pair in pairs if pair[2] > 0][:n]

    def check_dist_tol(
        self,
        n: int,
        n_round: int | None = None,
    ) -> set[float]:
        """Return up to ``n`` smallest positive rounded vertex distances.

        Args:
            n: Number of distinct distances to return.
            n_round: Number of decimal places used to round distances. If
                ``None``, uses the configured ``n_round`` default.

        Returns:
            set[float]: Up to ``n`` smallest positive rounded distances.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> group = Group([Shape([(0, 0), (5e-14, 0), (1, 0)])])
            >>> group.check_dist_tol(2, n_round=12) == {1.0}
            True
        """
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        if n_round is None:
            n_round = defaults["n_round"]
        if n_round < 0:
            raise ValueError("n_round must be a nonnegative integer.")

        vertices = self.all_vertices
        if len(vertices) < 2:
            return set()

        distances = set()
        for first_point, second_point in combinations(vertices, 2):
            point_distance = round(
                distance(first_point[:2], second_point[:2]),
                n_round,
            )
            if point_distance > 0:
                distances.add(point_distance)

        return set(sorted(distances)[:n])

    def check_angle_tol(
        self,
        n: int,
        n_round: int | None = None,
    ) -> set[float]:
        """Return up to ``n`` smallest positive rounded angle differences.

        Args:
            n: Number of distinct angle differences to return.
            n_round: Number of decimal places used to round angle differences.
                If ``None``, uses the configured ``n_round`` default.

        Returns:
            set[float]: Up to ``n`` smallest positive rounded angle differences.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> group = Group(
            ...     [
            ...         Shape([(0, 0), (1, 0)]),
            ...         Shape([(0, 0), (0, 1)]),
            ...     ]
            ... )
            >>> group.check_angle_tol(1, n_round=2) == {1.57}
            True
        """
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        if n_round is None:
            n_round = defaults["n_round"]
        if n_round < 0:
            raise ValueError("n_round must be a nonnegative integer.")

        angles = _segment_angles(self.all_segments)
        pair_count = len(angles) * (len(angles) - 1) // 2
        angle_differences = _closest_angle_differences(angles, pair_count)
        rounded_differences = set()
        for angle_difference in angle_differences:
            rounded_difference = round(angle_difference, n_round)
            if rounded_difference > 0:
                rounded_differences.add(rounded_difference)

        return set(sorted(rounded_differences)[:n])

    def append(self, element: Any) -> Self:
        """
        Appends the element to the group.

        Args:
            element (Any): The element to append.

        Returns:
            Self: The group object.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (1, 0)])])
            >>> g.append(Shape([(2, 0), (3, 0)])) is g
            True
            >>> len(g)
            2
        """
        if element in self.elements:
            issue_warning(
                f"Duplicate element added to Group: {element}",
                warning_type=WarningType.group.duplicate,
                stacklevel=2,
            )
        self.elements.append(element)
        return self

    def reverse(self) -> Self:
        """
        Reverses the order of the elements in the group.

        Returns:
            Self: The group object.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> first = Shape([(0, 0), (1, 0)])
            >>> second = Shape([(2, 0), (3, 0)])
            >>> g = Group([first, second])
            >>> _ = g.reverse()
            >>> g[0] is second
            True
        """
        self.elements = self.elements[::-1]
        return self

    def insert(self, index: int, element: Any) -> Self:
        """
        Inserts the element at the given index.

        Args:
            index (int): The index to insert the element at.
            element (Any): The element to insert.

        Returns:
            Self: The group object.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (1, 0)])])
            >>> _ = g.insert(0, Shape([(5, 0), (6, 0)]))
            >>> len(g)
            2
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

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> extra = Shape([(5, 0), (6, 0)])
            >>> g = Group([Shape([(0, 0), (1, 0)]), extra])
            >>> _ = g.remove(extra)
            >>> len(g)
            1
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

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (1, 0)]), Shape([(2, 0), (3, 0)])])
            >>> removed = g.pop(0)
            >>> removed.type.name
            'SHAPE'
            >>> len(g)
            1
        """
        return self.elements.pop(index)

    def clear(self) -> Self:
        """
        Removes all elements from the group.

        Returns:
            Self: The group object.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (1, 0)])])
            >>> _ = g.clear()
            >>> len(g)
            0
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

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (1, 0)])])
            >>> _ = g.extend([Shape([(2, 0), (3, 0)])])
            >>> len(g)
            2
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

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.base.all_enums import Types
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> nested = Group(
            ...     [Group([Shape([(0, 0), (1, 0)])]), Shape([(2, 0), (3, 0)])]
            ... )
            >>> len(list(nested.iter_elements(Types.SHAPE)))
            2
        """
        for elem in self.elements:
            if elem.type == Types.GROUP:
                yield from elem.iter_elements(element_type)
            else:
                if element_type is None or elem.type == element_type:
                    yield elem

    @property
    def all_elements(self) -> list[Any]:
        """Return a list of all elements in the group,
        including the elements in the nested groups.

        Returns:
            list[Any]: A list of all elements in the group.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> nested = Group(
            ...     [Group([Shape([(0, 0), (1, 0)])]), Shape([(2, 0), (3, 0)])]
            ... )
            >>> len(nested.all_elements)
            2
        """
        elements = []
        for elem in self.elements:
            if elem.type == Types.GROUP:
                elements.extend(elem.all_elements)
            else:
                elements.append(elem)
        return elements

    @property
    def all_shapes(self) -> list[Shape]:
        """Return a list of all shapes in the group.

        Returns:
            list[Shape]: A list of all shapes in the group.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> nested = Group(
            ...     [Group([Shape([(0, 0), (1, 0)])]), Shape([(2, 0), (3, 0)])]
            ... )
            >>> len(nested.all_shapes)
            2
        """
        elements = self.all_elements
        return [
            element for element in elements if element.type == Types.SHAPE
        ]

    @property
    def all_vertices(self) -> list[PointType]:
        """Return a list of all points in the group in their
        transformed positions.

        Returns:
            list[PointType]: A list of all points in the group in their transformed positions.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (10, 0), (10, 10)], closed=True)])
            >>> len(g.all_vertices)
            3
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

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (10, 0), (10, 10)], closed=True)])
            >>> len(g.all_segments)
            3
        """
        elements = self.all_elements
        segments = []
        for element in elements:
            if element.type == Types.SHAPE:
                segments.extend(element.vertex_pairs)
            elif element.type == Types.GROUP:
                segments.extend(element.all_segments)
        return segments

    @property
    def all_edges(self) -> list[LineType]:
        """Return a list of all segments in the group.

        Returns:
            list[LineType]: A list of all segments in the group.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (10, 0), (10, 10)], closed=True)])
            >>> g.all_edges is g.all_segments
            False
            >>> len(g.all_edges) == len(g.all_segments)
            True
        """
        # This is an alias for all_segments!

        return self.all_segments

    def merge_collinears(
        self,
        edges: list[tuple[int, int]],
        merge_angle_tol: float = 0.1,
        debug: bool = False,
        remove_duplicate_edges: bool = False,
    ) -> list[LineType]:
        """Merge connected collinear edges into longer segments.

        Args:
            edges: Edges as node-id pairs (see ``_set_node_dictionaries``).
            merge_angle_tol: Angle tolerance in radians for collinearity.
            debug: If True, print rejected-angle diagnostics.
            remove_duplicate_edges: If True, keep one copy of each congruent
                edge and drop the extra duplicates before merging.

        Returns:
            list[LineType]: Merged segments as coordinate pairs.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (10, 0)]), Shape([(10, 0), (20, 0)])])
            >>> g._set_node_dictionaries(g.all_vertices, dist_tol=0.01)
            >>> edges, _ = g._get_edges_and_segments()
            >>> len(g.merge_collinears(edges, merge_angle_tol=0.1))
            1
        """
        return _merge_collinears(
            self,
            edges,
            merge_angle_tol=merge_angle_tol,
            debug=debug,
            remove_duplicate_edges=remove_duplicate_edges,
        )

    def merge_shapes(
        self,
        dist_tol: float | None = None,
        merge_angle_tol: float = 0.1,
        debug: bool = False,
        remove_duplicate_edges: bool = True,
    ) -> Self:
        """Merge connected shapes into polygons and open polylines.

        Returns a new group containing reconstructed shapes from the edge
        graph. Unmerged content may be omitted depending on connectivity.

        Args:
            dist_tol: Vertex snap tolerance. Defaults to library ``dist_tol``.
            merge_angle_tol: Collinearity angle tolerance in radians.
            debug: If True, print merge diagnostics.
            remove_duplicate_edges: If True, keep one copy of each congruent
                edge and drop extra duplicates first.

        Returns:
            Group: New group of merged shapes.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group(
            ...     [Shape([(0, 0), (10, 0)]), Shape([(10, 0), (0, 0)])]
            ... )
            >>> merged = g.merge_shapes()
            >>> len(merged)
            1
            >>> len(merged[0])
            2
        """
        return _merge_shapes(
            self,
            dist_tol=dist_tol,
            merge_angle_tol=merge_angle_tol,
            debug=debug,
            remove_duplicate_edges=remove_duplicate_edges,
        )

    def _get_edges_and_segments(
        self,
        n_round: int = 2,
    ) -> tuple[list[tuple[int, int]], list[LineType]]:
        """Get the edges and segments for the group.

        Args:
            n_round (int, optional): The number of decimal places to round to. Defaults to None.

        Returns:
            tuple[list[tuple[int, int]], list[LineType]]: Node-id edges and
            rounded segment geometry.
        """
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
        self,
        coords: list[PointType],
        dist_tol: float,
        debug: bool = False,
    ) -> list[dict]:
        """Set dictionaries for nodes and coordinates.
        d_node_coord: Dictionary of node id to coordinates.
        d_coord_node: Dictionary of coordinates to node id.

        Args:
            nodes (list[PointType]): list of vertices.
            dist_tol (float): Distance tolerance for grouping coordinates.
            debug (bool, optional): Print node proximity diagnostics.
                Defaults to False.
        """
        from ..geom.polygons.polygon import node_dictionaries

        (
            self.d_node_coord,
            self.d_coord_node,
            self.d_rounded_coord,
        ) = node_dictionaries(coords, dist_tol, debug=debug)

    def all_polygons(self, dist_tol: float | None = None) -> list:
        """Return a list of all polygons in the group in their
        transformed positions.

        Args:
            dist_tol (float, optional): The distance tolerance for proximity. Defaults to None.

        Returns:
            list: A list of all polygons in the group.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (10, 0), (10, 10)], closed=True)])
            >>> len(g.all_polygons())
            1
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
        if polylines:
            fixed_polylines = [
                fix_degen_points(polyline, dist_tol=dist_tol, closed=True)
                for polyline in polylines
            ]
            polygons = get_polygons(fixed_polylines, dist_tol=dist_tol)
            res = polygons + exclude
        else:
            res = exclude
        return res

    def copy(self) -> Group:
        """Returns a copy of the group.

        The copy has the same class as this group, so subclasses keep their
        type, ``subtype``, ``visible`` flag, and their own attributes without
        writing a ``copy`` of their own. ``__init__`` is not called, since
        subclass constructors take different arguments.

        Elements are copied and ``modifiers`` is a new list. All other state
        is copied by reference.

        Returns:
            Group: A copy of the group.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (1, 0)])])
            >>> copy = g.copy()
            >>> len(copy) == len(g)
            True
            >>> copy is g
            False
        """

        b = object.__new__(type(self))
        b.__dict__.update(self.__dict__)
        b.elements = [elem.copy() for elem in self.elements]
        b.modifiers = self.modifiers[:]
        b.id = get_unique_id(b)
        return b

    @property
    def b_box(self) -> BoundingBox:
        """Returns the bounding box of the group.

        Built by uniting each element's own ``b_box`` (not raw vertices), so
        shapes such as ``Circle`` that store only a center still contribute
        their full extents.

        Returns:
            BoundingBox: The bounding box of the group.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (10, 0), (10, 10)], closed=True)])
            >>> g.b_box.southwest is not None
            True
        """
        # To do: memoize the bounding box
        corners = []
        for element in self.elements:
            box = element.b_box
            if box.southwest is None:
                continue
            corners.extend(box.corners)
        return bounding_box(array(corners))

    def _modify(self, modifier: Modifier) -> None:
        """Apply a modifier to the group.

        Args:
            modifier (Modifier): The modifier to apply.
        """
        modifier.apply()

    def _update(
        self,
        xform_matrix: NDArray,
        reps: int = 0,
        take: slice | None = None,
        incr: float
        | tuple[float, float]
        | tuple[callable, Any]
        | tuple[InPlace, Any]
        | None = None,
        dyn_ref: Callable | None = None,
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
            dyn_ref: Matrix factory built by the transform method when
                dynamic references are in use. Defaults to None.
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
            if dyn_ref:
                # self grows in place, so it is the accumulated pattern.
                targets = _Targets(Group(self.elements[:]), self)
                targets.active = Group()
            else:
                targets = None
            new = []
            for i in range(reps):
                if targets is not None:
                    targets.active.elements = elements
                xform_matrix = _next_xform_matrix(
                    xform_matrix, xform_type, incr, dyn_ref, targets, i
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
        # check if the bounding-boxes of the elements have changes

        return self

    def union(self, other: Group) -> Self:
        """Returns the union of two groups.

        Args:
            other (Group): The other group to union with.

        Returns:
            Group: The union of the two groups.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> shared = Shape([(2, 0), (3, 0)])
            >>> g1 = Group([Shape([(0, 0), (1, 0)]), shared])
            >>> g2 = Group([shared, Shape([(4, 0), (5, 0)])])
            >>> len(g1.union(g2))
            3
        """
        if not isinstance(other, Group):
            raise TypeError(
                "Invalid object. Only Group objects can be unioned!"
            )

        seen: set[Any] = set()
        merged: list[Any] = []
        for item in (*self.elements, *other.elements):
            key = item.id if hasattr(item, "id") else id(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)

        return Group(
            merged,
            modifiers=self.modifiers,
            subtype=self.subtype,
        )

    def intersection(self, other: Group) -> Self:
        """Returns the intersection of two groups.

        Args:
            other (Group): The other group to intersect with.

        Returns:
            Group: The intersection of the two groups.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> shared = Shape([(2, 0), (3, 0)])
            >>> g1 = Group([Shape([(0, 0), (1, 0)]), shared])
            >>> g2 = Group([shared, Shape([(4, 0), (5, 0)])])
            >>> len(g1.intersection(g2))
            1
        """
        if not isinstance(other, Group):
            raise TypeError(
                "Invalid object. Only Group objects can be intersected!"
            )

        self_ids = {item.id for item in self.elements}
        other_ids = {item.id for item in other.elements}

        intersection_ids = self_ids.intersection(other_ids)

        return Group(
            [
                item for item in self.elements if item.id in intersection_ids
            ],
            modifiers=self.modifiers,
            subtype=self.subtype,
        )

    def difference(self, other: Group) -> Self:
        """Returns the difference of two groups.

        Args:
            other (Group): The other group to subtract.

        Returns:
            Group: The difference of the two groups.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> shared = Shape([(2, 0), (3, 0)])
            >>> g1 = Group([Shape([(0, 0), (1, 0)]), shared])
            >>> g2 = Group([shared, Shape([(4, 0), (5, 0)])])
            >>> len(g1.difference(g2))
            1
        """
        if not isinstance(other, Group):
            raise TypeError(
                "Invalid object. Only Group objects can be subtracted!"
            )

        self_ids = {item.id for item in self.elements}
        other_ids = {item.id for item in other.elements}

        difference_ids = self_ids.difference(other_ids)

        return Group(
            [
                item for item in self.elements if item.id in difference_ids
            ],
            modifiers=self.modifiers,
            subtype=self.subtype,
        )

    def symmetric_difference(self, other: Group) -> Self:
        """Returns the symmetric difference of two groups.

        Args:
            other (Group): The other group to find the symmetric difference with.

        Returns:
            Group: The symmetric difference of the two groups.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> shared = Shape([(2, 0), (3, 0)])
            >>> g1 = Group([Shape([(0, 0), (1, 0)]), shared])
            >>> g2 = Group([shared, Shape([(4, 0), (5, 0)])])
            >>> len(g1.symmetric_difference(g2))
            2
        """
        if not isinstance(other, Group):
            raise TypeError(
                "Invalid object. Only Group objects can be symmetrically differenced!"
            )

        self_ids = {item.id for item in self.elements}
        other_ids = {item.id for item in other.elements}

        symmetric_difference_ids = self_ids.symmetric_difference(other_ids)

        seen: set[Any] = set()
        merged: list[Any] = []
        for item in (*self.elements, *other.elements):
            key = item.id if hasattr(item, "id") else id(item)
            if key not in symmetric_difference_ids or key in seen:
                continue
            seen.add(key)
            merged.append(item)

        return Group(
            merged,
            modifiers=self.modifiers,
            subtype=self.subtype,
        )

    def subset(self, other: Group) -> bool:
        """Checks if the current group is a subset of another group.

        Args:
            other (Group): The other group to check against.

        Returns:
            bool: True if the current group is a subset of the other group, False otherwise.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> shared = Shape([(0, 0), (1, 0)])
            >>> inner = Group([shared])
            >>> outer = Group([shared, Shape([(2, 0), (3, 0)])])
            >>> inner.subset(outer)
            True
        """
        if not isinstance(other, Group):
            raise TypeError(
                "Invalid object. Only Group objects can be checked for subset!"
            )

        self_ids = {item.id for item in self.elements}
        other_ids = {item.id for item in other.elements}

        return self_ids.issubset(other_ids)

    def superset(self, other: Group) -> bool:
        """Checks if the current group is a superset of another group.

        Args:
            other (Group): The other group to check against.

        Returns:
            bool: True if the current group is a superset of the other group, False otherwise.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> a = Shape([(0, 0), (1, 0)])
            >>> inner = Group([a])
            >>> outer = Group([a, Shape([(2, 0), (3, 0)])])
            >>> outer.superset(inner)
            True
        """
        if not isinstance(other, Group):
            raise TypeError(
                "Invalid object. Only Group objects can be checked for superset!"
            )

        self_ids = {item.id for item in self.elements}
        other_ids = {item.id for item in other.elements}

        return self_ids.issuperset(other_ids)

    @property
    def ids(self) -> list[Any]:
        """Return a list of ids of the elements in the group.

        If the element has an ``id`` attribute, it is used; otherwise ``id(element)``.

        Returns:
            list[Any]: Element ids for top-level members.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (1, 0)]), Shape([(2, 0), (3, 0)])])
            >>> len(g.ids)
            2
        """
        return [
            item.id if hasattr(item, "id") else id(item)
            for item in self.elements
        ]

    @property
    def all_ids(self) -> list[Any]:
        """Return ids for all elements, including nested groups.

        If the element has an ``id`` attribute, it is used; otherwise ``id(element)``.

        Returns:
            list[Any]: Flattened element ids.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> nested = Group(
            ...     [Group([Shape([(0, 0), (1, 0)])]), Shape([(2, 0), (3, 0)])]
            ... )
            >>> len(nested.all_ids)
            2
        """
        ids = []
        for item in self.elements:
            if hasattr(item, "type") and item.type == Types.GROUP:
                ids.extend(item.all_ids)
            else:
                ids.append(item.id if hasattr(item, "id") else id(item))

        return ids

    def __hash__(self) -> int:
        """Return the hash of the group.

        Returns:
            int: The hash of the group.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (1, 0)])])
            >>> isinstance(hash(g), int)
            True
        """
        return hash(tuple(self.ids))

    def __eq__(self, other: object) -> bool:
        """Check if two groups are equal.

        Args:
            other (object): The other group to compare.

        Returns:
            bool: True if the groups are equal, False otherwise.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.group.batch import Group
            >>> from simetri.shapes.shape import Shape
            >>> g = Group([Shape([(0, 0), (1, 0)])])
            >>> g == g
            True
            >>> g == Group([Shape([(0, 0), (1, 0)])])
            True
            >>> g == Group([Shape([(0, 0), (2, 0)])])
            False
        """
        if not isinstance(other, Group):
            return False

        if len(self.elements) != len(other.elements):
            return False

        return (
            self.elements == other.elements
            and self.modifiers == other.modifiers
        )


def custom_group_attributes(item: Group) -> list[str]:
    """
    Return a list of custom attributes of a Shape or
    Group instance.

    Args:
        item (Group): The group object.

    Returns:
        list[str]: A list of custom attributes.

    Examples:
        >>> from simetri.config.settings import set_defaults
        >>> set_defaults()
        >>> from simetri.group.batch import Group, custom_group_attributes
        >>> from simetri.shapes.shape import Shape
        >>> attrs = custom_group_attributes(Group([Shape([(0, 0), (1, 0)])]))
        >>> isinstance(attrs, list)
        True
    """
    from ..shapes.shape import Shape

    if isinstance(item, Group):
        dummy_shape = Shape([(0, 0), (1, 0)])
        dummy = Group([dummy_shape])
    else:
        raise TypeError("Invalid item type")
    native_attribs = set(dir(dummy))
    custom_attribs = set(dir(item)) - native_attribs

    return list(custom_attribs)

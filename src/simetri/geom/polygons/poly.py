"""Lightweight NumPy-backed polygon/polyline geometry (no style)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from numpy import around
from numpy.typing import NDArray

from simetri.base.all_enums import InPlace, Types
from simetri.config.settings import runtime_defaults as defaults
from simetri.geom.geom_utils import connected_pairs
from simetri.geom.points.point_utils import fix_degen_points

from ...base.all_enums import Anchor, Side, TransformationType
from ...base.common import LineType, PointType, get_unique_id
from ...base.core import _update_inplace
from ...helpers.utilities import decompose_transformations
from ..affine import mirror_matrix
from ..segments.line_utils import offset_line

if TYPE_CHECKING:
    from ...group.batch import Group


class TrackedArray(np.ndarray):
    """NumPy array subclass that clears a parent cache on item assignment."""

    def __setitem__(self, key: Any, value: Any) -> None:
        # print(# Here is your trigger event
        #     f"Alert: Index {key} is changing from {self[key]} to {value}!"
        # )
        super()._cache = {}
        super().__setitem__(key, value)


def to_array(points: list | tuple | NDArray) -> NDArray:
    """Convert points to a homogeneous NumPy array.

    Args:
        points: Sequence of ``(x, y)`` or already-homogeneous rows.

    Returns:
        NumPy array with shape ``(n, 3)``.

    Examples:
        >>> from simetri.geom.polygons.poly import to_array
        >>> arr = to_array([(0, 0), (1, 2)])
        >>> arr.shape
        (2, 3)
        >>> arr[0, 2]
        1.0
    """
    # convert points to a numpy array
    res = points
    if not isinstance(points, np.ndarray):
        res = np.array(points)

    # if they are not homogeneous coordinates, convert them
    if res.shape[1] == 2:
        res = np.column_stack((res, np.ones(res.shape[0])))

    return res


s_bbox_props = {
    "east",
    "west",
    "north",
    "south",
    "southwest",
    "southeast",
    "northwest",
    "northeast",
    "left",
    "bottom",
    "right",
    "top",
    "diagonal1",
    "diagonal2",
    "horiz_centerline",
    "vert_centerline",
}

bbox_aliases = {
    "e": "east",
    "w": "west",
    "n": "north",
    "s": "south",
    "sw": "southwest",
    "se": "southeast",
    "nw": "northwest",
    "ne": "northeast",
    "mid": "midpoint",
    "d1": "diagnoal1",
    "d2": "diagonal2",
    "vcl": "vert_centerline",
    "hcl": "horiz_centerline",
}


class Poly:
    """Light-weight polygon/polyline objects.
    They do not have any style properties, only geometry.
    All data is represented as numpy arrays.
    They can be transformed using translate, mirror, rotate, scale, shear,
    and glide methods.
    Transformations with reps > 0 return a Group object. The Poly object
    itself stays unchanged. reps = 0 modifies the xform_matrix.
    They are not meant to be modified.
    Most modifications create a new primary_points array.
    """

    __slots__ = (
        "_bbox",
        "_vertices",
        "closed",
        "id",
        "primary_points",
        "subtype",
        "type",
        "xform_matrix",
    )

    def __init__(self, points: list | tuple | NDArray, closed: bool = False) -> None:
        """Create a lightweight polygon/polyline from ``points``.

        Args:
            points: Vertex sequence or array (Cartesian or homogeneous).
            closed: If True, treat the polyline as a closed polygon.

        Examples:
            >>> from simetri.geom.polygons.poly import Poly
            >>> poly = Poly([(0, 0), (1, 0), (1, 1)], closed=True)
            >>> poly.closed
            True
            >>> poly.primary_points.shape
            (3, 3)
        """
        self.primary_points = to_array(points).view(TrackedArray)
        self.xform_matrix = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        self.closed = closed
        self.id = get_unique_id(self)
        self.type = Types.POLY
        self.subtype = Types.POLY
        self._bbox = PolyBBox()  # empty bounding box

    @property
    def vertices(self) -> tuple[PointType, ...]:
        # return self.primary_points @ self.xform_matrix
        """The final coordinates of the shape.

        Returns:
            Transformed vertex coordinates.
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

    @vertices.setter
    def vertices(self, value: object) -> None:
        """No-op setter; vertices are derived from ``primary_points``."""

    def __getattr__(self, name: str) -> Any:
        try:
            # try bounding-box properties first
            if name in s_bbox_props:
                if self._bbox._cache:
                    return getattr(self._bbox, name)
                else:
                    self._reset_bbox()
            else:
                raise AttributeError(f"Invalid attribute: {name}")
        except AttributeError:
            print(f"Invalid attribute: {name}")

    def __setattr__(self, name: str, value: object) -> None:
        if name in ("primary_points", "xform_matrix"):
            try:
                bbox = object.__getattribute__(self, "_bbox")
            except AttributeError:
                bbox = None
            if bbox is not None:
                object.__setattr__(bbox, "_cache", {})
        object.__setattr__(self, name, value)

    @property
    def vertices(self) -> tuple[PointType, ...]:
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

    def _reset_bbox(self) -> None:
        vertices = self.vertices
        xs = vertices[:, 0]
        ys = vertices[:, 1]
        min_x = xs.min()
        max_x = xs.max()
        min_y = ys.min()
        max_y = ys.max()

        self._bbox._reset((min_x, min_y, max_x, max_y))

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
        dyn_ref: bool | None = None,
        merge: bool = False,
        xform_type: TransformationType = None,
    ) -> Self | Group:
        """Used internally. Update the shape with a transformation matrix.

        Args:
            xform_matrix (array): The transformation matrix.
            reps (int, optional): The number of repetitions, defaults to 0.

        Returns:
            Shape or Group: The updated shape or a group of shapes.

        Raises:
            ValueError: If ``dyn_ref`` is used.
        """
        if dyn_ref:
            raise ValueError(
                "Poly does not support dynamic references. Only Shape and "
                "Group resolve dyn_ref."
            )
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
            polys = [self]
            poly = self
            for i in range(reps):
                poly = poly.copy()
                if incr is not None and i > 0:
                    xform_matrix = _update_inplace(
                        xform_matrix, xform_type, incr
                    )

                poly._update(xform_matrix)
                polys.append(poly)
            from ...group.batch import Group

            res = Group(polys)

        if merge and reps > 0:
            return res.merge_shapes()

        return res

    def mirror(
        self,
        about: LineType | PointType | NDArray,
        reps: int = 0,
        take: slice | None = None,
        incr: float
        | tuple[float, float]
        | tuple[callable, Any]
        | tuple[InPlace, Any]
        | None = None,
        merge: bool = False,
    ) -> Self:
        """
        Mirrors the object about the given line or point.

        Args:
            about (LineType | PointType): The line or point to mirror about.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The mirrored object.
        """
        transform = mirror_matrix(about)
        res = self._update(
            transform,
            reps=reps,
            incr=incr,
            merge=merge,
            xform_type=TransformationType.MIRROR,
        )

        return res


"""Bounding box class. Shape, Group, and Poly objects have a bounding box.
Bounding box is axis-aligned. Provides reference edges and points.
"""


class PolyBBox:
    """
    Light-weight boundingbox.
    Computes and caches only the requested property.
    """

    __slots__ = ["_cache"]

    def __init__(
        self, corners: tuple[float, float, float, float] | None = None
    ) -> None:
        """Create an empty or corner-initialized axis-aligned bbox cache.

        Args:
            corners: Optional ``(min_x, min_y, max_x, max_y)`` extents.
        """
        if corners:
            self._reset(corners)
        else:
            self._cache = {}

    def _reset(self, corners: tuple[float, float, float, float]) -> None:
        """
        corners : (min_x, min_y, max_x, max_y)
        When the _xs and -ys change, _cache needs to be reset.
        """
        min_x, min_y, max_x, max_y = corners

        self._cache = {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "mid_x": (max_x - min_x) / 2,
            "mid_y": (max_y - min_y) / 2,
        }

    def _set_value(self, name: str) -> Any:
        cache = self._cache
        d_ref = {
            "west": lambda: (cache["min_x"], cache["mid_y"]),
            "southwest": lambda: (cache["min_x"], cache["min_y"]),
            "south": lambda: (cache["mid_x"], cache["min_y"]),
            "southeast": lambda: (cache["max_x"], cache["min_y"]),
            "east": lambda: (cache["max_x"], cache["mid_y"]),
            "northeast": lambda: (cache["max_x"], cache["max_y"]),
            "north": lambda: (cache["mid_x"], cache["max_y"]),
            "northwest": lambda: (cache["min_x"], cache["max_y"]),
            "left": lambda: (
                (cache["min_x"], cache["max_y"]),
                (cache["min_x"], cache["min_y"]),
            ),
            "bottom": lambda: (
                (cache["min_x"], cache["min_y"]),
                (cache["max_x"], cache["min_y"]),
            ),
            "right": lambda: (
                (cache["max_x"], cache["min_y"]),
                (cache["max_x"], cache["max_y"]),
            ),
            "top": lambda: (
                (cache["max_x"], cache["max_y"]),
                (cache["min_x"], cache["max_y"]),
            ),
            "midpoint": lambda: (cache["mid_x"], cache["mid_y"]),
            "corners": lambda: (
                (cache["min_x"], cache["min_y"]),
                (cache["max_x"], cache["min_y"]),
                (cache["max_x"], cache["max_y"]),
                (cache["min_x"], cache["max_y"]),
            ),
            "diagonal1": lambda: (
                (cache["min_x"], cache["min_y"]),
                (cache["max_x"], cache["max_y"]),
            ),
            "diagonal2": lambda: (
                (cache["max_x"], cache["min_y"]),
                (cache["min_x"], cache["max_y"]),
            ),
            "width": lambda: cache["max_x"] - cache["min_x"],
            "height": lambda: cache["max_y"] - cache["min_"],
            "horiz_centerline": lambda: (
                (cache["min_x"], cache["mid_y"]),
                (cache["max_x"], cache["mid_y"]),
            ),
            "vert_centerline": lambda: (
                (cache["mid_x"], cache["max_y"]),
                (cache["mid_x"], cache["min_y"]),
            ),
        }

        res = d_ref[name]()
        self._cache[name] = res

        return res

    def __getattr__(self, name: str) -> Any:
        if not self._cache:
            return None
        alias = bbox_aliases.get(name, None)
        if alias:
            name = alias

        # return the property if it exists, otherwise set it
        return self._cache.get(name, self._set_value(name))

    def offset_line(
        self, side: Side | str, offset: float
    ) -> tuple[PointType, PointType]:
        """Return a bbox edge offset outward by ``offset``.

        Args:
            side: Bbox side (``Side`` or name string).
            offset: Outward distance; use negative for inward.

        Returns:
            Offset segment as two points.
        """
        if isinstance(side, str):
            side = Side[side.upper()]

        if side == Side.RIGHT:
            x1, y1 = self.southeast
            x2, y2 = self.northeast
            res = ((x1 + offset, y1), (x2 + offset, y2))
        elif side == Side.LEFT:
            x1, y1 = self.southwest
            x2, y2 = self.northwest
            res = ((x1 - offset, y1), (x2 - offset, y2))
        elif side == Side.TOP:
            x1, y1 = self.northwest
            x2, y2 = self.northeast
            res = ((x1, y1 + offset), (x2, y2 + offset))
        elif side == Side.BOTTOM:
            x1, y1 = self.southwest
            x2, y2 = self.southeast
            res = ((x1, y1 - offset), (x2, y2 - offset))
        elif side == Side.DIAGONAL1:
            res = offset_line(self.diagonal1, offset)
        elif side == Side.DIAGONAL2:
            res = offset_line(self.diagonal2, offset)
        elif side == Side.H_CENTERLINE:
            res = offset_line(self.horiz_center_line, offset)
        elif side == Side.V_CENTERLINE:
            res = offset_line(self.vert_center_line, offset)
        else:
            raise ValueError(f"Unknown side: {side}")

        return res

    def offset_point(
        self, anchor: Anchor | str, dx: float, dy: float
    ) -> list[float]:
        """Return a point offset from a bbox anchor.

        Args:
            anchor: ``Anchor`` or anchor name string.
            dx: x offset.
            dy: y offset.

        Returns:
            Offset point ``[x, y]``.
        """
        if isinstance(anchor, str):
            anchor = Anchor[anchor.upper()]
            x, y = getattr(self, anchor.value)
        elif isinstance(anchor, Anchor):
            x, y = getattr(self, anchor.value)
        else:
            raise TypeError(f"Unknown anchor: {anchor}")
        return [x + dx, y + dy]


def get_polygons(
    nested_points: Sequence[PointType],
    n_round_digits: int = 2,
    abs_tol: float | None = None,
) -> list:
    """Convert points to clean polygons. Points are vertices of polygons.

    Args:
        nested_points (Sequence[PointType]): List of nested points.
        n_round_digits (int, optional): Number of decimal places to round to. Defaults to 2.
        abs_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        list: List of clean polygons.

    Examples:
        >>> from simetri.geom.polygons.poly import get_polygons
        >>> polys = get_polygons([[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]])
        >>> polys
        [[(1, 0), (0, 0), (0, 1), (1, 1)]]
    """
    from ...helpers.graph import get_cycles, sanitize_graph_edges

    if abs_tol is None:
        abs_tol = defaults["abs_tol"]

    nested_rounded_points = []
    for points in nested_points:
        rounded_points = []
        for point in points:
            rounded_point = (around(point, n_round_digits)).tolist()
            rounded_points.append(tuple(rounded_point))
        nested_rounded_points.append(rounded_points)

    s_points = set()
    d_id__point = {}
    d_point__id = {}
    for points in nested_rounded_points:
        for point in points:
            s_points.add(point)

    for i, fs_point in enumerate(s_points):
        d_id__point[i] = fs_point  # we need a bidirectional dictionary
        d_point__id[fs_point] = i

    nested_point_ids = []
    for points in nested_rounded_points:
        point_ids = [d_point__id[point] for point in points]
        nested_point_ids.append(point_ids)

    graph_edges = []
    for point_ids in nested_point_ids:
        graph_edges.extend(connected_pairs(point_ids))
    polygons = []
    graph_edges = sanitize_graph_edges(graph_edges)
    cycles = get_cycles(graph_edges)
    if cycles is None:
        return []
    for cycle_ in cycles:
        nodes = cycle_
        points = [d_id__point[i] for i in nodes]
        points = fix_degen_points(points, closed=True, abs_tol=abs_tol)
        polygons.append(points)

    return polygons

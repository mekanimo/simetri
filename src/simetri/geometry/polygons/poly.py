from collections.abc import Sequence
from typing import Any, Self

import numpy as np
from simetri.core.all_enums import InPlace, Types
from numpy.typing import NDArray

from simetri.core.common import PointType
from numpy import around
from simetri.geometry.geometry import connected_pairs
from simetri.geometry.points.point_utils import fix_degen_points
from simetri.settings.settings import defaults

from ..segments.line_utils import offset_line
from ...helpers.utilities import decompose_transformations
from ..affine import mirror_matrix
from ...core.all_enums import Anchor, Side, TransformationType
from ...core.common import LineType, PointType, get_unique_id
from ...core.core import _update_inplace


class TrackedArray(np.ndarray):
    def __setitem__(self, key, value):
        # print(# Here is your trigger event
        #     f"Alert: Index {key} is changing from {self[key]} to {value}!"
        # )
        super()._cache = {}
        super().__setitem__(key, value)


def to_array(points: list | tuple | NDArray):
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

    __slots__ = ["_vertices", "closed", "id", "primary_points", "xform_matrix"]

    def __init__(self, points: list | tuple | NDArray, closed: bool = False):
        self.primary_points = to_array(points).view(TrackedArray)
        self.xform_matrix = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
        self.closed = closed
        self.id = get_unique_id(self)
        self.type = Types.POLY
        self.subtype = Types.POLY
        self._bbox = PolyBBox()  # empty bounding box

    @property
    def vertices(self):
        # return self.primary_points @ self.xform_matrix
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

    @vertices.setter
    def vertices(self, value):
        pass

    def __getattr__(self, name):
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

    def __setattr__(self, name, value):
        if name in ("primary_points", "xform_matrix"):
            self._cache = {}
        setattr(self, name, value)

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

    def _reset_bbox(self):
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
        merge: bool = False,
        xform_type: TransformationType = None,
    ) -> Self | "Group":
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
    ):
        if corners:
            self._reset(corners)
        else:
            self._cache = {}

    def _reset(self, corners: tuple[float, float, float, float]):
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

    def _set_value(self, name):
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

    def __getattr__(self, name):
        if not self._cache:
            return None
        alias = bbox_aliases.get(name, None)
        if alias:
            name = alias

        # return the property if it exists, otherwise set it
        return self._cache.get(name, self._set_value(name))

    def offset_line(self, side, offset):
        """
        Offset is applied outwards. Use negative values for inward offset.

        Args:
            side (Side): The side to offset.
            offset (float): The offset distance.

        Returns:
            tuple: The offset line.
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

    def offset_point(self, anchor, dx, dy):
        """
        Return an offset point from the given reference point.

        Args:
            anchor (Anchor): The anchor point.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            list: The offset point.
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
    dist_tol: float | None = None,
) -> list:
    """Convert points to clean polygons. Points are vertices of polygons.

    Args:
        nested_points (Sequence[PointType]): List of nested points.
        n_round_digits (int, optional): Number of decimal places to round to. Defaults to 2.
        dist_tol (float, optional): Distance tolerance. Defaults to None.

    Returns:
        list: List of clean polygons.
    """
    from ..helpers.graph import sanitize_graph_edges

    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    from ..helpers.graph import get_cycles

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
        point_ids = []
        for point in points:
            point_ids.append(d_point__id[point])
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
        points = fix_degen_points(points, closed=True, dist_tol=dist_tol)
        polygons.append(points)

    return polygons

"""Axis-aligned bounding boxes for Shape and Group objects.

Provides reference anchors (corners, mid-sides, diagonals) and placement
helpers such as ``left_of``, ``above``, and ``polar_pos``.

**Examples**

```python
import simetri.graphics as sg
box = sg.BoundingBox((0, 0), (100, 50))
box.width, box.height
# (100, 50)
box.midpoint
# (50.0, 25.0)
```
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..base.all_enums import Anchor, Side, Types, WarningType
from ..base.common import PointType, get_unique_id
from ..config.settings import VOID, issue_warning, runtime_defaults
from .geom_utils import midpoint
from .geometry import polar_to_cartesian, positive_angle
from .points.point_utils import distance
from .segments.line_utils import offset_line

if TYPE_CHECKING:
    from ..group.batch import Group
    from ..shapes.shape import Shape


_ALIASES = {
    "s": "south",
    "n": "north",
    "w": "west",
    "e": "east",
    "sw": "southwest",
    "se": "southeast",
    "nw": "northwest",
    "ne": "northeast",
    "d1": "diagonal1",
    "d2": "diagonal2",
    "m": "midpoint",
    "vcl": "vert_centerline",
    "hcl": "horiz_centerline",
    "center": "midpoint",
}

_EXCLUSIVE = frozenset(
    (
        "line_color",
        "line_width",
        "line_dash_array",
        "stroke",
        "fill",
    )
)


class BoundingBox:
    """Axis-aligned rectangular bounding box.

    For a Shape it encloses all vertices; for a Group it encloses all
    vertices of all nested shapes. Exposes corner/edge anchors and
    placement helpers used by transforms and tags.

    Attributes:
        southwest: Lower-left corner ``(x, y)``, or ``None`` if empty.
        northeast: Upper-right corner ``(x, y)``, or ``None`` if empty.
        type: Always ``Types.BOUNDING_BOX``.
        id: Unique object id.

    Examples:
        >>> import simetri.graphics as sg
        >>> bb = sg.BoundingBox((0, 0), (40, 80))
        >>> bb.northwest
        (0, 80)
    """

    def __init__(
        self, southwest: PointType = None, northeast: PointType = None
    ) -> None:
        """Initialize a BoundingBox from opposite corners.

        Args:
            southwest: Southwest (min-x, min-y) corner.
            northeast: Northeast (max-x, max-y) corner.

        Examples:
            >>> import simetri.graphics as sg
            >>> bb = sg.BoundingBox((0, 0), (40, 80))
            >>> bb.southwest, bb.northeast
            ((0, 0), (40, 80))
            >>> empty = sg.BoundingBox()
            >>> empty.southwest is None
            True
        """
        # define the four corners
        if southwest is None or northeast is None:
            self.__dict__["southwest"] = None
            self.__dict__["northeast"] = None
            self.__dict__["northwest"] = None
            self.__dict__["southeast"] = None
        else:
            self.__dict__["southwest"] = southwest
            self.__dict__["northeast"] = northeast
            self.__dict__["northwest"] = (southwest[0], northeast[1])
            self.__dict__["southeast"] = (northeast[0], southwest[1])
        self._aliases = _ALIASES

        self.type = Types.BOUNDING_BOX
        self.subtype = Types.BOUNDING_BOX
        self.visible = True
        self.exclusive = _EXCLUSIVE

        self.id = get_unique_id(self)

    def __repr__(self) -> str:
        """Return a BoundingBox string from the opposite corners.

        Examples:
            >>> import simetri.graphics as sg
            >>> repr(sg.BoundingBox((0, 0), (40, 80)))
            'BoundingBox((0, 0), (40, 80))'
            >>> repr(sg.BoundingBox())
            'BoundingBox()'
        """
        if self.southwest is None or self.northeast is None:
            return "BoundingBox()"
        return f"BoundingBox({self.southwest!r}, {self.northeast!r})"

    def __getattr__(self, name: str) -> Any:
        """
        Get the attribute with the given name.

        Args:
            name (str): The name of the attribute.

        Returns:
            Any: The attribute with the given name.

        Examples:
            >>> import simetri.graphics as sg
            >>> bb = sg.BoundingBox((-40, -20), (80, 100))
            >>> bb.sw
            (-40, -20)
        """
        if name in self._aliases:
            if name == "center":
                issue_warning(
                    '"center" is deprecated use "midpoint" instead.',
                    warning_type=WarningType.deprecation.bbox_center,
                    category=DeprecationWarning,
                )
            return getattr(self, self._aliases[name])
        if name in self.__dict__:
            return self.__dict__[name]
        raise AttributeError(name)

    def angle_point(self, angle: float) -> PointType:
        """Return where a ray from the midpoint hits the box boundary.

        Args:
            angle: Ray direction in radians (from positive x-axis).

        Returns:
            PointType: Intersection of the ray with the bounding-box edge.

        Examples:
            >>> from simetri.geom.bbox import BoundingBox
            >>> bb = BoundingBox((0, 0), (40, 80))
            >>> tuple(round(v, 10) for v in bb.angle_point(0))
            (40.0, 40.0)
        """
        angle = positive_angle(angle)
        direction_x = np.cos(angle)
        direction_y = np.sin(angle)
        midpoint_x, midpoint_y = self.midpoint[:2]
        southwest_x, southwest_y = self.southwest[:2]
        northeast_x, northeast_y = self.northeast[:2]

        distances = []
        if direction_x > 0:
            distances.append((northeast_x - midpoint_x) / direction_x)
        elif direction_x < 0:
            distances.append((southwest_x - midpoint_x) / direction_x)
        if direction_y > 0:
            distances.append((northeast_y - midpoint_y) / direction_y)
        elif direction_y < 0:
            distances.append((southwest_y - midpoint_y) / direction_y)

        distance_to_edge = min(distances)
        return (
            midpoint_x + distance_to_edge * direction_x,
            midpoint_y + distance_to_edge * direction_y,
        )

    @property
    def left(self) -> tuple[PointType, PointType]:
        """
        Return the left edge.

        Returns:
            tuple: The left edge.

        Examples:
            >>> from simetri.geom.bbox import BoundingBox
            >>> bb = BoundingBox((0, 0), (40, 80))
            >>> bb.left == ((0, 80), (0, 0))
            True
        """
        return (self.northwest, self.southwest)

    @property
    def right(self) -> tuple[PointType, PointType]:
        """
        Return the right edge.

        Returns:
            tuple: The right edge.

        Examples:
            >>> import simetri.graphics as sg
            >>> bb = sg.BoundingBox((0, 0), (40, 80))
            >>> bb.right == ((40, 80), (40, 0))
            True
        """
        return (self.northeast, self.southeast)

    @property
    def top(self) -> tuple[PointType, PointType]:
        """
        Return the top edge.

        Returns:
            tuple: The top edge.

        Examples:
            >>> import simetri.graphics as sg
            >>> bb = sg.BoundingBox((0, 0), (40, 80))
            >>> bb.top == ((0, 80), (40, 80))
            True
        """
        return (self.northwest, self.northeast)

    @property
    def bottom(self) -> tuple[PointType, PointType]:
        """
        Return the bottom edge.

        Returns:
            tuple: The bottom edge.

        Examples:
            >>> import simetri.graphics as sg
            >>> bb = sg.BoundingBox((0, 0), (40, 80))
            >>> bb.bottom == ((0, 0), (40, 0))
            True
        """
        return (self.southwest, self.southeast)

    @property
    def vert_centerline(self) -> tuple[PointType, PointType]:
        """
        Return the vertical centerline.

        Returns:
            tuple: The vertical centerline.

        Examples:
            >>> import simetri.graphics as sg
            >>> bb = sg.BoundingBox((0, 0), (40, 80))
            >>> bb.vert_centerline == ((20.0, 80.0), (20.0, 0.0))
            True
        """
        return (self.north, self.south)

    @property
    def horiz_centerline(self) -> tuple[PointType, PointType]:
        """
        Return the horizontal centerline.

        Returns:
            tuple: The horizontal centerline.

        Examples:
            >>> import simetri.graphics as sg
            >>> bb = sg.BoundingBox((0, 0), (40, 80))
            >>> bb.horiz_centerline == ((0.0, 40.0), (40.0, 40.0))
            True
        """
        return (self.west, self.east)

    @property
    def midpoint(self) -> PointType:
        """
        Return the center of the bounding box.

        Returns:
            tuple: The center of the bounding box.

        Examples:
            >>> from simetri.geom.bbox import BoundingBox
            >>> BoundingBox((0, 0), (40, 80)).midpoint
            (20.0, 40.0)
        """
        x1, y1 = self.southwest
        x2, y2 = self.northeast

        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2

        return (xc, yc)

    @property
    def corners(
        self,
    ) -> tuple[PointType, PointType, PointType, PointType]:
        """
        Return the four corners of the bounding box.

        Returns:
            tuple: The four corners of the bounding box.

        Examples:
            >>> import simetri.graphics as sg
            >>> bb = sg.BoundingBox((0, 0), (40, 80))
            >>> bb.corners
            ((0, 80), (0, 0), (40, 0), (40, 80))
        """
        return (self.northwest, self.southwest, self.southeast, self.northeast)

    @property
    def diamond(
        self,
    ) -> tuple[PointType, PointType, PointType, PointType]:
        """
        Return the four center points of the bounding box in a diamond shape.

        Returns:
            tuple: The four center points of the bounding box in a diamond shape.

        Examples:
            >>> import simetri.graphics as sg
            >>> bb = sg.BoundingBox((0, 0), (40, 80))
            >>> bb.diamond
            ((20.0, 80.0), (0.0, 40.0), (20.0, 0.0), (40.0, 40.0))
        """
        return (self.north, self.west, self.south, self.east)

    @property
    def all_anchors(self) -> tuple[PointType, ...]:
        """
        Return all anchors of the bounding box.

        Returns:
            tuple: All anchors of the bounding box.

        Examples:
            >>> import simetri.graphics as sg
            >>> bb = sg.BoundingBox((0, 0), (40, 80))
            >>> bb.all_anchors
            ((0.0, 40.0), (0, 0), (20.0, 0.0), (40, 0), (40.0, 40.0), (40, 80), (20.0, 80.0), (0, 80), (20.0, 40.0))
        """
        # Do not change the order. LiBeRTy (Left, Bottom, Right, Top) is the order.
        return (
            self.west,
            self.southwest,
            self.south,
            self.southeast,
            self.east,
            self.northeast,
            self.north,
            self.northwest,
            self.midpoint,
        )

    @property
    def all_lines(
        self,
    ) -> tuple[
        tuple[PointType, PointType],
        tuple[PointType, PointType],
        tuple[PointType, PointType],
        tuple[PointType, PointType],
        tuple[PointType, PointType],
        tuple[PointType, PointType],
        tuple[PointType, PointType],
        tuple[PointType, PointType],
    ]:
        """
        Return all lines of the bounding box.

        Returns:
            tuple: All lines of the bounding box.

        Examples:
            >>> import simetri.graphics as sg
            >>> bb = sg.BoundingBox((0, 0), (40, 80))
            >>> len(bb.all_lines)
            8
            >>> bb.all_lines
            (((0, 80), (0, 0)), ((0, 0), (40, 0)), ((40, 80), (40, 0)), ((0, 80),
            (40, 80)), ((0.0, 40.0), (40.0, 40.0)), ((20.0, 80.0), (20.0, 0.0)),
            ((0, 0), (40, 80)), ((40, 0), (0, 80)))
        """

        # Do not change the order. LiBeRTy (Left, Bottom, Right, Top) is the order.
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
    def width(self) -> float:
        """
        Return the width of the bounding box.

        Returns:
            float: The width of the bounding box.

        Examples:
            >>> from simetri.geom.bbox import BoundingBox
            >>> BoundingBox((0, 0), (40, 80)).width
            40.0
        """
        return distance(self.northwest, self.northeast)

    @property
    def height(self) -> float:
        """
        Return the height of the bounding box.

        Returns:
            float: The height of the bounding box.

        Examples:
            >>> from simetri.geom.bbox import BoundingBox
            >>> BoundingBox((0, 0), (40, 80)).height
            80.0
        """
        return distance(self.northwest, self.southwest)

    @property
    def size(self) -> tuple[float, float]:
        """
        Return the size of the bounding box.

        Returns:
            tuple: The size of the bounding box.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.BoundingBox((0, 0), (40, 80)).size
            (40.0, 80.0)
        """
        return (self.width, self.height)

    @property
    def west(self) -> PointType:
        """
        Return the left edge midpoint.

        Returns:
            tuple: The left edge midpoint.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.BoundingBox((0, 0), (40, 80)).west
            (0.0, 40.0)
        """
        return midpoint(*self.left)

    @property
    def south(self) -> PointType:
        """
        Return the bottom edge midpoint.

        Returns:
            tuple: The bottom edge midpoint.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.BoundingBox((0, 0), (40, 80)).south
            (20.0, 0.0)
        """
        return midpoint(*self.bottom)

    @property
    def east(self) -> PointType:
        """
        Return the right edge midpoint.

        Returns:
            tuple: The right edge midpoint.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.BoundingBox((0, 0), (40, 80)).east
            (40.0, 40.0)
        """
        return midpoint(*self.right)

    @property
    def north(self) -> PointType:
        """
        Return the top edge midpoint.

        Returns:
            tuple: The top edge midpoint.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.BoundingBox((0, 0), (40, 80)).north
            (20.0, 80.0)
        """
        return midpoint(*self.top)

    @property
    def northwest(self) -> PointType:
        """
        Return the top left corner.

        Returns:
            tuple: The top left corner.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.BoundingBox((0, 0), (40, 80)).northwest
            (0, 80)
        """
        return self.__dict__["northwest"]

    @property
    def northeast(self) -> PointType:
        """
        Return the top right corner.

        Returns:
            tuple: The top right corner.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.BoundingBox((0, 0), (40, 80)).northeast
            (40, 80)
        """
        return self.__dict__["northeast"]

    @property
    def southwest(self) -> PointType:
        """
        Return the bottom left corner.

        Returns:
            tuple: The bottom left corner.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.BoundingBox((0, 0), (40, 80)).southwest
            (0, 0)
        """
        return self.__dict__["southwest"]

    @property
    def southeast(self) -> PointType:
        """
        Return the bottom right corner.

        Returns:
            tuple: The bottom right corner.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.BoundingBox((0, 0), (40, 80)).southeast
            (40, 0)
        """
        return self.__dict__["southeast"]

    @property
    def diagonal1(self) -> tuple[PointType, PointType]:
        """
        Return the first diagonal. From the top left to the bottom right.

        Returns:
            tuple: The first diagonal.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.BoundingBox((0, 0), (40, 80)).diagonal1
            ((0, 0), (40, 80))
        """
        return (self.southwest, self.northeast)

    @property
    def diagonal2(self) -> tuple[PointType, PointType]:
        """
        Return the second diagonal. From the top right to the bottom left.

        Returns:
            tuple: The second diagonal.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.BoundingBox((0, 0), (40, 80)).diagonal2
            ((40, 0), (0, 80))
        """
        return (self.southeast, self.northwest)

    def get_inflated_b_box(
        self,
        left_margin: float | None = None,
        bottom_margin: float | None = None,
        right_margin: float | None = None,
        top_margin: float | None = None,
    ) -> BoundingBox:
        """
        Return a bounding box with offset edges.

        Args:
            left_margin (float, optional): The left margin.
            bottom_margin (float, optional): The bottom margin.
            right_margin (float, optional): The right margin.
            top_margin (float, optional): The top margin.

        Returns:
            BoundingBox: The inflated bounding box.

        Examples:
            >>> import simetri.graphics as sg
            >>> bb = sg.BoundingBox((0, 0), (40, 80))
            >>> inflated = bb.get_inflated_b_box(20, 40, 60, 80)
            >>> inflated.southwest, inflated.northeast
            ((-20, -40), (100, 160))
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

    def offset_line(
        self, side: Side | str, offset: float
    ) -> tuple[PointType, PointType]:
        """
        Offset is applied outwards. Use negative values for inward offset.

        Args:
            side (Side): The side to offset.
            offset (float): The offset distance.

        Returns:
            tuple: The offset line.

        Examples:
            >>> from simetri.base.all_enums import Side
            >>> import simetri.graphics as sg
            >>> bb = sg.BoundingBox((0, 0), (40, 80))
            >>> bb.offset_line(Side.LEFT, 40)
            ((-40, 0), (-40, 80))
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
            res = offset_line(self.horiz_centerline, offset)
        elif side == Side.V_CENTERLINE:
            res = offset_line(self.vert_centerline, offset)
        else:
            raise ValueError(f"Unknown side: {side}")

        return res

    def offset_point(
        self, anchor: Anchor | str | PointType, dx: float, dy: float
    ) -> list[float]:
        """
        Return an offset point from the given corner.

        Args:
            anchor (Anchor): The anchor point.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            list: The offset point.

        Examples:
            >>> from simetri.base.all_enums import Anchor
            >>> import simetri.graphics as sg
            >>> bb = sg.BoundingBox((0, 0), (40, 80))
            >>> bb.offset_point(Anchor.NORTHWEST, 20, -40)
            [20, 40]
        """
        if isinstance(anchor, str):
            anchor = Anchor[anchor.upper()]
            x, y = getattr(self, anchor.value)[:2]
        elif isinstance(anchor, Anchor):
            x, y = getattr(self, anchor.value)[:2]
        else:
            raise TypeError(f"Unknown anchor: {anchor}")
        return [x + dx, y + dy]

    def centered(
        self, item: Shape | Group, dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the center of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.midpoint of the reference item's bounding-box.

        Examples:
            >>> import simetri.graphics as sg
            >>> ref = sg.Shape([(80, 0), (120, 0), (120, 40), (80, 40)])
            >>> tag = sg.BoundingBox((0, 0), (40, 80))
            >>> tag.centered(ref)
            (100.0, 20.0)
        """

        x, y = item.midpoint[:2]
        x += dx
        y += dy
        return x, y

    def left_of(
        self, item: Shape | Group, dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the item.west of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.west of the reference item's bounding-box.

        Examples:
            >>> import simetri.graphics as sg
            >>> ref = sg.Shape([(80, 0), (120, 0), (120, 40), (80, 40)])
            >>> tag = sg.BoundingBox((0, 0), (40, 80))
            >>> tag.left_of(ref)
            (60.0, 20.0)
        """
        x, y = item.west[:2]
        w2 = self.width / 2
        x += dx - w2
        y += dy
        return x, y

    def right_of(
        self, item: Shape | Group, dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the item.east of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.east of the reference item's bounding-box.

        Examples:
            >>> import simetri.graphics as sg
            >>> ref = sg.Shape([(80, 0), (120, 0), (120, 40), (80, 40)])
            >>> tag = sg.BoundingBox((0, 0), (40, 80))
            >>> tag.right_of(ref)
            (140.0, 20.0)
        """
        x, y = item.east[:2]
        w2 = self.width / 2
        x += dx + w2
        y += dy
        return x, y

    def above(
        self, item: Shape | Group, dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the item.north of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.north of the reference item's bounding-box.

        Examples:
            >>> import simetri.graphics as sg
            >>> ref = sg.Shape([(80, 0), (120, 0), (120, 40), (80, 40)])
            >>> tag = sg.BoundingBox((0, 0), (40, 80))
            >>> tag.above(ref)
            (100.0, 80.0)
        """
        x, y = item.north[:2]
        h2 = self.height / 2
        x += dx
        y += dy + h2
        return x, y

    def below(
        self, item: Shape | Group, dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the item.south of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.south of the reference item's bounding-box.

        Examples:
            >>> import simetri.graphics as sg
            >>> ref = sg.Shape([(80, 0), (120, 0), (120, 40), (80, 40)])
            >>> tag = sg.BoundingBox((0, 0), (40, 80))
            >>> tag.below(ref)
            (100.0, -40.0)
        """
        x, y = item.south[:2]
        h2 = self.height / 2
        x += dx
        y += dy - h2
        return x, y

    def above_left(
        self, item: Shape | Group, dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the item.northwest of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.northwest of the reference item's bounding-box.

        Examples:
            >>> import simetri.graphics as sg
            >>> ref = sg.Shape([(80, 0), (120, 0), (120, 40), (80, 40)])
            >>> tag = sg.BoundingBox((0, 0), (40, 80))
            >>> tag.above_left(ref)
            (60.0, 80.0)
        """
        x, y = item.northwest[:]
        w2 = self.width / 2
        h2 = self.height / 2
        x += dx - w2
        y += dy + h2

        return x, y

    def above_right(
        self, item: Shape | Group, dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the item.northeast of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.northeast of the reference item's bounding-box.

        Examples:
            >>> import simetri.graphics as sg
            >>> ref = sg.Shape([(80, 0), (120, 0), (120, 40), (80, 40)])
            >>> tag = sg.BoundingBox((0, 0), (40, 80))
            >>> tag.above_right(ref)
            (140.0, 80.0)
        """
        x, y = item.northeast[:2]
        w2 = self.width / 2
        h2 = self.height / 2
        x += dx + w2
        y += dy + h2

        return x, y

    def below_left(
        self, item: Shape | Group, dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the item.southwest of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.southwest of the reference item's bounding-box.

        Examples:
            >>> import simetri.graphics as sg
            >>> ref = sg.Shape([(80, 0), (120, 0), (120, 40), (80, 40)])
            >>> tag = sg.BoundingBox((0, 0), (40, 80))
            >>> tag.below_left(ref)
            (60.0, -40.0)
        """
        x, y = item.southwest[:2]
        w2 = self.width / 2
        h2 = self.height / 2
        x += dx - w2
        y += dy - h2

        return x, y

    def below_right(
        self, item: Shape | Group, dx: float = 0, dy: float = 0
    ) -> PointType:
        """
        Get the item.southeast of the reference item.

        Args:
            item (object): The reference item. Shape or Group.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            PointType: The item.southeast of the reference item's bounding-box.

        Examples:
            >>> import simetri.graphics as sg
            >>> ref = sg.Shape([(80, 0), (120, 0), (120, 40), (80, 40)])
            >>> tag = sg.BoundingBox((0, 0), (40, 80))
            >>> tag.below_right(ref)
            (140.0, -40.0)
        """
        x, y = item.southeast[:2]
        w2 = self.width / 2
        h2 = self.height / 2
        x += dx + w2
        y += dy - h2

        return x, y

    def polar_pos(
        self, item: Shape | Group, angle: float, radius: float
    ) -> PointType:
        """Return a point at polar offset from ``item.midpoint``.

        Used to place this box's midpoint at that polar position.

        Args:
            item: Reference Shape or Group.
            angle: Angle in radians from the positive x-axis.
            radius: Distance from ``item.midpoint``.

        Returns:
            PointType: Target ``(x, y)`` for this box's midpoint.

        Examples:
            >>> import simetri.graphics as sg
            >>> ref = sg.Shape([(80, 0), (120, 0), (120, 40), (80, 40)])
            >>> tag = sg.BoundingBox((0, 0), (40, 80))
            >>> tag.polar_pos(ref, 0, 20)
            (120.0, 20.0)
        """

        x, y = item.midpoint[:2]

        x1, y1 = polar_to_cartesian(radius, angle)
        x += x1
        y += y1

        return x, y


def bounding_box(points: Sequence[PointType]) -> BoundingBox:
    """Build a ``BoundingBox`` from a sequence of points.

    Args:
        points: Sequence or ndarray of ``(x, y)`` points.

    Returns:
        BoundingBox: Axis-aligned box enclosing the points.

    Raises:
        ValueError: If ``points`` is empty.

    Examples:
        >>> import simetri.graphics as sg
        >>> bb = sg.bounding_box([(0, 0), (80, 40), (24, 64)])
        >>> bb.southwest, bb.northeast
        ((0, 0), (80, 64))
    """
    if isinstance(points, np.ndarray):
        points = points[:, :2]
    else:
        points = np.array(points)  # numpy array of points
    n_points = len(points)
    BB_EPSILON = runtime_defaults["BB_EPSILON"]
    if n_points == 0:  # empty list of points
        raise ValueError("Empty list of points")

    if len(points.shape) == 1:
        # single point
        min_x, min_y = points
        max_x = min_x + BB_EPSILON
        max_y = min_y + BB_EPSILON
    else:
        # find minimum and maximum coordinates
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        if min_x == max_x:  # this could be a vertical line or degenerate points
            max_x += BB_EPSILON
        if (
            min_y == max_y
        ):  # this could be a horizontal line or degenerate points
            max_y += BB_EPSILON
    # bounding box corners
    bottom_left = (min_x, min_y)
    top_right = (max_x, max_y)
    return BoundingBox(southwest=bottom_left, northeast=top_right)

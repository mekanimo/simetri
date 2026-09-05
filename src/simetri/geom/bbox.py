"""Axis-aligned bounding boxes for Shape and Group objects.

Provides reference anchors (corners, mid-sides, diagonals) and placement
helpers such as ``left_of``, ``above``, and ``polar_pos``.

Examples:
    >>> import simetri.graphics as sg
    >>> box = sg.BoundingBox((0, 0), (100, 50))
    >>> box.width, box.height
    (100, 50)
    >>> box.midpoint
    (50.0, 25.0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .geometry import polar_to_cartesian, positive_angle
from .points.point_utils import distance, midpoint
from .segments.line_utils import intersect, line_angle, offset_line
from ..config.settings import issue_warning
from ..base.all_enums import Anchor, Side, Types
from ..base.common import VOID, PointType, defaults, get_unique_id

if TYPE_CHECKING:
    from ..group.batch import Group
    from .shape import Shape


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
        >>> bb = sg.BoundingBox((0, 0), (10, 20))
        >>> bb.northwest
        (0, 20)
    """

    def __init__(
        self, southwest: PointType = None, northeast: PointType = None
    ):
        """Initialize a BoundingBox from opposite corners.

        Args:
            southwest: Southwest (min-x, min-y) corner.
            northeast: Northeast (max-x, max-y) corner.
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
        self._aliases = {
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

        self.type = Types.BOUNDING_BOX
        self.subtype = Types.BOUNDING_BOX
        self.visible = True
        self.exclusive = [
            "line_color",
            "line_width",
            "line_dash_array",
            "stroke",
            "fill",
        ]

        self.id = get_unique_id(self)

    def __getattr__(self, name):
        """
        Get the attribute with the given name.

        Args:
            name (str): The name of the attribute.

        Returns:
            Any: The attribute with the given name.
        """
        if name in self._aliases:
            if name == "center":
                issue_warning(
                    '"center" is deprecated use "midpoint" instead.',
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
        """
        angle = positive_angle(angle)
        line = ((0, 0), (np.cos(angle), np.sin(angle)))

        angle1 = line_angle(self.midpoint, self.northeast)
        angle2 = -angle1  # midpoint, southeast
        angle3 = np.pi - angle1  # midpoint, northwest
        angle4 = -angle3  # midpoint, southwest
        if angle3 >= angle >= angle1:
            res = intersect(line, self.top)
        elif angle4 <= angle <= angle2:
            res = intersect(line, self.bottom)
        elif angle1 <= angle <= angle2:
            res = intersect(line, self.right)
        else:
            res = intersect(line, self.left)

        return res

    @property
    def left(self):
        """
        Return the left edge.

        Returns:
            tuple: The left edge.
        """
        return (self.northwest, self.southwest)

    @property
    def right(self):
        """
        Return the right edge.

        Returns:
            tuple: The right edge.
        """
        return (self.northeast, self.southeast)

    @property
    def top(self):
        """
        Return the top edge.

        Returns:
            tuple: The top edge.
        """
        return (self.northwest, self.northeast)

    @property
    def bottom(self):
        """
        Return the bottom edge.

        Returns:
            tuple: The bottom edge.
        """
        return (self.southwest, self.southeast)

    @property
    def vert_centerline(self):
        """
        Return the vertical centerline.

        Returns:
            tuple: The vertical centerline.
        """
        return (self.north, self.south)

    @property
    def horiz_centerline(self):
        """
        Return the horizontal centerline.

        Returns:
            tuple: The horizontal centerline.
        """
        return (self.west, self.east)

    @property
    def midpoint(self):
        """
        Return the center of the bounding box.

        Returns:
            tuple: The center of the bounding box.
        """
        x1, y1 = self.southwest
        x2, y2 = self.northeast

        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2

        return (xc, yc)

    @property
    def corners(self):
        """
        Return the four corners of the bounding box.

        Returns:
            tuple: The four corners of the bounding box.
        """
        return (self.northwest, self.southwest, self.southeast, self.northeast)

    @property
    def diamond(self):
        """
        Return the four center points of the bounding box in a diamond shape.

        Returns:
            tuple: The four center points of the bounding box in a diamond shape.
        """
        return (self.north, self.west, self.south, self.east)

    @property
    def all_anchors(self):
        """
        Return all anchors of the bounding box.

        Returns:
            tuple: All anchors of the bounding box.
        """
        return (
            self.west,
            self.southwest,
            self.south,
            self.northeast,
            self.east,
            self.northeast,
            self.north,
            self.northwest,
            self.midpoint,
        )

    @property
    def all_lines(self):
        """
        Return all lines of the bounding box.

        Returns:
            tuple: All lines of the bounding box.
        """

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
    def width(self):
        """
        Return the width of the bounding box.

        Returns:
            float: The width of the bounding box.
        """
        return distance(self.northwest, self.northeast)

    @property
    def height(self):
        """
        Return the height of the bounding box.

        Returns:
            float: The height of the bounding box.
        """
        return distance(self.northwest, self.southwest)

    @property
    def size(self):
        """
        Return the size of the bounding box.

        Returns:
            tuple: The size of the bounding box.
        """
        return (self.width, self.height)

    @property
    def west(self):
        """
        Return the left edge midpoint.

        Returns:
            tuple: The left edge midpoint.
        """
        return midpoint(*self.left)

    @property
    def south(self):
        """
        Return the bottom edge midpoint.

        Returns:
            tuple: The bottom edge midpoint.
        """
        return midpoint(*self.bottom)

    @property
    def east(self):
        """
        Return the right edge midpoint.

        Returns:
            tuple: The right edge midpoint.
        """
        return midpoint(*self.right)

    @property
    def north(self):
        """
        Return the top edge midpoint.

        Returns:
            tuple: The top edge midpoint.
        """
        return midpoint(*self.top)

    @property
    def northwest(self):
        """
        Return the top left corner.

        Returns:
            tuple: The top left corner.
        """
        return self.__dict__["northwest"]

    @property
    def northeast(self):
        """
        Return the top right corner.

        Returns:
            tuple: The top right corner.
        """
        return self.__dict__["northeast"]

    @property
    def southwest(self):
        """
        Return the bottom left corner.

        Returns:
            tuple: The bottom left corner.
        """
        return self.__dict__["southwest"]

    @property
    def southeast(self):
        """
        Return the bottom right corner.

        Returns:
            tuple: The bottom right corner.
        """
        return self.__dict__["southeast"]

    @property
    def diagonal1(self):
        """
        Return the first diagonal. From the top left to the bottom right.

        Returns:
            tuple: The first diagonal.
        """
        return (self.southwest, self.northeast)

    @property
    def diagonal2(self):
        """
        Return the second diagonal. From the top right to the bottom left.

        Returns:
            tuple: The second diagonal.
        """
        return (self.southeast, self.northwest)

    def get_inflated_b_box(
        self,
        left_margin=None,
        bottom_margin=None,
        right_margin=None,
        top_margin=None,
    ):
        """
        Return a bounding box with offset edges.

        Args:
            left_margin (float, optional): The left margin.
            bottom_margin (float, optional): The bottom margin.
            right_margin (float, optional): The right margin.
            top_margin (float, optional): The top margin.

        Returns:
            BoundingBox: The inflated bounding box.
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
        Return an offset point from the given corner.

        Args:
            anchor (Anchor): The anchor point.
            dx (float): The x offset.
            dy (float): The y offset.

        Returns:
            list: The offset point.
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
        """

        x, y = item.midpoint[:2]

        x1, y1 = polar_to_cartesian(radius, angle)
        x += x1
        y += y1

        return x, y


def bounding_box(points):
    """Build a ``BoundingBox`` from a sequence of points.

    Args:
        points: Sequence or ndarray of ``(x, y)`` points.

    Returns:
        BoundingBox: Axis-aligned box enclosing the points.

    Raises:
        ValueError: If ``points`` is empty.

    Examples:
        >>> import simetri.graphics as sg
        >>> bb = sg.bounding_box([(0, 0), (10, 5), (3, 8)])
        >>> bb.southwest, bb.northeast  # doctest: +SKIP
        ((0, 0), (10, 8))
    """
    if isinstance(points, np.ndarray):
        points = points[:, :2]
    else:
        points = np.array(points)  # numpy array of points
    n_points = len(points)
    BB_EPSILON = defaults["BB_EPSILON"]
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

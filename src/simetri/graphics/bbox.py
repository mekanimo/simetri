import logging

import numpy as np

from .common import Point, common_properties, _set_Nones, defaults
from .all_enums import Side, Types, Anchor
from ..helpers.geometry import distance, mid_point, offset_line


class BoundingBox:
    """Rectangular bounding box.
    If the object is a Shape, it contains all points.
    If the object is a Batch, it contains all points of all Shapes.

    Provides reference edges and points as shown in the Book page ???.
    """

    def __init__(self, south_west: Point, north_east: Point):
        # define the four corners
        self.__dict__["south_west"] = south_west
        self.__dict__["north_east"] = north_east
        self.__dict__["north_west"] = (south_west[0], north_east[1])
        self.__dict__["south_east"] = (north_east[0], south_west[1])
        self._aliases = {
            "s": "south",
            "n": "north",
            "w": "west",
            "e": "east",
            "sw": "south_west",
            "se": "south_east",
            "nw": "north_west",
            "ne": "north_east",
            "d1": "diagonal1",
            "d2": "diagonal2",
            "c": "center",
            "vcl": "vert_centerline",
            "hcl": "horiz_centerline",
        }
        common_properties(self)

    def __getattr__(self, name):
        if name in self._aliases:
            res = getattr(self, self._aliases[name])
        else:
            res = self.__dict__[name]
        return res

    @property
    def type(self):
        """Return the type of the object."""
        return Types.BOUNDINGBOX

    @property
    def left(self):
        """Return the left edge."""
        return (self.north_west, self.south_west)

    @property
    def right(self):
        """Return the right edge."""
        return (self.north_east, self.south_east)

    @property
    def top(self):
        """Return the top edge."""
        return (self.north_west, self.north_east)

    @property
    def bottom(self):
        """Return the bottom edge."""
        return (self.south_west, self.south_east)

    @property
    def vert_centerline(self):
        """Return the vertical centerline."""
        return (self.north, self.south)

    @property
    def horiz_centerline(self):
        """Return the horizontal centerline."""
        return (self.west, self.east)

    @property
    def center(self):
        """Return the center of the bounding box."""
        x1, y1 = self.south_west
        x2, y2 = self.north_east

        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        return (xc, yc)

    @property
    def corners(self):
        """Return the four corners of the bounding box."""
        return (self.north_west, self.south_west, self.south_east, self.north_east)

    @property
    def all_anchors(self):
        """Return all the anchors of the bounding box."""
        return (
            self.north_west,
            self.west,
            self.south_west,
            self.south,
            self.north_east,
            self.east,
            self.north_east,
            self.north,
            self.center,
        )

    @property
    def width(self):
        """Return the width of the bounding box."""
        return distance(self.north_west, self.north_east)

    @property
    def height(self):
        """Return the height of the bounding box."""
        return distance(self.north_west, self.south_west)

    @property
    def size(self):
        """Return the size of the bounding box."""
        return (self.width, self.height)

    @property
    def west(self):
        """Return the left edge midpoint."""
        return mid_point(*self.left)

    @property
    def south(self):
        """Return the bottom edge midpoint."""
        return mid_point(*self.bottom)

    @property
    def east(self):
        """Return the right edge midpoint."""
        return mid_point(*self.right)

    @property
    def north(self):
        """Return the top edge midpoint."""
        return mid_point(*self.top)

    @property
    def north_west(self):
        """Return the top left corner."""
        return self.__dict__["north_west"]

    @property
    def north_east(self):
        """Return the top right corner."""
        return self.__dict__["north_east"]

    @property
    def south_west(self):
        """Return the bottom left corner."""
        return self.__dict__["south_west"]

    @property
    def south_east(self):
        """Return the bottom right corner."""
        return self.__dict__["south_east"]

    @property
    def diagonal1(self):
        """Return the first diagonal.From the top left to the bottom right."""
        return (self.south_west, self.north_east)

    @property
    def diagonal2(self):
        """Return the second diagonal.From the top right to the bottom left."""
        return (self.south_east, self.north_west)

    def get_inflated_b_box(
        self, left_margin=None, bottom_margin=None, right_margin=None, top_margin=None
    ):
        """Return a bounding box with offset edges."""
        _set_Nones(
            self,
            ["left_margin", "bottom_margin", "right_margin", "top_margin"],
            [left_margin, bottom_margin, right_margin, top_margin],
        )

        x, y = self.south_west
        south_west = (x - left_margin, y - bottom_margin)

        x, y = self.north_east
        north_east = (x + right_margin, y + top_margin)

        return BoundingBox(south_west, north_east)

    def offset_line(self, side, offset):
        """Offset is applied outwards. Use negative values for inward
        offset.
        """
        if isinstance(side, str):
            side = Side[side.upper()]

        if side == Side.RIGHT:
            x1, y1 = self.south_east
            x2, y2 = self.north_east
            res = ((x1 + offset, y1), (x2 + offset, y2))
        elif side == Side.LEFT:
            x1, y1 = self.south_west
            x2, y2 = self.north_west
            res = ((x1 - offset, y1), (x2 - offset, y2))
        elif side == Side.TOP:
            x1, y1 = self.north_west
            x2, y2 = self.north_east
            res = ((x1, y1 + offset), (x2, y2 + offset))
        elif side == Side.BOTTOM:
            x1, y1 = self.south_west
            x2, y2 = self.south_east
            res = ((x1, y1 - offset), (x2, y2 - offset))
        elif side == Side.DIAGONAL1:
            res = offset_line(self.diagonal1, offset)
        elif side == Side.DIAGONAL2:
            res = offset_line(self.diagonal2, offset)
        elif side == Side.HCENTER:
            res = offset_line(self.horiz_center_line, offset)
        elif side == Side.VCENTER:
            res = offset_line(self.vert_center_line, offset)
        else:
            logging.error("Unknown side: %s", side)
            res = None

        return res

    def offset_point(self, anchor, dx, dy):
        """Return an offset point from the given corner."""
        if isinstance(anchor, str):
            anchor = Anchor[anchor.upper()]
            x, y = getattr(self, anchor.value)
        else:
            x, y = anchor
        return [x + dx, y + dy]


def bounding_box(points):
    """Given a list of (x, y) points return the corresponding
    BoundingBox object."""
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
        if min_y == max_y:  # this could be a horizontal line or degenerate points
            max_y += BB_EPSILON
    # bounding box corners
    bottom_left = (min_x, min_y)
    top_right = (max_x, max_y)
    return BoundingBox(south_west=bottom_left, north_east=top_right)

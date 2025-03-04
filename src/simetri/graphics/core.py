"""Base class. This is the parent for Shape and Batch classes."""

__all__ = [
    "Base",
]


from typing import Sequence, Any, Union
from typing_extensions import Self

import numpy as np
from numpy import ndarray

from .all_enums import Anchor, Side, get_enum_value
from .common import (
    Point,
    Line,
)
from .affine import (
    translation_matrix,
    rotation_matrix,
    mirror_matrix,
    glide_matrix,
    scale_in_place_matrix,
    shear_matrix,
)
from ..geometry.geometry import line_angle


class Base:
    """Base class for Shape and Batch objects."""

    def __getattr__(self, name: str) -> Any:
        _anchors = [
            "southeast",
            "southwest",
            "northeast",
            "northwest",
            "south",
            "north",
            "east",
            "west",
            "center",
            "left",
            "right",
            "top",
            "bottom",
            "diagonal1",
            "diagonal2",
            "horiz_centerline",
            "vert_centerline",
            "s",
            "n",
            "e",
            "w",
            "sw",
            "se",
            "nw",
            "ne",
            "c",
            "d1",
            "d",
            "corners",
            "all_anchors",
            "width",
            "height",
        ]

        if name in _anchors:
            res = getattr(self.b_box, name)
        else:
            try:
                res = self.__dict__[name]
            except KeyError as exc:
                msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
                raise AttributeError(msg) from exc

        return res

    def translate(self, dx: float = 0, dy: float = 0, reps: int = 0) -> Self:
        """Translates the object by dx and dy."""
        transform = translation_matrix(dx, dy)
        return self._update(transform, reps=reps)

    def translate_along(
        self,
        path: Sequence[Point],
        step: int = 1,
        align_tangent: bool = False,
        scale: float = 1, # scale factor
        rotate: float = 0, # angle in radians
    ) -> Self:
        """Translates the object along the given curve.
        Every n-th point is used to calculate the translation vector.
        If align_tangent is True, the object is rotated to align with the tangent at each point.
        scale is the scale factor applied at each point.
        rotate is the angle in radians applied at each point.
        """
        x, y = path[0][:2]
        self.move_to((x, y))
        dup = self.copy()
        if align_tangent:
            tangent = line_angle(path[-1], path[0])
            self.rotate(tangent, about=path[0], reps=0)
        dup2 = dup.copy()
        for i, point in enumerate(path[1::step]):
            dup2 = dup2.copy()
            px, py = point[:2]
            dup2.move_to((px, py))
            if scale != 1:
                dup2.scale(scale, about=point)
            if rotate != 0:
                dup2.rotate(rotate, about=point)
            self.append(dup2)
            if align_tangent:
                tangent = line_angle(path[i - 1], path[i])
                dup2.rotate(tangent, about=point, reps=0)
            # scale *= scale
            # rotate += rotate
        return self

    def rotate(self, angle: float, about: Point = (0, 0), reps: int = 0) -> Self:
        """Rotates the object by the given angle (in radians) about the given point."""
        transform = rotation_matrix(angle, about)
        return self._update(transform, reps=reps)

    def mirror(self, about: Union[Line, Point], reps: int = 0) -> Self:
        """Mirrors the object about the given line or point."""
        transform = mirror_matrix(about)

        return self._update(transform, reps=reps)

    def glide(self, glide_line: Line, glide_dist: float, reps: int = 0) -> Self:
        """Glides (first mirror then translate) the object along the given line
        by the given glide_dist."""
        transform = glide_matrix(glide_line, glide_dist)

        return self._update(transform, reps=reps)

    def scale(
        self,
        scale_x: float,
        scale_y: Union[float, None] = None,
        about: Point = (0, 0),
        reps: int = 0,
    ) -> Self:
        """Scales the object by the given scale factors about the given point."""
        if scale_y is None:
            scale_y = scale_x
        transform = scale_in_place_matrix(scale_x, scale_y, about)

        return self._update(transform, reps=reps)

    def shear(self, theta_x: float, theta_y: float, reps: int = 0) -> Self:
        """Shears the object by the given angles."""
        transform = shear_matrix(theta_x, theta_y)
        return self._update(transform, reps=reps)

    def reset_xform_matrix(self) -> Self:
        """Resets the transformation matrix to the identity matrix."""
        self.__dict__["xform_matrix"] = np.identity(3)

        return self

    def transform(self, xform_matrix: ndarray, reps: int = 0) -> Self:
        """Transforms the object by the given transformation matrix."""
        return self._update(xform_matrix, reps=reps)

    def move_to(self, pos: Point, anchor: Anchor = Anchor.CENTER) -> Self:
        """Moves the object to the given position by using its center point."""
        x, y = pos[:2]
        anchor = get_enum_value(Anchor, anchor)
        x1, y1 = getattr(self.b_box, anchor)
        transform = translation_matrix(x - x1, y - y1)
        return self._update(transform, reps=0)

    def offset_line(self, side: Side, offset: float) -> Line:
        """side can be Side.LEFT, Side.RIGHT, Side.TOP, or Side.BOTTOM.
        offset is applied outwards."""
        side = get_enum_value(Side, side)
        return self.b_box.offset_line(side, offset)

    def offset_point(self, anchor: Anchor, dx: float, dy: float = 0) -> Point:
        """anchor can be Anchor.CENTER, Anchor.SOUTHWEST, Anchor.SOUTHEAST,
        Anchor.NORTHWEST, Anchor.NORTHEAST, Anchor.SOUTH, Anchor.WEST,
        Anchor.EAST, or Anchor.NORTH."""
        anchor = get_enum_value(Anchor, anchor)
        return self.b_box.offset_point(anchor, dx, dy)

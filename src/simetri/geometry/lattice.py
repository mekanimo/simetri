"""2D Lattice"""

from math import pi, cos, sin, ceil, floor
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from ..graphics.all_enums import IsometryType, LatType, Types, LatRef
from ..graphics.shape import Shape
from ..graphics.batch import Batch
from ..graphics.shapes import reg_poly_shape
from ..graphics.common import PointType
from ..geometry.geometry import (
    intersect,
    lerp_point,
    is_number,
    distance,
    clip_line_to_rect,
)
from ..geometry.vectors import Vector
from ..colors.colors import gray, green, red, navy, blue, yellow

r = 8
triangle = reg_poly_shape(3, r, angle=-pi / 6, color=navy).scale(.6)
hexagon = reg_poly_shape(6, r, angle=pi / 6, fill_color=blue).scale(.6)
diamond = Shape([(-5, 0), (0, 9), (5, 0), (0, -9)], closed=True, fill_color=red).scale(.6)
square = reg_poly_shape(4, r, angle=-pi / 4, fill_color=green).scale(.6)


def basis_to_cart(a_, b_, u, v):
    ux, uy = u
    vx, vy = v
    x = a_ * ux + b_ * vx
    y = a_ * uy + b_ * vy
    return x, y


def cart_to_basis(x, y, u, v):
    ux, uy = u
    vx, vy = v
    det = ux * vy - uy * vx
    if abs(det) < 1e-12:
        raise ValueError("Basis vectors are linearly dependent.")

    a = (x * vy - y * vx) / det
    b = (ux * y - uy * x) / det
    return a, b


def all_axes():
    """
    Returns all unique axes using a corner-crossing check.
    A segment is valid if it crosses a corner OR both endpoints are corners.
    """
    corners = {0, 4, 8, 12}

    return [
        (i, j)
        for i in range(16)
        for j in range(i + 1, 16)
        if (
            # 1. Is there a corner between them?
            # (Excluding the wrap-around side where i=0 and j > 12)
            (any(i < c < j for c in [4, 8, 12]) and not (i == 0 and j > 12))
            # 2. OR are both endpoints corners? (Covers the 4 main sides)
            or ({i, j} <= corners)
        )
    ]


@dataclass
class Isometry:
    """Isometries are transformations that preserve:
        - distances
        - angles
        - shape
        - size

    There are four different types:
        - Translation
        - Rotation
        - Reflection
        - Glide Reflection

    """

    subtype: IsometryType
    reference: Any = None
    quantifier: Any = None
    reps: int = 1
    take: slice = None

    def __post_init__(self):
        self.type = Types.ISOMETRY


class Lattice:
    """A 2D latttice that can have five different types.
    'PAR', 'RECT', 'RHOMB', 'SQR', and 'HEX'
    """

    def __init__(
        self, subtype: LatType = LatType.HEX, a=40, b=None, theta=None, origin=(0, 0)
    ):
        angles = {
            LatType.HEX: pi / 3,
            LatType.RECT: pi / 2,
            LatType.SQR: pi / 2,
        }
        self.type = Types.LATTICE
        self.subtype = subtype
        self.a = a
        self.b = b

        if subtype in [LatType.RECT, LatType.SQR, LatType.HEX]:
            self.theta = angles[self.subtype]
        else:
            self.theta = theta

        if subtype in [LatType.HEX, LatType.SQR, LatType.RHOMB]:
            self.b = self.a

        self.origin = (0, 0)

        if subtype == LatType.RHOMB:
            angle1 = theta / 2
            angle2 = angle1
        else:
            angle1 = 0
            angle2 = self.theta
        self.ax = self.a * cos(angle1)
        self.ay = -self.a * sin(angle1)
        self.bx = self.b * cos(angle2)
        self.by = self.b * sin(angle2)
        self.u = Vector(self.ax, self.ay)
        self.v = Vector(self.bx, self.by)
        self.origin = origin[:2]

        det = self.ax * self.by - self.ay * self.bx

        if abs(det) < 1e-6:
            raise ValueError(
                "Lattice vectors are linearly dependent (not a 2D lattice)."
            )

        # unit cell
        p1 = (0, 0)
        p2 = self.u[:2]
        p3 = (self.u + self.v)[:2]
        p4 = self.v[:2]

        self.unit = Shape(
            [p1, p2, p3, p4],
            closed=True,
            fill=False,
        )

        self.isometries = []
        self.pattern = None
        self.kernel = None

    def _resolve(self, reference: Any):
        ref, quantifier = reference
        if is_number(ref):
            res = ref
        elif ref == LatRef.COORD:
            # t1, t2 = quantifier
            res = self.basis_to_cartesian(quantifier)
        elif ref == LatRef.LERP:
            i, t = quantifier
            res = lerp_point(*self.unit.edges[i], t)
        elif ref == LatRef.VERTEX:
            res = self.unit[quantifier]
        elif ref == LatRef.EDGE:
            res = self.edges[quantifier]
        elif ref == LatRef.AXIS:
            p1, p2 = quantifier[:2]
            p1 = self._resolve(p1)
            p2 = self._resolve(p2)
            res = (p1, p2)
        elif ref == LatRef.DISTANCE:
            p1, p2 = quantifier
            p1 = self._resolve(p1)
            p2 = self._resolve(p2)
            res = distance(p1, p2)

        return res

    @property
    def center(self):
        orig = Vector(self.origin[:2])

        return (orig + self.u / 2 + self.v / 2).data

    def apply(self, isometry):
        """Applies an isometry using references of the lattice."""
        reference = isometry.reference
        quantifier = isometry.quantifier
        reps = isometry.reps
        take = isometry.take
        if isometry.subtype == IsometryType.TRANSLATION:
            if reference is None:
                dx, dy = quantifier
                dx = self._resolve(dx)
                dy = self._resolve(dy)

            return self.pattern.translate(dx, dy, reps=reps, take=take)
        elif isometry.subtype == IsometryType.ROTATION:
            about = self._resolve(reference)
            return self.pattern.rotate(
                angle=quantifier,
                about=about,
                reps=reps,
                take=take,
            )
        elif isometry.subtype == IsometryType.MIRROR:
            about = self._resolve(reference)

            return self.pattern.mirror(about=about, reps=reps, take=take)
        elif isometry.subtype == IsometryType.GLIDE_REFLECTION:
            about = self._resolve(reference)
            distance = self._resolve(quantifier)
            return self.pattern.glide(
                glide_line=about,
                glide_dist=distance,
                reps=reps,
                take=take,
            )
        elif isometry.subtype == IsometryType.IDENTITY:
            return self.pattern

    def populate_unit(self, kernel):
        if kernel.type == "SHAPE":
            self.pattern = Batch(kernel)
        elif kernel.type == "BATCH":
            self.pattern = kernel
        else:
            raise ValueError("kernel needs to be a Shape or Batch object!")

        for isom in self.isometries:
            self.apply(isom)

        return self

    def expand(self, kernel, reps: int = 1) -> Batch:
        self.populate_unit(kernel)
        pattern = self.pattern
        subtype = self.subtype
        reps1 = reps // 2
        reps2 = reps
        if subtype in [LatType.HEX, LatType.PAR]:
            dx1 = 0
            dy1 = 2 * self.by
            dx2 = self.a
            pattern.translate(dx=self.bx, dy=self.by, reps=1)
            pattern.translate(dx=dx1, dy=dy1, reps=reps1).translate(
                dx=dx2, reps=reps2
            )

        elif subtype in [LatType.SQR, LatType.RECT]:
            pattern.translate(self.a, reps=reps).translate(0, self.b, reps=reps)

        elif subtype == LatType.RHOMB:
            width = self.unit.width
            height = self.unit.height
            pattern.translate(dx=self.bx, dy=self.by, reps=1).translate(
                dx=0, dy=height, reps=reps2
            ).translate(dx=width, reps=reps2)

        return self

    def cell_structure(self) -> Batch:
        """Returns the cell structure of the lattice."""
        pass

    def cartesian_to_basis(self, point: PointType) -> PointType:
        x, y = point
        return cart_to_basis(x, y, self.u, self.v)

    def basis_to_cartesian(self, point: PointType) -> PointType:
        a, b = point
        return basis_to_cart(a, b, self.u, self.v)

    def clipped_points(
        self, lower_left: PointType, upper_right: PointType
    ) -> List:
        """Returns the lattice points within the given rectangle by two points."""
        # Define the 4 corners of the rectangle
        x_min, y_min = lower_left[:2]
        x_max, y_max = upper_right[:2]
        ox, oy = self.origin[:2]

        corners = [
            (x_min, y_min),
            (x_max, y_min),
            (x_min, y_max),
            (x_max, y_max),
        ]

        # Transform corners to (i, j) space
        # Formula: [i, j]^T = M^-1 * ([x, y]^T - [ox, oy]^T)
        i_vals = []
        j_vals = []
        ux, uy = self.u
        vx, vy = self.v
        det = ux * vy - uy * vx
        for px, py in corners:
            dx, dy = px - ox, py - oy
            # Inverse matrix multiplication
            i = (dx * vy - dy * vx) / det
            j = (-dx * uy + dy * ux) / det
            i_vals.append(i)
            j_vals.append(j)

        # Determine integer bounds for i and j
        i_start = floor(min(i_vals))
        i_end = ceil(max(i_vals))
        j_start = floor(min(j_vals))
        j_end = ceil(max(j_vals))

        # 5. Collect points and perform a final check
        # (The i,j range covers the parallelogram bounding the rectangle)
        points_inside = []
        for i in range(i_start, i_end + 1):
            for j in range(j_start, j_end + 1):
                px = ox + i * ux + j * vx
                py = oy + i * uy + j * vy

                # Final check against the axis-aligned rectangle
                if x_min <= px <= x_max and y_min <= py <= y_max:
                    points_inside.append((px, py))

        return points_inside

    def clipped_lines(
        self, lower_left: PointType, upper_right: PointType
    ) -> List:
        """Returns the lattice lines within the given rectangle by two points."""
        # Define the 4 corners of the rectangle
        x_min, y_min = lower_left[:2]
        x_max, y_max = upper_right[:2]
        origin = self.origin
        ux, uy = self.u
        vx, vy = self.v
        corners = np.array(
            [[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]]
        )

        # Solve for lattice coordinates (c_u, c_v) for each corner
        # Corner = Origin + c_u*u + c_v*v
        # [ux vx] [c_u] = [Corner_x - Origin_x]
        # [uy vy] [c_v]   [Corner_y - Origin_y]
        matrix = np.array([[ux, vx], [uy, vy]])
        inv_matrix = np.linalg.inv(matrix)

        diffs = corners - np.array(origin)
        coords = diffs @ inv_matrix.T

        # Get range of indices
        min_u, max_u = np.floor(coords[:, 0].min()), np.ceil(coords[:, 0].max())
        min_v, max_v = np.floor(coords[:, 1].min()), np.ceil(coords[:, 1].max())

        clipped_lines = []

        # Lines in direction v (indexed by u)
        u = self.u
        v = self.v
        for i in range(int(min_u), int(max_u) + 1):
            point = np.array(origin) + i * np.array(u)
            line = clip_line_to_rect(point, v, lower_left, upper_right)
            if line:
                clipped_lines.append(line)

        # Lines in direction u (indexed by v)
        for j in range(int(min_v), int(max_v) + 1):
            point = np.array(origin) + j * np.array(v)
            line = clip_line_to_rect(point, u, lower_left, upper_right)
            if line:
                clipped_lines.append(line)

        return clipped_lines


def draw_unit(canvas, lat, group, **kwargs):
    triangle = reg_poly_shape(3, r, angle=-pi / 6, color=navy).scale(.6)
    hexagon = reg_poly_shape(6, r, angle=pi / 6, fill_color=blue).scale(.6)
    diamond = Shape(
        [(-5, 0), (0, 9), (5, 0), (0, -9)], closed=True, fill_color=red
    ).scale(.6)
    points = []
    for x in (0, 0.25, 0.5, 0.75, 1):
        points.append((x, 0))
    for y in (0.25, 0.5, 0.75, 1):
        points.append((1, y))
    for x in [0.75, 0.5, 0.25, 0]:
        points.append((x, 1))
    for y in (0.75, 0.5, 0.25):
        points.append((0, y))

    p = points

    hexes = []
    diamonds = []
    triangles = []
    squares = []
    mirrors = []
    glides = []
    hairlines = []

    center = (0.5, 0.5)
    piv1 = (1 / 3, 1 / 3)
    piv2 = (2 / 3, 2 / 3)

    if group in ["p6", "p6m"]:
        hexes = [p[0], p[4], p[8], p[12]]
        diamonds = [p[2], p[6], p[10], p[14], center]
        triangles = [piv1, piv2]

    if group == "p6m":
        mirrors = [
            [p[0], p[8]],
            [p[4], p[12]],
            [p[2], p[12]],
            [p[4], p[10]],
            [p[14], p[4]],
            [p[12], p[6]],
        ]
        glides = [
            [p[1], p[14]],
            [p[2], p[1]],
            [p[2], p[14]],
            [p[2], p[6]],
            [p[2], p[10]],
            [p[2], p[15]],
            [p[3], p[11]],
            [p[5], p[13]],
            [p[6], p[14]],
            [p[6], p[10]],
            [p[6], p[9]],
            [p[7], p[10]],
            [p[10], p[14]],
        ]

    elif group == "p6":
        hairlines = [
            (p[0], piv1),
            (p[4], piv1),
            (p[12], piv1),
            (p[4], piv2),
            (p[8], piv2),
            (p[12], piv2),
            (p[4], p[12]),
        ]

    elif group == "pmm":
        mirrors = [
            (p[0], p[4]),
            (p[4], p[8]),
            (p[8], p[12]),
            (p[12], p[0]),
            (p[2], p[10]),
            (p[6], p[14]),
        ]

        diamonds = [p[0], p[4], p[8], p[12], center]

    elif group == "pm":
        mirrors = [
            (p[0], p[4]),
            (p[8], p[12]),
            (p[6], p[14]),
        ]

    elif group == "p1":
        pass

    elif group == "p31m":
        mirrors = [
            (p[0], p[4]),
            (p[4], p[8]),
            (p[8], p[12]),
            (p[12], p[0]),
            (p[4], p[12]),
        ]

        glides = [
            [p[2], p[14]],
            [p[2], p[10]],
            [p[6], p[14]],
            [p[6], p[10]],
        ]

        hairlines = [
            (p[0], piv1),
            (p[4], piv1),
            (p[12], piv1),
            (p[4], piv2),
            (p[8], piv2),
            (p[12], piv2),
        ]

        triangles = [p[0], p[4], p[8], p[12], piv1, piv2]

    elif group == "p3m1":
        mirrors = [
            [p[0], p[8]],
            [p[2], p[12]],
            [p[4], p[10]],
            [p[14], p[4]],
            [p[12], p[6]],
        ]
        glides = [
            [p[1], p[14]],
            [p[2], p[1]],
            [p[2], p[6]],
            [p[2], p[15]],
            [p[3], p[11]],
            [p[5], p[13]],
            [p[6], p[14]],
            [p[6], p[9]],
            [p[7], p[10]],
            [p[10], p[14]],
        ]
        triangles = [p[0], p[4], p[8], p[12], piv1, piv2]

    elif group == "p3":
        triangles = [p[0], p[4], p[8], p[12], piv1, piv2]
        hairlines = [
            (p[0], piv1),
            (p[4], piv1),
            (p[12], piv1),
            (p[4], piv2),
            (p[8], piv2),
            (p[12], piv2),
        ]

    elif group == "p4g":
        mirrors = [
            (p[2], p[6]),
            (p[6], p[10]),
            (p[10], p[14]),
            (p[14], p[2]),
        ]
        glides = [
            [p[0], p[8]],
            [p[4], p[12]],
            [p[1], p[11]],
            [p[3], p[9]],
            [p[5], p[15]],
            [p[7], p[13]],
        ]

        hairlines = [
            (p[2], p[10]),
            (p[6], p[14]),
        ]

        diamonds = [p[2], p[6], p[10], p[14]]
        squares = [p[0], p[4], p[8], p[12], center]

    elif group == "p4m":
        mirrors = [
            [p[0], p[4]],
            [p[4], p[8]],
            [p[8], p[12]],
            [p[12], p[0]],
            [p[0], p[8]],
            [p[4], p[12]],
        ]
        glides = [
            [p[2], p[6]],
            [p[6], p[10]],
            [p[10], p[14]],
            [p[14], p[2]],
        ]

        diamonds = [p[2], p[6], p[10], p[14]]
        squares = [p[0], p[4], p[8], p[12], center]

    elif group == "p4":
        hairlines = [
            (p[2], p[10]),
            (p[6], p[14]),
        ]

        diamonds = [p[2], p[6], p[10], p[14]]
        squares = [p[0], p[4], p[8], p[12], center]

    elif group == "cmm":
        mirrors = [
            [p[0], p[8]],
            [p[4], p[12]],
        ]
        glides = [
            [p[2], p[6]],
            [p[6], p[10]],
            [p[10], p[14]],
            [p[14], p[2]],
        ]

        diamonds = [p[2], p[6], p[10], p[14]]
        squares = [p[0], p[4], p[8], p[12], center]

    elif group == "cm":
        mirrors = [
            [p[0], p[8]],
        ]
        glides = [
            [p[2], p[6]],
            [p[10], p[14]],
        ]

    elif group == "p2":
        diamonds = [p[0], p[4], p[8], p[12], center, p[2], p[6], p[10], p[14]]

    elif group == "pg":
        glides = [
            [p[0], p[4]],
            [p[6], p[14]],
            [p[8], p[12]],
        ]

    elif group == "pmg":
        glides = [
            [p[2], p[10]],
        ]
        mirrors = [
            [p[0], p[4]],
            [p[6], p[14]],
            [p[8], p[12]],
        ]
        diamonds = [p[15], (0.5, 0.25), p[5], p[13], (0.5, 0.75), p[7]]

    elif group == "pgg":
        glides = [
            [p[1], p[11]],
            [p[3], p[9]],
            [p[5], p[15]],
            [p[7], p[13]],
        ]

        diamonds = [p[0], p[4], p[8], p[12], center, p[2], p[6], p[10], p[14]]

    canvas.draw(lat.unit, **kwargs)

    for line in hairlines:
        p1, p2 = line
        p1 = lat._resolve((LatRef.COORD, p1))
        p2 = lat._resolve((LatRef.COORD, p2))

        canvas.line(p1, p2, line_width=0.5, line_color=gray)

    for line in mirrors:
        p1, p2 = line
        p1 = lat._resolve((LatRef.COORD, p1))
        p2 = lat._resolve((LatRef.COORD, p2))

        canvas.line(
            p1,
            p2,
            draw_double=True,
            double_distance=3,
            double_color=yellow,
            line_color=red,
        )

    for line in glides:
        p1, p2 = line
        p1 = lat._resolve((LatRef.COORD, p1))
        p2 = lat._resolve((LatRef.COORD, p2))

        canvas.line(
            p1,
            p2,
            line_dash_array=[8, 3, 3, 3],
            line_width=2.5,
            line_color=blue,
        )

    for hex in hexes:
        pos = lat._resolve((LatRef.COORD, hex))
        canvas.draw(hexagon, pos=pos)

    for diam in diamonds:
        pos = lat._resolve((LatRef.COORD, diam))
        canvas.draw(diamond, pos=pos)

    for tri in triangles:
        pos = lat._resolve((LatRef.COORD, tri))
        canvas.draw(triangle, pos=pos)

    for sqr in squares:
        pos = lat._resolve((LatRef.COORD, sqr))
        canvas.draw(square, pos=pos)


def lattice_p6(a: float) -> Lattice:
    lat = Lattice(LatType.HEX, a=a)

    isom1 = Isometry(
        IsometryType.ROTATION,
        (LatRef.COORD, (1 / 3, 1 / 3)),
        2 * pi / 3,
        reps=2,
    )
    isom2 = Isometry(
        IsometryType.ROTATION, (LatRef.COORD, (0.5, 0.5)), pi, reps=1
    )
    lat.isometries = [isom1, isom2]

    return lat


def lattice_p6m(a: float = 40) -> Lattice:
    lat = Lattice(LatType.HEX, a=a)

    isom1 = Isometry(
        IsometryType.MIRROR,
        (LatRef.AXIS, ((LatRef.VERTEX, 0), (LatRef.VERTEX, 2))),
        pi,
        reps=1,
    )
    isom2 = Isometry(
        IsometryType.ROTATION,
        (LatRef.COORD, (1 / 3, 1 / 3)),
        2 * pi / 3,
        reps=2,
    )
    isom3 = Isometry(
        IsometryType.ROTATION, (LatRef.COORD, (0.5, 0.5)), pi, reps=1
    )
    lat.isometries = [isom1, isom2, isom3]

    return lat


def lattice_p31m(a: float = 40) -> Lattice:
    lat = Lattice(LatType.HEX, a=a)

    isom1 = Isometry(
        IsometryType.ROTATION,
        (LatRef.COORD, (1 / 3, 1 / 3)),
        2 * pi / 3,
        reps=2,
    )

    isom2 = Isometry(
        IsometryType.MIRROR,
        (LatRef.AXIS, ((LatRef.VERTEX, 1), (LatRef.VERTEX, 3))),
        pi,
        reps=1,
    )

    lat.isometries = [isom1, isom2]

    return lat


def lattice_p3m1(a: float = 40) -> Lattice:
    lat = Lattice(LatType.HEX, a=a)

    isom1 = Isometry(
        IsometryType.MIRROR,
        (LatRef.AXIS, ((LatRef.VERTEX, 0), (LatRef.VERTEX, 2))),
        pi,
        reps=1,
    )
    isom2 = Isometry(
        IsometryType.ROTATION,
        (LatRef.COORD, (1 / 3, 1 / 3)),
        2 * pi / 3,
        reps=2,
    )
    isom3 = Isometry(
        IsometryType.ROTATION,
        (LatRef.COORD, (2 / 3, 2 / 3)),
        2 * pi / 3,
        reps=2,
    )
    lat.isometries = [isom1, isom2, isom3]

    return lat


def lattice_p3(a: float = 40) -> Lattice:
    lat = Lattice(LatType.HEX, a=a)

    isom1 = Isometry(
        IsometryType.ROTATION,
        (LatRef.COORD, (1 / 3, 1 / 3)),
        2 * pi / 3,
        reps=2,
    )
    isom2 = Isometry(
        IsometryType.ROTATION,
        (LatRef.COORD, (2 / 3, 2 / 3)),
        2 * pi / 3,
        reps=2,
    )
    lat.isometries = [isom1, isom2]

    return lat


def lattice_p4(a: float = 40) -> Lattice:
    lat = Lattice(LatType.SQR, a=a)

    isom1 = Isometry(
        IsometryType.ROTATION, (LatRef.COORD, (0.5, 0.5)), pi / 2, reps=3
    )

    lat.isometries = [isom1]

    return lat


def lattice_p4m(a: float = 40) -> Lattice:
    lat = Lattice(LatType.SQR, a=a)

    isom1 = Isometry(
        IsometryType.MIRROR,
        (LatRef.AXIS, ((LatRef.VERTEX, 0), (LatRef.VERTEX, 2))),
        pi,
        reps=1,
    )
    isom2 = Isometry(
        IsometryType.ROTATION, (LatRef.COORD, (0.5, 0.5)), pi / 2, reps=3
    )

    lat.isometries = [isom1, isom2]

    return lat


def lattice_p4g(a: float = 40) -> Lattice:
    lat = Lattice(LatType.SQR, a=a)

    isom1 = Isometry(
        IsometryType.MIRROR,
        (LatRef.AXIS, ((LatRef.LERP, (0, 0.5)), (LatRef.LERP, (3, 0.5)))),
        reps=1,
    )
    isom2 = Isometry(
        IsometryType.ROTATION, (LatRef.COORD, (0.5, 0.5)), pi / 2, reps=3
    )

    lat.isometries = [isom1, isom2]

    return lat


def lattice_p1(
    a: float = 40, b: float = 40, theta=pi / 2, lat_type=LatType.SQR
) -> Lattice:
    lat = Lattice(lat_type, a=a, b=a, theta=theta)
    unit = lat.unit
    p1 = unit.edge_midpoint(0)
    p2 = unit.edge_midpoint(3)
    isom1 = Isometry(
        IsometryType.IDENTITY,
        reps=0,
    )

    lat.isometries = [isom1]

    return lat


def lattice_pm(a: float = 40, lat_type=LatType.SQR) -> Lattice:
    lat = Lattice(lat_type, a=a)

    if lat_type in [LatType.SQR, LatType.RECT, LatType.PAR]:
        axis = (LatRef.AXIS, ((LatRef.LERP, (1, 0.5)), (LatRef.LERP, (3, 0.5))))

    isom1 = Isometry(IsometryType.MIRROR, axis, reps=1)

    lat.isometries = [isom1]

    return lat


def lattice_pmm(a: float = 40, lat_type=LatType.SQR) -> Lattice:
    lat = Lattice(lat_type, a=a)

    axis1 = (LatRef.AXIS, ((LatRef.COORD, (0.5, 0)), (LatRef.COORD, (0.5, 1))))
    isom1 = Isometry(IsometryType.MIRROR, axis1, reps=1)

    axis2 = (LatRef.AXIS, ((LatRef.COORD, (1, 0.5)), (LatRef.COORD, (0, 0.5))))
    isom2 = Isometry(IsometryType.MIRROR, axis2, reps=1)

    lat.isometries = [isom1, isom2]

    return lat


def lattice_p2(
    a: float = 40, b: float = None, theta: float = None, lat_type=LatType.SQR
) -> Lattice:
    lat = Lattice(lat_type, a=a, b=b, theta=theta)

    isom = Isometry(
        IsometryType.ROTATION, (LatRef.COORD, (0.5, 0.5)), pi, reps=1
    )

    lat.isometries = [isom]

    return lat


def lattice_pg(
    a: float = 40, b: float = 20, glide_dist=10, lat_type=LatType.RECT
) -> Lattice:
    lat = Lattice(lat_type, a=a, b=b)

    axis = (LatRef.AXIS, ((LatRef.LERP, (3, 0.5)), (LatRef.LERP, (1, 0.5))))

    isom1 = Isometry(
        IsometryType.GLIDE_REFLECTION,
        axis,
        quantifier=(glide_dist, None),
        reps=1,
    )

    lat.isometries = [isom1]

    return lat


def lattice_pmg(a: float = 40, b: float = 30, lat_type=LatType.RECT) -> Lattice:
    lat = Lattice(lat_type, a=a, b=b)

    isom1 = Isometry(
        IsometryType.ROTATION, (LatRef.COORD, (0.5, 0.25)), pi, reps=1
    )

    axis = (LatRef.AXIS, ((LatRef.LERP, (1, 0.5)), (LatRef.LERP, (3, 0.5))))

    isom2 = Isometry(IsometryType.MIRROR, axis, reps=1)

    lat.isometries = [isom1, isom2]

    return lat


def lattice_pgg(a: float = 40, b: float = 30, lat_type=LatType.RECT) -> Lattice:
    lat = Lattice(lat_type, a=a, b=b)

    axis1 = (
        LatRef.AXIS,
        ((LatRef.COORD, (0, 0.25)), (LatRef.COORD, (1, 0.25))),
    )
    isom1 = Isometry(
        IsometryType.GLIDE_REFLECTION, axis1, (a / 2, None), reps=1
    )

    pivot = (LatRef.COORD, (0.5, 0.5))
    isom2 = Isometry(IsometryType.ROTATION, pivot, pi, reps=1)

    lat.isometries = [isom1, isom2]

    return lat


def lattice_cm(
    a: float = 100, theta: float = 2 * pi / 5, glide_dist: float = None
) -> Lattice:
    """Rhombic lattice with a glide_reflection and a mirror.
    theta cannot be 60 degrees or 90 degrees.
    """
    lat_type = LatType.RHOMB
    lat = Lattice(lat_type, a=a, b=a, theta=theta)
    if glide_dist is None:
        glide_dist = a

    axis1 = (LatRef.AXIS, ((LatRef.COORD, (0, 0.5)), (LatRef.COORD, (0.5, 1))))
    isom1 = Isometry(
        IsometryType.GLIDE_REFLECTION, axis1, (glide_dist, None), reps=1
    )

    axis2 = (LatRef.AXIS, ((LatRef.VERTEX, 0), (LatRef.VERTEX, 2)))

    isom2 = Isometry(IsometryType.MIRROR, axis2, reps=1)

    lat.isometries = [isom1, isom2]

    return lat


def lattice_cmm(a: float = 100, theta: float = 2 * pi / 5) -> Lattice:
    """Rhombic lattice with cross-mirrors. If theta is 90 degrees then it is a
    square lattice. Theta cannot be 60 degrees."""
    lat_type = LatType.RHOMB
    lat = Lattice(lat_type, a=a, b=a, theta=theta)

    axis1 = (LatRef.AXIS, ((LatRef.VERTEX, 0), (LatRef.VERTEX, 2)))
    isom1 = Isometry(IsometryType.MIRROR, axis1, reps=1)

    axis2 = (LatRef.AXIS, ((LatRef.VERTEX, 1), (LatRef.VERTEX, 3)))
    isom2 = Isometry(IsometryType.MIRROR, axis2, reps=1)

    lat.isometries = [isom1, isom2]

    return lat

"""2D lattices and wallpaper-group isometry helpers.

Provides ``Lattice`` / ``Isometry`` plus factory functions for the
seventeen wallpaper groups (``lattice_p1``, ``lattice_p6m``, …). Motifs are
typically ``Shape`` / ``Group`` instances from ``simetri.graphics``.

Examples:
    ::

        import simetri.graphics as sg

        lat = sg.lattice_p4(a=40)
        motif = sg.Circle((10, 10), 5)
        lat.populate_unit(motif)
"""

from dataclasses import dataclass
from math import ceil, cos, floor, pi, sin
from typing import Any

import numpy as np

from ..colors.colors import blue, gray, green, light_gold, navy, purple, red
from ..geometry.geom_utils import distance, is_number
from ..geometry.geometry import clip_line_to_rect, lerp_point
from ..geometry.vectors import Vector
from ..graphics.all_enums import IsometryType, LatRef, LatType, Types
from ..graphics.batch import Group
from ..graphics.common import PointType
from ..graphics.shape import Shape
from ..graphics.shapes import Line, reg_poly_shape

r = 8
triangle = reg_poly_shape(3, r, angle=-pi / 6, color=navy).scale(0.6)
hexagon = reg_poly_shape(6, r, angle=pi / 6, fill_color=blue).scale(0.6)
diamond = Shape(
    [(-5, 0), (0, 9), (5, 0), (0, -9)], closed=True, fill_color=purple
).scale(0.75)
square = reg_poly_shape(4, r, angle=-pi / 4, fill_color=green).scale(0.6)


def basis_to_cart(a_, b_, u, v):
    """Convert lattice basis coordinates ``(a_, b_)`` to Cartesian ``(x, y)``.

    Args:
        a_: Coefficient along basis vector ``u``.
        b_: Coefficient along basis vector ``v``.
        u: First basis vector ``(ux, uy)``.
        v: Second basis vector ``(vx, vy)``.

    Returns:
        tuple: Cartesian point ``(x, y)``.
    """
    ux, uy = u
    vx, vy = v
    x = a_ * ux + b_ * vx
    y = a_ * uy + b_ * vy
    return x, y


def cart_to_basis(x, y, u, v):
    """Convert Cartesian ``(x, y)`` to lattice basis coordinates.

    Args:
        x: Cartesian x-coordinate.
        y: Cartesian y-coordinate.
        u: First basis vector ``(ux, uy)``.
        v: Second basis vector ``(vx, vy)``.

    Returns:
        tuple: Basis coordinates ``(a, b)``.

    Raises:
        ValueError: If ``u`` and ``v`` are linearly dependent.
    """
    ux, uy = u
    vx, vy = v
    det = ux * vy - uy * vx
    if abs(det) < 1e-12:
        raise ValueError("Basis vectors are linearly dependent.")

    a = (x * vy - y * vx) / det
    b = (ux * y - uy * x) / det
    return a, b


def all_axes():
    """Return index pairs of unique lattice axes.

    A segment is kept if it crosses a corner or both endpoints are corners.

    Returns:
        list[tuple[int, int]]: Pairs of axis endpoint indices in ``0..15``.
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
    """Distance-preserving transformation used to fill a lattice unit cell.

    Supported subtypes: translation, rotation, reflection, and glide
    reflection (plus identity).

    Attributes:
        subtype: ``IsometryType`` kind of transformation.
        reference: Lattice reference locating the transform (axis, point, …).
        quantifier: Extra parameter (angle, distance, translation components).
        reps: How many times to repeat the isometry when applied.
        take: Optional slice selecting which copies to keep.
    """

    subtype: IsometryType
    reference: Any = None
    quantifier: Any = None
    reps: int = 1
    take: slice = None

    def __post_init__(self):
        self.type = Types.ISOMETRY


class Lattice:
    """A 2D lattice of type PAR, RECT, RHOMB, SQR, or HEX.

    Attributes:
        subtype: Lattice type (``LatType``).
        a: First lattice spacing.
        b: Second lattice spacing (equals ``a`` for HEX/SQR/RHOMB).
        theta: Angle between basis vectors.
        origin: Lattice origin in Cartesian coordinates.
        u: First basis vector.
        v: Second basis vector.
        unit: Unit-cell parallelogram as a ``Shape``.
        isometries: List of ``Isometry`` objects for the wallpaper group.
        pattern: Motif ``Group`` after ``populate_unit``.
    """

    def __init__(
        self,
        subtype: LatType = LatType.HEX,
        a=40,
        b=None,
        theta=None,
        origin=(0, 0),
    ):
        """Create a 2D lattice.

        Args:
            subtype: Lattice type. Defaults to ``LatType.HEX``.
            a: Primary spacing. Defaults to 40.
            b: Secondary spacing (RECT/PAR). Defaults to None.
            theta: Angle between basis vectors when not implied by type.
            origin: Cartesian origin. Defaults to ``(0, 0)``.

        Raises:
            ValueError: If the basis vectors are linearly dependent.
        """
        angles = {
            LatType.HEX: pi / 3,
            LatType.RECT: pi / 2,
            LatType.SQR: pi / 2,
        }
        self.type = Types.LATTICE
        self.subtype = subtype
        self.a = a
        self.b = b

        if subtype in (LatType.RECT, LatType.SQR, LatType.HEX):
            self.theta = angles[self.subtype]
        else:
            self.theta = theta

        if subtype in (LatType.HEX, LatType.SQR, LatType.RHOMB):
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

        self.unit_cell = None

        self.isometries = []
        self.pattern = None
        self.kernel = None

    def _resolve(self, reference: Any):
        """Resolve a lattice reference to a numeric/geometric value."""
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
        """Return the Cartesian center of the unit cell.

        Returns:
            list: Center coordinates ``[x, y]``.
        """
        orig = Vector(self.origin[:2])

        return (orig + self.u / 2 + self.v / 2).data

    def apply(self, isometry):
        """Apply an isometry to ``self.pattern`` using lattice references.

        Args:
            isometry: ``Isometry`` describing translation, rotation, mirror,
                glide, or identity.

        Returns:
            Group | Shape | None: Transformed pattern (or ``self.pattern`` for
            identity).
        """
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
        """Set ``pattern`` from ``kernel`` and apply configured isometries.

        Args:
            kernel: Motif ``Shape`` or ``Group`` to place in the unit cell.

        Returns:
            Lattice: ``self``, for chaining.

        Raises:
            ValueError: If ``kernel`` is not a Shape or Group.
        """
        if kernel.subtype == "LINPATH" or kernel.type == "SHAPE":
            self.pattern = Group(kernel)
        elif kernel.type == "BATCH" or kernel.type == "GROUP":
            self.pattern = kernel

        else:
            raise ValueError("kernel needs to be a Shape or Group object!")

        for isom in self.isometries:
            self.apply(isom)

        return self

    def span(self, kernel, horizontal=True, reps: int = 1) -> Group:
        """Populate the unit and translate the pattern along one axis.

        Args:
            kernel: Motif ``Shape`` or ``Group``.
            horizontal: If True, span along ``a``; otherwise along ``b``.
            reps: Number of translations. Defaults to 1.

        Returns:
            Group: Updated pattern (also stored on ``self.pattern``).
        """
        self.populate_unit(kernel)
        pattern = self.pattern
        if horizontal:
            dx = self.a
            self.pattern = pattern.translate(dx, 0, reps=reps)
        else:
            dy = self.b
            self.pattern = pattern.translate(0, dy, reps=reps)

    def expand(self, kernel, reps: int = 1) -> Group:
        """Populate the unit and expand the pattern by repeated translations.

        Args:
            kernel: Motif ``Shape`` or ``Group``.
            reps: Expansion repetitions. Defaults to 1.

        Returns:
            Lattice: ``self``, for chaining.
        """
        self.populate_unit(kernel)
        pattern = self.pattern
        subtype = self.subtype
        reps1 = reps // 2
        reps2 = reps
        if subtype in (LatType.HEX, LatType.PAR):
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

    def cell_structure(self) -> Group:
        """Return the cell structure of the lattice.

        Note:
            Currently unimplemented (returns None via ``pass``).

        Returns:
            Group: Cell structure visualization (when implemented).
        """
        pass

    def cartesian_to_basis(self, point: PointType) -> PointType:
        """Convert a Cartesian point to this lattice's basis coordinates.

        Args:
            point: Cartesian point ``(x, y)``.

        Returns:
            PointType: Basis coordinates ``(a, b)``.
        """
        x, y = point
        return cart_to_basis(x, y, self.u, self.v)

    def basis_to_cartesian(self, point: PointType) -> PointType:
        """Convert lattice-basis coordinates to Cartesian.

        Args:
            point: Basis coordinates ``(a, b)``.

        Returns:
            PointType: Cartesian point ``(x, y)``.
        """
        a, b = point
        return basis_to_cart(a, b, self.u, self.v)

    def clipped_points(
        self, lower_left: PointType, upper_right: PointType
    ) -> list:
        """Return lattice points that lie inside the given axis-aligned rectangle.

        Args:
            lower_left: Lower-left corner of the clip rectangle.
            upper_right: Upper-right corner of the clip rectangle.

        Returns:
            list: Lattice points inside the rectangle.
        """
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
    ) -> list:
        """Return lattice lines clipped to the given axis-aligned rectangle.

        Args:
            lower_left: Lower-left corner of the clip rectangle.
            upper_right: Upper-right corner of the clip rectangle.

        Returns:
            list: Line segments inside the rectangle.
        """
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


def get_unit(lat, group, vertical=False, **kwargs):
    """Build the fundamental-domain motif for a lattice wallpaper group.

    Args:
        lat: ``Lattice`` whose unit cell anchors the motif.
        group: Wallpaper group name (e.g. ``\"p4\"``) or a motif group.
        vertical: Orientation flag for some groups. Defaults to False.
        **kwargs: Extra style/options passed through to motif construction.

    Returns:
        Group | Shape: Unit motif for the group.
    """
    triangle = reg_poly_shape(3, r, angle=-pi / 6, color=navy).scale(0.6)
    hexagon = reg_poly_shape(6, r, angle=pi / 6, fill_color=blue).scale(0.6)
    diamond = Shape(
        [(-5, 0), (0, 9), (5, 0), (0, -9)], closed=True, fill_color=purple
    ).scale(0.75)
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

    unit = Group()

    if group in ("p6", "p6m"):
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

        fund_domain = [p[0], p[2], piv1]

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

        fund_domain = [p[0], p[4], piv1]

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

        fund_domain = [p[0], p[2], center, p[14]]

    elif group == "pm":
        if vertical:
            mirrors = [
                (p[0], p[12]),
                (p[4], p[8]),
                (p[2], p[10]),
            ]
        else:
            mirrors = [
                (p[0], p[4]),
                (p[8], p[12]),
                (p[6], p[14]),
            ]

        fund_domain = [p[0], p[4], p[6], p[14]]

    elif group == "p1":
        fund_domain = [p[0], p[4], p[8], p[12]]

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
        fund_domain = [p[0], p[4], piv1]

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
        fund_domain = [piv1, piv2, p[12]]

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
        fund_domain = [piv1, p[4], piv2, p[12]]

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

        fund_domain = [p[0], p[2], p[14]]

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

        fund_domain = [p[0], p[2], center]

        diamonds = [p[2], p[6], p[10], p[14]]
        squares = [p[0], p[4], p[8], p[12], center]

    elif group == "p4":
        hairlines = [
            (p[2], p[10]),
            (p[6], p[14]),
        ]

        fund_domain = [p[0], p[2], center, p[14]]

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
        fund_domain = [p[0], p[4], center]

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

        fund_domain = [p[0], p[4], p[8]]

    elif group == "p2":
        diamonds = [p[0], p[4], p[8], p[12], center, p[2], p[6], p[10], p[14]]

        fund_domain = [p[0], p[4], p[6], p[14]]

    elif group == "pg":
        glides = [
            [p[0], p[4]],
            [p[6], p[14]],
            [p[8], p[12]],
        ]

        fund_domain = [p[0], p[2], p[10], p[12]]

    elif group == "pmg":
        glides = [
            [p[2], p[10]],
        ]
        mirrors = [
            [p[0], p[4]],
            [p[6], p[14]],
            [p[8], p[12]],
        ]
        fund_domain = [p[0], p[2], center, p[14]]

        diamonds = [p[15], (0.5, 0.25), p[5], p[13], (0.5, 0.75), p[7]]

    elif group == "pgg":
        glides = [
            [p[1], p[11]],
            [p[3], p[9]],
            [p[5], p[15]],
            [p[7], p[13]],
        ]
        fund_domain = [p[2], p[6], p[14]]

        diamonds = [p[0], p[4], p[8], p[12], center, p[2], p[6], p[10], p[14]]

    points = [lat._resolve((LatRef.COORD, p)) for p in fund_domain]
    # canvas.draw(
    #     Shape(points, closed=True), color=light_gold, fill=True
    # )
    unit.append(Shape(points, closed=True, color=light_gold, fill=True))

    # canvas.draw(lat.unit)
    unit.append(lat.unit)

    for line in hairlines:
        p1, p2 = line
        p1 = lat._resolve((LatRef.COORD, p1))
        p2 = lat._resolve((LatRef.COORD, p2))

        # canvas.line(p1, p2, line_width=0.5, line_color=gray)
        unit.append(Line(p1, p2, line_width=0.5, line_color=gray))

    for line in mirrors:
        p1, p2 = line
        p1 = lat._resolve((LatRef.COORD, p1))
        p2 = lat._resolve((LatRef.COORD, p2))
        double_distance = 2.5 * kwargs.get("u_scale", 1)
        # canvas.line(
        #     p1,
        #     p2,
        #     draw_double=True,
        #     double_distance=double_distance,
        #     double_color=red,
        #     **kwargs
        # )
        unit.append(
            Line(
                p1,
                p2,
                draw_double=True,
                double_distance=double_distance,
                double_color=red,
            )
        )

    for line in glides:
        p1, p2 = line
        p1 = lat._resolve((LatRef.COORD, p1))
        p2 = lat._resolve((LatRef.COORD, p2))
        line_width = 2.5 * kwargs.get("u_scale", 1)
        # canvas.line(
        #     p1,
        #     p2,
        #     line_dash_array=[8, 3, 3, 3],
        #     line_width=line_width,
        #     line_color=blue,
        #     **kwargs
        # )
        unit.append(
            Line(
                p1,
                p2,
                line_dash_array=[8, 3, 3, 3],
                line_width=line_width,
                line_color=blue,
            )
        )

    sf = kwargs.get("u_scale", 1)
    hexagon.scale(sf)
    for hex in hexes:
        pos_ = lat._resolve((LatRef.COORD, hex))

        # canvas.draw(hexagon, pos=pos)
        unit.append(hexagon.copy().translate(*pos_))

    diamond.scale(sf)
    for diam in diamonds:
        pos_ = lat._resolve((LatRef.COORD, diam))

        # canvas.draw(diamond, pos=pos)
        unit.append(diamond.copy().translate(*pos_))

    triangle.scale(sf)
    for tri in triangles:
        pos_ = lat._resolve((LatRef.COORD, tri))

        # canvas.draw(triangle, pos=pos)
        unit.append(triangle.copy().translate(*pos_))

    square.scale(sf)
    for sqr in squares:
        pos_ = lat._resolve((LatRef.COORD, sqr))

        # canvas.draw(square, pos=pos)
        unit.append(square.copy().translate(*pos_))

    return unit


def draw_unit(canvas, lat, group, vertical=False, **kwargs):
    """Draw the lattice unit motif on ``canvas``.

    Args:
        canvas: Target canvas.
        lat: ``Lattice`` instance.
        group: Wallpaper group name or motif group.
        vertical: Orientation flag for some groups. Defaults to False.
        **kwargs: Passed to ``get_unit`` and ``canvas.draw``.
    """
    unit = get_unit(lat, group, vertical, **kwargs)
    canvas.draw(unit, **kwargs)


def lattice_p6(a: float) -> Lattice:
    """Return a hexagonal lattice configured for wallpaper group p6.

    Args:
        a: Lattice spacing.

    Returns:
        Lattice: Lattice with p6 isometries and unit cell.
    """
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

    lat.unit_cell = get_unit(lat, "p6")

    return lat


def lattice_p6m(a: float = 40) -> Lattice:
    """Return a hexagonal lattice configured for wallpaper group p6m.

    Args:
        a: Lattice spacing. Defaults to 40.

    Returns:
        Lattice: Lattice with p6m isometries and unit cell.
    """
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

    lat.unit_cell = get_unit(lat, "p6m")

    return lat


def lattice_p31m(a: float = 40) -> Lattice:
    """Return a hexagonal lattice configured for wallpaper group p31m.

    Args:
        a: Lattice spacing. Defaults to 40.

    Returns:
        Lattice: Lattice with p31m isometries and unit cell.
    """
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
    lat.unit_cell = get_unit(lat, "p31m")

    return lat


def lattice_p3m1(a: float = 40) -> Lattice:
    """Return a hexagonal lattice configured for wallpaper group p3m1.

    Args:
        a: Lattice spacing. Defaults to 40.

    Returns:
        Lattice: Lattice with p3m1 isometries and unit cell.
    """
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

    lat.unit_cell = get_unit(lat, "p3m1")

    return lat


def lattice_p3(a: float = 40) -> Lattice:
    """Return a hexagonal lattice configured for wallpaper group p3.

    Args:
        a: Lattice spacing. Defaults to 40.

    Returns:
        Lattice: Lattice with p3 isometries and unit cell.
    """
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

    lat.unit_cell = get_unit(lat, "p3")

    return lat


def lattice_p4(a: float = 40) -> Lattice:
    """Return a square lattice configured for wallpaper group p4.

    Args:
        a: Lattice spacing. Defaults to 40.

    Returns:
        Lattice: Lattice with p4 isometries and unit cell.
    """
    lat = Lattice(LatType.SQR, a=a)

    isom1 = Isometry(
        IsometryType.ROTATION, (LatRef.COORD, (0.5, 0.5)), pi / 2, reps=3
    )

    lat.isometries = [isom1]

    lat.unit_cell = get_unit(lat, "p4")

    return lat


def lattice_p4m(a: float = 40) -> Lattice:
    """Return a square lattice configured for wallpaper group p4m.

    Args:
        a: Lattice spacing. Defaults to 40.

    Returns:
        Lattice: Lattice with p4m isometries and unit cell.
    """
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

    lat.unit_cell = get_unit(lat, "p4m")

    return lat


def lattice_p4g(a: float = 40) -> Lattice:
    """Return a square lattice configured for wallpaper group p4g.

    Args:
        a: Lattice spacing. Defaults to 40.

    Returns:
        Lattice: Lattice with p4g isometries and unit cell.
    """
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

    lat.unit_cell = get_unit(lat, "p4g")

    return lat


def lattice_p1(
    a: float = 40, b: float = 40, theta=pi / 2, lat_type=LatType.PAR
) -> Lattice:
    """Return a lattice configured for wallpaper group p1 (translations only).

    Args:
        a: Primary spacing. Defaults to 40.
        b: Secondary spacing. Defaults to 40.
        theta: Angle between basis vectors. Defaults to ``pi/2``.
        lat_type: Lattice type. Defaults to ``LatType.PAR``.

    Returns:
        Lattice: Lattice with p1 isometries and unit cell.
    """
    lat = Lattice(lat_type, a=a, b=b, theta=theta)
    unit = lat.unit
    # p1 = unit.edge_midpoint(0)
    # p2 = unit.edge_midpoint(3)
    isom1 = Isometry(
        IsometryType.IDENTITY,
        reps=0,
    )

    lat.isometries = [isom1]

    lat.unit_cell = get_unit(lat, "p1")

    return lat


def lattice_pm(
    a: float = 40,
    b: float = None,
    lat_type=LatType.SQR,
    vertical=False,
) -> Lattice:
    """Return a lattice configured for wallpaper group pm (parallel mirrors).

    Args:
        a: Primary spacing. Defaults to 40.
        b: Secondary spacing for rectangular lattices. Defaults to None.
        lat_type: ``LatType.SQR`` or ``LatType.RECT``. Defaults to SQR.
        vertical: If True, orient mirrors vertically; otherwise horizontally.

    Returns:
        Lattice: Lattice with pm isometries and unit cell.

    Raises:
        ValueError: If ``lat_type`` is not SQR or RECT.
    """
    if lat_type == LatType.SQR:
        lat = Lattice(lat_type, a=a)
    elif lat_type == LatType.RECT:
        lat = Lattice(lat_type, a=a, b=b)
    else:
        raise ValueError("Invalid lattice type.")

    if vertical:
        axis = (
            LatRef.AXIS,
            ((LatRef.LERP, (0, 0.5)), (LatRef.LERP, (2, 0.5))),
        )
    else:
        axis = (
            LatRef.AXIS,
            ((LatRef.LERP, (1, 0.5)), (LatRef.LERP, (3, 0.5))),
        )

    isom1 = Isometry(IsometryType.MIRROR, axis, reps=1)

    lat.isometries = [isom1]

    lat.unit_cell = get_unit(lat, "pm")

    return lat


def lattice_pmm(
    a: float = 40, b: float = None, lat_type=LatType.SQR
) -> Lattice:
    """Return a lattice configured for wallpaper group pmm.

    Args:
        a: Primary spacing. Defaults to 40.
        b: If given, forces a rectangular lattice with this secondary spacing.
        lat_type: Lattice type when ``b`` is None. Defaults to SQR.

    Returns:
        Lattice: Lattice with pmm isometries and unit cell.
    """
    if b is not None:
        lat_type = LatType.RECT
        lat = Lattice(lat_type, a=a, b=b)
    else:
        lat = Lattice(lat_type, a=a)

    axis1 = (LatRef.AXIS, ((LatRef.COORD, (0.5, 0)), (LatRef.COORD, (0.5, 1))))
    isom1 = Isometry(IsometryType.MIRROR, axis1, reps=1)

    axis2 = (LatRef.AXIS, ((LatRef.COORD, (1, 0.5)), (LatRef.COORD, (0, 0.5))))
    isom2 = Isometry(IsometryType.MIRROR, axis2, reps=1)

    lat.isometries = [isom1, isom2]

    lat.unit_cell = get_unit(lat, "pmm")

    return lat


def lattice_p2(
    a: float = 40,
    b: float = None,
    theta: float = 2 * pi / 5,
    lat_type=LatType.SQR,
) -> Lattice:
    """Return a lattice configured for wallpaper group p2 (180° rotations).

    Args:
        a: Primary spacing. Defaults to 40.
        b: Secondary spacing; defaults to ``a`` when None.
        theta: Angle for non-square/rect types. Defaults to ``2*pi/5``.
        lat_type: Lattice type. Defaults to SQR.

    Returns:
        Lattice: Lattice with p2 isometries and unit cell.
    """
    if b is None:
        b = a
    lat = Lattice(lat_type, a=a, b=b, theta=theta)

    isom = Isometry(
        IsometryType.ROTATION, (LatRef.COORD, (0.5, 0.5)), pi, reps=1
    )

    lat.isometries = [isom]

    lat.unit_cell = get_unit(lat, "p2")

    return lat


def lattice_pg(
    a: float = 40, b: float = 20, glide_dist=10, lat_type=LatType.RECT
) -> Lattice:
    """Return a lattice configured for wallpaper group pg (glide reflections).

    Args:
        a: Primary spacing. Defaults to 40.
        b: Secondary spacing. Defaults to 20.
        glide_dist: Glide distance along the axis. Defaults to 10.
        lat_type: Lattice type. Defaults to RECT.

    Returns:
        Lattice: Lattice with pg isometries and unit cell.
    """
    lat = Lattice(lat_type, a=a, b=b)

    axis = (LatRef.AXIS, ((LatRef.LERP, (3, 0.5)), (LatRef.LERP, (1, 0.5))))

    isom1 = Isometry(
        IsometryType.GLIDE_REFLECTION,
        axis,
        quantifier=(glide_dist, None),
        reps=1,
    )

    lat.isometries = [isom1]

    lat.unit_cell = get_unit(lat, "pg")

    return lat


def lattice_pmg(a: float = 40, b: float = 30, lat_type=LatType.RECT) -> Lattice:
    """Return a lattice configured for wallpaper group pmg.

    Args:
        a: Primary spacing. Defaults to 40.
        b: Secondary spacing. Defaults to 30.
        lat_type: Lattice type. Defaults to RECT.

    Returns:
        Lattice: Lattice with pmg isometries and unit cell.
    """
    lat = Lattice(lat_type, a=a, b=b)

    isom1 = Isometry(
        IsometryType.ROTATION, (LatRef.COORD, (0.5, 0.25)), pi, reps=1
    )

    axis = (LatRef.AXIS, ((LatRef.LERP, (1, 0.5)), (LatRef.LERP, (3, 0.5))))

    isom2 = Isometry(IsometryType.MIRROR, axis, reps=1)

    lat.isometries = [isom1, isom2]

    lat.unit_cell = get_unit(lat, "pmg")

    return lat


def lattice_pgg(a: float = 40, b: float = 30, lat_type=LatType.RECT) -> Lattice:
    """Return a lattice configured for wallpaper group pgg.

    Args:
        a: Primary spacing. Defaults to 40.
        b: Secondary spacing. Defaults to 30.
        lat_type: Lattice type. Defaults to RECT.

    Returns:
        Lattice: Lattice with pgg isometries and unit cell.
    """
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

    lat.unit_cell = get_unit(lat, "pgg")

    return lat


def lattice_cm(
    a: float = 100, theta: float = 2 * pi / 5, glide_dist: float = None
) -> Lattice:
    """Return a rhombic lattice configured for wallpaper group cm.

    Combines a glide reflection and a mirror. ``theta`` must not be 60° or 90°.

    Args:
        a: Rhombus side length. Defaults to 100.
        theta: Acute angle of the rhombus. Defaults to ``2*pi/5``.
        glide_dist: Glide distance; defaults to ``a`` when None.

    Returns:
        Lattice: Lattice with cm isometries and unit cell.
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

    lat.unit_cell = get_unit(lat, "cm")

    return lat


def lattice_cmm(a: float = 100, theta: float = 2 * pi / 5) -> Lattice:
    """Return a rhombic lattice configured for wallpaper group cmm.

    Cross-mirrors on the diagonals. If ``theta`` is 90°, the lattice is square.
    ``theta`` must not be 60°.

    Args:
        a: Rhombus side length. Defaults to 100.
        theta: Angle of the rhombus. Defaults to ``2*pi/5``.

    Returns:
        Lattice: Lattice with cmm isometries and unit cell.
    """
    lat_type = LatType.RHOMB
    lat = Lattice(lat_type, a=a, b=a, theta=theta)

    axis1 = (LatRef.AXIS, ((LatRef.VERTEX, 0), (LatRef.VERTEX, 2)))
    isom1 = Isometry(IsometryType.MIRROR, axis1, reps=1)

    axis2 = (LatRef.AXIS, ((LatRef.VERTEX, 1), (LatRef.VERTEX, 3)))
    isom2 = Isometry(IsometryType.MIRROR, axis2, reps=1)

    lat.isometries = [isom1, isom2]

    lat.unit_cell = get_unit(lat, "cmm")

    return lat

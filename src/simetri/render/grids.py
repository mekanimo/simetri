"""Provides facilities for working with grids of cells."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import cos, isclose, pi, sin, sqrt

from ..base.all_enums import GridType, Types
from ..base.common import PointType
from ..coloring.colors import gray
from ..geom.geom_utils import reg_poly_points
from ..geom.geometry import (
    cartesian_to_polar,
    polar_to_cartesian,
)
from ..geom.nonlinear.circle import Circle
from ..geom.points.point_utils import distance, lerp_point
from ..geom.segments.line_utils import intersect
from ..group.batch import Group
from ..shapes.shape import Shape

d_grid_types = {
    GridType.CIRCULAR: Types.CIRCULAR_GRID,
    GridType.SQUARE: Types.SQUARE_GRID,
    GridType.HEXAGONAL: Types.HEX_GRID,
    GridType.MIXED: Types.MIXED_GRID,
}


class Grid(Group):
    """A base-class for all grids.

    Examples:
        >>> from simetri.config.settings import set_defaults
        >>> set_defaults()
        >>> from simetri.base.all_enums import GridType
        >>> from simetri.render.grids import Grid
        >>> pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
        >>> grid = Grid(GridType.SQUARE, points=pts, n=4)
        >>> len(grid.points)
        4
    """

    def __init__(
        self,
        grid_type: GridType,
        center: PointType = (0, 0),
        n: int = 9,
        radius: float = 100,
        points: Sequence[PointType] | None = None,
        n_circles: int = 1,
    ) -> None:
        """Initialize a geometric grid of points and connecting lines.

        Args:
            grid_type (GridType): Kind of grid to build.
            center (PointType, optional): Grid center. Defaults to ``(0, 0)``.
            n (int, optional): Number of points. Defaults to 9.
            radius (float, optional): Circumradius. Defaults to 100.
            points (Sequence[PointType], optional): Explicit grid points.
            n_circles (optional): Number of concentric circles for circular grids.

        Raises:
            ValueError: If ``grid_type`` is not recognized.
        """
        if grid_type not in d_grid_types:
            raise ValueError(f"Invalid grid type: {grid_type}.")
        super().__init__(subtype=d_grid_types[grid_type])
        self.center = center
        self.radius = radius
        self.n = n
        self.n_circles = n_circles
        pairs = list(product(points, repeat=2))

        self._points = Shape(points)
        self.append(self._points)

        for i, point in enumerate(self.points):
            next_point = self._points[(i + 1) % len(self._points)]
            self.append(Shape([point, next_point]))

        if grid_type == GridType.SQUARE:
            # Draw only the horizontal and vertical lines in the grid
            width = self.width
            for p1, p2 in pairs:
                x1, y1 = p1[:2]
                x2, y2 = p2[:2]
                cond1 = x1 == x2
                cond2 = y1 == y2
                if cond1 ^ cond2:
                    dist = distance(p1, p2)
                    if isclose(dist, width, rel_tol=0, abs_tol=1e-5):
                        self.append(Shape([p1, p2], line_color=gray))
        else:
            # Undirected chords between non-adjacent points. Consecutive ring
            # edges were already appended above; skip both (i, j) and (j, i).
            grid_points = self.points
            n_points = len(grid_points)
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    step = j - i
                    if step == 1 or step == n_points - 1:
                        continue
                    self.append(
                        Shape(
                            [grid_points[i], grid_points[j]],
                            line_color=gray,
                        )
                    )

    @property
    def points(self) -> list[PointType]:
        """Return the grid vertex points.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.base.all_enums import GridType
            >>> from simetri.render.grids import Grid
            >>> pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
            >>> Grid(GridType.SQUARE, points=pts, n=4).points[0]
            (0.0, 0.0)
        """
        return self._points.vertices

    def intersect(
        self, line1: Sequence[int], line2: Sequence[int]
    ) -> PointType:
        """Return the intersection of two grid chords given by vertex indices.

        Args:
            line1: Two vertex indices ``(ind1, ind2)``.
            line2: Two vertex indices ``(ind3, ind4)``.

        Returns:
            PointType: Intersection of the two lines.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.base.all_enums import GridType
            >>> from simetri.render.grids import Grid
            >>> pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
            >>> g = Grid(GridType.SQUARE, points=pts, n=4)
            >>> g.intersect((0, 2), (1, 3))[0]
            5.0
        """
        ind1, ind2 = line1
        ind3, ind4 = line2
        points = self.points
        line1 = (points[ind1], points[ind2])
        line2 = (points[ind3], points[ind4])

        return intersect(line1, line2)

    def line(self, ind1: int, ind2: int) -> tuple:
        """
        Returns the line connecting the given indices.

        Args:
            ind1 (int): The first index.
            ind2 (int): The second index.

        Returns:
            tuple: The line connecting the two points.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.base.all_enums import GridType
            >>> from simetri.render.grids import Grid
            >>> pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
            >>> g = Grid(GridType.SQUARE, points=pts, n=4)
            >>> g.line(0, 1)[1]
            (10.0, 0.0)
        """
        return (self.points[ind1], self.points[ind2])

    def radial_point(self, radius: float, index: int) -> PointType:
        """Return a point at ``radius`` from center along the ray to vertex ``index``.

        Args:
            radius: Distance from the grid center.
            index: Vertex index defining the ray direction.

        Returns:
            PointType: Cartesian coordinates of the point.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.render.grids import CircularGrid
            >>> g = CircularGrid(n=12, radius=10)
            >>> round(g.radial_point(5, 0)[0], 10)
            5.0
        """
        return polar_to_cartesian(radius, index * (2 * pi / self.n))

    def between(self, ind1: int, ind2: int, t: float = 0.5) -> PointType:
        """
        Returns the point on the line connecting the given indices interpolated
        by using the given t parameter.

        Args:
            ind1 (int): The first index.
            ind2 (int): The second index.
            t (float): The parameter used for interpolation. Default is 0.5.

        Returns:
            PointType: The point on the line connecting the two points.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> from simetri.base.all_enums import GridType
            >>> from simetri.render.grids import Grid
            >>> pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
            >>> g = Grid(GridType.SQUARE, points=pts, n=4)
            >>> g.between(0, 1, 0.5)
            (5.0, 0.0)
        """
        if t < 0 or t > 1:
            raise ValueError("t must be between 0 and 1.")
        if ind1 < 0 or ind1 >= len(self.points):
            raise ValueError(
                f"ind1 must be between 0 and {len(self.points) - 1}."
            )
        if ind1 == ind2:
            raise ValueError("ind1 and ind2 must be different.")

        return lerp_point(self.points[ind1], self.points[ind2], t)


class CircularGrid(Grid):
    """A grid formed by connections of regular polygon points.

    Examples:
        >>> from simetri.config.settings import set_defaults
        >>> set_defaults()
        >>> from simetri.render.grids import CircularGrid
        >>> CircularGrid(n=6, radius=20).n
        6
    """

    def __init__(
        self,
        center: PointType = (0, 0),
        n: int = 12,
        radius: float = 100,
        n_circles: int = 1,
    ) -> None:
        """Initialize a circular grid from a regular ``n``-gon.

        Args:
            center: Grid center.
            n: Number of vertices on the outer polygon.
            radius: Circumradius.
            n_circles: Concentric circle count used when drawing the grid.
        """
        points = reg_poly_points(center, n, radius)
        super().__init__(
            GridType.CIRCULAR, center, n, radius, points, n_circles
        )

        self.append(Circle(radius, center, fill=False))


class HexGrid(Grid):
    """A grid formed by connections of regular polygon points.

    Examples:
        >>> from simetri.config.settings import set_defaults
        >>> set_defaults()
        >>> from simetri.render.grids import HexGrid
        >>> HexGrid(radius=50).n
        6
    """

    def __init__(
        self,
        center: PointType = (0, 0),
        radius: float = 100,
        n_circles: int = 1,
    ) -> None:
        """Initialize a hexagonal grid (regular 6-gon).

        Args:
            center: Grid center.
            radius: Hexagon circumradius.
            n_circles: Concentric circle count used when drawing the grid.
        """
        points = reg_poly_points(center, 6, radius)
        super().__init__(
            GridType.HEXAGONAL, center, 6, radius, points, n_circles
        )


class SquareGrid(Grid):
    """A grid formed by connections of square cells.

    Examples:
        >>> from simetri.config.settings import set_defaults
        >>> set_defaults()
        >>> from simetri.render.grids import SquareGrid
        >>> SquareGrid(n=16, cell_size=25).cell_size
        25
    """

    def __init__(
        self, center: PointType = (0, 0), n: int = 16, cell_size: float = 25
    ) -> None:
        """
        Initializes the grid with the given center, number of rows, number of columns, and cell size.

        Args:
            center (PointType): The center point of the grid.
            n (int): The number of points in the grid. Square of an even integer.
            cell_size (float): The size of each cell in the grid.
        """
        self.cell_size = cell_size
        self.width = cell_size * n / 4
        hs = int(sqrt(n) // 2)  # half size
        c = cell_size
        vals = [c * x for x in range(-hs, hs + 1)]
        coords = list(product(vals, repeat=2))

        def sort_key(coord: PointType) -> float:
            """Polar radius used to sort grid candidate points.

            Examples:
                >>> pass  # doctest: +SKIP
            """
            r, _ = cartesian_to_polar(*coord)
            return r

        def sort_key2(coord: PointType) -> float:
            """Polar angle used to order the selected grid points.

            Examples:
                >>> pass  # doctest: +SKIP
            """
            _, theta = cartesian_to_polar(*coord)
            return theta

        coords.sort(key=sort_key, reverse=True)
        coords = coords[:n]
        coords.sort(key=sort_key2)
        points = coords
        radius = (2 * (cell_size * sqrt(n)) ** 2) ** 0.5
        super().__init__(GridType.SQUARE, center, n, radius, points)


# change of basis conversion


def convert_basis(
    x: float, y: float, basis: tuple[tuple[float, float], tuple[float, float]]
) -> tuple[float, float]:
    """Convert ``(x, y)`` from the standard basis to ``basis``.

    Args:
        x: X coordinate.
        y: Y coordinate.
        basis: Two basis vectors as ``((x0, y0), (x1, y1))``.

    Returns:
        ``(x', y')`` in the new basis.

    Examples:
        >>> from simetri.config.settings import set_defaults
        >>> set_defaults()
        >>> from simetri.render.grids import convert_basis
        >>> convert_basis(1, 0, ((1, 0), (0, 1)))
        (1, 0)
    """
    return basis[0][0] * x + basis[0][1] * y, basis[1][0] * x + basis[1][1] * y


def convert_to_cartesian(
    x: float, y: float, basis: tuple[tuple[float, float], tuple[float, float]]
) -> tuple[float, float]:
    """Convert ``(x, y)`` from ``basis`` to the standard Cartesian basis.

    Args:
        x: X coordinate in ``basis``.
        y: Y coordinate in ``basis``.
        basis: Two basis vectors as ``((x0, y0), (x1, y1))``.

    Returns:
        ``(x', y')`` in Cartesian coordinates.

    Examples:
        >>> from simetri.config.settings import set_defaults
        >>> set_defaults()
        >>> from simetri.render.grids import convert_to_cartesian
        >>> convert_to_cartesian(1, 0, ((1, 0), (0, 1)))
        (1, 0)
    """
    return basis[0][0] * x + basis[1][0] * y, basis[0][1] * x + basis[1][1] * y


def cartesian_to_isometric(x: float, y: float) -> tuple[float, float]:
    """Convert Cartesian ``(x, y)`` to isometric coordinates.

    Args:
        x: Cartesian x.
        y: Cartesian y.

    Returns:
        Isometric ``(x', y')``.

    Examples:
        >>> from simetri.config.settings import set_defaults
        >>> set_defaults()
        >>> from simetri.render.grids import cartesian_to_isometric
        >>> cartesian_to_isometric(1, 0)[0]
        1
    """
    return convert_basis(x, y, ((1, 0), (cos(pi / 3), sin(pi / 3))))


def isometric_to_cartesian(x: float, y: float) -> tuple[float, float]:
    """Convert isometric ``(x, y)`` to Cartesian coordinates.

    Args:
        x: Isometric x.
        y: Isometric y.

    Returns:
        Cartesian ``(x', y')``.

    Examples:
        >>> from simetri.config.settings import set_defaults
        >>> set_defaults()
        >>> from simetri.render.grids import isometric_to_cartesian
        >>> isometric_to_cartesian(1, 0)[0]
        1.0
    """
    return convert_to_cartesian(x, y, ((1, 0), (cos(pi / 3), sin(pi / 3))))

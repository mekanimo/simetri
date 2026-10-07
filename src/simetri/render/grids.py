"""Provides facilities for working with grids of cells."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import cos, isclose, pi, sin, sqrt

from ..base.all_enums import GridType, Types
from ..base.common import PointType
from ..config.settings import runtime_defaults
from ..geom.geom_utils import reg_poly_points
from ..geom.geometry import (
    cartesian_to_polar,
    polar_to_cartesian,
)
from ..geom.nonlinear.circle import Circle
from ..geom.points.point_utils import lerp_point
from ..geom.segments.line_utils import intersect
from ..group.batch import Group
from ..shapes.shape import Shape

d_grid_types = {
    GridType.CIRCULAR: Types.CIRCULAR_GRID,
    GridType.SQUARE: Types.SQUARE_GRID,
    GridType.HEXAGONAL: Types.HEX_GRID,
    GridType.MIXED: Types.MIXED_GRID,
}


def _coords_close(first: float, second: float) -> bool:
    """Return True if two coordinates match within runtime tolerances."""
    return isclose(
        first,
        second,
        rel_tol=runtime_defaults["rel_tol"],
        abs_tol=runtime_defaults["abs_tol"],
    )


def _adjacent_index_pairs(
    cluster: Sequence[tuple[int, PointType]],
) -> list[tuple[int, int]]:
    """Return consecutive undirected index pairs along a sorted collinear cluster."""
    if len(cluster) < 2:
        return []
    pairs: list[tuple[int, int]] = []
    for first_item, second_item in zip(cluster, cluster[1:]):
        first = first_item[0]
        second = second_item[0]
        if first > second:
            first, second = second, first
        pairs.append((first, second))
    return pairs


def _consecutive_pairs_sharing_axis(
    indexed_points: Sequence[tuple[int, PointType]],
    axis: int,
) -> list[tuple[int, int]]:
    """Connect consecutive vertices that share coordinate ``axis`` (0 is x, 1 is y)."""
    if not indexed_points:
        return []
    other = 1 - axis
    ordered = sorted(
        indexed_points,
        key=lambda item: (item[1][axis], item[1][other]),
    )
    pairs: list[tuple[int, int]] = []
    cluster: list[tuple[int, PointType]] = [ordered[0]]
    cluster_key = ordered[0][1][axis]
    for item in ordered[1:]:
        if _coords_close(item[1][axis], cluster_key):
            cluster.append(item)
            continue
        pairs.extend(_adjacent_index_pairs(cluster))
        cluster = [item]
        cluster_key = item[1][axis]
    pairs.extend(_adjacent_index_pairs(cluster))
    return pairs


def _extreme_vertex_index(
    points: Sequence[PointType],
    plus: bool,
    maximize: bool,
) -> int:
    """Return the index of the vertex with extreme ``x+y`` or ``x-y``.

    ``plus`` True uses ``x + y`` (NE / SW). False uses ``x - y``
    (SE / NW).
    """
    first = points[0]
    best_value = (first[0] + first[1]) if plus else (first[0] - first[1])
    best_index = 0
    for index, point in enumerate(points[1:], start=1):
        value = (point[0] + point[1]) if plus else (point[0] - point[1])
        if maximize:
            if value > best_value:
                best_value = value
                best_index = index
        elif value < best_value:
            best_value = value
            best_index = index
    return best_index


def _square_corner_indices(points: Sequence[PointType]) -> list[int]:
    """Return the four geometric corner indices of a square point set, sorted."""
    if len(points) < 4:
        raise ValueError(
            "corners_only requires a square grid with four distinct corners"
        )
    corners = [
        _extreme_vertex_index(points, True, True),
        _extreme_vertex_index(points, False, True),
        _extreme_vertex_index(points, True, False),
        _extreme_vertex_index(points, False, False),
    ]
    unique = sorted(set(corners))
    if len(unique) != 4:
        raise ValueError(
            "corners_only requires four distinct square corners"
        )
    return unique


class Grid(Group):
    """A base-class for all grids.

    ``connections``, ``skip``, ``corners_only``, ``border``,
    ``centerlines``, ``orthogonals``, and ``diagonals`` control what
    ``canvas.draw(grid)`` strokes. ``connections = [3, 5]`` draws chords ``(i, i+3)`` and
    ``(i, i+5)`` for every start index ``i`` (modulo the number of
    vertices). ``skip = 2`` draws ``(i, i+2)``, ``(i, i+4)``, … until
    the index wraps to ``i``. If ``corners_only`` is True, those start
    indices are only the geometric corner vertices (on a ``SquareGrid``
    with index ``0`` on +x, ``n=16``: ``2``, ``6``, ``10``, ``14``).
    On a circular or hexagonal grid every vertex is a corner, so the
    chords are unchanged. ``border``, ``centerlines``, ``orthogonals``,
    and ``diagonals`` are ``False``, ``True`` (default line color and
    width), or a style dict, like ``canvas.draw_bbox``. ``orthogonals``
    defaults to True and strokes adjacent horizontal and vertical
    segments between vertices that share an x- or y-coordinate.
    ``centerlines`` defaults to False and strokes the horizontal and
    vertical midlines of the grid's axis-aligned bounding box.
    ``diagonals`` defaults to False and, on a ``SquareGrid`` only,
    strokes the two main diagonals of the square (opposite geometric
    corners).

    Examples:
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
        self.connections: Sequence[int] | None = None
        self.skip: int | None = None
        self.border: bool | dict[str, object] = False
        self.centerlines: bool | dict[str, object] = False
        self.orthogonals: bool | dict[str, object] = True
        self.diagonals: bool | dict[str, object] = False
        self.corners_only: bool = False

        self._points = Shape(points)
        self.append(self._points)

    @property
    def points(self) -> list[PointType]:
        """Return the grid vertex points.

        Examples:
            >>> from simetri.base.all_enums import GridType
            >>> from simetri.render.grids import Grid
            >>> pts = [(0, 0), (10, 0), (10, 10), (0, 10)]
            >>> Grid(GridType.SQUARE, points=pts, n=4).points[0]
            (0.0, 0.0)
        """
        return self._points.vertices

    def _connection_start_indices(self) -> list[int]:
        """Return vertex indices that start ``connections`` and ``skip`` chords.

        If ``corners_only`` is False, every vertex is a start. If True, a
        square grid uses the four geometric corners; circular and hexagonal
        grids still use every vertex.
        """
        n_vertices = len(self.points)
        if n_vertices == 0:
            return []
        if not self.corners_only:
            return list(range(n_vertices))
        if self.subtype == Types.SQUARE_GRID:
            return _square_corner_indices(self.points)
        return list(range(n_vertices))

    def line_index_pairs(self) -> list[tuple[int, int]]:
        """Return undirected vertex-index pairs for ``connections`` and ``skip``.

        Indices wrap with ``mod n``. A chord is listed once.

        Returns:
            list[tuple[int, int]]: Pairs ``(i, j)`` with ``i < j``.

        Raises:
            TypeError: If ``connections`` is not a sequence of ints.
            ValueError: If ``skip`` is not a positive integer, or if
                ``corners_only`` is True on a square grid without four
                distinct corners.

        Examples:
            >>> import simetri.graphics as sg
            >>> grid = sg.CircularGrid(n=6, radius=10)
            >>> grid.connections = [3]
            >>> grid.line_index_pairs()
            [(0, 3), (1, 4), (2, 5)]
            >>> grid.connections = None
            >>> grid.skip = 2
            >>> grid.line_index_pairs()
            [(0, 2), (0, 4), (1, 3), (1, 5), (2, 4), (3, 5)]
            >>> sq = sg.SquareGrid(n=16, cell_size=25)
            >>> sq.skip = 2
            >>> sq.corners_only = True
            >>> pairs = sq.line_index_pairs()
            >>> (2, 4) in pairs and (2, 6) in pairs and (2, 8) in pairs
            True
            >>> (6, 8) in pairs and (6, 10) in pairs
            True
            >>> (1, 3) in pairs
            False
            >>> (0, 4) in pairs
            False
        """
        points = self.points
        n_vertices = len(points)
        if n_vertices == 0:
            return []
        starts = self._connection_start_indices()
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()

        def add_pair(start: int, end: int) -> None:
            first = start % n_vertices
            second = end % n_vertices
            if first == second:
                return
            if first > second:
                first, second = second, first
            key = (first, second)
            if key in seen:
                return
            seen.add(key)
            pairs.append(key)

        connections = self.connections
        if connections is not None:
            if isinstance(connections, (str, bytes)):
                raise TypeError("connections must be a sequence of integers")
            for offset in connections:
                if type(offset) is not int:
                    raise TypeError("connections must be a sequence of integers")
                for index in starts:
                    add_pair(index, index + offset)

        skip = self.skip
        if skip is not None:
            if type(skip) is not int or skip <= 0:
                raise ValueError("skip must be a positive integer")
            for index in starts:
                offset = skip
                while offset % n_vertices != 0:
                    add_pair(index, index + offset)
                    offset += skip

        return pairs

    def orthogonal_index_pairs(self) -> list[tuple[int, int]]:
        """Return undirected pairs of vertices that share an x or y coordinate.

        Along each vertical line (shared x) and each horizontal line
        (shared y), consecutive vertices are connected. A pair is listed
        once with ``i < j``.

        Returns:
            list[tuple[int, int]]: Adjacent axis-aligned index pairs.

        Examples:
            >>> import simetri.graphics as sg
            >>> circ = sg.CircularGrid(n=6, radius=10)
            >>> circ.orthogonal_index_pairs()
            [(2, 4), (1, 5), (4, 5), (0, 3), (1, 2)]
            >>> sq = sg.SquareGrid(n=16, cell_size=25)
            >>> len(sq.orthogonal_index_pairs())
            22
        """
        indexed_points = list(enumerate(self.points))
        vertical = _consecutive_pairs_sharing_axis(indexed_points, 0)
        horizontal = _consecutive_pairs_sharing_axis(indexed_points, 1)
        seen: set[tuple[int, int]] = set()
        pairs: list[tuple[int, int]] = []
        for pair in vertical + horizontal:
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
        return pairs

    def diagonal_index_pairs(self) -> list[tuple[int, int]]:
        """Return the two main-diagonal vertex pairs of a square grid.

        Only a ``SquareGrid`` has these pairs. Other grid types return
        an empty list. The pairs are opposite geometric corners:
        northeast–southwest (max ``x+y`` to min ``x+y``) and
        southeast–northwest (max ``x-y`` to min ``x-y``).

        Returns:
            list[tuple[int, int]]: At most two pairs ``(i, j)`` with
            ``i < j``.

        Examples:
            >>> import simetri.graphics as sg
            >>> sg.CircularGrid(n=6, radius=10).diagonal_index_pairs()
            []
            >>> sq = sg.SquareGrid(n=16, cell_size=25)
            >>> sq.diagonal_index_pairs()
            [(2, 10), (6, 14)]
        """
        if self.subtype != Types.SQUARE_GRID:
            return []
        points = self.points
        if len(points) < 2:
            return []
        northeast = _extreme_vertex_index(points, True, True)
        southwest = _extreme_vertex_index(points, True, False)
        southeast = _extreme_vertex_index(points, False, True)
        northwest = _extreme_vertex_index(points, False, False)
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for first, second in (
            (northeast, southwest),
            (southeast, northwest),
        ):
            if first == second:
                continue
            if first > second:
                first, second = second, first
            key = (first, second)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
        return pairs

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

    def __repr__(self) -> str:
        """Return a CircularGrid string from this grid's elements.

        Examples:
            >>> import simetri.graphics as sg
            >>> grid = sg.CircularGrid(n=6, radius=20)
            >>> repr(grid).startswith("CircularGrid(")
            True
            >>> str(grid).startswith("Group")
            True
        """
        if self.elements is None or len(self.elements) == 0:
            return "CircularGrid()"
        if len(self.elements) in [1, 2]:
            return f"CircularGrid({self.elements})"
        return f"CircularGrid({self.elements[0]}...{self.elements[-1]})"


class HexGrid(Grid):
    """A grid formed by connections of regular polygon points.

    Examples:
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

    def __repr__(self) -> str:
        """Return a HexGrid string from this grid's elements.

        Examples:
            >>> import simetri.graphics as sg
            >>> grid = sg.HexGrid(radius=50)
            >>> repr(grid).startswith("HexGrid(")
            True
            >>> str(grid).startswith("Group")
            True
        """
        if self.elements is None or len(self.elements) == 0:
            return "HexGrid()"
        if len(self.elements) in [1, 2]:
            return f"HexGrid({self.elements})"
        return f"HexGrid({self.elements[0]}...{self.elements[-1]})"


class SquareGrid(Grid):
    """A grid formed by connections of square cells.

    Examples:
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

    def __repr__(self) -> str:
        """Return a SquareGrid string from this grid's elements.

        Examples:
            >>> import simetri.graphics as sg
            >>> grid = sg.SquareGrid(n=16, cell_size=25)
            >>> repr(grid).startswith("SquareGrid(")
            True
            >>> str(grid).startswith("Group")
            True
        """
        if self.elements is None or len(self.elements) == 0:
            return "SquareGrid()"
        if len(self.elements) in [1, 2]:
            return f"SquareGrid({self.elements})"
        return f"SquareGrid({self.elements[0]}...{self.elements[-1]})"


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
        >>> from simetri.render.grids import isometric_to_cartesian
        >>> isometric_to_cartesian(1, 0)[0]
        1.0
    """
    return convert_to_cartesian(x, y, ((1, 0), (cos(pi / 3), sin(pi / 3))))

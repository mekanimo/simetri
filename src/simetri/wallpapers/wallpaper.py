"""Simetri graphics library's wallpaper patterns.

Note:
    Prefer ``lattice`` for newer lattice-based APIs.
    This module remains for the classic wallpaper group helpers.

Examples:
    >>> import simetri.graphics as sg
    >>> from simetri.wallpapers import wallpaper as wp
    >>> mark = sg.Shape([(0, 0), (40, 0)])
    >>> pattern = wp.wallpaper_p1(mark, (20, 0), (0, 60), reps1=1, reps2=1)
    >>> [[[round(c, 6) for c in p[:2]] for p in s.vertices] for s in pattern]
    [[[0.0, 0.0], [40.0, 0.0]], [[20.0, 0.0], [60.0, 0.0]], [[0.0, 60.0], [40.0, 60.0]], [[20.0, 60.0], [60.0, 60.0]]]
"""

# This is obsolete now! Replaced by lattice.py

from math import cos, pi, sqrt

from ..base.common import LineType, PointType, VecType
from ..geom.geom_utils import midpoint
from ..geom.segments.line_utils import line_through_point_and_angle
from ..group.batch import Group
from ..helpers.illustration import Tag
from ..shapes.shape import Shape

cos60 = cos(pi / 3)
cos30 = cos(pi / 6)


def cover_hex(
    item: Group | Shape | Tag,
    size: float,
    gap: float = 0,
    reps1: int = 2,
    reps2: int = 2,
    flat: bool = True,
) -> Group:
    """
    Covers an area with a hexagonal pattern.

    Args:
        item (Group | Shape | Tag): The item to be repeated.
        size (float): The size of the hexagons.
        gap (float, optional): The gap between hexagons. Defaults to 0.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 2.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 2.
        flat (bool, optional): If True, hexagons are flat-topped. Defaults to True.

    Returns:
        Group: The resulting pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = cover_hex(mark, 10, reps1=1, reps2=1)
        >>> [[[round(c, 6) for c in p[:2]] for p in s.vertices] for s in pattern]
        [[[0.0, 0.0], [40.0, 0.0]], [[15.0, 8.660254], [55.0, 8.660254]], [[30.0, 0.0], [70.0, 0.0]], [[45.0, 8.660254], [85.0, 8.660254]], [[0.0, 17.320508], [40.0, 17.320508]], [[15.0, 25.980762], [55.0, 25.980762]], [[30.0, 17.320508], [70.0, 17.320508]], [[45.0, 25.980762], [85.0, 25.980762]]]
    """
    gap_x = 2 * gap * cos60
    gap_y = gap * cos30
    if flat:
        w = 2 * size
        h = sqrt(3) * size
        dx = 3 * size + (gap_x * 2)
        dy = h + (gap_y * 2)
        item = item.translate((3 * size / 2) + gap_x, (h / 2) + gap_y, reps=1)
    else:
        w = sqrt(3) * size
        h = 2 * size
        dx = w + (gap_x * 2)
        dy = (2 * size) + (h / 2) + (gap_y * 2)
        item = item.translate((w / 2) + gap_x, (3 * h / 4) + gap_y, reps=1)

    item = item.translate(dx, 0, reps=reps1)
    item = item.translate(0, dy, reps=reps2)

    return item


def cover_rhombic(
    item: Group | Shape | Tag, size: float, reps1: int = 2, reps2: int = 2
) -> Group:
    """
    Covers an area with a rhombic pattern.

    Args:
        item (Group | Shape | Tag): The item to be repeated.
        size (float): The size of the rhombuses.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 2.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 2.

    Returns:
        Group: The resulting pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = cover_rhombic(mark, 10, reps1=1, reps2=1)
        >>> [[[round(c, 6) for c in p[:2]] for p in s.vertices] for s in pattern]
        [[[0.0, 0.0], [40.0, 0.0]], [[7.071068, 7.071068], [47.071068, 7.071068]], [[14.142136, 0.0], [54.142136, 0.0]], [[21.213203, 7.071068], [61.213203, 7.071068]], [[0.0, 14.142136], [40.0, 14.142136]], [[7.071068, 21.213203], [47.071068, 21.213203]], [[14.142136, 14.142136], [54.142136, 14.142136]], [[21.213203, 21.213203], [61.213203, 21.213203]]]
    """
    sqrt2 = sqrt(2)
    diag = (sqrt2 / 2) * size
    item = item.translate(diag, diag, reps=1)
    dx = dy = diag * 2
    item = item.translate(dx, 0, reps=reps1)
    item = item.translate(0, dy, reps=reps2)

    return item


def hex_grid_pointy(
    x: float, y: float, size: float, n_rows: int, n_cols: int
) -> Group:
    """
    Creates a hexagonal grid with pointy tops.

    Args:
        x (float): The x-coordinate of the starting point.
        y (float): The y-coordinate of the starting point.
        size (float): The size of the hexagons.
        n_rows (int): Number of rows in the grid.
        n_cols (int): Number of columns in the grid.

    Returns:
        Group: The resulting grid as a Group of Shapes.

    Examples:
        >>> len(hex_grid_pointy(0, 0, 40, 2, 2))
        4
    """
    height = sqrt(3) * size
    width = 2 * size
    edge_length = 2 * size * cos(pi / 6)
    # create the first row by translating a single hexagon in the x direction
    row = Group(Shape([(x, y)])).translate(size, 0, reps=n_cols - 1)
    # create the second row by translating the first row
    two_rows = row.translate(width, height + edge_length, reps=1)
    # create the grid by translating the first and second row in the y direction
    grid = two_rows.translate(0, 2 * size + edge_length, reps=n_rows / 2 - 1)

    return grid


def cover_hex_pointy(
    item: Shape | Group | Tag,
    size: float,
    gap: float = 0,
    reps1: int = 2,
    reps2: int = 2,
) -> Group:
    """
    Covers an area with a hexagonal pattern with pointy tops.

    Args:
        item (Shape | Group | Tag): The item to be repeated.
        size (float): The size of the hexagons.
        gap (float, optional): The gap between hexagons. Defaults to 0.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 2.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 2.

    Returns:
        Group: The resulting pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = cover_hex_pointy(mark, 10, reps1=1, reps2=1)
        >>> [[[round(c, 6) for c in p[:2]] for p in s.vertices] for s in pattern]
        [[[0.0, 0.0], [40.0, 0.0]], [[8.660254, 15.0], [48.660254, 15.0]], [[17.320508, 0.0], [57.320508, 0.0]], [[25.980762, 15.0], [65.980762, 15.0]], [[0.0, 30.0], [40.0, 30.0]], [[8.660254, 45.0], [48.660254, 45.0]], [[17.320508, 30.0], [57.320508, 30.0]], [[25.980762, 45.0], [65.980762, 45.0]]]
    """
    gap_x = 2 * gap * cos60
    gap_y = gap * cos30
    w = sqrt(3) * size
    h = 2 * size
    dx = w + (gap_x * 2)
    dy = (2 * size) + (h / 2) + (gap_y * 2)
    item = item.translate((w / 2) + gap_x, (3 * h / 4) + gap_y, reps=1)
    item = item.translate(dx, 0, reps=reps1)
    item = item.translate(0, dy, reps=reps2)

    return item


def cover_hex_flat(
    item: Group | Shape | Tag,
    size: float,
    gap: float = 0,
    reps1: int = 2,
    reps2: int = 2,
) -> Group:
    """
    Covers an area with a hexagonal pattern with flat tops.

    Args:
        item (Group | Shape | Tag): The item to be repeated.
        size (float): The size of the hexagons.
        gap (float, optional): The gap between hexagons. Defaults to 0.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 2.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 2.

    Returns:
        Group: The resulting pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = cover_hex_flat(mark, 10, reps1=1, reps2=1)
        >>> [[[round(c, 6) for c in p[:2]] for p in s.vertices] for s in pattern]
        [[[0.0, 0.0], [40.0, 0.0]], [[15.0, 8.660254], [55.0, 8.660254]], [[30.0, 0.0], [70.0, 0.0]], [[45.0, 8.660254], [85.0, 8.660254]], [[0.0, 17.320508], [40.0, 17.320508]], [[15.0, 25.980762], [55.0, 25.980762]], [[30.0, 17.320508], [70.0, 17.320508]], [[45.0, 25.980762], [85.0, 25.980762]]]
    """
    gap_x = 2 * gap * cos60
    gap_y = gap * cos30
    h = sqrt(3) * size
    dx = 3 * size + (gap_x * 2)
    dy = h + (gap_y * 2)
    item = item.translate((3 * size / 2) + gap_x, (h / 2) + gap_y, reps=1)
    item = item.translate(dx, 0, reps=reps1)
    item = item.translate(0, dy, reps=reps2)

    return item


# Wallpaper groups

# generator is the primary cell
# tile is the basic unit cell
# mirrors
# glide-mirror (m1, m2), glide-dist (dist1, dist2) if more than one
# rotocenters
# n rotations (n1, n2, ...) corresponding to each rotocenter
# translations (vec1, n1, vec2, n2) this is the lattice


def wallpaper_p1(
    generator: Group | Shape | Tag,
    vector1: VecType,
    vector2: VecType,
    reps1: int = 4,
    reps2: int = 4,
) -> Group:
    """
    Translation symmetry.
    IUC: p1
    Conway: o
    Oblique lattice
    PointType group: C1

    Args:
        generator (Group | Shape | Tag): The repeating motif.
        vector1 (VecType): The translation vector in the x direction.
        vector2 (VecType): The translation vector in the y direction.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Group: The resulting wallpaper pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = wallpaper_p1(mark, (20, 0), (0, 60), reps1=1, reps2=1)
        >>> [[[round(c, 6) for c in p[:2]] for p in s.vertices] for s in pattern]
        [[[0.0, 0.0], [40.0, 0.0]], [[20.0, 0.0], [60.0, 0.0]], [[0.0, 60.0], [40.0, 60.0]], [[20.0, 60.0], [60.0, 60.0]]]
    """
    dx1, dy1 = vector1
    wallpaper = generator.translate(dx1, dy1, reps=reps1)
    dx2, dy2 = vector2
    wallpaper.translate(dx2, dy2, reps=reps2)

    return wallpaper


def wallpaper_p2(
    generator: Shape | Group | Tag,
    vector1: VecType,
    vector2: VecType,
    reps1: int = 4,
    reps2: int = 4,
) -> Group:
    """
    Half-turn rotation symmetry.
    IUC: p2 (p211)
    Conway: 2222
    Oblique lattice
    PointType group: C2

    Args:
        generator (Shape | Group | Tag): The repeating motif.
        vector1 (VecType): The translation vector in the x direction.
        vector2 (VecType): The translation vector in the y direction.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Group: The resulting wallpaper pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = wallpaper_p2(mark, (20, 0), (0, 60), reps1=1, reps2=1)
        >>> [[[round(c, 6) for c in p[:2]] for p in s.vertices] for s in pattern]
        [[[0.0, 0.0], [40.0, 0.0]], [[20.0, 60.0], [-20.0, 60.0]], [[20.0, 0.0], [60.0, 0.0]], [[40.0, 60.0], [0.0, 60.0]], [[0.0, 60.0], [40.0, 60.0]], [[20.0, 120.0], [-20.0, 120.0]], [[20.0, 60.0], [60.0, 60.0]], [[40.0, 120.0], [0.0, 120.0]]]
    """
    rotocenter = midpoint(vector1, vector2)
    wallpaper = generator.rotate(pi, rotocenter, reps=1)
    dx1, dy1 = vector1
    wallpaper.translate(dx1, dy1, reps=reps1)
    dx2, dy2 = vector2
    wallpaper.translate(dx2, dy2, reps=reps2)

    return wallpaper


def wallpaper_p2_rect_lattice(
    generator: Shape | Group | Tag,
    rotocenter: PointType,
    vector1: VecType,
    vector2: VecType,
    reps1: int = 4,
    reps2: int = 4,
) -> Group:
    """Half-turn rotation symmetry on a rectangular lattice (IUC: p2).

    Args:
        generator: Motif to repeat.
        rotocenter: Center of the 180° rotation.
        vector1: First lattice translation vector.
        vector2: Second lattice translation vector.
        reps1: Repetitions along ``vector1``.
        reps2: Repetitions along ``vector2``.

    Returns:
        Group: Wallpaper pattern as a Group.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = wallpaper_p2_rect_lattice(
        ...     mark, (0, 0), (20, 0), (0, 60), reps1=1, reps2=1
        ... )
        >>> [[[round(c, 6) for c in p[:2]] for p in s.vertices] for s in pattern]
        [[[0.0, 0.0], [40.0, 0.0]], [[0.0, 0.0], [-40.0, 0.0]], [[20.0, 0.0], [60.0, 0.0]], [[20.0, 0.0], [-20.0, 0.0]], [[0.0, 60.0], [40.0, 60.0]], [[0.0, 60.0], [-40.0, 60.0]], [[20.0, 60.0], [60.0, 60.0]], [[20.0, 60.0], [-20.0, 60.0]]]
    """
    wallpaper = generator.rotate(pi, rotocenter, reps=1)
    dx1, dy1 = vector1
    wallpaper.translate(dx1, dy1, reps=reps1)
    dx2, dy2 = vector2
    wallpaper.translate(dx2, dy2, reps=reps2)

    return wallpaper


def wallpaper_p3(
    generator: Shape | Group | Tag,
    rotocenter: PointType,
    distance: float,
    reps1: int = 4,
    reps2: int = 4,
    flat_hex: bool = False,
) -> Group:
    """
    Three rotations.
    IUC: p3
    Conway: 333
    Hexagonal lattice.
    PointType group: C3

    Args:
        generator (Shape | Group | Tag): The repeating motif.
        rotocenter (PointType): The center of rotation.
        distance (float): The distance between the centers of the hexagons.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.
        flat_hex (bool, optional): If True, hexagons are flat-topped. Defaults to False.

    Returns:
        Group: The resulting wallpaper pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = wallpaper_p3(mark, (0, 0), 10, reps1=1, reps2=1)
        >>> len(pattern)
        24
        >>> [[[round(c, 6) for c in p[:2]] for p in s.vertices] for s in pattern[:4]]
        [[[0.0, 0.0], [40.0, 0.0]], [[0.0, 0.0], [-20.0, 34.641016]], [[0.0, 0.0], [-20.0, -34.641016]], [[8.660254, 15.0], [48.660254, 15.0]]]
        >>> [[round(c, 6) for c in p[:2]] for p in pattern[-1].vertices]
        [[25.980762, 45.0], [5.980762, 10.358984]]
    """
    wallpaper = generator.rotate(2 * pi / 3, rotocenter, reps=2)
    if flat_hex:
        cover_hex_flat(wallpaper, distance, reps1=reps1, reps2=reps2)
    else:
        cover_hex_pointy(wallpaper, distance, reps1=reps1, reps2=reps2)

    return wallpaper


def wallpaper_p4(
    generator: Group | Shape | Tag,
    rotocenter: PointType,
    distance: float,
    reps1: int = 4,
    reps2: int = 4,
) -> Group:
    """
    Pinwheel symmetry.
    IUC: p4
    Conway: 442
    Square lattice
    PointType group: C4

    Args:
        generator (Group | Shape | Tag): The repeating motif.
        rotocenter (PointType): The center of rotation.
        distance (float): The distance between the centers of the squares.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Group: The resulting wallpaper pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = wallpaper_p4(mark, (0, 0), 10, reps1=1, reps2=1)
        >>> [[[round(c, 6) or 0.0 for c in p[:2]] for p in s.vertices] for s in pattern]
        [[[0.0, 0.0], [40.0, 0.0]], [[0.0, 0.0], [0.0, 40.0]], [[0.0, 0.0], [-40.0, 0.0]], [[0.0, 0.0], [0.0, -40.0]], [[10.0, 0.0], [50.0, 0.0]], [[10.0, 0.0], [10.0, 40.0]], [[10.0, 0.0], [-30.0, 0.0]], [[10.0, 0.0], [10.0, -40.0]], [[0.0, 10.0], [40.0, 10.0]], [[0.0, 10.0], [0.0, 50.0]], [[0.0, 10.0], [-40.0, 10.0]], [[0.0, 10.0], [0.0, -30.0]], [[10.0, 10.0], [50.0, 10.0]], [[10.0, 10.0], [10.0, 50.0]], [[10.0, 10.0], [-30.0, 10.0]], [[10.0, 10.0], [10.0, -30.0]]]
    """
    wallpaper = generator.rotate(pi / 2, rotocenter, reps=3)
    wallpaper.translate(distance, 0, reps=reps1)
    wallpaper.translate(0, distance, reps=reps2)

    return wallpaper


def wallpaper_p6(
    generator: Group | Shape | Tag,
    rotocenter: PointType,
    hex_size: float,
    reps1: int = 4,
    reps2: int = 4,
    flat_hex: bool = False,
) -> Group:
    """
    Six rotations.
    IUC: p6
    Conway : 632
    Hexagonal lattice
    PointType group: C6

    Args:
        generator: The repeating motif.
        rotocenter: The center of rotation.
        hex_size: Hexagon radius/size passed to ``cover_hex_*``.
        reps1: Repetitions in the cover grid x direction.
        reps2: Repetitions in the cover grid y direction.
        flat_hex: If True, use flat-topped hex cover; otherwise pointy-topped.

    Returns:
        Group: The resulting pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = wallpaper_p6(mark, (0, 0), 10, reps1=1, reps2=1)
        >>> len(pattern)
        48
        >>> [[[round(c, 6) for c in p[:2]] for p in s.vertices] for s in pattern[:4]]
        [[[0.0, 0.0], [40.0, 0.0]], [[0.0, 0.0], [20.0, 34.641016]], [[0.0, 0.0], [-20.0, 34.641016]], [[0.0, 0.0], [-40.0, 0.0]]]
        >>> [[round(c, 6) for c in p[:2]] for p in pattern[-1].vertices]
        [[25.980762, 45.0], [45.980762, 10.358984]]
    """
    wallpaper = generator.rotate(pi / 3, rotocenter, reps=5)
    if flat_hex:
        cover_hex_flat(wallpaper, hex_size, reps1=reps1, reps2=reps2)
    else:
        cover_hex_pointy(wallpaper, hex_size, reps1=reps1, reps2=reps2)

    return wallpaper


def wallpaper_pm(
    generator: Group | Shape | Tag,
    mirror_line: LineType,
    dx: float,
    dy: float,
    reps1: int = 4,
    reps2: int = 4,
) -> Group:
    """
    Mirror symmetry.
    Mirror could be horizontal or vertical.
    IUC: pm(p1m1)
    Conway : **
    Rectangular lattice
    PointType group: D1

    Args:
        generator (Group | Shape | Tag): The repeating motif.
        mirror_line (LineType): The line of symmetry.
        dx (float): Translation distance in the x direction.
        dy (float): Translation distance in the y direction.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Group: The resulting pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> axis = ((0, -20), (100, -20))
        >>> pattern = wallpaper_pm(mark, axis, 20, 20, reps1=1, reps2=1)
        >>> [[[round(c, 6) for c in p[:2]] for p in s.vertices] for s in pattern]
        [[[0.0, 0.0], [40.0, 0.0]], [[0.0, -40.0], [40.0, -40.0]], [[20.0, 0.0], [60.0, 0.0]], [[20.0, -40.0], [60.0, -40.0]], [[0.0, 20.0], [40.0, 20.0]], [[0.0, -20.0], [40.0, -20.0]], [[20.0, 20.0], [60.0, 20.0]], [[20.0, -20.0], [60.0, -20.0]]]
    """
    wallpaper = generator.mirror(mirror_line, reps=1)
    wallpaper.translate(dx, 0, reps=reps1)
    wallpaper.translate(0, dy, reps=reps2)

    return wallpaper


def wallpaper_pg(
    generator: Group | Shape | Tag,
    mirror_line: LineType,
    distance: float,
    dx: float,
    dy: float,
    reps1: int = 4,
    reps2: int = 4,
) -> Group:
    """
    Glide symmetry.
    IUC: pg(p1g1)
    Conway : xx
    Rectangular lattice
    PointType group: D1

    Args:
        generator (Group | Shape | Tag): The repeating motif.
        mirror_line (LineType): The line of symmetry.
        distance (float): The distance for the glide reflection.
        dx (float): Translation distance in the x direction.
        dy (float): Translation distance in the y direction.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Group: The resulting pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> axis = ((0, -20), (100, -20))
        >>> pattern = wallpaper_pg(mark, axis, 5, 20, 20, reps1=1, reps2=1)
        >>> [[[round(c, 6) for c in p[:2]] for p in s.vertices] for s in pattern]
        [[[0.0, 0.0], [40.0, 0.0]], [[5.0, -40.0], [45.0, -40.0]], [[20.0, 0.0], [60.0, 0.0]], [[25.0, -40.0], [65.0, -40.0]], [[0.0, 20.0], [40.0, 20.0]], [[5.0, -20.0], [45.0, -20.0]], [[20.0, 20.0], [60.0, 20.0]], [[25.0, -20.0], [65.0, -20.0]]]
    """
    wallpaper = generator.glide(mirror_line, distance, reps=1)
    wallpaper.translate(dx, 0, reps=reps1)
    wallpaper.translate(0, dy, reps=reps2)

    return wallpaper


def wallpaper_cm(
    generator: Group | Shape | Tag,
    mirror_point: PointType,
    rhomb_size: float,
    reps1: int = 4,
    reps2: int = 4,
    horizontal: bool = True,
) -> Group:
    """
    Spinning-sidle symmetry.
    IUC: cm(c1m1)
    Conway : *x
    Rhombic lattice
    PointType group: D1

    Args:
        generator (Group | Shape | Tag): The repeating motif.
        mirror_point (PointType): The point of symmetry.
        rhomb_size (float): The size of the rhombuses.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.
        horizontal (bool, optional): If True, the mirror line is horizontal. Defaults to True.

    Returns:
        Group: The resulting pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = wallpaper_cm(mark, (0, 0), 10, reps1=1, reps2=1)
        >>> [[[round(c, 6) for c in p[:2]] for p in s.vertices] for s in pattern[:4]]
        [[[0.0, 0.0], [40.0, 0.0]], [[0.0, 0.0], [40.0, 0.0]], [[7.071068, 7.071068], [47.071068, 7.071068]], [[7.071068, 7.071068], [47.071068, 7.071068]]]
        >>> [[round(c, 6) for c in p[:2]] for p in pattern[-1].vertices]
        [[21.213203, 21.213203], [61.213203, 21.213203]]
    """
    x1, y1 = mirror_point[:2]
    if horizontal:
        x2, y2 = x1 + 1, y1
    else:
        x2, y2 = x1, y1 + 1
    wallpaper = generator.mirror(((x1, y1), (x2, y2)), reps=1)
    cover_rhombic(
        wallpaper,
        rhomb_size,
        reps1=reps1,
        reps2=reps2,
    )

    return wallpaper


def wallpaper_pmm(
    generator: Group | Shape | Tag,
    mirror_cross: PointType,
    dx: float,
    dy: float,
    reps1: int = 4,
    reps2: int = 4,
) -> Group:
    """
    Double mirror symmetry.
    IUC: pmm(p2mm)
    Conway : *2222
    Rectangular lattice
    PointType group: D2

    Args:
        generator (Group | Shape | Tag): The repeating motif.
        mirror_cross (PointType): The point where the mirror lines cross.
        dx (float): Translation distance in the x direction.
        dy (float): Translation distance in the y direction.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Group: The resulting pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = wallpaper_pmm(mark, (0, 0), 20, 20, reps1=1, reps2=1)
        >>> [[[round(c, 6) for c in p[:2]] for p in s.vertices] for s in pattern[:4]]
        [[[0.0, 0.0], [40.0, 0.0]],
         [[0.0, 0.0], [40.0, 0.0]],
         [[0.0, 0.0], [-40.0, 0.0]],
         [[0.0, 0.0], [-40.0, 0.0]]]
        >>> [[round(c, 6) for c in p[:2]] for p in pattern[-1].vertices]
        [[20.0, 20.0], [-20.0, 20.0]]
    """
    x, y = mirror_cross[:2]
    mirror_line1 = ((x, y), (x + 1, y))
    mirror_line2 = ((x, y), (x, y + 1))
    wallpaper = generator.mirror(mirror_line1, reps=1)
    wallpaper.mirror(mirror_line2, reps=1)
    wallpaper.translate(dx, 0, reps=reps1)
    wallpaper.translate(0, dy, reps=reps2)

    return wallpaper


def wallpaper_pmg(
    generator: Group | Shape | Tag,
    center_point: PointType,
    dx: float,
    dy: float,
    reps1: int = 4,
    reps2: int = 4,
    horizontal: bool = True,
) -> Group:
    """
    Glided staggered symmetry.
    IUC: pmg(p2mg)
    Conway : 22*
    Rectangular lattice
    PointType group: D2

    Args:
        generator (Group | Shape | Tag): The repeating motif.
        center_point (PointType): The center point for the symmetry.
        dx (float): Translation distance in the x direction.
        dy (float): Translation distance in the y direction.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.
        horizontal (bool, optional): If True, the mirror line is horizontal. Defaults to True.

    Returns:
        Group: The resulting pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = wallpaper_pmg(mark, (0, 0), 20, 20, reps1=1, reps2=1)
        >>> [[[round(c, 6) for c in p[:2]] for p in s.vertices] for s in pattern[:4]]
        [[[0.0, 0.0], [40.0, 0.0]], [[0.0, 10.0], [-40.0, 10.0]], [[0.0, 0.0], [40.0, 0.0]], [[0.0, -10.0], [-40.0, -10.0]]]
        >>> [[round(c, 6) for c in p[:2]] for p in pattern[-1].vertices]
        [[20.0, 10.0], [-20.0, 10.0]]
    """
    x, y = center_point[:2]
    if horizontal:
        rotocenter = midpoint((x, y), (x, (y + dy) / 2))
        mirror_line = ((x, y), (x + 1, y))
    else:
        rotocenter = midpoint((x, y), (-(x + dx) / 2, y))
        mirror_line = ((x, y), (x, y + 1))
    wallpaper = generator.rotate(pi, rotocenter, reps=1)
    wallpaper.mirror(mirror_line, reps=1)
    wallpaper.translate(dx, 0, reps=reps1)
    wallpaper.translate(0, dy, reps=reps2)

    return wallpaper


def wallpaper_pgg(
    generator: Group | Shape | Tag,
    rotocenter: PointType,
    dx: float,
    dy: float,
    reps1: int = 4,
    reps2: int = 4,
    horizontal: bool = True,
) -> Group:
    """
    Double glide symmetry.
    IUC: pgg(p2gg)
    Conway : 22x
    Rectangular lattice
    PointType group: D2

    Args:
        generator (Group | Shape | Tag): The repeating motif.
        rotocenter (PointType): The center of rotation.
        dx (float): Translation distance in the x direction.
        dy (float): Translation distance in the y direction.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.
        horizontal (bool, optional): If True, the glide reflection is horizontal. Defaults to True.

    Returns:
        Group: The resulting pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = wallpaper_pgg(mark, (20, 0), 20, 20, reps1=1, reps2=1)
        >>> [[[round(c, 6) or 0.0 for c in p[:2]] for p in s.vertices] for s in pattern[:4]]
        [[[0.0, 0.0], [40.0, 0.0]],
         [[0.0, 0.01], [40.0, 0.01]],
         [[40.0, 0.0], [0.0, 0.0]],
         [[40.0, -0.01], [0.0, -0.01]]]
        >>> [[round(c, 6) for c in p[:2]] for p in pattern[-1].vertices]
        [[60.0, 19.99], [20.0, 19.99]]
    """
    if horizontal:
        dist = rotocenter[0] - generator.center[0]
        wallpaper = generator.glide(
            generator.horiz_centerline, 2 * dist, reps=1
        )
        wallpaper.rotate(pi, rotocenter, reps=1)
    else:
        dist = rotocenter[1] - generator.center[1]
        wallpaper = generator.glide(generator.vert_centerline, 2 * dist, reps=1)
        wallpaper.rotate(pi, rotocenter, reps=1)
    wallpaper.translate(dx, 0, reps=reps1)
    wallpaper.translate(0, dy, reps=reps2)

    return wallpaper


def wallpaper_cmm(
    generator: Group | Shape | Tag,
    mirror_cross: PointType,
    rhomb_size: float,
    reps1: int = 4,
    reps2: int = 4,
) -> Group:
    """
    Staggered double mirror symmetry.
    IUC: cmm(c2mm)
    Conway : 2*22
    Rhombic lattice
    PointType group: D2

    Args:
        generator (Group | Shape | Tag): The repeating motif.
        mirror_cross (PointType): The point where the mirror lines cross.
        rhomb_size (float): The size of the rhombuses.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Group: The resulting pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = wallpaper_cmm(mark, (0, 0), 10, reps1=1, reps2=1)
        >>> len(pattern)
        32
        >>> [[[round(c, 6) for c in p[:2]] for p in s.vertices] for s in pattern[:4]]
        [[[0.0, 0.0], [40.0, 0.0]],
         [[0.0, 0.0], [40.0, 0.0]],
         [[0.0, 0.0], [-40.0, 0.0]],
         [[0.0, 0.0], [-40.0, 0.0]]]
        >>> [[round(c, 6) for c in p[:2]] for p in pattern[-1].vertices]
        [[21.213203, 21.213203], [-18.786797, 21.213203]]
    """
    x, y = mirror_cross[:2]
    mirror_line1 = ((x, y), (x + 1, y))
    mirror_line2 = ((x, y), (x, y + 1))
    wallpaper = generator.mirror(mirror_line1, reps=1)
    wallpaper.mirror(mirror_line2, reps=1)
    cover_rhombic(wallpaper, rhomb_size, reps1=reps1, reps2=reps2)

    return wallpaper


def wallpaper_p4m(
    generator: Group | Shape | Tag,
    mirror_cross: PointType,
    side_length: float,
    reps1: int = 4,
    reps2: int = 4,
) -> Group:
    """
    Block symmetry.
    IUC: p4m(p4mm)
    Conway : *442
    Square lattice
    PointType group: D4

    Args:
        generator (Group | Shape | Tag): The repeating motif.
        mirror_cross (PointType): The point where the mirror lines cross.
        side_length (float): The side length of the squares.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Group: The resulting pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = wallpaper_p4m(mark, (0, 0), 10, reps1=1, reps2=1)
        >>> len(pattern)
        32
        >>> [[[round(c, 6) or 0.0 for c in p[:2]] for p in s.vertices] for s in pattern[:4]]
        [[[0.0, 0.0], [40.0, 0.0]],
         [[0.0, 0.0], [-40.0, 0.0]],
         [[0.0, 0.0], [0.0, 40.0]],
         [[0.0, 0.0], [0.0, -40.0]]]
        >>> [[round(c, 6) for c in p[:2]] for p in pattern[-1].vertices]
        [[10.0, 10.0], [10.0, 50.0]]
    """
    x, y = mirror_cross[:2]
    mirror_line = ((x, y), (x, y + 1))
    rotocenter = x, y
    wallpaper = generator.mirror(mirror_line, reps=1)
    wallpaper.rotate(pi / 2, rotocenter, reps=3)
    wallpaper.translate(side_length, 0, reps=reps1)
    wallpaper.translate(0, side_length, reps=reps2)

    return wallpaper


def wallpaper_p4g(
    generator: Group | Shape | Tag,
    dist: float,
    reps1: int = 4,
    reps2: int = 4,
) -> Group:
    """
    Mirrored pinwheel symmetry.
    IUC: p4g(p4gm)
    Conway : 4*2
    Square lattice
    PointType group: D4

    Args:
        generator (Group | Shape | Tag): The repeating motif.
        dist (float): The distance between the centers of the squares.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Group: The resulting pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = wallpaper_p4g(mark, 20, reps1=1, reps2=1)
        >>> len(pattern)
        64
        >>> [[[round(c, 6) or 0.0 for c in p[:2]] for p in s.vertices] for s in pattern[:4]]
        [[[0.0, 0.0], [40.0, 0.0]],
         [[0.0, 0.0], [0.0, 40.0]],
         [[0.0, 0.0], [-40.0, 0.0]],
         [[0.0, 0.0], [0.0, -40.0]]]
        >>> [[round(c, 6) for c in p[:2]] for p in pattern[-1].vertices]
        [[30.0, 30.0], [30.0, 70.0]]
    """
    # rotocenter should be (0, 0) and mirror_cross should be (d/4,d/4 )
    # translations are (d, d)

    wallpaper = generator.rotate(pi / 2, (0, 0), reps=3)
    x, y = (dist / 4, dist / 4)
    wallpaper.mirror(((x, y), (x + 1, y)), reps=1)
    wallpaper.mirror(((x, y), (x, y + 1)), reps=1)
    wallpaper.translate(dist, 0, reps=reps1)
    wallpaper.translate(0, dist, reps=reps2)

    return wallpaper


def wallpaper_p3m1(
    generator: Group | Shape | Tag,
    center_point: PointType,
    hex_size: float,
    reps1: int = 4,
    reps2: int = 4,
) -> Group:
    """
    Mirror and three rotations.
    IUC: p3m1
    Conway : *333
    Hexagonal lattice
    PointType group: D3

    Args:
        generator (Group | Shape | Tag): The repeating motif.
        center_point (PointType): The center point for the symmetry.
        hex_size (float): The size of the hexagons.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Group: The resulting pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = wallpaper_p3m1(mark, (0, 0), 10, reps1=1, reps2=1)
        >>> len(pattern)
        48
        >>> [[[round(c, 6) or 0.0 for c in p[:2]] for p in s.vertices] for s in pattern[:4]]
        [[[0.0, 0.0], [40.0, 0.0]], [[0.0, 0.0], [-20.0, -34.641016]], [[0.0, 0.0], [-20.0, 34.641016]], [[0.0, 0.0], [40.0, 0.0]]]
        >>> [[round(c, 6) for c in p[:2]] for p in pattern[-1].vertices]
        [[45.0, 25.980762], [25.0, 60.621778]]
    """
    x, y = center_point[:2]
    mirror_line = line_through_point_and_angle((x, y), 2 * pi / 3)
    wallpaper = generator.mirror(mirror_line, reps=1)
    wallpaper.rotate(2 * pi / 3, center_point, reps=2)
    cover_hex(wallpaper, hex_size, reps1=reps1, reps2=reps2, flat=True)

    return wallpaper


def wallpaper_p31m(
    generator: Group | Shape | Tag,
    center_point: PointType,
    hex_size: float,
    reps1: int = 4,
    reps2: int = 4,
) -> Group:
    """
    Three rotations and a mirror.
    IUC: p31m
    Conway : 3*3
    Hexagonal lattice
    PointType group: D3

    Args:
        generator (Group | Shape | Tag): The repeating motif.
        center_point (PointType): The center point for the symmetry.
        hex_size (float): The size of the hexagons.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Group: The resulting pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = wallpaper_p31m(mark, (0, 0), 10, reps1=1, reps2=1)
        >>> len(pattern)
        144
        >>> [[[round(c, 6) for c in p[:2]] for p in s.vertices] for s in pattern[:4]]
        [[[0.0, 0.0], [40.0, 0.0]], [[0.0, 0.0], [-20.0, 34.641016]], [[0.0, 0.0], [-20.0, -34.641016]], [[0.0, 5.7732], [40.0, 5.7732]]]
        >>> [[round(c, 6) for c in p[:2]] for p in pattern[-1].vertices]
        [[54.999869, 31.754189], [94.999869, 31.754189]]
    """
    x, y = center_point[:2]
    dy = 0.28866 * hex_size
    mirror_line = ((x, y + dy), (x + 1, y + dy))
    wallpaper = generator.rotate(2 * pi / 3, center_point, reps=2)
    wallpaper.mirror(mirror_line, reps=1)

    rotocenter = (x + hex_size / 2, y + dy)
    wallpaper.rotate(2 * pi / 3, rotocenter, reps=2)
    cover_hex(wallpaper, hex_size, reps1=reps1, reps2=reps2, flat=True)

    return wallpaper


def wallpaper_p6m(
    generator: Group | Shape | Tag,
    rotocenter: PointType,
    mirror_cross: PointType,
    hex_size: float,
    reps1: int = 4,
    reps2: int = 4,
    flat_hex: bool = False,
) -> Group:
    """
    Kaleidoscope.
    IUC: p6m(p6mm)
    Conway : *632
    Hexagonal lattice
    PointType group: D6

    Args:
        generator (Group | Shape | Tag): The repeating motif.
        rotocenter (PointType): The center of rotation.
        mirror_cross (PointType): The point where the mirror lines cross.
        hex_size (float): The size of the hexagons.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.
        flat_hex (bool, optional): If True, hexagons are flat-topped. Defaults to False.

    Returns:
        Group: The resulting pattern as a Group object.

    Examples:
        >>> import simetri.graphics as sg
        >>> mark = sg.Shape([(0, 0), (40, 0)])
        >>> pattern = wallpaper_p6m(mark, (0, 0), (0, 0), 10, reps1=1, reps2=1)
        >>> len(pattern)
        24
        >>> [[[round(c, 6) or 0.0 for c in p[:2]] for p in s.vertices] for s in pattern[:4]]
        [[[40.0, 0.0], [-40.0, 0.0]], [[20.0, 34.641016], [-20.0, -34.641016]], [[-20.0, 34.641016], [20.0, -34.641016]], [[48.660254, 15.0], [-31.339746, 15.0]]]
        >>> [[round(c, 6) for c in p[:2]] for p in pattern[-1].vertices]
        [[5.980762, 79.641016], [45.980762, 10.358984]]
    """
    x, y = mirror_cross[:2]
    mirror1 = [(x, y), (x + 1, y)]
    mirror2 = [(x, y), (x, y + 1)]
    wallpaper = generator.mirror(mirror1, reps=1)
    wallpaper.mirror(mirror2, reps=1)
    wallpaper.rotate(pi / 3, rotocenter, reps=5)
    wallpaper = wallpaper.merge_shapes(abs_tol=1)
    if flat_hex:
        cover_hex_flat(wallpaper, hex_size, reps1=reps1, reps2=reps2)
    else:
        cover_hex_pointy(wallpaper, hex_size, reps1=reps1, reps2=reps2)

    return wallpaper

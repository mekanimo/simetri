"""Simetri graphics library's wallpaper patterns.
"""

# Only six of the 17 wallpaper groups are tested yet.

from math import sqrt, pi, cos

from typing import Union

from ..geometry.geometry import mid_point, line_through_point_and_angle
from ..graphics.common import VecType, Point, Line
from ..helpers.illustration import Tag
from ..graphics.batch import Batch
from ..graphics.shape import Shape


cos60 = cos(pi / 3)
cos30 = cos(pi / 6)


def cover_hex(
    item: Union[Batch, Shape, Tag],
    size: float,
    gap: float = 0,
    reps1: int = 2,
    reps2: int = 2,
    flat: bool = True,
) -> Batch:
    """
    Covers an area with a hexagonal pattern.

    Args:
        item (Union[Batch, Shape, Tag]): The item to be repeated.
        size (float): The size of the hexagons.
        gap (float, optional): The gap between hexagons. Defaults to 0.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 2.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 2.
        flat (bool, optional): If True, hexagons are flat-topped. Defaults to True.

    Returns:
        Batch: The resulting pattern as a Batch object.
    """
    gap_x = 2 * gap * cos60
    gap_y = gap * cos30
    if (flat):
        w = 2 * size
        h = sqrt(3) * size
        dx = 3 * size + (gap_x * 2)
        dy = h + (gap_y * 2)
        item.translate((3 * size / 2) + gap_x, (h / 2) + gap_y, reps=1)
    else:
        w = sqrt(3) * size
        h = 2 * size
        dx = w + (gap_x * 2)
        dy = (2 * size) + (h / 2) + (gap_y * 2)
        item.translate((w / 2) + gap_x, (3 * h / 4) + gap_y, reps=1)

    item.translate(dx, 0, reps=reps1)
    item.translate(0, dy, reps=reps2)

    return item


def cover_rhombic(
    item: Union[Batch, Shape, Tag], size: float, reps1: int = 2, reps2: int = 2
) -> Batch:
    """
    Covers an area with a rhombic pattern.

    Args:
        item (Union[Batch, Shape, Tag]): The item to be repeated.
        size (float): The size of the rhombuses.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 2.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 2.

    Returns:
        Batch: The resulting pattern as a Batch object.
    """
    sqrt2 = sqrt(2)
    diag = (sqrt2 / 2) * size
    item.translate(diag, diag, reps=1)
    dx = dy = diag * 2
    item.translate(dx, 0, reps=reps1)
    item.translate(0, dy, reps=reps2)

    return item


def hex_grid_pointy(x: float, y: float, size: float, n_rows: int, n_cols: int) -> Batch:
    """
    Creates a hexagonal grid with pointy tops.

    Args:
        x (float): The x-coordinate of the starting point.
        y (float): The y-coordinate of the starting point.
        size (float): The size of the hexagons.
        n_rows (int): Number of rows in the grid.
        n_cols (int): Number of columns in the grid.

    Returns:
        Batch: The resulting grid as a Batch of Shapes.
    """
    height = sqrt(3) * size
    width = 2 * size
    edge_length = 2 * size * cos(pi / 6)
    # create the first row by translating a single hexagon in the x direction
    row = Batch(Shape([(x, y)])).translate(size, 0, reps=n_cols - 1)
    # create the second row by translating the first row
    two_rows = row.translate(width, height + edge_length, reps=1)
    # create the grid by translating the first and second row in the y direction
    grid = two_rows.translate(0, 2 * size + edge_length, reps=n_rows / 2 - 1)

    return grid


def cover_hex_pointy(
    item: Union[Shape, Batch, Tag],
    size: float,
    gap: float = 0,
    reps1: int = 2,
    reps2: int = 2,
) -> Batch:
    """
    Covers an area with a hexagonal pattern with pointy tops.

    Args:
        item (Union[Shape, Batch, Tag]): The item to be repeated.
        size (float): The size of the hexagons.
        gap (float, optional): The gap between hexagons. Defaults to 0.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 2.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 2.

    Returns:
        Batch: The resulting pattern as a Batch object.
    """
    gap_x = 2 * gap * cos60
    gap_y = gap * cos30
    w = sqrt(3) * size
    h = 2 * size
    dx = w + (gap_x * 2)
    dy = (2 * size) + (h / 2) + (gap_y * 2)
    item.translate((w / 2) + gap_x, (3 * h / 4) + gap_y, reps=1)
    item.translate(dx, 0, reps=reps1)
    item.translate(0, dy, reps=reps2)

    return item


def cover_hex_flat(
    item: Union[Batch, Shape, Tag],
    size: float,
    gap: float = 0,
    reps1: int = 2,
    reps2: int = 2,
) -> Batch:
    """
    Covers an area with a hexagonal pattern with flat tops.

    Args:
        item (Union[Batch, Shape, Tag]): The item to be repeated.
        size (float): The size of the hexagons.
        gap (float, optional): The gap between hexagons. Defaults to 0.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 2.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 2.

    Returns:
        Batch: The resulting pattern as a Batch object.
    """
    gap_x = 2 * gap * cos60
    gap_y = gap * cos30
    h = sqrt(3) * size
    dx = 3 * size + (gap_x * 2)
    dy = h + (gap_y * 2)
    item.translate((3 * size / 2) + gap_x, (h / 2) + gap_y, reps=1)
    item.translate(dx, 0, reps=reps1)
    item.translate(0, dy, reps=reps2)

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
    generator: Union[Batch, Shape, Tag],
    vector1: VecType,
    vector2: VecType,
    reps1: int = 4,
    reps2: int = 4,
) -> Batch:
    """
    Translation symmetry.
    IUC: p1
    Conway: o
    Oblique lattice
    Point group: C1

    Args:
        generator (Union[Batch, Shape, Tag]): The repeating motif.
        vector1 (VecType): The translation vector in the x direction.
        vector2 (VecType): The translation vector in the y direction.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Batch: The resulting wallpaper pattern as a Batch object.
    """
    dx1, dy1 = vector1
    wallpaper = generator.translate(dx1, dy1, reps1)
    dx2, dy2 = vector2
    wallpaper.translate(dx2, dy2, reps2)

    return wallpaper


def wallpaper_p2(
    generator: Union[Shape, Batch, Tag],
    vector1: VecType,
    vector2: VecType,
    reps1: int = 4,
    reps2: int = 4,
) -> Batch:
    """
    Half-turn rotation symmetry.
    IUC: p2 (p211)
    Conway: 2222
    Oblique lattice
    Point group: C2

    Args:
        generator (Union[Shape, Batch, Tag]): The repeating motif.
        vector1 (VecType): The translation vector in the x direction.
        vector2 (VecType): The translation vector in the y direction.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Batch: The resulting wallpaper pattern as a Batch object.
    """
    rotocenter = mid_point(vector1, vector2)
    wallpaper = generator.rotate(pi, rotocenter, reps=1)
    dx1, dy1 = vector1
    wallpaper.translate(dx1, dy1, reps=reps1)
    dx2, dy2 = vector2
    wallpaper.translate(dx2, dy2, reps=reps2)

    return wallpaper


def wallpaper_p2_rect_lattice(
    generator: Union[Shape, Batch, Tag],
    rotocenter: Point,
    vector1: VecType,
    vector2: VecType,
    reps1: int = 4,
    reps2: int = 4,
) -> Batch:
    # """
    # Half-turn rotation symmetry.
    # IUC: p2 (p211)
    # Conway: 2222
    # Oblique lattice
    # Point group: C2

    # Point argument can be an Anchor object or a tuple, or two points can be given
    # as a sequence.

    # Example:
    # import simetri.graphics as sg
    # import simetri.wallpaper as wp

    # directory = 'dir_path'
    # canvas = sg.Canvas()

    # F = sg.letter_F()
    # vec1 = (2 * F.width + 10, 0)
    # vec2 = (0, F.height + 10)

    # pattern = wp.wallpaper_p2(F, vec1, vec2, reps1=2, reps2=2)
    # file_path = os.path.join(directory, 'wallpaper_test_p2.pdf')
    # canvas.draw(pattern, file_path=file_path)

    # """

    rotocenter = mid_point(vector1, vector2)
    wallpaper = generator.rotate(pi, rotocenter, reps=1)
    dx1, dy1 = vector1
    wallpaper.translate(dx1, dy1, reps=reps1)
    dx2, dy2 = vector2
    wallpaper.translate(dx2, dy2, reps=reps2)

    return wallpaper


def wallpaper_p3(
    generator: Union[Shape, Batch, Tag],
    rotocenter: Point,
    distance: float,
    reps1: int = 4,
    reps2: int = 4,
    flat_hex: bool = False,
) -> Batch:
    """
    Three rotations.
    IUC: p3
    Conway: 333
    Hexagonal lattice.
    Point group: C3

    Args:
        generator (Union[Shape, Batch, Tag]): The repeating motif.
        rotocenter (Point): The center of rotation.
        distance (float): The distance between the centers of the hexagons.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.
        flat_hex (bool, optional): If True, hexagons are flat-topped. Defaults to False.

    Returns:
        Batch: The resulting wallpaper pattern as a Batch object.
    """
    wallpaper = generator.rotate(2 * pi / 3, rotocenter, reps=2)
    if flat_hex:
        cover_hex_flat(wallpaper, distance, reps1=reps1, reps2=reps2)
    else:
        cover_hex_pointy(wallpaper, distance, reps1=reps1, reps2=reps2)

    return wallpaper


def wallpaper_p4(
    generator: Union[Batch, Shape, Tag],
    rotocenter: Point,
    distance: float,
    reps1: int = 4,
    reps2: int = 4,
) -> Batch:
    """
    Pinwheel symmetry.
    IUC: p4
    Conway: 442
    Square lattice
    Point group: C4

    Args:
        generator (Union[Batch, Shape, Tag]): The repeating motif.
        rotocenter (Point): The center of rotation.
        distance (float): The distance between the centers of the squares.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Batch: The resulting wallpaper pattern as a Batch object.
    """
    wallpaper = generator.rotate(pi / 2, rotocenter, reps=3)
    wallpaper.translate(distance, 0, reps1)
    wallpaper.translate(0, distance, reps2)

    return wallpaper


def wallpaper_p6(
    generator: Union[Batch, Shape, Tag],
    rotocenter: Point,
    hex_size: float,
    reps1: int = 4,
    reps2: int = 4,
    flat_hex=False,
) -> Batch:
    """
    Six rotations.
    IUC: p6
    Conway : 632
    Hexagonal lattice
    Point group: C6

    Args:
        generator (Union[Batch, Shape, Tag]): The repeating motif.
        rotocenter (Point): The center of rotation.
        hex_size (float): The size of the hexagons.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.
        flat_hex (bool, optional): If True, hexagons are flat-topped. Defaults to False.

    Returns:
        Batch: The resulting pattern as a Batch object.
    """
    wallpaper = generator.rotate(pi / 3, rotocenter, reps=5)
    if flat_hex:
        cover_hex_flat(wallpaper, hex_size, reps1=reps1, reps2=reps2)
    else:
        cover_hex_pointy(wallpaper, hex_size, reps1=reps1, reps2=reps2)

    return wallpaper


def wallpaper_pm(
    generator: Union[Batch, Shape, Tag],
    mirror_line: Line,
    dx: float,
    dy: float,
    reps1: int = 4,
    reps2: int = 4,
) -> Batch:
    """
    Mirror symmetry.
    Mirror could be horizontal or vertical.
    IUC: pm(p1m1)
    Conway : **
    Rectangular lattice
    Point group: D1

    Args:
        generator (Union[Batch, Shape, Tag]): The repeating motif.
        mirror_line (Line): The line of symmetry.
        dx (float): Translation distance in the x direction.
        dy (float): Translation distance in the y direction.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Batch: The resulting pattern as a Batch object.
    """
    wallpaper = generator.mirror(mirror_line, reps=1)
    wallpaper.translate(dx, 0, reps=reps1)
    wallpaper.translate(0, dy, reps=reps2)

    return wallpaper


def wallpaper_pg(
    generator: Union[Batch, Shape, Tag],
    mirror_line: Line,
    distance: float,
    dx: float,
    dy: float,
    reps1: int = 4,
    reps2: int = 4,
) -> Batch:
    """
    Glide symmetry.
    IUC: pg(p1g1)
    Conway : xx
    Rectangular lattice
    Point group: D1

    Args:
        generator (Union[Batch, Shape, Tag]): The repeating motif.
        mirror_line (Line): The line of symmetry.
        distance (float): The distance for the glide reflection.
        dx (float): Translation distance in the x direction.
        dy (float): Translation distance in the y direction.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Batch: The resulting pattern as a Batch object.
    """
    wallpaper = generator.glide(mirror_line, distance, reps=1)
    wallpaper.translate(dx, 0, reps=reps1)
    wallpaper.translate(0, dy, reps=reps2)

    return wallpaper


def wallpaper_cm(
    generator: Union[Batch, Shape, Tag],
    mirror_point: Point,
    rhomb_size: float,
    reps1: int = 4,
    reps2: int = 4,
    horizontal: bool = True,
) -> Batch:
    """
    Spinning-sidle symmetry.
    IUC: cm(c1m1)
    Conway : *x
    Rhombic lattice
    Point group: D1

    Args:
        generator (Union[Batch, Shape, Tag]): The repeating motif.
        mirror_point (Point): The point of symmetry.
        rhomb_size (float): The size of the rhombuses.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.
        horizontal (bool, optional): If True, the mirror line is horizontal. Defaults to True.

    Returns:
        Batch: The resulting pattern as a Batch object.
    """
    x1, y1 = mirror_point
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
    generator: Union[Batch, Shape, Tag],
    mirror_cross: Point,
    dx: float,
    dy: float,
    reps1=4,
    reps2=4,
) -> Batch:
    """
    Double mirror symmetry.
    IUC: pmm(p2mm)
    Conway : *2222
    Rectangular lattice
    Point group: D2

    Args:
        generator (Union[Batch, Shape, Tag]): The repeating motif.
        mirror_cross (Point): The point where the mirror lines cross.
        dx (float): Translation distance in the x direction.
        dy (float): Translation distance in the y direction.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Batch: The resulting pattern as a Batch object.
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
    generator: Union[Batch, Shape, Tag],
    center_point: Point,
    dx: float,
    dy: float,
    reps1=4,
    reps2=4,
    horizontal=True,
) -> Batch:
    """
    Glided staggered symmetry.
    IUC: pmg(p2mg)
    Conway : 22*
    Rectangular lattice
    Point group: D2

    Args:
        generator (Union[Batch, Shape, Tag]): The repeating motif.
        center_point (Point): The center point for the symmetry.
        dx (float): Translation distance in the x direction.
        dy (float): Translation distance in the y direction.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.
        horizontal (bool, optional): If True, the mirror line is horizontal. Defaults to True.

    Returns:
        Batch: The resulting pattern as a Batch object.
    """
    x, y = center_point[:2]
    if horizontal:
        rotocenter = mid_point((x, y), (x, (y + dy) / 2))
        mirror_line = ((x, y), (x + 1, y))
    else:
        rotocenter = mid_point((x, y), (-(x + dx) / 2, y))
        mirror_line = ((x, y), (x, y + 1))
    wallpaper = generator.rotate(pi, rotocenter, reps=1)
    wallpaper.mirror(mirror_line, reps=1)
    wallpaper.translate(dx, 0, reps=reps1)
    wallpaper.translate(0, dy, reps=reps2)

    return wallpaper


def wallpaper_pgg(
    generator: Union[Batch, Shape, Tag],
    rotocenter: Point,
    dx: float,
    dy: float,
    reps1: int = 4,
    reps2: int = 4,
    horizontal=True,
) -> Batch:
    """
    Double glide symmetry.
    IUC: pgg(p2gg)
    Conway : 22x
    Rectangular lattice
    Point group: D2

    Args:
        generator (Union[Batch, Shape, Tag]): The repeating motif.
        rotocenter (Point): The center of rotation.
        dx (float): Translation distance in the x direction.
        dy (float): Translation distance in the y direction.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.
        horizontal (bool, optional): If True, the glide reflection is horizontal. Defaults to True.

    Returns:
        Batch: The resulting pattern as a Batch object.
    """
    if horizontal:
        dist = rotocenter[0] - generator.center[0]
        wallpaper = generator.glide(generator.horiz_centerline, 2 * dist, reps=1)
        wallpaper.rotate(pi, rotocenter, reps=1)
    else:
        dist = rotocenter[1] - generator.center[1]
        wallpaper = generator.glide(generator.vert_centerline, 2 * dist, reps=1)
        wallpaper.rotate(pi, rotocenter, reps=1)
    wallpaper.translate(dx, 0, reps1)
    wallpaper.translate(0, dy, reps2)

    return wallpaper


def wallpaper_cmm(
    generator: Union[Batch, Shape, Tag],
    mirror_cross: Point,
    rhomb_size: float,
    reps1: int = 4,
    reps2: int = 4,
) -> Batch:
    """
    Staggered double mirror symmetry.
    IUC: cmm(c2mm)
    Conway : 2*22
    Rhombic lattice
    Point group: D2

    Args:
        generator (Union[Batch, Shape, Tag]): The repeating motif.
        mirror_cross (Point): The point where the mirror lines cross.
        rhomb_size (float): The size of the rhombuses.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Batch: The resulting pattern as a Batch object.
    """
    x, y = mirror_cross[:2]
    mirror_line1 = ((x, y), (x + 1, y))
    mirror_line2 = ((x, y), (x, y + 1))
    wallpaper = generator.mirror(mirror_line1, reps=1)
    wallpaper.mirror(mirror_line2, reps=1)
    cover_rhombic(wallpaper, rhomb_size, reps1=reps1, reps2=reps2)

    return wallpaper


def wallpaper_p4m(
    generator: Union[Batch, Shape, Tag],
    mirror_cross: Point,
    side_length: float,
    reps1: int = 4,
    reps2: int = 4,
) -> Batch:
    """
    Block symmetry.
    IUC: p4m(p4mm)
    Conway : *442
    Square lattice
    Point group: D4

    Args:
        generator (Union[Batch, Shape, Tag]): The repeating motif.
        mirror_cross (Point): The point where the mirror lines cross.
        side_length (float): The side length of the squares.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Batch: The resulting pattern as a Batch object.
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
    generator: Union[Batch, Shape, Tag], dist: float, reps1: int = 4, reps2: int = 4
) -> Batch:
    """
    Mirrored pinwheel symmetry.
    IUC: p4g(p4gm)
    Conway : 4*2
    Square lattice
    Point group: D4

    Args:
        generator (Union[Batch, Shape, Tag]): The repeating motif.
        dist (float): The distance between the centers of the squares.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Batch: The resulting pattern as a Batch object.
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
    generator: Union[Batch, Shape, Tag],
    center_point: Point,
    hex_size: float,
    reps1: int = 4,
    reps2: int = 4,
) -> Batch:
    """
    Mirror and three rotations.
    IUC: p3m1
    Conway : *333
    Hexagonal lattice
    Point group: D3

    Args:
        generator (Union[Batch, Shape, Tag]): The repeating motif.
        center_point (Point): The center point for the symmetry.
        hex_size (float): The size of the hexagons.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Batch: The resulting pattern as a Batch object.
    """
    x, y = center_point[:2]
    mirror_line = line_through_point_and_angle((x, y), 2 * pi / 3)
    wallpaper = generator.mirror(mirror_line, reps=1)
    wallpaper.rotate(2 * pi / 3, center_point, reps=2)
    cover_hex(wallpaper, hex_size, reps1=reps1, reps2=reps2, flat=True)

    return wallpaper


def wallpaper_p31m(
    generator: Union[Batch, Shape, Tag],
    center_point: Point,
    hex_size: float,
    reps1: int = 4,
    reps2: int = 4,
) -> Batch:
    """
    Three rotations and a mirror.
    IUC: p31m
    Conway : 3*3
    Hexagonal lattice
    Point group: D3

    Args:
        generator (Union[Batch, Shape, Tag]): The repeating motif.
        center_point (Point): The center point for the symmetry.
        hex_size (float): The size of the hexagons.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.

    Returns:
        Batch: The resulting pattern as a Batch object.
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
    generator: Union[Batch, Shape, Tag],
    rotocenter: Point,
    mirror_cross: Point,
    hex_size: float,
    reps1: int = 4,
    reps2: int = 4,
    flat_hex: bool = False,
) -> Batch:
    """
    Kaleidoscope.
    IUC: p6m(p6mm)
    Conway : *632
    Hexagonal lattice
    Point group: D6

    Args:
        generator (Union[Batch, Shape, Tag]): The repeating motif.
        rotocenter (Point): The center of rotation.
        mirror_cross (Point): The point where the mirror lines cross.
        hex_size (float): The size of the hexagons.
        reps1 (int, optional): Number of repetitions in the x direction. Defaults to 4.
        reps2 (int, optional): Number of repetitions in the y direction. Defaults to 4.
        flat_hex (bool, optional): If True, hexagons are flat-topped. Defaults to False.

    Returns:
        Batch: The resulting pattern as a Batch object.
    """
    x, y = mirror_cross[:2]
    mirror1 = [(x, y), (x + 1, y)]
    mirror2 = [(x, y), (x, y + 1)]
    wallpaper = generator.mirror(mirror1, reps=1)
    wallpaper.mirror(mirror2, reps=1)
    wallpaper.rotate(pi / 3, rotocenter, reps=5)
    wallpaper = wallpaper.merge_shapes(1, n_round=0)
    if flat_hex:
        cover_hex_flat(wallpaper, hex_size, reps1=reps1, reps2=reps2)
    else:
        cover_hex_pointy(wallpaper, hex_size, reps1=reps1, reps2=reps2)

    return wallpaper

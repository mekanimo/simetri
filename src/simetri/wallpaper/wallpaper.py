"""Simetri graphics library's wallpaper patterns.
"""
#Only six of the 17 wallpaper groups are tested yet.

from math import sqrt, pi, cos

from typing import Union

from ..helpers.geometry import mid_point, line_through_point_and_angle
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
    gap_x = 2 * gap * cos60
    gap_y = gap * cos30
    if flat:
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
    """Cover a rhombic lattice with a given size."""
    sqrt2 = sqrt(2)
    diag = (sqrt2 / 2) * size
    item.translate(diag, diag, reps=1)
    dx = dy = diag * 2
    item.translate(dx, 0, reps=reps1)
    item.translate(0, dy, reps=reps2)

    return item


def hex_grid_pointy(x: float, y: float, size: float, n_rows: int, n_cols: int) -> Batch:
    """Create a hexagonal grid with pointy tops.
    size is the distance between the center and a corner.
    Return a Batch of Shapes."""
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
    """Cover a hexagonal lattice with pointy tops."""
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
    """Cover a hexagonal lattice with flat tops."""
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

    generator is the repeating motif that could be a Shape, Batch, or Tag object
    vector1 is the translation vector in the x direction given as a tuple (x, y)
    vector2 is the translation vector in the y direction given as a tuple (x, y)
    reps1 is the number of repetitions in the x direction
    reps2 is the number of repetitions in the y direction
    Resulting wallpaper pattern is returned as a Batch object

    Example:
    import simetri.graphics as sg
    import simetri.wallpaper as wp

    directory = 'c:/tmp'
    canvas = sg.Canvas()
    F = sg.letter_F()
    vec1 = (F.width + 10, 20)
    vec2 = (30, F.height + 10)

    pattern = wp.wallpaper_p1(F, vec1, vec2, reps1=4, reps2=4)
    file_path = os.path.join(directory, 'wallpaper_test_p1.pdf')
    canvas.draw(pattern, file_path=file_path)
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

    Example:
    import simetri.graphics as sg
    import simetri.wallpaper as wp

    directory = 'c:/tmp'
    canvas = sg.Canvas()

    F = sg.letter_F()
    vec1 = (2 * F.width + 10, 0)
    vec2 = (0, F.height + 10)

    pattern = wp.wallpaper_p2(F, vec1, vec2, reps1=2, reps2=2)
    file_path = os.path.join(directory, 'wallpaper_test_p2.pdf')
    canvas.draw(pattern, file_path=file_path)

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
    """
    Half-turn rotation symmetry.
    IUC: p2 (p211)
    Conway: 2222
    Oblique lattice
    Point group: C2

    Point argument can be an Anchor object or a tuple, or two points can be given
    as a sequence.

    Example:
    import simetri.graphics as sg
    import simetri.wallpaper as wp

    directory = 'c:/tmp'
    canvas = sg.Canvas()

    F = sg.letter_F()
    vec1 = (2 * F.width + 10, 0)
    vec2 = (0, F.height + 10)

    pattern = wp.wallpaper_p2(F, vec1, vec2, reps1=2, reps2=2)
    file_path = os.path.join(directory, 'wallpaper_test_p2.pdf')
    canvas.draw(pattern, file_path=file_path)

    """

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

    generator is the repeating motif that could be a Shape, Batch, or Tag object
    rotocenter is the center of rotation
    distance is the distance between the centers of the hexagons
    reps1 is the number of repetitions in the x direction
    reps2 is the number of repetitions in the y direction
    flat_hex is True if the hexagons are oriented flat-side up,
    False if they are oriented flat-side on the side
    Resulting wallpaper pattern is returned as a Batch object

    Example:
    import simetri.graphics as sg
    import simetri.wallpaper as wp

    directory = 'c:/tmp'
    canvas = sg.Canvas()
    F = sg.letter_F()
    rotocenter = F.north_west
    distance = 100

    pattern = wp.wallpaper_p3(F, rotocenter, distance, reps1=4, reps2=4)
    file_path = os.path.join(directory, 'wallpaper_test_p3.pdf')
    canvas.draw(pattern, file_path=file_path)

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

    Example:

    import simetri.graphics as sg
    import simetri.wallpaper as wp

    directory = 'c:/tmp'
    canvas = sg.Canvas()

    F = sg.letter_F()
    x, y = F.north_west
    rotocenter = (x-5, y+5)
    distance = 170
    reps1 = 4
    reps2 = 4

    pattern = wp.wallpaper_p4(F, rotocenter, distance, reps1, reps2)
    file_path = os.path.join(directory, 'wallpaper_test_p4.pdf')
    canvas.draw(pattern, file_path=file_path)

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
    Glide symmetry
    IUC: pg(p1g1)
    Conway : xx
    Rectangular lattice
    Point group: D1
    """
    # This should be mirrorline, glide_dist, translation
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
    """
    x, y = mirror_cross
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
    """
    x, y = center_point
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
    """
    x, y = mirror_cross
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
    """
    x, y = mirror_cross
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
    rotocenter can be Anchor.SOUTHWEST, Anchor.SOUTHEAST,
                Anchor.NORTHWEST, or Anchor.NORTHEAST
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
    """
    x, y = center_point
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
    """
    x, y = center_point
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
    """
    x, y = mirror_cross
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

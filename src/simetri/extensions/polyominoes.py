"""Generate free, fixed, and chiral polyominoes as cell centers or Figures."""

from collections.abc import Iterable, Iterator
from typing import Literal

import simetri.graphics as sg
from simetri.shapes.figure import Figure

Cell = tuple[int, int]
NormalizedPoly = tuple[Cell, ...]
PolyoType = Literal["fixed", "free", "chiral"]


def _rotate_90(p_set: NormalizedPoly) -> NormalizedPoly:
    return tuple((y, -x) for x, y in p_set)


def _reflect_x(p_set: NormalizedPoly) -> NormalizedPoly:
    return tuple((-x, y) for x, y in p_set)


def _normalize_variant(p_set: Iterable[Cell]) -> NormalizedPoly:
    min_x = min(x for x, _ in p_set)
    min_y = min(y for _, y in p_set)
    return tuple(sorted((x - min_x, y - min_y) for x, y in p_set))


def _canonical(
    poly: set[Cell] | tuple[Cell, ...], polyo_type: PolyoType
) -> NormalizedPoly:
    min_x = min(x for x, _ in poly)
    min_y = min(y for _, y in poly)
    normalized = tuple(sorted((x - min_x, y - min_y) for x, y in poly))
    if polyo_type == "fixed":
        return normalized

    variants = [normalized]
    current = normalized
    for _ in range(3):
        current = _normalize_variant(_rotate_90(current))
        variants.append(current)

    if polyo_type == "free":
        current = _normalize_variant(_reflect_x(normalized))
        variants.append(current)
        for _ in range(3):
            current = _normalize_variant(_rotate_90(current))
            variants.append(current)

    return min(variants)


def generate_centers(
    n: int, polyo_type: PolyoType = "free"
) -> list[list[Cell]]:
    """Return all n-omino cell-center sets for the given equivalence type.

    Args:
        n: Number of unit squares in each polyomino.
        polyo_type: One of ``"fixed"``, ``"free"``, or ``"chiral"``.

    Returns:
        List of polyominoes, each a list of ``(x, y)`` integer cell centers
        normalized so the minimum coordinates are at the origin.

    Raises:
        ValueError: If ``polyo_type`` is not a supported value.

    Examples:
        >>> generate_centers(0)
        []
        >>> generate_centers(1)
        [[(0, 0)]]
        >>> len(generate_centers(2))
        1
    """
    if n <= 0:
        return []

    # Validation check for type
    valid_types = {"fixed", "free", "chiral"}
    if polyo_type not in valid_types:
        raise ValueError(f"polyo_type must be one of {valid_types}")

    # Core Redelmeier-like cell growth algorithm
    # Start with a single square at the origin
    current_level = {((0, 0),)}

    for _ in range(2, n + 1):
        next_level = set()
        for poly in current_level:
            # Find all neighboring open grid squares
            neighbors = set()
            for x, y in poly:
                for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                    if (nx, ny) not in poly:
                        neighbors.add((nx, ny))
            # Expand the shape by one cell and canonicalize it
            for neighbor in neighbors:
                new_poly = set(poly)
                new_poly.add(neighbor)
                next_level.add(_canonical(new_poly, polyo_type))
        current_level = next_level

    return [list(p) for p in current_level]


def iter_centers(
    n: int, polyo_type: PolyoType = "free"
) -> Iterator[list[Cell]]:
    """Yield n-omino cell-center sets one at a time.

    Args:
        n: Number of unit squares in each polyomino.
        polyo_type: One of ``"fixed"``, ``"free"``, or ``"chiral"``.

    Yields:
        Lists of ``(x, y)`` integer cell centers for each distinct n-omino.

    Raises:
        ValueError: If ``polyo_type`` is not a supported value.

    Examples:
        >>> centers = list(iter_centers(2))
        >>> len(centers)
        1
        >>> sorted(centers[0])
        [(0, 0), (0, 1)]
    """
    if n <= 0:
        return

    valid_types = {"fixed", "free", "chiral"}
    if polyo_type not in valid_types:
        raise ValueError(f"polyo_type must be one of {valid_types}")

    current_level = {((0, 0),)}

    for _ in range(2, n + 1):
        next_level = set()
        for poly in current_level:
            neighbors = set()
            for x, y in poly:
                for neighbor in (
                    (x + 1, y),
                    (x - 1, y),
                    (x, y + 1),
                    (x, y - 1),
                ):
                    if neighbor not in poly:
                        neighbors.add(neighbor)
            for neighbor in neighbors:
                new_poly = set(poly)
                new_poly.add(neighbor)
                next_level.add(_canonical(new_poly, polyo_type))
        current_level = next_level

    for poly in current_level:
        yield list(poly)


def iter_polyominoes(
    n: int, polyo_type: PolyoType = "free", size: float = 20
) -> Iterator[Figure]:
    """Yield ``Figure`` objects for each distinct n-omino.

    Each figure has merged-outline ``geometry`` and a non-filled unit-square
    ``skin``.

    Args:
        n: Number of unit squares in each polyomino.
        polyo_type: ``"fixed"``, ``"free"``, or ``"chiral"``.
        size: Side length of each unit square in points.

    Yields:
        ``Figure`` instances with cells on a ``size``-point grid.

    Examples:
        >>> fig = next(iter_polyominoes(1, size=10))
        >>> fig.__class__.__name__
        'Figure'
        >>> len(list(iter_polyominoes(2, size=10)))
        1
    """
    res = iter_centers(n=n, polyo_type=polyo_type)

    unit = sg.square(size, (0, 0))
    for polyo in res:
        units = sg.Group()
        for x, y in polyo:
            units.append(unit.copy().move_to((x * size, y * size)))
        skin = units.copy()
        skin.set_attribs("fill", False)
        geometry = units.merge_shapes(keep_one_duplicate=True)[0]
        figure = Figure(geometry, skin)
        yield (figure)


# Example:

# canvas = sg.Canvas()

# n = 6
# size = 15
# polyos = iter_polyominoes(n=n, size=size)
# for i in range(640):
#     pos = sg.get_cell_position(
#         index=i,
#         n_columns=12,
#         cell_width=(size - 4) * n,
#         cell_height=(size - 2) * n,
#         gap=size / 8,
#         margin=size,
#     )
#     polyo = next(polyos, None)

#     if polyo is None:
#         print(i)
#         break
#     if len(polyo) == 2 or not simetri.geom.polygons.polygon_utils.is_simple(
#         polyo[0]
#     ):
#         color = sg.red
#         alpha = 1
#     else:
#         color = sg.random_color()
#         alpha = 0.5
#     canvas.draw(polyo.move_to(pos), fill_color=color, alpha=alpha)

# canvas.save(f"c:/tmp/polyominoes_{n}.svg", overwrite=True)

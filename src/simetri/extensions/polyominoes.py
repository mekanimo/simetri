import simetri.geom.polygons.polygon_utils
import simetri.graphics as sg
from simetri.shapes.figure import Figure


def generate_centers(n, polyo_type="free"):
    if n <= 0:
        return []

    # Validation check for type
    valid_types = {"fixed", "free", "chiral"}
    if polyo_type not in valid_types:
        raise ValueError(f"polyo_type must be one of {valid_types}")

    def canonical(poly):
        # Shift coordinates to start at the origin (0, 0)
        min_x = min(x for x, y in poly)
        min_y = min(y for x, y in poly)
        normalized = tuple(sorted((x - min_x, y - min_y) for x, y in poly))

        # Track all valid variations based on the polyomino type
        variants = [normalized]

        # 1. Fixed: No rotations, no reflections. Only shifted to origin.
        if polyo_type == "fixed":
            return normalized

        # Helper to rotate a polyomino 90 degrees clockwise
        def rotate_90(p_set):
            return tuple((y, -x) for x, y in p_set)

        # Helper to reflect a polyomino horizontally across the y-axis
        def reflect_x(p_set):
            return tuple((-x, y) for x, y in p_set)

        # Helper to re-normalize a transformed shape back to the origin
        def normalize_variant(p_set):
            mx = min(x for x, y in p_set)
            my = min(y for x, y in p_set)
            return tuple(sorted((x - mx, y - my) for x, y in p_set))

        # 2. Chiral: 4 rotations allowed, no reflections
        curr = normalized
        for _ in range(3):
            curr = normalize_variant(rotate_90(curr))
            variants.append(curr)

        # 3. Free: 4 rotations AND their reflections (8 shapes total)
        if polyo_type == "free":
            reflected = normalize_variant(reflect_x(normalized))
            curr_ref = reflected
            variants.append(curr_ref)
            for _ in range(3):
                curr_ref = normalize_variant(rotate_90(curr_ref))
                variants.append(curr_ref)

        # Return the unique lexicographical minimum shape as the canonical ID
        return min(variants)

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
                next_level.add(canonical(new_poly))
        current_level = next_level

    return [list(p) for p in current_level]


def iter_centers(n, polyo_type="free"):
    if n <= 0:
        return

    valid_types = {"fixed", "free", "chiral"}
    if polyo_type not in valid_types:
        raise ValueError(f"polyo_type must be one of {valid_types}")

    def canonical(poly):
        min_x = min(x for x, y in poly)
        min_y = min(y for x, y in poly)
        normalized = tuple(sorted((x - min_x, y - min_y) for x, y in poly))

        variants = [normalized]

        if polyo_type == "fixed":
            return normalized

        def rotate_90(p_set):
            return tuple((y, -x) for x, y in p_set)

        def reflect_x(p_set):
            return tuple((-x, y) for x, y in p_set)

        def normalize_variant(p_set):
            min_x = min(x for x, y in p_set)
            min_y = min(y for x, y in p_set)
            return tuple(sorted((x - min_x, y - min_y) for x, y in p_set))

        current = normalized
        for _ in range(3):
            current = normalize_variant(rotate_90(current))
            variants.append(current)

        if polyo_type == "free":
            current = normalize_variant(reflect_x(normalized))
            variants.append(current)
            for _ in range(3):
                current = normalize_variant(rotate_90(current))
                variants.append(current)

        return min(variants)

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
                next_level.add(canonical(new_poly))
        current_level = next_level

    for poly in current_level:
        yield list(poly)


def iter_polyominoes(n, polyo_type="free", size=20):
    res = iter_centers(n=n, polyo_type=polyo_type)

    unit = sg.square((0, 0), size)
    for polyo in res:
        units = sg.Group()
        for x, y in polyo:
            units.append(unit.copy().move_to((x * size, y * size)))
        skin = units.copy()
        skin.set_attribs("fill", False)
        geometry = units.merge_shapes(remove_duplicate_edges=True)[0]
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

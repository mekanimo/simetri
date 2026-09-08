"""Simetri graphics library's utility functions."""

import ast
import base64
import cmath
import collections
import inspect
import os
import random
import re
import string
import sys
from bisect import bisect_left
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from functools import cmp_to_key, reduce, wraps
from math import atan2, ceil, cos, factorial, floor, isclose, sin, sqrt
from pathlib import Path
from time import monotonic, perf_counter, sleep, time

import numpy as np
from numpy import array, ndarray
from PIL import ImageFont

from ..base.common import LineType, PointType
from ..config.settings import (
    _print_options,
    defaults,
    issue_warning,
)


@contextmanager
def print_options(**kwargs):
    """Temporarily override library print formatting options.

    Args:
        **kwargs: Options such as ``precision`` and ``suppress`` to apply
            for the duration of the context.

    Yields:
        None: Control returns to the caller with options applied.

    Examples:
        >>> import simetri.graphics as sg
        >>> with sg.print_options(precision=2):
        ...     sg.format_data(1 / 3)
        '0.33'
        >>> sg.format_data(1 / 3)
        '0.3333'
    """
    # Save the current state
    old_options = _print_options.copy()

    # Update with the new user choices
    _print_options.update(kwargs)
    try:
        yield
    finally:
        # Always restore the old state, even if an error happens
        _print_options.update(old_options)


def format_data(data):
    """Format a value using the current print options.

    Args:
        data: Scalar, sequence, or mapping to format.

    Returns:
        str: Formatted representation.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.format_data(1 / 3)
        '0.3333'
        >>> sg.format_data((1.0, 2.5))
        '(1.0000, 2.5000)'
    """
    # Get current formatting options
    precision = _print_options["precision"]
    suppress = _print_options["suppress"]

    # Handle floats
    if isinstance(data, float):
        if suppress and abs(data) < 1e-4:
            return "0.0"
        return f"{data:.{precision}f}"

    # Handle tuples (must convert to string representation)
    if isinstance(data, tuple):
        return f"({', '.join(format_data(x) for x in data)})"

    # Handle lists, arrays, or sequences (excluding strings)
    if isinstance(data, collections.abc.Sequence) and not isinstance(
        data, (str, bytes)
    ):
        return f"[{', '.join(format_data(x) for x in data)}]"

    # Handle dictionaries
    if isinstance(data, dict):
        items = [f"{k!r}: {format_data(v)}" for k, v in data.items()]
        return f"{{{', '.join(items)}}}"

    # Return everything else (ints, strings, etc.) as their default string
    return str(data)


def p_print(*data):
    """Print values using ``format_data`` and current print options.

    Args:
        *data: Values to print.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.p_print(1 / 3)
        0.3333
    """
    output = " ".join(format_data(item) for item in data)
    print(output)


# alias for get_cell_position
def get_cell_pos(
    index: int,
    n_columns: int,
    cell_width: float,
    cell_height: float,
    gap: float | None = None,
    horiz_gap: float | None = None,
    vert_gap: float | None = None,
    margin: float | None = None,
    left_margin: float | None = None,
    bot_margin: float | None = None,
    right_margin: float | None = None,
    top_margin: float | None = None,
    from_bottom_left: bool = True,
) -> tuple[float, float]:
    """Return the center of a grid cell by linear index.

    Cells are laid out in row-major order: index increases along columns first,
    then along rows. With ``from_bottom_left=True`` (default), row 0 is the
    bottom row and later rows stack upward. With ``from_bottom_left=False``,
    row 0 is the top row and later rows stack downward.

    Args:
        index: Linear cell index (0-based).
        n_columns: Number of columns in the grid.
        cell_width: Width of each cell.
        cell_height: Height of each cell.
        gap: Uniform spacing between cells. Defaults to 0. Do not pass together
            with ``horiz_gap`` or ``vert_gap``.
        horiz_gap: Horizontal spacing between adjacent cells. Defaults to
            ``gap`` when not given.
        vert_gap: Vertical spacing between adjacent cells. Defaults to ``gap``
            when not given.
        margin: Uniform offset for unset sides. Defaults to 0. May be combined
            with individual side margins; any side not passed explicitly uses
            ``margin``. Do not pass together with all four per-side margins.
        left_margin: Offset of the grid from the left. Defaults to ``margin``
            when not given.
        bot_margin: Offset of the grid from the bottom when
            ``from_bottom_left=True``. Defaults to ``margin`` when not given.
        right_margin: Right inset. With the other side margins, rejects
            ``margin`` together with all four per-side margins.
        top_margin: Offset of the grid from the top when
            ``from_bottom_left=False``. Defaults to ``margin`` when not given.
        from_bottom_left: If True, index 0 is the bottom-left cell and rows
            increase upward. If False, index 0 is the top-left cell and rows
            increase downward.

    Returns:
        ``(x, y)`` coordinates of the cell center.

    Raises:
        ValueError: If ``gap`` is passed together with ``horiz_gap`` or
            ``vert_gap``, or if ``margin`` is passed together with all four
            per-side margins.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.get_cell_pos(0, 3, 10, 20)
        (5.0, 10.0)
        >>> sg.get_cell_pos(1, 2, 10, 10)
        (15.0, 5.0)
    """

    return get_cell_position(
        index=index,
        n_columns=n_columns,
        cell_width=cell_width,
        cell_height=cell_height,
        gap=gap,
        horiz_gap=horiz_gap,
        vert_gap=vert_gap,
        margin=margin,
        left_margin=left_margin,
        bot_margin=bot_margin,
        right_margin=right_margin,
        top_margin=top_margin,
        from_bottom_left=from_bottom_left,
    )


def get_cell_position(
    index: int,
    n_columns: int,
    cell_width: float,
    cell_height: float,
    gap: float | None = None,
    horiz_gap: float | None = None,
    vert_gap: float | None = None,
    margin: float | None = None,
    left_margin: float | None = None,
    bot_margin: float | None = None,
    right_margin: float | None = None,
    top_margin: float | None = None,
    from_bottom_left: bool = True,
) -> tuple[float, float]:
    """Return the center of a grid cell by linear index.

    Cells are laid out in row-major order: index increases along columns first,
    then along rows. With ``from_bottom_left=True`` (default), row 0 is the
    bottom row and later rows stack upward. With ``from_bottom_left=False``,
    row 0 is the top row and later rows stack downward.

    Args:
        index: Linear cell index (0-based).
        n_columns: Number of columns in the grid.
        cell_width: Width of each cell.
        cell_height: Height of each cell.
        gap: Uniform spacing between cells. Defaults to 0. Do not pass together
            with ``horiz_gap`` or ``vert_gap``.
        horiz_gap: Horizontal spacing between adjacent cells. Defaults to
            ``gap`` when not given.
        vert_gap: Vertical spacing between adjacent cells. Defaults to ``gap``
            when not given.
        margin: Uniform offset for unset sides. Defaults to 0. May be combined
            with individual side margins; any side not passed explicitly uses
            ``margin``. Do not pass together with all four per-side margins.
        left_margin: Offset of the grid from the left. Defaults to ``margin``
            when not given.
        bot_margin: Offset of the grid from the bottom when
            ``from_bottom_left=True``. Defaults to ``margin`` when not given.
        right_margin: Right inset. With the other side margins, rejects
            ``margin`` together with all four per-side margins.
        top_margin: Offset of the grid from the top when
            ``from_bottom_left=False``. Defaults to ``margin`` when not given.
        from_bottom_left: If True, index 0 is the bottom-left cell and rows
            increase upward. If False, index 0 is the top-left cell and rows
            increase downward.

    Returns:
        ``(x, y)`` coordinates of the cell center.

    Raises:
        ValueError: If ``gap`` is passed together with ``horiz_gap`` or
            ``vert_gap``, or if ``margin`` is passed together with all four
            per-side margins.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.get_cell_position(0, 3, 10, 20)
        (5.0, 10.0)
        >>> sg.get_cell_position(3, 3, 10, 20)
        (5.0, 30.0)
        >>> sg.get_cell_position(0, 3, 10, 20, from_bottom_left=False)
        (5.0, -10.0)
    """

    row = index // n_columns
    col = index % n_columns

    if gap is not None and (horiz_gap is not None or vert_gap is not None):
        raise ValueError(
            "Cannot set both gap and horiz_gap/vert_gap; pass one or the other."
        )

    all_margins = (
        margin is not None
        and left_margin is not None
        and bot_margin is not None
        and right_margin is not None
        and top_margin is not None
    )
    if all_margins:
        raise ValueError(
            "Cannot set margin together with all per-side margins."
        )

    if gap is None:
        gap = 0
    if horiz_gap is None:
        horiz_gap = gap
    if vert_gap is None:
        vert_gap = gap

    if margin is None:
        margin = 0
    if left_margin is None:
        left_margin = margin
    if bot_margin is None:
        bot_margin = margin
    if top_margin is None:
        top_margin = margin

    pitch_x = cell_width + horiz_gap
    pitch_y = cell_height + vert_gap
    x = left_margin + col * pitch_x + cell_width / 2

    if from_bottom_left:
        y = bot_margin + row * pitch_y + cell_height / 2
    else:
        y = -(top_margin + row * pitch_y + cell_height / 2)

    res = (x, y)
    return res


def all_cells_connected(
    indices,
    n_rows: int = 3,
    n_cols: int = 3,
    diagonal_neighbors: bool = True,
) -> bool:
    """Return True if every cell in ``indices`` belongs to one connected group.

    Args:
        indices: Cell indices on the grid (e.g. ``range(9)`` for 3×3).
        n_rows: Number of grid rows.
        n_cols: Number of grid columns.
        diagonal_neighbors: If True, cells sharing a corner are adjacent. If
            False, only edge-adjacent cells are adjacent.

    Returns:
        True when all given cells form a single connected component (including
        when the set is empty or has one cell); False when the cells form two
        or more separate groups.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.all_cells_connected(range(9), 3, 3, diagonal_neighbors=False)
        True
        >>> sg.all_cells_connected([0, 1, 8], 3, 3, diagonal_neighbors=False)
        False
    """

    def connected(
        index1: int,
        index2: int,
        n_rows: int = 3,
        n_cols: int = 3,
        diagonal_neighbors: bool = True,
    ) -> bool:
        row1, col1 = divmod(index1, n_cols)
        row2, col2 = divmod(index2, n_cols)
        if row2 < 0 or row2 >= n_rows or col2 < 0 or col2 >= n_cols:
            return False
        dr = abs(row1 - row2)
        dc = abs(col1 - col2)
        if dr > 1 or dc > 1 or (dr == 0 and dc == 0):
            return False
        if diagonal_neighbors:
            return True
        return dr + dc == 1

    cells = set(indices)
    if len(cells) <= 1:
        res = True
    else:
        visited = set()
        stack = [next(iter(cells))]
        while stack:
            i = stack.pop()
            if i in visited:
                continue
            visited.add(i)
            for j in cells:
                if j not in visited and connected(
                    i, j, n_rows, n_cols, diagonal_neighbors
                ):
                    stack.append(j)

        res = len(visited) == len(cells)
    return res


def get_island_cells(
    starting_cell_index: int,
    all_cells: Sequence[Sequence],
    empty=None,
    diagonal_neighbors: bool = True,
) -> tuple[tuple, tuple]:
    """Return values and indices of non-empty cells connected to the start cell.

    ``all_cells`` is a rectangular grid stored row by row. Index 0 is
    ``all_cells[0][0]``. Whether that row is the bottom or the top is
    decided by how the caller builds the grid.

    Args:
        starting_cell_index: Linear index of the cell to grow the island from.
        all_cells: Grid of cell values (e.g. ``((2, 3, None), ...)``).
        empty: Value(s) treated as empty. Default ``None`` means only ``None``
            is empty. Pass a collection to mark several values as empty.
        diagonal_neighbors: If True, cells sharing a corner are neighbors. If
            False, only edge-adjacent cells are neighbors.

    Returns:
        ``(values, indices)`` — parallel tuples of connected non-empty cell
        values and their indices. If the starting cell is out of range or
        empty, both tuples are empty.

    Examples:
        >>> import simetri.graphics as sg
        >>> grid = ((2, 3, None), (5, None, 6), (7, None, 4))
        >>> sg.get_island_cells(0, grid, diagonal_neighbors=False)
        ((2, 5, 7, 3), (0, 3, 6, 1))
        >>> sg.get_island_cells(8, grid, diagonal_neighbors=False)
        ((4, 6), (8, 5))
        >>> sg.get_island_cells(7, grid)
        ((), ())
    """
    n_rows = len(all_cells)
    if n_rows == 0:
        res = ((), ())
        return res
    n_cols = len(all_cells[0])
    n_cells = n_rows * n_cols

    def cell_empty(value) -> bool:
        if isinstance(empty, (tuple, list, set, frozenset)):
            return value in empty
        if empty is None:
            return value is None
        return value == empty

    def cells_neighbors(index1: int, index2: int) -> bool:
        row1, col1 = divmod(index1, n_cols)
        row2, col2 = divmod(index2, n_cols)
        if row2 < 0 or row2 >= n_rows or col2 < 0 or col2 >= n_cols:
            return False
        dr = abs(row1 - row2)
        dc = abs(col1 - col2)
        if dr > 1 or dc > 1 or (dr == 0 and dc == 0):
            return False
        if diagonal_neighbors:
            return True
        return dr + dc == 1

    if starting_cell_index < 0 or starting_cell_index >= n_cells:
        res = ((), ())
        return res

    start_row, start_col = divmod(starting_cell_index, n_cols)
    start_value = all_cells[start_row][start_col]
    if cell_empty(start_value):
        res = ((), ())
        return res

    visited: set[int] = set()
    stack = [starting_cell_index]
    indices_list: list[int] = []
    values_list: list = []

    while stack:
        i = stack.pop()
        if i in visited:
            continue
        row, col = divmod(i, n_cols)
        value = all_cells[row][col]
        if cell_empty(value):
            continue
        visited.add(i)
        indices_list.append(i)
        values_list.append(value)
        for j in range(n_cells):
            if j not in visited and cells_neighbors(i, j):
                stack.append(j)

    res = (tuple(values_list), tuple(indices_list))
    return res


def sort_points(points):
    """Return a new list of points sorted by x, then y.

    Args:
        points: Points to sort.

    Returns:
        list: Points in sorted order.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.sort_points([(2, 1), (1, 0), (1, 2)])
        [(1, 0), (1, 2), (2, 1)]
        >>> pts = [(2, 1), (0, 0)]
        >>> sg.sort_points(pts)
        [(0, 0), (2, 1)]
        >>> pts
        [(2, 1), (0, 0)]
    """
    return sorted(points, key=lambda p: (p[0], p[1]))


def time_it(func):
    """Decorator that prints how long ``func`` takes to run.

    Args:
        func: Callable to wrap.

    Returns:
        callable: Wrapped function that reports elapsed time.
    """

    @wraps(func)
    def time_it_wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.6f} seconds")
        return result

    return time_it_wrapper


def round_symmetric(n, inc):
    """Round ``n`` to a multiple of ``inc``.

    Positive numbers round away from zero. Negative numbers round
    away from zero.

    Args:
        n: Number to round.
        inc: Increment.

    Returns:
        The rounded number.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.round_symmetric(3.1, 2)
        4
        >>> sg.round_symmetric(-3.1, 2)
        -4
    """
    if n >= 0:
        return ceil(n / inc) * inc
    else:
        return floor(n / inc) * inc


def close_logger(logger):
    """Close the logger and remove all handlers.

    Args:
        logger (mutated): Logger whose handlers are closed and removed.
    """
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def get_file_path_with_rev(directory, script_path, ext=".pdf"):
    """Get the file path with a revision number.

    Args:
        directory: The directory to search for files.
        script_path: The script file path.
        ext: The file extension.

    Returns:
        The file path with a revision number.
    """

    # Get the file path of the script
    def get_rev_number(file_name):
        match = re.search(r"_\d+$", file_name)
        if match:
            rev = match.group()[1:]  # remove the underscore
            if rev is not None:
                return int(rev)
        return 0

    # script_path = __file__
    filename = os.path.basename(script_path)
    filename, _ = os.path.splitext(filename)
    # check if the file is in the current directory
    files = os.listdir(directory)
    file_names = [
        os.path.splitext(item)[0]
        for item in files
        if os.path.isfile(os.path.join(directory, item))
    ]
    existing = [item for item in file_names if item.startswith(filename)]
    if not existing:
        return os.path.join(directory, filename + ext)

    revs = [get_rev_number(file) for file in existing]
    if revs is None:
        rev = 1
    else:
        rev = max(revs) + 1

    return os.path.join(directory, f"{filename}_{rev}" + ext)


def remove_file_handler(logger, handler):
    """Remove a handler from a logger.

    Args:
        logger (mutated): Logger to remove the handler from.
        handler: Handler to close and remove.
    """
    logger.removeHandler(handler)
    handler.close()


def pretty_print_coords(coords: Sequence[PointType]) -> str:
    """Print the coordinates with a precision of 2.

    Args:
        coords: A sequence of PointType objects.

    Returns:
        A string representation of the coordinates.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.pretty_print_coords([(1, 2), (3.14159, 0)])
        '((1.00, 2.00), (3.14, 0.00))'
        >>> sg.pretty_print_coords([(0, 0)])
        '((0.00, 0.00))'
    """
    return (
        "("
        + ", ".join([f"({coord[0]:.2f}, {coord[1]:.2f})" for coord in coords])
        + ")"
    )


def is_file_empty(file_path):
    """Check if a file is empty.

    Args:
        file_path: The path to the file.

    Returns:
        True if the file is empty, False otherwise.
    """
    return os.path.getsize(file_path) == 0


def wait_for_file_availability(file_path, timeout=None, check_interval=1):
    """Check if a file is available for writing.

    Args:
        file_path: The path to the file.
        timeout: The timeout period in seconds.
        check_interval: The interval to check the file availability.

    Returns:
        True if the file is available, False otherwise.
    """
    start_time = monotonic()
    while True:
        try:
            # Attempt to open the file in write mode. This will raise an exception
            # if the file is currently locked or being written to.
            with open(file_path, "a", encoding="utf-8"):
                # If the file was successfully opened, it's available.
                return True
        except OSError:
            # The file is likely in use.
            if timeout is not None and (monotonic() - start_time) > timeout:
                # Timeout period elapsed.
                return False  # Or raise a TimeoutError if you prefer
            sleep(check_interval)
        except (TypeError, ValueError) as e:
            # Handle other potential exceptions (e.g., file not found) as needed
            print(f"An error occurred: {e}")
            return False


def random_characters(
    n=4, lower=True, upper=True, digit=False, exclude_chars=None
):
    """Return ``n`` random letters or digits as a string.

    Defaults are lowercase and uppercase letters, no digits.
    exclude_chars is a list of characters to be excluded.
    Usually l, 0, O are not desirable in variable names.
    Usually used for creating unique names.

    n = 3: 17576 unique lowercase or uppercase words
    n = 3: 140608 unique mixed-case words
    n = 3: 46656 unique lowercase or uppercase words and digits
    n = 3: 238328 unique mixed-case words and digits
    n = 4: 456976 unique lowercase or uppercase words
    n = 4: 7311616 unique mixed-case words
    n = 4: 1679616 unique lowercase or uppercase words and digits
    n = 4: 14776336 unique mixed-case words and digits
    n = 5: 11881376 unique lowercase or uppercase words
    n = 5: 380204032 unique mixed-case words
    n = 5: 60466176 unique lowercase or uppercase words and digits
    n = 5: 916132832 unique mixed-case words and digits
    n = 6: 308915776 unique lowercase or uppercase words
    n = 6: 19770609664 unique mixed-case words
    n = 6: 2176782336 unique lowercase or uppercase words and digits
    n = 6: 56800235584 unique mixed-case words and digits

    Examples:
        >>> import simetri.graphics as sg
        >>> excluded = ["l"]
        >>> token = sg.random_characters(
        ...     4, lower=True, upper=False, digit=False, exclude_chars=excluded
        ... )
        >>> len(token) == 4 and token.isalpha() and token.islower() and "l" not in token
        True
        >>> excluded
        ['l']
    """
    letters = string.ascii_letters
    uppers = string.ascii_uppercase
    lowers = string.ascii_lowercase
    digits = string.digits

    lowers_uppers_digits = letters + digits
    lowers_uppers = letters
    lowers_digits = lowers + digits
    uppers_digits = uppers + digits

    lookup = {
        (True, True, True): lowers_uppers_digits,
        (True, True, False): lowers_uppers,
        (True, False, True): lowers_digits,
        (True, False, False): lowers,
        (False, True, True): uppers_digits,
        (False, True, False): uppers,
        (False, False, True): digits,
    }

    characters = lookup[(lower, upper, digit)]
    if exclude_chars:
        characters = list(characters)
        for char in exclude_chars:
            try:
                characters.remove(char)
            except ValueError:
                issue_warning(f"{char} is not valid.")

    return "".join([random.choice(characters) for _ in range(n)])


def detokenize(text: str) -> str:
    """Replace the special Latex characters with their Latex commands.
    Inline math segments delimited by $ are preserved as-is.

    Args:
        text: The text to detokenize.

    Returns:
        The detokenized text.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.detokenize("a_b & c") == "a\\_b \\& c"
        True
        >>> sg.detokenize("a $b_c$ d")
        'a $b_c$ d'
    """
    replacements = {
        "\\": r"\textbackslash ",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_",
        "^": r"\^{}",
        "~": r"\textasciitilde{}",
    }

    def escape_plain_text(plain_text):
        for char, replacement in replacements.items():
            plain_text = plain_text.replace(char, replacement)
        return plain_text

    parts = text.split("$")
    if len(parts) == 1:
        return escape_plain_text(text)

    result_parts = []
    for index, part in enumerate(parts):
        if index % 2 == 0:
            result_parts.append(escape_plain_text(part))
        else:
            result_parts.append(f"${part}$")
    return "".join(result_parts)


def get_text_dimensions(text, font_path, font_size):
    """Return the width and height of the text.

    Args:
        text: The text to measure.
        font_path: The path to the font file.
        font_size: The size of the font.

    Returns:
        A tuple containing the width and height of the text.
    """
    font = ImageFont.truetype(font_path, font_size)
    _, descent = font.getmetrics()
    text_width = font.getmask(text).getbbox()[2]
    text_height = font.getmask(text).getbbox()[3] + descent
    return text_width, text_height


def function_module(func):
    """Return the module name of ``func``.

    Args:
        func: Function whose module is requested.

    Returns:
        str: Module name of ``func``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.function_module(sg.function_module)
        'simetri.helpers.utilities'
    """
    return inspect.getmodule(func).__name__


def timing(func):
    """Print the execution time of a function.

    Args:
        func: The function to time.

    Returns:
        The wrapped function.
    """

    @wraps(func)
    def wrap(*args, **kw):
        start_time = time()
        result = func(*args, **kw)
        end_time = time()
        elapsed_time = end_time - start_time
        print(f"function:{func.__name__} took: {elapsed_time:.4f} sec")

        return result

    return wrap


def grid_positions(
    rows: int,
    cols: int,
    cell_width: float,
    cell_height: float,
    pos: PointType,
    offset: PointType = (0, 0),
    page_height: float | None = None,
    from_top_left=True,
) -> Generator[PointType]:
    """Given number of rows and columns and row height and
    column width and an origin point, returns a generator of grid positions. If from_top_left is False then it starts from
    bottom right.

    Examples:
        >>> import simetri.graphics as sg
        >>> list(sg.grid_positions(2, 2, 10, 10, (0, 0), page_height=100))
        [(0, 100), (10, 100), (0, 90), (10, 90)]
    """

    width, height = cell_width, cell_height
    x, y = pos[:2]
    offset_x, offset_y = offset[:2]
    grid = []

    if from_top_left:
        if page_height is None:
            raise ValueError(
                "page_height is required when from_top_left is True"
            )
        row_indices = range(rows)
        col_indices = range(cols)
    else:
        row_indices = range(rows)
        col_indices = range(cols - 1, -1, -1)

    for row in row_indices:
        if from_top_left:
            y_row = page_height - y - offset_y - row * height
        else:
            y_row = y + offset_y + row * height
        for col in col_indices:
            x_col = col * width
            grid.append((x + offset_x + x_col, y_row))
    return (p for p in grid)


def find_nearest_value(values: array, value: float) -> float:
    """Find the closest value in an array to a given number.

    Args:
        values: A NumPy array.
        value: The number to find the closest value to.

    Returns:
        The closest value in the array to the given number.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.find_nearest_value([1, 4, 9], 5)
        4
    """
    arr = np.asarray(values)
    idx = (np.abs(arr - value)).argmin()

    return arr[idx]


def nested_count(nested_sequence):
    """Return the total number of items in a nested sequence.

    Args:
        nested_sequence: A nested sequence.

    Returns:
        The total number of items in the nested sequence.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.nested_count([1, [2, 3], (4,)])
        4
    """
    return sum(
        nested_count(item) if isinstance(item, (list, tuple, ndarray)) else 1
        for item in nested_sequence
    )


def decompose_transformations(transformation_matrix):
    """Decompose a 3x3 transformation matrix into translation, rotation, and scale components.

    Args:
        transformation_matrix: A 3x3 transformation matrix.

    Returns:
        A tuple containing the translation, rotation, and scale components.

    Examples:
        >>> import simetri.graphics as sg
        >>> import numpy as np
        >>> matrix = np.array([[1.0, 0, 0], [0, 1.0, 0], [3.0, 4.0, 1.0]])
        >>> translation, rotation, scale = sg.decompose_transformations(matrix)
        >>> list(translation), float(rotation), (float(scale[0]), float(scale[1]))
        ([3.0, 4.0], 0.0, (1.0, 1.0))
    """
    xform = transformation_matrix
    translation = xform[2, :2]
    rotation = np.arctan2(xform[0, 1], xform[0, 0])
    scale = np.linalg.norm(xform[:2, 0]), np.linalg.norm(xform[:2, 1])

    return translation, rotation, scale


def check_directory(dir_path):
    """Check if a directory is valid and writable.

    Args:
        dir_path: The path to the directory.

    Returns:
        A tuple containing a boolean indicating validity and an error message.
    """
    error_msg = []

    def dir_exists():
        nonlocal error_msg
        parent_dir = os.path.dirname(dir_path)
        if not os.path.exists(parent_dir):
            error_msg.append("Error! Parent directory doesn't exist")

    def is_writable():
        nonlocal error_msg
        parent_dir = os.path.dirname(dir_path)
        if not os.access(parent_dir, os.W_OK):
            error_msg.append("Error! Path is not writable.")

    dir_exists()
    is_writable()
    if error_msg:
        res = False, "\n".join(error_msg)
    else:
        res = True, ""

    return res


def analyze_path(file_path, overwrite):
    """Check if a file path is valid and writable.

    Args:
        file_path: The path to the file.
        overwrite: Whether to overwrite the file if it exists.

    Returns:
        A tuple containing a boolean indicating validity, the file extension, and an error message.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.analyze_path("notes.txt", True)
        (False, 'Error! Only .pdf, .svg, .ps, .eps, .tex supported.', '')
    """
    supported_types = (".pdf", ".svg", ".ps", ".eps", ".tex")
    error_msg = ""

    def is_writable():
        nonlocal error_msg
        parent_dir = os.path.dirname(file_path)
        if os.access(parent_dir, os.W_OK):
            res = True
        else:
            error_msg = "Error! Path is not writable."
            res = False

        return res

    def is_supported():
        nonlocal error_msg
        extension = Path(file_path).suffix
        if extension in supported_types:
            res = True
        else:
            error_msg = f"Error! Only {', '.join(supported_types)} supported."
            res = False

        return res

    def can_overwrite(overwrite):
        nonlocal error_msg
        if os.path.exists(file_path):
            if overwrite is None:
                overwrite = defaults["overwrite_files"]
            if overwrite:
                res = True
            else:
                error_msg = (
                    "Error! File exists. Use canvas."
                    "save(f_path, overwrite=True) to overwrite."
                )
                res = False
        else:
            res = True

        return res

    try:
        file_path = os.path.abspath(file_path)
        if is_writable() and is_supported() and can_overwrite(overwrite):
            res = (True, "", Path(file_path).suffix)
        else:
            res = (False, error_msg, "")

        return res
    except (
        OSError,
        TypeError,
        ValueError,
    ) as e:  # Million other ways a file path is not valid but life is short!
        return False, f"Path Error! {e}", ""


def can_be_xform_matrix(seq):
    """Check if a sequence can be converted to a transformation matrix.

    Args:
        seq: The sequence to check.

    Returns:
        True if the sequence can be converted to a transformation matrix, False otherwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.can_be_xform_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        True
        >>> sg.can_be_xform_matrix([1, 2, 3])
        False
    """
    # check if it is a sequence that can be
    # converted to a transformation matrix
    try:
        arr = array(seq)
        return is_xform_matrix(arr)
    except (TypeError, ValueError):
        return False


def is_sequence(value):
    """Check if a value is a sequence.

    Args:
        value: The value to check.

    Returns:
        True if the value is a sequence, False otherwise.
        Strings are not sequences for this check.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.is_sequence([1, 2])
        True
        >>> sg.is_sequence((1, 2))
        True
        >>> sg.is_sequence("ab")
        False
    """
    return isinstance(value, (list, tuple, ndarray))


def rel_coord(dx: float, dy: float, center: PointType) -> PointType:
    """Return the relative coordinates.

    Args:
        dx: The x-coordinate difference.
        dy: The y-coordinate difference.
        center: The center coordinates.

    Returns:
        The relative coordinates.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.rel_coord(2, 3, (10, 20))
        (12, 23)
    """
    return dx + center[0], dy + center[1]


def rel_polar(r: float, angle: float, center: PointType) -> PointType:
    """Return the coordinates.

    Args:
        r: The radius.
        angle: The angle in radians.
        center: The center coordinates.

    Returns:
        The coordinates.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.rel_polar(2, 0, (1, 1))
        (3.0, 1.0)
    """
    x, y = center[:2]
    x1 = x + r * cos(angle)
    y1 = y + r * sin(angle)

    return x1, y1


rc = rel_coord  # alias for rel_coord
rp = rel_polar  # alias for rel_polar


def axis(angle: float, length: float = 10) -> LineType:
    """Return a line [(x1, y1), (x2, y2)] with the given angle
    and length.
    Args:
        angle: The angle between the line and the x-axis, in radians.
        length: The length of the line.

    Returns:
        A line represented as a tuple of two points.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.axis(0, 10)
        ((5.0, 0.0), (-5.0, -0.0))
    """
    length2 = length / 2
    x1 = cos(angle) * length2
    y1 = sin(angle) * length2
    x2 = -x1
    y2 = -y1

    return (x1, y1), (x2, y2)


def flatten(points):
    """Flatten the points and return it as a list.

    Args:
        points: A sequence of points.

    Returns:
        A flattened list of points.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.flatten([(1, 2), (3, 4)])
        [1, 2, 3, 4]
    """
    if isinstance(points, set):
        points = list(points)
    if isinstance(points, np.ndarray):
        flat = list(points[:, :2].flatten())
    elif isinstance(points, collections.abc.Sequence):
        if isinstance(points[0], collections.abc.Sequence):
            flat = list(
                reduce(lambda x, y: x + y, [list(pnt[:2]) for pnt in points])
            )
        else:
            flat = list(points)
    else:
        raise TypeError("Error! Invalid data type.")

    return flat


def find_closest_value(a_sorted_list, value):
    """Return the index of the closest value and the value itself in a sorted list.

    Args:
        a_sorted_list: A sorted list of values.
        value: The value to find the closest match for.

    Returns:
        A tuple containing the closest value and its index.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.find_closest_value([1, 3, 8], 4)
        (3, 1)
        >>> sg.find_closest_value([1, 3, 8], 0)
        (1, 0)
        >>> sg.find_closest_value([1, 3, 8], 9)
        (8, 2)
    """
    ind = bisect_left(a_sorted_list, value)

    if ind == 0:
        return a_sorted_list[0], 0

    if ind == len(a_sorted_list):
        return a_sorted_list[-1], len(a_sorted_list) - 1

    left = a_sorted_list[ind - 1]
    right = a_sorted_list[ind]

    if right - value < value - left:
        return right, ind
    else:
        return left, ind - 1


def value_from_intervals(value, values, intervals):
    """Return the value from the intervals.
    Args:
        value: The value to find.
        values: The values to search.
        intervals: The intervals to search.
    Returns:
        The value from the intervals.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.value_from_intervals(2.5, ["a", "b", "c"], [1, 3, 5])
        'b'
    """

    return values[bisect_left(intervals, value)]


def get_transform(transform):
    """Return the transformation matrix.

    Args:
        transform: The transformation matrix or sequence.

    Returns:
        The transformation matrix.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.get_transform(None).tolist()
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    """
    if transform is None:
        # return identity
        res = array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    else:
        if is_xform_matrix(transform):
            res = transform
        elif can_be_xform_matrix(transform):
            res = array(transform)
        else:
            raise RuntimeError("Invalid transformation matrix!")
    return res


def is_numeric_numpy_array(array_):
    """Check if it is an array of numbers.

    Args:
        array_: The array to check.

    Returns:
        True if the array is numeric, False otherwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> import numpy as np
        >>> sg.is_numeric_numpy_array(np.array([1.0, 2.0]))
        True
        >>> sg.is_numeric_numpy_array([1, 2])
        False
    """
    if not isinstance(array_, np.ndarray):
        return False

    numeric_types = {
        "u",  # unsigned integer
        "i",  # signed integer
        "f",  # floating-point
        "c",
    }  # complex number
    try:
        return array_.dtype.kind in numeric_types
    except AttributeError:
        return False


def is_xform_matrix(matrix):
    """Check if it is a 3x3 transformation matrix.

    Args:
        matrix: The matrix to check.

    Returns:
        True if the matrix is a 3x3 transformation matrix, False otherwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> import numpy as np
        >>> sg.is_xform_matrix(np.eye(3))
        True
        >>> sg.is_xform_matrix(np.eye(2))
        False
    """
    return (
        is_numeric_numpy_array(matrix)
        and matrix.shape == (3, 3)
        and matrix.size == 9
    )


def prime_factors(n):
    """Return the prime factors of ``n``.

    Args:
        n: Positive integer to factorize.

    Returns:
        list: Prime factors of ``n`` (with multiplicity).

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.prime_factors(12)
        [2, 2, 3]
        >>> sg.prime_factors(1)
        []
    """
    factors = []
    p = 2
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n = n // p
        if p == 2:
            p += 1
        else:
            p += 2
    if n > 1:
        factors.append(n)

    return factors


def random_id():
    """Generate a random ID.

    Returns:
        A random ID string.

    Examples:
        >>> import simetri.graphics as sg
        >>> isinstance(sg.random_id(), str) and len(sg.random_id()) > 0
        True
    """
    return base64.b64encode(os.urandom(6)).decode("ascii")


def decompose_svg_transform(transform):
    """Decompose a SVG transformation string.

    Args:
        transform: The SVG transformation string.

    Returns:
        A tuple containing the decomposed transformation components.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.decompose_svg_transform((1, 0, 0, 1, 3, 4))
        (3, 4, 1.0, 1.0, 0.0)
    """
    a, b, c, d, e, f = transform
    # [[a, c, e],
    #  [b, d, f],
    #  [0, 0, 1]]
    dx = e
    dy = f

    sx = np.sign(a) * sqrt(a**2 + c**2)
    sy = np.sign(d) * sqrt(b**2 + d**2)

    angle = atan2(b, d)

    return dx, dy, sx, sy, angle


def abcdef_svg(transform_matrix):
    """Return the a, b, c, d, e, f for SVG transformations.

    Args:
        transform_matrix: A Numpy array representing the transformation matrix.

    Returns:
        A tuple containing the a, b, c, d, e, f components.

    Examples:
        >>> import simetri.graphics as sg
        >>> import numpy as np
        >>> sg.abcdef_svg(np.eye(3))
        (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    """
    # [[a, c, e],
    #  [b, d, f],
    #  [0, 0, 1]]
    a, b, _, c, d, _, e, f, _ = list(transform_matrix.flat)
    return (a, b, c, d, e, f)


def abcdef_pil(xform_matrix):
    """Return the a, b, c, d, e, f for PIL transformations.

    Args:
        xform_matrix: A Numpy array representing the transformation matrix.

    Returns:
        A tuple containing the a, b, c, d, e, f components.

    Examples:
        >>> import simetri.graphics as sg
        >>> import numpy as np
        >>> matrix = np.array([[1.0, 0, 0], [0, 1.0, 0], [3.0, 4.0, 1.0]])
        >>> sg.abcdef_pil(matrix)
        (1.0, 0.0, 3.0, 0.0, 1.0, 4.0)
    """
    a, d, _, b, e, _, c, f, _ = list(xform_matrix.flat)
    return (a, b, c, d, e, f)


def abcdef_reportlab(xform_matrix):
    """Return the a, b, c, d, e, f for Reportlab transformations.

    Args:
        xform_matrix: A Numpy array representing the transformation matrix.

    Returns:
        A tuple containing the a, b, c, d, e, f components.

    Examples:
        >>> import simetri.graphics as sg
        >>> import numpy as np
        >>> matrix = np.array([[1.0, 0, 0], [0, 1.0, 0], [3.0, 4.0, 1.0]])
        >>> sg.abcdef_reportlab(matrix)
        (1.0, 0.0, 0.0, 1.0, 3.0, 4.0)
    """
    # a, b, _, c, d, _, e, f, _ = list(np.transpose(xform_matrix).flat)
    a, b, _, c, d, _, e, f, _ = list(xform_matrix.flat)
    return (a, b, c, d, e, f)


def lerp(start, end, t):
    """Linear interpolation of two values.

    Args:
        start: The start value.
        end: The end value.
        t: The interpolation factor (0 <= t <= 1).

    Returns:
        The interpolated value.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.lerp(0, 10, 0.5)
        5.0
    """
    return start + t * (end - start)


def inv_lerp(start, end, value):
    """Inverse linear interpolation of two values.

    Args:
        start: The start value.
        end: The end value.
        value: The value to interpolate.

    Returns:
        The interpolation factor (0 <= t <= 1).

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.inv_lerp(0, 10, 4)
        0.4
    """
    return (value - start) / (end - start)


def zip_points(points1, points2):
    """Interleave two point sequences into one flat list.

    Args:
        points1: First sequence of points.
        points2: Second sequence of points.

    Returns:
        list: Alternating points from ``points1`` and ``points2``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.zip_points([(0, 0)], [(1, 1)])
        [(0, 0), (1, 1)]
    """
    res = []
    zipped = list(zip(points1, points2))
    for a, b in zipped:
        res.append(a)
        res.append(b)

    return res


def flatten2(nested_list):
    """Flatten a nested list.

    Args:
        nested_list: The nested list to flatten.

    Yields:
        The flattened elements.

    Examples:
        >>> import simetri.graphics as sg
        >>> list(sg.flatten2([1, [2, (3, 4)]]))
        [1, 2, 3, 4]
    """
    for i in nested_list:
        if isinstance(i, (list, tuple)):
            yield from flatten2(i)
        else:
            yield i


def round2(n: float, cutoff: int = 25) -> int:
    """Round a number to the nearest multiple of cutoff.

    Args:
        n: The number to round.
        cutoff: The cutoff value.

    Returns:
        The rounded number.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.round2(30)
        25
        >>> sg.round2(10)
        0
    """
    return cutoff * round(n / cutoff)


def is_nested_sequence(value):
    """Check if a value is a nested sequence.

    Args:
        value: The value to check.

    Returns:
        True if the value is a nested sequence, False otherwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.is_nested_sequence([(1, 2), [3, 4]])
        True
        >>> sg.is_nested_sequence([1, 2])
        False
    """
    if not isinstance(value, (list, tuple, ndarray)):
        return False  # Not a sequence

    for item in value:
        if not isinstance(item, (list, tuple, ndarray)):
            return False  # At least one element is not a sequence

    return True  # All elements are sequences


def group_into_bins(values, delta, compare_function=None, with_objects=True):
    """Group values into bins.

    Args:
        values: A list of (value, obj) pairs, or just values.
        delta: Bin size tolerance value.
        with_objects: If True, values are (value, obj) pairs.

    Returns:
        A list of bins.

    Examples:
        >>> import simetri.graphics as sg
        >>> values = [10, 1, 2]
        >>> sg.group_into_bins(values, 2, with_objects=False)
        [[1, 2], [10]]
        >>> values
        [10, 1, 2]
        >>> sg.group_into_bins([(1, "a"), (2, "b"), (10, "c")], 2)
        [[(1, 'a'), (2, 'b')], [(10, 'c')]]
    """
    if compare_function:
        values = sorted(values, key=cmp_to_key(compare_function))
    else:
        values = sorted(values)
    bins = []
    bin_ = [values[0]]
    if with_objects:
        for value in values[1:]:
            if value[0] - bin_[0][0] <= delta:
                bin_.append(value)
            else:
                bins.append(bin_)
                bin_ = [value]
        bins.append(bin_)
    else:
        for value in values[1:]:
            if value - bin_[0] <= delta:
                bin_.append(value)
            else:
                bins.append(bin_)
                bin_ = [value]
        bins.append(bin_)

    return bins


def equal_cycles(
    cycle1: list[float], cycle2: list[float], rel_tol=None, abs_tol=None
) -> bool:
    """Check if two cycles are circularly equal.

    Args:
        cycle1: The first cycle.
        cycle2: The second cycle.
        rel_tol: The relative tolerance.
        abs_tol: The absolute tolerance.

    Returns:
        True if the cycles are circularly equal, False otherwise.

    Examples:
        >>> import simetri.graphics as sg
        >>> cycle = [1.0, 2.0, 3.0]
        >>> sg.equal_cycles(cycle, [3.0, 1.0, 2.0], rel_tol=0, abs_tol=1e-9)
        True
        >>> cycle
        [1.0, 2.0, 3.0]
        >>> sg.equal_cycles([1.0, 2.0], [1.0, 3.0], rel_tol=0, abs_tol=1e-9)
        False
        >>> sg.equal_cycles([1.0, 2.0, 3.0], [3.0, 1.0, 2.0])
        True
    """
    if rel_tol is None:
        rel_tol = defaults["rel_tol"]

    if abs_tol is None:
        abs_tol = defaults["abs_tol"]

    def check_cycles(cyc1, cyc2, rel_tol):
        for i, val in enumerate(cyc1):
            if not isclose(val, cyc2[i], rel_tol=rel_tol, abs_tol=abs_tol):
                return False
        return True

    len_cycle1 = len(cycle1)
    len_cycle2 = len(cycle2)
    if len_cycle1 != len_cycle2:
        return False
    cycle1 = cycle1[:]
    cycle1.extend(cycle1)
    for i in range(len_cycle1):
        if check_cycles(cycle2, cycle1[i : i + len_cycle2], rel_tol):
            return True

    return False


def map_ranges(
    value: float,
    range1_min: float,
    range1_max: float,
    range2_min: float,
    range2_max: float,
) -> float:
    """Map a value from one range to another.

    Args:
        value: The value to map.
        range1_min: The minimum of the first range.
        range1_max: The maximum of the first range.
        range2_min: The minimum of the second range.
        range2_max: The maximum of the second range.

    Returns:
        The mapped value.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.map_ranges(5, 0, 10, 0, 100)
        50.0
    """
    delta1 = range1_max - range1_min
    delta2 = range2_max - range2_min
    return (value - range1_min) / delta1 * delta2 + range2_min


def binomial(n: int, k: int) -> int:
    """Calculate the binomial coefficient.

    Args:
        n: The number of trials.
        k: The number of successes.

    Returns:
        The binomial coefficient.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.binomial(5, 2)
        10
    """
    if k == 0:
        res = 1
    else:
        res = factorial(n) / (factorial(k) * factorial(n - k))

    return int(res)


def n_permutations(n: int, k: int) -> int:
    """Calculate the binomial coefficient.

    Args:
        n: The sequence length.
        k: The number of distinct elements.

    Returns:
        The number of nPk permutations.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.n_permutations(5, 2)
        20
    """

    return int(factorial(n) / factorial(n - k))


def catalan(n):
    """Calculate the nth Catalan number.

    Args:
        n: The index of the Catalan number.

    Returns:
        The nth Catalan number.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.catalan(0)
        1
        >>> sg.catalan(3)
        5.0
    """
    if n <= 1:
        res = 1
    else:
        res = factorial(2 * n) / (factorial(n + 1) * factorial(n))
    return res


def solve_quadratic_eq(a, b, c, abs_tolerance=1e-5):
    """Solve ``ax^2 + bx + c = 0``.

    Args:
        a: Quadratic coefficient.
        b: Linear coefficient.
        c: Constant term.
        abs_tolerance (float, optional): Tolerance for treating the
            discriminant as zero. Defaults to ``1e-5``.

    Returns:
        list: Real roots (empty, one, or two values).

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.solve_quadratic_eq(1, -3, 2)
        [2.0, 1.0]
        >>> sg.solve_quadratic_eq(1, 0, 1)
        []
    """

    discr = b**2 - (4 * a * c)  # discriminant

    if discr < 0:
        res = []
    elif isclose(discr, 0, rel_tol=0, abs_tol=abs_tolerance):
        # one solution
        res = [(-b + discr) / (2 * a)]
    else:
        a2 = a * 2
        sqrt_discr = sqrt(discr)
        x1 = (-b + sqrt_discr) / a2
        x2 = (-b - sqrt_discr) / a2
        res = [x1, x2]

    return res


def solve_quartic_eq(
    a: float, b: float, c: float, d: float, e: float
) -> list[float]:
    """
    Solves a quartic equation of the form ax^4 + bx^3 + cx^2 + dx + e = 0.

    Args:
        a: The coefficient of x^4.
        b: The coefficient of x^3.
        c: The coefficient of x^2.
        d: The coefficient of x.
        e: The constant term.

    Returns:
        A numpy array containing the four roots of the equation.

    Examples:
        >>> import simetri.graphics as sg
        >>> sorted(round(float(root), 10) for root in sg.solve_quartic_eq(1, 0, -5, 0, 4))
        [-2.0, -1.0, 1.0, 2.0]
    """

    return np.roots((a, b, c, d, e)).tolist()


def solve_complex_quadratic_eq(
    a: complex, b: complex, c: complex, tolerance: float = 1e-5
) -> list[complex]:
    """Solves a quadratic equation of the form ax^2 + bx + c = 0,
    where a, b, and c can be complex numbers.

    Args:
        a: The complex coefficient of x^2.
        b: The complex coefficient of x.
        c: The complex constant term.
        tolerance: The tolerance for floating-point comparisons.

    Returns:
        list: The two roots. A repeated root is returned once.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.solve_complex_quadratic_eq(1 + 0j, -3 + 0j, 2 + 0j)
        [(1+0j), (2+0j)]
    """
    discr = (b**2) - 4 * (a * c)  # discriminant
    a2 = a * 2
    if abs(discr) <= tolerance:
        roots = [-b / a2]
    else:
        sqrt_discr = cmath.sqrt(discr)
        roots = [(-b - sqrt_discr) / a2, (-b + sqrt_discr) / a2]
    return [complex(root.real + 0.0, root.imag + 0.0) for root in roots]


def get_function_dependencies(func):
    """Extract called-name dependencies of a function via AST parsing.

    Args:
        func: Function object whose source is inspected.

    Returns:
        set: Names of functions/attributes referenced in calls.

    Examples:
        >>> import simetri.graphics as sg
        >>> sorted(sg.get_function_dependencies(sg.prime_factors))
        [('ctx', 'factors'), ('ctx', 'n'), ('ctx', 'p')]
    """
    source = inspect.getsource(func)
    tree = ast.parse(source)
    dependencies = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            dependencies.add(("ctx", node.id))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                dependencies.add(("import", alias.name))
        elif isinstance(node, ast.ImportFrom):
            dependencies.add(("module", node.module))

    # Remove the function name itself from dependencies
    dependencies.discard(func.__name__)
    # Remove built-in names
    dependencies.discard("print")  # Add more built-in names if needed

    return list(dependencies)


def analyze_function_dependencies(func):
    """
    Analyzes a function's dependencies, separating arguments, function calls, and variables.

    Args:
        func: The function to analyze.

    Returns:
        A dictionary containing lists of arguments, function calls, and variables.

    Examples:
        >>> import simetri.graphics as sg
        >>> info = sg.analyze_function_dependencies(sg.prime_factors)
        >>> info["arguments"]
        ['n']
        >>> info["function_calls"]
        ['factors.append', 'factors.append']
    """

    source_code = inspect.getsource(func)
    tree = ast.parse(source_code)

    arguments = []
    function_calls = []
    variables = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            arguments.extend([arg.arg for arg in node.args.args])
        elif isinstance(node, ast.Call):
            function_calls.append(ast.unparse(node.func))
        elif (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in arguments
        ):
            variables.append(node.id)
    # frame = None
    # try:
    #     # Use inspect.trace to get the frame of the function execution
    #     trace = inspect.trace(lambda: func(*args, **kwargs), context=0)
    #     frame = trace[0][0]

    #     # Extract local and global variables from the frame
    #     local_vars = frame.f_locals
    #     global_vars = frame.f_globals
    # finally:
    #     # Ensure the frame is deleted to avoid resource leaks
    #     del frame
    code = func.__code__
    local_vars = code.co_varnames[: code.co_argcount + code.co_nlocals]
    global_names = [name for name in func.__globals__ if name in code.co_names]
    # get the types of local_vars and make a dictionary d_local_vars = {name: type}
    d_local_vars = {
        name: type(func.__code__.co_varnames)
        for name in local_vars
        if name in func.__code__.co_varnames
    }
    # get the types of global_vars and make a dictionary d_global_vars = {name: type}
    d_global_names = {
        name: type(func.__globals__[name])
        for name in global_names
        if name in func.__globals__
    }

    return {
        "local_vars": local_vars,
        "global_vars": global_names,
        "d_local_vars": d_local_vars,
        "d_global_vars": d_global_names,
        "arguments": arguments,
        "function_calls": function_calls,
        "variables": list(
            set(variables) - set(function_calls)
        ),  # Remove function calls that are also variables
    }


def get_local_variables_info(func, *args, **kwargs):
    """
    Inspects a function call to retrieve local variable names and types
    without modifying the function's execution.

    Args:
        func: The function to inspect.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        A dictionary where keys are local variable names and values are their types.

    Examples:
        >>> import simetri.graphics as sg
        >>> def add(x, y):
        ...     total = x + y
        ...     return total
        >>> sg.get_local_variables_info(add, 2, 3)
        {'x': 'int', 'y': 'int', 'total': 'int', 'return': 'int'}
    """

    captured = {}

    def tracer(frame, event, arg):
        if frame.f_code is not func.__code__:
            return tracer
        if event == "return":
            captured.clear()
            captured.update(
                {
                    name: type(value).__name__
                    for name, value in frame.f_locals.items()
                }
            )
            captured["return"] = type(arg).__name__
        return tracer

    sys.settrace(tracer)
    try:
        func(*args, **kwargs)
    finally:
        sys.settrace(None)
    return captured


def best_fit_exponent(pairs):
    """Return the best-fit exponent ``p`` in ``T(n) = k * n^p``.

    Fits a line to ``log(time)`` versus ``log(n)``. Pairs with a time of
    zero or less are dropped so the logarithm is defined.

    Args:
        pairs: Sequence of ``(n, time)`` pairs.

    Returns:
        float: Slope of the log-log fit.

    Examples:
        >>> import simetri.graphics as sg
        >>> round(sg.best_fit_exponent([(10, 1.0), (100, 100.0)]), 10)
        2.0
    """
    pairs = np.array(pairs, dtype=float)
    n = pairs[:, 0]
    t = pairs[:, 1]

    mask = t > 0
    n = n[mask]
    t = t[mask]

    logn = np.log(n)
    logt = np.log(t)

    exponent, _ = np.polyfit(logn, logt, 1)

    return exponent

"""Simetri graphics library's utility functions."""

import collections
import os
import re
import base64
import cmath
from functools import wraps, reduce
from time import time, monotonic, perf_counter
from math import factorial, cos, sin, pi, atan2, sqrt
from pathlib import Path
from bisect import bisect_left

from typing import Sequence

from PIL import ImageFont
from numpy import array, ndarray
import numpy as np
from numpy import isclose

from ..settings.settings import defaults
from ..graphics.common import get_defaults, Point


def time_it(func):
    """Decorator to time a function"""

    @wraps(func)
    def time_it_wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = func(*args, **kwargs)
        end_time = perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.6f} seconds")
        return result

    return time_it_wrapper


def close_logger(logger):
    """Close the logger and remove all handlers.

    Args:
        logger: The logger instance to close.
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
    else:
        revs = [get_rev_number(file) for file in existing]
        if revs is None:
            rev = 1
        else:
            rev = max(revs) + 1

        return os.path.join(directory, f"{filename}_{rev}" + ext)


def remove_file_handler(logger, handler):
    """Remove a handler from a logger.

    Args:
        logger: The logger instance.
        handler: The handler to remove.
    """
    logger.removeHandler(handler)
    handler.close()


def pretty_print_coords(coords: Sequence[Point]) -> str:
    """Print the coordinates with a precision of 2.

    Args:
        coords: A sequence of Point objects.

    Returns:
        A string representation of the coordinates.
    """
    return (
        "(" + ", ".join([f"({coord[0]:.2f}, {coord[1]:.2f})" for coord in coords]) + ")"
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
        except IOError:
            # The file is likely in use.
            if timeout is not None and (monotonic() - start_time) > timeout:
                # Timeout period elapsed.
                return False  # Or raise a TimeoutError if you prefer
            time.sleep(check_interval)
        except Exception as e:
            # Handle other potential exceptions (e.g., file not found) as needed
            print(f"An error occurred: {e}")
            return False


def detokenize(text: str) -> str:
    """Replace the special Latex characters with their Latex commands.

    Args:
        text: The text to detokenize.

    Returns:
        The detokenized text.
    """
    if text.startswith("$") and text.endswith("$"):
        res = text
    else:
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
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
            res = text

    return res


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


def find_nearest_value(values: array, value: float) -> float:
    """Find the closest value in an array to a given number.

    Args:
        values: A NumPy array.
        value: The number to find the closest value to.

    Returns:
        The closest value in the array to the given number.
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
        Exception
    ) as e:  # Million other ways a file path is not valid but life is short!
        return False, f"Path Error! {e}", ""


def can_be_xform_matrix(seq):
    """Check if a sequence can be converted to a transformation matrix.

    Args:
        seq: The sequence to check.

    Returns:
        True if the sequence can be converted to a transformation matrix, False otherwise.
    """
    # check if it is a sequence that can be
    # converted to a transformation matrix
    try:
        arr = array(seq)
        return is_xform_matrix(arr)
    except Exception:
        return False


def is_sequence(value):
    """Check if a value is a sequence.

    Args:
        value: The value to check.

    Returns:
        True if the value is a sequence, False otherwise.
    """
    return isinstance(value, (list, tuple, array))


def rel_coord(dx: float, dy: float, origin):
    """Return the relative coordinates.

    Args:
        dx: The x-coordinate difference.
        dy: The y-coordinate difference.
        origin: The origin coordinates.

    Returns:
        The relative coordinates.
    """
    return dx + origin[0], dy + origin[1]


def rel_polar(r: float, angle: float, origin):
    """Return the coordinates.

    Args:
        r: The radius.
        angle: The angle in radians.
        origin: The origin coordinates.

    Returns:
        The coordinates.
    """
    x, y = origin[:2]
    x1 = x + r * cos(angle)
    y1 = y + r * sin(angle)

    return x1, y1


rc = rel_coord  # alias for rel_coord
rp = rel_polar  # alias for rel_polar


def flatten(points):
    """Flatten the points and return it as a list.

    Args:
        points: A sequence of points.

    Returns:
        A flattened list of points.
    """
    if isinstance(points, set):
        points = list(points)
    if isinstance(points, np.ndarray):
        flat = list(points[:, :2].flatten())
    elif isinstance(points, collections.abc.Sequence):
        if isinstance(points[0], collections.abc.Sequence):
            flat = list(reduce(lambda x, y: x + y, [list(pnt[:2]) for pnt in points]))
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
    """
    ind = bisect_left(a_sorted_list, value)

    if ind == 0:
        return a_sorted_list[0]

    if ind == len(a_sorted_list):
        return a_sorted_list[-1]

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
    """

    return values[bisect_left(intervals, value)]


def get_transform(transform):
    """Return the transformation matrix.

    Args:
        transform: The transformation matrix or sequence.

    Returns:
        The transformation matrix.
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
    """
    return (
        is_numeric_numpy_array(matrix) and matrix.shape == (3, 3) and matrix.size == 9
    )


def prime_factors(n):
    """Prime factorization.

    Args:
        n: The number to factorize.

    Returns:
        A list of prime factors.
    """
    p = 2
    factors = []
    while n > 1:
        if n % p:
            p += 1
        else:
            factors.append(p)
            n = n / p
    return factors


def random_id():
    """Generate a random ID.

    Returns:
        A random ID string.
    """
    return base64.b64encode(os.urandom(6)).decode("ascii")


def decompose_svg_transform(transform):
    """Decompose a SVG transformation string.

    Args:
        transform: The SVG transformation string.

    Returns:
        A tuple containing the decomposed transformation components.
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
    """
    a, d, _, b, e, _, c, f, _ = list(xform_matrix.flat)
    return (a, b, c, d, e, f)


def abcdef_reportlab(xform_matrix):
    """Return the a, b, c, d, e, f for Reportlab transformations.

    Args:
        xform_matrix: A Numpy array representing the transformation matrix.

    Returns:
        A tuple containing the a, b, c, d, e, f components.
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
    """
    return (value - start) / (end - start)


def sanitize_weighted_graph_edges(edges):
    """Sanitize weighted graph edges.

    Args:
        edges: A list of weighted graph edges.

    Returns:
        A sanitized list of weighted graph edges.
    """
    clean_edges = []
    s_seen = set()
    for edge in edges:
        e1, e2, _ = edge
        frozen_edge = frozenset((e1, e2))
        if frozen_edge in s_seen:
            continue
        s_seen.add(frozen_edge)
        clean_edges.append(edge)
    clean_edges.sort()
    return clean_edges


def sanitize_graph_edges(edges):
    """Sanitize graph edges.

    Args:
        edges: A list of graph edges.

    Returns:
        A sanitized list of graph edges.
    """
    s_edge_set = set()
    for edge in edges:
        s_edge_set.add(frozenset(edge))
    edges = [tuple(x) for x in s_edge_set]
    edges = [(min(x), max(x)) for x in edges]
    edges.sort()
    return edges


def flatten2(nested_list):
    """Flatten a nested list.

    Args:
        nested_list: The nested list to flatten.

    Yields:
        The flattened elements.
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
    """
    return cutoff * round(n / cutoff)


def is_nested_sequence(value):
    """Check if a value is a nested sequence.

    Args:
        value: The value to check.

    Returns:
        True if the value is a nested sequence, False otherwise.
    """
    if not isinstance(value, (list, tuple, ndarray)):
        return False  # Not a sequence

    for item in value:
        if not isinstance(item, (list, tuple, ndarray)):
            return False  # At least one element is not a sequence

    return True  # All elements are sequences


def group_into_bins(values, delta):
    """Group values into bins.

    Args:
        values: A list of numbers.
        delta: The bin size.

    Returns:
        A list of bins.
    """
    values.sort()
    bins = []
    bin_ = [values[0]]
    for value in values[1:]:
        if value[0] - bin_[0][0] <= delta:
            bin_.append(value)
        else:
            bins.append(bin_)
            bin_ = [value]
    bins.append(bin_)
    return bins


def equal_cycles(
    cycle1: list[float], cycle2: list[float], rtol=None, atol=None
) -> bool:
    """Check if two cycles are circularly equal.

    Args:
        cycle1: The first cycle.
        cycle2: The second cycle.
        rtol: The relative tolerance.
        atol: The absolute tolerance.

    Returns:
        True if the cycles are circularly equal, False otherwise.
    """
    rtol, atol = get_defaults(["rtol", "atol"], [rtol, atol])

    def check_cycles(cyc1, cyc2, rtol=defaults["rtol"]):
        for i, val in enumerate(cyc1):
            if not isclose(val, cyc2[i], rtol=rtol, atol=atol):
                return False
        return True

    len_cycle1 = len(cycle1)
    len_cycle2 = len(cycle2)
    if len_cycle1 != len_cycle2:
        return False
    cycle1 = cycle1[:]
    cycle1.extend(cycle1)
    for i in range(len_cycle1):
        if check_cycles(cycle2, cycle1[i : i + len_cycle2], rtol):
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
    """
    delta1 = range1_max - range1_min
    delta2 = range2_max - range2_min
    return (value - range1_min) / delta1 * delta2 + range2_min


def binomial(n, k):
    """Calculate the binomial coefficient.

    Args:
        n: The number of trials.
        k: The number of successes.

    Returns:
        The binomial coefficient.
    """
    if k == 0:
        res = 1
    else:
        res = factorial(n) / (factorial(k) * factorial(n - k))
    return res


def catalan(n):
    """Calculate the nth Catalan number.

    Args:
        n: The index of the Catalan number.

    Returns:
        The nth Catalan number.
    """
    if n <= 1:
        res = 1
    else:
        res = factorial(2 * n) / (factorial(n + 1) * factorial(n))
    return res


def reg_poly_points(pos: Point, n: int, r: float) -> Sequence[Point]:
    """Return a regular polygon points list with n sides, r radius, and pos center.

    Args:
        pos: The center position of the polygon.
        n: The number of sides.
        r: The radius.

    Returns:
        A sequence of points representing the polygon.
    """
    angle = 2 * pi / n
    x, y = pos[:2]
    points = [[cos(angle * i) * r + x, sin(angle * i) * r + y] for i in range(n)]
    points.append(points[0])
    return points


def solve_quadratic_eq(a, b, c, tolerance=1e-5):
    """Solves a quadratic equation of the form ax^2 + bx + c = 0."""

    discr = b**2 - (4 * a * c)  # discriminant

    if discr < 0:
        res = []
    elif isclose(discr, 0, rtol=0, atol=tolerance):
        # one solution
        res = [(-b + discr) / (2 * a)]
    else:
        a2 = a * 2
        sqrt_discr = sqrt(discr)
        x1 = (-b + sqrt_discr) / a2
        x2 = (-b - sqrt_discr) / a2
        res = [x1, x2]

    return res


def solve_quartic_eq(a: float, b: float, c: float, d: float, e: float) -> list[float]:
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
    """
    discr = (b**2) - 4 * (a * c)  # discriminant
    a2 = a * 2
    if isclose(discr, 0, rtol=0, atol=tolerance):
        x1 = -b / a2
        res = [x1]
    elif discr > 0:
        sqrt_discr = sqrt(discr)
        x1 = (-b - sqrt_discr) / a2
        x2 = (-b + sqrt_discr) / a2
        res = [x1, x2]
    else:
        sqrt_discr = cmath.sqrt(discr)
        x1 = (-b - sqrt_discr) / a2
        x2 = (-b + sqrt_discr) / a2
        res = [x1, x2]

    return res

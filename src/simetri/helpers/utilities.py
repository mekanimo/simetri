"""Simetri graphics library's utility functions."""

import collections
import os
import base64
from functools import wraps, reduce
from time import time, monotonic
from math import factorial, cos, sin, pi
from pathlib import Path

from typing import Sequence

from PIL import ImageFont
from numpy import array, ndarray
import numpy as np
from numpy import isclose

from ..settings.settings import defaults
from ..graphics.common import get_defaults, Point


def close_logger(logger):
    """Close the logger and remove all handlers."""
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def remove_file_handler(logger, handler):
    """Used for individually removing a handler from a logger.
    Example: remove_file_handler(logger, file_handler)
    """
    logger.removeHandler(handler)
    handler.close()


def pretty_print_coords(coords: Sequence[Point]) -> str:
    """Print the coordinates with a precision of 2"""
    return (
        "(" + ", ".join([f"({coord[0]:.2f}, {coord[1]:.2f})" for coord in coords]) + ")"
    )


def is_file_empty(file_path):
    """Check if a file is empty."""
    return os.path.getsize(file_path) == 0


def wait_for_file_availability(file_path, timeout=None, check_interval=1):
    """Check if a file is available for writing."""
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


# replace the special Latex characters with their Latex commands
def detokenize(text: str) -> str:
    """Replace the special Latex characters with their Latex commands."""
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
    """Return the width and height of the text."""
    font = ImageFont.truetype(font_path, font_size)
    _, descent = font.getmetrics()
    text_width = font.getmask(text).getbbox()[2]
    text_height = font.getmask(text).getbbox()[3] + descent
    return text_width, text_height


# Example usage
# font_path = "path/to/your/font.ttf"  # Replace with the path to your font file
# font_size = 12
# text = "Hello, TeX!"
# width, height = get_text_dimensions(text, font_path, font_size)
# print(f"Text width: {width}, Text height: {height}")


def timing(func):
    """Print the execution time of a function."""

    @wraps(func)
    def wrap(*args, **kw):
        start_time = time()
        result = func(*args, **kw)
        end_time = time()
        elapsed_time = end_time - start_time
        print(f"function:{func.__name__} took: {elapsed_time:.4f} sec")

        return result

    return wrap


def nested_count(nested_sequence):
    """Return the total number of items in a nested sequence."""
    return sum(
        nested_count(item) if isinstance(item, (list, tuple, ndarray)) else 1
        for item in nested_sequence
    )


def decompose_transformations(transformation_matrix):
    """
    Decompose a 3x3 transformation matrix into translation, rotation,
    and scale components.
    This function uses row major order, which is consistent with our convention.
    """
    xform = transformation_matrix
    translation = xform[2, :2]
    rotation = np.arctan2(xform[0, 1], xform[0, 0])
    scale = np.linalg.norm(xform[:2, 0]), np.linalg.norm(xform[:2, 1])

    return translation, rotation, scale


def check_directory(dir_path):
    """Check if a directory is valid and writable."""
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
    """Given a file_path, check if it is valid and writeable.
    If file_path is valid return: (True, extension, "")
    If file_path is invalid return: (False, "", error message)
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
    """Check if a sequence can be converted to a transformation matrix."""
    # check if it is a sequence that can be
    # converted to a transformation matrix
    try:
        arr = array(seq)
        return is_xform_matrix(arr)
    except Exception:
        return False


def is_sequence(value):
    """Check if a value is a sequence."""
    return isinstance(value, (list, tuple, array))


def flatten(points):
    """points can be: sequences, sequence of sequences, set of
    sequences, or arrays.
    Flatten the points and return it as a list.
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


def get_transform(transform):
    """Return the transformation matrix."""
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
    """Check if it is an array of numbers"""
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
    """Check if it is a 3x3 transformation matrix."""
    return (
        is_numeric_numpy_array(matrix) and matrix.shape == (3, 3) and matrix.size == 9
    )


def prime_factors(n):
    """Prime factorization."""
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
    """If you need a close to 100% unique ID, use UUID.
    This should be good enough for small number of objects.
    ID function in common.py can be used for temporary unique IDs that can be
    used in numpy arrays.
    """
    return base64.b64encode(os.urandom(6)).decode("ascii")


def abcdef_pil(xform_matrix):
    """xform_matrix is a Numpy array.
    Return the a, b, c, d, e, f for PIL transformations."""
    a, d, _, b, e, _, c, f, _ = list(xform_matrix.flat)
    return (a, b, c, d, e, f)


def abcdef_reportlab(xform_matrix):
    """xform_matrix is a Numpy array.
    Return the a, b, c, d, e, f for Reportlab transformations."""
    # a, b, _, c, d, _, e, f, _ = list(np.transpose(xform_matrix).flat)
    a, b, _, c, d, _, e, f, _ = list(xform_matrix.flat)
    return (a, b, c, d, e, f)


def lerp(start, end, t):
    """Linear interpolation of two values.
    start: number
    end: number
     0 <= t <= 1."""
    return start + t * (end - start)


def sanitize_weighted_graph_edges(edges):
    """edges: [(node1, node2), ...]"""
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
    """edges: [(node1, node2), ...]"""
    s_edge_set = set()
    for edge in edges:
        s_edge_set.add(frozenset(edge))
    edges = [tuple(x) for x in s_edge_set]
    edges = [(min(x), max(x)) for x in edges]
    edges.sort()
    return edges


def flatten2(nested_list):
    """Flatten a nested list."""
    for i in nested_list:
        if isinstance(i, (list, tuple)):
            yield from flatten2(i)
        else:
            yield i


def round2(n: float, cutoff: int = 25) -> int:
    """
    Round a number to the nearest multiple of cutoff.
    This is useful for grouping numbers into bins.
    canvas.draw uses this to color fragments of lace objects.
    """
    return cutoff * round(n / cutoff)


def is_nested_sequence(value):
    """Check if a value is a nested sequence."""
    if not isinstance(value, (list, tuple, ndarray)):
        return False  # Not a sequence

    for item in value:
        if not isinstance(item, (list, tuple, ndarray)):
            return False  # At least one element is not a sequence

    return True  # All elements are sequences


def group_into_bins(values, delta):
    """values: list of numbers
    delta: bin size
    return: list of bins
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
    """
    cycle1: list of floats
    cycle2: list of floats
    return: True if cycles are circularly equal, False otherwise
    [1., 2., 3.] == [2., 3., 1.] == [3., 1., 2.]
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
    It doesn't clamp the value to the range."""
    delta1 = range1_max - range1_min
    delta2 = range2_max - range2_min
    return (value - range1_min) / delta1 * delta2 + range2_min


def binomial(n, k):
    """Binomial coefficient.
    n: number of trials
    k: number of successes
    n choose k"""
    if k == 0:
        res = 1
    else:
        res = factorial(n) / (factorial(k) * factorial(n - k))
    return res


def catalan(n):
    """Catalan numbers"""
    if n <= 1:
        res = 1
    else:
        res = factorial(2 * n) / (factorial(n + 1) * factorial(n))
    return res


def reg_poly_points(pos: Point, n: int, r: float) -> Sequence[Point]:
    """
    Return a regular polygon points list with n sides, r radius, and pos center.
    """

    angle = 2 * pi / n
    x, y = pos[:2]
    points = [[cos(angle * i) * r + x, sin(angle * i) * r + y] for i in range(n)]
    points.append(points[0])
    return points

"""Shared constants, type aliases, and ID helpers for Simetri graphics.

Unit constants convert physical lengths to PostScript points (1 inch = 72 pt).
Type aliases such as ``PointType`` and ``LineType`` are used throughout the
graphics and geometry APIs.

Examples:
    >>> import simetri.graphics as sg
    >>> sg.INCH, round(sg.CM, 4), round(sg.phi, 12)
    (72, 28.3464, 1.61803398875)
    >>> 2 * sg.INCH
    144
"""

from __future__ import annotations

from functools import wraps
from collections.abc import Callable, Iterator, Mapping, Sequence
from math import cos, pi, sin
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from ..shapes.shape import Shape

# Not sg.defaults (factory-only). temp, then user, then factory.
from ..config.settings import runtime_defaults

# These are used for type hinting and annotations
GraphEdgeType = tuple[int, int]
LineType = Sequence[Sequence]
MatrixType = Sequence[Sequence[float]]
PointType = Sequence[float]
PolygonLike = Union["Shape", Sequence[PointType]]
PolygonType = Sequence[PointType]
PolylineType = Sequence[PointType]
TurnPair = tuple[float, float]
TurnSequence = Sequence[TurnPair]
VecType = Sequence[float]

INCH = 72  # (used for converting inches to points)
CM = 28.3464  # (used for converting centimeters to points)
MM = 2.83464  # (used for converting millimeters to points)
# 2 * inch is equal to 144 points
# 10 * cm is equal to 283.46456 points


UNDER: bool = True

# Pre-computed values
two_pi = 2 * pi  # 360 degrees
tau = 2 * pi  # 360 degrees
phi = (1 + 5**0.5) / 2  # golden ratio


def gen_unique_ids() -> Iterator[int]:
    """Yield an infinite sequence of unique integer IDs.

    Every drawable object in Simetri receives an ID from this generator
    (via ``get_unique_id``).

    Yields:
        int: The next unique identifier, starting at 0.

    Examples:
        >>> gen = gen_unique_ids()
        >>> next(gen), next(gen)
        (0, 1)
    """
    id_ = 0
    while True:
        yield id_
        id_ += 1


unique_id = gen_unique_ids()

d_id_obj = {}  # for Shape objects


def get_unique_id(item: object) -> int:
    """Allocate a unique ID and register ``item`` in ``d_id_obj``.

    Args:
        item: Object to register (typically a Shape, Group, or sketch).

    Returns:
        int: Newly assigned unique identifier.

    Examples:
        >>> from simetri.base.common import d_id_obj, get_unique_id
        >>> class _Item: pass
        >>> item = _Item()
        >>> uid = get_unique_id(item)
        >>> d_id_obj[uid] is item

        True
    """
    id_ = next(unique_id)
    d_id_obj[id_] = item
    return id_


origin = (0.0, 0.0)  # used for a point at the origin
axis_x = (origin, (1.0, 0.0))  # used for a line along x axis
axis_y = (origin, (0.0, 1.0))  # used for a line along y axis
axis_diag1 = (origin, (1.0, 1.0))  # used for a line along y = x
axis_diag2 = (origin, (1.0, -1.0))

axis_hex = (
    (0.0, 0.0),
    (cos(pi / 3), sin(pi / 3)),
)  # used for 3 and 6 rotation symmetries


def _set_Nones(obj: object, args: Sequence[str], values: Sequence[Any]) -> None:
    """
    Internally used in instance construction to set default values for None values.

    Args:
        obj (Any): The object to set values for.
        args (list): The arguments to set.
        values (list): The values to set.
    """
    for i, arg in enumerate(args):
        if values[i] is None:
            setattr(obj, arg, runtime_defaults[arg])
        else:
            setattr(obj, arg, values[i])


def resolve_tol(
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> tuple[float, float]:
    """Fill ``None`` tolerances from ``runtime_defaults``.

    ``runtime_defaults`` is not ``sg.defaults``. A missing argument uses
    ``temp_defaults``, then ``user_defaults``, then factory
    (``rel_tol=0``, ``abs_tol=0.001``). Comparisons use

    ``abs(a - b) <= abs_tol + rel_tol * abs(b)``
    (NumPy ``isclose``; map ``rel_tol`` → ``rtol``, ``abs_tol`` → ``atol``).

    Args:
        rel_tol: Relative tolerance. ``None`` uses ``runtime_defaults["rel_tol"]``.
        abs_tol: Absolute tolerance. ``None`` uses ``runtime_defaults["abs_tol"]``.

    Returns:
        tuple: ``(rel_tol, abs_tol)``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.resolve_tol()
        (0, 0.001)
        >>> sg.resolve_tol(abs_tol=0.05)
        (0, 0.05)
        >>> sg.resolve_tol(rel_tol=0.01, abs_tol=0.05)
        (0.01, 0.05)
    """
    if rel_tol is None:
        rel_tol = runtime_defaults["rel_tol"]
    if abs_tol is None:
        abs_tol = runtime_defaults["abs_tol"]
    return (rel_tol, abs_tol)


def get_defaults(args: Sequence[str], values: Sequence[Any]) -> list[Any]:
    """Fill ``None`` entries from ``runtime_defaults`` (not ``sg.defaults``).

    Args:
        args (list): Setting names to resolve.
        values (list): Values parallel to ``args``; ``None`` picks a default.

    Returns:
        list: Resolved values in the same order as ``args``.

    Examples:
        >>> from simetri.base.common import get_defaults
        >>> get_defaults(["rel_tol", "abs_tol"], [None, None])
        [0, 0.001]
        >>> get_defaults(["abs_tol"], [0.05])
        [0.05]
    """
    res = []
    for i, arg in enumerate(args):
        if values[i] is None:
            res.append(runtime_defaults[arg])
        else:
            res.append(values[i])
    return res


# def alias_argument(official_name, alias_name):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             if alias_name in kwargs:
#                 if official_name in kwargs:
#                     raise TypeError(f"Received both '{official_name}' and its alias '{alias_name}'!")

#                 # Warn the user (optional)
#                 warnings.warn(f"'{alias_name}' is deprecated; use '{official_name}' instead.", DeprecationWarning)

#                 # Map the alias value to the official parameter name
#                 kwargs[official_name] = kwargs.pop(alias_name)
#             return func(*args, **kwargs)
#         return wrapper
#     return decorator

# @alias_argument(official_name='username', alias_name='user')
# def greet(username):
#     print(f"Hello, {username}!")


def _alias_argument_map(
    official_name: str | Mapping[str, str | Sequence[str]],
    alias_name: str | Sequence[str] | None,
) -> dict[str, tuple[str, ...]]:
    if isinstance(official_name, Mapping):
        if alias_name is not None:
            raise TypeError(
                "alias_name cannot be used when official_name is a mapping."
            )
        raw: Mapping[str, str | Sequence[str]] = official_name
    else:
        if alias_name is None:
            raise TypeError(
                "alias_name is required when official_name is not a mapping."
            )
        raw = {official_name: alias_name}

    alias_map: dict[str, tuple[str, ...]] = {}
    seen_aliases: dict[str, str] = {}
    for official, aliases in raw.items():
        if isinstance(aliases, str):
            alias_tuple = (aliases,)
        else:
            alias_tuple = tuple(aliases)
        if not alias_tuple:
            raise ValueError(
                f"No aliases given for official argument '{official}'."
            )
        for alias in alias_tuple:
            if alias == official:
                raise ValueError(
                    f"Alias '{alias}' is the same as the official name."
                )
            if alias in seen_aliases:
                raise ValueError(
                    f"Alias '{alias}' is already used for "
                    f"'{seen_aliases[alias]}'."
                )
            seen_aliases[alias] = official
        alias_map[official] = alias_tuple
    return alias_map


def _alias_conflict_message(official: str, present: list[str]) -> str:
    aliases_present = [name for name in present if name != official]
    if official in present and len(aliases_present) == 1:
        return (
            f"Received both '{official}' and its alias '{aliases_present[0]}'!"
        )
    if official in present:
        quoted = ", ".join(f"'{name}'" for name in aliases_present)
        return f"Received both '{official}' and its aliases {quoted}!"
    quoted = ", ".join(f"'{name}'" for name in present)
    return f"Received multiple aliases for '{official}': {quoted}!"


def alias_argument(
    official_name: str | Mapping[str, str | Sequence[str]],
    alias_name: str | Sequence[str] | None = None,
) -> Callable:
    """Rewrite keyword aliases to official parameter names.

    One official name may have several aliases. Several official names may
    be aliased in a single mapping.

    Args:
        official_name: Official parameter name, or a mapping of official
            names to one alias or a sequence of aliases.
        alias_name: Alias or sequence of aliases for ``official_name``.
            Required when ``official_name`` is a string; omitted when it
            is a mapping.

    Returns:
        Callable: Decorator that maps alias keywords onto official names.

    Raises:
        TypeError: If an official name and one of its aliases are both
            given, or if two aliases for the same official name are both
            given.
        TypeError: If ``alias_name`` is omitted for a string
            ``official_name``, or supplied together with a mapping.
        ValueError: If an official name has no aliases, an alias repeats
            the official name, or one alias is used for two official names.

    Examples:
        >>> import simetri.graphics as sg
        >>> @sg.alias_argument("count", "n")
        ... def _double(count):
        ...     return count * 2
        >>> _double(n=3)
        6
        >>> @sg.alias_argument("count", ["n", "num"])
        ... def _double2(count):
        ...     return count * 2
        >>> _double2(num=4)
        8
        >>> @sg.alias_argument({"x": "width", "y": ["height", "h"]})
        ... def _size(x, y):
        ...     return (x, y)
        >>> _size(width=2, h=3)
        (2, 3)
        >>> _double(count=1, n=2)
        Traceback (most recent call last):
            ...
        TypeError: Received both 'count' and its alias 'n'!
    """
    alias_map = _alias_argument_map(official_name, alias_name)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            for official, aliases in alias_map.items():
                present = []
                if official in kwargs:
                    present.append(official)
                for alias in aliases:
                    if alias in kwargs:
                        present.append(alias)  # noqa
                if len(present) > 1:
                    raise TypeError(_alias_conflict_message(official, present))
                if present and present[0] != official:
                    # Warn the user (optional)
                    # warnings.warn(
                    #     f"'{present[0]}' is deprecated; "
                    #     f"use '{official}' instead.",
                    #     DeprecationWarning,
                    # )

                    # Map the alias value to the official parameter name
                    kwargs[official] = kwargs.pop(present[0])
            return func(*args, **kwargs)

        return wrapper

    return decorator

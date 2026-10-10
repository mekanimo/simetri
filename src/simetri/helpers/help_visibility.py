"""Ground rules for which names appear in user-facing help browse and lookup.

Edit the constants in this file, then re-apply:

    uv run python -m simetri.helpers.check_help_visibility
    uv run python -m simetri.helpers.compile_help_topics

See ``topic_guides/building_help.qmd`` (Help visibility ground rules).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# Topic browse lists: ``sg.help("points")`` when no ``.qmd`` is loaded.
INCLUDE_COMPILED_IN_TOPIC_BROWSE: bool = False

# ``sg.*`` / ``Class.method`` lines always shown when listed in ``d_help_topic``
# or ``help_topic_supplements`` (curated by hand).
ALWAYS_SHOW_CURATED_HELP_LINES: bool = True

# Defining ``__module__`` prefixes that are never user-facing help targets.
EXCLUDE_MODULE_PREFIXES: tuple[str, ...] = (
    "simetri.scrap.",
)

# Casefold substring match on the full help name (``sg.foo``, ``Canvas.draw``).
EXCLUDE_NAME_SUBSTRINGS: tuple[str, ...] = (
    "_sketch",
)

# Exceptions: always visible for browse / lookup / compile (alphabetical).
HELP_VISIBILITY_ALLOWLIST: frozenset[str] = frozenset(
    {
        # "sg.Canvas.draw",
    }
)


class HelpVisibilityContext(str, Enum):
    """Where a help name is being surfaced."""

    COMPILED_TOPIC = "compiled_topic"
    STRING_LOOKUP = "string_lookup"
    SIMILAR_NAMES = "similar_names"
    TOPIC_BROWSE_LINE = "topic_browse_line"


def help_line_leaf_name(help_line: str) -> str:
    """Return the symbol leaf (``draw`` from ``Canvas.draw`` or ``sg.distance``)."""
    token = help_line.strip().split()[0]
    token = token.removeprefix("sg.")
    return token.rsplit(".", maxsplit=1)[-1]


def normalize_help_line_name(help_line: str) -> str:
    """Normalize one browse/lookup line to ``sg.*`` or ``Class.method`` form."""
    token = help_line.strip().split()[0]
    if token.startswith("See also:") or token.startswith("("):
        return token
    if token.startswith("sg."):
        return token
    if "." in token:
        return token
    return f"sg.{token}"


def object_for_help_name(name: str) -> object | None:
    """Resolve a help string to an object on ``simetri.graphics``, if any."""
    import simetri.graphics as sg

    key = name.removeprefix("sg.")
    if not key:
        return None
    try:
        if "." not in key:
            return getattr(sg, key)
        parts = key.split(".")
        obj: object = getattr(sg, parts[0])
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj
    except AttributeError:
        return None


def _defining_module(obj: object) -> str | None:
    mod = getattr(obj, "__module__", None)
    if isinstance(mod, str):
        return mod
    return None


def is_help_name_visible(
    name: str,
    obj: object | None = None,
    *,
    context: HelpVisibilityContext,
) -> bool:
    """Return whether ``name`` may appear for the given help surface."""
    normalized = normalize_help_line_name(name)
    if normalized in HELP_VISIBILITY_ALLOWLIST:
        return True
    if normalized.startswith("See also:") or normalized.startswith("("):
        return True

    if (
        context is HelpVisibilityContext.COMPILED_TOPIC
        and not INCLUDE_COMPILED_IN_TOPIC_BROWSE
    ):
        return False

    leaf = help_line_leaf_name(normalized)
    if leaf.endswith("_"):
        return False

    folded = normalized.casefold()
    for fragment in EXCLUDE_NAME_SUBSTRINGS:
        if fragment.casefold() in folded:
            return False

    if obj is None:
        obj = object_for_help_name(normalized)
    if obj is not None:
        module = _defining_module(obj)
        if module is not None:
            for prefix in EXCLUDE_MODULE_PREFIXES:
                if module.startswith(prefix):
                    return False

    return True


def filter_sg_export_names(names: Iterable[str]) -> tuple[str, ...]:
    """Filter top-level ``graphics`` export names for compile / indexing."""
    if not INCLUDE_COMPILED_IN_TOPIC_BROWSE:
        return ()
    visible: list[str] = []
    for name in names:
        display = name if name.startswith("sg.") else name
        obj = object_for_help_name(display)
        if is_help_name_visible(
            display,
            obj,
            context=HelpVisibilityContext.COMPILED_TOPIC,
        ):
            visible.append(name.removeprefix("sg."))
    return tuple(sorted(set(visible), key=str.casefold))


def filter_named_help_registration(name: str, obj: object) -> bool:
    """Whether to register ``name`` for string lookup and similar-name search."""
    return is_help_name_visible(
        name,
        obj,
        context=HelpVisibilityContext.STRING_LOOKUP,
    )


def exclusion_reason(name: str, obj: object | None = None) -> str | None:
    """Return why ``name`` is hidden, or ``None`` if visible for lookup."""
    normalized = normalize_help_line_name(name)
    if normalized in HELP_VISIBILITY_ALLOWLIST:
        return None
    leaf = help_line_leaf_name(normalized)
    if leaf.endswith("_"):
        return "leaf name ends with '_'"
    folded = normalized.casefold()
    for fragment in EXCLUDE_NAME_SUBSTRINGS:
        if fragment.casefold() in folded:
            return f"name contains {fragment!r}"
    if obj is None:
        obj = object_for_help_name(normalized)
    if obj is not None:
        module = _defining_module(obj)
        if module is not None:
            for prefix in EXCLUDE_MODULE_PREFIXES:
                if module.startswith(prefix):
                    return f"module {module!r}"
    return None


def excluded_export_report(public_names: Iterable[str]) -> list[tuple[str, str]]:
    """Return ``(name, reason)`` for each excluded top-level export."""
    rows: list[tuple[str, str]] = []
    for name in sorted(set(public_names), key=str.casefold):
        display = name.removeprefix("sg.")
        obj = object_for_help_name(display)
        if is_help_name_visible(
            display,
            obj,
            context=HelpVisibilityContext.STRING_LOOKUP,
        ):
            continue
        reason = exclusion_reason(display, obj) or "excluded"
        rows.append((display, reason))
    return rows

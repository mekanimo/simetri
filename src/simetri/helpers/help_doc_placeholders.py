"""Searchable documentation cross-reference placeholders for help and mkdocs.

Placeholders are emitted verbatim in ``sg.help`` topic listings. A later
mkdocs (or site) build can find and replace them with real links.

Format (always lowercase kind, no spaces)::

    {{doc:guide:<topic_key>}}   — ``topic_guides/<topic_key>.qmd`` / ``sg.doc``
    {{doc:api:<name>}}          — API symbol; ``name`` is ``sg.foo`` or dotted path
    {{doc:defaults:<key>}}      — ``defaults_help`` setting
    {{doc:site:<path>}}         — future mkdocs path (POSIX-style, no leading slash)

Examples::

    {{doc:guide:viewer}}
    {{doc:api:sg.subdivide_segment}}
    {{doc:site:topics/segments.md}}
"""

from __future__ import annotations

import re
from typing import Final

DOC_PLACEHOLDER_PATTERN: Final = re.compile(
    r"\{\{doc:([a-z]+):([^}]+)\}\}"
)

_VALID_KINDS: Final = frozenset({"guide", "api", "defaults", "site", "source"})


def doc_placeholder(kind: str, target: str) -> str:
    """Build one ``{{doc:kind:target}}`` token."""
    normalized_kind = kind.strip().lower()
    if normalized_kind not in _VALID_KINDS:
        raise ValueError(
            f"doc placeholder kind must be one of {sorted(_VALID_KINDS)!r}, "
            f"got {kind!r}"
        )
    cleaned = target.strip()
    if not cleaned:
        raise ValueError("doc placeholder target must be non-empty")
    return f"{{{{doc:{normalized_kind}:{cleaned}}}}}"


def parse_doc_placeholders(text: str) -> list[tuple[str, str]]:
    """Return ``(kind, target)`` for each placeholder in ``text``."""
    return [
        (match.group(1), match.group(2))
        for match in DOC_PLACEHOLDER_PATTERN.finditer(text)
    ]


def format_doc_placeholder_for_help(line: str) -> str:
    """Keep placeholder searchable; add a short REPL hint after it."""
    parsed = parse_doc_placeholders(line)
    if len(parsed) != 1 or line.strip() != doc_placeholder(parsed[0][0], parsed[0][1]):
        return line
    kind, target = parsed[0]
    if kind == "guide":
        return f"{line}  → sg.doc({target!r})"
    if kind == "defaults":
        return f"{line}  → sg.help({target!r})"
    if kind == "api" and target.startswith("sg."):
        return f"{line}  → sg.help({target!r})"
    if kind == "source":
        return f"{line}  → mentioned in guide {target!r}"
    return line

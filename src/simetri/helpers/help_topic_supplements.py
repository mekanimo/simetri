"""Manual semantic links for help topic browse lists.

Token matching (``compile_help_topics``) lists ``sg.*`` exports whose
names share words with a topic key. Use the structures below when
related names do **not** match by spelling but belong together by meaning.

Topic keys must exist in ``help_utils.all_help_topic_keys()`` (curated
``d_help_topic`` plus every ``topic_guides/*.qmd`` stem).

TOPIC_SG_ENTRIES
    ``dict[str, tuple[str, ...]]`` — topic → extra ``sg.*`` lines.

SG_NAME_TO_TOPICS
    ``dict[str, tuple[str, ...]]`` — one export → topic keys (many-to-many
    graph). Keys may be ``"distance"`` or ``"sg.distance"``.

TOPIC_ALIASES
    ``dict[str, str]`` — help query → canonical topic key (beyond compiled
    singular/plural aliases).

RELATED_TOPICS
    ``dict[str, tuple[str, ...]]`` — topic → other topic keys for
    cross-browse hints at the bottom of a topic listing.

TOPIC_DOC_PLACEHOLDERS
    ``dict[str, tuple[str, ...]]`` — topic → ``{{doc:kind:target}}`` tokens
    (see ``help_doc_placeholders.py``). Used for mkdocs link conversion later.
"""

from __future__ import annotations

TOPIC_SG_ENTRIES: dict[str, tuple[str, ...]] = {
    # "lines": (
    #     "sg.some_helper",  # semantic link; name has no "line" token
    # ),
}

SG_NAME_TO_TOPICS: dict[str, tuple[str, ...]] = {
    # "sg.stitch": ("segments", "lines"),
}

TOPIC_ALIASES: dict[str, str] = {
    # "Segments": "segments",
}

RELATED_TOPICS: dict[str, tuple[str, ...]] = {
    # "segments": ("lines", "points"),
}

TOPIC_DOC_PLACEHOLDERS: dict[str, tuple[str, ...]] = {
    # "segments": (
    #     "{{doc:site:geometry/segments.md}}",
    # ),
}

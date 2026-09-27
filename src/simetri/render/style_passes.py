"""Renderer-neutral style-pass helpers for sketch styling."""

from __future__ import annotations

from typing import Any

from ..config.settings import runtime_defaults

NON_SCOPABLE_SCOPE_KEYS = frozenset(
    ["draw_double", "double_color", "double_distance"]
)


def resolve_style_value(sketch_dict: dict[str, Any], style_key: str) -> object:
    """Resolve a style value from sketch data or ``runtime_defaults``.

    Args:
        sketch_dict: Sketch ``__dict__`` or equivalent attribute bag.
        style_key: Style attribute name to resolve.

    Returns:
        object: Resolved value from the sketch or ``runtime_defaults``.

    Raises:
        KeyError: If ``style_key`` is missing from the sketch and
        ``runtime_defaults``.

    Examples:
        >>> from simetri.render.style_passes import resolve_style_value
        >>> from simetri.config.settings import runtime_defaults
        >>> resolve_style_value({}, "grid_line_width") == runtime_defaults["grid_line_width"]
        True
    """

    if style_key in sketch_dict:
        return sketch_dict[style_key]
    if style_key in runtime_defaults:
        return runtime_defaults[style_key]
    raise KeyError(
        f"Missing style key '{style_key}' in sketch and runtime_defaults"
    )


def create_style_signature(
    sketch_dict: dict[str, Any], style_keys: list[str]
) -> tuple[tuple[str, str], ...]:
    """Create a hashable style signature for a sketch using style_keys.

    Examples:
        >>> from simetri.render.style_passes import create_style_signature
        >>> sig = create_style_signature({"line_width": 1.0}, ["line_width"])
        >>> sig[0][0]
        'line_width'
    """

    signature = []
    for style_key in style_keys:
        style_value = resolve_style_value(sketch_dict, style_key)
        signature.append((style_key, repr(style_value)))
    return tuple(signature)


def build_styles_dict(
    sketches: list[Any],
    style_domain_key_sets: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    """Pass 1: Build neutral styles dictionary as {style_id: style_obj}.

    Examples:
        >>> from simetri.render.style_passes import build_styles_dict
        >>> build_styles_dict([], {"line": ["line_width"]})
        {}
    """

    signature_to_style_id = {}
    styles_dict = {}
    style_counter = 1

    for sketch in sketches:
        sketch_dict = sketch.__dict__
        for domain_name, style_keys in style_domain_key_sets.items():
            style_signature = create_style_signature(sketch_dict, style_keys)
            signature_key = (domain_name, style_signature)
            if signature_key not in signature_to_style_id:
                style_id = f"style_{style_counter}"
                style_counter += 1
                style_obj = {
                    "id": style_id,
                    "domain": domain_name,
                    "keys": list(style_keys),
                    "signature": style_signature,
                }
                styles_dict[style_id] = style_obj
                signature_to_style_id[signature_key] = style_id

    return styles_dict


def build_style_sketch_dict(
    sketches: list[Any],
    style_domain_key_sets: dict[str, list[str]],
    styles_dict: dict[str, dict[str, Any]],
) -> dict[str, list[int]]:
    """Pass 2: Build style-to-sketch dictionary as {style_id: [sketch_id, ...]}.

    Examples:
        >>> from simetri.render.style_passes import build_style_sketch_dict
        >>> build_style_sketch_dict([], {"line": ["line_width"]}, {})
        {}
    """

    style_sketch_dict = {style_id: [] for style_id in styles_dict}
    signature_to_style_id = {}
    for style_id, style_obj in styles_dict.items():
        signature_key = (style_obj["domain"], style_obj["signature"])
        signature_to_style_id[signature_key] = style_id

    for sketch in sketches:
        sketch_dict = sketch.__dict__
        sketch_id = sketch.id
        for domain_name, style_keys in style_domain_key_sets.items():
            style_signature = create_style_signature(sketch_dict, style_keys)
            signature_key = (domain_name, style_signature)
            style_id = signature_to_style_id[signature_key]
            style_sketch_dict[style_id].append(sketch_id)

    return style_sketch_dict


def build_sketch_style_ids(style_sketch_dict: dict[str, list[int]]) -> dict[int, list[str]]:
    """Build reverse style mapping as {sketch_id: [style_id, ...]}.

    Examples:
        >>> from simetri.render.style_passes import build_sketch_style_ids
        >>> build_sketch_style_ids({"style_1": [3, 4]})
        {3: ['style_1'], 4: ['style_1']}
    """

    sketch_style_ids = {}
    for style_id, sketch_ids in style_sketch_dict.items():
        for sketch_id in sketch_ids:
            if sketch_id not in sketch_style_ids:
                sketch_style_ids[sketch_id] = []
            sketch_style_ids[sketch_id].append(style_id)
    return sketch_style_ids


def validate_style_sketch_coverage(
    sketches: list[Any],
    sketch_style_ids: dict[int, list[str]],
) -> None:
    """Validate that every sketch has at least one style assignment.

    Raises:
        ValueError: If any sketch has no style mapping.

    Examples:
        >>> from simetri.render.style_passes import validate_style_sketch_coverage
        >>> validate_style_sketch_coverage([], {})
    """

    for sketch in sketches:
        sketch_id = sketch.id
        if sketch_id not in sketch_style_ids:
            raise ValueError(
                f"Sketch {sketch_id} has no style mapping in style_sketch dictionary"
            )

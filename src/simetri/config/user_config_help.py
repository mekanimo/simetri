"""Help text for ``simetri_config.toml`` sections and keys (``sg.help`` strings)."""

from __future__ import annotations

# (section, key) -> help body. Section-only intros use key "".
_USER_CONFIG_ENTRIES: dict[tuple[str, str], str] = {
    ("converters", ""): (
        "Personal external format converters. Native saves: .svg, .pdf, .tex, "
        ".eps, .ps. Other extensions need ``[converters.<format>]`` tables. "
        "Never put this section in shared ``use_settings`` tomls. "
        "See sg.help('external_converters')."
    ),
    ("converters", "command"): (
        "Per-format CLI argv or shell string. Used on ``[converters.<format>]`` "
        "tables (not the global ``[converters]`` table). Placeholders: "
        "``{input}``, ``{output}``, ``{outdir}``, ``{stem}``. Prefer a TOML "
        "string array with ``shell = false``."
    ),
    ("converters", "enabled"): (
        "When false, Simetri does not run external converters; only native "
        "extensions can be saved."
    ),
    ("converters", "shell"): (
        "When false (default), ``command`` on ``[converters.<format>]`` is a TOML "
        "array of argv strings. When true, ``command`` is one shell string."
    ),
    ("converters", "source"): (
        "Intermediate format Simetri writes before conversion (for example "
        "``\"svg\"`` for PNG output)."
    ),
    ("converters", "timeout_seconds"): (
        "Maximum seconds to wait for a converter subprocess."
    ),
    ("defaults", ""): (
        "Optional overrides for library ``defaults`` applied at import. Only "
        "**uncommented** keys in your toml replace built-ins; commented lines are "
        "documentation. Use ``sg.save_user_defaults(...)`` to write this section. "
        "For help on a specific default name, use ``sg.help('<name>')`` (for "
        "example ``sg.help('line_width')``)."
    ),
    ("paths", ""): (
        "Directory defaults for saves and tests. Edit ``simetri_config.toml`` "
        "under your ``simetri_user`` folder (``sg.user_config_path()``)."
    ),
    ("paths", "default_output_directory"): (
        "Directory joined with bare filenames in ``canvas.save('figure.svg')``. "
        "When empty, bare names raise an error; pass a full path instead. "
        "Does not affect paths that already include a folder component."
    ),
    ("paths", "default_test_directory"): (
        "Default directory for test or gallery output when tools use it. "
        "May be empty."
    ),
    ("styles", ""): (
        "Named draw-time style recipes ``[styles.<name>]``. Look up with "
        "``sg.user_styles['<name>']``. Keys mirror shape style fields (``fill``, "
        "``line_width``, ``line_color``, …). Personal config only — not shared "
        "tomls. See sg.help('style_definitions')."
    ),
    ("tex", ""): (
        "Personal LaTeX/PDF compiler for ``canvas.save('….pdf')``. When "
        "``command`` is omitted, Simetri uses ``defaults['latex_compiler']`` on "
        "PATH. Never put this section in shared tomls. See sg.help('tex_compiler')."
    ),
    ("tex", "command"): (
        "LaTeX engine argv or shell string. Placeholders: ``{input}`` (.tex), "
        "``{output}`` (.pdf), ``{outdir}``, ``{stem}``. Omit to use "
        "``defaults['latex_compiler']``."
    ),
    ("tex", "shell"): (
        "When false (default), ``command`` is a TOML string array. When true, "
        "``command`` is one shell string passed to the system shell."
    ),
    ("tex", "timeout_seconds"): (
        "Maximum seconds to wait for the TeX subprocess."
    ),
    ("viewer", ""): (
        "How ``canvas.save`` opens a file after a successful save (when "
        "``show`` / ``defaults['show_browser']`` allow it). Modes: "
        "``\"system\"`` (OS default), ``\"command\"`` (run ``command``), "
        "``\"none\"`` (skip). Never put this section in shared tomls. "
        "See sg.help('viewer')."
    ),
    ("viewer", "command"): (
        "Program to run when ``mode = \"command\"``. Placeholders: "
        "``{filepath}``, ``{file}``, ``{url}``. If missing, Simetri warns and "
        "falls back to ``\"system\"``."
    ),
    ("viewer", "mode"): (
        "``\"system\"`` | ``\"command\"`` | ``\"none\"``. Controls how saved "
        "files are opened when opening is allowed."
    ),
    ("viewer", "shell"): (
        "When false (default), ``command`` is a TOML string array. When true, "
        "``command`` is one shell string."
    ),
    ("warnings", ""): (
        "Permanent warning toggles loaded at import. ``warnings_on`` is the "
        "master switch; nested ``[warnings.<group>]`` tables mirror "
        "``WarningType.<group>.<leaf>``. Use ``sg.save_user_warning`` / "
        "``sg.set_warning_on`` for API updates. See sg.help('warnings')."
    ),
    ("warnings", "warnings_on"): (
        "When false, all Simetri warnings are suppressed for the session "
        "(equivalent to turning warnings off globally)."
    ),
}

_SECTION_ORDER = (
    "converters",
    "defaults",
    "paths",
    "styles",
    "tex",
    "viewer",
    "warnings",
)

# Bare section names that win over ``sg.help`` topic aliases (e.g. ``paths``).
USER_CONFIG_TOPIC_PREEMPT: frozenset[str] = frozenset(
    {
        "converters",
        "defaults",
        "paths",
        "styles",
        "warnings",
    }
)


def user_config_sections() -> frozenset[str]:
    """Return top-level ``simetri_config.toml`` section names with help text."""
    return frozenset(_SECTION_ORDER)


def _entries_for_section(section: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for (sec, key), text in _USER_CONFIG_ENTRIES.items():
        if sec == section and key:
            items.append((key, text))
    items.sort(key=lambda item: item[0])
    return items


def _format_section_help(section: str) -> str | None:
    intro = _USER_CONFIG_ENTRIES.get((section, ""))
    if intro is None:
        return None
    lines = [f"simetri_config.toml [{section}]", "", intro, ""]
    for key, text in _entries_for_section(section):
        first_line = text.splitlines()[0]
        lines.append(f"{key}:")
        lines.append(f"  {first_line}")
        lines.append("")
    lines.append("Full guide: sg.help('user_settings')")
    return "\n".join(lines).strip()


def _format_qualified_help(section: str, key: str) -> str | None:
    text = _USER_CONFIG_ENTRIES.get((section, key))
    if text is None:
        return None
    return f"simetri_config.toml [{section}]\n{key}:\n{text}"


def _format_bare_key_help(key: str) -> str | None:
    matches = [
        (section, entry_key)
        for (section, entry_key), _ in _USER_CONFIG_ENTRIES.items()
        if entry_key == key
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: (_SECTION_ORDER.index(item[0])
        if item[0] in _SECTION_ORDER else len(_SECTION_ORDER), item[0]))
    parts: list[str] = []
    for section, _ in matches:
        text = _USER_CONFIG_ENTRIES[(section, key)]
        parts.append(f"[{section}]")
        parts.append(f"{key}:")
        parts.append(text)
        parts.append("")
    if len(matches) > 1:
        parts.insert(0, f"Setting {key!r} appears under multiple sections:\n")
    parts.append("Full guide: sg.help('user_settings')")
    return "\n".join(parts).strip()


def user_config_help(query: str) -> str | None:
    """Return help for a config section or key name, or ``None`` if unknown."""
    name = query.strip()
    if not name:
        return None

    if "." in name:
        section, _, key = name.partition(".")
        if key:
            return _format_qualified_help(section, key)
        return _format_section_help(section)

    section_help = _format_section_help(name)
    if section_help is not None:
        return section_help

    return _format_bare_key_help(name)


def user_config_help_keys() -> frozenset[str]:
    """Return string keys registered for ``sg.help`` lookup."""
    keys: set[str] = set(_SECTION_ORDER)
    for section, key in _USER_CONFIG_ENTRIES:
        if key:
            keys.add(key)
            keys.add(f"{section}.{key}")
    return frozenset(keys)

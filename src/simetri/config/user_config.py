"""Load and apply per-user Simetri settings from ``simetri_config.toml``."""

from __future__ import annotations

import ast
import json
import platform
import runpy
import tomllib
from datetime import date
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any

from ..base.all_enums import WarningType
from ..coloring import colors
from ..coloring.colors import Color, check_color
from ..helpers.validation import check_version

CONFIG_FILENAME = "simetri_config.toml"
_TEMPLATE_FILENAME = "config_template.txt"
_USER_SUBDIR = "simetri_user"
_PACKAGE_TEMPLATE = Path(__file__).with_name(_TEMPLATE_FILENAME)

# Populated by ``apply_user_config``.
_user_paths: dict[str, str] = {
    "default_output_directory": "",
    "default_test_directory": "",
}
_DIRECT_USER_DEFAULT_EDIT_MESSAGE = (
    "Direct edits to user default overrides are not saved to "
    "simetri_config.toml and may be lost when the config is reloaded. "
    "Use sg.save_user_defaults(...) to change personal defaults for this "
    "session and on disk, or edit simetri_config.toml and restart the "
    "Python kernel (or call sg.apply_user_config())."
)

_internal_user_default_write_depth = 0


@contextmanager
def _internal_user_default_writes() -> Iterator[None]:
    """Allow library code to mutate overrides without the direct-edit warning."""
    global _internal_user_default_write_depth
    _internal_user_default_write_depth += 1
    try:
        yield
    finally:
        _internal_user_default_write_depth -= 1


class _UserDefaultOverridesDict(dict[str, Any]):
    """In-memory ``[defaults]`` overrides; warns on user-facing mutation."""

    def _warn_if_direct_edit(self) -> None:
        if _internal_user_default_write_depth:
            return
        from .settings import issue_warning

        issue_warning(
            _DIRECT_USER_DEFAULT_EDIT_MESSAGE,
            warning_type=WarningType.file.config,
            stacklevel=4,
        )

    def __setitem__(self, key: str, value: Any) -> None:
        self._warn_if_direct_edit()
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self._warn_if_direct_edit()
        super().__delitem__(key)

    def clear(self) -> None:
        if self and not _internal_user_default_write_depth:
            self._warn_if_direct_edit()
        super().clear()

    def pop(self, key: str, default: Any = ...) -> Any:  # type: ignore[override]
        self._warn_if_direct_edit()
        if default is ...:
            return super().pop(key)
        return super().pop(key, default)

    def popitem(self) -> tuple[str, Any]:
        self._warn_if_direct_edit()
        return super().popitem()

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self and not _internal_user_default_write_depth:
            self._warn_if_direct_edit()
        return super().setdefault(key, default)

    def update(  # type: ignore[override]
        self,
        other: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        if not _internal_user_default_write_depth and (other or kwargs):
            self._warn_if_direct_edit()
        super().update(other, **kwargs)


_user_default_overrides = _UserDefaultOverridesDict()
_config_applied = False


def _set_user_default_override(key: str, value: Any) -> None:
    """Set one override without the direct-edit warning (library use only)."""
    with _internal_user_default_writes():
        _user_default_overrides[key] = value

# Personal ``[converters]`` settings (never loaded from shared tomls).
_NATIVE_SAVE_EXTENSIONS = frozenset({".pdf", ".eps", ".ps", ".svg", ".tex"})
_CONVERTER_SOURCES = frozenset({"svg", "pdf", "tex"})
_converter_globals: dict[str, Any] = {
    "enabled": True,
    "timeout_seconds": 120,
    "shell": False,
}
# format key without dot → {"source": "svg", "command": list[str] | str}
_converter_formats: dict[str, dict[str, Any]] = {}

# Personal ``[tex]`` compiler (never loaded from shared tomls).
# command is None → use defaults["latex_compiler"] on PATH.
_tex_settings: dict[str, Any] = {
    "timeout_seconds": 120,
    "shell": False,
    "command": None,
}

# Personal ``[viewer]`` for canvas.save() (never loaded from shared tomls).
# mode "system" → webbrowser / OS default; "command" → command; "none" → skip.
_VIEWER_MODES = frozenset({"system", "command", "none"})
_viewer_settings: dict[str, Any] = {
    "mode": "system",
    "shell": False,
    "command": None,
}

# Personal ``[styles.<name>]`` recipes (never loaded from shared tomls).
user_styles: dict[str, Any] = {}

# Personal ``[script_export]`` defaults for ``save_as`` (commented in template).
_script_export_settings: dict[str, Any] = {
    "include_script_header": False,
    "include_defaults": False,
    "author": "",
    "copyright": "",
    "description": "",
    "doc_version": "1.0.0",
    "license": "",
}
_script_export_custom_slots: dict[str, str] = {}


def set_user_settings_path() -> Path:
    """Return (and create) the OS-specific ``simetri_user`` config directory.

    Source - https://stackoverflow.com/a/77658488
    Posted by Het Vaghani
    Retrieved 2026-09-11, License - CC BY-SA 4.0

    Examples:

        >>> sg.set_user_settings_path().is_dir()
        True
    """
    system_name = platform.system()
    if system_name == "Windows":
        path = Path.home() / "AppData" / "Local" / _USER_SUBDIR
    elif system_name == "Darwin":
        path = Path.home() / "Library" / "Application Support" / _USER_SUBDIR
    elif system_name == "Linux":
        path = Path.home() / ".config" / _USER_SUBDIR
    else:
        raise ValueError(f"Unsupported operating system: {system_name}")

    path.mkdir(parents=True, exist_ok=True)
    return path


def user_config_path() -> Path:
    """Return the full path to the user's ``simetri_config.toml``.

    Examples:

        >>> sg.user_config_path().name
        'simetri_config.toml'
    """
    return set_user_settings_path() / CONFIG_FILENAME


def ensure_user_config() -> Path:
    """Create the user config directory and template file if missing.

    Returns:
        Path to ``simetri_config.toml``.

    Examples:

        >>> from simetri.config.user_config import ensure_user_config
        >>> ensure_user_config().name
        'simetri_config.toml'
    """
    config_path = user_config_path()
    if not config_path.exists():
        config_path.write_text(
            _PACKAGE_TEMPLATE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return config_path


def get_default_output_directory() -> str:
    """Return the configured default output directory (may be empty).

    Examples:

        >>> from simetri.config.user_config import get_default_output_directory
        >>> isinstance(get_default_output_directory(), str)
        True
    """
    return _user_paths["default_output_directory"]


def get_default_test_directory() -> str:
    """Return the configured default test directory (may be empty).

    Examples:

        >>> from simetri.config.user_config import get_default_test_directory
        >>> isinstance(get_default_test_directory(), str)
        True
    """
    return _user_paths["default_test_directory"]


def get_user_default_overrides() -> dict[str, Any]:
    """Return the live ``[defaults]`` override mapping (library internal)."""
    return _user_default_overrides


def get_converter_globals() -> dict[str, Any]:
    """Return personal ``[converters]`` global flags (copy).

    Examples:

        >>> from simetri.config.user_config import get_converter_globals
        >>> get_converter_globals()["enabled"]
        True
    """
    return dict(_converter_globals)


def get_converter_formats() -> dict[str, dict[str, Any]]:
    """Return personal per-format converter entries (shallow copy).

    Examples:

        >>> from simetri.config.user_config import get_converter_formats
        >>> isinstance(get_converter_formats(), dict)
        True
    """
    return {key: dict(value) for key, value in _converter_formats.items()}


def native_save_extensions() -> frozenset[str]:
    """Extensions Simetri writes without an external converter.

    Examples:

        >>> from simetri.config.user_config import native_save_extensions
        >>> ".svg" in native_save_extensions()
        True
    """
    return _NATIVE_SAVE_EXTENSIONS


def converter_supports_extension(extension: str) -> bool:
    """Return True if personal config defines a converter for ``extension``.

    Args:
        extension: File extension including the leading dot (e.g. ``.png``).

    Examples:

        >>> from simetri.config.user_config import converter_supports_extension
        >>> converter_supports_extension(".png")
        False
    """
    if not _converter_globals["enabled"]:
        return False
    format_key = extension.lstrip(".").lower()
    return format_key in _converter_formats


def get_tex_compiler() -> dict[str, Any]:
    """Return personal ``[tex]`` compiler settings (copy).

    ``command`` is ``None`` when the user did not set ``[tex].command``.

    Examples:

        >>> from simetri.config.user_config import get_tex_compiler
        >>> get_tex_compiler()["timeout_seconds"]
        120
    """
    return dict(_tex_settings)


def get_viewer_settings() -> dict[str, Any]:
    """Return personal ``[viewer]`` settings (copy).

    Loaded from ``[viewer]`` in ``simetri_config.toml`` when the config is
    applied. Shared ``use_settings`` tomls never include this table.

    Keys in the returned dict:

    ``mode`` (str)
        How ``canvas.save`` opens the file when opening is allowed
        (``show=True`` or ``defaults['show_browser']``). Allowed values only:

        - ``"system"`` — ``webbrowser.open`` with a ``file:`` URL (OS default
          app for the extension). Default when ``[viewer]`` is absent.
        - ``"command"`` — run ``command`` via subprocess; on failure or missing
          ``command``, Simetri warns and falls back to ``"system"``.
        - ``"none"`` — do not open (save still succeeds).

    ``shell`` (bool)
        When ``mode`` is ``"command"``: ``False`` means ``command`` is a list
        of argv strings; ``True`` means ``command`` is one shell string.

    ``command`` (list[str] | str | None)
        Program and arguments. ``None`` if unset. Placeholders ``{filepath}``,
        ``{file}``, ``{url}`` are substituted at launch. See ``sg.help('viewer')``.

    Examples:

        >>> from simetri.config.user_config import get_viewer_settings
        >>> get_viewer_settings()["mode"] in ("system", "command", "none")
        True
    """
    return dict(_viewer_settings)


def get_converter_for_extension(extension: str) -> dict[str, Any]:
    """Return converter config for ``extension``.

    Args:
        extension: File extension including the leading dot (e.g. ``.png``).

    Returns:
        dict: ``source``, ``command``, plus global ``timeout_seconds`` / ``shell``.

    Raises:
        KeyError: No converter for this extension, or converters disabled.

    Examples:

        >>> from simetri.config.user_config import get_converter_for_extension
        >>> get_converter_for_extension(".png")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        KeyError: ...
    """
    if not _converter_globals["enabled"]:
        raise KeyError(
            "External converters are disabled "
            "([converters].enabled = false in simetri_config.toml)."
        )
    format_key = extension.lstrip(".").lower()
    if format_key not in _converter_formats:
        raise KeyError(
            f"No [converters.{format_key}] entry in simetri_config.toml."
        )
    entry = dict(_converter_formats[format_key])
    entry["timeout_seconds"] = _converter_globals["timeout_seconds"]
    entry["shell"] = _converter_globals["shell"]
    entry["format_key"] = format_key
    return entry


def resolve_save_filepath(filepath: str | Path) -> str:
    """Resolve a save path, using ``default_output_directory`` for bare names.

    Args:
        filepath: User-supplied save path.

    Returns:
        Absolute or joined path string ready for ``validate_output_filepath``.

    Raises:
        ValueError: Bare filename and ``default_output_directory`` is unset.

    Examples:

        >>> from pathlib import Path
        >>> Path(sg.resolve_save_filepath("out/sub/file.svg")).name
        'file.svg'
    """
    path = Path(filepath)
    if path.parent != Path("") and path.parent != Path("."):
        return str(path)

    output_directory = _user_paths["default_output_directory"]
    if not output_directory:
        config_file = user_config_path()
        raise ValueError(
            "canvas.save() was given a filename without a directory, and "
            "paths.default_output_directory is not set in simetri_config.toml.\n"
            f"Either set paths.default_output_directory in {config_file}, "
            "or pass a full path, e.g. "
            r'canvas.save(r"C:\out\filename.svg").'
        )
    return str(Path(output_directory) / path.name)


def _convert_default_value(key: str, value: Any, expected_type: type) -> Any:
    """Convert a TOML value to the type expected by ``defaults``."""
    if expected_type is Color or (
        isinstance(expected_type, type) and issubclass(expected_type, Color)
    ):
        return check_color(value)

    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        if isinstance(value, expected_type):
            return value
        if isinstance(value, str):
            return expected_type[value.upper()]
        raise TypeError(
            f"Cannot convert {value!r} to {expected_type.__name__} for {key!r}"
        )

    if expected_type is float:
        return float(value)
    if expected_type is int:
        return int(value)
    if expected_type is bool:
        return bool(value)
    if expected_type is str:
        return str(value)
    return value


def _warn_invalid_key(message: str) -> None:
    """Emit an invalid-config-key warning without importing settings at top."""
    from .settings import issue_warning

    issue_warning(message, warning_type=WarningType.file.config)


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _update_config_lines(
    *,
    warnings_on: bool | None = None,
    leaf_states: dict[tuple[str, str], bool] | None = None,
) -> None:
    """Update warning keys in the user toml while preserving comments."""
    config_path = ensure_user_config()
    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    current_section: str | None = None
    leaf_states = {} if leaf_states is None else leaf_states
    updated_leaves: set[tuple[str, str]] = set()
    warnings_on_updated = warnings_on is None

    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1].strip()
            new_lines.append(line)
            continue

        code = stripped
        if "#" in code:
            code = code.split("#", 1)[0].strip()
        if not code or "=" not in code:
            new_lines.append(line)
            continue

        key_part, _, value_part = code.partition("=")
        key_name = key_part.strip()
        trailing = ""
        if "#" in stripped:
            trailing = " #" + stripped.split("#", 1)[1]

        if (
            current_section == "warnings"
            and key_name == "warnings_on"
            and warnings_on is not None
        ):
            newline = "\n" if line.endswith("\n") else ""
            indent = line[: len(line) - len(line.lstrip(" \t"))]
            new_lines.append(
                f"{indent}warnings_on = {_toml_bool(warnings_on)}{trailing}{newline}"
            )
            warnings_on_updated = True
            continue

        if current_section is not None and current_section.startswith(
            "warnings."
        ):
            group_name = current_section.partition(".")[2]
            leaf_key = (group_name, key_name)
            if leaf_key in leaf_states:
                newline = "\n" if line.endswith("\n") else ""
                indent = line[: len(line) - len(line.lstrip(" \t"))]
                enabled = leaf_states[leaf_key]
                new_lines.append(
                    f"{indent}{key_name} = {_toml_bool(enabled)}{trailing}{newline}"
                )
                updated_leaves.add(leaf_key)
                continue

        new_lines.append(line)

    if warnings_on is not None and not warnings_on_updated:
        _warn_invalid_key(
            f"Could not find warnings_on in {config_path}"
        )
    missing = set(leaf_states) - updated_leaves
    for group_name, leaf_name in sorted(missing):
        _warn_invalid_key(
            f"Could not find warnings.{group_name}.{leaf_name} in {config_path}"
        )

    config_path.write_text("".join(new_lines), encoding="utf-8")


def persist_warnings_on_flag(enabled: bool) -> None:
    """Write ``[warnings].warnings_on`` without rewriting the whole file."""
    _update_config_lines(warnings_on=enabled)


def persist_warning_leaves(leaves: Any, enabled: bool) -> None:
    """Write leaf enable flags under ``[warnings.<group>]``."""
    leaf_states: dict[tuple[str, str], bool] = {}
    for leaf in leaves:
        group_name, _, leaf_name = leaf.value.partition(".")
        leaf_states[(group_name, leaf_name)] = enabled
    if enabled:
        _update_config_lines(warnings_on=True, leaf_states=leaf_states)
    else:
        _update_config_lines(leaf_states=leaf_states)


def persist_all_warnings(enabled: bool) -> None:
    """Write global ``warnings_on`` and every known warning leaf flag."""
    leaf_states: dict[tuple[str, str], bool] = {}
    for group_name, group in vars(WarningType).items():
        if group_name.startswith("_"):
            continue
        if not (isinstance(group, type) and issubclass(group, Enum)):
            continue
        for member in group:
            leaf_states[(group_name, member.name)] = enabled
    _update_config_lines(warnings_on=enabled, leaf_states=leaf_states)


def _defaults_assignment_key(stripped: str) -> str | None:
    """Return the defaults key on an assignment line, commented or not."""
    code = stripped
    if code.startswith("#"):
        code = code[1:].strip()
    if "=" not in code:
        return None
    key_part, _, _value_part = code.partition("=")
    key_name = key_part.strip()
    if not _is_toml_bare_key(key_name):
        return None
    return key_name


def _update_default_keys(toml_literals: dict[str, str]) -> None:
    """Uncomment or insert ``[defaults]`` assignments in the personal toml."""
    config_path = ensure_user_config()
    text = config_path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    current_section: str | None = None
    defaults_start: int | None = None
    defaults_end: int | None = None
    live_index: dict[str, int] = {}
    comment_index: dict[str, int] = {}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip()
            if current_section == "defaults" and section != "defaults":
                defaults_end = index
            current_section = section
            if section == "defaults" and defaults_start is None:
                defaults_start = index
            continue
        if current_section != "defaults":
            continue
        key_name = _defaults_assignment_key(stripped)
        if key_name is None:
            continue
        if stripped.startswith("#"):
            if key_name not in comment_index:
                comment_index[key_name] = index
        else:
            live_index[key_name] = index
    if current_section == "defaults":
        defaults_end = len(lines)

    new_lines = list(lines)

    def replace_defaults_line(index: int, key_name: str) -> None:
        line = new_lines[index]
        indent = line[: len(line) - len(line.lstrip(" \t"))]
        stripped = line.strip()
        trailing = ""
        if not stripped.startswith("#") and "#" in stripped:
            trailing = " #" + stripped.split("#", 1)[1]
        new_lines[index] = (
            f"{indent}{key_name} = {toml_literals[key_name]}{trailing}{newline}"
        )

    replaced: set[str] = set()
    for key_name in toml_literals:
        if key_name in live_index:
            replace_defaults_line(live_index[key_name], key_name)
            replaced.add(key_name)
        elif key_name in comment_index:
            replace_defaults_line(comment_index[key_name], key_name)
            replaced.add(key_name)

    missing_keys = [key for key in toml_literals if key not in replaced]
    extra = [f"{key} = {toml_literals[key]}{newline}" for key in missing_keys]
    if extra:
        extra.append(newline)
        if defaults_start is None:
            if new_lines:
                last_line = new_lines[-1]
                if not last_line.endswith("\n"):
                    new_lines[-1] = last_line + newline
                if new_lines[-1].strip():
                    new_lines.append(newline)
            new_lines.append(f"[defaults]{newline}")
            new_lines.extend(extra)
        else:
            insert_at = defaults_end if defaults_end is not None else len(new_lines)
            new_lines = new_lines[:insert_at] + extra + new_lines[insert_at:]
    config_path.write_text("".join(new_lines), encoding="utf-8")


def _apply_paths(paths_table: dict[str, Any]) -> None:
    """Apply the ``[paths]`` table."""
    known = frozenset(_user_paths)
    for key, value in paths_table.items():
        if key not in known:
            _warn_invalid_key(
                f"Unknown key in [paths]: {key!r} (simetri_config.toml)"
            )
            continue
        if value is None:
            _user_paths[key] = ""
        else:
            _user_paths[key] = str(value)


def _apply_defaults_table(defaults_table: dict[str, Any]) -> None:
    """Apply uncommented ``[defaults]`` entries into user overrides."""
    from .settings import default_types, defaults

    for key, value in defaults_table.items():
        if key not in defaults.defaults:
            _warn_invalid_key(
                f"Unknown key in [defaults]: {key!r} (simetri_config.toml)"
            )
            continue
        if key not in default_types:
            _warn_invalid_key(
                f"Key {key!r} in [defaults] has no registered type "
                "(simetri_config.toml)"
            )
            continue
        try:
            converted = _convert_default_value(key, value, default_types[key])
        except (TypeError, ValueError, KeyError) as error:
            _warn_invalid_key(
                f"Invalid value for [defaults].{key}: {value!r} ({error})"
            )
            continue
        _set_user_default_override(key, converted)


def _warning_member(group_name: str, leaf_name: str) -> Any | None:
    """Return ``WarningType.<group>.<leaf>`` or None if unknown."""
    if group_name not in vars(WarningType):
        return None
    group = vars(WarningType)[group_name]
    if not (isinstance(group, type) and issubclass(group, Enum)):
        return None
    if leaf_name not in group.__members__:
        return None
    return group[leaf_name]


def _apply_warnings_table(warnings_table: dict[str, Any]) -> None:
    """Apply ``[warnings]`` and nested ``[warnings.<group>]`` tables."""
    from .settings import (
        _apply_warning_enabled_session,
        _disabled_warning_types,
    )

    _disabled_warning_types.clear()

    for key, value in warnings_table.items():
        if key == "warnings_on":
            _set_user_default_override("show_warnings", bool(value))
            continue
        if not isinstance(value, dict):
            _warn_invalid_key(
                f"Unknown key in [warnings]: {key!r} (simetri_config.toml)"
            )
            continue
        group_name = key
        group_cls = (
            vars(WarningType)[group_name]
            if group_name in vars(WarningType)
            else None
        )
        if not (
            isinstance(group_cls, type) and issubclass(group_cls, Enum)
        ):
            _warn_invalid_key(
                f"Unknown warning group [warnings.{group_name}] "
                "(simetri_config.toml)"
            )
            continue
        for leaf_name, enabled in value.items():
            member = _warning_member(group_name, leaf_name)
            if member is None:
                _warn_invalid_key(
                    f"Unknown warning [warnings.{group_name}.{leaf_name}] "
                    "(simetri_config.toml)"
                )
                continue
            _apply_warning_enabled_session(member, enabled=bool(enabled))

    if "warnings_on" in warnings_table:
        _set_user_default_override(
            "show_warnings", bool(warnings_table["warnings_on"])
        )


def _apply_converters_table(converters_table: dict[str, Any]) -> None:
    """Apply personal ``[converters]`` and ``[converters.<format>]`` tables."""
    _converter_formats.clear()
    _converter_globals["enabled"] = True
    _converter_globals["timeout_seconds"] = 120
    _converter_globals["shell"] = False

    global_keys = frozenset({"enabled", "timeout_seconds", "shell"})
    for key, value in converters_table.items():
        if key in global_keys:
            if key == "enabled":
                _converter_globals["enabled"] = bool(value)
            elif key == "timeout_seconds":
                timeout = float(value)
                if timeout <= 0:
                    _warn_invalid_key(
                        "[converters].timeout_seconds must be positive "
                        f"(got {value!r})"
                    )
                    continue
                _converter_globals["timeout_seconds"] = timeout
            else:
                _converter_globals["shell"] = bool(value)
            continue

        if not isinstance(value, dict):
            _warn_invalid_key(
                f"Unknown key in [converters]: {key!r} "
                "(expected a [converters.<format>] table)"
            )
            continue

        format_key = str(key).lstrip(".").lower()
        if not format_key:
            _warn_invalid_key(
                f"Invalid converter format name {key!r} (simetri_config.toml)"
            )
            continue

        if "source" not in value:
            _warn_invalid_key(
                f"[converters.{format_key}] missing required key 'source'"
            )
            continue
        if "command" not in value:
            _warn_invalid_key(
                f"[converters.{format_key}] missing required key 'command'"
            )
            continue

        source = str(value["source"]).lstrip(".").lower()
        if source not in _CONVERTER_SOURCES:
            _warn_invalid_key(
                f"[converters.{format_key}].source must be one of "
                f"{sorted(_CONVERTER_SOURCES)} (got {value['source']!r})"
            )
            continue

        command = value["command"]
        if isinstance(command, str):
            command_value: str | list[str] = command
        elif isinstance(command, list):
            if not command:
                _warn_invalid_key(
                    f"[converters.{format_key}].command must not be empty"
                )
                continue
            command_value = [str(part) for part in command]
        else:
            _warn_invalid_key(
                f"[converters.{format_key}].command must be a string or "
                f"array of strings (got {type(command).__name__})"
            )
            continue

        for extra_key in value:
            if extra_key not in ("source", "command"):
                _warn_invalid_key(
                    f"Unknown key in [converters.{format_key}]: {extra_key!r}"
                )

        _converter_formats[format_key] = {
            "source": source,
            "command": command_value,
        }


def _reset_tex_settings() -> None:
    """Restore personal ``[tex]`` settings to library defaults."""
    _tex_settings["timeout_seconds"] = 120
    _tex_settings["shell"] = False
    _tex_settings["command"] = None


def _reset_viewer_settings() -> None:
    """Restore personal ``[viewer]`` settings to library defaults."""
    _viewer_settings["mode"] = "system"
    _viewer_settings["shell"] = False
    _viewer_settings["command"] = None


def _apply_tex_table(tex_table: dict[str, Any]) -> None:
    """Apply personal ``[tex]`` compiler table."""
    _reset_tex_settings()
    known_keys = frozenset({"timeout_seconds", "shell", "command"})
    for key, value in tex_table.items():
        if key not in known_keys:
            _warn_invalid_key(f"Unknown key in [tex]: {key!r}")
            continue
        if key == "timeout_seconds":
            timeout = float(value)
            if timeout <= 0:
                _warn_invalid_key(
                    f"[tex].timeout_seconds must be positive (got {value!r})"
                )
                continue
            _tex_settings["timeout_seconds"] = timeout
        elif key == "shell":
            _tex_settings["shell"] = bool(value)
        else:
            if isinstance(value, str):
                if not value:
                    _warn_invalid_key("[tex].command must not be empty")
                    continue
                _tex_settings["command"] = value
            elif isinstance(value, list):
                if not value:
                    _warn_invalid_key("[tex].command must not be empty")
                    continue
                _tex_settings["command"] = [str(part) for part in value]
            else:
                _warn_invalid_key(
                    "[tex].command must be a string or array of strings "
                    f"(got {type(value).__name__})"
                )


def _apply_viewer_table(viewer_table: dict[str, Any]) -> None:
    """Apply personal ``[viewer]`` table for ``canvas.save`` previews."""
    _reset_viewer_settings()
    known_keys = frozenset({"mode", "shell", "command"})
    for key, value in viewer_table.items():
        if key not in known_keys:
            _warn_invalid_key(f"Unknown key in [viewer]: {key!r}")
            continue
        if key == "mode":
            mode = str(value).strip().lower()
            if mode not in _VIEWER_MODES:
                _warn_invalid_key(
                    "[viewer].mode must be 'system', 'command', or 'none' "
                    f"(got {value!r})"
                )
                continue
            _viewer_settings["mode"] = mode
        elif key == "shell":
            _viewer_settings["shell"] = bool(value)
        else:
            if isinstance(value, str):
                if not value:
                    _warn_invalid_key("[viewer].command must not be empty")
                    continue
                _viewer_settings["command"] = value
            elif isinstance(value, list):
                if not value:
                    _warn_invalid_key("[viewer].command must not be empty")
                    continue
                _viewer_settings["command"] = [str(part) for part in value]
            else:
                _warn_invalid_key(
                    "[viewer].command must be a string or array of strings "
                    f"(got {type(value).__name__})"
                )


def _apply_styles_table(styles_table: dict[str, Any]) -> None:
    """Apply personal ``[styles.<name>]`` tables into ``user_styles``."""
    from .settings import default_types
    from ..base.common_style import STYLE_ALIAS_KEYS, Style

    user_styles.clear()
    for style_name, body in styles_table.items():
        if not isinstance(body, dict):
            raise TypeError(
                f"[styles.{style_name}] in simetri_config.toml must be a table, "
                f"got {type(body).__name__}"
            )
        converted: dict[str, Any] = {}
        for key, value in body.items():
            if key not in STYLE_ALIAS_KEYS:
                raise ValueError(
                    f"Unknown style key {key!r} in [styles.{style_name}]"
                )
            if key not in default_types:
                raise ValueError(
                    f"Key {key!r} in [styles.{style_name}] has no registered type"
                )
            converted[key] = _convert_default_value(key, value, default_types[key])
        user_styles[style_name] = Style(converted)


def _is_toml_bare_key(name: str) -> bool:
    """Return True if ``name`` is a TOML bare key (letters, digits, ``_``, ``-``)."""
    if not name:
        return False
    for character in name:
        is_letter = ("A" <= character <= "Z") or ("a" <= character <= "z")
        is_digit = "0" <= character <= "9"
        if not (is_letter or is_digit or character in "_-"):
            return False
    return True


def _upsert_toml_table(config_path: Path, table_header: str, body_lines: list[str]) -> None:
    """Replace or append ``[table_header]``, preserving other file contents."""
    text = config_path.read_text(encoding="utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    start: int | None = None
    end: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not (stripped.startswith("[") and stripped.endswith("]")):
            continue
        section = stripped[1:-1].strip()
        if start is not None:
            end = index
            break
        if section == table_header:
            start = index
    replacement = [f"[{table_header}]{newline}"]
    replacement.extend(f"{body_line}{newline}" for body_line in body_lines)
    replacement.append(newline)
    if start is not None:
        if end is None:
            end = len(lines)
        new_lines = lines[:start] + replacement + lines[end:]
    else:
        new_lines = list(lines)
        if new_lines:
            last_line = new_lines[-1]
            if not last_line.endswith("\n"):
                new_lines[-1] = last_line + newline
            if new_lines[-1].strip():
                new_lines.append(newline)
        new_lines.extend(replacement)
    config_path.write_text("".join(new_lines), encoding="utf-8")


def save_user_style(
    name: str, mapping: Any = None, **kwargs: object
) -> Path:
    """Write ``[styles.<name>]`` to the personal ``simetri_config.toml``.

    Overwrites that table if it already exists. Updates ``user_styles``
    in this session. Fields whose value is ``None`` are not written.

    Args:
        name: Style name. Must be a TOML bare key (letters, digits, ``_``, ``-``).
        mapping: A ``Style``, a dict of draw aliases, or omitted.
        **kwargs: Draw-alias fields; overwrite ``mapping`` for those keys.

    Returns:
        Path to the personal config file.

    Examples:

        >>> path = sg.save_user_style(
        ...     "outline", fill=False, line_width=3, line_color=sg.blue
        ... )
        >>> path.name
        'simetri_config.toml'
        >>> "outline" in sg.user_styles
        True
    """
    from ..base.common_style import Style, coerce_style_overlay

    if mapping is None and not kwargs:
        raise TypeError(
            "save_user_style() requires a Style, a dict, or keyword arguments"
        )
    if not _is_toml_bare_key(name):
        raise ValueError(
            f"Style name {name!r} is not a TOML bare key "
            "(use letters, digits, '_' or '-')"
        )
    overlay = coerce_style_overlay(mapping, kwargs)
    written: dict[str, Any] = {}
    body_lines: list[str] = []
    for key in sorted(overlay):
        value = overlay[key]
        if value is None:
            continue
        serialized = _serialize_shared_value(value)
        written[key] = value
        body_lines.append(f"{key} = {_format_toml_value(serialized)}")
    if not body_lines:
        raise ValueError(
            f"Style {name!r} has no fields to save (all values are None)"
        )
    config_path = ensure_user_config()
    _upsert_toml_table(config_path, f"styles.{name}", body_lines)
    user_styles[name] = Style(written)
    return config_path


def save_user_defaults(mapping: Any = None, **kwargs: object) -> Path:
    """Write ``[defaults]`` keys to the personal ``simetri_config.toml``.

    Uncomments an existing catalog line for that key when present; otherwise
    inserts the assignment under ``[defaults]``. Updates user overrides in
    this session. Does not change factory ``sg.defaults``.

    Args:
        mapping: A dict of defaults keys to values, or omitted.
        **kwargs: Defaults keys; overwrite ``mapping`` for those keys.

    Returns:
        Path to the personal config file.

    Raises:
        TypeError: No mapping or kwargs, or a value cannot be serialized.
        KeyError: Unknown defaults key, or the key has no registered type.

    Examples:

        >>> path = sg.save_user_defaults(line_width=1.5, page_size="A4")
        >>> path.name
        'simetri_config.toml'
        >>> sg.save_user_defaults({"fill_color": sg.blue}).suffix
        '.toml'
        >>> _ = sg.save_user_defaults(line_width=2.5)
        >>> sg.user_defaults["line_width"]
        2.5
    """
    from .settings import _default_store, default_types, defaults

    if mapping is None and not kwargs:
        raise TypeError(
            "save_user_defaults() requires a dict or keyword arguments"
        )
    updates: dict[str, Any] = {}
    if mapping is not None:
        if not isinstance(mapping, dict):
            raise TypeError(
                "save_user_defaults() mapping must be a dict, "
                f"got {type(mapping).__name__}"
            )
        updates.update(mapping)
    updates.update(kwargs)
    if not updates:
        raise TypeError(
            "save_user_defaults() requires at least one defaults key"
        )

    converted_map: dict[str, Any] = {}
    toml_literals: dict[str, str] = {}
    for key in sorted(updates):
        if key not in defaults.defaults:
            raise KeyError(
                f"Unknown defaults key {key!r} (simetri_config.toml)"
            )
        if key not in default_types:
            raise KeyError(
                f"Defaults key {key!r} has no registered type"
            )
        expected_type = default_types[key]
        if not _is_exportable_default_type(expected_type):
            raise TypeError(
                f"Cannot serialize defaults key {key!r} of type "
                f"{expected_type!r}"
            )
        converted = _convert_default_value(key, updates[key], expected_type)
        serialized = _serialize_shared_value(converted)
        converted_map[key] = converted
        toml_literals[key] = _format_toml_value(serialized)

    _update_default_keys(toml_literals)
    for key, converted in converted_map.items():
        _set_user_default_override(key, converted)
    if _default_store.user_overrides is not _user_default_overrides:
        _default_store.user_overrides = _user_default_overrides
    return user_config_path()


def apply_user_config() -> Path:
    """Ensure, load, and apply ``simetri_config.toml``.

    Called once when you ``import simetri.graphics as sg``. Hand-edits to the
    toml while a session is already running are not picked up until you restart
    the kernel, re-import in a fresh process, or call this function again.
    APIs such as ``sg.save_user_defaults`` update the file and this session
    immediately.

    Returns:
        Path to the config file that was read.

    Examples:

        >>> sg.apply_user_config().name
        'simetri_config.toml'
    """
    global _config_applied
    from .settings import _default_store, defaults

    config_path = ensure_user_config()
    with _internal_user_default_writes():
        _user_default_overrides.clear()
    _user_paths["default_output_directory"] = ""
    _user_paths["default_test_directory"] = ""
    _converter_formats.clear()
    _converter_globals["enabled"] = True
    _converter_globals["timeout_seconds"] = 120
    _converter_globals["shell"] = False
    _reset_tex_settings()
    _reset_viewer_settings()
    user_styles.clear()
    _script_export_settings["include_script_header"] = False
    _script_export_settings["include_defaults"] = False
    _script_export_settings["author"] = ""
    _script_export_settings["copyright"] = ""
    _script_export_settings["description"] = ""
    _script_export_settings["doc_version"] = "1.0.0"
    _script_export_settings["license"] = ""
    _script_export_custom_slots.clear()

    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    known_sections = frozenset(
        {
            "paths",
            "warnings",
            "defaults",
            "converters",
            "tex",
            "viewer",
            "styles",
            "script_export",
        }
    )
    for section_name in data:
        if section_name not in known_sections:
            _warn_invalid_key(
                f"Unknown section [{section_name}] in simetri_config.toml"
            )

    if "paths" in data:
        _apply_paths(data["paths"])
    if "defaults" in data:
        _apply_defaults_table(data["defaults"])
    if "warnings" in data:
        _apply_warnings_table(data["warnings"])
    if "converters" in data:
        _apply_converters_table(data["converters"])
    if "tex" in data:
        tex_table = data["tex"]
        if not isinstance(tex_table, dict):
            raise TypeError(
                "[tex] in simetri_config.toml must be a table, "
                f"got {type(tex_table).__name__}"
            )
        _apply_tex_table(tex_table)
    if "viewer" in data:
        viewer_table = data["viewer"]
        if not isinstance(viewer_table, dict):
            raise TypeError(
                "[viewer] in simetri_config.toml must be a table, "
                f"got {type(viewer_table).__name__}"
            )
        _apply_viewer_table(viewer_table)
    if "styles" in data:
        styles_table = data["styles"]
        if not isinstance(styles_table, dict):
            raise TypeError(
                "[styles] in simetri_config.toml must be a table, "
                f"got {type(styles_table).__name__}"
            )
        _apply_styles_table(styles_table)
    if "script_export" in data:
        export_table = data["script_export"]
        if not isinstance(export_table, dict):
            raise TypeError(
                "[script_export] in simetri_config.toml must be a table, "
                f"got {type(export_table).__name__}"
            )
        _apply_script_export_table(export_table)

    _default_store.user_overrides = _user_default_overrides
    _config_applied = True
    return config_path


def _color_toml_value(color: Color) -> str | list[float]:
    """Serialize a ``Color`` as a named string or RGB list."""
    for name, named_color in colors.__dict__.items():
        if isinstance(named_color, Color) and named_color == color:
            return name
    red, green, blue, alpha = color.rgba
    if alpha == 1.0:
        return [red, green, blue]
    return [red, green, blue, alpha]


def _serialize_shared_value(value: Any) -> Any:
    """Convert a defaults value to a TOML-friendly Python object."""
    from .settings import VOID

    if value is VOID:
        raise TypeError("VOID cannot be written to a shared settings toml")
    if isinstance(value, bool):
        return value
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Color):
        return _color_toml_value(value)
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_serialize_shared_value(item) for item in value]
    raise TypeError(
        f"Cannot serialize default value of type {type(value).__name__}: {value!r}"
    )


def _format_toml_value(value: Any) -> str:
    """Format a Python value as a TOML literal."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        items = ", ".join(_format_toml_value(item) for item in value)
        return f"[{items}]"
    raise TypeError(f"Cannot format TOML value: {value!r}")


def _is_exportable_default_type(expected_type: Any) -> bool:
    """Return True if ``expected_type`` can round-trip through shared toml."""
    if expected_type is object:
        return False
    if isinstance(expected_type, tuple):
        return all(_is_exportable_default_type(item) for item in expected_type)
    if expected_type is Sequence:
        return True
    if expected_type is Color or (
        isinstance(expected_type, type) and issubclass(expected_type, Color)
    ):
        return True
    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        return True
    return expected_type in (bool, int, float, str)


def _apply_script_export_table(export_table: dict[str, Any]) -> None:
    """Apply uncommented ``[script_export]`` and ``[script_export.custom]`` entries."""
    known = frozenset(_script_export_settings)
    for key, value in export_table.items():
        if key == "custom":
            if isinstance(value, dict):
                _script_export_custom_slots.clear()
                for slot_name, slot_value in value.items():
                    _script_export_custom_slots[str(slot_name)] = str(slot_value)
            else:
                _warn_invalid_key(
                    "[script_export].custom must be a table (simetri_config.toml)"
                )
            continue
        if key not in known:
            _warn_invalid_key(
                f"Unknown key in [script_export]: {key!r} (simetri_config.toml)"
            )
            continue
        if key in ("include_script_header", "include_defaults"):
            _script_export_settings[key] = bool(value)
        else:
            _script_export_settings[key] = str(value)


def _capture_exportable_defaults_from_script(script_path: Path) -> dict[str, Any]:
    """Run ``script_path`` and return exportable defaults read during the run."""
    from .settings import _default_store, default_types, runtime_defaults

    _default_store.log.clear()
    runpy.run_path(str(script_path), run_name="__main__")

    accessed_keys = sorted({key for key, _str_value in _default_store.log})
    defaults_table: dict[str, Any] = {}
    for key in accessed_keys:
        if key not in default_types:
            continue
        expected_type = default_types[key]
        if not _is_exportable_default_type(expected_type):
            continue
        value = runtime_defaults[key]
        defaults_table[key] = _serialize_shared_value(value)
    return defaults_table


def _extract_script_body(source: str) -> str:
    """Return script text after shebang, encoding cookie, and module docstring."""
    lines = source.splitlines(keepends=True)
    offset = 0
    if lines and lines[0].startswith("#!"):
        offset = 1
    if offset < len(lines) and "coding" in lines[offset] and "utf-8" in lines[offset]:
        offset += 1
    chunk = "".join(lines[offset:])
    try:
        tree = ast.parse(chunk, mode="exec")
    except SyntaxError:
        return chunk
    if not tree.body:
        return chunk
    first = tree.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        body_index = offset + first.end_lineno
        return "".join(lines[body_index:])
    return chunk


def _format_script_header_default_line(key: str, serialized: Any) -> str:
    """Format one ``[defaults]`` line inside a script header docstring."""
    return f"{key} = {_format_toml_value(serialized)}"


def _build_script_header_docstring(
    *,
    filename: str,
    author: str,
    export_date: str,
    simetri_version: str,
    doc_version: str,
    description: str,
    copyright_text: str,
    license_text: str,
    custom_slots: dict[str, str],
    defaults_table: dict[str, Any],
) -> str:
    """Build the module docstring for ``save_as`` (metadata + optional defaults)."""
    lines = [
        f"Filename: {filename}",
        f"Author: {author}",
        f"Date: {export_date}",
        f"sg.__version__: {simetri_version}",
        f"doc_version: {doc_version}",
        f"Description: {description}",
    ]
    if copyright_text.strip():
        lines.append(f"Copyright: {copyright_text.strip()}")
    if license_text.strip():
        lines.append(f"License: {license_text.strip()}")
    for slot_name in sorted(custom_slots):
        slot_value = custom_slots[slot_name].strip()
        if slot_value:
            lines.append(f"{slot_name}: {slot_value}")
    if defaults_table:
        lines.append("[defaults]")
        lines.extend(
            _format_script_header_default_line(key, defaults_table[key])
            for key in sorted(defaults_table)
        )
    body = "\n".join(lines)
    return f'"""\n{body}\n"""\n\n'


def _parse_header_assignment_value(literal: str) -> Any:
    """Parse one ``[defaults]`` rhs from a script header docstring."""
    text = literal.strip()
    lower = text.lower()
    if lower in ("true", "false"):
        return lower == "true"
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        pass
    table = tomllib.loads(f"value = {text}".encode("utf-8"))
    return table["value"]


def _parse_script_header_defaults(docstring: str) -> dict[str, str]:
    """Parse ``[defaults]`` assignment lines from a script module docstring."""
    defaults_lines: dict[str, str] = {}
    in_defaults = False
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped == "[defaults]":
            in_defaults = True
            continue
        if not in_defaults or not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key_part, _, value_part = stripped.partition("=")
        key_name = key_part.strip()
        if key_name:
            defaults_lines[key_name] = value_part.strip()
    return defaults_lines


def _parse_script_header_meta(docstring: str) -> dict[str, str]:
    """Parse ``Key: value`` metadata lines from a script module docstring."""
    meta: dict[str, str] = {}
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped == "[defaults]" or not stripped:
            break
        if ":" not in stripped:
            continue
        key_part, _, value_part = stripped.partition(":")
        key_name = key_part.strip()
        if key_name:
            meta[key_name] = value_part.strip()
    return meta


def _load_script_header_defaults(docstring: str) -> dict[str, Any]:
    """Convert parsed header defaults into typed runtime values."""
    from .settings import default_types, defaults

    raw = _parse_script_header_defaults(docstring)
    overlay: dict[str, Any] = {}
    for key, literal in raw.items():
        if key not in defaults.defaults:
            raise KeyError(f"Unknown defaults key {key!r} in script header")
        if key not in default_types:
            raise KeyError(
                f"Defaults key {key!r} has no registered type (script header)"
            )
        overlay[key] = _convert_default_value(
            key, _parse_header_assignment_value(literal), default_types[key]
        )
    return overlay


def _script_module_docstring(source: str) -> str | None:
    """Return the module docstring text from ``source``, if present."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    return ast.get_docstring(tree, clean=False)


def _resolve_save_as_bool(name: str, override: bool | None) -> bool:
    if override is not None:
        return bool(override)
    return bool(_script_export_settings.get(name, False))


def _resolve_save_as_str(name: str, override: str | None) -> str:
    if override is not None:
        return str(override)
    return str(_script_export_settings.get(name, ""))


def _resolve_save_as_custom(
    override: dict[str, str] | None,
) -> dict[str, str]:
    merged = dict(_script_export_custom_slots)
    if override:
        for key, value in override.items():
            merged[str(key)] = str(value)
    return merged


def save_as(
    source_path: str | Path,
    target_path: str | Path,
    *,
    include_script_header: bool | None = None,
    include_defaults: bool | None = None,
    doc_version: str | None = None,
    simetri_version: str | None = None,
    author: str | None = None,
    export_date: str | None = None,
    description: str | None = None,
    filename: str | None = None,
    copyright: str | None = None,
    license: str | None = None,
    custom: dict[str, str] | None = None,
) -> Path:
    """Write a copy of a script to ``target_path`` with an optional Simetri header.

    Runs ``source_path`` once to capture which defaults the drawing reads when
    ``include_defaults`` is true. Never modifies ``source_path``. Personal
    ``[script_export]`` keys in ``simetri_config.toml`` apply when call-time
    arguments are omitted.

    Args:
        source_path: Script to run for capture and to copy body from.
        target_path: Destination ``.py`` file (created or overwritten).
        include_script_header: When true, write shebang, encoding, and metadata
            docstring (all metadata fields together). When false, only shebang
            and body.
        include_defaults: When true, append a ``[defaults]`` block to the header
            docstring from the capture run.
        doc_version: Header schema version (script header parser).
        simetri_version: Minimum Simetri version recorded in the header.
        author: Author line in the header docstring.
        export_date: ``Date:`` line (defaults to today, ISO).
        description: Description line in the header docstring.
        filename: ``Filename:`` line (defaults to ``target_path`` name).
        copyright: Optional ``Copyright:`` line (omitted when empty).
        license: Optional ``License:`` line (omitted when empty).
        custom: Extra ``Label: value`` header lines; merged with
            ``[script_export.custom]`` from ``simetri_config.toml``.

    Returns:
        Path to ``target_path``.

    Raises:
        FileNotFoundError: If ``source_path`` does not exist.
        TypeError: If a captured default cannot be serialized.

    Examples:

        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     src = Path(tmp) / "figure.py"
        ...     dst = Path(tmp) / "figure_export.py"
        ...     _ = src.write_text(
        ...         "import simetri.graphics as sg\\n"
        ...         "from simetri.config.settings import runtime_defaults\\n"
        ...         "_ = runtime_defaults['line_width']\\n",
        ...         encoding="utf-8",
        ...     )
        ...     out = sg.save_as(
        ...         src,
        ...         dst,
        ...         include_script_header=True,
        ...         include_defaults=True,
        ...     )
        ...     out == dst and dst.is_file()
        True
    """
    from .. import __version__

    source = Path(source_path)
    target = Path(target_path)
    if not source.is_file():
        raise FileNotFoundError(f"Script not found: {source}")

    include_header = _resolve_save_as_bool(
        "include_script_header", include_script_header
    )
    include_defs = _resolve_save_as_bool("include_defaults", include_defaults)

    defaults_table: dict[str, Any] = {}
    if include_defs:
        defaults_table = _capture_exportable_defaults_from_script(source)

    source_text = source.read_text(encoding="utf-8")
    body = _extract_script_body(source_text)

    parts: list[str] = ["#!/usr/bin/env python3\n"]
    if include_header:
        parts.append("# -*- coding: utf-8 -*-\n")
        resolved_version = (
            str(simetri_version)
            if simetri_version is not None
            else __version__
        )
        parts.append(
            _build_script_header_docstring(
                filename=filename if filename is not None else target.name,
                author=_resolve_save_as_str("author", author),
                export_date=export_date or date.today().isoformat(),
                simetri_version=resolved_version,
                doc_version=_resolve_save_as_str(
                    "doc_version",
                    doc_version if doc_version is not None else None,
                ),
                description=_resolve_save_as_str("description", description),
                copyright_text=_resolve_save_as_str("copyright", copyright),
                license_text=_resolve_save_as_str("license", license),
                custom_slots=_resolve_save_as_custom(custom),
                defaults_table=defaults_table if include_defs else {},
            )
        )
    parts.append(body.lstrip("\n") if body else "")
    if parts[-1] and not parts[-1].endswith("\n"):
        parts[-1] = parts[-1] + "\n"

    target.write_text("".join(parts), encoding="utf-8")
    return target


@contextmanager
def use_script_header(script_path: str | Path) -> Iterator[None]:
    """Apply ``[defaults]`` from a script module docstring for a ``with`` block.

    Reads ``sg.__version__`` / ``simetri_version`` from the header when present
    and calls ``check_version``. Installs header defaults as shared overrides
    (personal ``simetri_config.toml`` defaults are suppressed). Does not write
    personal config.

    Args:
        script_path: Path to a ``.py`` file (typically ``__file__``).

    Yields:
        None.

    Raises:
        FileNotFoundError: If ``script_path`` does not exist.
        KeyError: Unknown defaults key in the header.
        VersionConflict: If this install is older than the header version.

    Examples:

        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     script = Path(tmp) / "fig.py"
        ...     _ = script.write_text(
        ...         '\\\"\\\"\\\"\\nsg.__version__: 0.0.0\\n'
        ...         '[defaults]\\nline_width = 2.0\\n\\\"\\\"\\\"\\n',
        ...         encoding="utf-8",
        ...     )
        ...     with sg.use_script_header(script):
        ...         pass
    """
    from .settings import _default_store

    path = Path(script_path)
    if not path.is_file():
        raise FileNotFoundError(f"Script not found: {path}")

    docstring = _script_module_docstring(path.read_text(encoding="utf-8"))
    overlay: dict[str, Any] = {}
    if docstring:
        meta = _parse_script_header_meta(docstring)
        version_text = meta.get("sg.__version__") or meta.get("simetri_version")
        if version_text:
            check_version(version_text.strip())
        overlay = _load_script_header_defaults(docstring)

    previous_shared = dict(_default_store.shared_overrides)
    previous_suppress = _default_store.suppress_user_overrides
    _default_store.shared_overrides = overlay
    _default_store.suppress_user_overrides = True
    try:
        yield
    finally:
        _default_store.shared_overrides = previous_shared
        _default_store.suppress_user_overrides = previous_suppress


def _write_shared_toml(
    output_path: Path,
    *,
    simetri_version: str,
    defaults_table: dict[str, Any],
) -> None:
    """Write ``[meta]`` and ``[defaults]`` for a shared settings file."""
    lines = [
        "[meta]",
        f"simetri_version = {_format_toml_value(simetri_version)}",
        "",
        "[defaults]",
    ]
    lines.extend(
        f"{key} = {_format_toml_value(defaults_table[key])}"
        for key in sorted(defaults_table)
    )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_shared_toml(script_path: str | Path, output_path: str | Path) -> Path:
    """Run ``script_path`` and write accessed defaults to a shared settings toml.

    Clears ``defaults.log``, executes the script with ``runpy.run_path``, then
    writes ``[meta].simetri_version`` and every exportable default key that was
    read during the run to ``output_path``.

    Args:
        script_path: Path to the Python script to run under capture.
        output_path: Destination path for the shared settings toml.

    Returns:
        ``Path`` to the written toml file.

    Raises:
        FileNotFoundError: If ``script_path`` does not exist.
        TypeError: If an accessed exportable default cannot be serialized.

    Examples:

        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     script = Path(tmp) / "read_defaults.py"
        ...     _ = script.write_text(
        ...         "import simetri.graphics as sg\\n"
        ...         "_ = sg.defaults['line_width']\\n",
        ...         encoding="utf-8",
        ...     )
        ...     out = Path(tmp) / "shared.toml"
        ...     _ = sg.generate_shared_toml(script, out)
        ...     out.is_file()
        True
    """
    from .. import __version__

    script = Path(script_path)
    output = Path(output_path)
    if not script.is_file():
        raise FileNotFoundError(f"Script not found: {script}")

    defaults_table = _capture_exportable_defaults_from_script(script)

    _write_shared_toml(
        output,
        simetri_version=__version__,
        defaults_table=defaults_table,
    )
    return output


def _load_shared_settings(toml_path: Path) -> tuple[str, dict[str, Any]]:
    """Load version and converted defaults from a shared settings toml."""
    from .settings import default_types, defaults

    if not toml_path.is_file():
        raise FileNotFoundError(f"Shared settings file not found: {toml_path}")

    with toml_path.open("rb") as handle:
        data = tomllib.load(handle)

    if "meta" not in data:
        raise ValueError(f"Missing [meta] section in {toml_path}")
    meta = data["meta"]
    if "simetri_version" not in meta:
        raise ValueError(f"Missing meta.simetri_version in {toml_path}")
    simetri_version = str(meta["simetri_version"])

    if "defaults" not in data:
        raise ValueError(f"Missing [defaults] section in {toml_path}")
    defaults_table = data["defaults"]

    overlay: dict[str, Any] = {}
    for key, value in defaults_table.items():
        if key not in defaults.defaults:
            raise KeyError(
                f"Unknown defaults key {key!r} in shared settings {toml_path}"
            )
        if key not in default_types:
            raise KeyError(
                f"Defaults key {key!r} has no registered type "
                f"(shared settings {toml_path})"
            )
        overlay[key] = _convert_default_value(key, value, default_types[key])
    return simetri_version, overlay


@contextmanager
def use_settings(toml_path: str | Path) -> Iterator[None]:
    """Apply a shared settings toml for the duration of a ``with`` block.

    Reads ``[meta].simetri_version`` and ``[defaults]`` from ``toml_path``,
    checks that this Simetri is at least that version, then installs those
    values as ``shared_overrides`` (personal ``[defaults]`` from
    ``simetri_config.toml`` are suppressed for the block).

    Args:
        toml_path: Path to a toml produced by ``generate_shared_toml``.

    Yields:
        None.

    Raises:
        FileNotFoundError: If ``toml_path`` does not exist.
        ValueError: If required sections/keys are missing.
        VersionConflict: If this package is older than ``simetri_version``.

    Examples:

        >>> import tempfile
        >>> from pathlib import Path
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     toml = Path(tmp) / "shared.toml"
        ...     _ = toml.write_text(
        ...         '[meta]\\nsimetri_version = "0.0.0"\\n\\n'
        ...         '[defaults]\\nline_width = 2.0\\n',
        ...         encoding="utf-8",
        ...     )
        ...     with sg.use_settings(toml):
        ...         sg.user_defaults["line_width"]
        2.0
    """
    from .settings import _default_store, defaults

    path = Path(toml_path)
    simetri_version, overlay = _load_shared_settings(path)
    check_version(simetri_version)

    previous_shared = dict(_default_store.shared_overrides)
    previous_suppress = _default_store.suppress_user_overrides
    _default_store.shared_overrides = overlay
    _default_store.suppress_user_overrides = True
    try:
        yield
    finally:
        _default_store.shared_overrides = previous_shared
        _default_store.suppress_user_overrides = previous_suppress

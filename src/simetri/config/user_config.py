"""Load and apply per-user Simetri settings from ``simetri_config.toml``."""

from __future__ import annotations

import json
import platform
import runpy
import tomllib
from collections.abc import Sequence
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

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
_user_default_overrides: dict[str, Any] = {}
_config_applied = False


def set_user_settings_path() -> Path:
    """Return (and create) the OS-specific ``simetri_user`` config directory.

    Source - https://stackoverflow.com/a/77658488
    Posted by Het Vaghani
    Retrieved 2026-09-11, License - CC BY-SA 4.0
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
    """Return the full path to the user's ``simetri_config.toml``."""
    return set_user_settings_path() / CONFIG_FILENAME


def ensure_user_config() -> Path:
    """Create the user config directory and template file if missing.

    Returns:
        Path to ``simetri_config.toml``.
    """
    config_path = user_config_path()
    if not config_path.exists():
        config_path.write_text(
            _PACKAGE_TEMPLATE.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return config_path


def get_default_output_directory() -> str:
    """Return the configured default output directory (may be empty)."""
    return _user_paths["default_output_directory"]


def get_default_test_directory() -> str:
    """Return the configured default test directory (may be empty)."""
    return _user_paths["default_test_directory"]


def get_user_default_overrides() -> dict[str, Any]:
    """Return the mapping of uncommented ``[defaults]`` overrides."""
    return _user_default_overrides


def resolve_save_filepath(filepath: str | Path) -> str:
    """Resolve a save path, using ``default_output_directory`` for bare names.

    Args:
        filepath: User-supplied save path.

    Returns:
        Absolute or joined path string ready for ``validate_filepath``.

    Raises:
        ValueError: Bare filename and ``default_output_directory`` is unset.
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
        _user_default_overrides[key] = converted


def _warning_member(group_name: str, leaf_name: str):
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
        defaults,
    )

    _disabled_warning_types.clear()

    for key, value in warnings_table.items():
        if key == "warnings_on":
            defaults["show_warnings"] = bool(value)
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
        defaults["show_warnings"] = bool(warnings_table["warnings_on"])


def apply_user_config() -> Path:
    """Ensure, load, and apply ``simetri_config.toml``.

    Returns:
        Path to the config file that was read.
    """
    global _config_applied
    from .settings import defaults

    config_path = ensure_user_config()
    _user_default_overrides.clear()
    _user_paths["default_output_directory"] = ""
    _user_paths["default_test_directory"] = ""

    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    known_sections = frozenset({"paths", "warnings", "defaults"})
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

    defaults.user_overrides = _user_default_overrides
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
    for key in sorted(defaults_table):
        lines.append(f"{key} = {_format_toml_value(defaults_table[key])}")
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
    """
    from .. import __version__
    from .settings import default_types, defaults

    script = Path(script_path)
    output = Path(output_path)
    if not script.is_file():
        raise FileNotFoundError(f"Script not found: {script}")

    defaults.log.clear()
    runpy.run_path(str(script), run_name="__main__")

    accessed_keys = sorted({key for key, _str_value in defaults.log})
    defaults_table: dict[str, Any] = {}
    for key in accessed_keys:
        if key not in default_types:
            continue
        expected_type = default_types[key]
        if not _is_exportable_default_type(expected_type):
            continue
        value = defaults[key]
        defaults_table[key] = _serialize_shared_value(value)

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
    """
    from .settings import defaults

    path = Path(toml_path)
    simetri_version, overlay = _load_shared_settings(path)
    check_version(simetri_version)

    previous_shared = dict(defaults.shared_overrides)
    previous_suppress = defaults.suppress_user_overrides
    defaults.shared_overrides = overlay
    defaults.suppress_user_overrides = True
    try:
        yield
    finally:
        defaults.shared_overrides = previous_shared
        defaults.suppress_user_overrides = previous_suppress

"""Load and apply per-user Simetri settings from ``simetri_config.toml``."""

from __future__ import annotations

import platform
import tomllib
from enum import Enum
from pathlib import Path
from typing import Any

from ..base.all_enums import WarningType
from ..coloring.colors import Color, check_color

CONFIG_FILENAME = "simetri_config.toml"
_USER_SUBDIR = "simetri_user"
_PACKAGE_TEMPLATE = Path(__file__).with_name(CONFIG_FILENAME)

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
    from .settings import defaults, warnings_off, warnings_on

    for key, value in warnings_table.items():
        if key == "warnings_on":
            if value:
                warnings_on()
            else:
                warnings_off()
            continue
        if not isinstance(value, dict):
            _warn_invalid_key(
                f"Unknown key in [warnings]: {key!r} (simetri_config.toml)"
            )
            continue
        group_name = key
        group_cls = vars(WarningType)[group_name] if group_name in vars(WarningType) else None
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
            if enabled:
                warnings_on(member)
            else:
                warnings_off(member)

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

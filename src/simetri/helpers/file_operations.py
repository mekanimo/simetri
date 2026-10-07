"""File-path and I/O helpers used by the GUI and exporters."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
import webbrowser
from pathlib import Path, PurePosixPath, PureWindowsPath

import pymupdf as fitz

from ..base.all_enums import WarningType
from ..config.settings import issue_warning
from ..config.user_config import (
    converter_supports_extension,
    get_converter_for_extension,
    get_tex_compiler,
    get_viewer_settings,
    native_save_extensions,
    user_config_path,
)

if platform.system() == "Windows":
    import winreg

# Win32 CREATE_BREAKAWAY_FROM_JOB: child is not killed with the parent job.
_WINDOWS_CREATE_BREAKAWAY_FROM_JOB = 0x01000000

_WINDOWS_INVALID_FILENAME_CHARS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def _skip_windows_path_part(part: str) -> bool:
    """Return True for drive roots, UNC anchors, and special dir names."""
    if part in (".", ".."):
        return True
    if len(part) >= 2 and part[1] == ":" and part[0].isalpha():
        return True
    if part.startswith("\\\\"):
        return True
    return False


def _windows_filename_is_valid(name: str) -> bool:
    if not name:
        return False
    if name.endswith(" ") or name.endswith("."):
        return False
    if any(ch in _WINDOWS_INVALID_FILENAME_CHARS for ch in name):
        return False
    if any(ord(ch) < 32 for ch in name):
        return False
    stem = name.split(".")[0].upper()
    if stem in _WINDOWS_RESERVED_STEMS:
        return False
    return True


def _is_valid_windows_filepath(path_str: str) -> bool:
    for part in PureWindowsPath(path_str).parts:
        if _skip_windows_path_part(part):
            continue
        if not _windows_filename_is_valid(part):
            return False
    return True


def _is_valid_posix_filepath(path_str: str) -> bool:
    for part in PurePosixPath(path_str).parts:
        if "\0" in part:
            return False
    return True


def is_valid_filepath(path: str | os.PathLike[str]) -> bool:
    """Return whether ``path`` is syntactically valid on this operating system.

    Checks filename rules for the **current** platform (Windows vs Linux/macOS).
    Does not require the path to exist. Does not check save formats or
    writability; for ``canvas.save`` output, use :func:`validate_output_filepath`.

    Windows: invalid characters, trailing spaces or dots on names, reserved
    device names (``CON``, ``PRN``, ``COM1``, …). Linux and macOS: non-empty
    path with no null bytes in any component (other characters are allowed).

    Args:
        path: Candidate path string or pathlike object.

    Returns:
        bool: ``True`` when the path satisfies platform rules.

    Examples:
        >>> from simetri.helpers.file_operations import is_valid_filepath
        >>> is_valid_filepath("out/figure.svg")
        True
        >>> is_valid_filepath("")
        False
    """
    path_str = os.fspath(path)
    if not path_str or "\0" in path_str:
        return False
    if platform.system() == "Windows":
        return _is_valid_windows_filepath(path_str)
    return _is_valid_posix_filepath(path_str)


def validate_output_filepath(
    filepath: Path, overwrite: bool
) -> tuple[str, str, str]:
    """Validate a ``canvas.save`` output path (extension, parent dir, overwrite).

    Checks that the extension is a native save format or a configured converter
    format, that the parent directory exists and is writable, and that an
    existing file is only allowed when ``overwrite`` is true. For syntactic
    path rules only, use :func:`is_valid_filepath`.

    Args:
        filepath: Resolved output path (after ``resolve_save_filepath``).
        overwrite: Whether an existing file at ``filepath`` may be replaced.

    Returns:
        ``(parent_dir, stem, extension)`` with ``extension`` lowercased.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from simetri.helpers.file_operations import validate_output_filepath
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "figure.svg"
        ...     parent, stem, ext = validate_output_filepath(p, False)
        >>> ext
        '.svg'
    """
    path_exists = os.path.exists(filepath)
    if path_exists and not overwrite:
        raise FileExistsError(
            f"File {filepath} already exists. \n"
            "Use canvas.save(filepath, overwrite=True) to overwrite the file."
        )
    parent_dir, file_name = os.path.split(filepath)
    file_name, extension = os.path.splitext(file_name)
    extension = extension.lower()
    if extension not in native_save_extensions() and not converter_supports_extension(
        extension
    ):
        config_file = user_config_path()
        raise RuntimeError(
            f"File type {extension!r} is not supported.\n"
            f"Native formats: {', '.join(sorted(native_save_extensions()))}.\n"
            "For other formats, add a personal converter in "
            f"{config_file}, e.g.\n"
            f"[converters.{extension.lstrip('.')}]\n"
            'source = "svg"\n'
            'command = ["resvg", "{input}", "{output}"]'
        )
    if not os.path.exists(parent_dir):
        raise NotADirectoryError(f"Directory {parent_dir} does not exist.")
    if not os.access(parent_dir, os.W_OK):
        raise PermissionError(f"Directory {parent_dir} is not writable.")

    return parent_dir, file_name, extension


def _substitute_viewer_placeholders(template: str, filepath: str) -> str:
    """Replace ``{filepath}``, ``{file}``, and ``{url}`` in a viewer command."""
    file_url = Path(filepath).as_uri()
    return (
        template.replace("{filepath}", filepath)
        .replace("{file}", filepath)
        .replace("{url}", file_url)
    )


def _windows_app_path(program: str) -> str | None:
    """Return the Windows App Paths executable for ``program``.

    ``CreateProcess`` (``shell = false``) does not search App Paths, so a
    name such as ``msedge`` fails with WinError 2 even though it works in
    a shell. Returns None when no registered executable exists.
    """
    executable_name = Path(program).name
    if Path(executable_name).suffix.lower() != ".exe":
        executable_name = f"{executable_name}.exe"
    subkey = (
        rf"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{executable_name}"
    )
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        try:
            key = winreg.OpenKey(hive, subkey)
        except FileNotFoundError:
            continue
        try:
            value, _value_type = winreg.QueryValueEx(key, "")
        except FileNotFoundError:
            continue
        finally:
            winreg.CloseKey(key)
        resolved = Path(value)
        if resolved.is_file():
            return str(resolved)
    return None


def _resolve_viewer_program(program: str) -> str:
    """Return an executable path for ``program``.

    Looks on PATH, then (Windows) App Paths. Raises ``FileNotFoundError``
    if the program cannot be resolved.

    Args:
        program: Executable name or path from ``[viewer].command``.

    Returns:
        str: Path to the executable.

    Raises:
        FileNotFoundError: ``program`` is not a file, not on PATH, and not
            in Windows App Paths.
    """
    candidate = Path(program)
    if candidate.is_file():
        return str(candidate)
    found = shutil.which(program)
    if found is not None:
        return found
    if platform.system() == "Windows":
        found = _windows_app_path(program)
        if found is not None:
            return found
    raise FileNotFoundError(
        f"Viewer program {program!r} was not found on PATH"
    )


def open_saved_file(filepath: str | Path) -> None:
    """Open a file written by ``canvas.save`` using personal ``[viewer]``.

    ``show=False`` / ``defaults['show_browser']`` is handled by the caller.
    ``mode = "system"`` uses the OS default (``webbrowser.open``).
    ``mode = "command"`` runs ``[viewer].command``.
    ``mode = "none"`` does nothing.

    Args:
        filepath: Saved output path (not a ``file://`` URL).

    Examples:
        >>> from simetri.helpers.file_operations import open_saved_file
        >>> open_saved_file('README.md')  # doctest: +SKIP
    """
    path = str(Path(filepath).resolve())
    viewer = get_viewer_settings()
    mode = viewer["mode"]
    if mode == "none":
        return
    if mode == "command":
        command = viewer["command"]
        if command is None:
            issue_warning(
                "[viewer].mode is 'command' but [viewer].command is not set; "
                "opening with the system default instead.",
                WarningType.file.config,
            )
        else:
            try:
                _run_viewer_command(path, viewer)
                return
            except (FileNotFoundError, OSError, ValueError) as error:
                issue_warning(
                    f"Could not run [viewer].command ({error}); "
                    "opening with the system default instead.",
                    WarningType.file.config,
                )
    webbrowser.open(Path(path).as_uri())


def _run_viewer_command(filepath: str, viewer: dict) -> None:
    """Launch the configured viewer command without waiting for it."""
    command = viewer["command"]
    use_shell = bool(viewer["shell"])
    outdir = str(Path(filepath).parent)
    launch_env = os.environ.copy()
    for key in list(launch_env):
        if key.startswith(("ELECTRON_", "VSCODE_", "CURSOR_")):
            del launch_env[key]
    popen_kwargs = {
        "cwd": outdir,
        "env": launch_env,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if platform.system() == "Windows":
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | _WINDOWS_CREATE_BREAKAWAY_FROM_JOB
        )
    else:
        popen_kwargs["start_new_session"] = True
        popen_kwargs["close_fds"] = True

    if use_shell:
        if not isinstance(command, str):
            raise ValueError(
                "[viewer].command must be a string when [viewer].shell = true."
            )
        rendered_command = _substitute_viewer_placeholders(command, filepath)
        subprocess.Popen(rendered_command, shell=True, **popen_kwargs)
        return

    if isinstance(command, str):
        raise ValueError(
            "[viewer].command must be an array of strings when "
            "[viewer].shell = false. Set shell = true only if you "
            "intentionally need a shell."
        )
    rendered_argv = [
        _substitute_viewer_placeholders(part, filepath) for part in command
    ]
    rendered_argv[0] = _resolve_viewer_program(rendered_argv[0])
    subprocess.Popen(rendered_argv, shell=False, **popen_kwargs)


def _substitute_converter_placeholders(
    template: str,
    *,
    input_path: str,
    output_path: str,
) -> str:
    """Replace ``{input}``, ``{output}``, ``{outdir}``, and ``{stem}``."""
    output = Path(output_path)
    return (
        template.replace("{input}", input_path)
        .replace("{output}", output_path)
        .replace("{outdir}", str(output.parent))
        .replace("{stem}", output.stem)
    )


def run_external_converter(
    *,
    input_path: str | Path,
    output_path: str | Path,
    extension: str | None = None,
) -> None:
    """Run the personal ``simetri_config.toml`` converter for ``output_path``.

    Args:
        input_path: Native Simetri file written as the converter source.
        output_path: Destination path from ``canvas.save``.
        extension: Output extension including the dot. Defaults to the
            extension of ``output_path``.

    Raises:
        KeyError: No converter configured for the extension.
        RuntimeError: Converter process failed or did not create the output.
        ValueError: ``shell`` is false but ``command`` is a string (or the
            reverse expectation is violated in a way that cannot run safely).

    Examples:
        >>> from simetri.helpers.file_operations import run_external_converter
        >>> run_external_converter('a.svg', '.png')  # doctest: +SKIP
    """
    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())
    if extension is None:
        extension = Path(output_path).suffix.lower()
    else:
        extension = extension.lower()

    converter = get_converter_for_extension(extension)
    command = converter["command"]
    use_shell = bool(converter["shell"])
    timeout_seconds = float(converter["timeout_seconds"])
    outdir = str(Path(output_path).parent)

    if use_shell:
        if not isinstance(command, str):
            raise ValueError(
                f"[converters.{converter['format_key']}].command must be a "
                "string when [converters].shell = true."
            )
        rendered_command = _substitute_converter_placeholders(
            command, input_path=input_path, output_path=output_path
        )
        completed = subprocess.run(
            rendered_command,
            shell=True,
            cwd=outdir,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    else:
        if isinstance(command, str):
            raise ValueError(
                f"[converters.{converter['format_key']}].command must be an "
                "array of strings when [converters].shell = false. "
                "Set shell = true only if you intentionally need a shell."
            )
        rendered_argv = [
            _substitute_converter_placeholders(
                part, input_path=input_path, output_path=output_path
            )
            for part in command
        ]
        completed = subprocess.run(
            rendered_argv,
            shell=False,
            cwd=outdir,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        details = stderr or stdout or "(no output)"
        raise RuntimeError(
            f"External converter for {extension!r} failed "
            f"(exit {completed.returncode}):\n{details}"
        )
    if not os.path.isfile(output_path):
        raise RuntimeError(
            f"External converter for {extension!r} exited 0 but did not "
            f"create {output_path}."
        )


def run_tex_compiler(
    *,
    input_path: str | Path,
    output_path: str | Path,
) -> None:
    """Run the personal ``[tex].command`` from ``simetri_config.toml``.

    Args:
        input_path: Absolute ``.tex`` file Simetri wrote.
        output_path: Expected ``.pdf`` path (``{output}`` placeholder).

    Raises:
        RuntimeError: ``[tex].command`` is unset, the process failed, or
            the PDF was not created.
        ValueError: ``shell`` does not match the ``command`` type.

    Examples:
        >>> from simetri.helpers.file_operations import run_tex_compiler
        >>> run_tex_compiler('doc.tex')  # doctest: +SKIP
    """
    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())
    tex = get_tex_compiler()
    command = tex["command"]
    if command is None:
        raise RuntimeError(
            "[tex].command is not set in simetri_config.toml."
        )
    use_shell = bool(tex["shell"])
    timeout_seconds = float(tex["timeout_seconds"])
    outdir = str(Path(output_path).parent)

    if use_shell:
        if not isinstance(command, str):
            raise ValueError(
                "[tex].command must be a string when [tex].shell = true."
            )
        rendered_command = _substitute_converter_placeholders(
            command, input_path=input_path, output_path=output_path
        )
        completed = subprocess.run(
            rendered_command,
            shell=True,
            cwd=outdir,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    else:
        if isinstance(command, str):
            raise ValueError(
                "[tex].command must be an array of strings when "
                "[tex].shell = false. Set shell = true only if you "
                "intentionally need a shell."
            )
        rendered_argv = [
            _substitute_converter_placeholders(
                part, input_path=input_path, output_path=output_path
            )
            for part in command
        ]
        completed = subprocess.run(
            rendered_argv,
            shell=False,
            cwd=outdir,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        details = stderr or stdout or "(no output)"
        raise RuntimeError(
            f"TeX compiler failed (exit {completed.returncode}):\n{details}"
        )
    if not os.path.isfile(output_path):
        raise RuntimeError(
            "TeX compiler exited 0 but did not create "
            f"{output_path}."
        )


def inject_snippet(
    code: str, snippet: list[str], mark: str, before: bool = True
) -> str | None:
    """Insert the given snippet before/after the line that contains the mark.

    Args:
        code (str): Source code to modify.
        snippet (list[str]): Lines to insert.
        mark (str): Substring identifying the insertion anchor line.
        before (bool, optional): If True, insert before the mark line;
            otherwise insert after. Defaults to True.

    Returns:
        str: Modified source code, or None if the mark was not found.

    Examples:
        >>> from simetri.helpers.file_operations import inject_snippet
        >>> inject_snippet('a\\nMARK\\nb', ['X'], 'MARK')
        'a\\nX\\nMARK\\nb'
    """

    lines = code.split("\n")
    res_lines = []
    flag = True
    count = 0
    for line in lines:
        if mark in line:
            flag = False
            break
        count += 1

    if not before:
        count += 1

    res_lines = lines[:count] + snippet + lines[count:]

    if flag:
        issue_warning(
            f"Could not find '{mark}'",
            warning_type=WarningType.file.mark,
        )
    else:
        return "\n".join(res_lines)


def replace_token(code: str, token: str, replace: str) -> str:
    """Replace occurrences of ``token`` in each matching line.

    Args:
        code (str): Source code to modify.
        token (str): Substring to find within lines.
        replace (str): Replacement text for ``token``.

    Returns:
        str: Modified source code, or None if the token was not found.

    Examples:
        >>> from simetri.helpers.file_operations import replace_token
        >>> replace_token('a TOKEN b', 'TOKEN', 'X')
        'a X b'
    """
    lines = code.split("\n")
    res_lines = []
    flag = True
    for line in lines:
        if token in line:
            res_line = line.replace(token, replace)
            res_lines.append(res_line)
            flag = False
        else:
            res_lines.append(line)

    if flag:
        issue_warning(
            "Could not find 'token'",
            warning_type=WarningType.file.token,
        )
    else:
        return "\n".join(res_lines)


def inject_filepath(code: str, pic_path: str) -> str:
    """Replace ``canvas.display()`` with ``canvas.save(...)``.

    Args:
        code (str): Source code to modify.
        pic_path (str): Output path passed to ``canvas.save``.

    Returns:
        str: Modified source code, or None if ``canvas.display()`` was not found.

    Examples:
        >>> from simetri.helpers.file_operations import inject_filepath
        >>> 'out.svg' in inject_filepath('canvas.display()', 'out.svg')
        True
    """
    lines = code.split("\n")
    res_lines = []
    flag = True
    disp = "canvas.display()"
    save_file = f'canvas.save("{pic_path}", overwrite=True)'
    for line in lines:
        if "canvas.display()" in line:
            res_line = line.replace(disp, save_file)
            res_lines.append(res_line)
            flag = False
        else:
            res_lines.append(line)

    if flag:
        issue_warning(
            "Could not find 'canvas.display()'",
            warning_type=WarningType.file.display,
        )
    else:
        return "\n".join(res_lines)


def inject_border(
    code: str,
    caption: str,
    width: float | None = None,
    height: float | None = None,
) -> str | None:
    """Inject an ``auto_border`` call before ``canvas.save(``.

    Args:
        code (str): Source code to modify.
        caption (str): Caption passed to ``auto_border``.
        width (float | None, optional): Optional border width.
        height (float | None, optional): Optional border height.

    Returns:
        str: Modified source code with the border snippet inserted.

    Examples:
        >>> from simetri.helpers.file_operations import inject_border
        >>> 'auto_border' in inject_border('canvas.save(x)', 'cap')
        True
    """
    # inject auto_border(canvas)
    w, h = width, height
    mark = "canvas.save("
    snippet = [
        "import sys",
        'new_path = "D:/potams/pages"',
        "sys.path.append(new_path)",
        "from border import auto_border",
        f'auto_border(canvas, caption="{caption}", width={w}, height={h})',
    ]

    code_border = inject_snippet(
        code=code, snippet=snippet, mark=mark, before=True
    )

    return code_border


def inject_border_and_filepath(
    code: str,
    pic_path: str,
    pic_caption: str,
    width: float | None = None,
    height: float | None = None,
) -> str:
    """Inject ``auto_border`` and replace ``canvas.display()`` with save.

    Args:
        code (str): Source code to modify.
        pic_path (str): Output path for ``canvas.save``.
        pic_caption (str): Caption passed to ``auto_border``.
        width (float | None, optional): Optional border width.
        height (float | None, optional): Optional border height.

    Returns:
        str: Modified source code.

    Examples:
        >>> from simetri.helpers.file_operations import inject_border_and_filepath
        >>> 'auto_border' in inject_border_and_filepath('canvas.display()', 'out.svg', 'cap')
        True
    """
    w, h = width, height
    mark = "canvas.display()"
    snippet = [
        "import sys",
        'new_path = "D:/potams/pages"',
        "sys.path.append(new_path)",
        "from border import auto_border",
        f'auto_border(canvas, caption="{pic_caption}", width={w}, height={h})',
    ]

    code_border = inject_snippet(code=code, snippet=snippet, mark=mark)

    # replace canvas.display() with canvas.save(pic_path)
    token = "canvas.display()"
    pic_path = pic_path.replace(os.sep, "/")
    replace = f"canvas.save('{pic_path}', overwrite=True)"
    code_save = replace_token(code=code_border, token=token, replace=replace)

    return code_save


def forward_slash_path(path: str | os.PathLike[str]) -> str:
    """Return ``path`` with backslashes written as forward slashes.

    TeX and SVG accept this form on Linux, macOS, and Windows.

    Args:
        path: A filesystem path.

    Returns:
        str: The same path using ``/`` separators.

    Examples:
        >>> from simetri.helpers.file_operations import forward_slash_path
        >>> forward_slash_path("C:\\\\FB\\\\m336_block_three_gr4.png")
        'C:/FB/m336_block_three_gr4.png'
        >>> forward_slash_path("/home/user/fig.png")
        '/home/user/fig.png'
    """
    return os.fspath(path).replace("\\", "/")


def path_join(
    path: str | os.PathLike[str], *paths: str | os.PathLike[str]
) -> str:
    """Join path segments using ``os.path.join``.

    Args:
        path: First path segment.
        *paths: Additional path segments.

    Returns:
        str: Joined path string.

    Examples:
        >>> from simetri.helpers.file_operations import path_join
        >>> path_join('a', 'b', 'c.txt')
        'a/b/c.txt'
    """
    joined_path = os.path.join(path, *paths)
    return joined_path.replace(os.sep, "/")


def join_path_with_ext(
    *folders: str | os.PathLike[str], filename: str, ext: str
) -> str:
    """Build a full path from folders, filename, and extension.

    Args:
        *folders: Parent folder segments.
        filename: File stem without extension.
        ext: Extension including the leading dot (for example ``.pdf``).

    Returns:
        str: Joined path (intended to use forward slashes).

    Examples:
        >>> from simetri.helpers.file_operations import join_path_with_ext
        >>> join_path_with_ext('out', filename='fig', ext='.svg')
        'out/fig.svg'
    """

    return path_join(*folders, filename + ext)


def path_exists(path: str | os.PathLike[str]) -> bool:
    """Return True if the given path exists (file or directory).

    Args:
        path (str | os.PathLike[str]): Path to check.

    Returns:
        bool: True if the path exists.

    Examples:
        >>> from simetri.helpers.file_operations import path_exists
        >>> path_exists('__no_such_path__')
        False
    """
    return Path(path).exists()


def wait_for_file_availability(
    filepath: str | os.PathLike[str],
    timeout: float | None = None,
    check_interval: float = 1,
) -> bool | None:
    """Check if a file is available for writing.

    Args:
        filepath: The path to the file.
        timeout: The timeout period in seconds.
        check_interval: The interval to check the file availability.

    Returns:
        True if the file is available, False otherwise.

    Examples:
        >>> from simetri.helpers.file_operations import wait_for_file_availability
        >>> wait_for_file_availability('__no_such_parent__/__missing__', timeout=0)
        False
    """
    start_time = time.monotonic()
    while True:
        try:
            # Attempt to open the file in write mode. This will raise an exception
            # if the file is currently locked or being written to.
            with open(filepath, "a", encoding="utf-8"):
                # If the file was successfully opened, it's available.
                return True
        except OSError:
            # The file is likely in use or not yet present.
            if timeout is not None and (
                time.monotonic() - start_time
            ) >= timeout:
                return False
            if timeout is None:
                time.sleep(check_interval)
                continue
            remaining = timeout - (time.monotonic() - start_time)
            if remaining <= 0:
                return False
            time.sleep(min(check_interval, remaining))
        except (TypeError, ValueError) as e:
            # Handle other potential exceptions (e.g., file not found) as needed
            print(f"An error occurred: {e}")
            return False


def remove_aux_files(filepath: str | Path | os.PathLike[str]) -> None:
    """
    Remove auxiliary files generated during compilation.

    Args:
        filepath (Path): The path to the file.

    Examples:
        >>> from simetri.helpers.file_operations import remove_aux_files
        >>> remove_aux_files('doc.tex')  # doctest: +SKIP
    """
    time_out = 1  # seconds
    folder, filename = os.path.split(filepath)
    stem, extension = os.path.splitext(filename)
    aux_filepath = path_join(folder, stem + ".aux")
    if os.path.exists(aux_filepath):
        if not wait_for_file_availability(aux_filepath, time_out):
            print(
                f"File '{aux_filepath}' is not available after waiting for "
                f"{time_out} seconds."
            )
        else:
            os.remove(aux_filepath)
    log_filepath = path_join(folder, stem + ".log")
    if os.path.exists(log_filepath) and not wait_for_file_availability(
        log_filepath, time_out
    ):
        print(
            f"File '{log_filepath}' is not available after waiting for "
            f"{time_out} seconds."
        )
        # else:
        #     if not defaults["keep_log_files"]:
        #         os.remove(log_file)
    tex_filepath = path_join(folder, stem + ".tex")
    if os.path.exists(tex_filepath):
        if not wait_for_file_availability(tex_filepath, time_out):
            print(
                f"File '{tex_filepath}' is not available after waiting for "
                f"{time_out} seconds."
            )
        else:
            os.remove(tex_filepath)
    stem, extension = os.path.splitext(filename)
    if extension not in (".pdf", ".tex"):
        pdf_filepath = path_join(folder, stem + ".pdf")
        if os.path.exists(pdf_filepath):
            if not wait_for_file_availability(pdf_filepath, time_out):
                print(
                    f"File '{pdf_filepath}' is not available after waiting for "
                    f"{time_out} seconds."
                )
            else:
                # os.remove(pdf_file)
                pass
    log_filepath = path_join(folder, f"{stem}.log")
    if os.path.exists(log_filepath):
        try:
            os.remove(log_filepath)
        except PermissionError:
            # to do: log the error
            pass


def replace_extension(filepath: str, ext: str) -> str:
    """Return ``filepath`` with its extension replaced.

    Args:
        filepath (str): Original file path.
        ext (str): New extension including the leading dot.

    Returns:
        str: Path with the new extension.

    Examples:
        >>> from simetri.helpers.file_operations import replace_extension
        >>> replace_extension('a.b.tex', '.pdf')
        'a.b.pdf'
    """
    return os.path.splitext(filepath)[0] + ext


def convert_pdf(pdf_path: str, extension: str) -> None:
    """Convert a PDF file to another supported vector format.

    Only ``.ps``, ``.eps``, and ``.svg`` extensions are supported.

    Args:
        pdf_path (str): Path to the source PDF.
        extension (str): Target extension including the leading dot.

    Raises:
        RuntimeError: If PDF-to-PS conversion fails.

    Examples:
        >>> from simetri.helpers.file_operations import convert_pdf
        >>> convert_pdf('a.pdf', '.svg')  # doctest: +SKIP
    """
    parent_dir, file_name = os.path.split(pdf_path)
    file_name, _ = os.path.splitext(file_name)
    output_path = os.path.join(parent_dir, file_name + extension)
    if extension in (".eps", ".ps"):
        os.chdir(parent_dir)
        cmd = f"pdf2ps {pdf_path} {output_path}"
        res = subprocess.run(cmd, shell=True, check=False)
        if res.returncode != 0:
            raise RuntimeError("Failed to convert pdf to ps.")
    elif extension == ".svg":
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        svg = page.get_svg_image()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg)

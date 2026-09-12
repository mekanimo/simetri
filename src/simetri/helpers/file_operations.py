"""File-path and I/O helpers used by the GUI and exporters."""

import os
import platform
import subprocess
import time
from pathlib import Path

import pymupdf as fitz

from ..base.all_enums import WarningType
from ..config.settings import issue_warning
from ..config.user_config import (
    converter_supports_extension,
    get_converter_for_extension,
    get_tex_compiler,
    native_save_extensions,
    user_config_path,
)


def validate_filepath(filepath: Path, overwrite: bool):
    """
    Validate the file path.

    Args:
        filepath (Path): The path to the file.
        overwrite (bool): Whether to overwrite the file if it exists.

    Returns:
        Result: The parent directory, file name, and extension.
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
    code: str, snippet: list[str], mark: str, before=True
) -> str:
    """Insert the given snippet before/after the line that contains the mark.

    Args:
        code (str): Source code to modify.
        snippet (list[str]): Lines to insert.
        mark (str): Substring identifying the insertion anchor line.
        before (bool, optional): If True, insert before the mark line;
            otherwise insert after. Defaults to True.

    Returns:
        str: Modified source code, or None if the mark was not found.
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
):
    """Inject an ``auto_border`` call before ``canvas.save(``.

    Args:
        code (str): Source code to modify.
        caption (str): Caption passed to ``auto_border``.
        width (float | None, optional): Optional border width.
        height (float | None, optional): Optional border height.

    Returns:
        str: Modified source code with the border snippet inserted.
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


def path_join(path, *paths):
    """Join path segments using ``os.path.join``.

    Args:
        path: First path segment.
        *paths: Additional path segments.

    Returns:
        str: Joined path string.
    """
    joined_path = os.path.join(path, *paths)
    joined_path.replace(os.sep, "/")

    return joined_path


def join_path_with_ext(*folders, filename, ext):
    """Build a full path from folders, filename, and extension.

    Args:
        *folders: Parent folder segments.
        filename: File stem without extension.
        ext: Extension including the leading dot (for example ``.pdf``).

    Returns:
        str: Joined path (intended to use forward slashes).
    """

    return path_join(*folders, filename + ext)


def path_exists(path: str | os.PathLike[str]) -> bool:
    """Return True if the given path exists (file or directory).

    Args:
        path (str | os.PathLike[str]): Path to check.

    Returns:
        bool: True if the path exists.

    Examples:
        >>> path_exists("/tmp/test.txt")
        False
    """
    return Path(path).exists()


def wait_for_file_availability(filepath, timeout=None, check_interval=1):
    """Check if a file is available for writing.

    Args:
        filepath: The path to the file.
        timeout: The timeout period in seconds.
        check_interval: The interval to check the file availability.

    Returns:
        True if the file is available, False otherwise.
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
            # The file is likely in use.
            if (
                timeout is not None
                and (time.monotonic() - start_time) > timeout
            ):
                # Timeout period elapsed.
                return False  # Or raise a TimeoutError if you prefer
            time.sleep(check_interval)
        except (TypeError, ValueError) as e:
            # Handle other potential exceptions (e.g., file not found) as needed
            print(f"An error occurred: {e}")
            return False


def remove_aux_files(filepath):
    """
    Remove auxiliary files generated during compilation.

    Args:
        filepath (Path): The path to the file.
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
    """
    return os.path.splitext(filepath)[0] + ext


def convert_pdf(pdf_path: str, extension: str):
    """Convert a PDF file to another supported vector format.

    Only ``.ps``, ``.eps``, and ``.svg`` extensions are supported.

    Args:
        pdf_path (str): Path to the source PDF.
        extension (str): Target extension including the leading dot.

    Raises:
        RuntimeError: If PDF-to-PS conversion fails.
    """
    if extension in (".eps", ".ps"):
        ps_path = os.path.join(parent_dir, file_name + extension)
        os.chdir(parent_dir)
        cmd = f"pdf2ps {pdf_path} {ps_path}"
        res = subprocess.run(cmd, shell=True, check=False)
        if res.returncode != 0:
            raise RuntimeError("Failed to convert pdf to ps.")
    elif extension == ".svg":
        doc = fitz.open(pdf_path)
        page = doc.load_page(0)
        svg = page.get_svg_image()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg)

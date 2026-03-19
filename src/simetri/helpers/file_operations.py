"""
File operation utilities for the GUI.
"""

import os
from pathlib import Path
from string import Template
import subprocess
import sys
import time
from typing import List, Optional, Union
import warnings

import fitz

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
    if extension not in [".pdf", ".eps", ".ps", ".svg", ".png", ".tex"]:
        raise RuntimeError("File type is not supported.")
    if not os.path.exists(parent_dir):
        raise NotADirectoryError(f"Directory {parent_dir} does not exist.")
    if not os.access(parent_dir, os.W_OK):
        raise PermissionError(f"Directory {parent_dir} is not writable.")

    return parent_dir, file_name, extension

def inject_snippet(code: str, snippet: List[str], mark: str, before=True) -> str:
    """Insert the given snippet before/after the line that contains the
    given mark.
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
        warnings.warn(f"Could not find '{mark}'")
    else:
        return "\n".join(res_lines)


def replace_token(code: str, token: str, replace: str) -> str:
    """'Replace the line with the given mark with the given
    new_line.
    Return the modified snippet.
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
        warnings.warn("Could not find 'token'")
    else:
        return "\n".join(res_lines)


def inject_filepath(code: str, pic_path: str) -> str:
    """'Replace 'canvas.display()' with
    'canvas.save(pic_path, overwrite=True)'
    and returns the modified script.
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
        warnings.warn("Could not find 'canvas.display()'")
    else:
        return "\n".join(res_lines)


def inject_border(
    code: str,
    caption: str,
    width: Optional[float] = None,
    height: Optional[float] = None,
):
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

    code_border = inject_snippet(code=code, snippet=snippet, mark=mark, before=True)

    return code_border


def inject_border_and_filepath(
    code: str,
    pic_path: str,
    pic_caption: str,
    width: Optional[float] = None,
    height: Optional[float] = None,
) -> str:
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
    joined_path = os.path.join(path, *paths)
    joined_path.replace(os.sep, "/")

    return joined_path


def join_path_with_ext(*folders, filename, ext):
    """Given a folder-path filename and ext (".ext") returns the full path
    with forward slashes."""

    return path_join(*folders, filename + ext)


def path_exists(path: Union[str, os.PathLike[str]]) -> bool:
    """
    Return True if the given path exists (file or directory), otherwise False.

    Parameters:
      path: str or PathLike - the path to check

    Example:
      path_exists("/tmp/test.txt")  # True if the file exists
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
        except IOError:
            # The file is likely in use.
            if timeout is not None and (time.monotonic() - start_time) > timeout:
                # Timeout period elapsed.
                return False  # Or raise a TimeoutError if you prefer
            time.sleep(check_interval)
        except Exception as e:
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
                (
                    f"File '{aux_filepath}' is not available after waiting for "
                    f"{time_out} seconds."
                )
            )
        else:
            os.remove(aux_filepath)
    log_filepath = path_join(folder, stem + ".log")
    if os.path.exists(log_filepath):
        if not wait_for_file_availability(log_filepath, time_out):
            print(
                (
                    f"File '{log_filepath}' is not available after waiting for "
                    f"{time_out} seconds."
                )
            )
        # else:
        #     if not defaults["keep_log_files"]:
        #         os.remove(log_file)
    tex_filepath = path_join(folder, stem + ".tex")
    if os.path.exists(tex_filepath):
        if not wait_for_file_availability(tex_filepath, time_out):
            print(
                (
                    f"File '{tex_filepath}' is not available after waiting for "
                    f"{time_out} seconds."
                )
            )
        else:
            os.remove(tex_filepath)
    stem, extension = os.path.splitext(filename)
    if extension not in [".pdf", ".tex"]:
        pdf_filepath = path_join(folder, stem + ".pdf")
        if os.path.exists(pdf_filepath):
            if not wait_for_file_availability(pdf_filepath, time_out):
                print(
                    (
                        f"File '{pdf_filepath}' is not available after waiting for "
                        f"{time_out} seconds."
                    )
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
    """Given a fileapth and an extension ('.pdf', '.py', etc.)
    returns a filepath with the given extension.
    """
    return os.path.splitext(filepath)[0] + ext

def convert_pdf(pdf_path:str, extension:str):
    '''Converts the given PDF file to given extension.
       Only .ps, .eps, .svg, and .png extensions are supported.
    '''
    if extension in [".eps", ".ps"]:
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
    elif extension == ".png":
        pdf_file = fitz.open(pdf_path)
        page = pdf_file[0]
        pix = page.get_pixmap()
        pix.save(output_path)
        pdf_file.close()
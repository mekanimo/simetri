"""LaTeX / TeX compilation helpers for Simetri canvas export.

Builds TeX documents from canvas TikZ code, runs the configured LaTeX
compiler, and cleans auxiliary files.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import fitz

from simetri.canvas.pre_render import (
    canvas_uses_label_halos,
    collect_tikz_preamble_requirements,
    label_halo_preamble_line,
)
from simetri.graphics.all_enums import TexLoc, Types
from simetri.helpers.utilities import *
from simetri.settings.settings import defaults
from simetri.tikz.tikz import (
    color_to_tikz,
    get_canvas_scope,
    get_limits_code,
    scope_code_required,
)

if TYPE_CHECKING:
    from simetri.canvas import Canvas


def remove_aux_files(file_path):
    """
    Remove auxiliary files generated during compilation.

    Args:
        file_path (Path): The path to the file.
    """
    time_out = 1  # seconds
    parent_dir, file_name = os.path.split(file_path)
    file_name, extension = os.path.splitext(file_name)
    aux_file = os.path.join(parent_dir, file_name + ".aux")
    if os.path.exists(aux_file):
        if not wait_for_file_availability(aux_file, time_out):
            print(
                (
                    f"File '{aux_file}' is not available after waiting for "
                    f"{time_out} seconds."
                )
            )
        else:
            os.remove(aux_file)
    log_file = os.path.join(parent_dir, file_name + ".log")
    if os.path.exists(log_file):
        if not wait_for_file_availability(log_file, time_out):
            print(
                f"File '{log_file}' is not available after waiting for "
                f"{time_out} seconds."
            )
        else:
            if not defaults["keep_log_files"]:
                os.remove(log_file)
    tex_file = os.path.join(parent_dir, file_name + ".tex")
    if os.path.exists(tex_file):
        if not wait_for_file_availability(tex_file, time_out):
            print(
                f"File '{tex_file}' is not available after waiting for "
                f"{time_out} seconds."
            )
        else:
            os.remove(tex_file)
    file_name, extension = os.path.splitext(file_name)
    if extension not in (".pdf", ".tex"):
        pdf_file = os.path.join(parent_dir, file_name + ".pdf")
        if os.path.exists(pdf_file):
            if not wait_for_file_availability(pdf_file, time_out):
                print(
                    f"File '{pdf_file}' is not available after waiting for "
                    f"{time_out} seconds."
                )
            else:
                # os.remove(pdf_file)
                pass
    log_file = os.path.join(parent_dir, "simetri.log")
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
        except PermissionError:
            # to do: log the error
            pass


def run_job(parent_dir, file_name, extension, tex_path):
    """Compile a TeX file and write the requested output format.

    Args:
        parent_dir: Directory containing the TeX file and outputs.
        file_name: Base name without extension.
        extension: Desired output extension (e.g. ``.pdf``, ``.png``).
        tex_path: Full path to the ``.tex`` source file.
    """
    output_path = os.path.join(parent_dir, file_name + extension)
    compiler = defaults["latex_compiler"].lower()
    cmd = f'{compiler} "{tex_path}" --output-directory "{parent_dir}"'
    res = compile_tex(cmd, parent_dir, print_output=False)
    if "No pages of output" in res:
        raise RuntimeError("Failed to compile the tex file.")
    pdf_path = os.path.join(parent_dir, file_name + ".pdf")
    if not os.path.exists(pdf_path):
        raise RuntimeError("Failed to compile the tex file.")

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
    elif extension == ".png":
        pdf_file = fitz.open(pdf_path)
        page = pdf_file[0]
        pix = page.get_pixmap()
        pix.save(output_path)
        pdf_file.close()


def compile_tex(cmd, parent_dir, print_output):
    """
    Compile the TeX file.

    Args:
        cmd (str): The command to compile the TeX file.

    Returns:
        str: The output of the compilation.
    """
    os.chdir(parent_dir)
    with subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        shell=True,
        text=True,
    ) as p:
        output = p.communicate("_s\n_l\n")[0]
    if print_output:
        print(output.split("\n")[-3:])
    return output


@dataclass
class Tex:
    """Tex class for generating tex code.

    Attributes:
        begin_document (str): The beginning of the document.
        end_document (str): The end of the document.
        begin_tikz (str): The beginning of the TikZ environment.
        end_tikz (str): The end of the TikZ environment.
        packages (list[str]): List of required TeX packages.
        tikz_libraries (list[str]): List of required TikZ libraries.
        tikz_code (str): The generated TikZ code.
        sketches (list["Sketch"]): List of Sketch objects.
    """

    begin_document: str = defaults["begin_doc"]
    end_document: str = defaults["end_doc"]
    begin_tikz: str = defaults["begin_tikz"]
    end_tikz: str = defaults["end_tikz"]
    packages: list[str] = None
    tikz_libraries: list[str] = None
    tikz_code: str = ""  # Generated by the canvas by using sketches
    sketches: list["Sketch"] = field(
        default_factory=list
    )  # List of TexSketch objects

    def __post_init__(self):
        """Post-initialization method."""
        self.type = Types.TEX

    def tex_code(self, canvas: Canvas, aux_code: str) -> str:
        """Generate the final TeX code.

        Args:
            canvas (Canvas): The canvas object.
            aux_code (str): Auxiliary code to include.

        Returns:
            str: The final TeX code.
        """
        doc_code = []
        for sketch in self.sketches:
            if sketch.location == TexLoc.DOCUMENT:
                doc_code.append(sketch.code)
        doc_code = "\n".join(doc_code)
        if canvas.back_color is None:
            back_color = ""
        else:
            back_color = f"\\pagecolor{color_to_tikz(canvas.back_color)}"
        self.begin_document = self.begin_document + back_color + "\n"
        if canvas.overlay:
            begin_t = self.begin_tikz
            i = begin_t.index("]\n")
            overlay = ", remember picture, overlay"
            self.begin_tikz = begin_t[:i] + overlay + begin_t[i:]
        if canvas.limits is not None or canvas.inset != 0:
            begin_tikz = self.begin_tikz + get_limits_code(canvas) + "\n"
        else:
            begin_tikz = self.begin_tikz + "\n"
        if scope_code_required(canvas):
            scope = get_canvas_scope(canvas)
            code = (
                self.get_preamble(canvas)
                + self.begin_document
                + doc_code
                + begin_tikz
                + scope
                + self.get_tikz_code()
                + aux_code
                + "\\end{scope}\n"
                + self.end_tikz
                + self.end_document
            )
        else:
            code = (
                self.get_preamble(canvas)
                + self.begin_document
                + doc_code
                + begin_tikz
                + self.get_tikz_code()
                + aux_code
                + self.end_tikz
                + self.end_document
            )

        return code

    def get_doc_class(self, border: float, font_size: int) -> str:
        """Returns the document class.

        Args:
            border (float): The border size.
            font_size (int): The font size.

        Returns:
            str: The document class string.
        """
        if isinstance(border, str):
            border_value = border
        else:
            border_value = f"{border}pt"
        return f"\\documentclass[{font_size}pt,tikz,border={border_value}]{{standalone}}\n"

    def get_tikz_code(self) -> str:
        """Returns the TikZ code.

        Returns:
            str: The TikZ code.
        """
        code = ""
        for sketch in self.sketches:
            if sketch.location == TexLoc.PICTURE:
                code += sketch.text + "\n"

        return code

    def get_tikz_libraries(self) -> str:
        """Returns the TikZ libraries.

        Returns:
            str: The TikZ libraries string.
        """
        return f"\\usetikzlibrary{{{','.join(self.tikz_libraries)}}}\n"

    def get_packages(self, canvas) -> str:
        """Returns the required TeX packages.

        Args:
            canvas: The canvas object.

        Returns:
            str: The required TeX packages.
        """
        if self.tikz_libraries is not None and self.packages is not None:
            return self.tikz_libraries, self.packages

        return collect_tikz_preamble_requirements(canvas)

    def get_preamble(self, canvas) -> str:
        """Returns the TeX preamble.

        Args:
            canvas: The canvas object.

        Returns:
            str: The TeX preamble.
        """
        libraries, packages = self.get_packages(canvas)

        if packages:
            packages = f"\\usepackage{{{','.join(packages)}}}\n"
            if canvas_uses_label_halos(canvas):
                packages += label_halo_preamble_line()
            if "fontspec" in packages:
                fonts_section = f"""\\setmainfont{{{defaults["main_font"]}}}
\\setsansfont{{{defaults["sans_font"]}}}
\\setmonofont{{{defaults["mono_font"]}}}\n"""

        if libraries:
            libraries = f"\\usetikzlibrary{{{','.join(libraries)}}}\n"
        if canvas.page_size is not None:
            border = 0
        else:
            if canvas.border is None:
                border = defaults["border"]
            elif isinstance(canvas.border, (int, float)):
                border = canvas.border
            elif (
                isinstance(canvas.border, (list, tuple))
                and len(canvas.border) == 4
            ):
                if any(item < 0 for item in canvas.border):
                    raise ValueError(
                        "Canvas.border must be a positive numeric value or a tuple of 4 positive numeric values."
                    )
                left, bottom, right, top = canvas.border
                border = f"{{{left}pt {bottom}pt {right}pt {top}pt}}"
            else:
                raise ValueError(
                    "Canvas.border must be a positive numeric value or a tuple of 4 positive numeric values."
                )
            if isinstance(border, (int, float)) and border < 0:
                raise ValueError(
                    "Canvas.border must be a positive numeric value or a tuple of 4 positive numeric values."
                )
        doc_class = self.get_doc_class(border, defaults["font_size"])
        # Check if different fonts are used
        fonts_section = ""
        fonts = canvas.get_fonts_list()
        for font in fonts:
            if font is None:
                continue
            font_family = font.replace(" ", "")
            fonts_section += (
                f"\\newfontfamily\\{font_family}[Scale=1.0]{{{font}}}\n"
            )
        preamble = f"{doc_class}{packages}{libraries}{fonts_section}"

        indices = False
        for sketch in canvas.active_page.sketches:
            if (
                hasattr(sketch, "marker_type")
                and sketch.marker_type == "indices"
            ):
                indices = True
                break
        if indices:
            font_family = defaults["indices_font_family"]
            font_size = defaults["index_font_size"]
            if isinstance(font_size, (int, float)):
                baseline = ceil(float(font_size) * 1.2)
                font_spec = (
                    f"\\{font_family}\\fontsize{{{font_size}}}{{{baseline}}}\\selectfont"
                )
            else:
                font_spec = f"\\{font_family}\\{font_size}"
            count = 0
            for sketch in canvas.active_page.sketches:
                if (
                    hasattr(sketch, "marker_type")
                    and sketch.marker_type == "indices"
                ):
                    preamble += "\\tikzset{\n"
                    node_style = (
                        f"nodestyle{count}/.style={{draw, circle, gray, "
                        f"text=black, fill=white, line width = .5, inner sep=.5, "
                        f"font={font_spec}}}\n}}\n"
                    )
                    preamble += node_style
                    count += 1
        for sketch in canvas.active_page.sketches:
            if sketch.subtype == Types.TEX_SKETCH:  # noqa: SIM102
                if sketch.location == TexLoc.PREAMBLE:
                    preamble += sketch.code + "\n"
        return preamble

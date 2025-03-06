"""This module contains the functions to display the output of the canvas
in the Jupyter notebook."""

import tempfile
import shutil
import os

import fitz
from IPython.display import Image
from IPython.display import display as ipy_display


def display(canvas):
    """Show the output of the canvas in the Jupyter notebook.

    Args:
        canvas: The canvas object to be displayed.

    """
    # open a temporary directory
    with tempfile.TemporaryDirectory(
        ignore_cleanup_errors=True, delete=True
    ) as tmpdirname:
        file_name = next(tempfile._get_candidate_names())
        file_path = os.path.join(tmpdirname, file_name + ".png")
        canvas.save(file_path, show=False, print_output=False)
        ipy_display(Image(file_path))

    shutil.rmtree(tmpdirname, ignore_errors=True)

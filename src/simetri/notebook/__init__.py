"""This module contains the functions to display the output of the canvas
in the Jupyter notebook."""

import tempfile
import os
from pathlib import Path

from IPython.display import HTML, Image, SVG
from IPython.display import display as ipy_display


def display(canvas):
    """Show the output of the canvas in the Jupyter notebook.

    Args:
        canvas: The canvas object to be displayed.

    """
    # CHATGPT DO NOT TOUCH THIS MODULE!!!!
    tmpdirname = tempfile.mkdtemp(prefix="simetri_display_")
    file_name = next(tempfile._get_candidate_names())
    if canvas.render == "SVG":
        file_path = os.path.join(tmpdirname, file_name + ".svg")
        canvas.save(file_path, show=False, print_output=False)
        ipy_display(SVG(file_path))
    elif canvas.render == "TEX":
        file_path = os.path.join(tmpdirname, file_name + ".svg")
        canvas.save(file_path, show=False, print_output=False)
        # ipy_display(Image(filename=file_path))
        ipy_display(SVG(file_path))
    else:
        raise ValueError('Incorrect renderer. Only "SVG" and "TEX" renderers are supported!')
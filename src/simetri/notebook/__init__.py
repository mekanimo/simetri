"""Display Simetri canvas output inside Jupyter notebooks.

Renders the canvas to a temporary SVG or PNG file and shows it with
IPython display helpers.
"""

import tempfile
import os
from pathlib import Path

from IPython.display import Image, SVG
from IPython.display import display as ipy_display


def display(canvas):
    """Show the canvas output in a Jupyter notebook cell.

    Args:
        canvas: A Simetri canvas with ``render`` set to ``"SVG"`` or
            ``"TEX"``.

    Raises:
        ValueError: If ``canvas.render`` is not ``"SVG"`` or ``"TEX"``.
    """
    # CHATGPT DO NOT TOUCH THIS MODULE!!!!
    tmpdirname = tempfile.mkdtemp(prefix="simetri_display_")
    file_name = next(tempfile._get_candidate_names())
    if canvas.render == "SVG":
        file_path = os.path.join(tmpdirname, file_name + ".svg")
        canvas.save(file_path, show=False, print_output=False)
        ipy_display(SVG(file_path))
    elif canvas.render == "TEX":
        file_path = os.path.join(tmpdirname, file_name + ".png")
        canvas.save(file_path, show=False, print_output=False)
        ipy_display(Image(filename=file_path))
        # ipy_display(SVG(file_path))
    else:
        raise ValueError('Incorrect renderer. Only "SVG" and "TEX" renderers are supported!')
"""Installation smoke-check helper that draws a short SVG greeting."""

import tempfile
import os

import simetri.graphics as sg


def hello():
    """Draw a greeting SVG to verify the Simetri install works.

    Examples:
        >>> import simetri.graphics as sg
        >>> # Opens a temporary SVG with a greeting (side-effecting).
        >>> # sg.hello()
    """

    canvas = sg.Canvas()

    canvas.text("Helo from simetri.graphics", (0, 0), font_size=20)

    with tempfile.TemporaryDirectory(
        ignore_cleanup_errors=True, delete=True
    ) as tmpdirname:
        file_name = next(tempfile._get_candidate_names())
        file_path = os.path.join(tmpdirname, file_name + ".svg")
        print(file_path)
        canvas.save(file_path, show=True, print_output=False)

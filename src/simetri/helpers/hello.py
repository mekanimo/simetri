import tempfile
import os

import simetri.graphics as sg


def hello():
    """Hello world function.
    To check if the installation is successful, start a Python interpreter
    in a terminal, then run:
    >> import simetri.graphics as sg
    >> sg.hello()
    >>
    You should see an svg file with a message.
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

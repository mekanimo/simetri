'''TikZ related sketches are handled here.'''

from dataclasses import dataclass

from ..graphics.all_enums import Types, TexLoc


@dataclass
class TexSketch:
    """TexSketch is a dataclass for inserting code into the tex file.

    Attributes:
        code (str, optional): The code to be inserted. Defaults to None.
        location (TexLoc, optional): The location of the code. Defaults to TexLoc.NONE.

    Returns:
        None
    """

    code: str = None
    location: TexLoc = TexLoc.NONE

    def __post_init__(self):
        """Initialize the TexSketch object."""
        self.type = Types.SKETCH
        self.subtype = Types.TEX_SKETCH

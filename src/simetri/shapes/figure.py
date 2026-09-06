"""A shape with a skin. Skin could be any drawable object."""

from typing import Any, Self
from .shape import Shape
from ..group.batch import Group


class Figure:
    """A shape object with a skin. Skin can be any drawable object"""

    def __init__(self, geometry: Shape | Group, skin: Any) -> Self:
        self.geometry = geometry
        self.skin = skin

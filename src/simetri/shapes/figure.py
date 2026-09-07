"""A shape with a skin. Skin could be any drawable object."""

from __future__ import annotations

from typing import Any

from ..base.all_enums import Types
from ..group.batch import Group
from .shape import Shape


class Figure(Group):
    """A ``Group`` whose elements are ``geometry`` and optional ``skin``.

    ``type`` is ``Types.GROUP``; ``subtype`` is ``Types.FIGURE``. Bounding-box
    and transform APIs come from ``Group``.

    Attributes:
        geometry: Underlying geometric object (first element).
        skin: Drawable overlay (second element), if any.
        draw_geometry: If True, canvas draws ``geometry``.
        draw_skin: If True, canvas draws ``skin``.
    """

    def __init__(
        self,
        geometry: Shape | Group,
        skin: Any = None,
    ) -> None:
        """Create a figure from geometry and optional skin.

        Args:
            geometry: Underlying geometric object (first element).
            skin: Optional drawable overlay (second element).
        """
        if skin is not None and isinstance(skin, (list, tuple)):
            skin = Group(list(skin))
        elements = [geometry] if skin is None else [geometry, skin]
        super().__init__(elements=elements, subtype=Types.FIGURE)
        self.draw_geometry = True
        self.draw_skin = True

    @property
    def geometry(self) -> Shape | Group | None:
        """Underlying geometry (first group element)."""
        return self.elements[0] if self.elements else None

    @geometry.setter
    def geometry(self, value: Shape | Group) -> None:
        """Replace the underlying geometry (first group element)."""
        if self.elements:
            self.elements[0] = value
        else:
            self.elements = [value]

    @property
    def skin(self) -> Any:
        """Optional skin drawable (second group element)."""
        return self.elements[1] if len(self.elements) > 1 else None

    @skin.setter
    def skin(self, value: Any) -> None:
        """Set or clear the optional skin (second group element).

        Args:
            value: Drawable overlay, or ``None`` to remove the skin.

        Raises:
            ValueError: If the figure has no geometry yet.
        """
        if value is not None and isinstance(value, (list, tuple)):
            value = Group(list(value))
        if not self.elements:
            raise ValueError("Figure has no geometry; set geometry before skin.")
        if value is None:
            self.elements = self.elements[:1]
        elif len(self.elements) == 1:
            self.elements.append(value)
        else:
            self.elements[1] = value

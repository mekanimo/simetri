"""Regular polygons and patches for k-uniform tiling construction."""

from typing import Self

import numpy as np

from ..shapes.shape import Shape
from ..group.batch import Group
from ..base.all_enums import Types
from ..shapes.geom_items import reg_poly_points_side_length, snap
from ..base.common import PointType


class UniPoly(Shape):
    """A regular polygon with standard side-length (100 pts.).
    Used with k-uniform tilings and UniPatch objects.
    """

    def __init__(
        self,
        n_sides: int,
        pos: PointType = (0, 0),
        side_length: float = 100.0,
        xform_matrix: np.array = None,
        **kwargs,
    ):
        """Create a regular n-gon at ``pos`` with the given side length.

        Args:
            n_sides: Number of sides.
            pos: Placement position.
            side_length: Edge length in points. Defaults to 100.
            xform_matrix: Optional initial affine matrix.
            **kwargs: Passed to ``Shape``.
        """
        points = reg_poly_points_side_length(
            pos=post, n=n, side_len=side_length
        )
        super().__init__(
            points, closed=closed, xform_matrix=xform_matrix, **kwargs
        )
        self.subtype = Types.UNIPOLY
        range_n = range(self.n)
        self.free_edges = list(range_n)
        self.free_vertices = list(range_n)
        self.connected_edges = []
        self.connected_vertices = []

    def _snap(
        free_shape: UniPoly,
        ref1: int | float,
        ref2: int | float,
        angle: float = 0,
    ):
        snap(
            free_shape=free_shape,
            ref1=ref1,
            fixed_shape=self,
            ref2=ref2,
            angle=angle,
        )
        # update connected-edges
        # update free-edges
        # update connected-vertices


class UniPatch(Group):
    """Used with k-uniform tilings. A group of UniPoly objects."""

    def __init__(
        self, signature: str | int | list[int], pos: PointType = (0, 0)
    ):
        """Create a patch from a tiling signature at ``pos``.

        Args:
            signature: Patch identity (string, int, or list of ints).
            pos: Placement position. Defaults to ``(0, 0)``.
        """

        elements = self.init(signature)
        super().__init__(elements)
        self.subtype = Types.UNIPATCH

    def init(self, signature):
        """Build element list for ``signature`` (stub).

        Args:
            signature: Patch identity.

        Returns:
            Elements used to initialize the group (currently ``None``/pass).
        """
        pass

    def snap(
        free_shape: UniPoly,
        ref1: int | float,
        fixed_shape: UniPoly,
        ref2: int | float,
        angle: float = 0,
    ) -> Self:
        """Snap ``free_shape`` onto ``fixed_shape`` at matching refs (stub).

        Args:
            free_shape: Polygon to move.
            ref1: Reference on ``free_shape``.
            fixed_shape: Anchored polygon.
            ref2: Reference on ``fixed_shape``.
            angle: Optional rotation after snap.

        Returns:
            Self after snapping (stub).
        """
        pass

    def connect(uni_patch: UniPatch, indices: List | None = None) -> Self:
        """Connect this patch to another along shared indices (stub).

        Args:
            uni_patch: Other patch to connect.
            indices: Optional edge/vertex indices to join.

        Returns:
            Self after connecting (stub).
        """
        pass

    @property
    def free_boundary(self) -> list[pointType]:
        """Return free boundary points of the patch (stub)."""
        pass

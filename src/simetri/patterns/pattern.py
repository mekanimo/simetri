"""Repeated geometric patterns built from a kernel and transforms.

A ``Pattern`` stores a kernel Shape/Group plus a
``PatternTransformation`` (list of ``TransformMat`` matrices with
repetitions). Calling transform helpers such as ``translate`` / ``rotate``
appends transforms rather than baking them into the kernel.

Examples:
    >>> from simetri.config.settings import set_defaults
    >>> set_defaults()
    >>> kernel = Shape([(0, 0), (10, 0), (10, 10), (0, 10)], closed=True)
    >>> pattern = Pattern(kernel)
    >>> _ = pattern.rotate(1.0471975511965976, about=(0, 0), reps=2)
    >>> pattern.count
    3
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import md5
from itertools import product
from math import prod
from types import FunctionType
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from ..base.all_enums import (
    Anchor,
    InPlace,
    TransformationType,
    Types,
    get_enum_value,
)
from ..base.common import LineType, PointType
from ..base.common_style import COLOR_ALPHA_ATTRS, STYLE_COPY_ATTRS, CommonStyle
from ..base.core import DynRef, resolve_dyn_ref
from ..geom.affine import *
from ..geom.bbox import BoundingBox, bounding_box
from ..group.batch import Group
from ..helpers.validation import validate_args
from ..shapes.shape import Shape


@dataclass
class TransformMat:
    """A single transformation matrix with optional repetitions.

    Used inside ``PatternTransformation`` to build a composite matrix stack.

    Attributes:
        xform_matrix: 3×3 affine matrix (row form).
        reps: Number of repetitions (0 means identity only in partitions).
        incr: Optional increment between repetitions.
        take: Optional slice for selective application.

    Examples:
        >>> from simetri.config.settings import set_defaults
        >>> set_defaults()
        >>> step = TransformMat(translation_matrix(5, 0), reps=1)
        >>> len(step.partitions)
        2
        >>> step.reps
        1
    """

    xform_matrix: NDArray
    reps: int = 0
    incr: (
        float
        | tuple[float, float]
        | tuple[callable, Any]
        | tuple[InPlace, Any]
        | None
    ) = None
    take: slice = None

    def __post_init__(self) -> None:
        self.type = Types.TRANSFORM
        self.subtype = Types.TRANSFORM
        self.__dict__["_xform_matrix"] = self.xform_matrix
        self.__dict__["_reps"] = self.reps
        self._update()

    def _update(self) -> None:
        self.hash = md5(self.xform_matrix.tobytes()).hexdigest()
        self._set_partitions()
        self._composite = np.concatenate(self._partitions, axis=1)
        self._reps = self.reps

    def __repr__(self) -> str:
        return f"TransformMat(xform_matrix={self.xform_matrix}, reps={self.reps})"

    def __str__(self) -> str:
        return f"TransformMat(xform_matrix={self.xform_matrix}, reps={self.reps})"

    # @property
    # def reps(self) -> int:
    #     return self._reps

    # @reps.setter
    # def reps(self, value: int):
    #     if value < 0:
    #         raise ValueError("x cannot be negative")
    #     self._reps = value

    def _changed(self) -> bool:
        """
        Checks if the transformation matrix or reps value has changed.

        Returns:
            bool: True if the transformation state has changed, False otherwise.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> step = TransformMat(identity_matrix(), reps=0)
            >>> step._changed()
            False
        """
        return not (
            (self.hash == md5(self.xform_matrix.tobytes()).hexdigest())
            and (self.reps == self._reps)
        )

    def _set_partitions(self) -> None:
        if self.reps == 0:
            partition_list = [identity_matrix()]
        elif self.reps == 1:
            partition_list = [identity_matrix(), self.xform_matrix]
        else:
            xform_mat = self.xform_matrix
            partition_list = [identity_matrix(), xform_mat]
            last = xform_mat
            for _ in range(self.reps - 1):
                last = xform_mat @ last
                partition_list.append(last)

        self._partitions = partition_list

    def update(self) -> None:
        """Recompute ``partitions`` and ``composite`` from the current matrix.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> step = TransformMat(translation_matrix(1, 0), reps=2)
            >>> step.reps = 2
            >>> step.update()
            >>> len(step.partitions)
            3
        """
        self._update()

    @property
    def xform_matrix(self) -> NDArray:
        """
        Returns the transformation matrix.

        Returns:
            ndarray: The transformation matrix.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> step = TransformMat(translation_matrix(3, 4), reps=0)
            >>> float(step.xform_matrix[2, 0])
            3.0
        """

        return self._xform_matrix

    @xform_matrix.setter
    def xform_matrix(self, value: object) -> None:
        """Set the base transform matrix.

        Args:
            value: 3x3 affine matrix as a NumPy array.

        Raises:
            ValueError: If ``value`` is not a NumPy array.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> step = TransformMat(identity_matrix(), reps=0)
            >>> step.xform_matrix = identity_matrix()
            >>> step.xform_matrix.shape
            (3, 3)
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("xform_matrix must be a numpy array")
        self._xform_matrix = value

    @property
    def partitions(self) -> list:
        """
        Returns the submatrices in the transformation.

        Returns:
            list: A list of submatrices.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> step = TransformMat(translation_matrix(2, 0), reps=2)
            >>> len(step.partitions)
            3
        """
        if self._changed():
            self.update()

        return self._partitions

    @property
    def composite(self) -> NDArray:
        """
        Returns the compound transformation matrix.

        Returns:
            ndarray: The compound transformation matrix.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> step = TransformMat(translation_matrix(1, 0), reps=1)
            >>> step.composite.shape
            (3, 6)
        """
        if self._changed():
            self.update()

        return self._composite

    def copy(self) -> "TransformMat":
        """
        Creates a copy of the TransformMat instance.

        Returns:
            TransformMat: A new TransformMat instance with the same attributes.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> step = TransformMat(translation_matrix(1, 0), reps=1)
            >>> copy = step.copy()
            >>> copy.reps
            1
            >>> copy is step
            False
        """
        return TransformMat(self.xform_matrix.copy(), self.reps)


@dataclass
class PatternTransformation:
    """Ordered list of ``TransformMat`` components forming a pattern.

    Attributes:
        components: List of ``TransformMat`` instances applied in order.
        type: Always ``Types.TRANSFORMATION``.

    Examples:
        >>> from simetri.config.settings import set_defaults
        >>> set_defaults()
        >>> stack = PatternTransformation(
        ...     [TransformMat(translation_matrix(10, 0), reps=1)]
        ... )
        >>> stack.count
        2
    """

    components: list[TransformMat] = None

    def __post_init__(self) -> None:
        self.type = Types.TRANSFORMATION
        self.subtype = Types.TRANSFORMATION
        if self.components is None:
            self.components = []

    def __repr__(self) -> str:
        return f"PatternTransformation(components={self.components})"

    def __str__(self) -> str:
        return f"PatternTransformation(components={self.components})"

    def apply(self, kernel: Shape) -> Group:
        """Apply the composite transform to ``kernel`` and return copies.

        Args:
            kernel: Source shape whose ``final_coords`` are transformed.

        Returns:
            Group: One shape per transform partition, with kernel style copied.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> kernel = Shape([(0, 0), (5, 0), (5, 5)], closed=True)
            >>> stack = PatternTransformation(
            ...     [TransformMat(translation_matrix(10, 0), reps=1)]
            ... )
            >>> expanded = stack.apply(kernel)
            >>> len(expanded)
            2
        """
        all_vertices = kernel.final_coords @ self.composite
        vertices_list = np.hsplit(all_vertices, self.count)
        res = Group()
        for vertices in vertices_list:
            shape = Shape(vertices)
            shape.copy_style(kernel)
            res.append(shape)

        return res

    @property
    def count(self) -> int:
        """Return the number of shapes produced by this transformation stack.

        Returns:
            int: Product of ``(reps + 1)`` over all components.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> stack = PatternTransformation(
            ...     [
            ...         TransformMat(translation_matrix(1, 0), reps=1),
            ...         TransformMat(translation_matrix(0, 1), reps=1),
            ...     ]
            ... )
            >>> stack.count
            4
        """
        return prod([comp.reps + 1 for comp in self.components])

    @property
    def partitions(self) -> list:
        """
        Returns the submatrices in the transformation.

        Returns:
            list of ndarrays.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> stack = PatternTransformation()
            >>> len(stack.partitions)
            1
        """
        if len(self.components) == 0:
            return [identity_matrix()]
        elif len(self.components) == 1:
            partitions = [identity_matrix(), self.components[0].xform_matrix]
        else:
            partitions = []
            for component in self.components:
                partitions.extend(component.partitions)

        return partitions

    @property
    def composite(self) -> NDArray:
        """
        Returns the compound transformation matrix.

        Returns:
            ndarray: The compound transformation matrix.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> stack = PatternTransformation(
            ...     [TransformMat(translation_matrix(5, 0), reps=0)]
            ... )
            >>> stack.composite.shape
            (3, 3)
        """
        if len(self.components) == 0:
            return identity_matrix()
        matrices = [component.partitions for component in self.components]
        res = []
        if len(matrices) == 1:
            if len(matrices[0]) == 1:
                return matrices[0][0]
            else:
                return np.concatenate(matrices[0], axis=1)
        else:
            res.extend(np.linalg.multi_dot(mats) for mats in product(*matrices))

        return np.concatenate(res, axis=1)

    def copy(self) -> "PatternTransformation":
        """
        Creates a copy of the PatternTransformation instance.

        Returns:
            PatternTransformation: A new PatternTransformation with the same
            components.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> stack = PatternTransformation(
            ...     [TransformMat(translation_matrix(1, 0), reps=1)]
            ... )
            >>> copy = stack.copy()
            >>> copy.count
            2
            >>> copy is stack
            False
        """
        return PatternTransformation(
            [component.copy() for component in self.components]
        )


class Pattern(Group, CommonStyle):
    """Drawable pattern: a kernel repeated by a PatternTransformation.

    Transform methods (``translate``, ``rotate``, …) append to
    ``transformation`` instead of mutating the kernel geometry directly.

    Style lives on the Pattern (``CommonStyle``), same model as Shape/Path2D.
    ``get_shapes`` copies this pattern's style onto each expanded Shape.

    Attributes:
        kernel: Shape or Group that is repeated.
        transformation: Accumulated ``PatternTransformation``.
        subtype: Always ``Types.PATTERN``.

    Examples:
        >>> from simetri.config.settings import set_defaults
        >>> set_defaults()
        >>> pattern = Pattern(Shape([(0, 0), (5, 0), (5, 5)], closed=True))
        >>> _ = pattern.translate(10, 0, reps=3)
        >>> pattern.count
        4
    """

    # Group.__setattr__ adds a frame above the color/alpha property setters.
    _style_warning_stacklevel: int = 4

    def __init__(
        self,
        kernel: Shape | Group = None,
        transformation: PatternTransformation = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a Pattern.

        Args:
            kernel: Shape or Group to repeat.
            transformation: Optional existing PatternTransformation.
            **kwargs: Style attributes (``CommonStyle`` / ``STYLE_COPY_ATTRS``).

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> kernel = Shape([(0, 0), (1, 0), (1, 1)], closed=True)
            >>> pattern = Pattern(kernel)
            >>> pattern.kernel is kernel
            True
            >>> pattern.count
            1
        """
        self.kernel = kernel
        if transformation is None:
            transformation = PatternTransformation()

        self.transformation = transformation
        super().__init__()
        self.subtype = Types.PATTERN

        valid_args = list(COLOR_ALPHA_ATTRS) + list(STYLE_COPY_ATTRS)
        validate_args(kwargs, valid_args)
        self._init_from_style_kwargs(kwargs)
        if kwargs:
            raise TypeError(
                f"Unexpected keyword arguments: {sorted(kwargs)}"
            )

    def __repr__(self) -> str:
        return f"Pattern(kernel={self.kernel}, transformation={self.transformation})"

    def __str__(self) -> str:
        return f"Pattern(kernel={self.kernel}, transformation={self.transformation})"

    @property
    def closed(self) -> bool:
        """
        Returns True if the pattern is closed.

        Returns:
            bool: True if the pattern is closed, False otherwise.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> pattern = Pattern(Shape([(0, 0), (1, 0), (1, 1)], closed=True))
            >>> pattern.closed
            True
        """
        return self.kernel.closed

    @closed.setter
    def closed(self, value: bool) -> None:
        """
        Sets the closed property of the pattern.

        Args:
            value (bool): True to set the pattern as closed, False otherwise.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> pattern = Pattern(Shape([(0, 0), (1, 0), (1, 1)], closed=True))
            >>> pattern.closed = False
            >>> pattern.kernel.closed
            False
        """
        self.kernel.closed = value

    @property
    def composite(self) -> NDArray:
        """Return the pattern's composite transform from ``transformation``.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> pattern = Pattern(Shape([(0, 0), (1, 0)]))
            >>> _ = pattern.translate(5, 0, reps=0)
            >>> pattern.composite.shape[0]
            3
        """
        return self.transformation.composite

    def __bool__(self) -> bool:
        return bool(self.kernel)

    @property
    def all_vertices(self) -> NDArray:
        """
        Returns flat (x, y) coordinates for all shapes in the pattern.

        Returns:
            ndarray: Array of shape (n_verts * count, 2) with all (x, y) positions.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> pattern = Pattern(Shape([(0, 0), (1, 0)]))
            >>> _ = pattern.translate(10, 0, reps=1)
            >>> pattern.all_vertices.shape
            (4, 2)
        """
        raw = self.kernel.final_coords @ self.composite
        splits = np.hsplit(raw, self.count)
        return np.vstack([s[:, :2] for s in splits])

    @property
    def b_box(self) -> BoundingBox:
        """
        Returns the bounding box of the pattern.

        Returns:
            BoundingBox: The bounding box of the pattern.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> pattern = Pattern(Shape([(0, 0), (10, 0), (10, 10)], closed=True))
            >>> pattern.b_box.width
            10.0
        """
        return bounding_box(self.all_vertices)

    def get_vertices_list(self) -> list:
        """
        Returns the per-shape vertex submatrices (homogeneous coords).

        Returns:
            list: A list of ndarrays of shape (n_verts, 3), one per copy.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> pattern = Pattern(Shape([(0, 0), (1, 0)]))
            >>> _ = pattern.translate(2, 0, reps=1)
            >>> len(pattern.get_vertices_list())
            2
        """
        raw = self.kernel.final_coords @ self.composite
        return np.hsplit(raw, self.count)

    def get_shapes(self) -> Group:
        """
        Expands the pattern into a group of shapes.

        Returns:
            Group: A new Group instance with the expanded shapes.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> pattern = Pattern(Shape([(0, 0), (1, 0)]))
            >>> _ = pattern.translate(3, 0, reps=2)
            >>> len(pattern.get_shapes())
            3
        """
        vertices_list = self.get_vertices_list()
        res = Group()
        kernel = self.kernel
        for vertices in vertices_list:
            shape = Shape(vertices, closed=kernel.closed)
            shape.copy_style(self)
            res.append(shape)

        return res

    @property
    def count(self) -> int:
        """Return the total number of expanded shapes in the pattern.

        Returns:
            int: Same as ``transformation.count``.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> pattern = Pattern(Shape([(0, 0), (1, 0)]))
            >>> _ = pattern.rotate(1.5707963267948966, reps=1)
            >>> pattern.count
            2
        """
        return self.transformation.count

    def copy(self) -> "Pattern":
        """
        Creates a copy of the Pattern instance.

        Returns:
            Pattern: A new Pattern instance with the same attributes.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> pattern = Pattern(Shape([(0, 0), (1, 0)]))
            >>> _ = pattern.translate(1, 0, reps=1)
            >>> duplicate = pattern.copy()
            >>> duplicate.count
            2
            >>> duplicate is pattern
            False
        """
        kernel = None
        if self.kernel is not None:
            kernel = self.kernel.copy()

        transformation = None
        if self.transformation is not None:
            transformation = self.transformation.copy()

        pattern = Pattern(kernel, transformation)
        pattern.copy_style(self)
        return pattern

    def translate(self, dx: float = 0, dy: float = 0, reps: int = 0) -> Self:
        """
        Translates the object by dx and dy.

        Args:
            dx (float): The translation distance along the x-axis.
            dy (float): The translation distance along the y-axis.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The transformed object.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> pattern = Pattern(Shape([(0, 0), (1, 0)]))
            >>> pattern.translate(10, 0, reps=2) is pattern
            True
            >>> pattern.count
            3
        """

        component = TransformMat(translation_matrix(dx, dy), reps)
        self.transformation.components.append(component)

        return self

    def rotate(
        self, angle: float, about: PointType = (0, 0), reps: int = 0
    ) -> Self:
        """
        Rotates the object by the given angle (in radians) about the given point.

        Args:
            angle (float): The rotation angle in radians.
            about (PointType, optional): The point to rotate about. Defaults to (0, 0).
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The rotated object.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> pattern = Pattern(Shape([(0, 0), (1, 0)]))
            >>> _ = pattern.rotate(1.5707963267948966, about=(0, 0), reps=1)
            >>> pattern.count
            2
        """
        component = TransformMat(rotation_matrix(angle, about), reps)
        self.transformation.components.append(component)

        return self

    def mirror(self, about: LineType | PointType, reps: int = 0) -> Self:
        """
        Mirrors the object about the given line or point.

        Args:
            about (Line | PointType): The line or point to mirror about.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The mirrored object.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> pattern = Pattern(Shape([(0, 0), (1, 0)]))
            >>> _ = pattern.mirror(((0, 0), (1, 0)), reps=1)
            >>> pattern.count
            2
        """
        component = TransformMat(mirror_matrix(about), reps)
        self.transformation.components.append(component)

        return self

    def glide(
        self, glide_line: LineType, glide_dist: float, reps: int = 0
    ) -> Self:
        """
        Glides (first mirror then translate) the object along the given line
        by the given glide_dist.

        Args:
            glide_line (Line): The line to glide along.
            glide_dist (float): The distance to glide.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The glided object.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> pattern = Pattern(Shape([(0, 0), (1, 0)]))
            >>> _ = pattern.glide(((0, 0), (10, 0)), 5, reps=1)
            >>> pattern.count
            2
        """
        component = TransformMat(glide_matrix(glide_line, glide_dist), reps)
        self.transformation.components.append(component)

        return self

    def scale(
        self,
        scale_x: float,
        scale_y: float | None = None,
        about: PointType = (0, 0),
        reps: int = 0,
    ) -> Self:
        """
        Scales the object by the given scale factors about the given point.

        Args:
            scale_x (float): The scale factor in the x direction.
            scale_y (float, optional): The scale factor in the y direction. Defaults to None.
            about (PointType, optional): The point to scale about. Defaults to (0, 0).
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The scaled object.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> pattern = Pattern(Shape([(0, 0), (1, 0)]))
            >>> _ = pattern.scale(2, reps=1)
            >>> pattern.count
            2
        """
        if scale_y is None:
            scale_y = scale_x
        component = TransformMat(
            scale_in_place_matrix(scale_x, scale_y, about), reps
        )
        self.transformation.components.append(component)

        return self

    def shear(self, theta_x: float, theta_y: float, reps: int = 0) -> Self:
        """
        Shears the object by the given angles.

        Args:
            theta_x (float): The shear angle in the x direction.
            theta_y (float): The shear angle in the y direction.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The sheared object.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> pattern = Pattern(Shape([(0, 0), (1, 0)]))
            >>> _ = pattern.shear(0.1, 0.0, reps=0)
            >>> pattern.count
            1
        """
        component = TransformMat(shear_matrix(theta_x, theta_y), reps)
        self.transformation.components.append(component)

        return self

    def transform(self, transform_matrix: NDArray, reps: int = 0) -> Self:
        """
        Transforms the pattern by the given transformation matrix.

        Args:
            transform_matrix (ndarray): The transformation matrix.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The transformed pattern.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> pattern = Pattern(Shape([(0, 0), (1, 0)]))
            >>> _ = pattern.transform(translation_matrix(4, 0), reps=1)
            >>> pattern.count
            1
        """
        return self._update(transform_matrix, reps=reps)

    def move_to(self, pos: PointType, anchor: Anchor = Anchor.CENTER) -> Self:
        """
        Moves the object to the given position by using its center point.

        Args:
            pos (PointType): The position to move to.
            anchor (Anchor, optional): The anchor point. Defaults to Anchor.CENTER.

        Returns:
            Self: The moved object.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> pattern = Pattern(Shape([(0, 0), (10, 0), (10, 10)], closed=True))
            >>> _ = pattern.move_to((50, 50))
            >>> len(pattern.transformation.components)
            1
        """
        x, y = pos[:2]
        anchor = get_enum_value(Anchor, anchor)
        x1, y1 = getattr(self.b_box, anchor)
        component = TransformMat(translation_matrix(x - x1, y - y1), reps=0)
        self.transformation.components.append(component)

        return self


# The difference between Group and Pattern is the way they are drawn.
# Group objects are drawn only one way. Their sketches are handled differently.
# Groups behave like SVG groups and TikZ \pic


# class Group(Pattern):
#     """A class representing a group of objects.
#     Groups are optimized for repeating geometry to reduce file size and allow
#     for automatic simultaneous updates for all instances.

#     Attributes:
#         kernel (Shape/Group): The repeated form.
#         transformation: A Transformation object.
#     """

#     def __init__(
#         self,
#         kernel: Shape | Group = None,
#         transformation: PatternTransformation = None,
#         **kwargs,
#     ):
#         super().__init__(kernel, transformation, **kwargs)
#         self.subtype = Types.GROUP

#         valid_args = shape_args
#         validate_args(kwargs, valid_args)


@dataclass
class TransformDef:
    """One transform step in a ``PatternDef``.

    ``Def`` means definition: the operation and its arguments, applied
    later by ``PatternDef.apply``.

    Attributes:
        type: Transformation kind (translate, rotate, mirror, glide, ...).
        ref: Optional ``DynRef`` or literal for pivot/axis resolution.
        args: Transform arguments (angle, distance, ``(dx, dy)``, etc.).
        take: Optional slice selecting which elements to transform.
        incr: Optional increment between repetitions.
        reps: Number of repetitions.
        modifier: Optional callable applied to the transform.

    Examples:
        >>> from simetri.config.settings import set_defaults
        >>> set_defaults()
        >>> step = TransformDef(TransformationType.TRANSLATE, None, (1, 0), reps=2)
        >>> step.reps
        2
    """

    type: TransformationType  # translation, rotation, ...
    ref: DynRef | PointType | None
    args: DynRef | PointType | float | None = None
    take: slice = None
    incr: Any = None
    reps: int = 0
    modifier: Callable = None

    def copy(self) -> TransformDef:
        """Return a copy of this transform definition.

        Returns:
            TransformDef: Shallow copy with ``DynRef`` kwargs duplicated.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> step = TransformDef(TransformationType.ROTATE, (0, 0), 1.0, reps=1)
            >>> copy = step.copy()
            >>> copy.reps
            1
            >>> copy is step
            False
        """
        if isinstance(self.ref, DynRef):
            ref = replace(
                self.ref,
                kwargs=None
                if self.ref.kwargs is None
                else dict(self.ref.kwargs),
            )
        else:
            ref = self.ref
        if isinstance(self.args, DynRef):
            args = replace(
                self.args,
                kwargs=None
                if self.args.kwargs is None
                else dict(self.args.kwargs),
            )
        else:
            args = self.args
        return TransformDef(
            type=self.type,
            ref=ref,
            args=args,
            take=self.take,
            incr=self.incr,
            reps=self.reps,
            modifier=self.modifier,
        )


@dataclass
class PatternDef:
    """Sequence of ``TransformDef`` steps that build a pattern ``Group``.

    Attributes:
        transform_defs: Ordered list of transform definition steps.
        modifier: Optional callable applied to the finished pattern.

    Examples:
        >>> from simetri.config.settings import set_defaults
        >>> set_defaults()
        >>> definition = PatternDef(
        ...     [TransformDef(TransformationType.TRANSLATE, None, (5, 0), reps=1)]
        ... )
        >>> len(definition.transform_defs)
        1
    """

    transform_defs: list[TransformDef]
    modifier: Callable = None

    def apply(self, kernel: Shape | Group) -> Group:
        """Apply transform defs to ``kernel`` and return a pattern ``Group``.

        Args:
            kernel: Seed shape or group to transform.

        Returns:
            Group: Transformed copies after all definition steps.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> kernel = Shape([(0, 0), (1, 0)])
            >>> definition = PatternDef(
            ...     [TransformDef(TransformationType.TRANSLATE, None, (10, 0), reps=1)]
            ... )
            >>> group = definition.apply(kernel)
            >>> len(group)
            2
        """
        pattern = Group(kernel)
        for t_def in self.transform_defs:
            take = t_def.take
            reps = t_def.reps
            incr = t_def.incr
            if t_def.type == TransformationType.TRANSLATE:
                dx, dy = self.resolve_tuple(t_def.args, kernel, pattern)
                pattern.translate(dx, dy, take=take, reps=reps, incr=incr)
            elif t_def.type == TransformationType.ROTATE:
                pivot = self.resolve_reference(t_def.ref, kernel, pattern)
                angle = t_def.args
                pattern.rotate(angle, pivot, take=take, reps=reps, incr=incr)
            elif t_def.type == TransformationType.MIRROR:
                about = self.resolve_reference(t_def.ref, kernel, pattern)
                pattern.mirror(about, take=take, reps=reps, incr=incr)
            elif t_def.type == TransformationType.GLIDE:
                about = self.resolve_reference(t_def.ref, kernel, pattern)
                dist = self.resolve_value(t_def.args, kernel, pattern)
                pattern.glide(about, dist, take=take, reps=reps, incr=incr)
            elif t_def.type == TransformationType.SCALE:
                about = self.resolve_reference(t_def.ref, kernel, pattern)
                sx, sy = self.resolve_tuple(t_def.args, kernel.pattern)
                pattern.translate(
                    sx, sy, about, take=take, reps=reps, incr=incr
                )
            elif t_def.type == TransformationType.TRANSFORM:
                pattern.transform()

        return pattern

    def resolve_reference(
        self,
        reference: DynRef | Any,
        kernel: Shape | Group,
        pattern: Group,
    ) -> Any:
        """Resolve a ``DynRef`` against ``kernel`` / ``pattern``.

        Args:
            reference: A ``DynRef`` or an already-resolved value.
            kernel: Seed object for ``ReferenceTarget.KERNEL``.
            pattern: Growing pattern group for ``ReferenceTarget.PATTERN``.

        Returns:
            Any: Resolved point, line, or numeric value.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> kernel = Shape([(0, 0), (10, 0)])
            >>> pattern = Group(kernel)
            >>> definition = PatternDef([])
            >>> definition.resolve_reference((0, 0), kernel, pattern)
            (0, 0)
        """
        return resolve_dyn_ref(reference, kernel=kernel, pattern=pattern)

    def resolve_tuple(
        self,
        args: DynRef | tuple[Any, ...] | list[Any],
        kernel: Shape | Group,
        pattern: Group,
    ) -> Any:
        """Resolve a 2-tuple argument (or ``DynRef``) for transforms.

        Args:
            args: ``DynRef``, ``(x, y)``, or callable pair.
            kernel: Seed object used for nested resolution.
            pattern: Pattern group used for nested resolution.

        Returns:
            Any: Resolved ``(x, y)`` or other two-value result.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> kernel = Shape([(0, 0), (10, 0)])
            >>> pattern = Group(kernel)
            >>> definition = PatternDef([])
            >>> definition.resolve_tuple((3, 4), kernel, pattern)
            (3, 4)
        """
        if isinstance(args, DynRef):
            res = self.resolve_reference(args, kernel, pattern)
        elif isinstance(args, (tuple, list)):
            x, y = args
            if callable(x):
                res = x(**y)
            else:
                x = self.resolve_value(x, kernel, pattern)
                y = self.resolve_value(y, kernel, pattern)
                res = x, y

        return res

    def resolve_value(
        self,
        value: DynRef | Callable[..., Any] | Any,
        kernel: Shape | Group,
        pattern: Group,
    ) -> Any:
        """Resolve a scalar/reference argument for a transform.

        Args:
            value: ``DynRef``, callable, or literal value.
            kernel: Seed object used for nested resolution.
            pattern: Pattern group used for nested resolution.

        Returns:
            Any: Resolved numeric or geometric value.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> kernel = Shape([(0, 0), (10, 0)])
            >>> pattern = Group(kernel)
            >>> definition = PatternDef([])
            >>> definition.resolve_value(7, kernel, pattern)
            7
        """
        if isinstance(value, DynRef):
            res = self.resolve_reference(value, kernel, pattern)
        elif isinstance(value, FunctionType):
            value(kernel, pattern)
        else:
            res = value

        return res

    def copy(self) -> PatternDef:
        """Return a deep-enough copy of transform defs and modifier.

        Returns:
            PatternDef: New instance with copied ``TransformDef`` entries.

        Examples:
            >>> from simetri.config.settings import set_defaults
            >>> set_defaults()
            >>> definition = PatternDef(
            ...     [TransformDef(TransformationType.TRANSLATE, None, (1, 0))]
            ... )
            >>> copy = definition.copy()
            >>> len(copy.transform_defs)
            1
            >>> copy is definition
            False
        """
        return PatternDef(
            transform_defs=[t_def.copy() for t_def in self.transform_defs],
            modifier=self.modifier,
        )

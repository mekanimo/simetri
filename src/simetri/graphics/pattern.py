from math import prod
from itertools import product
from dataclasses import dataclass
from hashlib import md5
from typing import Any, List
from collections.abc import Callable, Sequence

import numpy as np
from typing_extensions import Union, Self

from .shape import Shape
from .batch import Batch
from .affine import *
from .common import PointType, LineType, common_properties
from .all_enums import (
    Types,
    get_enum_value,
    Anchor,
    InPlace,
    TransformationType,
    Reference,
    ReferenceTarget,
)
from .core import StyleMixin
from .bbox import bounding_box, BoundingBox
from ..canvas.style_map import ShapeStyle, shape_style_map, shape_args
from ..helpers.validation import validate_args
from ..geometry.geometry import offset_line, offset_point


@dataclass
class Transform:
    """
    A class representing a single transformation.
    Used in the Transformation class to represent a transformation matrix and its repetitions.

    Attributes:
        xform_matrix (ndarray): The transformation matrix.
        reps (int): The number of repetitions of the transformation.
    """

    xform_matrix: "ndarray"
    reps: int = 0
    incr: Union[
        float,
        tuple[float, float],
        tuple[callable, Any],
        tuple[InPlace, Any],
        None,
    ] = None
    take: slice = None

    def __post_init__(self):
        self.type = Types.TRANSFORM
        self.subtype = Types.TRANSFORM
        common_properties(self, graphics_object=False, id_only=True)
        self.__dict__["_xform_matrix"] = self.xform_matrix
        self.__dict__["_reps"] = self.reps
        self._update()

    def _update(self):
        self.hash = md5(self.xform_matrix.tobytes()).hexdigest()
        self._set_partitions()
        self._composite = np.concatenate(self._partitions, axis=1)
        self._reps = self.reps

    def __repr__(self):
        return f"Transform(xform_matrix={self.xform_matrix}, reps={self.reps})"

    def __str__(self):
        return f"Transform(xform_matrix={self.xform_matrix}, reps={self.reps})"

    # @property
    # def reps(self) -> int:
    #     return self._reps

    # @reps.setter
    # def reps(self, value: int):
    #     if value < 0:
    #         raise ValueError("x cannot be negative")
    #     self._reps = value

    def _changed(self):
        """
        Checks if the transformation matrix or reps value has changed.

        Returns:
            bool: True if the transformation state has changed, False otherwise.
        """
        return not (
            (self.hash == md5(self.xform_matrix.tobytes()).hexdigest())
            and (self.reps == self._reps)
        )

    def _set_partitions(self):
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

    def update(self):
        self._update()

    @property
    def xform_matrix(self) -> "ndarray":
        """
        Returns the transformation matrix.

        Returns:
            ndarray: The transformation matrix.
        """

        return self._xform_matrix

    @xform_matrix.setter
    def xform_matrix(self, value: "ndarray"):
        if not isinstance(value, np.ndarray):
            raise ValueError("xform_matrix must be a numpy array")
        self._xform_matrix = value
        self._update()

    @property
    def partitions(self) -> list:
        """
        Returns the submatrices in the transformation.

        Returns:
            list: A list of submatrices.
        """
        if self._changed():
            self.update()

        return self._partitions

    @partitions.setter
    def partitions(self, value: list):
        raise AttributeError(
            (
                "Cannot set partitions directly. "
                "Use the update method to update the partitions."
            )
        )

    @property
    def composite(self) -> "ndarray":
        """
        Returns the compound transformation matrix.

        Returns:
            ndarray: The compound transformation matrix.
        """
        if self._changed():
            self.update()

        return self._composite

    @composite.setter
    def composite(self, value: "ndarray"):
        raise AttributeError(
            (
                "Cannot set composition directly. "
                "Use the update method to update the composition."
            )
        )

    def copy(self) -> "Tranform":
        """
        Creates a copy of the Transform instance.

        Returns:
            Transform: A new Transform instance with the same attributes.
        """
        return Transform(self.xform_matrix.copy(), self.reps)


@dataclass
class Transformation:
    """
    A class representing a transformation that can be composite or not.

    Attributes:
        components (list): A list of Transform instances representing the transformations.
    """

    components: List[Transform] = None

    def __post_init__(self):
        self.type = Types.TRANSFORMATION
        self.subtype = Types.TRANSFORMATION
        if self.components is None:
            self.components = []
        common_properties(self, graphics_object=False, id_only=True)

    def __repr__(self):
        return f"Transformation(components={self.components})"

    def __str__(self):
        return f"Transformation(components={self.components})"

    def apply(self, kernel: Shape) -> List[Shape]:
        all_vertices = kernel.final_coords @ self.composite
        vertices_list = np.hsplit(all_vertices, self.count)
        res = Batch()
        style = kernel.style
        for vertices in vertices_list:
            shape = Shape(vertices)
            shape.style = style
            res.append(shape)

        return res

    @property
    def count(self):
        """
        Returns the number of individual shapes.

        Returns:
            int: The total number of shapes in the pattern.
        """

        return prod([comp.reps + 1 for comp in self.components])

    @property
    def partitions(self) -> list:
        """
        Returns the submatrices in the transformation.

        Returns:
            list of ndarrays.
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
    def composite(self) -> "array":
        """
        Returns the compound transformation matrix.

        Returns:
            ndarray: The compound transformation matrix.
        """
        if len(self.components) == 0:
            return identity_matrix()
        matrices = []
        for component in self.components:
            matrices.append(component.partitions)
        res = []
        if len(matrices) == 1:
            if len(matrices[0]) == 1:
                return matrices[0][0]
            else:
                return np.concatenate(matrices[0], axis=1)
        else:
            for mats in product(*matrices):
                res.append(np.linalg.multi_dot(mats))

        return np.concatenate(res, axis=1)

    @composite.setter
    def composite(self, value: "ndarray"):
        raise AttributeError(
            (
                "Cannot set composition directly. "
                "Use the update method to update the composition."
            )
        )

    def copy(self) -> "Transformation":
        """
        Creates a copy of the Transform instance.

        Returns:
            Transform: A new Transform instance with the same components.
        """
        return Transformation(
            [component.copy() for component in self.components]
        )


class Pattern(Batch, StyleMixin):
    """
    A class representing a pattern of a shape or batch object.

    Attributes:
        kernel (Shape/Batch): The repeated form.
        transformation: A Transformation object.
    """

    def __init__(
        self,
        kernel: Union[Shape, Batch] = None,
        transformation: Transformation = None,
        **kwargs,
    ):
        """
        Initializes the Pattern instance with a pattern and its count.

        Args:
            kernel (Shape/Batch): The repeated form of the pattern.
            transformation (Transformation): The transformation applied to the pattern.
            **kwargs: Additional keyword arguments.
        """
        self.__dict__["style"] = ShapeStyle()
        self.__dict__["_style_map"] = shape_style_map
        self._set_aliases()
        self.kernel = kernel
        if transformation is None:
            transformation = Transformation()

        self.transformation = transformation
        super().__init__(**kwargs)
        self.subtype = Types.PATTERN
        common_properties(self)

        valid_args = shape_args
        validate_args(kwargs, valid_args)

    def __repr__(self):
        return f"Pattern(kernel={self.kernel}, transformation={self.transformation})"

    def __str__(self):
        return f"Pattern(kernel={self.kernel}, transformation={self.transformation})"

    @property
    def closed(self) -> bool:
        """
        Returns True if the pattern is closed.

        Returns:
            bool: True if the pattern is closed, False otherwise.
        """
        return self.kernel.closed

    @closed.setter
    def closed(self, value: bool):
        """
        Sets the closed property of the pattern.

        Args:
            value (bool): True to set the pattern as closed, False otherwise.
        """
        self.kernel.closed = value

    @property
    def composite(self) -> "ndarray":
        return self.transformation.composite

    def __bool__(self):
        return bool(self.kernel)

    @property
    def all_vertices(self) -> "ndarray":
        """
        Returns flat (x, y) coordinates for all shapes in the pattern.

        Returns:
            ndarray: Array of shape (n_verts * count, 2) with all (x, y) positions.
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
        """
        return bounding_box(self.all_vertices)

    def get_vertices_list(self) -> list:
        """
        Returns the per-shape vertex submatrices (homogeneous coords).

        Returns:
            list: A list of ndarrays of shape (n_verts, 3), one per copy.
        """
        raw = self.kernel.final_coords @ self.composite
        return np.hsplit(raw, self.count)

    def get_shapes(self) -> Batch:
        """
        Expands the pattern into a batch of shapes.

        Returns:
            Batch: A new Batch instance with the expanded shapes.
        """
        vertices_list = self.get_vertices_list()
        res = Batch()
        kernel = self.kernel
        style = kernel.style
        for vertices in vertices_list:
            shape = Shape(vertices, closed=kernel.closed)
            shape.style = style
            res.append(shape)

        return res

    @property
    def count(self):
        """
        Returns the total number of shapes in the pattern.

        Returns:
            int: The total number of shapes in the pattern.
        """

        return self.transformation.count

    def copy(self) -> "Pattern":
        """
        Creates a copy of the Pattern instance.

        Returns:
            Pattern: A new Pattern instance with the same attributes.
        """
        kernel = None
        if self.kernel is not None:
            kernel = self.kernel.copy()

        transformation = None
        if self.transformation is not None:
            transformation = self.transformation.copy()

        pattern = Pattern(kernel, transformation)
        for attrib in shape_style_map:
            setattr(pattern, attrib, getattr(self, attrib))
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
        """

        component = Transform(translation_matrix(dx, dy), reps)
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
        """
        component = Transform(rotation_matrix(angle, about), reps)
        self.transformation.components.append(component)

        return self

    def mirror(self, about: Union[LineType, PointType], reps: int = 0) -> Self:
        """
        Mirrors the object about the given line or point.

        Args:
            about (Union[Line, PointType]): The line or point to mirror about.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The mirrored object.
        """
        component = Transform(mirror_matrix(about), reps)
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
        """
        component = Transform(glide_matrix(glide_line, glide_dist), reps)
        self.transformation.components.append(component)

        return self

    def scale(
        self,
        scale_x: float,
        scale_y: Union[float, None] = None,
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
        """
        if scale_y is None:
            scale_y = scale_x
        component = Transform(
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
        """
        component = Transform(shear_matrix(theta_x, theta_y), reps)
        self.transformation.components.append(component)

        return self

    def transform(self, transform_matrix: "ndarray", reps: int = 0) -> Self:
        """
        Transforms the pattern by the given transformation matrix.

        Args:
            transform_matrix (ndarray): The transformation matrix.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The transformed pattern.
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
        """
        x, y = pos[:2]
        anchor = get_enum_value(Anchor, anchor)
        x1, y1 = getattr(self.b_box, anchor)
        component = Transform(translation_matrix(x - x1, y - y1), reps=0)
        self.transformation.components.append(component)

        return self


# The difference between Group and Pattern is the way they are drawn.
# Group objects are drawn only one way. Their sketches are handled differently.
# Groups behave like SVG groups and TikZ \pic


class Group(Pattern):
    """A class representing a group of objects.
    Groups are optimized for repeating geometry to reduce file size and allow
    for automatic simultaneous updates for all instances.

    Attributes:
        kernel (Shape/Batch): The repeated form.
        transformation: A Transformation object.
    """

    def __init__(
        self,
        kernel: Union[Shape, Batch] = None,
        transformation: Transformation = None,
        **kwargs,
    ):
        super().__init__(kernel, transformation, **kwargs)
        self.subtype = Types.GROUP
        common_properties(self)

        valid_args = shape_args
        validate_args(kwargs, valid_args)


@dataclass
class ReferenceDef:
    reference: Reference  # Bounding-box references
    target: Union[ReferenceTarget | None] = None  # kernel, pattern, or None
    offset: Union[PointType | float] = (
        None  # float for line, <dx, dy> for point offset
    )
    multiplier: float = None
    modifier: Callable = None
    kwargs: Union[dict | None] = None

    def copy(self):
        return ReferenceDef(
            reference=self.reference,
            target=self.target,
            offset=self.offset,
            multiplier=self.multiplier,
            modifier=self.modifier,
            kwargs=self.kwargs.copy() if self.kwargs is not None else None,
        )


@dataclass
class TransformDef:
    type: TransformationType  # translation, rotation, ...
    ref: ReferenceDef
    args: Union[ReferenceDef | PointType | float] = None
    take: slice = None
    incr: Any = None
    reps: int = 0
    modifier: Callable = None

    def copy(self):
        return TransformDef(
            type=self.type,
            ref=self.ref.copy() if self.ref is not None else None,
            args=self.args.copy()
            if isinstance(self.args, ReferenceDef)
            else self.args,
            take=self.take,
            incr=self.incr,
            reps=self.reps,
            modifier=self.modifier,
        )


@dataclass
class PatternDef:
    transform_defs: List[TransformDef]
    modifier: Callable = None

    def apply(self, kernel) -> Batch:
        pattern = Batch(kernel)
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

    def resolve_reference(self, reference_def, kernel, pattern):
        if not isinstance(reference_def, ReferenceDef):
            return reference_def

        def apply_offset(value, offset):
            if isinstance(offset, ReferenceDef):
                offset = self.resolve_reference(offset, kernel, pattern)
            elif isinstance(offset, (tuple, List)):
                offset = tuple(
                    self.resolve_value(item, kernel, pattern) for item in offset
                )

            if isinstance(value, (float, int)):
                # number
                offset_val = value + offset
            elif isinstance(value, (tuple, List)):
                x, y = value
                if isinstance(x, (tuple, list)):
                    # line
                    offset_val = offset_line(value, offset)
                elif isinstance(x, (float, int)):
                    # point type
                    offset_val = offset_point(value, offset[0], offset[1])
                elif callable(x):
                    offset_val = x(**y)

            return offset_val

        ref_def = reference_def
        if ref_def.target is None:
            ref_def.target = ReferenceTarget.PATTERN

        if ref_def.target == ReferenceTarget.KERNEL:
            res = getattr(kernel, ref_def.reference)
        elif ref_def.target == ReferenceTarget.PATTERN:
            res = getattr(pattern, ref_def.reference)

        if ref_def.multiplier is not None:
            res *= ref_def.multiplier

        if ref_def.modifier is not None:
            res = ref_def.modifier(res)

        offset = ref_def.offset
        if offset is not None:
            res = apply_offset(res, offset)

        return res

    def resolve_tuple(self, args, kernel, pattern):
        if isinstance(args, ReferenceDef):
            res = self.resolve_reference(args, kernel, pattern)
        elif isinstance(args, (tuple, List)):
            x, y = args
            if callable(x):
                res = x(**y)
            else:
                x = self.resolve_value(x, kernel, pattern)
                y = self.resolve_value(y, kernel, pattern)
                res = x, y

        return res

    def resolve_value(self, value, kernel, pattern):
        if isinstance(value, ReferenceDef):
            res = self.resolve_reference(value, kernel, pattern)
        else:
            res = value

        return res

    def copy(self):
        return PatternDef(
            transform_defs=[t_def.copy() for t_def in self.transform_defs],
            modifier=self.modifier,
        )

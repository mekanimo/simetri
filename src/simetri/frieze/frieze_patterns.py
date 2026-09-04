"""Creates pattern definitions for the Frieze groups.

Each ``*Def`` dataclass builds a ``PatternDef``
for one frieze symmetry (hop, step, jump, sidle, and spinning variants).
Factory helpers ``hop_def``, ``step_def``, ``jump_def``, and ``sidle_def``
return plain ``PatternDef`` objects.

Examples:
    >>> import simetri.graphics as sg
    >>> from simetri.frieze.frieze_patterns import HopDef
    >>> motif = sg.Circle(10)
    >>> HopDef(dx=40, reps=4).apply(motif)
"""

from math import pi
from dataclasses import dataclass

from ..graphics.pattern import ReferenceDef, PatternDef, TransformDef
from ..graphics.all_enums import (
    TransformationType,
    ReferenceTarget,
    Reference,
    Types,
)
from ..graphics.common import PointType


@dataclass
class HopDef:
    """p1 (translation) frieze pattern definition.

    Attributes:
        dx: Horizontal translation distance or reference.
        dy: Vertical translation distance or reference.
        reps: Number of translation repetitions.
        pattern_def: Built ``PatternDef``.
    """

    dx: float | ReferenceDef
    dy: float | ReferenceDef = 0
    reps: int = 3

    def __post_init__(self):
        self.type = Types.PATTERN_DEF
        self.subtype = Types.HOP_DEF
        self._build_pattern()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == "reps" and "pattern_def" in self.__dict__:
            self._build_pattern()
        elif name in ["dx", "dy"] and "pattern_def" in self.__dict__:
            self.pattern_def.transform_defs[0].args = (self.dx, self.dy)

    def _build_pattern(self):
        """Rebuild pattern_def from current distance."""
        t_type = TransformationType.TRANSLATE
        args = (self.dx, self.dy)
        trans_def = TransformDef(t_type, None, args, reps=self.reps)
        self.pattern_def = PatternDef([trans_def])

    def apply(self, design):
        """Apply this hop pattern to ``design``.

        Args:
            design: Shape or group to transform.

        Returns:
            The transformed design (pattern result).
        """
        return self.pattern_def.apply(design)


@dataclass
class StepDef:
    """p11g (glide) frieze pattern definition.

    Attributes:
        mirror_offset: Offset of the glide axis from the kernel side.
        distance: Glide distance.
        side: Which side of the kernel supplies the glide axis.
        reps: Number of glide repetitions.
        pattern_def: Built ``PatternDef``.
    """

    mirror_offset: float | ReferenceDef
    distance: float | ReferenceDef
    side: Reference = Reference.BOTTOM
    reps: int = 3

    def __post_init__(self):
        self.type = Types.PATTERN_DEF
        self.subtype = Types.STEP_DEF
        self._build_pattern()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (
            name in ["mirror_offset", "distance", "side", "reps"]
            and "pattern_def" in self.__dict__
        ):
            self._build_pattern()

    def _build_pattern(self):
        """Rebuild pattern_def from current glide parameters."""
        t_type = TransformationType.GLIDE
        target = ReferenceTarget.KERNEL
        ref_def = ReferenceDef(self.side, target, self.mirror_offset)
        glide_def = TransformDef(t_type, ref_def, self.distance, reps=self.reps)
        self.pattern_def = PatternDef([glide_def])

    def apply(self, design):
        """Apply this step (glide) pattern to ``design``.

        Args:
            design: Shape or group to transform.

        Returns:
            The transformed design (pattern result).
        """
        return self.pattern_def.apply(design)


@dataclass
class JumpDef:
    """p11m (mirror then translate) frieze pattern definition.

    Attributes:
        mirror_offset: Offset of the mirror line from the kernel side.
        distance: Translation distance after mirroring.
        side: Which side of the kernel supplies the mirror line.
        reps: Number of translation repetitions.
        pattern_def: Built ``PatternDef``.
    """

    mirror_offset: float | ReferenceDef
    distance: float | ReferenceDef
    side: Reference = Reference.BOTTOM
    reps: int = 3

    def __post_init__(self):
        self.type = Types.PATTERN_DEF
        self.subtype = Types.JUMP_DEF
        self._build_pattern()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (
            name in ["mirror_offset", "distance", "side", "reps"]
            and "pattern_def" in self.__dict__
        ):
            self._build_pattern()

    def _build_pattern(self):
        """Rebuild pattern_def from current mirror/translate parameters."""
        t_type = TransformationType.MIRROR
        target = ReferenceTarget.KERNEL
        ref_def = ReferenceDef(self.side, target, self.mirror_offset)
        mirror_def = TransformDef(t_type, ref_def, reps=1)
        t_type2 = TransformationType.TRANSLATE
        trans_def = TransformDef(
            t_type2,
            ref=None,
            args=(self.distance, 0),
            reps=self.reps,
        )
        self.pattern_def = PatternDef([mirror_def, trans_def])

    def apply(self, design):
        """Apply this jump pattern to ``design``.

        Args:
            design: Shape or group to transform.

        Returns:
            The transformed design (pattern result).
        """
        return self.pattern_def.apply(design)


@dataclass
class SidleDef:
    """p1m1 (vertical mirror then translate) frieze pattern definition.

    Attributes:
        mirror_offset: Offset of the right-side mirror from the kernel.
        dx: Horizontal translation after mirroring.
        reps: Number of translation repetitions.
        pattern_def: Built ``PatternDef``.
    """

    mirror_offset: float | ReferenceDef
    dx: float | ReferenceDef
    reps: int = 0

    def __post_init__(self):
        self.type = Types.PATTERN_DEF
        self.subtype = Types.SIDLE_DEF
        self._build_pattern()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (
            name in ["mirror_offset", "dx", "reps"]
            and "pattern_def" in self.__dict__
        ):
            self._build_pattern()

    def _build_pattern(self):
        """Rebuild pattern_def from current mirror/translate parameters."""
        mirror_def = TransformDef(
            TransformationType.MIRROR,
            ReferenceDef(
                Reference.RIGHT, ReferenceTarget.KERNEL, self.mirror_offset
            ),
            reps=1,
        )
        trans_def = TransformDef(
            TransformationType.TRANSLATE,
            None,
            (self.dx, 0),
            reps=self.reps,
        )
        self.pattern_def = PatternDef([mirror_def, trans_def])

    def apply(self, design):
        """Apply this sidle pattern to ``design``.

        Args:
            design: Shape or group to transform.

        Returns:
            The transformed design (pattern result).
        """
        return self.pattern_def.apply(design)


@dataclass
class SpinningHopDef:
    """p2 (180° rotation then translate) frieze pattern definition.

    Attributes:
        rotocenter: Center of the 180° rotation.
        dx: Horizontal translation after rotation.
        dy: Vertical translation after rotation.
        reps: Number of translation repetitions.
        pattern_def: Built ``PatternDef``.
    """

    rotocenter: PointType | ReferenceDef
    dx: float | ReferenceDef
    dy: float | ReferenceDef = 0
    reps: int = 3

    def __post_init__(self):
        self.type = Types.PATTERN_DEF
        self.subtype = Types.SPINNING_HOP_DEF
        self._build_pattern()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (
            name in ["rotocenter", "dx", "dy", "reps"]
            and "pattern_def" in self.__dict__
        ):
            self._build_pattern()

    def _build_pattern(self):
        """Rebuild pattern_def from current rotation/translate parameters."""
        rotate_def = TransformDef(
            TransformationType.ROTATE,
            self.rotocenter,
            pi,
            reps=1,
        )
        trans_def = TransformDef(
            TransformationType.TRANSLATE,
            None,
            (self.dx, self.dy),
            reps=self.reps,
        )
        self.pattern_def = PatternDef([rotate_def, trans_def])

    def apply(self, design):
        """Apply this spinning-hop pattern to ``design``.

        Args:
            design: Shape or group to transform.

        Returns:
            The transformed design (pattern result).
        """
        return self.pattern_def.apply(design)


@dataclass
class SpinningJumpDef:
    """p2mm (two mirrors then translate) frieze pattern definition.

    Attributes:
        mirror_offset1: Offset for the vertical (right) mirror.
        mirror_offset2: Offset for the horizontal (bottom) mirror.
        dx: Horizontal translation after mirroring.
        dy: Vertical translation after mirroring.
        reps: Number of translation repetitions.
        pattern_def: Built ``PatternDef``.
    """

    mirror_offset1: float | ReferenceDef
    mirror_offset2: float | ReferenceDef
    dx: float | ReferenceDef
    dy: float | ReferenceDef = 0
    reps: int = 3

    def __post_init__(self):
        self.type = Types.PATTERN_DEF
        self.subtype = Types.SPINNING_JUMP_DEF
        self._build_pattern()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (
            name in ["mirror_offset1", "mirror_offset2", "dx", "dy", "reps"]
            and "pattern_def" in self.__dict__
        ):
            self._build_pattern()

    def _build_pattern(self):
        """Rebuild pattern_def from current mirror/translate parameters."""
        mirror_def1 = TransformDef(
            TransformationType.MIRROR,
            ReferenceDef(
                Reference.RIGHT, ReferenceTarget.KERNEL, self.mirror_offset1
            ),
            reps=1,
        )
        mirror_def2 = TransformDef(
            TransformationType.MIRROR,
            ReferenceDef(
                Reference.BOTTOM, ReferenceTarget.PATTERN, self.mirror_offset2
            ),
            reps=1,
        )
        trans_def = TransformDef(
            TransformationType.TRANSLATE,
            None,
            (self.dx, self.dy),
            reps=self.reps,
        )
        self.pattern_def = PatternDef([mirror_def1, mirror_def2, trans_def])

    def apply(self, design):
        """Apply this spinning-jump pattern to ``design``.

        Args:
            design: Shape or group to transform.

        Returns:
            The transformed design (pattern result).
        """
        return self.pattern_def.apply(design)


@dataclass
class SpinningSidleDef:
    """p2mg (mirror, glide, then translate) frieze pattern definition.

    Attributes:
        mirror_offset: Offset for the vertical mirror.
        glide_distance: Distance of the subsequent glide.
        dx: Horizontal translation after mirror/glide.
        dy: Vertical translation after mirror/glide.
        reps: Number of translation repetitions.
        pattern_def: Built ``PatternDef``.
    """

    mirror_offset: float | ReferenceDef
    glide_distance: float | ReferenceDef
    dx: float | ReferenceDef
    dy: float | ReferenceDef = 0
    reps: int = 3

    def __post_init__(self):
        self.type = Types.PATTERN_DEF
        self.subtype = Types.SPINNING_SIDLE_DEF
        self._build_pattern()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (
            name in ["mirror_offset", "glide_distance", "dx", "dy", "reps"]
            and "pattern_def" in self.__dict__
        ):
            self._build_pattern()

    def _build_pattern(self):
        """Rebuild pattern_def from current mirror/glide/translate parameters."""
        mirror_def = TransformDef(
            TransformationType.MIRROR,
            ReferenceDef(
                Reference.RIGHT, ReferenceTarget.KERNEL, self.mirror_offset
            ),
            reps=1,
        )
        glide_def = TransformDef(
            TransformationType.GLIDE,
            ReferenceDef(Reference.BOTTOM, ReferenceTarget.KERNEL, 0),
            self.glide_distance,
            reps=1,
        )
        trans_def = TransformDef(
            TransformationType.TRANSLATE,
            None,
            (self.dx, self.dy),
            reps=self.reps,
        )
        self.pattern_def = PatternDef([mirror_def, glide_def, trans_def])

    def apply(self, design):
        """Apply this spinning-sidle pattern to ``design``.

        Args:
            design: Shape or group to transform.

        Returns:
            The transformed design (pattern result).
        """
        return self.pattern_def.apply(design)


def hop_def(distance: float | ReferenceDef, reps: int = 3) -> PatternDef:
    """Build a p1 (translation) pattern definition.

    Args:
        distance: Horizontal translation distance.
        reps: Number of repetitions.

    Returns:
        PatternDef: Translation-only pattern definition.
    """
    t_type = TransformationType.TRANSLATE
    args = (distance, 0)
    trans_def = TransformDef(t_type, None, args, reps=reps)
    pattern_def = PatternDef([trans_def])

    return pattern_def


def step_def(
    mirror_offset: float | ReferenceDef,
    distance: float | ReferenceDef,
    side: Reference = Reference.BOTTOM,
    reps: int = 3,
) -> PatternDef:
    """Build a p11g (glide) pattern definition.

    Args:
        mirror_offset: Offset of the glide axis from the kernel side.
        distance: Glide distance.
        side: Kernel side used for the glide axis.
        reps: Number of repetitions.

    Returns:
        PatternDef: Glide pattern definition.
    """
    t_type = TransformationType.GLIDE
    target = ReferenceTarget.KERNEL
    ref_def = ReferenceDef(side, target, mirror_offset)
    glide_def = TransformDef(t_type, ref_def, distance, reps=reps)
    pattern_def = PatternDef([glide_def])

    return pattern_def


def jump_def(
    mirror_offset: float | ReferenceDef,
    distance: float | ReferenceDef,
    side: Reference = Reference.BOTTOM,
    reps: int = 3,
) -> PatternDef:
    """Build a p11m (mirror then translate) pattern definition.

    Args:
        mirror_offset: Offset of the mirror line from the kernel side.
        distance: Translation distance after mirroring.
        side: Kernel side used for the mirror line.
        reps: Number of translation repetitions.

    Returns:
        PatternDef: Mirror-then-translate pattern definition.
    """
    t_type = TransformationType.MIRROR
    target = ReferenceTarget.KERNEL
    ref_def = ReferenceDef(side, target, mirror_offset)
    mirror_def = TransformDef(t_type, ref_def, reps=1)
    t_type2 = TransformationType.TRANSLATE
    trans_def = TransformDef(t_type2, ref=None, args=(distance, 0), reps=reps)
    pattern_def = PatternDef([mirror_def, trans_def])

    return pattern_def


def sidle_def(mirror_offset, dx, reps: int = 0):
    """Build a p1m1 (right mirror then translate) pattern definition.

    Args:
        mirror_offset: Offset of the right-side mirror from the kernel.
        dx: Horizontal translation after mirroring.
        reps: Number of translation repetitions.

    Returns:
        PatternDef: Mirror-then-translate pattern definition.
    """
    # reflect over right+offset, then translate by pattern width+dx
    mirror_def = TransformDef(
        TransformationType.MIRROR,
        ReferenceDef(Reference.RIGHT, ReferenceTarget.KERNEL, mirror_offset),
        reps=1,
    )
    trans_def = TransformDef(
        TransformationType.TRANSLATE, None, (dx, 0), reps=reps
    )
    return PatternDef([mirror_def, trans_def])

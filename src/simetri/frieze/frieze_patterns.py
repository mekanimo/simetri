"""Creates pattern defintions for the Frieze groups"""

from math import pi
from collections.abc import Callable
from typing import Union
from dataclasses import dataclass

from ..graphics.pattern import ReferenceDef, PatternDef, TransformDef
from ..graphics.all_enums import (
    TransformationType,
    ReferenceTarget,
    Reference,
    Types,
)
from ..graphics.common import PointType, common_properties


@dataclass
class HopDef:
    dx: Union[float, ReferenceDef]
    dy: Union[float, ReferenceDef] = 0
    reps: int = 3

    def __post_init__(self):
        self.type = Types.PATTERN_DEF
        self.subtype = Types.HOP_DEF
        common_properties(self, False)
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
        return self.pattern_def.apply(design)


@dataclass
class StepDef:
    mirror_offset: Union[float, ReferenceDef]
    distance: Union[float, ReferenceDef]
    side: Reference = Reference.BOTTOM
    reps: int = 3

    def __post_init__(self):
        self.type = Types.PATTERN_DEF
        self.subtype = Types.STEP_DEF
        common_properties(self, False)
        self._build_pattern()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (
            name in ["mirror_offset", "distance", "side", "reps"]
            and "pattern_def" in self.__dict__
        ):
            self._build_pattern()

    def _build_pattern(self):
        t_type = TransformationType.GLIDE
        target = ReferenceTarget.KERNEL
        ref_def = ReferenceDef(self.side, target, self.mirror_offset)
        glide_def = TransformDef(t_type, ref_def, self.distance, reps=self.reps)
        self.pattern_def = PatternDef([glide_def])

    def apply(self, design):
        return self.pattern_def.apply(design)


@dataclass
class JumpDef:
    mirror_offset: Union[float, ReferenceDef]
    distance: Union[float, ReferenceDef]
    side: Reference = Reference.BOTTOM
    reps: int = 3

    def __post_init__(self):
        self.type = Types.PATTERN_DEF
        self.subtype = Types.JUMP_DEF
        common_properties(self, False)
        self._build_pattern()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (
            name in ["mirror_offset", "distance", "side", "reps"]
            and "pattern_def" in self.__dict__
        ):
            self._build_pattern()

    def _build_pattern(self):
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
        return self.pattern_def.apply(design)


@dataclass
class SidleDef:
    mirror_offset: Union[float, ReferenceDef]
    dx: Union[float, ReferenceDef]
    reps: int = 0

    def __post_init__(self):
        self.type = Types.PATTERN_DEF
        self.subtype = Types.SIDLE_DEF
        common_properties(self, False)
        self._build_pattern()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ["mirror_offset", "dx", "reps"] and "pattern_def" in self.__dict__:
            self._build_pattern()

    def _build_pattern(self):
        mirror_def = TransformDef(
            TransformationType.MIRROR,
            ReferenceDef(Reference.RIGHT, ReferenceTarget.KERNEL, self.mirror_offset),
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
        return self.pattern_def.apply(design)


@dataclass
class SpinningHopDef:
    rotocenter: Union[PointType, ReferenceDef]
    dx: Union[float, ReferenceDef]
    dy: Union[float, ReferenceDef] = 0
    reps: int = 3

    def __post_init__(self):
        self.type = Types.PATTERN_DEF
        self.subtype = Types.SPINNING_HOP_DEF
        common_properties(self, False)
        self._build_pattern()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ["rotocenter", "dx", "dy", "reps"] and "pattern_def" in self.__dict__:
            self._build_pattern()

    def _build_pattern(self):
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
        return self.pattern_def.apply(design)


@dataclass
class SpinningJumpDef:
    mirror_offset1: Union[float, ReferenceDef]
    mirror_offset2: Union[float, ReferenceDef]
    dx: Union[float, ReferenceDef]
    dy: Union[float, ReferenceDef] = 0
    reps: int = 3

    def __post_init__(self):
        self.type = Types.PATTERN_DEF
        self.subtype = Types.SPINNING_JUMP_DEF
        common_properties(self, False)
        self._build_pattern()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (
            name in ["mirror_offset1", "mirror_offset2", "dx", "dy", "reps"]
            and "pattern_def" in self.__dict__
        ):
            self._build_pattern()

    def _build_pattern(self):
        mirror_def1 = TransformDef(
            TransformationType.MIRROR,
            ReferenceDef(Reference.RIGHT, ReferenceTarget.KERNEL, self.mirror_offset1),
            reps=1,
        )
        mirror_def2 = TransformDef(
            TransformationType.MIRROR,
            ReferenceDef(Reference.BOTTOM, ReferenceTarget.PATTERN, self.mirror_offset2),
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
        return self.pattern_def.apply(design)


@dataclass
class SpinningSidleDef:
    mirror_offset: Union[float, ReferenceDef]
    glide_distance: Union[float, ReferenceDef]
    dx: Union[float, ReferenceDef]
    dy: Union[float, ReferenceDef] = 0
    reps: int = 3

    def __post_init__(self):
        self.type = Types.PATTERN_DEF
        self.subtype = Types.SPINNING_SIDLE_DEF
        common_properties(self, False)
        self._build_pattern()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if (
            name in ["mirror_offset", "glide_distance", "dx", "dy", "reps"]
            and "pattern_def" in self.__dict__
        ):
            self._build_pattern()

    def _build_pattern(self):
        mirror_def = TransformDef(
            TransformationType.MIRROR,
            ReferenceDef(Reference.RIGHT, ReferenceTarget.KERNEL, self.mirror_offset),
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
        return self.pattern_def.apply(design)


def hop_def(distance: Union[float, ReferenceDef], reps: int = 3) -> PatternDef:
    t_type = TransformationType.TRANSLATE
    args = (distance, 0)
    trans_def = TransformDef(t_type, None, args, reps=reps)
    pattern_def = PatternDef([trans_def])

    return pattern_def


def step_def(
    mirror_offset: Union[float, ReferenceDef],
    distance: Union[float, ReferenceDef],
    side: Reference = Reference.BOTTOM,
    reps: int = 3,
) -> PatternDef:
    t_type = TransformationType.GLIDE
    target = ReferenceTarget.KERNEL
    ref_def = ReferenceDef(side, target, mirror_offset)
    glide_def = TransformDef(t_type, ref_def, distance, reps=reps)
    pattern_def = PatternDef([glide_def])

    return pattern_def


def jump_def(
    mirror_offset: Union[float, ReferenceDef],
    distance: Union[float, ReferenceDef],
    side: Reference = Reference.BOTTOM,
    reps: int = 3,
) -> PatternDef:
    t_type = TransformationType.MIRROR
    target = ReferenceTarget.KERNEL
    ref_def = ReferenceDef(side, target, mirror_offset)
    mirror_def = TransformDef(t_type, ref_def, reps=1)
    t_type2 = TransformationType.TRANSLATE
    trans_def = TransformDef(t_type2, ref=None, args=(distance, 0), reps=reps)
    pattern_def = PatternDef([mirror_def, trans_def])

    return pattern_def


def sidle_def(mirror_offset, dx, reps: int = 0):
    # reflect over right+offset, then translate by pattern width+dx
    mirror_def = TransformDef(
        TransformationType.MIRROR,
        ReferenceDef(Reference.RIGHT, ReferenceTarget.KERNEL, mirror_offset),
        reps=1,
    )
    trans_def = TransformDef(TransformationType.TRANSLATE, None, (dx, 0), reps=reps)
    return PatternDef([mirror_def, trans_def])
"""Frieze / strip pattern definitions (PS1–PS15) built from ``PatternDef``.

Each ``PSn`` returns a ``PatternDef`` describing translations, mirrors,
glides, and/or 180° rotations used to repeat a kernel along a strip.
"""

from __future__ import annotations

import simetri.graphics as sg
from simetri.base.all_enums import ReferenceTarget, TransformationType
from simetri.patterns.pattern import PatternDef, ReferenceDef, TransformDef


def PS1(dx, reps: int = 0):
    """Translation-only strip: repeat by ``(dx, 0)``.

    Args:
        dx: Horizontal translation step.
        reps: Number of repetitions (``0`` means unlimited / pattern default).

    Returns:
        PatternDef for a pure translation strip.
    """
    t_type = sg.TransformationType.TRANSLATE
    args = (dx, 0)
    trans_def = sg.TransformDef(t_type, None, args, reps=reps)
    pattern_def = sg.PatternDef([trans_def])

    return pattern_def


def PS2(mirror_offset, distance, reps: int = 0):
    """Glide reflection along the kernel bottom edge.

    Args:
        mirror_offset: Offset of the glide axis from ``Reference.BOTTOM``.
        distance: Glide distance along the axis.
        reps: Number of repetitions.

    Returns:
        PatternDef for a glide strip.
    """
    t_type = sg.TransformationType.GLIDE
    target = sg.ReferenceTarget.KERNEL
    ref_def = sg.ReferenceDef(sg.Reference.BOTTOM, target, mirror_offset)
    glide_def = sg.TransformDef(t_type, ref_def, distance, reps=reps)
    pattern_def = sg.PatternDef([glide_def])

    return pattern_def


def PS3(mirror_offset, dx, reps: int = 0):
    """Mirror across bottom, then translate by ``(dx, 0)``.

    Args:
        mirror_offset: Offset of the mirror line from ``Reference.BOTTOM``.
        dx: Horizontal translation step after the mirror.
        reps: Number of translation repetitions.

    Returns:
        PatternDef combining mirror and translation.
    """
    t_type = sg.TransformationType.MIRROR
    target = sg.ReferenceTarget.KERNEL
    ref_def = sg.ReferenceDef(sg.Reference.BOTTOM, target, mirror_offset)
    mirror_def = sg.TransformDef(t_type, ref_def, reps=1)
    t_type2 = sg.TransformationType.TRANSLATE
    trans_def = sg.TransformDef(t_type2, ref=None, args=(dx, 0), reps=reps)
    pattern_def = sg.PatternDef([mirror_def, trans_def])

    return pattern_def


def PS4(dx, reps: int = 0):
    """Reflect over bottom (no offset), then translate by ``(dx, 0)``.

    Args:
        dx: Horizontal translation step.
        reps: Number of translation repetitions.

    Returns:
        PatternDef combining mirror and translation.
    """
    # reflect over bottom, then translate by dx
    mirror_def = TransformDef(
        TransformationType.MIRROR,
        ReferenceDef(sg.Reference.BOTTOM, ReferenceTarget.KERNEL, 0),
        reps=1,
    )
    trans_def = TransformDef(
        TransformationType.TRANSLATE, None, (dx, 0), reps=reps
    )
    return PatternDef([mirror_def, trans_def])


def PS5(mirror_offset, dx, reps: int = 0):
    """Reflect over right+offset, then translate by ``(dx, 0)``.

    Args:
        mirror_offset: Offset of the mirror from ``Reference.RIGHT``.
        dx: Horizontal translation step.
        reps: Number of translation repetitions.

    Returns:
        PatternDef combining mirror and translation.
    """
    # reflect over right+offset, then translate by pattern width+dx
    mirror_def = TransformDef(
        TransformationType.MIRROR,
        ReferenceDef(sg.Reference.RIGHT, ReferenceTarget.KERNEL, mirror_offset),
        reps=1,
    )
    trans_def = TransformDef(
        TransformationType.TRANSLATE, None, (dx, 0), reps=reps
    )
    return PatternDef([mirror_def, trans_def])


def PS6(mirror_offset, dx, reps: int = 0):
    """Reflect over right+offset, then translate by ``(dx, 0)``.

    Args:
        mirror_offset: Offset of the mirror from ``Reference.RIGHT``.
        dx: Horizontal translation step.
        reps: Number of translation repetitions.

    Returns:
        PatternDef combining mirror and translation.
    """
    # reflect over right, then translate by dx
    mirror_def = TransformDef(
        TransformationType.MIRROR,
        ReferenceDef(sg.Reference.RIGHT, ReferenceTarget.KERNEL, mirror_offset),
        reps=1,
    )
    trans_def = TransformDef(
        TransformationType.TRANSLATE, None, (dx, 0), reps=reps
    )
    return PatternDef([mirror_def, trans_def])


def PS7(about_offset, dx, reps: int = 0):
    """Rotate 180° about southeast+offset, then translate by ``(dx, 0)``.

    Args:
        about_offset: Offset of the rotation center from ``Reference.SOUTHEAST``.
        dx: Horizontal translation step.
        reps: Number of translation repetitions.

    Returns:
        PatternDef combining 180° rotation and translation.
    """
    # rotate 180° about southeast+(about_offset_x, 0), then translate by dx
    rotate_def = TransformDef(
        TransformationType.ROTATE,
        ReferenceDef(
            sg.Reference.SOUTHEAST, ReferenceTarget.KERNEL, about_offset
        ),
        sg.pi,
        reps=1,
    )
    trans_def = TransformDef(
        TransformationType.TRANSLATE, None, (dx, 0), reps=reps
    )
    return PatternDef([rotate_def, trans_def])


def PS8(about_offset_y, dx, reps: int = 0):
    """Rotate 180° about southeast+(0, offset), then translate by ``(dx, 0)``.

    Args:
        about_offset_y: Vertical offset of the rotation center from southeast.
        dx: Horizontal translation step.
        reps: Number of translation repetitions.

    Returns:
        PatternDef combining 180° rotation and translation.
    """
    # rotate 180° about southeast+(0, about_offset_y), then translate by dx
    rotate_def = TransformDef(
        TransformationType.ROTATE,
        ReferenceDef(
            sg.Reference.SOUTHEAST, ReferenceTarget.KERNEL, (0, about_offset_y)
        ),
        sg.pi,
        reps=1,
    )
    trans_def = TransformDef(
        TransformationType.TRANSLATE, None, (dx, 0), reps=reps
    )
    return PatternDef([rotate_def, trans_def])


def PS9(mirror_offset, distance, reps: int = 0):
    """Reflect over right+offset, then glide along the bottom.

    Args:
        mirror_offset: Offset of the mirror from ``Reference.RIGHT``.
        distance: Glide distance along the bottom axis.
        reps: Number of glide repetitions.

    Returns:
        PatternDef combining mirror and glide.
    """
    # reflect over right+offset, then glide along bottom
    mirror_def = TransformDef(
        TransformationType.MIRROR,
        ReferenceDef(sg.Reference.RIGHT, ReferenceTarget.KERNEL, mirror_offset),
        reps=1,
    )
    glide_def = TransformDef(
        TransformationType.GLIDE,
        ReferenceDef(sg.Reference.BOTTOM, ReferenceTarget.KERNEL, 0),
        distance,
        reps=reps,
    )
    return PatternDef([mirror_def, glide_def])


def PS10(about_offset, mirror_offset, dx, reps: int = 0):
    """Rotate 180° about southeast, reflect over right+offset, then translate.

    Args:
        about_offset: Offset of the rotation center from ``Reference.SOUTHEAST``.
        mirror_offset: Offset of the mirror from pattern ``Reference.RIGHT``.
        dx: Horizontal translation step.
        reps: Number of translation repetitions.

    Returns:
        PatternDef combining rotation, mirror, and translation.
    """
    # rotate 180° about southeast+(0, about_offset_y), reflect over right+offset, then translate
    rotate_def = TransformDef(
        TransformationType.ROTATE,
        ReferenceDef(
            sg.Reference.SOUTHEAST, ReferenceTarget.KERNEL, about_offset
        ),
        sg.pi,
        reps=1,
    )
    mirror_def = TransformDef(
        TransformationType.MIRROR,
        ReferenceDef(
            sg.Reference.RIGHT, ReferenceTarget.PATTERN, mirror_offset
        ),
        reps=1,
    )
    trans_def = TransformDef(
        TransformationType.TRANSLATE, None, (dx, 0), reps=reps
    )
    return PatternDef([rotate_def, mirror_def, trans_def])


def PS11(mirror_offset, distance, reps: int = 0):
    """Reflect over right+offset, then glide along the bottom.

    Args:
        mirror_offset: Offset of the mirror from ``Reference.RIGHT``.
        distance: Glide distance along the bottom axis.
        reps: Number of glide repetitions.

    Returns:
        PatternDef combining mirror and glide.
    """
    # reflect over right, then glide along bottom
    mirror_def = TransformDef(
        TransformationType.MIRROR,
        ReferenceDef(sg.Reference.RIGHT, ReferenceTarget.KERNEL, mirror_offset),
        reps=1,
    )
    glide_def = TransformDef(
        TransformationType.GLIDE,
        ReferenceDef(sg.Reference.BOTTOM, ReferenceTarget.KERNEL, 0),
        distance,
        reps=reps,
    )
    return PatternDef([mirror_def, glide_def])


def PS12(mirror_offset1, mirror_offset2, dx, reps: int = 0):
    """Reflect over right and bottom (with offsets), then translate.

    Args:
        mirror_offset1: Offset of the first mirror from kernel ``Reference.RIGHT``.
        mirror_offset2: Offset of the second mirror from pattern ``Reference.BOTTOM``.
        dx: Horizontal translation step.
        reps: Number of translation repetitions.

    Returns:
        PatternDef combining two mirrors and a translation.
    """
    # reflect over right+offset1, reflect over bottom+offset2, then translate
    mirror_def1 = TransformDef(
        TransformationType.MIRROR,
        ReferenceDef(
            sg.Reference.RIGHT, ReferenceTarget.KERNEL, mirror_offset1
        ),
        reps=1,
    )
    mirror_def2 = TransformDef(
        TransformationType.MIRROR,
        ReferenceDef(
            sg.Reference.BOTTOM, ReferenceTarget.PATTERN, mirror_offset2
        ),
        reps=1,
    )
    trans_def = TransformDef(
        TransformationType.TRANSLATE, None, (dx, 0), reps=reps
    )
    return PatternDef([mirror_def1, mirror_def2, trans_def])


def PS13(mirror_offset, dx, reps: int = 0):
    """Reflect over right (exact), then bottom+offset, then translate.

    Args:
        mirror_offset: Offset of the bottom mirror from pattern ``Reference.BOTTOM``.
        dx: Horizontal translation step.
        reps: Number of translation repetitions.

    Returns:
        PatternDef combining two mirrors and a translation.
    """
    # reflect over right (exact), reflect over bottom+offset, then translate
    mirror_def1 = TransformDef(
        TransformationType.MIRROR,
        ReferenceDef(sg.Reference.RIGHT, ReferenceTarget.KERNEL, 0),
        reps=1,
    )
    mirror_def2 = TransformDef(
        TransformationType.MIRROR,
        ReferenceDef(
            sg.Reference.BOTTOM, ReferenceTarget.PATTERN, mirror_offset
        ),
        reps=1,
    )
    trans_def = TransformDef(
        TransformationType.TRANSLATE, None, (dx, 0), reps=reps
    )
    return PatternDef([mirror_def1, mirror_def2, trans_def])


def PS14(mirror_offset, dx, reps: int = 0):
    """Reflect over bottom (exact), then right+offset, then translate.

    Args:
        mirror_offset: Offset of the right mirror from pattern ``Reference.RIGHT``.
        dx: Horizontal translation step.
        reps: Number of translation repetitions.

    Returns:
        PatternDef combining two mirrors and a translation.
    """
    # reflect over bottom (exact), reflect over right+offset, then translate
    mirror_def1 = TransformDef(
        TransformationType.MIRROR,
        ReferenceDef(sg.Reference.BOTTOM, ReferenceTarget.KERNEL, 0),
        reps=1,
    )
    mirror_def2 = TransformDef(
        TransformationType.MIRROR,
        ReferenceDef(
            sg.Reference.RIGHT, ReferenceTarget.PATTERN, mirror_offset
        ),
        reps=1,
    )
    trans_def = TransformDef(
        TransformationType.TRANSLATE, None, (dx, 0), reps=reps
    )
    return PatternDef([mirror_def1, mirror_def2, trans_def])


def PS15(dx, reps: int = 0):
    """Reflect over bottom and left (exact), then translate by ``(dx, 0)``.

    Args:
        dx: Horizontal translation step.
        reps: Number of translation repetitions.

    Returns:
        PatternDef combining two mirrors and a translation.
    """
    # reflect over bottom (exact), reflect over left (exact), then translate
    mirror_def1 = TransformDef(
        TransformationType.MIRROR,
        ReferenceDef(sg.Reference.BOTTOM, ReferenceTarget.KERNEL, 0),
        reps=1,
    )
    mirror_def2 = TransformDef(
        TransformationType.MIRROR,
        ReferenceDef(sg.Reference.LEFT, ReferenceTarget.PATTERN, 0),
        reps=1,
    )
    trans_def = TransformDef(
        TransformationType.TRANSLATE, None, (dx, 0), reps=reps
    )
    return PatternDef([mirror_def1, mirror_def2, trans_def])

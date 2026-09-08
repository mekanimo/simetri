"""Mask objects for SVG and TikZ backends.

Masks wrap a shape used for clipping or luminance/opacity masking.
Gradient and Stop types live in ``simetri.render.gradient`` and are
re-exported from ``simetri.graphics``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..base.all_enums import Axis, SvgUnits, Types
from ..config.settings import defaults
from ..geom.matrices import identity_matrix
from ..group.batch import Group
from ..shapes.shape import Shape
from .gradient import Stop, normalize_stops

if TYPE_CHECKING:
    from .canvas import Canvas
    from .sketch import Sketch


@dataclass
class Mask:
    """Mask payload used by clipping and masking APIs.

    A mask always has a shape plus optional opacity and gradient stop data.
    ``subtype`` may be ``Types.CLIP``, ``Types.LUMINANCE``, or ``Types.OPACITY``.

    Attributes:
        shape: Shape defining the mask geometry (must have ``type == Types.SHAPE``).
        opacity: Overall opacity in ``[0, 1]``. Defaults to 1.0.
        stops: Optional gradient stops for opacity/color masking.
        axis: Optional axis for linear gradient masks.
        subtype: Mask kind (clip / luminance / opacity).
        center: Radial gradient center. Defaults to ``(0, 0)``.
        focal: Radial gradient focal point. Defaults to ``(0, 0)``.

    Raises:
        TypeError: If ``shape`` is not a Shape.
        ValueError: If ``opacity`` is outside ``[0, 1]``.
    """

    shape: Shape
    opacity: float | None = None
    stops: list[Stop] = None
    axis: Axis | None = None
    subtype: Types = None
    center: tuple[float, float] = (0, 0)
    focal: tuple[float, float] = (0, 0)

    def __post_init__(self):
        if self.shape.type != Types.SHAPE:
            raise TypeError("mask.shape must be a Shape.")

        self.type = Types.MASK

        if self.opacity is None:
            self.opacity = 1.0
        self.opacity = float(self.opacity)
        if not (0.0 <= self.opacity <= 1.0):
            raise ValueError("mask opacity must be between 0 and 1.")


def normalize_axis(axis):
    """Return a usable mask axis.

    Args:
        axis: Explicit axis value or ``None``.

    Returns:
        tuple | object: Axis value for downstream renderers.
    """
    if axis is None:
        return defaults["mask_axis"]
    return axis


def _normalize_units(value: str | SvgUnits | None, field_name: str) -> SvgUnits:
    """Normalize a units string or enum to ``SvgUnits``.

    Args:
        value: Units value or ``None`` (uses defaults for known field names).
        field_name: One of ``mask_units``, ``mask_content_units``,
            ``gradient_units``.

    Returns:
        SvgUnits: Normalized units enum.

    Raises:
        ValueError: If the field name or value is unsupported.
    """
    if value is None:
        if field_name == "mask_units":
            default_value = defaults["mask_units"]
        elif field_name == "mask_content_units":
            default_value = defaults["mask_content_units"]
        elif field_name == "gradient_units":
            default_value = defaults["gradient_units"]
        else:
            raise ValueError(f"unsupported units field: {field_name}")
        value = default_value

    if isinstance(value, SvgUnits):
        return value

    text = str(value).strip()
    lowered = text.lower()
    aliases = {
        "userspaceonuse": SvgUnits.USER_SPACE_ON_USE,
        "usersapceonuse": SvgUnits.USER_SPACE_ON_USE,
        "objectboundingbox": SvgUnits.OBJECT_BOUNDING_BOX,
    }
    normalized = aliases.get(lowered, None)
    if normalized is None:
        if text == SvgUnits.USER_SPACE_ON_USE.value:
            normalized = SvgUnits.USER_SPACE_ON_USE
        elif text == SvgUnits.OBJECT_BOUNDING_BOX.value:
            normalized = SvgUnits.OBJECT_BOUNDING_BOX
        else:
            normalized = None

    if normalized is None:
        raise ValueError(
            f"{field_name} must be 'userSpaceOnUse' or 'objectBoundingBox'."
        )
    return normalized


# This is no longer used! Will be deleted soon.
# We will use canvas.clip(target, mask), canvas.mask(target, mask)
def clip_mask_(
    self: Canvas,
    target: Shape | Group | None = None,
    mask: Mask = None,
    **kwargs,
):
    """Apply a ``Mask`` to a target and draw it (legacy API).

    Note:
        Prefer ``canvas.clip`` / ``canvas.mask`` / ``canvas.apply_mask``.
        This helper is retained temporarily for compatibility.

    Args:
        self: Canvas instance (bound method style).
        target: Shape or Group to mask, or ``None`` to open a mask scope.
        mask: Mask instance or Shape used as the mask.
        **kwargs: Legacy private kwargs such as ``_mask_opacity``.

    Returns:
        Canvas or result of ``clip`` / ``apply_mask``.

    Raises:
        TypeError: If ``mask`` or ``target`` has an unsupported type.
        ValueError: If opacity or unsupported units are invalid.
    """
    mask_opacity = defaults.get("alpha", 1.0)
    mask_stops = None
    mask_axis = normalize_axis(None)
    mask_units = _normalize_units(
        defaults.get("mask_units", SvgUnits.USER_SPACE_ON_USE.value),
        "mask_units",
    )
    mask_content_units = _normalize_units(
        defaults.get("mask_content_units", SvgUnits.USER_SPACE_ON_USE.value),
        "mask_content_units",
    )
    if isinstance(mask, Mask):
        mask_shape = mask.shape
        mask_opacity = mask.opacity
        mask_stops = mask.stops
        mask_axis = mask.axis
        mask_units = mask.mask_units
        mask_content_units = mask.mask_content_units
    elif isinstance(mask, Shape):
        mask_shape = mask
        if "_mask_opacity" in kwargs:
            mask_opacity = kwargs["_mask_opacity"]
        if "_mask_stops" in kwargs:
            mask_stops = kwargs["_mask_stops"]
        if "_mask_axis" in kwargs:
            mask_axis = kwargs["_mask_axis"]
        if "_mask_units" in kwargs:
            mask_units = _normalize_units(kwargs["_mask_units"], "mask_units")
        if "_mask_content_units" in kwargs:
            mask_content_units = _normalize_units(
                kwargs["_mask_content_units"], "mask_content_units"
            )
    else:
        raise TypeError("mask must be a Mask instance or a Shape.")

    # Apply the canvas xform_matrix to a copy of the mask shape
    xform = self.xform_matrix  # property returns a copy
    if not np.allclose(xform, identity_matrix()):
        mask_shape = mask_shape.copy()
        mask_shape.transform(xform)

    if mask_opacity is None:
        mask_opacity = defaults.get("alpha", 1.0)
    if not (0.0 <= mask_opacity <= 1.0):
        raise ValueError("mask opacity must be between 0 and 1.")
    if mask_stops is not None:
        mask_stops = normalize_stops(mask_stops)
        mask_axis = normalize_axis(mask_axis)

    use_gradient_opacity = mask_stops is not None

    if target is None:
        scope_sketch = MaskSketch(
            mask=mask_shape,
            clip=True,
            mask_opacity=mask_opacity,
            mask_stops=mask_stops,
            mask_axis=mask_axis,
            mask_units=mask_units,
            mask_content_units=mask_content_units,
        )
        self.active_page.sketches.append(scope_sketch)
        if mask_shape is not None:
            self._all_vertices.extend(mask_shape.b_box.corners)
        return self

    if not isinstance(target, (Shape, Group)):
        raise TypeError("target must be a Shape, Group, or None.")

    if mask_opacity >= 1.0 and not use_gradient_opacity:
        return self.clip(target, mask_shape, **kwargs)

    if mask_units != _normalize_units(None, "mask_units"):
        raise ValueError("clip_mask_ does not support mask_units.")
    if mask_content_units != _normalize_units(None, "mask_content_units"):
        raise ValueError("clip_mask_ does not support mask_content_units.")

    mask_data = Mask(
        shape=mask_shape,
        opacity=mask_opacity,
        stops=mask_stops,
        axis=mask_axis,
    )
    return self.apply_mask(target, mask_data)

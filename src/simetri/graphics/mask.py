"""Gradient and Mask objects for both SVG and TikZ"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from ..graphics.common import common_properties
from ..graphics.all_enums import Types, SvgUnits, GradientType

from ..colors.colors import Color
from ..settings.settings import defaults
from ..helpers.validation import check_percent, check_color

if TYPE_CHECKING:
    from ..graphics.shape import Shape


@dataclass
class Mask:
    """Mask object used by clipping/masking APIs.

    A mask always has a shape payload plus optional opacity/gradient data.
    mask.subtype can be Types.CLIP, Types.LUMINANCE, Types.OPACITY
    """

    shape: Shape
    opacity: float = None
    stops: list[Stop] = None
    axis: Optional["Axis"] = None
    subtype: Types = None

    def __post_init__(self):
        if not isinstance(self.shape, Shape):
            raise TypeError("mask.shape must be a Shape.")

        common_properties(self, graphics_object=False)
        self.type = Types.MASK

        if self.opacity is None:
            self.opacity = defaults.get("alpha", 1.0)
        self.opacity = float(self.opacity)
        if not (0.0 <= self.opacity <= 1.0):
            raise ValueError("mask opacity must be between 0 and 1.")



@dataclass
class Axis:
    """Mask direction axis represented by normalized start/end points.
    start end end must be between [0, 0], and [1.0, 1.0]
    """

    start: tuple[float, float]
    end: tuple[float, float]

    def __init__(self, start: tuple[float, float], end: tuple[float, float]):
        self,
        start: tuple[float, float] = start,
        end: tuple[float, float] = end


    def __post_init__(self):
        if not (check_percent(self.start[0]) and
                check_percent(self.start[1]) and
                check_percent(self.end[0]) and
                check_percent(self.end[0])
        ):
            raise ValueError("start and end points must be between (0, 0) and (1.0, 1.0) ")
        common_properties(self, graphics_object=False)



@dataclass(init=False)
class Stop:
    """A gradient stop used by mask gradients.
    offset: 0 <= offset <= 1.0
    opacity: 0 <= opacity <= 1.0
    color: Color
    """

    offset: float
    color: Optional[Color] = None
    opacity: Optional[float] = None

    def __init__(
        self,
        offset: float,
        color: Optional[Color] = None,
        opacity: Optional[float] = None,
    ):
        if not check_percent(offset):
            raise ValueError("Stop offset must be between 0 and 1.0")
        if color is None and opacity is None:
            raise ValueError("Specify a color, opacity, or both.")
        if color is not None:
            if not check_color(color):
                raise ValueError("Incorrect color value.")
        if opacity is not None:
            if not check_percent(opacity):
                raise ValueError("Stop opacity must be between 0 and 1.0")
        self.offset = offset
        self.color = color
        self.opacity = opacity
        self.__post_init__()

    def __post_init__(self):
        common_properties(self, graphics_object=False)
        self.type = Types.STOP
        self.subtype = Types.STOP

def _resolve_stops(stops):
    if len(stops) < 2:
        raise ValueError("Invalid stop values.")
    if isinstance(stops[0], Stop):
        for stop in stops[1:]:
            if not isinstance(stop, Stop):
                raise ValueError("All stops must have the same type.")
        return stops
    else:
        stops_list = []
        for stop in stops:
            offset = stop[0]
            if not check_percent(offset):
                raise ValueError("Offset must be between 0 and 1")
            color = None
            opacity = None
            if isinstance(stop[1], float):
                opacity = stop[1]
                if not check_percent(opacity):
                    raise ValueError("Offset must be between 0 and 1")
            elif isinstance(stop[1], Color):
                color = stop[1]
            if len(stop) > 2:
                color = stop[2]
                if not check_color(color):
                    raise ValueError("Invalid color.")
            stops_list.append(Stop(offset, color, opacity))

        return stops_list


@dataclass
class Gradient:
    """Gradient object for shape fills in SVG/TikZ backends.
    stops can be a list of Stop objects or
    [(offset1, opacity1, color1), (offset2, opacity2, color2), ...] or
    [(offset1, color1), (offset2, color2), ...] or
    [(offset1, opacity1), (offset2, opacity2), ...]

    """
    gradient_type: GradientType = None
    stops: Union[list[Stop], list[tuple]] = None
    axis: Optional[Axis] = None
    center: Optional[tuple[float, float]] = None
    focal: Optional[tuple[float, float]] = None
    radius: Optional[float] = None
    units: SvgUnits = None
    spread_method: str = None
    transform: Optional[str] = None
    subtype: Types = None

    def __post_init__(self):
        common_properties(self, graphics_object=False)
        self.type = Types.GRADIENT

        if self.spread_method is None:
            self.spread_method = defaults["gradient_spread_method"]
        self.stops = _resolve_stops(self.stops)
        if self.gradient_type == GradientType.LINEAR:
            self.center = None
            self.focal = None
            self.radius = None
            self.subtype = Types.LINEAR
        else:
            self.axis = None
            if self.center is None:
                self.center = defaults["gradient_center"]
            if self.focal is None:
                self.focal = defaults["gradient_focal"]
            if self.radius is None:
                self.radius = defaults["gradient_radius"]
            if self.radius <= 0.0:
                raise ValueError("gradient radius must be positive.")
            self.subtype = Types.RADIAL



def _normalize_units(value: Optional[Union[str, SvgUnits]], field_name: str) -> SvgUnits:
    if value is None:
        if field_name == "mask_units":
            default_value = defaults["mask_units"]
        elif field_name == "mask_content_units":
            default_value = defaults["mask_content_units"]
        elif field_name == "gr_units":
            default_value = defaults["gr_units"]
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
def clip_mask_(self: "Canvas", target: Union[Shape, Batch, None]=None, mask: Mask=None, **kwargs):
    """Apply a `Mask` to a target and draw it.
    """
    mask_opacity = defaults.get("alpha", 1.0)
    mask_stops = None
    mask_axis = normalize_axis(None)
    mask_units = _normalize_units(defaults.get("mask_units", SvgUnits.USER_SPACE_ON_USE.value), "mask_units")
    mask_content_units = _normalize_units(defaults.get("mask_content_units", SvgUnits.USER_SPACE_ON_USE.value), "mask_content_units")
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

    def _next_mask_context_id() -> str:
        current = getattr(self, "_mask_context_counter", 0) + 1
        self._mask_context_counter = current
        return f"mask_target_{current}"

    def _same_vertices(sketch, shape) -> bool:
        sketch_vertices = getattr(sketch, "vertices", None)
        shape_vertices = getattr(shape, "vertices", None)
        if not sketch_vertices or not shape_vertices:
            return False
        if len(sketch_vertices) != len(shape_vertices):
            return False
        for sk_v, sh_v in zip(sketch_vertices, shape_vertices):
            if len(sk_v) < 2 or len(sh_v) < 2:
                return False
            if abs(float(sk_v[0]) - float(sh_v[0])) > 1e-9:
                return False
            if abs(float(sk_v[1]) - float(sh_v[1])) > 1e-9:
                return False
        return True

    def _apply_mask_to_existing_target() -> bool:
        if not isinstance(target, Shape):
            return False

        mask_context_id = None
        for sketch in reversed(self.active_page.sketches):
            if not _same_vertices(sketch, target):
                continue

            sketch.mask = mask_shape
            for key, value in kwargs.items():
                setattr(sketch, key, value)
            if mask_opacity >= 1.0 and not use_gradient_opacity:
                sketch.clip = True
            else:
                sketch.clip = False
                mask_context_id = _next_mask_context_id()
                sketch._mask_context_id = mask_context_id
                sketch._mask_opacity = mask_opacity
                sketch._mask_stops = mask_stops
                sketch._mask_axis = mask_axis
                sketch._mask_units = mask_units
                sketch._mask_content_units = mask_content_units

            if mask_shape is not None:
                self._all_vertices.extend(mask_shape.b_box.corners)
            return True

        return False

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

    if not isinstance(target, (Shape, Batch)):
        raise TypeError("target must be a Shape, Batch, or None.")

    if _apply_mask_to_existing_target():
        return self

    draw_kwargs = {"mask": mask_shape}
    draw_kwargs.update(kwargs)
    if mask_opacity >= 1.0 and not use_gradient_opacity:
        draw_kwargs["clip"] = True
    else:
        draw_kwargs["clip"] = False
        draw_kwargs["_mask_context_id"] = _next_mask_context_id()
        draw_kwargs["_mask_opacity"] = mask_opacity
        draw_kwargs["_mask_stops"] = mask_stops
        draw_kwargs["_mask_axis"] = mask_axis
        draw_kwargs["_mask_units"] = mask_units
        draw_kwargs["_mask_content_units"] = mask_content_units

    vertices_len = len(self._all_vertices)
    canvas_draw.draw(self, target, **draw_kwargs)
    del self._all_vertices[vertices_len:]
    self._all_vertices.extend(mask_shape.b_box.corners)
    return self

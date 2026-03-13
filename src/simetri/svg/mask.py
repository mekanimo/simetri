"""SVG mask API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from ..graphics.common import common_properties
from ..graphics.all_enums import Types, SvgUnits, GradientType
from ..graphics.affine import identity_matrix
from ..graphics.batch import Batch
from ..graphics.shape import Shape
from ..graphics.sketch import MaskSketch
from ..canvas import draw as canvas_draw
from ..colors.colors import Color
from ..settings.settings import defaults

if TYPE_CHECKING:
    from ..canvas.canvas import Canvas


@dataclass
class Mask:
    """Mask object used by clipping/masking APIs.

    A mask always has a shape payload plus optional opacity/gradient data.
    """

    shape: Shape
    opacity: float = None
    stops: list[Stop] = None
    axis: Optional["Axis"] = None
    mask_units: SvgUnits = None
    mask_content_units: SvgUnits = None
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

        self.axis = normalize_axis(self.axis)
        self.stops = normalize_stops(self.stops)
        self.mask_units = _normalize_units(self.mask_units, "mask_units")
        self.mask_content_units = _normalize_units(self.mask_content_units, "mask_content_units")

        self.subtype = _normalize_mask_subtype(self.subtype, self.stops, self.opacity)


@dataclass
class Axis:
    """Mask direction axis represented by normalized start/end points."""

    start: tuple[float, float]
    end: tuple[float, float]

    def __post_init__(self):
        common_properties(self, graphics_object=False)
        self.start = _normalize_axis_point(self.start, "start")
        self.end = _normalize_axis_point(self.end, "end")


@dataclass(init=False)
class Stop:
    """A gradient stop used by mask gradients."""

    offset: float
    color: Optional[Union[Color, str]] = None
    opacity: Optional[float] = None
    style: Optional[dict[str, Any]] = None

    def __init__(
        self,
        offset: float,
        color: Optional[Union[Color, str]] = None,
        opacity: Optional[float] = None,
        style: Optional[dict[str, Any]] = None,
    ):
        self.offset = offset
        self.color = color
        self.opacity = opacity
        self.style = style
        self.__post_init__()

    def __post_init__(self):
        common_properties(self, graphics_object=False)
        self.type = Types.STOP
        self.subtype = Types.STOP

        self.offset = _normalize_offset(self.offset)

        if self.opacity is not None:
            self.opacity = float(self.opacity)
            if not (0.0 <= self.opacity <= 1.0):
                raise ValueError("mask stop opacity must be between 0 and 1.")

        self.style = dict(self.style or {})


@dataclass
class Gradient:
    """Gradient object for shape fills in SVG/TikZ backends."""

    gradient_type: GradientType = None
    stops: list[Stop] = None
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

        self.gradient_type = _normalize_gradient_type(self.gradient_type)
        self.stops = normalize_stops(self.stops)
        self.units = _normalize_units(self.units, "gr_units")

        if self.spread_method is None:
            self.spread_method = defaults["gradient_spread_method"]

        if self.gradient_type == GradientType.LINEAR:
            self.axis = normalize_axis(self.axis)
            self.center = None
            self.focal = None
            self.radius = None
            self.subtype = Types.LINEAR
        else:
            self.axis = None
            self.center = _normalize_axis_point(
                self.center if self.center is not None else (defaults["gr_cx"], defaults["gr_cy"]),
                "center",
            )
            default_focal_x = defaults["gr_fx"] if defaults["gr_fx"] is not None else self.center[0]
            default_focal_y = defaults["gr_fy"] if defaults["gr_fy"] is not None else self.center[1]
            self.focal = _normalize_axis_point(
                self.focal if self.focal is not None else (default_focal_x, default_focal_y),
                "focal",
            )
            self.radius = float(self.radius if self.radius is not None else defaults["gr_r"])
            if self.radius <= 0.0:
                raise ValueError("gradient radius must be positive.")
            self.subtype = Types.RADIAL


def _normalize_gradient_type(value: Optional[Union[GradientType, Types]]) -> GradientType:
    if value is None:
        value = defaults["gradient_type"]

    if isinstance(value, GradientType):
        return value

    if value == Types.LINEAR:
        return GradientType.LINEAR
    if value == Types.RADIAL:
        return GradientType.RADIAL

    value_text = str(value).strip()
    if value_text in GradientType.__members__:
        return GradientType[value_text]
    lowered_value = value_text.lower()
    if lowered_value == GradientType.LINEAR.value:
        return GradientType.LINEAR
    if lowered_value == GradientType.RADIAL.value:
        return GradientType.RADIAL

    raise ValueError("gradient_type must be GradientType.LINEAR or GradientType.RADIAL.")


def _normalize_offset(offset: Union[float, str]) -> float:
    if isinstance(offset, str) and offset.endswith("%"):
        offset = float(offset[:-1]) / 100.0
    value = float(offset)
    if not (0.0 <= value <= 1.0):
        raise ValueError("mask stop offset must be between 0 and 1.")
    return value


def _normalize_axis_point(point: Union[tuple[float, float], list[float]], name: str) -> tuple[float, float]:
    if not isinstance(point, (tuple, list)) or len(point) != 2:
        raise TypeError(f"mask axis {name} must be a pair (x, y).")
    x = float(point[0])
    y = float(point[1])
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"mask axis {name}.x must be between 0 and 1.")
    if not (0.0 <= y <= 1.0):
        raise ValueError(f"mask axis {name}.y must be between 0 and 1.")
    return (x, y)


def normalize_axis(axis: Optional[Union[Axis, dict, tuple, list]]) -> Axis:
    if axis is None:
        return Axis(
            start=(defaults.get("msk_x1", 0.0), defaults.get("msk_y1", 0.0)),
            end=(defaults.get("msk_x2", 1.0), defaults.get("msk_y2", 0.0)),
        )
    if isinstance(axis, Axis):
        return axis
    if isinstance(axis, dict):
        if "start" not in axis or "end" not in axis:
            raise ValueError("mask axis dict must include 'start' and 'end'.")
        return Axis(start=axis["start"], end=axis["end"])
    if isinstance(axis, (tuple, list)) and len(axis) == 2:
        return Axis(start=axis[0], end=axis[1])
    raise TypeError("mask axis must be an Axis, {'start','end'} dict, or ((x1,y1),(x2,y2)) pair.")


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


def stop_from_input(stop: Union[Stop, dict, tuple, list]) -> Stop:
    if isinstance(stop, Stop):
        return stop

    if isinstance(stop, dict):
        if "offset" not in stop:
            raise ValueError("each mask stop dict must include 'offset'.")

        allowed_keys = {"offset", "color", "opacity", "style"}
        unknown_keys = set(stop.keys()) - allowed_keys
        if unknown_keys:
            unknown_keys_text = ", ".join(sorted(unknown_keys))
            raise ValueError(f"unsupported mask stop dict keys: {unknown_keys_text}")

        stop_offset = stop["offset"]
        if "color" in stop:
            color = stop["color"]
        else:
            color = None
        if "opacity" in stop:
            opacity = stop["opacity"]
        else:
            opacity = None
        if "style" in stop:
            style = stop["style"]
        else:
            style = None

        return Stop(stop_offset, color=color, opacity=opacity, style=style)

    raise ValueError("each mask stop must be a Stop or a dict with keys: offset, color, opacity, style.")


def normalize_stops(stops: Optional[list]) -> Optional[list[Stop]]:
    if stops is None:
        return None
    if not isinstance(stops, (list, tuple)):
        raise ValueError("mask stops must be a list with at least two stop entries.")
    normalized = [stop_from_input(stop) for stop in stops]
    if len(normalized) < 2:
        raise ValueError("mask stops must contain at least two stops.")
    return normalized


def _normalize_mask_subtype(subtype, stops: Optional[list[Stop]], opacity: float) -> Types:
    allowed = {Types.CLIP, Types.LUMINANCE, Types.OPACITY}

    if subtype is not None:
        if not isinstance(subtype, Types):
            subtype_str = str(subtype).strip()
            if subtype_str.startswith("Types."):
                subtype_str = subtype_str.split(".", 1)[1]
            subtype_str = subtype_str.upper()
            if subtype_str in Types.__members__:
                subtype = Types[subtype_str]
            else:
                subtype = Types(subtype_str)
        if subtype not in allowed:
            raise ValueError("mask subtype must be Types.CLIP, Types.LUMINANCE, or Types.OPACITY.")
        return subtype

    if stops:
        has_color = any(stop.color is not None for stop in stops)
        return Types.LUMINANCE if has_color else Types.OPACITY

    if opacity < 1.0:
        return Types.OPACITY

    return Types.CLIP

def clip_mask(self: "Canvas", target: Union[Shape, Batch, None]=None, mask: Mask=None, **kwargs):
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

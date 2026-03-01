"""SVG mask API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from ..graphics.batch import Batch
from ..graphics.shape import Shape

if TYPE_CHECKING:
    from ..canvas.canvas import Canvas


@dataclass
class Mask:
    """Mask object used by clipping/masking APIs.

    First step implementation stores a single shape payload.
    """

    shape: Shape
    opacity: float = 1.0
    stops: list = None
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 1.0
    y2: float = 0.0


def clip_mask(self: "Canvas", target: Union[Shape, Batch, None]=None, mask: Mask=None):
    """Apply a `Mask` to a target and draw it.
    """
    mask_opacity = 1.0
    mask_stops = None
    mask_x1 = 0.0
    mask_y1 = 0.0
    mask_x2 = 1.0
    mask_y2 = 0.0
    if isinstance(mask, Mask):
        mask_shape = mask.shape
        mask_opacity = mask.opacity
        mask_stops = mask.stops
        mask_x1 = mask.x1
        mask_y1 = mask.y1
        mask_x2 = mask.x2
        mask_y2 = mask.y2
    elif isinstance(mask, Shape):
        mask_shape = mask
    else:
        raise TypeError("mask must be a Mask instance or a Shape.")

    if mask_opacity is None:
        mask_opacity = 1.0
    if not (0.0 <= mask_opacity <= 1.0):
        raise ValueError("mask opacity must be between 0 and 1.")
    if mask_stops is not None:
        if not isinstance(mask_stops, (list, tuple)) or len(mask_stops) < 2:
            raise ValueError("mask stops must be a list with at least two SVG stop entries.")
        for stop in mask_stops:
            if isinstance(stop, dict):
                if "offset" not in stop:
                    raise ValueError("each mask stop dict must include 'offset'.")
                stop_opacity = stop.get("stop-opacity", stop.get("opacity", None))
            elif isinstance(stop, (list, tuple)):
                if len(stop) < 2:
                    raise ValueError("each mask stop tuple must be (offset, opacity), (offset, stop-color), or (offset, stop-color, stop-opacity).")
                second = stop[1]
                if len(stop) >= 3:
                    stop_opacity = stop[2]
                elif isinstance(second, (int, float)):
                    stop_opacity = second
                else:
                    stop_opacity = None
            else:
                raise ValueError("each mask stop must be a tuple/list or dict.")

            if stop_opacity is not None and not (0.0 <= float(stop_opacity) <= 1.0):
                raise ValueError("mask stop opacity must be between 0 and 1.")
        for coord_name, coord in (("x1", mask_x1), ("y1", mask_y1), ("x2", mask_x2), ("y2", mask_y2)):
            if not isinstance(coord, (int, float)):
                raise TypeError(f"mask {coord_name} must be a float between 0 and 1.")
            if not (0.0 <= float(coord) <= 1.0):
                raise ValueError(f"mask {coord_name} must be between 0 and 1.")

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
            if mask_opacity >= 1.0 and not use_gradient_opacity:
                sketch.clip = True
            else:
                sketch.clip = False
                mask_context_id = _next_mask_context_id()
                sketch._mask_context_id = mask_context_id
                sketch._mask_opacity = mask_opacity
                sketch._mask_stops = mask_stops
                sketch._mask_x1 = mask_x1
                sketch._mask_y1 = mask_y1
                sketch._mask_x2 = mask_x2
                sketch._mask_y2 = mask_y2

            if mask_shape is not None:
                self._all_vertices.extend(mask_shape.b_box.corners)
            return True

        return False

    if target is None:
        self.clip = True
        self._mask = mask_shape
        self._mask_opacity = mask_opacity
        self._mask_stops = mask_stops
        self._mask_x1 = mask_x1
        self._mask_y1 = mask_y1
        self._mask_x2 = mask_x2
        self._mask_y2 = mask_y2
        return self

    if not isinstance(target, (Shape, Batch)):
        raise TypeError("target must be a Shape, Batch, or None.")

    if _apply_mask_to_existing_target():
        return self

    draw_kwargs = {"mask": mask_shape}
    if mask_opacity >= 1.0 and not use_gradient_opacity:
        draw_kwargs["clip"] = True
    else:
        draw_kwargs["clip"] = False
        draw_kwargs["_mask_context_id"] = _next_mask_context_id()
        draw_kwargs["_mask_opacity"] = mask_opacity
        draw_kwargs["_mask_stops"] = mask_stops
        draw_kwargs["_mask_x1"] = mask_x1
        draw_kwargs["_mask_y1"] = mask_y1
        draw_kwargs["_mask_x2"] = mask_x2
        draw_kwargs["_mask_y2"] = mask_y2

    vertices_len = len(self._all_vertices)
    self.draw(target, **draw_kwargs)
    del self._all_vertices[vertices_len:]
    self._all_vertices.extend(mask_shape.b_box.corners)
    return self

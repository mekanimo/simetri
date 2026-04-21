"""TikZ clipping and masking helpers.

This module implements mask application helpers for TeX rendering
without changing existing TikZ renderer internals.
"""

from __future__ import annotations

from math import atan2, degrees
from types import SimpleNamespace
from typing import TYPE_CHECKING, Union

from ..colors.colors import Color
from ..graphics.all_enums import TexLoc
from ..graphics.batch import Batch
from ..graphics.shape import Shape
from ..graphics.sketch import MaskSketch
from .tikz_sketch import TexSketch
from ..canvas import draw as canvas_draw
from ..tikz.tikz import get_clip_code
from ..svg.mask import Mask, Stop

if TYPE_CHECKING:
	from ..canvas.canvas import Canvas


def _normalize_mask_inputs(mask, **kwargs):
	mask_opacity = 1.0
	mask_stops = None
	mask_axis = ((0.0, 0.0), (1.0, 0.0))

	if isinstance(mask, Mask) or (
		hasattr(mask, "shape") and hasattr(mask, "opacity") and hasattr(mask, "stops")
	):
		mask_shape = mask.shape
		mask_opacity = mask.opacity
		mask_stops = mask.stops
		mask_axis = (mask.axis.start, mask.axis.end)
	elif isinstance(mask, Shape):
		mask_shape = mask
		if "_mask_opacity" in kwargs:
			mask_opacity = kwargs["_mask_opacity"]
		if "_mask_stops" in kwargs:
			mask_stops = kwargs["_mask_stops"]
		if "_mask_axis" in kwargs:
			mask_axis = kwargs["_mask_axis"]
	else:
		raise TypeError("mask must be a Mask instance or a Shape.")

	if mask_opacity is None:
		mask_opacity = 1.0
	if not (0.0 <= float(mask_opacity) <= 1.0):
		raise ValueError("mask opacity must be between 0 and 1.")

	if mask_stops is not None:
		if not isinstance(mask_stops, (list, tuple)) or len(mask_stops) < 2:
			raise ValueError("mask stops must be a list with at least two SVG stop entries.")
		for coord_name, coord in (("x1", mask_axis[0][0]), ("y1", mask_axis[0][1]), ("x2", mask_axis[1][0]), ("y2", mask_axis[1][1])):
			if not isinstance(coord, (int, float)):
				raise TypeError(f"mask {coord_name} must be a float between 0 and 1.")
			if not (0.0 <= float(coord) <= 1.0):
				raise ValueError(f"mask {coord_name} must be between 0 and 1.")

	return mask_shape, float(mask_opacity), mask_stops, mask_axis


def _parse_offset(offset):
	if isinstance(offset, (int, float)):
		return float(offset)
	if isinstance(offset, str) and offset.endswith("%"):
		return float(offset[:-1]) / 100.0
	return float(offset)


def _luminance_from_color(stop_color):
	if isinstance(stop_color, Color):
		r, g, b = stop_color.rgb255
	elif isinstance(stop_color, str):
		c = stop_color.strip().lower()
		if c == "white":
			return 1.0
		if c == "black":
			return 0.0
		if c.startswith("#") and len(c) == 7:
			r = int(c[1:3], 16)
			g = int(c[3:5], 16)
			b = int(c[5:7], 16)
		else:
			return 1.0
	else:
		return 1.0
	return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def _effective_alpha_from_stop(stop):
	if isinstance(stop, Stop) or (hasattr(stop, "offset") and hasattr(stop, "color")):
		offset = _parse_offset(stop.offset)
		stop_color = stop.color if stop.color is not None else "white"
		stop_opacity = stop.opacity if stop.opacity is not None else 1.0
	elif isinstance(stop, dict):
		offset = _parse_offset(stop["offset"])
		stop_color = stop.get("stop-color", stop.get("stop_color", "white"))
		stop_opacity = stop.get("stop-opacity", stop.get("stop_opacity", stop.get("opacity", 1.0)))
	else:
		offset = _parse_offset(stop[0])
		second = stop[1]
		if len(stop) >= 3:
			stop_color = second
			stop_opacity = stop[2]
		elif isinstance(second, (int, float)):
			stop_color = "white"
			stop_opacity = second
		else:
			stop_color = second
			stop_opacity = 1.0

	lum = _luminance_from_color(stop_color)
	alpha = max(0.0, min(1.0, float(stop_opacity) * lum))
	return offset, alpha


def _pgf_gray(transparency: int) -> str:
	"""Convert transparency % (0=opaque, 100=fully transparent) to a valid pgf color.

	In TikZ fadings, white=fully opaque, black=fully transparent.
	'transparent!N' is NOT a valid pgf color — use black!N instead.
	"""
	if transparency <= 0:
		return "white"
	if transparency >= 100:
		return "black"
	return f"black!{transparency}"


def _build_fading_code(fade_id, stops, x1, y1, x2, y2):
	parsed = [_effective_alpha_from_stop(stop) for stop in stops]
	parsed.sort(key=lambda x: x[0])
	if not parsed:
		parsed = [(0.0, 1.0), (1.0, 1.0)]

	shade_id = f"{fade_id}Shade"
	color_stops = []
	for offset, alpha in parsed:
		offset = max(0.0, min(1.0, float(offset)))
		pos = int(round(offset * 100))
		transparency = int(round((1.0 - float(alpha)) * 100))
		color_stops.append(f"color({pos}bp)=({_pgf_gray(transparency)})")

	if parsed[0][0] > 0.0:
		first_t = int(round((1.0 - float(parsed[0][1])) * 100))
		color_stops.insert(0, f"color(0bp)=({_pgf_gray(first_t)})")
	if parsed[-1][0] < 1.0:
		last_t = int(round((1.0 - float(parsed[-1][1])) * 100))
		color_stops.append(f"color(100bp)=({_pgf_gray(last_t)})")

	shading_decl = "; ".join(color_stops)

	angle = degrees(atan2(y2 - y1, x2 - x1))
	return (
		f"\\pgfdeclarehorizontalshading{{{shade_id}}}{{100bp}}{{{shading_decl}}}\n"
		f"\\tikzfadingfrompicture[name={fade_id}]\n"
		f"  \\shade[shading={shade_id}, shading angle={angle:.2f}] (0, 0) rectangle (100bp, 100bp);\n"
		f"\\endtikzfadingfrompicture\n"
	)


def _get_clip_from_mask(mask_shape):
	proxy = SimpleNamespace(mask=mask_shape)
	return get_clip_code(proxy)


def _get_scope_fading_path(mask_shape, fade_id):
	bbox = mask_shape.b_box
	x1, y1 = bbox.southwest
	x2, y2 = bbox.northeast
	return f"\\path [scope fading={fade_id}] ({x1}, {y1}) rectangle ({x2}, {y2});\n"


def clip_mask(self: "Canvas", target: Union[Shape, Batch, None] = None, mask: Mask = None, **kwargs):
	"""Apply a mask for TeX rendering using additive scope/TexSketch logic."""
	mask_shape, mask_opacity, mask_stops, mask_axis = _normalize_mask_inputs(mask, **kwargs)
	mask_x1, mask_y1 = mask_axis[0]
	mask_x2, mask_y2 = mask_axis[1]

	use_gradient = mask_stops is not None

	if target is None:
		scope_sketch = MaskSketch(
			mask=mask_shape,
			clip=True,
			mask_opacity=mask_opacity,
			mask_stops=mask_stops,
				mask_axis=mask_axis,
		)
		self.active_page.sketches.append(scope_sketch)
		if mask_shape is not None:
			self._all_vertices.extend(mask_shape.b_box.corners)

		if use_gradient:
			fade_id = f"simetriCanvasMaskFade{len(self.active_page.sketches)}"
			fade_code = _build_fading_code(fade_id, mask_stops, mask_x1, mask_y1, mask_x2, mask_y2)
			fade_sketch = TexSketch(fade_code)
			fade_sketch.library = "fadings"
			fade_sketch.location = TexLoc.PREAMBLE
			self.active_page.sketches.append(fade_sketch)
			scope_sketch._mask_fade_id = fade_id
		return self

	if not isinstance(target, (Shape, Batch)):
		raise TypeError("target must be a Shape, Batch, or None.")

	clip_code = _get_clip_from_mask(mask_shape)
	if not clip_code:
		raise ValueError("TikZ masking currently requires a clip-compatible mask shape.")

	if use_gradient:
		fade_id = f"simetriMaskFade{id(target)}{len(self.active_page.sketches)}"
		fade_code = _build_fading_code(fade_id, mask_stops, mask_x1, mask_y1, mask_x2, mask_y2)
		fade_sketch = TexSketch(fade_code)
		fade_sketch.library = "fadings"
		fade_sketch.location = TexLoc.PREAMBLE
		self.active_page.sketches.append(fade_sketch)
		fading_path = _get_scope_fading_path(mask_shape, fade_id)
		start_scope = f"\\begin{{scope}}\n{fading_path}{clip_code}"
	elif mask_opacity < 1.0:
		start_scope = f"\\begin{{scope}}[opacity={mask_opacity}]\n{clip_code}"
	else:
		start_scope = f"\\begin{{scope}}\n{clip_code}"

	self.active_page.sketches.append(TexSketch(start_scope))
	vertices_len = len(self._all_vertices)
	draw_kwargs = dict(kwargs)
	canvas_draw.draw(self, target, **draw_kwargs)
	del self._all_vertices[vertices_len:]
	self._all_vertices.extend(mask_shape.b_box.corners)
	self.active_page.sketches.append(TexSketch("\\end{scope}"))
	return self

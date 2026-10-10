"""Planar segment arrangement helpers for bounded face extraction.

The functions in this module work on 2D segment arrangements after
intersections have been made explicit as graph vertices. They are intended
for partition extraction and related topology queries over shape groups.

**Examples**

```python
import simetri.graphics as sg
square = sg.Shape([(0, 0), (10, 0), (10, 10), (0, 10)], closed=True)
parts, unused = partitions_and_unused_segments_from_group(sg.Group(square))
len(parts), unused
# (1, [])
```
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import atan2
from typing import TYPE_CHECKING

from ...base.common import LineType, PointType
from ..points.point_utils import remove_duplicate_points, round_points
from ..segments.line_utils import all_segments_sorted
from .polygon import remove_duplicate_edges
from .polygon_utils import right_handed

if TYPE_CHECKING:
	from simetri.group.batch import Group
	from simetri.shapes.shape import Shape

SNAP_DIGITS = 1

ArrangementPoint = tuple[float, float]
ArrangementEdge = tuple[ArrangementPoint, ArrangementPoint]
ArrangementPolygon = list[ArrangementPoint]


def _snap_points(
	points: Iterable[PointType], snap_digits: int = SNAP_DIGITS
) -> list[ArrangementPoint]:
	"""Snap point coordinates to the active arrangement grid.

	Args:
		points: Points to round.
		snap_digits: Decimal digits retained after snapping.

	Returns:
		list[ArrangementPoint]: Rounded points.
	"""
	return [tuple(point) for point in round_points(list(points), snap_digits)]


def _segment_key(
	start: ArrangementPoint, end: ArrangementPoint
) -> tuple[ArrangementPoint, ArrangementPoint]:
	"""Return an order-independent key for a segment."""
	return tuple(sorted((start, end)))


def _trace_face(
	start_edge: ArrangementEdge,
	ordered_neighbors: dict[ArrangementPoint, list[ArrangementPoint]],
	visited_half_edges: set[ArrangementEdge],
) -> ArrangementPolygon | None:
	"""Trace one face boundary from a directed edge.

	The walk follows the neighbor immediately before the reverse direction in
	the cyclic angle order around each vertex.
	"""
	face: ArrangementPolygon = []
	current_edge = start_edge
	seen_in_face: set[ArrangementEdge] = set()

	while True:
		if current_edge in seen_in_face:
			return None

		seen_in_face.add(current_edge)
		visited_half_edges.add(current_edge)

		start_point, end_point = current_edge
		face.append(start_point)

		neighbors = ordered_neighbors[end_point]
		reverse_index = neighbors.index(start_point)
		next_point = neighbors[reverse_index - 1]
		current_edge = (end_point, next_point)

		if current_edge == start_edge:
			return face


def collect_shape_edges(
	group: Group, snap_digits: int = SNAP_DIGITS
) -> list[ArrangementEdge]:
	"""Collect normalized edges from every shape in a group.

	Args:
		group: Group whose shapes provide the arrangement boundary.
		snap_digits: Decimal digits retained after snapping.

	Returns:
		list[ArrangementEdge]: Deduplicated normalized shape edges.

	Examples:
		>>> import simetri.graphics as sg
		>>> from simetri.geom.polygons.arrangement import collect_shape_edges
		>>> square = sg.Shape([(0, 0), (40, 0), (40, 40), (0, 40)], closed=True)
		>>> collect_shape_edges(sg.Group(square))
		[((0.0, 0.0), (40.0, 0.0)), ((40.0, 0.0), (40.0, 40.0)), ((40.0, 40.0), (0.0, 40.0)), ((0.0, 40.0), (0.0, 0.0))]
	"""
	edges: list[ArrangementEdge] = []
	for start, end in group.all_segments:
		normalized_start, normalized_end = _snap_points([start, end], snap_digits)
		if normalized_start == normalized_end:
			continue
		edges.append((normalized_start, normalized_end))

	return remove_duplicate_edges(edges, keep_one=True)


def split_edges(
	edges: Sequence[LineType], snap_digits: int = SNAP_DIGITS
) -> list[ArrangementEdge]:
	"""Split segments at all pairwise intersections.

	Args:
		edges: Input segments.
		snap_digits: Decimal digits retained after snapping.

	Returns:
		list[ArrangementEdge]: Deduplicated consecutive sub-segments.

	Examples:
		>>> from simetri.geom.polygons.arrangement import split_edges
		>>> split_edges([((0, 0), (40, 0)), ((20, -20), (20, 20))])
		[((0, 0), (20.0, 0.0)), ((20.0, 0.0), (40, 0)), ((20, -20), (20.0, 0.0)), ((20.0, 0.0), (20, 20))]
	"""
	if not edges:
		return []

	rounded_edges = [tuple(_snap_points(edge, snap_digits)) for edge in edges]
	split_segments = [
		tuple(_snap_points(edge, snap_digits))
		for edge in all_segments_sorted(rounded_edges)
	]

	return remove_duplicate_edges(split_segments, keep_one=True)


def bounded_faces(
	edges: Sequence[LineType], snap_digits: int = SNAP_DIGITS
) -> tuple[list[ArrangementPolygon], set[tuple[ArrangementPoint, ArrangementPoint]]]:
	"""Extract bounded faces from a split planar segment arrangement.

	Args:
		edges: Split planar segments.
		snap_digits: Decimal digits retained after snapping.

	Returns:
		tuple[list[ArrangementPolygon], set[tuple[ArrangementPoint, ArrangementPoint]]]:
		Closed face boundaries with repeated start point appended, and the
		undirected segment keys used by those faces.

	Examples:
		>>> from simetri.geom.polygons.arrangement import bounded_faces
		>>> faces, _ = bounded_faces(
		... [((0, 0), (40, 0)), ((40, 0), (40, 40)), ((40, 40), (0, 40)), ((0, 40), (0, 0))]
		... )
		>>> faces
		[[(0, 0), (40, 0), (40, 40), (0, 40), (0, 0)]]
	"""
	rounded_edges = [tuple(_snap_points(edge, snap_digits)) for edge in edges]

	adjacency: dict[ArrangementPoint, list[ArrangementPoint]] = {}
	for start_point, end_point in rounded_edges:
		adjacency.setdefault(start_point, []).append(end_point)
		adjacency.setdefault(end_point, []).append(start_point)

	ordered_neighbors = {
		point: sorted(
			neighbors,
			key=lambda neighbor: atan2(
				neighbor[1] - point[1], neighbor[0] - point[0]
			),
		)
		for point, neighbors in adjacency.items()
	}

	visited_half_edges: set[ArrangementEdge] = set()
	faces: list[ArrangementPolygon] = []
	seen_face_keys: set[frozenset[tuple[ArrangementPoint, ArrangementPoint]]] = set()
	used_segment_keys: set[tuple[ArrangementPoint, ArrangementPoint]] = set()

	for start_point, end_point in rounded_edges:
		for half_edge in ((start_point, end_point), (end_point, start_point)):
			if half_edge in visited_half_edges:
				continue

			face = _trace_face(half_edge, ordered_neighbors, visited_half_edges)
			if not face:
				continue
			face = remove_duplicate_points(_snap_points(face, snap_digits))
			if len(face) < 3:
				continue

			if not right_handed(face):
				continue

			face_keys = frozenset(
				_segment_key(face[index], face[(index + 1) % len(face)])
				for index in range(len(face))
			)
			if face_keys in seen_face_keys:
				continue

			seen_face_keys.add(face_keys)
			used_segment_keys.update(face_keys)
			faces.append(
				remove_duplicate_points(_snap_points(face + [face[0]], snap_digits))
			)

	return faces, used_segment_keys


def partitions_from_edges(
	edges: Sequence[LineType], snap_digits: int = SNAP_DIGITS
) -> list[ArrangementPolygon]:
	"""Return bounded partitions from split planar segments.

	Args:
		edges: Split planar segments.
		snap_digits: Decimal digits retained after snapping.

	Returns:
		list[ArrangementPolygon]: Closed polygon coordinate lists.

	Examples:
		>>> from simetri.geom.polygons.arrangement import partitions_from_edges
		>>> parts = partitions_from_edges(
		... [((0, 0), (40, 0)), ((40, 0), (40, 40)), ((40, 40), (0, 40)), ((0, 40), (0, 0))]
		... )
		>>> parts
		[[(0, 0), (40, 0), (40, 40), (0, 40), (0, 0)]]
	"""
	partitions, _ = bounded_faces(edges, snap_digits=snap_digits)
	return partitions


def partitions_and_unused_segments_from_edges(
	edges: Sequence[LineType], snap_digits: int = SNAP_DIGITS
) -> tuple[list[ArrangementPolygon], list[ArrangementEdge]]:
	"""Return bounded partitions and segments outside any bounded face.

	Args:
		edges: Split planar segments.
		snap_digits: Decimal digits retained after snapping.

	Returns:
		tuple[list[ArrangementPolygon], list[ArrangementEdge]]: Closed face
		boundaries and leftover split segments.

	Examples:
		>>> from simetri.geom.polygons.arrangement import (
		... partitions_and_unused_segments_from_edges,
		... )
		>>> edges = [((0, 0), (40, 0)), ((40, 0), (40, 40)), ((40, 40), (0, 40)), ((0, 40), (0, 0))]
		>>> parts, unused = partitions_and_unused_segments_from_edges(edges)
		>>> parts, unused
		([[(0, 0), (40, 0), (40, 40), (0, 40), (0, 0)]], [])
	"""
	partitions, used_segment_keys = bounded_faces(edges, snap_digits=snap_digits)
	rounded_edges = [tuple(_snap_points(edge, snap_digits)) for edge in edges]
	unused_segments = [
		edge
		for edge in rounded_edges
		if _segment_key(edge[0], edge[1]) not in used_segment_keys
	]
	return partitions, unused_segments


def partitions_and_unused_segments_from_group(
	group: Group, snap_digits: int = SNAP_DIGITS
) -> tuple[list[ArrangementPolygon], list[ArrangementEdge]]:
	"""Extract bounded partitions and leftover segments from a group.

	Args:
		group: Group containing polygonal shapes and linework.
		snap_digits: Decimal digits retained after snapping.

	Returns:
		tuple[list[ArrangementPolygon], list[ArrangementEdge]]: Bounded
		partitions and unused split segments.

	Examples:
		>>> import simetri.graphics as sg
		>>> from simetri.geom.polygons.arrangement import (
		... partitions_and_unused_segments_from_group,
		... )
		>>> square = sg.Shape([(0, 0), (40, 0), (40, 40), (0, 40)], closed=True)
		>>> parts, unused = partitions_and_unused_segments_from_group(sg.Group(square))
		>>> parts, unused
		([[(0.0, 0.0), (40.0, 0.0), (40.0, 40.0), (0.0, 40.0), (0.0, 0.0)]], [])
	"""
	edges = collect_shape_edges(group, snap_digits=snap_digits)
	split_segments = split_edges(edges, snap_digits=snap_digits)
	return partitions_and_unused_segments_from_edges(
		split_segments, snap_digits=snap_digits
	)


def prune_shapes(
	group: Group, snap_digits: int = SNAP_DIGITS
) -> tuple[list[Shape], list[Shape]]:
	"""Extract bounded partitions and leftover linework from a group.

	Collects shape edges, splits at intersections, then returns closed
	partition polygons and open two-point shapes for segments that belong to
	no bounded face (matching the usual draw loop: filled closed regions plus
	open strokes).

	Args:
		group: Group containing polygonal shapes and linework.
		snap_digits: Decimal digits retained after snapping.

	Returns:
		tuple[list[Shape], list[Shape]]: ``(closed_shapes, open_shapes)``.
		Closed shapes use ``closed=True``; open shapes are two-vertex polylines
		with ``closed=False``.

	Examples:
		>>> import simetri.graphics as sg
		>>> square = sg.Shape([(0, 0), (40, 0), (40, 40), (0, 40)], closed=True)
		>>> closed, open_shapes = sg.prune_shapes(sg.Group(square))
		>>> [[round(c, 6) for c in p[:2]] for p in closed[0].vertices], open_shapes
		([[0.0, 0.0], [40.0, 0.0], [40.0, 40.0], [0.0, 40.0]], [])
	"""
	from simetri.shapes.shape import Shape

	partitions, unused_segments = partitions_and_unused_segments_from_group(
		group, snap_digits=snap_digits
	)
	closed_shapes = [Shape(ring, closed=True) for ring in partitions]
	open_shapes = [
		Shape([start, end], closed=False) for start, end in unused_segments
	]
	return closed_shapes, open_shapes


__all__ = [
	"ArrangementEdge",
	"ArrangementPoint",
	"ArrangementPolygon",
	"SNAP_DIGITS",
	"bounded_faces",
	"collect_shape_edges",
	"partitions_and_unused_segments_from_edges",
	"partitions_and_unused_segments_from_group",
	"partitions_from_edges",
	"prune_shapes",
	"split_edges",
]

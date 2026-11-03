"""Interlacing paths with both linear and curvilinear parts.

Provides ``Interlace``, a ``Lace`` subclass that builds over/under weaving
patterns from mixed geometry: polyline and polygon ``Shape`` objects,
sampled curves (``Arc``, ``Ellipse``, ``Bezier``, Hobby and sine shapes),
parametric ``Circle`` objects, ``Path2D`` paths mixing linear and
curvilinear sections, and ``Group`` objects nesting any of these.

Curvilinear entities are reduced to their exact on-curve sample points
(parametric sampling for circles; library-sampled vertices for the other
curve types), then the ``Lace`` pipeline computes offsets, intersections,
overlaps, fragments, and plaits on the resulting polylines and polygons.

Examples:
    >>> import simetri.graphics as sg
    >>> from simetri.interlace.interlacing import Interlace
    >>> circle1 = sg.Circle(radius=50, center=(0, 0))
    >>> circle2 = sg.Circle(radius=50, center=(60, 0))
    >>> interlaced = Interlace([circle1, circle2], offset=5)
    >>> len(interlaced.overlaps)
    2
"""

from __future__ import annotations

from math import pi
from typing import Any

import networkx as nx
import numpy as np

from ..base.all_enums import Types
from ..coloring import colors
from ..config.settings import runtime_defaults
from ..geom.geom_utils import connected_pairs
from ..geom.nonlinear.path import Path2D
from ..geom.polygons.polygon_utils import right_handed
from ..group.batch import Group
from ..helpers.graph import get_cycles
from ..shapes.geom_items import Circle
from ..shapes.shape import Shape
from .lace import (
    Fragment,
    Lace,
    ParallelPolyline,
    Polyline,
    Section,
    all_intersections,
)

# Overlap-band membership is a strict inequality: section midpoints on a
# band boundary (a strand's own rails) sit at exactly ``offset`` distance
# from the generating spine, while tiny zone-corner sections at shallow
# crossings can lie inside another band by arbitrarily small amounts. The
# epsilon only needs to absorb the floating-point noise of the exact
# offset construction, so it is kept near machine precision; a larger
# margin would punch holes into the overlap-zone boundary rings.
_OVERLAP_DISTANCE_EPSILON = 1e-6


def _point_segment_distances(
    point: tuple, segments: np.ndarray
) -> np.ndarray:
    """Return distances from one point to many segments.

    Args:
        point: Query point; only ``point[:2]`` is used.
        segments: Array of shape ``(n, 4)`` with rows ``(x1, y1, x2, y2)``.

    Returns:
        np.ndarray: Distance from the point to each segment.
    """
    point_x, point_y = point[:2]
    x1 = segments[:, 0]
    y1 = segments[:, 1]
    x2 = segments[:, 2]
    y2 = segments[:, 3]
    delta_x = x2 - x1
    delta_y = y2 - y1
    seg_len2 = delta_x * delta_x + delta_y * delta_y
    safe_len2 = np.where(seg_len2 == 0, 1.0, seg_len2)
    t = ((point_x - x1) * delta_x + (point_y - y1) * delta_y) / safe_len2
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(point_x - (x1 + t * delta_x), point_y - (y1 + t * delta_y))


def circle_to_shape(circle: Circle, n_points: int | None = None) -> Shape:
    """Convert a parametric circle into a closed polygon shape.

    Sample points are computed parametrically in the circle's local frame
    and mapped through the circle's transformation matrix, so every vertex
    lies exactly on the transformed circle (an ellipse under non-uniform
    scale or shear).

    Args:
        circle: Circle to convert.
        n_points: Number of sample points. Defaults to
            ``runtime_defaults["n_circle_points"]``.

    Returns:
        Shape: Closed polygon shape approximating the circle.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.interlace.interlacing import circle_to_shape
        >>> shp = circle_to_shape(sg.Circle(radius=40, center=(10, 0)), n_points=4)
        >>> shp.closed
        True
        >>> [[round(float(c), 6) or 0.0 for c in p[:2]] for p in shp.vertices]
        [[50.0, 0.0], [10.0, 40.0], [-30.0, 0.0], [10.0, -40.0]]
    """
    if n_points is None:
        n_points = runtime_defaults["n_circle_points"]
    center_x, center_y = circle.primary_points[0][:2]
    radius = circle._radius
    angles = np.linspace(0.0, 2 * pi, n_points, endpoint=False)
    local_points = np.column_stack(
        (
            center_x + radius * np.cos(angles),
            center_y + radius * np.sin(angles),
            np.ones(n_points),
        )
    )
    points = local_points @ circle.xform_matrix
    return Shape([(float(x), float(y)) for x, y, _ in points], closed=True)


def _dedupe_points(points: list, closed: bool) -> list:
    """Drop consecutive duplicate points from a vertex sequence.

    Points closer than ``runtime_defaults["abs_tol"]`` in both
    coordinates count as duplicates; the pipeline cannot resolve them as
    separate nodes and the zero-length segments they form break shape
    merging. ``Path2D`` repeats the pen position where a line meets an
    arc, which is the typical source. For closed sequences a trailing
    duplicate of the first point is dropped as well.

    Args:
        points: Vertex sequence; only ``point[:2]`` is compared.
        closed: Whether the sequence describes a closed ring.

    Returns:
        list: Deduplicated vertex sequence.
    """
    tol = runtime_defaults["abs_tol"]
    result = [points[0]]
    for point in points[1:]:
        x, y = point[:2]
        last_x, last_y = result[-1][:2]
        if abs(x - last_x) <= tol and abs(y - last_y) <= tol:
            continue
        result.append(point)
    if closed and len(result) > 1:
        x, y = result[-1][:2]
        first_x, first_y = result[0][:2]
        if abs(x - first_x) <= tol and abs(y - first_y) <= tol:
            result.pop()
    return result


def path2d_to_shapes(path: Path2D) -> list[Shape]:
    """Extract the sampled subpaths of a path as plain shapes.

    Each pen-down subpath of the path becomes one shape; curvilinear
    sections (arcs, beziers, Hobby and sine curves) contribute their
    sampled on-curve vertices. Subpaths closed with ``close`` produce
    closed shapes. Bare ``move`` subpaths carry a single point and
    produce no shape. Consecutive duplicate vertices (``Path2D`` repeats
    the pen position where a line meets an arc) are dropped.

    Args:
        path: Path to convert.

    Returns:
        list[Shape]: One polyline or polygon shape per drawable subpath.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.interlace.interlacing import path2d_to_shapes
        >>> path = sg.Path2D((0, 0)).line_to((40, 0)).line_to((40, 30))
        >>> shapes = path2d_to_shapes(path)
        >>> len(shapes)
        1
        >>> [[round(float(c), 6) for c in p[:2]] for p in shapes[0].vertices]
        [[0.0, 0.0], [40.0, 0.0], [40.0, 30.0]]
    """
    shapes = []
    for element in path.elements:
        if element.type != Types.SHAPE:
            continue
        vertices = _dedupe_points(list(element.vertices), element.closed)
        if len(vertices) < 2:
            continue
        shapes.append(Shape(vertices, closed=element.closed))
    return shapes


def entity_to_shapes(
    entity: Shape | Group,
    n_circle_points: int | None = None,
) -> list[Shape]:
    """Convert a drawable entity into plain polyline/polygon shapes.

    Dispatches on the entity type: ``Path2D`` paths are split into their
    subpath shapes, ``Circle`` objects are sampled parametrically, other
    ``Shape`` objects (including ``Arc``, ``Ellipse``, ``Bezier``, Hobby
    and sine shapes) contribute their transformed vertices, and ``Group``
    objects are flattened recursively.

    Args:
        entity: Shape, Circle, Path2D, or (nested) Group to convert.
        n_circle_points: Sample count for circles. Defaults to
            ``runtime_defaults["n_circle_points"]``.

    Returns:
        list[Shape]: Plain shapes carrying the entity's final coordinates.

    Raises:
        TypeError: If the entity type is not supported.
        ValueError: If a shape entity has fewer than two vertices.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.interlace.interlacing import entity_to_shapes
        >>> group = sg.Group([sg.Circle(radius=20), sg.Shape([(0, 0), (40, 0)])])
        >>> shapes = entity_to_shapes(group, n_circle_points=12)
        >>> len(shapes)
        2
        >>> len(shapes[0].vertices), shapes[0].closed
        (12, True)
    """
    if isinstance(entity, Path2D):
        return path2d_to_shapes(entity)
    if isinstance(entity, Circle):
        return [circle_to_shape(entity, n_circle_points)]
    if isinstance(entity, Shape):
        vertices = _dedupe_points(list(entity.vertices), entity.closed)
        if len(vertices) < 2:
            raise ValueError(
                f"entity_to_shapes: {type(entity).__name__} with fewer than "
                "two vertices has no segments to interlace."
            )
        return [Shape(vertices, closed=entity.closed)]
    if isinstance(entity, Group):
        shapes = []
        for element in entity.elements:
            shapes.extend(entity_to_shapes(element, n_circle_points))
        return shapes
    raise TypeError(
        f"entity_to_shapes: unsupported entity type: {type(entity).__name__}."
    )


class Interlace(Lace):
    """Interlace pattern built from mixed linear and curvilinear geometry.

    Accepts any combination of ``Shape`` polylines/polygons, sampled curves
    (``Arc``, ``Ellipse``, ``Bezier``, Hobby and sine shapes), parametric
    ``Circle`` objects, ``Path2D`` paths, and ``Group`` objects nesting any
    of these. Entities are converted to polylines and polygons whose
    vertices lie exactly on the source curves, then the ``Lace`` pipeline
    builds the over/under weaving pattern.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.interlace.interlacing import Interlace
        >>> circle = sg.Circle(radius=50, center=(0, 0))
        >>> zigzag = sg.Shape([(-30, -80), (0, 80), (30, -80)])
        >>> interlaced = Interlace([circle, zigzag], offset=3)
        >>> len(interlaced.overlaps)
        4
        >>> interlaced.subtype.name
        'LACE'
    """

    def __init__(
        self,
        entities: Shape | Group | list | tuple = None,
        offset: float = 2,
        rel_tol: float | None = None,
        swatch: list | None = None,
        plait_color: colors.Color = None,
        draw_fragments: bool = True,
        palette: list | None = None,
        color_step: int = 1,
        with_plaits: bool = True,
        area_threshold: float | None = None,
        radius_threshold: float | None = None,
        abs_tol: float | None = None,
        merge_angle_tol: float = 0.1,
        n_circle_points: int | None = None,
        debug: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create an interlace pattern from mixed geometry.

        Args:
            entities: Entity, or sequence of entities, to interlace. Each
                entity is a ``Shape`` (polyline, polygon, ``Arc``,
                ``Ellipse``, ``Bezier``, Hobby or sine shape), ``Circle``,
                ``Path2D``, or a ``Group`` nesting any of these.
            offset: Offset distance for parallel polylines.
            rel_tol: Relative tolerance for geometric comparisons.
            swatch: Optional color swatch list.
            plait_color: Color used for plaits.
            draw_fragments: Whether fragments should be drawn.
            palette: Optional palette list.
            color_step: Step when cycling palette colors.
            with_plaits: Whether plaits are included.
            area_threshold: Minimum area filter for fragments.
            radius_threshold: Minimum radius filter for fragments.
            abs_tol: Distance tolerance when merging shapes.
            merge_angle_tol: Angle tolerance (radians) for merging
                collinear edges.
            n_circle_points: Sample count used when converting circles.
                Defaults to ``runtime_defaults["n_circle_points"]``.
            debug: Print shape-merge diagnostics when True.
            **kwargs: Reserved for internal ``_copy`` use only.
        """
        if isinstance(entities, (list, tuple)):
            items = list(entities)
        elif entities is not None:
            items = [entities]
        else:
            raise TypeError("Interlace.__init__ : Invalid entities argument.")
        shapes = []
        for item in items:
            shapes.extend(entity_to_shapes(item, n_circle_points))
        self.n_circle_points = n_circle_points
        super().__init__(
            Group(shapes),
            offset=offset,
            rel_tol=rel_tol,
            swatch=swatch,
            plait_color=plait_color,
            draw_fragments=draw_fragments,
            palette=palette,
            color_step=color_step,
            with_plaits=with_plaits,
            area_threshold=area_threshold,
            radius_threshold=radius_threshold,
            abs_tol=abs_tol,
            merge_angle_tol=merge_angle_tol,
            debug=debug,
            **kwargs,
        )

    def __repr__(self) -> str:
        """Return an Interlace string from this interlace's elements.

        Examples:
            >>> import simetri.graphics as sg
            >>> from simetri.interlace.interlacing import Interlace
            >>> circle1 = sg.Circle(radius=50, center=(0, 0))
            >>> circle2 = sg.Circle(radius=50, center=(60, 0))
            >>> interlaced = Interlace([circle1, circle2], offset=5)
            >>> repr(interlaced).startswith("Interlace(")
            True
        """
        if self.elements is None or len(self.elements) == 0:
            return "Interlace()"
        if len(self.elements) in [1, 2]:
            return f"Interlace({self.elements})"
        return f"Interlace({self.elements[0]}...{self.elements[-1]})"

    def _set_polyline_list(self) -> None:
        """Populate ``self.polyline_list`` from polygon and polyline shapes.

        Closed rings are normalized to right-handed vertex order before the
        ``Polyline`` objects are created. The offset construction reverses
        clockwise rings, so normalizing here keeps the division order of the
        main polyline and its two offset polylines in correspondence.
        """
        self.polyline_list = []
        if self.polygon_shapes:
            for polygon in self.polygon_shapes:
                vertices = list(polygon.vertices)
                if not right_handed(vertices):
                    vertices.reverse()
                self.polyline_list.append(Polyline(vertices, closed=True))
        if self.polyline_shapes:
            for polyline in self.polyline_shapes:
                self.polyline_list.append(
                    Polyline(polyline.vertices, closed=False)
                )

    def _spine_segment_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the main-division segments and their strand indices.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(segments, strands)`` where
            ``segments`` has rows ``(x1, y1, x2, y2)`` for every main
            division and ``strands`` holds the index of the parallel
            polyline each segment belongs to.
        """
        rows = []
        strands = []
        for strand_index, ppoly in enumerate(self.parallel_poly_list):
            for division in ppoly.polyline.divisions:
                x1, y1 = division.p1[:2]
                x2, y2 = division.p2[:2]
                rows.append((x1, y1, x2, y2))
                strands.append(strand_index)
        return np.array(rows), np.array(strands)

    def _set_intersections(self) -> None:
        """Compute intersections and sections for main and offset divisions.

        Unlike ``Lace``, overlap sections are detected geometrically: a
        section is an overlap-boundary section when its midpoint lies
        strictly inside the offset band of a strand (closer to a main
        polyline than ``offset``). Points on a section's own rail sit at
        exactly ``offset`` distance from their spine, so they are excluded
        automatically and self-crossing strands are handled. This removes
        the ``Lace`` requirement that a crossing zone must fall inside a
        single polyline edge, which densely sampled curves cannot satisfy.

        Main sections are flagged against the bands of *other* strands
        only; this flag is informational (serialization) and does not
        drive the weaving.
        """
        main_divisions = self.main_divisions
        offset_divisions = self.offset_divisions
        self.main_intersections = all_intersections(
            main_divisions, self.d_intersections, self.d_connections
        )
        spine_segments, spine_strands = self._spine_segment_arrays()
        threshold = self.offset * (1.0 - _OVERLAP_DISTANCE_EPSILON)

        self.main_sections = []
        for strand_index, ppoly in enumerate(self.parallel_poly_list):
            other_strands = spine_strands != strand_index
            for division in ppoly.polyline.divisions:
                division.intersections[0].endpoint = True
                division.intersections[-1].endpoint = True
                division.sections = []
                for pair in connected_pairs(division.intersections):
                    section = Section(*pair)
                    if other_strands.any():
                        start_x, start_y = section.start.point[:2]
                        end_x, end_y = section.end.point[:2]
                        mid = (
                            (start_x + end_x) / 2,
                            (start_y + end_y) / 2,
                        )
                        distances = _point_segment_distances(
                            mid, spine_segments[other_strands]
                        )
                        section.is_overlap = bool(
                            distances.min() < threshold
                        )
                    division.sections.append(section)
                    self.main_sections.append(section)

        self.offset_intersections = all_intersections(
            offset_divisions, self.d_intersections, self.d_connections
        )

        self.offset_sections = []
        for division in offset_divisions:
            division.intersections[0].endpoint = True
            division.intersections[-1].endpoint = True
            division.sections = []
            for pair in connected_pairs(division.intersections):
                section = Section(*pair)
                start_x, start_y = section.start.point[:2]
                end_x, end_y = section.end.point[:2]
                mid = ((start_x + end_x) / 2, (start_y + end_y) / 2)
                distances = _point_segment_distances(mid, spine_segments)
                section.is_overlap = bool(distances.min() < threshold)
                division.sections.append(section)
                self.offset_sections.append(section)

    def _set_twin_sections(self) -> None:
        """Pair each offset section with its twin on the other rail.

        Sections are paired per corresponding division. When a crossing
        zone spans a polyline vertex, the two rails of a strand can carry
        different section counts within corresponding divisions; such
        divisions keep ``twin`` as ``None`` because index pairing would
        mismatch sections.
        """
        for par_poly in self.parallel_poly_list:
            poly1, poly2 = par_poly.offset_poly_list
            for division1, division2 in zip(poly1.divisions, poly2.divisions):
                if len(division1.sections) != len(division2.sections):
                    continue
                for section1, section2 in zip(
                    division1.sections, division2.sections
                ):
                    section1.twin = section2
                    section2.twin = section1

    def _overlap_runs(self, poly: Polyline) -> list[tuple]:
        """Collect maximal runs of overlap sections along an offset polyline.

        A run is a maximal sequence of consecutive overlap sections that
        belong to the same overlap zone; a zone that spans polyline
        vertices contributes one run covering several sections. For closed
        polylines, a run straddling the seam is merged into one.

        Args:
            poly: Offset polyline to walk.

        Returns:
            list[tuple]: ``(overlap, division_indices, sections)`` triples
            in walk order.

        Raises:
            RuntimeError: If an overlap section has no overlap object.
        """
        runs = []
        current_overlap = None
        current_divisions = None
        current_sections = None
        first_section_overlap = None
        last_section_overlap = False
        for division_index, division in enumerate(poly.divisions):
            for section in division.sections:
                if first_section_overlap is None:
                    first_section_overlap = section.is_overlap
                if section.is_overlap:
                    if section.overlap is None:
                        msg = (
                            "Overlap section in the interlace has no "
                            "overlap object.\n"
                            "Try different offset value and/or tolerance."
                        )
                        raise RuntimeError(msg)
                    if current_overlap is section.overlap:
                        current_divisions.append(division_index)
                        current_sections.append(section)
                    else:
                        if current_overlap is not None:
                            runs.append(
                                (
                                    current_overlap,
                                    current_divisions,
                                    current_sections,
                                )
                            )
                        current_overlap = section.overlap
                        current_divisions = [division_index]
                        current_sections = [section]
                    last_section_overlap = True
                else:
                    if current_overlap is not None:
                        runs.append(
                            (
                                current_overlap,
                                current_divisions,
                                current_sections,
                            )
                        )
                        current_overlap = None
                    last_section_overlap = False
        if current_overlap is not None:
            runs.append((current_overlap, current_divisions, current_sections))
        if (
            poly.closed
            and len(runs) > 1
            and first_section_overlap
            and last_section_overlap
            and runs[0][0] is runs[-1][0]
        ):
            overlap, divisions, sections = runs.pop()
            runs[0] = (
                overlap,
                divisions + runs[0][1],
                sections + runs[0][2],
            )
        return runs

    def _mark_run_over(
        self,
        ppoly: ParallelPolyline,
        run: tuple,
        d_runs_b: dict,
    ) -> None:
        """Mark a strand's crossing run (both rails) as going over.

        The matching run on the strand's second rail is selected by
        circular division-index proximity; for a self-crossing strand this
        separates the two passes through the same overlap zone.

        Args:
            ppoly: Parallel polyline whose strand goes over.
            run: ``(overlap, division_indices, sections)`` from the first
                rail.
            d_runs_b: Mapping of parallel polyline id to the runs of its
                second rail.

        Raises:
            RuntimeError: If the overlap zone has no run on the second rail.
        """
        overlap, division_indices, sections = run
        for section in sections:
            section.is_over = True
        poly_b = ppoly.offset_poly_list[1]
        candidates = [r for r in d_runs_b[ppoly.id] if r[0] is overlap]
        if not candidates:
            msg = (
                "Interlace: overlap zone is missing on one of the offset "
                "polylines.\n"
                "Try different offset value and/or tolerance."
            )
            raise RuntimeError(msg)
        n_divisions = len(poly_b.divisions)
        center = sum(division_indices) / len(division_indices)

        def circular_gap(candidate: tuple) -> float:
            candidate_center = sum(candidate[1]) / len(candidate[1])
            gap = abs(candidate_center - center)
            return min(gap, n_divisions - gap)

        best = min(candidates, key=circular_gap)
        for section in best[2]:
            section.is_over = True

    def _set_over_under(self) -> None:
        """Assign which overlapping offset sections pass over or under.

        Follows the same greedy alternation as ``Lace`` but walks overlap
        *zones* (runs of consecutive overlap sections) instead of single
        sections, and marks the matching run on the second rail
        geometrically instead of by section index. This supports zones
        that span polyline vertices and strands whose two rails carry
        different section counts.
        """
        d_runs_a = {}
        d_runs_b = {}
        for ppoly in self.parallel_poly_list:
            poly_a, poly_b = ppoly.offset_poly_list
            d_runs_a[ppoly.id] = self._overlap_runs(poly_a)
            d_runs_b[ppoly.id] = self._overlap_runs(poly_b)

        def next_ppoly(exclude: set) -> tuple:
            for candidate in self.parallel_poly_list:
                if candidate.id in exclude:
                    continue
                for ind, run in enumerate(d_runs_a[candidate.id]):
                    overlap = run[0]
                    if overlap.drawable and overlap.visited:
                        return candidate, ind % 2
            return None, None

        for start_ppoly in self.parallel_poly_list:
            ppoly = start_ppoly
            even_odd = 0
            exclude = set()
            while ppoly is not None:
                for ind, run in enumerate(d_runs_a[ppoly.id]):
                    overlap = run[0]
                    overlap.visited = True
                    if ind % 2 == even_odd and overlap.drawable:
                        overlap.drawable = False
                        self._mark_run_over(ppoly, run, d_runs_b)
                exclude.add(ppoly.id)
                ppoly, even_odd = next_ppoly(exclude)

    def _set_fragments(self) -> None:
        """Build fragment polygons from non-overlap offset-section cycles.

        Follows ``Lace._set_fragments`` but also closes the fragment graph
        across the ends of open strands: the two offset rails of an open
        polyline are joined with a synthetic end section at each end, so a
        region whose boundary wraps around an open strand end is still
        detected as a fragment. Because the end sections also close the
        outer boundary of the pattern, the largest cycle is stored as
        ``self.outline`` and removed from the fragments when open strands
        are present (``Lace`` skips ``_set_outline`` in that case).
        """
        G = nx.Graph()
        for section in self.iter_offset_sections():
            if section.is_overlap:
                continue
            G.add_edge(section.start.id, section.end.id, section=section)
        for ppoly in self.parallel_poly_list:
            if not ppoly.closed:
                polyline1, polyline2 = ppoly.offset_poly_list
                end_pairs = (
                    (polyline1.intersections[0], polyline2.intersections[0]),
                    (polyline1.intersections[-1], polyline2.intersections[-1]),
                )
                for x1, x2 in end_pairs:
                    end_section = Section(x1, x2)
                    self.d_intersections[x1.id] = x1
                    self.d_intersections[x2.id] = x2
                    G.add_edge(x1.id, x2.id, section=end_section)

        cycles = nx.cycle_basis(G)
        fragments = []
        d_x = self.d_intersections
        for cycle in cycles:
            cycle.append(cycle[0])
            nodes = cycle
            edges = connected_pairs(cycle)
            sections = [G.edges[edge]["section"] for edge in edges]
            s_intersections = set()
            for section in sections:
                s_intersections.add(section.start.id)
                s_intersections.add(section.end.id)
            intersections = [self.d_intersections[i] for i in s_intersections]
            points = [d_x[x_id].point for x_id in nodes]
            if not right_handed(points):
                points.reverse()
                sections.reverse()
            fragment = Fragment(points)
            fragment.sections = sections
            fragment.intersections = intersections
            fragments.append(fragment)

        for fragment in fragments:
            for section in fragment.sections:
                section.fragment = fragment

        for fragment in fragments:
            fragment._set_divisions()
        for fragment in fragments:
            fragment._set_twin_divisions()

        if self.polyline_shapes and fragments:
            areas = [(fragment.area, fragment) for fragment in fragments]
            areas.sort(reverse=True, key=lambda pair: pair[0])
            self.outline = areas[0][1]
            fragments.remove(self.outline)

        self.fragments = fragments

    def _plait_chains(self, poly: Polyline) -> list[list]:
        """Merge sections of an offset polyline into intersection chains.

        Chains are maximal runs of consecutive non-over sections along the
        whole polyline, continuing across division boundaries. ``Lace``
        merges per division, which duplicates a crossing edge when a
        division lies entirely inside an overlap zone; merging across
        divisions keeps chain endpoints distinct from the crossing edges.

        Args:
            poly: Offset polyline to walk.

        Returns:
            list[list]: Chains of intersections along the polyline.
        """
        chains = []
        current = None
        for division in poly.divisions:
            for section in division.sections:
                if section.is_over:
                    if current is not None:
                        chains.append(current)
                        current = None
                    continue
                if current is None:
                    current = [section.start, section.end]
                elif section.start.id == current[-1].id:
                    current.append(section.end)
                else:
                    chains.append(current)
                    current = [section.start, section.end]
        if current is not None:
            chains.append(current)
        if (
            poly.closed
            and len(chains) > 1
            and chains[-1][-1].id == chains[0][0].id
        ):
            first = chains.pop(0)
            chains[-1] = chains[-1] + first[1:]
        return chains

    def set_plaits(self) -> None:
        """Populate ``self.plaits`` with the over/under plait shapes.

        Follows ``Lace.set_plaits`` but merges plait rail chains across
        division boundaries (see ``_plait_chains``), which densely sampled
        curves require because their crossing zones span polyline vertices.
        The plait graph uses one contracted edge per chain to keep the
        cycles of neighboring plaits edge-disjoint; the full vertex detail
        of each chain is restored when the plait polygons are built. A
        chain that forms a complete ring (a strand that never passes over)
        becomes a ring plait directly.
        """
        if self.plaits:
            return
        graph_edges = []
        d_edge_chain = {}
        ring_chains = []
        for ppoly in self.parallel_poly_list:
            for poly in ppoly.offset_poly_list:
                for chain in self._plait_chains(poly):
                    key = frozenset((chain[0].id, chain[-1].id))
                    if len(key) == 1:
                        ring_chains.append(chain)
                        continue
                    graph_edges.append((chain[0].id, chain[-1].id))
                    d_edge_chain[key] = chain

        # connect the open ends of the polyline_shapes
        for ppoly in self.parallel_poly_list:
            if not ppoly.closed:
                polyline1 = ppoly.offset_poly_list[0]
                polyline2 = ppoly.offset_poly_list[1]
                p1_start_x = polyline1.intersections[0]
                p1_end_x = polyline1.intersections[-1]
                p2_start_x = polyline2.intersections[0]
                p2_end_x = polyline2.intersections[-1]

                graph_edges.append((p1_start_x.id, p2_start_x.id))
                graph_edges.append((p1_end_x.id, p2_end_x.id))
        graph_edges.extend(
            (sec.start.id, sec.end.id)
            for sec in self.iter_offset_sections()
            if not sec.is_over and sec.is_overlap
        )

        cycles = get_cycles(graph_edges)
        if cycles is None:
            cycles = []
        node_cycles = []
        for cycle in cycles:
            cycle = connected_pairs(cycle)
            dup = cycle[1:]
            node_cycle = [cycle[0][0], cycle[0][1]]
            for _ in range(len(cycle) - 1):
                last = node_cycle[-1]
                for edge in dup:
                    if edge[0] == last:
                        node_cycle.append(edge[1])
                        dup.remove(edge)
                        break
                    if edge[1] == last:
                        node_cycle.append(edge[0])
                        dup.remove(edge)
                        break
            node_cycles.append(node_cycle)
        d_x = self.d_intersections
        plait_intersections = []
        for node_cycle in node_cycles:
            expanded = []
            for node1, node2 in connected_pairs(node_cycle):
                key = frozenset((node1, node2))
                if key in d_edge_chain:
                    chain = d_edge_chain[key]
                    if chain[0].id == node1:
                        expanded.extend(chain[:-1])
                    else:
                        expanded.extend(chain[::-1][:-1])
                else:
                    expanded.append(d_x[node1])
            expanded.append(expanded[0])
            plait_intersections.append(expanded)
        plait_intersections.extend(ring_chains)
        for intersections in plait_intersections:
            vertices = [x.point for x in intersections]
            if not right_handed(vertices):
                vertices.reverse()
                intersections.reverse()
            shape = Shape(vertices)
            shape.intersections = intersections
            shape.fill_color = runtime_defaults["plait_color"]
            shape.inner_polygons = None
            shape.inner_lines = None
            shape.subtype = Types.PLAIT
            shape.lerp_points = []
            self.plaits.append(shape)

        # set the plait-ends
        self._set_plait_ends()
        # set the plait_connections
        self._set_plait_connections()

    def _set_plait_connections(self) -> None:
        """Create the index pairs used to draw interior plait connectors.

        Same as ``Lace._set_plait_connections``, but a plait without
        matched end edges (a ring plait that never passes over) gets no
        connections instead of failing.
        """
        for plait in self.plaits:
            if not plait.ends:
                plait.connections = []
                continue
            s = min(plait.ends) + 1
            n = len(plait)
            connections = []
            for i, j in enumerate(range(s, s + n // 2)):
                k = (n + s - 1 - i) % n
                connections.append((j, k))
            plait.connections = connections

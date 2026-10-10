"""Label font helpers and vertex and edge label layout."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from math import hypot
from typing import Any

from ..base.all_enums import (
    Align,
    FontFamily,
    FontSize,
    MarkerType,
    Types,
)
from ..coloring import colors
from ..config.settings import runtime_defaults
from ..geom.geom_utils import midpoint
from ..geom.geometry import bbox_overlap
from ..geom.vectors import Vector, perp_unit_vector, v_from_points
from ..shapes.shape import Shape
from .illustration import Tag, default_font_size_pt, latex_font_size_to_pt
from .label_overlap import LabelRect, resolve_all_overlaps

Color = colors.Color


def sketch_label_font_size_pt(sketch: Any, label_kind: str) -> float:
    """Label font size in points from sketch kwargs or defaults.

    Args:
        sketch: Sketch providing optional font-size attributes.
        label_kind (str): ``index`` or ``vertex``.

    Returns:
        float: Font size in points.

    Examples:
        >>> import simetri.graphics as sg
        >>> from types import SimpleNamespace
        >>> sketch = SimpleNamespace(index_font_size=12)
        >>> sg.sketch_label_font_size_pt(sketch, 'index')
        12.0
    """
    if label_kind == "index":
        attr = "index_font_size"
    else:
        attr = "vertex_font_size"
    if attr in sketch.__dict__:
        size = sketch.__dict__[attr]
        if size is not None:
            if isinstance(size, (int, float)):
                return float(size)
            return latex_font_size_to_pt(FontSize(size))
    return default_font_size_pt(attr)


def sketch_label_offset(sketch: Any, label_kind: str) -> float:
    """Label radial offset in points from sketch kwargs or defaults.

    Args:
        sketch: Sketch providing optional offset attributes.
        label_kind (str): ``index`` or ``vertex``.

    Returns:
        float: Radial offset in points.

    Examples:
        >>> import simetri.graphics as sg
        >>> from types import SimpleNamespace
        >>> sketch = SimpleNamespace(index_offset=7)
        >>> sg.sketch_label_offset(sketch, 'index')
        7.0
    """
    if label_kind == "index":
        attr = "index_offset"
    else:
        attr = "vertex_offset"
    try:
        return float(object.__getattribute__(sketch, attr))
    except AttributeError:
        return float(runtime_defaults[attr])


def sketch_label_font_color(sketch: Any, label_kind: str) -> Color:
    """Label text color from sketch kwargs or defaults.

    Args:
        sketch: Sketch providing optional font-color attributes.
        label_kind (str): ``index`` or ``vertex``.

    Returns:
        Color: Label text color.

    Examples:
        >>> import simetri.graphics as sg
        >>> from types import SimpleNamespace
        >>> sketch = SimpleNamespace(index_font_color=sg.red)
        >>> sg.sketch_label_font_color(sketch, 'index')
        Color(0.898, 0.0, 0.0)
    """
    if label_kind == "index":
        attr = "index_font_color"
    else:
        attr = "vertex_font_color"
    if attr in sketch.__dict__ and sketch.__dict__[attr] is not None:
        return sketch.__dict__[attr]
    return runtime_defaults[attr]


def sketch_label_font_family(sketch: Any, label_kind: str) -> str | FontFamily:
    """Label font family from sketch kwargs or defaults.

    Args:
        sketch: Sketch providing optional font-family attributes.
        label_kind (str): ``index`` or ``vertex``.

    Returns:
        str | FontFamily: TeX switch name, CSS-ish name, or FontFamily.

    Examples:
        >>> import simetri.graphics as sg
        >>> from types import SimpleNamespace
        >>> sketch = SimpleNamespace(index_font_family=sg.FontFamily.MONOSPACE)
        >>> sg.sketch_label_font_family(sketch, 'index')
        <FontFamily.MONOSPACE: 'monospace'>
    """
    if label_kind == "index":
        attr = "index_font_family"
    else:
        attr = "vertex_font_family"
    if attr in sketch.__dict__ and sketch.__dict__[attr] is not None:
        return sketch.__dict__[attr]
    return runtime_defaults[attr]


def label_font_family_tikz(family: FontFamily | str) -> str:
    """Map a label font-family value to a TeX font switch (no backslash).

    Args:
        family: ``FontFamily``, or a string such as ``ttfamily`` / ``monospace``.

    Returns:
        str: One of ``ttfamily``, ``rmfamily``, ``sffamily``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.label_font_family_tikz(sg.FontFamily.SANSSERIF)
        'sffamily'
        >>> sg.label_font_family_tikz(sg.FontFamily.MONOSPACE)
        'ttfamily'
        >>> sg.label_font_family_tikz('sans')
        'sffamily'
        >>> sg.label_font_family_tikz(sg.FontFamily.SERIF)
        'rmfamily'
    """
    if isinstance(family, FontFamily):
        if family == FontFamily.MONOSPACE:
            return "ttfamily"
        if family == FontFamily.SANSSERIF:
            return "sffamily"
        return "rmfamily"

    normalized = str(family).strip().lower().replace("-", "").replace("_", "")
    if normalized in ("ttfamily", "texttt", "monospace", "mono"):
        return "ttfamily"
    if normalized in ("sffamily", "textsf", "sansserif", "sans"):
        return "sffamily"
    if normalized in ("rmfamily", "textrm", "serif", "rm"):
        return "rmfamily"
    raise ValueError(
        f"Unsupported label font family for TikZ: {family!r}. "
        "Use FontFamily or ttfamily/rmfamily/sffamily."
    )


def _tag_font_family_for_label_bounds(
    family: FontFamily | str,
) -> FontFamily | str:
    """Map label font-family settings to ``Tag.font_family`` for ``text_bounds``.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.helpers.labels import _tag_font_family_for_label_bounds
        >>> _tag_font_family_for_label_bounds(sg.FontFamily.SANSSERIF)
        <FontFamily.SANSSERIF: 'sansserif'>
        >>> _tag_font_family_for_label_bounds('ttfamily')
        <FontFamily.MONOSPACE: 'monospace'>
        >>> _tag_font_family_for_label_bounds('sans')
        <FontFamily.SANSSERIF: 'sansserif'>
    """
    css = label_font_family_svg(family)
    if css == "monospace":
        return FontFamily.MONOSPACE
    if css == "sans-serif":
        return FontFamily.SANSSERIF
    return FontFamily.SERIF


def label_font_family_svg(family: FontFamily | str) -> str:
    """Map a label font-family value to a CSS ``font-family`` keyword.

    Args:
        family: ``FontFamily``, or a string such as ``ttfamily`` / ``monospace``.

    Returns:
        str: ``monospace``, ``serif``, or ``sans-serif``.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.label_font_family_svg('sans')
        'sans-serif'
        >>> sg.label_font_family_svg(sg.FontFamily.MONOSPACE)
        'monospace'
        >>> sg.label_font_family_svg(sg.FontFamily.SERIF)
        'serif'
        >>> sg.label_font_family_svg(sg.FontFamily.SANSSERIF)
        'sans-serif'
    """
    if isinstance(family, FontFamily):
        if family == FontFamily.MONOSPACE:
            return "monospace"
        if family == FontFamily.SANSSERIF:
            return "sans-serif"
        return "serif"

    normalized = str(family).strip().lower().replace("-", "").replace("_", "")
    if normalized in ("ttfamily", "texttt", "monospace", "mono"):
        return "monospace"
    if normalized in ("sffamily", "textsf", "sansserif", "sans"):
        return "sans-serif"
    if normalized in ("rmfamily", "textrm", "serif", "rm"):
        return "serif"
    raise ValueError(
        f"Unsupported label font family for SVG: {family!r}. "
        "Use FontFamily or ttfamily/rmfamily/sffamily."
    )


def label_halo_color() -> Color:
    """Stroke/halo color behind vertex index and coordinate labels.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.label_halo_color()
        Color(1.0, 1.0, 1.0)
    """
    return runtime_defaults["label_halo_color"]


def label_halo_stroke_width(font_size_pt: float) -> float:
    """SVG halo stroke width / TikZ ``\\contourlength`` in points.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.label_halo_stroke_width(10.0)
        1.4000000000000001
    """
    scale = float(runtime_defaults["label_halo_width_scale"])
    return max(0.2, font_size_pt * scale)


def label_halo_scale() -> float:
    """Legacy scale factor (SVG/TikZ use stroke width / contour length instead).

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.label_halo_scale()
        1.14
    """
    return float(runtime_defaults["label_halo_scale"])


def svg_label_paint_attrs(fill_color: Color, font_size_pt: float) -> str:
    """SVG fill/stroke attributes for halo-backed label text.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.svg_label_paint_attrs(sg.black, 12.0)
        'fill="rgb(0, 0, 0)" stroke="rgb(255, 255, 255)" stroke-width="1.6800000000000002" paint-order="stroke fill"'
    """
    fill_r, fill_g, fill_b = fill_color.rgb255
    halo_r, halo_g, halo_b = label_halo_color().rgb255
    width = label_halo_stroke_width(font_size_pt)
    return (
        f'fill="rgb({fill_r}, {fill_g}, {fill_b})" '
        f'stroke="rgb({halo_r}, {halo_g}, {halo_b})" '
        f'stroke-width="{width}" paint-order="stroke fill"'
    )


def vert_label_layout(shape: Shape, offset: float) -> list[dict[str, object]]:
    """Return label anchor, outward direction, and vertex for each vertex.

    Examples:
        >>> import simetri.graphics as sg
        >>> layout = sg.vert_label_layout(sg.Shape([(0, 0), (40, 0), (0, 40)]), 5.0)
        >>> [(tuple(round(c, 6) for c in item['position']), item['vertex']) for item in layout]
        [((-3.535534, -3.535534), (0.0, 0.0)), ((44.619398, -1.913417), (40.0, 0.0)), ((-1.913417, 44.619398), (0.0, 40.0))]
    """
    from simetri.geom.polygons.polygon import in_polygon

    vertices = list(shape.vertices)

    vec1 = v_from_points(vertices[0], vertices[-1])
    count = len(vertices)

    layout = []
    for i, vert in enumerate(vertices):
        prev = vertices[i - 1][:2]
        next = vertices[(i + 1) % count][:2]
        point = vert

        vec2 = v_from_points(point, next)
        vert_vec = Vector(point)

        bisector = vec1.bisector(vec2)
        if bisector.norm() < 1e-9:
            direction = Vector(perp_unit_vector((prev, next)))
        else:
            direction = bisector.normalize()

        test_point = vert_vec + direction
        if in_polygon(test_point, vertices):
            pos = vert_vec - direction * offset
        else:
            pos = vert_vec + direction * offset

        to_label = pos - vert_vec
        if to_label.norm() > 1e-9:
            placement_dir = to_label.normalize()
        else:
            placement_dir = direction

        layout.append(
            {
                "position": (pos.x, pos.y),
                "direction": (placement_dir.x, placement_dir.y),
                "vertex": tuple(vert[:2]),
            }
        )
        vec1 = -vec2

    return layout


# Rule-of-thumb tiers replaced by Tag.text_bounds (Pillow glyph bbox).


def _label_size_from_tag_text_bounds(
    text: str,
    font_size_pt: float,
    font_family: FontFamily | str | None = None,
) -> tuple[float, float]:
    """Return ``(width, height)`` from ``Tag.text_bounds`` (no frame padding).

    Uses a centered Tag with ``inner_sep=0`` so the size matches the Pillow
    ink box. ``font_family`` must match the label draw path (sketch / defaults).
    """
    if font_family is None:
        tag_family = runtime_defaults["font_family"]
    else:
        tag_family = _tag_font_family_for_label_bounds(font_family)
    tag = Tag(
        str(text),
        pos=(0.0, 0.0),
        font_size=font_size_pt,
        font_family=tag_family,
        align=Align.CENTER,
    )
    tag.frame.inner_sep = 0
    xmin, ymin, xmax, ymax = tag.text_bounds()
    return xmax - xmin, ymax - ymin


def estimate_index_label_bbox(
    label: object,
    font_size_pt: float,
    font_family: FontFamily | str | None = None,
) -> tuple[float, float]:
    """Width/height for an index label from ``Tag.text_bounds``.

    Args:
        label: Index label value (converted with ``str``).
        font_size_pt (float): Font size in points.
        font_family: TeX switch, ``FontFamily``, or ``None`` for
            ``runtime_defaults['index_font_family']``.

    Returns:
        tuple[float, float]: ``(width, height)`` of the label box.

    Note:
        Size comes from Pillow glyph metrics and varies by installed font,
        so the call is skipped in doctests.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.estimate_index_label_bbox('0', 12.0)  # doctest: +SKIP
    """
    if font_family is None:
        font_family = runtime_defaults["index_font_family"]
    return _label_size_from_tag_text_bounds(
        str(label), font_size_pt, font_family
    )


def estimate_vertex_coord_label_bbox(
    text: str,
    font_size_pt: float,
    font_family: FontFamily | str | None = None,
) -> tuple[float, float]:
    """Width/height for a vertex coordinate label from ``Tag.text_bounds``.

    Args:
        text (str): Coordinate label text.
        font_size_pt (float): Font size in points.
        font_family: TeX switch, ``FontFamily``, or ``None`` for
            ``runtime_defaults['vertex_font_family']``.

    Returns:
        tuple[float, float]: ``(width, height)`` of the label box.

    Note:
        Size comes from Pillow glyph metrics and varies by installed font,
        so the call is skipped in doctests.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.estimate_vertex_coord_label_bbox('0, 0', 12.0)  # doctest: +SKIP
    """
    if font_family is None:
        font_family = runtime_defaults["vertex_font_family"]
    return _label_size_from_tag_text_bounds(text, font_size_pt, font_family)


def _centered_label_bbox(
    center: tuple[float, float], size: tuple[float, float]
) -> tuple[float, float, float, float]:
    cx, cy = center
    w, h = size
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _label_bboxes_overlap(
    center_a: tuple[float, float],
    size_a: tuple[float, float],
    center_b: tuple[float, float],
    size_b: tuple[float, float],
    shrink: float = 1.0,
) -> bool:
    if shrink != 1.0:
        size_a = (size_a[0] * shrink, size_a[1] * shrink)
        size_b = (size_b[0] * shrink, size_b[1] * shrink)
    a = _centered_label_bbox(center_a, size_a)
    b = _centered_label_bbox(center_b, size_b)
    return bbox_overlap(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3])


def _label_axis_overlaps(
    center_a: tuple[float, float],
    size_a: tuple[float, float],
    center_b: tuple[float, float],
    size_b: tuple[float, float],
) -> tuple[float, float]:
    """Return horizontal and vertical overlap amounts (0 if disjoint)."""
    ax, ay = center_a
    bx, by = center_b
    aw, ah = size_a
    bw, bh = size_b
    overlap_h = min(ax + aw / 2, bx + bw / 2) - max(ax - aw / 2, bx - bw / 2)
    overlap_v = min(ay + ah / 2, by + bh / 2) - max(ay - ah / 2, by - bh / 2)
    return max(0.0, overlap_h), max(0.0, overlap_v)


def format_vertex_coord(x: float, y: float, ndigits: int | None = None) -> str:
    """Return ``(x, y)`` formatted for vertex-coordinate labels.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.format_vertex_coord(1.2345, 6.789, ndigits=2)
        '(1.23, 6.79)'
        >>> sg.format_vertex_coord(1.2345, 6.789, ndigits=1)
        '(1.2, 6.8)'
    """
    if ndigits is None:
        ndigits = runtime_defaults["n_vert_digits"]
    return f"({round(float(x), ndigits)}, {round(float(y), ndigits)})"


_RESOLVED_LABELS_KEY = "_simetri_resolved_vertex_labels"
_LABEL_RECTS_KEY = "_simetri_label_rects"
_LABEL_META_KEY = "_simetri_label_meta"


def _vertices_on_hull_points(
    vertices: Sequence,
    hull_pts: Sequence | None = None,
) -> list[int]:
    """Return vertex indices that lie on the given or computed convex hull."""
    from ..geom.polygons.convex_hull import convex_hull

    verts = [tuple(v[:2]) for v in vertices]
    if len(verts) <= 1:
        return list(range(len(verts)))

    if hull_pts is None:
        hull_pts = convex_hull(verts, on_edge=True)
    else:
        hull_pts = [tuple(p[:2]) for p in hull_pts]

    ndigits = int(runtime_defaults["n_vert_digits"])
    tol = 1e-6
    indices: list[int] = []
    for i, (vx, vy) in enumerate(verts):
        for hx, hy in hull_pts:
            if hypot(vx - hx, vy - hy) <= tol:
                indices.append(i)
                break
            if (
                ndigits >= 0
                and round(vx, ndigits) == round(hx, ndigits)
                and round(vy, ndigits) == round(hy, ndigits)
            ):
                indices.append(i)
                break
    return indices


def _hull_vertex_indices(vertices: Sequence) -> list[int]:
    """Return vertex indices on this shape's own convex hull."""
    return _vertices_on_hull_points(vertices)


def sketch_requests_vertex_labels(sketch: Any) -> bool:
    """Return True if a sketch should render index or vertex-coordinate labels.

    Used by SVG/TikZ renderers and page-level label overlap resolution.

    Args:
        sketch: Sketch being drawn or inspected.

    Returns:
        bool: True when label geometry should be emitted.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.render.draw import create_sketch
        >>> canvas = sg.Canvas()
        >>> shape = sg.Shape([(0, 0), (40, 0)])
        >>> sketch = create_sketch(shape, canvas)
        >>> sketch.indices = True
        >>> sg.sketch_requests_vertex_labels(sketch)
        True
        >>> sg.sketch_requests_vertex_labels(sg.Shape([(0, 0), (40, 0)]))
        False
    """
    if not hasattr(sketch, "vertices"):
        return False
    if getattr(sketch, "indices", False):
        return True
    if getattr(sketch, "show_vertex_coords", False):
        return True
    if (
        getattr(sketch, "draw_markers", False)
        and getattr(sketch, "marker_type", None) == MarkerType.INDICES
    ):
        return True
    return False


def _iter_label_sketches(sketches: object) -> Generator[Any, None, None]:
    """Yield shape sketches that show vertex or index labels."""
    for sketch in sketches:
        subtype = getattr(sketch, "subtype", None)
        if subtype in (Types.CLIPPED_SKETCH, Types.MASKED_SKETCH):
            for sketch_list in sketch.sketches:
                yield from _iter_label_sketches(sketch_list)
        elif subtype == Types.COMPOSITE_SKETCH:
            yield from _iter_label_sketches(sketch.sketches)
        elif sketch_requests_vertex_labels(sketch):
            yield sketch


def _coord_label_vertex_indices(sketch: Any, n: int) -> list[int]:
    """Vertex indices that receive coordinate (not index) labels."""
    if getattr(sketch, "vertex_on_hull", False):
        group_hull = getattr(sketch, "_group_hull_points", None)
        return _vertices_on_hull_points(sketch.vertices, group_hull)
    return list(range(n))


def _index_label_pairs(sketch: Any, n: int) -> list[tuple[int, Any]]:
    """Return ``(vertex_index, label_value)`` for each index label."""
    raw = getattr(sketch, "indices", False)
    if not raw:
        return []
    if isinstance(raw, bool):
        return [(i, i) for i in range(n)]
    values = list(raw)
    if not values:
        raise ValueError("indices sequence must not be empty")
    if len(values) > n:
        raise ValueError(
            f"indices sequence length {len(values)} exceeds vertex count {n}"
        )
    if all(0 <= v < n for v in values):
        return [(v, v) for v in values]
    if all(v >= n for v in values):
        return [(i, values[i]) for i in range(len(values))]
    raise ValueError(
        f"indices {values} must be all in range 0..{n - 1} (label selected "
        f"vertices) or all >= {n} (custom label numbers for vertices 0..)"
    )


def _index_label_vertex_indices(sketch: Any, n: int) -> list[int]:
    """Vertex indices that receive index labels (never hull-filtered)."""
    return [vertex for vertex, _ in _index_label_pairs(sketch, n)]


def _build_shape_label_rects(sketch: Any) -> list[LabelRect]:
    """Build centered label boxes at layout anchors (no overlap pass)."""
    existing = getattr(sketch, _LABEL_RECTS_KEY, None)
    if existing is not None:
        return existing

    has_index = bool(getattr(sketch, "indices", False))
    has_vertex = bool(getattr(sketch, "show_vertex_coords", False))
    if not hasattr(sketch, "vertices"):
        return []
    vertices = sketch.vertices
    n = len(vertices)
    index_pairs = _index_label_pairs(sketch, n) if has_index else []
    index_label_indices = [vertex for vertex, _ in index_pairs]
    coord_label_indices = (
        _coord_label_vertex_indices(sketch, n) if has_vertex else []
    )

    entries: list[
        tuple[str, int, tuple[float, float], tuple[float, float]]
    ] = []
    index_labels = None
    coord_texts = None

    if has_index:
        index_offset = sketch_label_offset(sketch, "index")
        index_layout = vert_label_layout(sketch, index_offset)
        index_labels = [None] * n
        index_font = sketch_label_font_size_pt(sketch, "index")
        index_family = sketch_label_font_family(sketch, "index")
        for vertex, label in index_pairs:
            pos = index_layout[vertex]["position"]
            size = estimate_index_label_bbox(label, index_font, index_family)
            entries.append(("index", vertex, pos, size))
            index_labels[vertex] = label

    if has_vertex:
        vertex_offset = sketch_label_offset(sketch, "vertex")
        vertex_layout = vert_label_layout(sketch, vertex_offset)
        coord_texts = [format_vertex_coord(*vertices[i]) for i in range(n)]
        vertex_font = sketch_label_font_size_pt(sketch, "vertex")
        vertex_family = sketch_label_font_family(sketch, "vertex")
        for i in coord_label_indices:
            pos = vertex_layout[i]["position"]
            size = estimate_vertex_coord_label_bbox(
                coord_texts[i], vertex_font, vertex_family
            )
            entries.append(("vertex", i, pos, size))

    rects = [
        LabelRect(sketch, kind, i, pos[0], pos[1], size[0], size[1])
        for kind, i, pos, size in entries
    ]
    setattr(sketch, _LABEL_RECTS_KEY, rects)
    setattr(
        sketch,
        _LABEL_META_KEY,
        {
            "n": n,
            "index_labels": index_labels,
            "coord_texts": coord_texts,
            "has_index": has_index,
            "has_vertex": has_vertex,
            "index_label_indices": index_label_indices,
            "coord_label_indices": coord_label_indices,
            "entries": entries,
        },
    )
    return rects


def _apply_label_rects_to_sketch(sketch: Any) -> None:
    """Write label rect centers into the sketch resolved-label cache."""
    meta = getattr(sketch, _LABEL_META_KEY, None)
    rects = getattr(sketch, _LABEL_RECTS_KEY, None)
    if meta is None or rects is None:
        return

    n = meta["n"]
    has_index = meta["has_index"]
    has_vertex = meta["has_vertex"]
    index_labels = meta["index_labels"]
    coord_texts = meta["coord_texts"]
    index_label_indices = meta.get("index_label_indices", list(range(n)))
    coord_label_indices = meta.get("coord_label_indices", list(range(n)))

    index_positions = [None] * n if has_index else None
    coord_positions = [None] * n if has_vertex else None
    for rect in rects:
        pos = (rect.x, rect.y)
        if rect.kind == "index":
            index_positions[rect.vertex_index] = pos
        else:
            coord_positions[rect.vertex_index] = pos

    result: dict = {"index": None, "vertex": None}
    if has_index:
        result["index"] = (
            [index_positions[i] for i in index_label_indices],
            [index_labels[i] for i in index_label_indices],
        )
    if has_vertex:
        result["vertex"] = (
            [coord_positions[i] for i in coord_label_indices],
            [coord_texts[i] for i in coord_label_indices],
        )
    setattr(sketch, _RESOLVED_LABELS_KEY, result)


def resolve_page_vertex_labels(sketches: object) -> None:
    """Resolve overlaps for all vertex/index labels on a sketch list.

    Args:
        sketches: Sketches belonging to one page (or comparable group).

    Returns:
        None

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.resolve_page_vertex_labels([])
    """
    label_sketches = list(_iter_label_sketches(sketches))
    all_rects: list[LabelRect] = []
    for sketch in label_sketches:
        all_rects.extend(_build_shape_label_rects(sketch))

    if runtime_defaults["vertices_label_avoid_overlap"] and len(all_rects) > 1:
        gap = float(runtime_defaults["vertices_label_overlap_gap"])
        max_iters = int(runtime_defaults["vertices_label_overlap_max_iters"])
        debug = any(bool(getattr(s, "debug", False)) for s in label_sketches)
        if debug:
            print(
                f"resolve_page_vertex_labels: {len(all_rects)} labels, "
                f"gap={gap}, max_iters={max_iters}"
            )
        resolve_all_overlaps(all_rects, gap=gap, max_iters=max_iters)

    for sketch in label_sketches:
        _apply_label_rects_to_sketch(sketch)


def _label_rect_outline_shape(rect: LabelRect) -> Shape:
    """Canvas-space outline for one overlap-resolution ``LabelRect``."""
    half_w = rect.width / 2
    half_h = rect.height / 2
    return Shape(
        [
            (rect.x - half_w, rect.y - half_h),
            (rect.x + half_w, rect.y - half_h),
            (rect.x + half_w, rect.y + half_h),
            (rect.x - half_w, rect.y + half_h),
        ],
        closed=True,
    )


def _draw_kind_label_bboxes(
    canvas: Any,
    kind: str,
    *,
    line_color: colors.Color,
    line_width: float,
    resolve: bool,
) -> Any:
    sketches = canvas.active_page.sketches
    if resolve:
        resolve_page_vertex_labels(sketches)
    for sketch in _iter_label_sketches(sketches):
        rects = getattr(sketch, _LABEL_RECTS_KEY, None)
        if not rects:
            continue
        for rect in rects:
            if rect.kind != kind:
                continue
            canvas.draw(
                _label_rect_outline_shape(rect),
                fill=False,
                line_color=line_color,
                line_width=line_width,
            )
    return canvas


def draw_index_label_bboxes(
    canvas: Any,
    *,
    line_color: colors.Color | None = None,
    line_width: float = 0.5,
    resolve: bool = True,
) -> Any:
    """Draw overlap boxes for index labels on ``canvas`` (maintainer debug).

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.draw(sg.Shape([(0, 0), (40, 0), (0, 40)], closed=True), indices=True)
        Canvas()
        >>> sg.draw_index_label_bboxes(canvas)
        Canvas()
        >>> len(canvas.active_page.sketches)
        4
    """
    if line_color is None:
        line_color = colors.red
    return _draw_kind_label_bboxes(
        canvas,
        "index",
        line_color=line_color,
        line_width=line_width,
        resolve=resolve,
    )


def draw_vertex_label_bboxes(
    canvas: Any,
    *,
    line_color: colors.Color | None = None,
    line_width: float = 0.5,
    resolve: bool = True,
) -> Any:
    """Draw overlap boxes for vertex coordinate labels (maintainer debug).

    Examples:
        >>> import simetri.graphics as sg
        >>> canvas = sg.Canvas()
        >>> canvas.draw(
        ...     sg.Shape([(0, 0), (40, 0), (0, 40)], closed=True),
        ...     show_vertex_coords=True,
        ... )
        Canvas()
        >>> sg.draw_vertex_label_bboxes(canvas)
        Canvas()
        >>> len(canvas.active_page.sketches)
        4
    """
    if line_color is None:
        line_color = colors.teal
    return _draw_kind_label_bboxes(
        canvas,
        "vertex",
        line_color=line_color,
        line_width=line_width,
        resolve=resolve,
    )


def _resolve_shape_labels(sketch: Any) -> dict:
    """Return cached label layout for a shape sketch."""
    cached = getattr(sketch, _RESOLVED_LABELS_KEY, None)
    if cached is not None:
        return cached

    resolve_page_vertex_labels([sketch])
    return getattr(sketch, _RESOLVED_LABELS_KEY)


def prepare_shape_index_labels(
    sketch: Any,
) -> tuple[list[tuple[float, float]], list] | None:
    """Return index label positions and values for a shape sketch.

    Uses ``index_offset`` from the sketch or defaults. When coordinate labels
    are also shown, overlap resolution considers both label types together.

    Args:
        sketch: Shape sketch that may request index labels.

    Returns:
        tuple | None: ``(positions, labels)`` or ``None`` if indices are off.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.prepare_shape_index_labels(sg.Shape([(0, 0), (40, 0), (0, 40)]))
    """
    if not getattr(sketch, "indices", False):
        return None
    if not hasattr(sketch, "vertices"):
        return None
    return _resolve_shape_labels(sketch)["index"]


def prepare_shape_vertex_coord_labels(
    sketch: Any,
) -> tuple[list[tuple[float, float]], list[str]] | None:
    """Return vertex coordinate label positions and texts for a shape sketch.

    Uses ``vertex_offset`` from the sketch or defaults. When index labels are
    also shown, overlap resolution considers both label types together.

    Args:
        sketch: Shape sketch that may request coordinate labels.

    Returns:
        tuple | None: ``(positions, texts)`` or ``None`` if coords are off.

    Examples:
        >>> import simetri.graphics as sg
        >>> sg.prepare_shape_vertex_coord_labels(sg.Shape([(0, 0), (40, 0), (0, 40)]))
    """
    if not getattr(sketch, "show_vertex_coords", False):
        return None
    if not hasattr(sketch, "vertices"):
        return None
    return _resolve_shape_labels(sketch)["vertex"]


def edge_label_positions(shape: Shape, offset: float) -> list:
    """Return edge-label positions using the given radial offset.

    Args:
        shape: Shape whose edges are labeled.
        offset: Distance from each edge midpoint to the label.

    Returns:
        list: Label positions for each edge.

    Examples:
        >>> import simetri.graphics as sg
        >>> shape = sg.Shape([(0, 0), (40, 0), (0, 40)])
        >>> sg.edge_label_positions(shape, 2.0)
        [[20.0, -2.0], [21.414213562373096, 21.414213562373096]]
        >>> closed = sg.Shape([(0, 0), (40, 0), (0, 40)], closed=True)
        >>> sg.edge_label_positions(closed, 2.0)
        [[20.0, -2.0], [21.414213562373096, 21.414213562373096], [-2.0, 20.0]]
    """
    from simetri.geom.polygons.polygon import in_polygon

    vertices = list(shape.vertices)
    count = len(vertices)
    num_edges = count if shape.closed else count - 1

    # Initialize with edge vector for edge 0
    edge_vec = v_from_points(vertices[0][:2], vertices[1][:2])
    positions = []
    for i in range(num_edges):
        prev_point = vertices[i][:2]
        next_point = vertices[(i + 1) % count][:2]
        point = midpoint(prev_point, next_point)

        mid_vec = Vector(point)
        direction = edge_vec.perp().normalize()

        test_point = mid_vec + direction
        if in_polygon(test_point, vertices):
            pos = mid_vec - direction * offset
        else:
            pos = mid_vec + direction * offset

        positions.append(pos[:])

        # Compute edge vector for next iteration (if there is one)
        if i < num_edges - 1:
            edge_vec = v_from_points(next_point, vertices[(i + 2) % count][:2])

    return positions


def edge_label_pos(
    shape: Shape, index: int, offset: float = 10
) -> tuple[float, float]:
    """Return the edge label position for ``index`` at ``offset`` from the edge.

    Examples:
        >>> import simetri.graphics as sg
        >>> shape = sg.Shape([(0, 0), (40, 0), (0, 40)])
        >>> sg.edge_label_pos(shape, 0, 2.0)
        (20.0, -2.0)
    """
    from simetri.geom.polygons.polygon import in_polygon

    vertices = shape.vertices
    count = len(vertices)
    prev_point = vertices[index][:2]
    next_point = vertices[(index + 1) % count][:2]
    point = midpoint(prev_point, next_point)

    vec1 = v_from_points(point, next_point)
    edge_vec = Vector(point)

    direction = vec1.perp().normalize()

    test_point = edge_vec + direction
    if in_polygon(test_point, shape.vertices):
        pos = edge_vec - direction * offset
    else:
        pos = edge_vec + direction * offset

    return (pos.x, pos.y)

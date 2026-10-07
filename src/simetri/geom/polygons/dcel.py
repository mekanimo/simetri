"""Doubly connected edge list (DCEL) / half-edge data structure helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from math import inf, isclose

from ...base.all_enums import Connection, Types
from ...base.common import LineType, PointType, resolve_tol
from ...group.batch import Group
from ...shapes.points import Points
from ...shapes.shape import Shape
from ..bbox import BoundingBox, bounding_box
from ..points.point_utils import distance, on_segment
from ..segments.line_utils import (
    intersect,
    intersection as segment_intersection,
)
from .polygon import in_polygon, polygon_area
from .polygon_utils import is_simple


class Vertex(Shape):
    """A 2D vertex in a DCEL, with one outgoing half-edge.

    Attributes:
        half_edge: One outgoing half-edge from this vertex.
    """

    def __init__(self, x: float, y: float) -> None:
        """Create a vertex at ``(x, y)``.

        Args:
            x: X coordinate.
            y: Y coordinate.
        """
        super().__init__([(x, y)], closed=False, subtype=Types.VERTEX)
        self.half_edge = None
        self.subtype = Types.VERTEX

    @property
    def x(self) -> float:
        """X coordinate."""
        return float(self.vertices[0][0])

    @property
    def y(self) -> float:
        """Y coordinate."""
        return float(self.vertices[0][1])

    @property
    def point(self) -> tuple[float, float]:
        """Return ``(x, y)``."""
        vertex = self.vertices[0]
        return (float(vertex[0]), float(vertex[1]))

    def __str__(self) -> str:
        """Return ``Vertex((x, y))``."""
        return f"Vertex({self.point})"

    def __repr__(self) -> str:
        """Return ``Vertex((x, y))``."""
        return self.__str__()

    def outgoing_halfedges(self) -> Iterator[HalfEdge]:
        """Yield outgoing half-edges around this vertex (mutated: none).

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (40, 0), (40, 40), (0, 40)]])
            >>> origin = [vertex for vertex in mesh.vertices if vertex.point == (0.0, 0.0)][0]
            >>> len(list(origin.outgoing_halfedges()))
            2
        """
        start = self.half_edge
        if start is None:
            return
        half_edge = start
        while True:
            if half_edge.twin is None:
                raise ValueError("half-edge has no twin")
            yield half_edge
            half_edge = half_edge.twin.next
            if half_edge is None:
                raise ValueError("half-edge twin.next is None")
            if half_edge is start:
                break

    def incident_faces(self) -> list[Face]:
        """Return unique faces incident to this vertex.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (40, 0), (40, 40), (0, 40)]])
            >>> origin = [vertex for vertex in mesh.vertices if vertex.point == (0.0, 0.0)][0]
            >>> len(origin.incident_faces())
            2
        """
        faces = []
        for half_edge in self.outgoing_halfedges():
            if half_edge.face not in faces:
                faces.append(half_edge.face)
        return faces

    def copy(self) -> Vertex:
        """Return a geometric copy of this vertex.

        The copy keeps the position and style. It is not linked into the
        mesh: ``half_edge`` is None.

        Returns:
            Vertex: The copy.

        Examples:
            >>> import simetri.graphics as sg
            >>> mesh = sg.DCEL()
            >>> _ = mesh.build_from_polygons(
            ...     [[(0, 0), (40, 0), (40, 40), (0, 40)]]
            ... )
            >>> vertex = mesh.vertices[0]
            >>> copied = vertex.copy()
            >>> copied is vertex
            False
            >>> copied.half_edge is None
            True
            >>> copied.point == vertex.point
            True
            >>> vertex.half_edge is None
            False
        """
        saved_half_edge = self.half_edge
        self.half_edge = None
        try:
            copied = super().copy()
        finally:
            self.half_edge = saved_half_edge
        return copied


class Face(Shape):
    """A face in a DCEL, referenced by one bounding half-edge.

    Attributes:
        half_edge: One half-edge on the outer boundary.
        inner_boundaries: Start half-edges of hole cycles.
        mesh: Owning ``DCEL``, or ``None`` if unlinked.
    """

    def __init__(self) -> None:
        """Create an empty face with no half-edge yet."""
        super().__init__(closed=True, subtype=Types.FACE)
        self.half_edge = None
        self.inner_boundaries: list[HalfEdge] = []
        self.mesh: DCEL | None = None
        self.subtype = Types.FACE

    def __str__(self) -> str:
        """Return a Face string using the outer-cycle vertices."""
        n = len(self.primary_points)
        if n == 0:
            return "Face()"
        if n < 4:
            return f"Face({self.vertices})"
        return f"Face([{self.vertices[0]}, ..., {self.vertices[-1]}])"

    def __repr__(self) -> str:
        """Return a Face string using the outer-cycle vertices."""
        return self.__str__()

    def copy(self) -> Face:
        """Return a geometric copy of this face.

        The copy keeps the vertices and style. It is not linked into the
        mesh: ``half_edge`` is None, ``inner_boundaries`` is empty, and
        ``mesh`` is None.

        Returns:
            Face: The copy.

        Examples:
            >>> import simetri.graphics as sg
            >>> mesh = sg.DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (40, 0), (40, 40), (0, 40)]])
            >>> face = mesh.bounded_faces[0]
            >>> copied = face.copy()
            >>> copied is face
            False
            >>> copied.half_edge is None
            True
            >>> [tuple(point[:2]) for point in copied.vertices] == [
            ...     tuple(point[:2]) for point in face.vertices
            ... ]
            True
            >>> face.half_edge is None
            False
        """
        saved_half_edge = self.half_edge
        saved_inner = self.inner_boundaries
        saved_mesh = self.mesh
        self.half_edge = None
        self.inner_boundaries = []
        self.mesh = None
        try:
            copied = super().copy()
        finally:
            self.half_edge = saved_half_edge
            self.inner_boundaries = saved_inner
            self.mesh = saved_mesh
        copied.mesh = None
        return copied

    def add_hole(
        self,
        polygon: Shape | Sequence[PointType],
        abs_tol: float | None = None,
        rel_tol: float | None = None,
    ) -> Face:
        """Add a hole polygon inside this face (mutated).

        Args:
            polygon: Hole as a ``Shape`` or vertex ring.
            abs_tol: Absolute tolerance. ``None`` uses runtime defaults.
            rel_tol: Relative tolerance. ``None`` uses runtime defaults.

        Returns:
            Face: This face.

        Examples:
            >>> import simetri.graphics as sg
            >>> mesh = sg.DCEL()
            >>> face = mesh.add_face(
            ...     [(0, 0), (80, 0), (80, 80), (0, 80)]
            ... )
            >>> _ = face.add_hole([(20, 20), (60, 20), (60, 60), (20, 60)])
            >>> len(face.inner_boundaries)
            1
            >>> mesh.contains(face, (40, 40))
            False
            >>> mesh.validate()
            True
        """
        if self.mesh is None:
            raise ValueError("face is not linked to a DCEL")
        self.mesh._add_hole(self, polygon, abs_tol=abs_tol, rel_tol=rel_tol)
        return self

    def remove_hole(
        self,
        polygon: Shape | Sequence[PointType],
        abs_tol: float | None = None,
        rel_tol: float | None = None,
    ) -> Face:
        """Remove a hole polygon from this face (mutated).

        Args:
            polygon: Hole as a ``Shape`` or vertex ring (same geometry as
                when added).
            abs_tol: Absolute tolerance. ``None`` uses runtime defaults.
            rel_tol: Relative tolerance. ``None`` uses runtime defaults.

        Returns:
            Face: This face.

        Examples:
            >>> import simetri.graphics as sg
            >>> mesh = sg.DCEL()
            >>> face = mesh.add_face(
            ...     [(0, 0), (80, 0), (80, 80), (0, 80)]
            ... )
            >>> hole = [(20, 20), (60, 20), (60, 60), (20, 60)]
            >>> _ = face.add_hole(hole)
            >>> _ = face.remove_hole(hole)
            >>> face.inner_boundaries
            []
            >>> mesh.validate()
            True
        """
        if self.mesh is None:
            raise ValueError("face is not linked to a DCEL")
        self.mesh._remove_hole(self, polygon, abs_tol=abs_tol, rel_tol=rel_tol)
        return self

    def _load_boundary(self, start: HalfEdge | None = None) -> None:
        """Load Shape vertices from the outer cycle at ``start``."""
        cycle_start = self.half_edge if start is None else start
        if cycle_start is None:
            _set_shape_points(self, [], True)
            return
        _set_shape_points(self, _cycle_points(cycle_start), True)

    @property
    def outer_boundary(self) -> HalfEdge | None:
        """Start half-edge of the outer cycle."""
        return self.half_edge

    def boundary_halfedges(self) -> Iterator[HalfEdge]:
        """Yield half-edges of the outer boundary cycle.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (40, 0), (40, 40), (0, 40)]])
            >>> interior = mesh.bounded_faces[0]
            >>> len(list(interior.boundary_halfedges()))
            4
        """
        yield from _cycle_halfedges(self.half_edge)

    def boundary_vertices(self) -> list[Vertex]:
        """Return origin vertices of the outer boundary cycle.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (40, 0), (40, 40), (0, 40)]])
            >>> interior = mesh.bounded_faces[0]
            >>> sorted(vertex.point for vertex in interior.boundary_vertices())
            [(0.0, 0.0), (0.0, 40.0), (40.0, 0.0), (40.0, 40.0)]
        """
        return [half_edge.origin for half_edge in self.boundary_halfedges()]


class HalfEdge(Shape):
    """Directed half-edge linking vertices, opposite edge, and face.

    Attributes:
        vertex: Destination vertex of this half-edge.
        pair: Opposite half-edge (twin).
        next: Next half-edge around the face.
        prev: Previous half-edge around the face.
        face: Face on the left side of this half-edge.
    """

    def __init__(self) -> None:
        """Create an unlinked half-edge."""
        super().__init__(closed=False, subtype=Types.HALF_EDGE)
        self.vertex = None
        self.pair = None
        self.next = None
        self.prev = None
        self.face = None
        self.subtype = Types.HALF_EDGE

    def __str__(self) -> str:
        """Return ``HalfEdge(origin, destination)`` when endpoints are set."""
        origin = self.origin
        destination = self.destination
        if origin is None or destination is None:
            return "HalfEdge()"
        return f"HalfEdge({origin.point}, {destination.point})"

    def __repr__(self) -> str:
        """Return ``HalfEdge(origin, destination)`` when endpoints are set."""
        return self.__str__()

    def copy(self) -> HalfEdge:
        """Return a geometric copy of this half-edge.

        The copy keeps the segment vertices and style. It is not linked
        into the mesh: ``vertex``, ``pair``, ``next``, ``prev``, and
        ``face`` are None.

        Returns:
            HalfEdge: The copy.

        Examples:
            >>> import simetri.graphics as sg
            >>> mesh = sg.DCEL()
            >>> _ = mesh.build_from_polygons(
            ...     [[(0, 0), (40, 0), (40, 40), (0, 40)]]
            ... )
            >>> half_edge = mesh.half_edges[0]
            >>> copied = half_edge.copy()
            >>> copied is half_edge
            False
            >>> copied.pair is None
            True
            >>> copied.next is None
            True
            >>> copied.face is None
            True
            >>> half_edge.pair is None
            False
        """
        saved_vertex = self.vertex
        saved_pair = self.pair
        saved_next = self.next
        saved_prev = self.prev
        saved_face = self.face
        self.vertex = None
        self.pair = None
        self.next = None
        self.prev = None
        self.face = None
        try:
            copied = super().copy()
        finally:
            self.vertex = saved_vertex
            self.pair = saved_pair
            self.next = saved_next
            self.prev = saved_prev
            self.face = saved_face
        return copied

    def _load_segment(self) -> None:
        """Load Shape vertices from origin and destination."""
        origin = self.origin
        destination = self.destination
        if origin is None or destination is None:
            return
        _set_shape_points(self, [origin.point, destination.point], False)

    @property
    def destination(self) -> Vertex | None:
        """Destination vertex."""
        return self.vertex

    @destination.setter
    def destination(self, value: Vertex | None) -> None:
        self.vertex = value

    @property
    def twin(self) -> HalfEdge | None:
        """Opposite half-edge."""
        return self.pair

    @twin.setter
    def twin(self, value: HalfEdge | None) -> None:
        self.pair = value

    @property
    def origin(self) -> Vertex | None:
        """Origin vertex (twin destination, else previous destination)."""
        if self.twin is not None:
            return self.twin.destination
        if self.prev is None:
            return None
        return self.prev.vertex


class Edge(Shape):
    """Undirected edge represented by a twin half-edge pair.

    Attributes:
        half_edge: One of the two twin half-edges.v
    """

    def __init__(self, half_edge: HalfEdge) -> None:
        """Wrap a half-edge as an undirected edge.

        Args:
            half_edge: One directed side of the edge.
        """
        origin = half_edge.origin
        destination = half_edge.destination
        if origin is not None and destination is not None:
            super().__init__(
                [origin.point, destination.point],
                closed=False,
                subtype=Types.EDGE,
            )
        else:
            super().__init__(closed=False, subtype=Types.EDGE)
        self.half_edge = half_edge
        self.subtype = Types.EDGE

    def __str__(self) -> str:
        """Return ``Edge((origin, destination))`` when endpoints are set."""
        origin = self.origin
        destination = self.destination
        if origin is None or destination is None:
            return "Edge()"
        return f"Edge({(origin.point, destination.point)})"

    def __repr__(self) -> str:
        """Return ``Edge((origin, destination))`` when endpoints are set."""
        return self.__str__()

    def copy(self) -> Edge:
        """Return a geometric copy of this edge.

        The copy keeps the segment vertices and style. It is not linked
        into the mesh: ``half_edge`` is None.

        Returns:
            Edge: The copy.

        Examples:
            >>> import simetri.graphics as sg
            >>> mesh = sg.DCEL()
            >>> _ = mesh.build_from_polygons(
            ...     [[(0, 0), (40, 0), (40, 40), (0, 40)]]
            ... )
            >>> edge = mesh.edges[0]
            >>> copied = edge.copy()
            >>> copied is edge
            False
            >>> copied.half_edge is None
            True
            >>> [tuple(point[:2]) for point in copied.vertices] == [
            ...     tuple(point[:2]) for point in edge.vertices
            ... ]
            True
            >>> edge.half_edge is None
            False
        """
        saved_half_edge = self.half_edge
        self.half_edge = None
        try:
            copied = super().copy()
        finally:
            self.half_edge = saved_half_edge
        return copied

    def _load_segment(self) -> None:
        """Load Shape vertices from the current endpoints."""
        origin = self.origin
        destination = self.destination
        if origin is None or destination is None:
            return
        _set_shape_points(self, [origin.point, destination.point], False)

    def adjacent_faces(self) -> tuple[Face | None, Face | None]:
        """Return the faces on either side of this edge.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (40, 0), (40, 40), (0, 40)]])
            >>> edge = mesh.edges[0]
            >>> len(edge.adjacent_faces())
            2
        """
        half_edge = self.half_edge
        if half_edge.twin is None:
            raise ValueError("edge has no twin")
        return (half_edge.face, half_edge.twin.face)

    @property
    def origin(self) -> Vertex | None:
        """Origin of ``half_edge``."""
        return self.half_edge.origin

    @property
    def destination(self) -> Vertex | None:
        """Destination of ``half_edge``."""
        return self.half_edge.destination

    @property
    def segment(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Endpoint coordinates ``(origin, destination)``."""
        origin = self.origin
        destination = self.destination
        if origin is None or destination is None:
            raise ValueError("edge endpoints are not set")
        return (origin.point, destination.point)


@dataclass
class SplitEdgeResult:
    """Result of ``split_edge``.

    Attributes:
        vertex: Inserted vertex.
        first_edge: Edge from the original origin to the new vertex.
        second_edge: Edge from the new vertex to the original destination.
    """

    vertex: Vertex
    first_edge: Edge
    second_edge: Edge


@dataclass
class SplitFaceResult:
    """Result of ``split_face``.

    Attributes:
        edge: New chord.
        first_face: Face that kept the original object.
        second_face: Newly created face.
    """

    edge: Edge
    first_face: Face
    second_face: Face


@dataclass
class SplitResult:
    """Result of a geometric cut that may produce several faces.

    Attributes:
        faces: Faces after the cut.
        cut_edges: Edges created along the cut.
        inserted_vertices: Vertices inserted on existing edges.
    """

    faces: list[Face] = field(default_factory=list)
    cut_edges: list[Edge] = field(default_factory=list)
    inserted_vertices: list[Vertex] = field(default_factory=list)


def _cycle_halfedges(start: HalfEdge | None) -> Iterator[HalfEdge]:
    """Yield half-edges of a cycle starting at ``start``."""
    if start is None:
        raise ValueError("cycle has no start half-edge")
    half_edge = start
    seen: set[int] = set()
    while True:
        if id(half_edge) in seen:
            raise ValueError("half-edge cycle is not closed")
        seen.add(id(half_edge))
        yield half_edge
        half_edge = half_edge.next
        if half_edge is None:
            raise ValueError("half-edge next is None")
        if half_edge is start:
            break


def _cycle_points(start: HalfEdge | None) -> list[tuple[float, float]]:
    """Return origin coordinates of a half-edge cycle."""
    return [half_edge.origin.point for half_edge in _cycle_halfedges(start)]


def _set_shape_points(
    shape: Shape, points: Sequence[PointType], closed: bool
) -> None:
    """Replace ``shape`` geometry with ``points``."""
    if not points:
        shape.primary_points = Points()
        shape.closed = closed
        shape.primary_points.nd_array_changed = True
        return
    closed, points = shape._get_closed(list(points), closed)
    shape.primary_points = Points(points)
    shape.closed = closed
    shape.primary_points.nd_array_changed = True


def _same_point(
    point_a: PointType, point_b: PointType, abs_tol: float | None = None
) -> bool:
    """Return True if ``point_a`` and ``point_b`` are within ``abs_tol``."""
    _, abs_tol = resolve_tol(None, abs_tol)
    return distance(point_a[:2], point_b[:2]) <= abs_tol


def _segment_param(start: PointType, end: PointType, point: PointType) -> float:
    """Return a scalar parameter of ``point`` along ``start``→``end``."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) >= abs(dy):
        if isclose(dx, 0.0):
            return 0.0
        return (point[0] - start[0]) / dx
    if isclose(dy, 0.0):
        return 0.0
    return (point[1] - start[1]) / dy


def _midpoint(point_a: PointType, point_b: PointType) -> tuple[float, float]:
    """Return the midpoint of two points."""
    return ((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0)


def _polygon_centroid(vertices: Sequence[PointType]) -> tuple[float, float]:
    """Return the centroid of a polygon ring."""
    area = polygon_area(vertices)
    if isclose(area, 0.0):
        n_vertices = len(vertices)
        if n_vertices == 0:
            raise ValueError("polygon has no vertices")
        x_sum = sum(point[0] for point in vertices)
        y_sum = sum(point[1] for point in vertices)
        return (x_sum / n_vertices, y_sum / n_vertices)
    cx = 0.0
    cy = 0.0
    points = list(vertices)
    if points[0][:2] != points[-1][:2]:
        points.append(points[0])
    for i, point in enumerate(points[:-1]):
        x1, y1 = point[:2]
        x2, y2 = points[i + 1][:2]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    factor = 1.0 / (6.0 * area)
    return (cx * factor, cy * factor)


def get_face_vertices(
    face: Face, repeat_start: bool = False
) -> list[tuple[float, float]]:
    """Return origin coordinates walking around ``face``.

    Args:
        face: A ``Face`` whose ``half_edge`` cycle is complete.
        repeat_start: If True, append a copy of the first point.

    Returns:
        ``(x, y)`` origin vertices in boundary order.

    Examples:
        >>> from simetri.geom.polygons.dcel import create_square_patch, get_face_vertices
        >>> face, _ = create_square_patch()
        >>> get_face_vertices(face)
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        >>> get_face_vertices(face, repeat_start=True)[-1]
        (0.0, 0.0)
    """
    coords = []
    start_he = face.half_edge
    he = start_he
    while True:
        # The vertex field stores the target of the half-edge
        # So the origin of 'he' is he.prev.vertex
        origin_vertex = he.prev.vertex
        coords.append((origin_vertex.x, origin_vertex.y))
        he = he.next
        if he is start_he:
            break
    if repeat_start:
        coords.append(coords[0])
    return coords


def create_square_patch() -> tuple[Face, list[Vertex]]:
    """Build a unit-square DCEL face for testing / illustration.

    Returns:
        ``(face, vertices)`` for the square ``[0, 1] × [0, 1]``.

    Examples:
        >>> from simetri.geom.polygons.dcel import create_square_patch
        >>> face, vertices = create_square_patch()
        >>> [(v.x, v.y) for v in vertices]
        [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    """
    # 1. Create vertices
    v0 = Vertex(0.0, 0.0)
    v1 = Vertex(1.0, 0.0)
    v2 = Vertex(1.0, 1.0)
    v3 = Vertex(0.0, 1.0)

    vertices = [v0, v1, v2, v3]

    # 2. Create half-edges for a single square face
    he0 = HalfEdge()
    he1 = HalfEdge()
    he2 = HalfEdge()
    he3 = HalfEdge()

    # Link face
    face = Face()
    face.half_edge = he0

    # Wire next/prev and vertex targets
    he0.vertex, he0.next, he0.prev, he0.face = v1, he1, he3, face
    he1.vertex, he1.next, he1.prev, he1.face = v2, he2, he0, face
    he2.vertex, he2.next, he2.prev, he2.face = v3, he3, he1, face
    he3.vertex, he3.next, he3.prev, he3.face = v0, he0, he2, face

    # Attach outgoing pointers to vertices
    v0.half_edge = he0
    v1.half_edge = he1
    v2.half_edge = he2
    v3.half_edge = he3

    return face, vertices


class DCEL(Group):
    """Planar subdivision as a doubly connected edge list.

    The empty mesh has one unbounded outer face. Primitive edits preserve
    twin, ``next``/``prev``, and face-cycle invariants.

    Examples:
        >>> import simetri.graphics as sg
        >>> mesh = sg.DCEL()
        >>> mesh.build_from_polygons([[(0, 0), (40, 0), (40, 40), (0, 40)]]) is mesh
        True
        >>> len(mesh.bounded_faces)
        1
        >>> mesh.validate()
        True
        >>> square = sg.Shape([(0, 0), (80, 0), (80, 80), (0, 80)], closed=True)
        >>> from_shape = sg.DCEL(polygons=square)
        >>> len(from_shape.bounded_faces)
        1
        >>> from_shape.validate()
        True
    """

    def __init__(
        self,
        polygons: (
            Shape
            | Group
            | Sequence[Shape]
            | Sequence[Sequence[PointType]]
            | None
        ) = None,
    ) -> None:
        """Create a DCEL, optionally built from polygons.

        Args:
            polygons: Optional polygon source: one ``Shape``, a ``Group`` of
                polygons, a sequence of shapes, or a sequence of vertex rings.
        """
        self.vertices: list[Vertex] = []
        self.half_edges: list[HalfEdge] = []
        self.edges: list[Edge] = []
        self.outer_face = Face()
        self.faces: list[Face] = [self.outer_face]
        super().__init__(subtype=Types.DCEL)
        self.elements = self.faces
        self.outer_face.mesh = self
        if polygons is not None:
            self.build_from_polygons(_polygon_rings(polygons))

    def __str__(self) -> str:
        """Return a summary ``DCEL(...)`` string."""
        return (
            f"DCEL(vertices={len(self.vertices)}, "
            f"edges={len(self.edges)}, "
            f"faces={len(self.faces)})"
        )

    def __repr__(self) -> str:
        """Return a summary ``DCEL(...)`` string."""
        return self.__str__()

    def copy(self) -> DCEL:
        """Return a geometric copy of this mesh.

        Faces, vertices, half-edges, and edges are copied. The copies are
        not linked to each other or to this mesh. ``elements`` and
        ``faces`` stay the same list, as on a new mesh.

        Returns:
            DCEL: The copy.

        Examples:
            >>> import simetri.graphics as sg
            >>> mesh = sg.DCEL()
            >>> _ = mesh.build_from_polygons(
            ...     [[(0, 0), (40, 0), (40, 40), (0, 40)]]
            ... )
            >>> copied = mesh.copy()
            >>> copied is mesh
            False
            >>> copied.elements is copied.faces
            True
            >>> len(copied.bounded_faces) == len(mesh.bounded_faces)
            True
            >>> copied.edges[0] is mesh.edges[0]
            False
            >>> copied.edges[0].half_edge is None
            True
            >>> mesh.edges[0].half_edge is None
            False
        """
        copied = super().copy()
        copied.vertices = [vertex.copy() for vertex in self.vertices]
        copied.half_edges = [half_edge.copy() for half_edge in self.half_edges]
        copied.edges = [edge.copy() for edge in self.edges]
        copied.faces = list(copied.elements)
        copied.elements = copied.faces
        copied.outer_face = copied.faces[0]
        for face in copied.faces:
            face.mesh = copied
        return copied

    @property
    def bounded_faces(self) -> list[Face]:
        """Faces other than the unbounded outer face."""
        return [face for face in self.faces if face is not self.outer_face]

    def add_vertex(
        self, point: PointType, abs_tol: float | None = None
    ) -> Vertex:
        """Add a vertex at ``point``, reusing one already at that location.

        Args:
            point: Coordinates ``(x, y)``.
            abs_tol: Absolute tolerance. ``None`` uses runtime defaults.

        Returns:
            Vertex: New or existing vertex.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> vertex = mesh.add_vertex((0, 0))
            >>> vertex.point
            (0.0, 0.0)
            >>> mesh.add_vertex((0, 0)) is vertex
            True
        """
        x, y = point[:2]
        location = (float(x), float(y))
        existing = self._vertex_at(location, abs_tol=abs_tol)
        if existing is not None:
            return existing
        vertex = Vertex(location[0], location[1])
        self.vertices.append(vertex)
        return vertex

    def add_edge(
        self,
        vertex_a: Vertex,
        vertex_b: Vertex,
        face: Face | None = None,
    ) -> Edge:
        """Insert an edge between two vertices (mutated).

        Isolated endpoints grow a slit. Two vertices on the same face
        split that face (MEF).

        Args:
            vertex_a: First endpoint.
            vertex_b: Second endpoint.
            face: Face that should own a slit or receive the chord.
                Defaults to the unbounded face when both vertices are new.

        Returns:
            Edge: The new undirected edge.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> a = mesh.add_vertex((0, 0))
            >>> b = mesh.add_vertex((40, 0))
            >>> edge = mesh.add_edge(a, b)
            >>> edge.segment
            ((0.0, 0.0), (40.0, 0.0))
        """
        if vertex_a is vertex_b:
            raise ValueError("Cannot add a self-loop")
        if self.edge_between(vertex_a, vertex_b) is not None:
            raise ValueError("Edge already exists")
        isolated_a = vertex_a.half_edge is None
        isolated_b = vertex_b.half_edge is None
        if isolated_a and isolated_b:
            return self._mev_slit(vertex_a, vertex_b, face)
        if isolated_a:
            return self._mev_from(vertex_b, vertex_a, face)
        if isolated_b:
            return self._mev_from(vertex_a, vertex_b, face)
        host = self._common_face(vertex_a, vertex_b, face)
        return self.split_face(host, vertex_a, vertex_b).edge

    def connect_vertices(
        self,
        vertex_a: Vertex,
        vertex_b: Vertex,
        face: Face | None = None,
    ) -> Edge:
        """Insert an edge between two vertices on a face (mutated).

        Args:
            vertex_a: First endpoint.
            vertex_b: Second endpoint.
            face: Face that contains both vertices.

        Returns:
            Edge: The new edge.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (80, 0), (80, 80), (0, 80)]])
            >>> interior = mesh.bounded_faces[0]
            >>> va = [v for v in mesh.vertices if v.point == (0.0, 0.0)][0]
            >>> vb = [v for v in mesh.vertices if v.point == (80.0, 80.0)][0]
            >>> edge = mesh.connect_vertices(va, vb, face=interior)
            >>> len(mesh.bounded_faces)
            2
        """
        return self.add_edge(vertex_a, vertex_b, face=face)

    def build_from_polygons(
        self, polygons: Sequence[Sequence[PointType]]
    ) -> DCEL:
        """Insert closed rings into this mesh (mutated).

        Shared locations reuse vertices. Each ring is inserted into the
        face that contains its first edge.

        Args:
            polygons: Vertex rings. A repeated closing point is ignored.

        Returns:
            DCEL: This mesh.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (40, 0), (40, 40), (0, 40)]])
            >>> len(mesh.edges)
            4
            >>> mesh.validate()
            True
        """
        for polygon in polygons:
            self._insert_polygon_ring(_closed_ring(polygon))
        return self

    def add_face(
        self,
        polygon: Shape | Sequence[PointType],
        abs_tol: float | None = None,
        rel_tol: float | None = None,
    ) -> Face:
        """Insert one polygon as a new bounded face (mutated).

        Args:
            polygon: Face as a ``Shape`` or vertex ring.
            abs_tol: Absolute tolerance. ``None`` uses runtime defaults.
            rel_tol: Relative tolerance. ``None`` uses runtime defaults.

        Returns:
            Face: The new bounded face.

        Examples:
            >>> import simetri.graphics as sg
            >>> mesh = sg.DCEL()
            >>> face = mesh.add_face([(0, 0), (80, 0), (80, 60), (0, 60)])
            >>> len(mesh.bounded_faces)
            1
            >>> mesh.face_area(face)
            4800.0
            >>> mesh.validate()
            True
        """
        rings = _polygon_rings(polygon)
        if len(rings) != 1:
            raise ValueError("add_face expects a single polygon")
        before = {id(face) for face in self.faces}
        _, abs_tol = resolve_tol(rel_tol, abs_tol)
        self._insert_polygon_ring(
            _closed_ring(rings[0], abs_tol=abs_tol), abs_tol=abs_tol
        )
        created = [face for face in self.faces if id(face) not in before]
        if len(created) != 1:
            raise ValueError("polygon did not create a single new face")
        created[0].mesh = self
        return created[0]

    def _insert_polygon_ring(
        self,
        ring: Sequence[PointType],
        abs_tol: float | None = None,
    ) -> None:
        """Insert one closed vertex ring into the mesh (mutated)."""
        vertices = [self.add_vertex(point, abs_tol=abs_tol) for point in ring]
        n_vertices = len(vertices)
        if n_vertices < 2:
            raise ValueError("polygon must have at least two vertices")
        for i in range(n_vertices):
            start = vertices[i]
            end = vertices[(i + 1) % n_vertices]
            if self.edge_between(start, end) is not None:
                continue
            host = self._face_for_new_edge(start, end)
            self.add_edge(start, end, face=host)

    def _add_hole(
        self,
        face: Face,
        polygon: Shape | Sequence[PointType],
        abs_tol: float | None = None,
        rel_tol: float | None = None,
    ) -> None:
        """Insert ``polygon`` as a hole cycle of ``face`` (mutated)."""
        if face is self.outer_face:
            raise ValueError("cannot add a hole to the unbounded face")
        if face not in self.faces:
            raise ValueError("face is not in this DCEL")
        _, abs_tol = resolve_tol(rel_tol, abs_tol)
        rings = _polygon_rings(polygon)
        if len(rings) != 1:
            raise ValueError("add_hole expects a single polygon")
        ring = _closed_ring(rings[0], abs_tol=abs_tol)
        if len(ring) < 3:
            raise ValueError("hole must have at least three vertices")
        if polygon_area(ring) > 0:
            ring = list(reversed(ring))
        sample = _polygon_centroid(ring)
        if not self.contains(face, sample):
            raise ValueError("hole does not lie inside its face")
        for existing in face.inner_boundaries:
            if in_polygon(sample, _cycle_points(existing), exclude_border=True):
                raise ValueError("hole lies inside an existing hole")
        vertices = [self.add_vertex(point, abs_tol=abs_tol) for point in ring]
        for vertex in vertices:
            if vertex.half_edge is not None:
                raise ValueError("hole vertex already has incident edges")
        n_vertices = len(vertices)
        half_edges: list[HalfEdge] = []
        twins: list[HalfEdge] = []
        for i in range(n_vertices):
            half_edge, twin, _edge = self._register_pair(
                vertices[i],
                vertices[(i + 1) % n_vertices],
                face,
                self.outer_face,
            )
            half_edges.append(half_edge)
            twins.append(twin)
        for i in range(n_vertices):
            half_edges[i].next = half_edges[(i + 1) % n_vertices]
            half_edges[i].prev = half_edges[i - 1]
            twins[i].next = twins[i - 1]
            twins[i].prev = twins[(i + 1) % n_vertices]
            vertices[i].half_edge = half_edges[i]
            half_edges[i]._load_segment()
            twins[i]._load_segment()
        face.inner_boundaries.append(half_edges[0])
        self._rebuild_inner_boundaries(face)
        self._rebuild_inner_boundaries(self.outer_face)

    def remove_face(self, face: Face) -> DCEL:
        """Remove a bounded face and its boundary edges (mutated).

        The face may only share edges with the unbounded face (and its
        own holes). Holes of the face are removed with it.

        Args:
            face: Bounded face to remove.

        Returns:
            DCEL: This mesh.

        Examples:
            >>> import simetri.graphics as sg
            >>> mesh = sg.DCEL()
            >>> face = mesh.add_face([(0, 0), (60, 0), (60, 80), (0, 80)])
            >>> mesh.remove_face(face) is mesh
            True
            >>> mesh.bounded_faces
            []
            >>> mesh.validate()
            True
        """
        if face is self.outer_face:
            raise ValueError("cannot remove the unbounded face")
        if face not in self.faces:
            raise ValueError("face is not in this DCEL")
        for half_edge in self._face_boundary_halfedges(face):
            twin = half_edge.twin
            if twin is None:
                raise ValueError("half-edge has no twin")
            neighbor = twin.face
            if (
                neighbor is not None
                and neighbor is not face
                and neighbor is not self.outer_face
            ):
                raise ValueError(
                    "face shares an edge with another bounded face"
                )
        edges: list[Edge] = []
        seen: set[int] = set()
        for half_edge in list(self._face_boundary_halfedges(face)):
            edge = self._edge_of_halfedge(half_edge)
            edge_id = id(edge)
            if edge_id in seen:
                continue
            seen.add(edge_id)
            edges.append(edge)
        for edge in edges:
            self._destroy_edge_records(edge)
        self.faces.remove(face)
        face.mesh = None
        face.half_edge = None
        face.inner_boundaries = []
        self._repair_vertex_links()
        self._remove_orphan_vertices()
        self.outer_face.half_edge = None
        self._rebuild_inner_boundaries(self.outer_face)
        return self

    def _remove_hole(
        self,
        face: Face,
        polygon: Shape | Sequence[PointType],
        abs_tol: float | None = None,
        rel_tol: float | None = None,
    ) -> None:
        """Remove a hole cycle of ``face`` matching ``polygon`` (mutated)."""
        if face is self.outer_face:
            raise ValueError("unbounded face has no removable holes")
        if face not in self.faces:
            raise ValueError("face is not in this DCEL")
        _, abs_tol = resolve_tol(rel_tol, abs_tol)
        rings = _polygon_rings(polygon)
        if len(rings) != 1:
            raise ValueError("remove_hole expects a single polygon")
        ring = _closed_ring(rings[0], abs_tol=abs_tol)
        if polygon_area(ring) > 0:
            ring = list(reversed(ring))
        target: HalfEdge | None = None
        for start in face.inner_boundaries:
            if _rings_match(_cycle_points(start), ring, abs_tol=abs_tol):
                target = start
                break
        if target is None:
            raise ValueError("no matching hole on face")
        edges: list[Edge] = []
        seen: set[int] = set()
        for half_edge in _cycle_halfedges(target):
            edge = self._edge_of_halfedge(half_edge)
            edge_id = id(edge)
            if edge_id in seen:
                continue
            seen.add(edge_id)
            edges.append(edge)
        for edge in edges:
            self._destroy_edge_records(edge)
        self._drop_inner(face, target)
        self._repair_vertex_links()
        self._remove_orphan_vertices()
        self._rebuild_inner_boundaries(face)
        self._rebuild_inner_boundaries(self.outer_face)

    def _destroy_edge_records(self, edge: Edge) -> None:
        """Remove an edge and its twins from mesh lists without splicing."""
        half_edge = edge.half_edge
        twin = half_edge.twin
        for face in (half_edge.face, None if twin is None else twin.face):
            if face is None:
                continue
            if face.half_edge is half_edge or (
                twin is not None and face.half_edge is twin
            ):
                face.half_edge = None
            face.inner_boundaries = [
                item
                for item in face.inner_boundaries
                if item is not half_edge and item is not twin
            ]
        self._unlink_edge_pair(edge)

    def _repair_vertex_links(self) -> None:
        """Reset each vertex ``half_edge`` to a live outgoing half-edge."""
        for vertex in self.vertices:
            vertex.half_edge = None
        for half_edge in self.half_edges:
            origin = half_edge.origin
            if origin is not None and origin.half_edge is None:
                origin.half_edge = half_edge

    def _remove_orphan_vertices(self) -> None:
        """Drop vertices that are not endpoints of any remaining edge."""
        used: set[Vertex] = set()
        for half_edge in self.half_edges:
            if half_edge.origin is not None:
                used.add(half_edge.origin)
            if half_edge.destination is not None:
                used.add(half_edge.destination)
        self.vertices = [vertex for vertex in self.vertices if vertex in used]

    def split_edge(self, edge: Edge, point: PointType) -> SplitEdgeResult:
        """Insert a vertex on ``edge`` and split both twins (mutated).

        Args:
            edge: Edge to split.
            point: Location of the new vertex. Must lie on ``edge``.

        Returns:
            SplitEdgeResult: New vertex and the two resulting edges.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (80, 0), (80, 40), (0, 40)]])
            >>> bottom = _edge_with_points(mesh, (0.0, 0.0), (80.0, 0.0))
            >>> result = mesh.split_edge(bottom, (40, 0))
            >>> result.vertex.point
            (40.0, 0.0)
            >>> mesh.validate()
            True
        """
        origin = edge.origin
        destination = edge.destination
        if origin is None or destination is None:
            raise ValueError("edge endpoints are not set")
        if not on_segment(origin.point, destination.point, point[:2]):
            raise ValueError("point does not lie on the edge")
        if _same_point(point, origin.point):
            raise ValueError("point coincides with the edge origin")
        if _same_point(point, destination.point):
            raise ValueError("point coincides with the edge destination")
        vertex = self.add_vertex(point)
        if vertex is origin or vertex is destination:
            raise ValueError("point coincides with an existing endpoint")
        if vertex.half_edge is not None:
            raise ValueError("a vertex already exists at this point")

        h = edge.half_edge
        t = h.twin
        if t is None:
            raise ValueError("edge has no twin")
        h_next = h.next
        t_next = t.next
        if h_next is None or t_next is None:
            raise ValueError("edge cycle is incomplete")

        n = HalfEdge()
        tn = HalfEdge()
        n.destination = destination
        tn.destination = origin
        h.destination = vertex
        t.destination = vertex
        n.twin = t
        t.twin = n
        h.twin = tn
        tn.twin = h
        n.face = h.face
        tn.face = t.face

        n.next = h_next
        n.prev = h
        h_next.prev = n
        h.next = n

        tn.next = t_next
        tn.prev = t
        t_next.prev = tn
        t.next = tn

        vertex.half_edge = n
        if destination.half_edge is h:
            destination.half_edge = n
        if origin.half_edge is t:
            origin.half_edge = tn

        self.half_edges.append(n)
        self.half_edges.append(tn)
        second = Edge(n)
        self.edges.append(second)
        edge._load_segment()
        h._load_segment()
        t._load_segment()
        n._load_segment()
        tn._load_segment()
        if h.face is not None:
            h.face._load_boundary()
        if t.face is not None:
            t.face._load_boundary()
        return SplitEdgeResult(
            vertex=vertex, first_edge=edge, second_edge=second
        )

    def split_face(
        self,
        face: Face,
        vertex_a: Vertex | PointType,
        vertex_b: Vertex | PointType,
    ) -> SplitFaceResult:
        """Split ``face`` by a chord between two boundary locations (mutated).

        Args:
            face: Face to split.
            vertex_a: Existing vertex, or a point on the boundary.
            vertex_b: Existing vertex, or a point on the boundary.

        Returns:
            SplitFaceResult: New edge and the two faces.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (80, 0), (80, 80), (0, 80)]])
            >>> interior = mesh.bounded_faces[0]
            >>> va = [v for v in mesh.vertices if v.point == (0.0, 0.0)][0]
            >>> vb = [v for v in mesh.vertices if v.point == (80.0, 80.0)][0]
            >>> result = mesh.split_face(interior, va, vb)
            >>> len(mesh.bounded_faces)
            2
            >>> mesh.validate()
            True
        """
        vertex_a = self._as_boundary_vertex(face, vertex_a)
        vertex_b = self._as_boundary_vertex(face, vertex_b)
        if vertex_a is vertex_b:
            raise ValueError("split_face requires two distinct vertices")
        if self.edge_between(vertex_a, vertex_b) is not None:
            raise ValueError("vertices are already connected")
        he_a = self._halfedge_leaving_on_face(vertex_a, face)
        he_b = self._halfedge_leaving_on_face(vertex_b, face)
        if not _same_cycle(he_a, he_b):
            return self._bridge_face_cycles(face, he_a, he_b)
        return self._mef(face, he_a, he_b)

    def split_face_by_edge(
        self, face: Face, vertex_a: Vertex, vertex_b: Vertex
    ) -> SplitFaceResult:
        """Split ``face`` by connecting two boundary vertices (mutated).

        Args:
            face: Face to split.
            vertex_a: First boundary vertex.
            vertex_b: Second boundary vertex.

        Returns:
            SplitFaceResult: New edge and the two faces.
        """
        return self.split_face(face, vertex_a, vertex_b)

    def merge_faces(self, face_a: Face, face_b: Face) -> Face:
        """Merge two faces that share exactly one edge (mutated).

        Args:
            face_a: First face.
            face_b: Second face.

        Returns:
            Face: The surviving face.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (80, 0), (80, 80), (0, 80)]])
            >>> interior = mesh.bounded_faces[0]
            >>> va = [v for v in mesh.vertices if v.point == (0.0, 0.0)][0]
            >>> vb = [v for v in mesh.vertices if v.point == (80.0, 80.0)][0]
            >>> _ = mesh.split_face(interior, va, vb)
            >>> shared = mesh.shared_edges(mesh.bounded_faces[0], mesh.bounded_faces[1])
            >>> survivor = mesh.merge_faces(mesh.bounded_faces[0], mesh.bounded_faces[1])
            >>> len(mesh.bounded_faces)
            1
            >>> mesh.validate()
            True
        """
        shared = self.shared_edges(face_a, face_b)
        if len(shared) == 0:
            raise ValueError("faces do not share an edge")
        survivor = self.remove_edge(shared[0])
        pending = shared[1:]
        while pending:
            tip = None
            for edge in pending:
                if edge not in self.edges:
                    continue
                half_edge = edge.half_edge
                twin = half_edge.twin
                if twin is not None and (
                    half_edge.next is twin or half_edge.prev is twin
                ):
                    tip = edge
                    break
            if tip is None:
                raise ValueError("shared boundary is not a single chain")
            self._remove_slit(tip)
            pending = [edge for edge in pending if edge in self.edges]
        self._rebuild_inner_boundaries(survivor)
        return survivor

    def remove_edge(self, edge: Edge) -> Face:
        """Remove a shared edge and merge its two faces (mutated).

        Args:
            edge: Edge whose twins bound two distinct faces.

        Returns:
            Face: The surviving face.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (80, 0), (80, 80), (0, 80)]])
            >>> interior = mesh.bounded_faces[0]
            >>> va = [v for v in mesh.vertices if v.point == (0.0, 0.0)][0]
            >>> vb = [v for v in mesh.vertices if v.point == (80.0, 80.0)][0]
            >>> chord = mesh.split_face(interior, va, vb).edge
            >>> mesh.remove_edge(chord) is mesh.bounded_faces[0]
            True
        """
        return self._kef(edge)

    def remove_vertex(self, vertex: Vertex) -> None:
        """Remove an isolated vertex (mutated).

        Args:
            vertex: Vertex with no incident edges.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> vertex = mesh.add_vertex((0, 0))
            >>> mesh.remove_vertex(vertex)
            >>> mesh.vertices
            []
        """
        if vertex.half_edge is not None:
            raise ValueError(
                "vertex has incident edges; use collapse_edge to remove it"
            )
        self.vertices.remove(vertex)

    def collapse_edge(self, edge: Edge, target: Vertex | None = None) -> Vertex:
        """Merge the endpoints of ``edge`` onto ``target`` (mutated).

        Args:
            edge: Edge to collapse.
            target: Surviving vertex. Defaults to the origin of ``edge``.

        Returns:
            Vertex: The surviving vertex.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> a = mesh.add_vertex((0, 0))
            >>> b = mesh.add_vertex((40, 0))
            >>> edge = mesh.add_edge(a, b)
            >>> mesh.collapse_edge(edge) is a
            True
            >>> b in mesh.vertices
            False
        """
        origin = edge.origin
        destination = edge.destination
        if origin is None or destination is None:
            raise ValueError("edge endpoints are not set")
        if target is None:
            target = origin
        if target is not origin and target is not destination:
            raise ValueError("target must be an endpoint of the edge")
        other = destination if target is origin else origin
        self._reject_collapse(edge, target, other)
        half_edge = edge.half_edge
        twin = half_edge.twin
        if twin is None:
            raise ValueError("edge has no twin")
        for outgoing in list(other.outgoing_halfedges()):
            if outgoing is half_edge or outgoing is twin:
                continue
            outgoing.twin.destination = target
        if target is origin:
            keep_outgoing = twin.next
        else:
            keep_outgoing = half_edge.next
        self._splice_out_edge(edge)
        self._unlink_edge_pair(edge)
        if keep_outgoing is not None and keep_outgoing.origin is target:
            target.half_edge = keep_outgoing
        else:
            target.half_edge = None
        other.half_edge = None
        self.vertices.remove(other)
        return target

    def insert_segment(self, segment: LineType) -> SplitResult:
        """Insert a finite segment, splitting crossed edges and faces (mutated).

        Args:
            segment: Segment ``(p1, p2)``.

        Returns:
            SplitResult: Faces, cut edges, and inserted vertices.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (80, 0), (80, 80), (0, 80)]])
            >>> result = mesh.insert_segment(((0, 0), (80, 80)))
            >>> len(mesh.bounded_faces)
            2
            >>> mesh.validate()
            True
        """
        start = segment[0][:2]
        end = segment[1][:2]
        if _same_point(start, end):
            raise ValueError("segment has zero length")
        inserted: list[Vertex] = []
        hits = self._segment_hits(start, end)
        vertices: list[Vertex] = []
        for _param, point, hit_edge, vertex in hits:
            if vertex is not None:
                vertices.append(vertex)
                continue
            if hit_edge is None:
                vertices.append(self.add_vertex(point))
                continue
            if hit_edge not in self.edges:
                existing = self._vertex_at(point)
                if existing is None:
                    raise ValueError("intersection edge was removed")
                vertices.append(existing)
                continue
            origin = hit_edge.origin
            destination = hit_edge.destination
            if origin is not None and _same_point(point, origin.point):
                vertices.append(origin)
                continue
            if destination is not None and _same_point(
                point, destination.point
            ):
                vertices.append(destination)
                continue
            split = self.split_edge(hit_edge, point)
            inserted.append(split.vertex)
            vertices.append(split.vertex)
        unique_vertices = []
        for vertex in vertices:
            if vertex not in unique_vertices:
                unique_vertices.append(vertex)
        cut_edges: list[Edge] = []
        for i, vertex in enumerate(unique_vertices[:-1]):
            nxt = unique_vertices[i + 1]
            existing = self.edge_between(vertex, nxt)
            if existing is not None:
                cut_edges.append(existing)
                continue
            host = self._face_for_new_edge(vertex, nxt)
            cut_edges.append(self.add_edge(vertex, nxt, face=host))
        return SplitResult(
            faces=list(self.faces),
            cut_edges=cut_edges,
            inserted_vertices=inserted,
        )

    def split_face_by_line(self, face: Face, line: LineType) -> SplitResult:
        """Split ``face`` by an infinite line (mutated).

        Args:
            face: Face to cut.
            line: Two points defining an infinite line.

        Returns:
            SplitResult: Resulting faces, cut edges, and inserted vertices.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (80, 0), (80, 80), (0, 80)]])
            >>> interior = mesh.bounded_faces[0]
            >>> result = mesh.split_face_by_line(interior, [(40, -40), (40, 60)])
            >>> len(result.faces)
            2
            >>> mesh.validate()
            True
        """
        p1 = line[0][:2]
        p2 = line[1][:2]
        if _same_point(p1, p2):
            raise ValueError("line has zero length")
        inserted: list[Vertex] = []
        hits: list[tuple[float, Vertex]] = []
        for half_edge in self._face_boundary_halfedges(face):
            origin = half_edge.origin
            destination = half_edge.destination
            if origin is None or destination is None:
                raise ValueError("half-edge endpoints are not set")
            point = intersect((p1, p2), (origin.point, destination.point))
            if point is None:
                continue
            if not on_segment(origin.point, destination.point, point):
                continue
            param = _segment_param(p1, p2, point)
            if _same_point(point, origin.point):
                hits.append((param, origin))
                continue
            if _same_point(point, destination.point):
                hits.append((param, destination))
                continue
            edge = self._edge_of_halfedge(half_edge)
            split = self.split_edge(edge, point)
            inserted.append(split.vertex)
            hits.append((param, split.vertex))
        unique: list[tuple[float, Vertex]] = []
        seen: set[int] = set()
        for param, vertex in sorted(hits, key=lambda item: item[0]):
            if id(vertex) in seen:
                continue
            seen.add(id(vertex))
            unique.append((param, vertex))
        cut_edges: list[Edge] = []
        current_face = face
        for i, (_param, vertex) in enumerate(unique[:-1]):
            nxt = unique[i + 1][1]
            mid = _midpoint(vertex.point, nxt.point)
            if not self.contains(current_face, mid):
                host = self.locate_face(mid)
                if host is self.outer_face or not self.contains(host, mid):
                    continue
                current_face = host
            if self.edge_between(vertex, nxt) is not None:
                continue
            if current_face not in vertex.incident_faces():
                current_face = self._common_face(vertex, nxt, None)
            result = self.split_face(current_face, vertex, nxt)
            cut_edges.append(result.edge)
            current_face = result.first_face
        faces = [
            item
            for item in self.faces
            if item is face
            or item
            in [edge.adjacent_faces()[0] for edge in cut_edges]
            + [edge.adjacent_faces()[1] for edge in cut_edges]
        ]
        if not faces:
            faces = [face]
        unique_faces = []
        for item in faces:
            if item not in unique_faces:
                unique_faces.append(item)
        return SplitResult(
            faces=unique_faces,
            cut_edges=cut_edges,
            inserted_vertices=inserted,
        )

    def split_face_by_segment(
        self, face: Face, segment: LineType
    ) -> SplitResult:
        """Insert a finite segment and split faces it crosses (mutated).

        Args:
            face: Face that should contain the cut; used to reject misses.
            segment: Segment ``(p1, p2)``.

        Returns:
            SplitResult: Faces, cut edges, and inserted vertices.
        """
        mid = _midpoint(segment[0][:2], segment[1][:2])
        if not self.contains(face, mid) and not self._segment_hits_face(
            face, segment
        ):
            raise ValueError("segment does not meet the face")
        return self.insert_segment(segment)

    def locate_face(self, point: PointType) -> Face:
        """Return the face that contains ``point``.

        Args:
            point: Query point.

        Returns:
            Face: A bounded face, or the unbounded outer face.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (40, 0), (40, 40), (0, 40)]])
            >>> mesh.locate_face((0.5, 0.5)) is mesh.bounded_faces[0]
            True
            >>> mesh.locate_face((80, 80)) is mesh.outer_face
            True
        """
        for face in self.bounded_faces:
            if self.contains(face, point):
                return face
        return self.outer_face

    def contains(self, face: Face, point: PointType) -> bool:
        """Return whether ``point`` lies in ``face``.

        Args:
            face: Face to test.
            point: Query point.

        Returns:
            bool: True if the point is in the face.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (40, 0), (40, 40), (0, 40)]])
            >>> mesh.contains(mesh.bounded_faces[0], (0.5, 0.5))
            True
        """
        if face is self.outer_face:
            for bounded in self.bounded_faces:
                if self.contains(bounded, point):
                    return False
            return True
        if face.half_edge is None:
            return False
        outer = _cycle_points(face.half_edge)
        if not in_polygon(point, outer):
            return False
        for hole in face.inner_boundaries:
            if in_polygon(point, _cycle_points(hole), exclude_border=True):
                return False
        return True

    def face_area(self, face: Face) -> float:
        """Return the signed area of ``face`` (outer minus holes).

        Args:
            face: Face to measure.

        Returns:
            float: Area. Unbounded face is ``inf``.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (40, 0), (40, 40), (0, 40)]])
            >>> mesh.face_area(mesh.bounded_faces[0])
            1600.0
        """
        if face is self.outer_face:
            return inf
        if face.half_edge is None:
            raise ValueError("face has no boundary")
        area = polygon_area(_cycle_points(face.half_edge))
        for hole in face.inner_boundaries:
            area += polygon_area(_cycle_points(hole))
        return area

    def face_centroid(self, face: Face) -> tuple[float, float]:
        """Return the centroid of a bounded face.

        Args:
            face: Bounded face.

        Returns:
            tuple[float, float]: Centroid.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (40, 0), (40, 40), (0, 40)]])
            >>> mesh.face_centroid(mesh.bounded_faces[0])
            (20.0, 20.0)
        """
        if face is self.outer_face:
            raise ValueError("unbounded face has no centroid")
        if face.half_edge is None:
            raise ValueError("face has no boundary")
        return _polygon_centroid(_cycle_points(face.half_edge))

    def face_bounds(self, face: Face) -> BoundingBox:
        """Return the axis-aligned bounds of ``face``.

        Args:
            face: Face to bound.

        Returns:
            BoundingBox: Bounds of the outer cycle.
        """
        if face.half_edge is None:
            raise ValueError("face has no boundary")
        return bounding_box(_cycle_points(face.half_edge))

    def is_boundary_edge(self, edge: Edge) -> bool:
        """Return True if one side of ``edge`` is the unbounded face.

        Args:
            edge: Edge to test.

        Returns:
            bool: True if the edge is on the outer boundary.
        """
        left, right = edge.adjacent_faces()
        return left is self.outer_face or right is self.outer_face

    def is_outer_face(self, face: Face) -> bool:
        """Return True if ``face`` is the unbounded face.

        Args:
            face: Face to test.

        Returns:
            bool: True for the unbounded face.
        """
        return face is self.outer_face

    def adjacent_faces(self, face: Face) -> list[Face]:
        """Return faces that share an edge with ``face``.

        Args:
            face: Face whose neighbors are requested.

        Returns:
            list[Face]: Adjacent faces, each once.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (40, 0), (40, 40), (0, 40)]])
            >>> mesh.adjacent_faces(mesh.bounded_faces[0]) == [mesh.outer_face]
            True
        """
        neighbors: list[Face] = []
        for half_edge in self._face_boundary_halfedges(face):
            if half_edge.twin is None:
                raise ValueError("half-edge has no twin")
            neighbor = half_edge.twin.face
            if (
                neighbor is not None
                and neighbor is not face
                and neighbor not in neighbors
            ):
                neighbors.append(neighbor)
        return neighbors

    def shared_edges(self, face_a: Face, face_b: Face) -> list[Edge]:
        """Return edges that have ``face_a`` and ``face_b`` on opposite sides.

        Args:
            face_a: First face.
            face_b: Second face.

        Returns:
            list[Edge]: Shared edges.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (40, 0), (40, 40), (0, 40)]])
            >>> len(mesh.shared_edges(mesh.bounded_faces[0], mesh.outer_face))
            4
        """
        shared: list[Edge] = []
        faces = {face_a, face_b}
        for edge in self.edges:
            left, right = edge.adjacent_faces()
            if {left, right} == faces:
                shared.append(edge)
        return shared

    def are_adjacent(self, face_a: Face, face_b: Face) -> bool:
        """Return True if the faces share an edge.

        Args:
            face_a: First face.
            face_b: Second face.

        Returns:
            bool: True if adjacent.
        """
        return len(self.shared_edges(face_a, face_b)) > 0

    def connected_components(self) -> list[list[Vertex]]:
        """Return vertex-connected components of the mesh.

        Returns:
            list[list[Vertex]]: Components as vertex lists.
        """
        remaining = set(self.vertices)
        components: list[list[Vertex]] = []
        while remaining:
            start = next(iter(remaining))
            stack = [start]
            component: list[Vertex] = []
            remaining.remove(start)
            while stack:
                vertex = stack.pop()
                component.append(vertex)
                if vertex.half_edge is None:
                    continue
                for half_edge in vertex.outgoing_halfedges():
                    neighbor = half_edge.destination
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
            components.append(component)
        return components

    def boundary_components(self, face: Face) -> list[HalfEdge]:
        """Return start half-edges of the outer cycle and holes.

        Args:
            face: Face whose cycles are requested.

        Returns:
            list[HalfEdge]: Outer start, then hole starts.
        """
        if face.half_edge is None:
            return list(face.inner_boundaries)
        return [face.half_edge, *face.inner_boundaries]

    def holes(self, face: Face) -> list[list[Vertex]]:
        """Return hole cycles of ``face`` as vertex lists.

        Args:
            face: Face whose holes are requested.

        Returns:
            list[list[Vertex]]: Hole boundaries.
        """
        return [
            [half_edge.origin for half_edge in _cycle_halfedges(start)]
            for start in face.inner_boundaries
        ]

    def face_vertices(self, face: Face) -> list[Vertex]:
        """Return outer-boundary vertices of ``face``.

        Args:
            face: Face to walk.

        Returns:
            list[Vertex]: Origin vertices of the outer cycle.
        """
        return face.boundary_vertices()

    def face_halfedges(self, face: Face) -> list[HalfEdge]:
        """Return outer-boundary half-edges of ``face``.

        Args:
            face: Face to walk.

        Returns:
            list[HalfEdge]: Outer cycle.
        """
        return list(face.boundary_halfedges())

    def vertex_halfedges(self, vertex: Vertex) -> list[HalfEdge]:
        """Return outgoing half-edges of ``vertex``.

        Args:
            vertex: Vertex to walk.

        Returns:
            list[HalfEdge]: Outgoing half-edges.
        """
        return list(vertex.outgoing_halfedges())

    def overlay(self, other: DCEL) -> DCEL:
        """Return a new mesh containing the overlay of both subdivisions.

        Args:
            other: Second mesh.

        Returns:
            DCEL: Overlay arrangement.
        """
        result = DCEL()
        for mesh in (self, other):
            for edge in mesh.edges:
                result.insert_segment(edge.segment)
        return result

    def union(self, other: DCEL) -> DCEL:
        """Return the union of the bounded regions of two meshes.

        Args:
            other: Second mesh.

        Returns:
            DCEL: Boolean union.
        """
        return self._boolean(other, lambda in_a, in_b: in_a or in_b)

    def intersection(self, other: DCEL) -> DCEL:
        """Return the intersection of the bounded regions of two meshes.

        Args:
            other: Second mesh.

        Returns:
            DCEL: Boolean intersection.
        """
        return self._boolean(other, lambda in_a, in_b: in_a and in_b)

    def difference(self, other: DCEL) -> DCEL:
        """Return this mesh minus ``other``.

        Args:
            other: Mesh subtracted from this one.

        Returns:
            DCEL: Boolean difference.
        """
        return self._boolean(other, lambda in_a, in_b: in_a and not in_b)

    def symmetric_difference(self, other: DCEL) -> DCEL:
        """Return the symmetric difference of two meshes.

        Args:
            other: Second mesh.

        Returns:
            DCEL: Boolean symmetric difference.
        """
        return self._boolean(other, lambda in_a, in_b: in_a != in_b)

    def validate(self, check_geometry: bool = False) -> bool:
        """Verify DCEL invariants. Raise ``ValueError`` if they fail.

        Args:
            check_geometry: If True, also check planarity, zero-length
                edges, duplicate directed edges, and simple face cycles.

        Returns:
            bool: True if the mesh is valid.

        Examples:
            >>> from simetri.geom.polygons.dcel import DCEL
            >>> mesh = DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (40, 0), (40, 40), (0, 40)]])
            >>> mesh.validate()
            True
            >>> mesh.validate(check_geometry=True)
            True
        """
        for half_edge in self.half_edges:
            if half_edge.twin is None or half_edge.twin.twin is not half_edge:
                raise ValueError("half-edge twin.twin is not identity")
            if half_edge.next is None or half_edge.next.prev is not half_edge:
                raise ValueError("half-edge next.prev is not identity")
            if half_edge.prev is None or half_edge.prev.next is not half_edge:
                raise ValueError("half-edge prev.next is not identity")
            if half_edge.origin is None:
                raise ValueError("half-edge origin is None")
            if half_edge.face is None:
                raise ValueError("half-edge face is None")
        for face in self.faces:
            for start in self.boundary_components(face):
                for half_edge in _cycle_halfedges(start):
                    if half_edge.face is not face:
                        raise ValueError("half-edge face does not match cycle")
        seen_ids: set[int] = set()
        for face in self.faces:
            for start in self.boundary_components(face):
                for half_edge in _cycle_halfedges(start):
                    key = id(half_edge)
                    if key in seen_ids:
                        raise ValueError("half-edge belongs to multiple cycles")
                    seen_ids.add(key)
        if len(seen_ids) != len(self.half_edges):
            raise ValueError("some half-edges are not in any face cycle")
        if check_geometry:
            self._validate_geometry()
        return True

    def _validate_geometry(self) -> None:
        """Raise if geometric invariants fail."""
        directed: set[tuple[tuple[float, float], tuple[float, float]]] = set()
        for edge in self.edges:
            origin = edge.origin
            destination = edge.destination
            if origin is None or destination is None:
                raise ValueError("edge endpoints are not set")
            if _same_point(origin.point, destination.point):
                raise ValueError("zero-length edge")
            key = (origin.point, destination.point)
            if key in directed:
                raise ValueError("duplicate directed edge")
            directed.add(key)
        for i, edge_a in enumerate(self.edges):
            for edge_b in self.edges[i + 1 :]:
                kind, point = segment_intersection(
                    edge_a.segment, edge_b.segment
                )
                if kind != Connection.INTERSECT or point is None:
                    continue
                endpoints = {
                    edge_a.origin.point,
                    edge_a.destination.point,
                    edge_b.origin.point,
                    edge_b.destination.point,
                }
                if any(_same_point(point, end) for end in endpoints):
                    continue
                raise ValueError("edges intersect away from vertices")
        for face in self.bounded_faces:
            if face.half_edge is None:
                continue
            ring = _cycle_points(face.half_edge)
            if not is_simple(ring):
                raise ValueError("face boundary is not simple")
            for hole in face.inner_boundaries:
                hole_ring = _cycle_points(hole)
                sample = _polygon_centroid(hole_ring)
                if not in_polygon(sample, ring):
                    raise ValueError("hole does not lie inside its face")

    def _vertex_at(
        self, point: PointType, abs_tol: float | None = None
    ) -> Vertex | None:
        """Return an existing vertex at ``point``, else ``None``."""
        for vertex in self.vertices:
            if _same_point(vertex.point, point, abs_tol=abs_tol):
                return vertex
        return None

    def edge_between(self, vertex_a: Vertex, vertex_b: Vertex) -> Edge | None:
        """Return the undirected edge joining two vertices, else ``None``.

        Argument order does not matter.

        Args:
            vertex_a: First endpoint.
            vertex_b: Second endpoint.

        Returns:
            Edge | None: The edge, or ``None`` if none exists.

        Examples:
            >>> import simetri.graphics as sg
            >>> mesh = sg.DCEL()
            >>> _ = mesh.build_from_polygons([[(0, 0), (40, 0), (40, 40), (0, 40)]])
            >>> a = mesh.add_vertex((0, 0))
            >>> b = mesh.add_vertex((40, 0))
            >>> mesh.edge_between(a, b).segment
            ((0.0, 0.0), (40.0, 0.0))
            >>> mesh.edge_between(a, mesh.add_vertex((80, 80))) is None
            True
        """
        target = {vertex_a, vertex_b}
        for edge in self.edges:
            ends = {edge.origin, edge.destination}
            if ends == target:
                return edge
        return None

    def _edge_of_halfedge(self, half_edge: HalfEdge) -> Edge:
        """Return the ``Edge`` wrapping ``half_edge`` or its twin."""
        for edge in self.edges:
            if edge.half_edge is half_edge or edge.half_edge.twin is half_edge:
                return edge
        raise ValueError("half-edge is not registered as an edge")

    def _halfedge_leaving_on_face(self, vertex: Vertex, face: Face) -> HalfEdge:
        """Return an outgoing half-edge of ``vertex`` that bounds ``face``."""
        for half_edge in vertex.outgoing_halfedges():
            if half_edge.face is face:
                return half_edge
        raise ValueError("vertex is not on the face")

    def _common_face(
        self, vertex_a: Vertex, vertex_b: Vertex, face: Face | None
    ) -> Face:
        """Return a face incident to both vertices."""
        faces_a = set(vertex_a.incident_faces())
        faces_b = set(vertex_b.incident_faces())
        shared = faces_a & faces_b
        if face is not None:
            if face not in shared:
                raise ValueError("vertices do not both lie on the given face")
            return face
        if not shared:
            raise ValueError("vertices do not share a face")
        if self.outer_face in shared and len(shared) > 1:
            shared.remove(self.outer_face)
        return next(iter(shared))

    def _face_for_new_edge(self, vertex_a: Vertex, vertex_b: Vertex) -> Face:
        """Choose the face that should receive a new edge."""
        if vertex_a.half_edge is None and vertex_b.half_edge is None:
            mid = _midpoint(vertex_a.point, vertex_b.point)
            return self.locate_face(mid)
        if vertex_a.half_edge is None:
            return self.locate_face(vertex_a.point)
        if vertex_b.half_edge is None:
            return self.locate_face(vertex_b.point)
        return self._common_face(vertex_a, vertex_b, None)

    def _as_boundary_vertex(
        self, face: Face, vertex_or_point: Vertex | PointType
    ) -> Vertex:
        """Return a vertex on ``face``, splitting an edge if needed."""
        if isinstance(vertex_or_point, Vertex):
            return vertex_or_point
        point = vertex_or_point[:2]
        existing = self._vertex_at(point)
        if existing is not None:
            return existing
        for half_edge in self._face_boundary_halfedges(face):
            origin = half_edge.origin
            destination = half_edge.destination
            if origin is None or destination is None:
                raise ValueError("half-edge endpoints are not set")
            if on_segment(origin.point, destination.point, point):
                edge = self._edge_of_halfedge(half_edge)
                return self.split_edge(edge, point).vertex
        raise ValueError("point does not lie on the face boundary")

    def _face_boundary_halfedges(self, face: Face) -> Iterator[HalfEdge]:
        """Yield every half-edge on outer and inner cycles of ``face``."""
        for start in self.boundary_components(face):
            yield from _cycle_halfedges(start)

    def _register_pair(
        self,
        origin: Vertex,
        destination: Vertex,
        face_left: Face,
        face_right: Face,
    ) -> tuple[HalfEdge, HalfEdge, Edge]:
        """Create twins, register them, and return ``(h, twin, edge)``."""
        half_edge = HalfEdge()
        twin = HalfEdge()
        half_edge.destination = destination
        twin.destination = origin
        half_edge.twin = twin
        twin.twin = half_edge
        half_edge.face = face_left
        twin.face = face_right
        self.half_edges.append(half_edge)
        self.half_edges.append(twin)
        edge = Edge(half_edge)
        self.edges.append(edge)
        half_edge._load_segment()
        twin._load_segment()
        return half_edge, twin, edge

    def _mev_slit(
        self, vertex_a: Vertex, vertex_b: Vertex, face: Face | None
    ) -> Edge:
        """Insert a dangling edge between two isolated vertices (MEV)."""
        host = self.outer_face if face is None else face
        half_edge, twin, edge = self._register_pair(
            vertex_a, vertex_b, host, host
        )
        half_edge.next = twin
        half_edge.prev = twin
        twin.next = half_edge
        twin.prev = half_edge
        vertex_a.half_edge = half_edge
        vertex_b.half_edge = twin
        if host.half_edge is None:
            host.half_edge = half_edge
        elif host is not self.outer_face or host.half_edge is not half_edge:
            if half_edge not in host.inner_boundaries:
                host.inner_boundaries.append(half_edge)
        return edge

    def _mev_from(
        self, attached: Vertex, isolated: Vertex, face: Face | None
    ) -> Edge:
        """Grow a new edge from ``attached`` to isolated ``isolated`` (MEV)."""
        host = face
        if host is None:
            if not attached.incident_faces():
                raise ValueError("attached vertex has no incident face")
            host = attached.incident_faces()[0]
            if self.outer_face in attached.incident_faces():
                mid = _midpoint(attached.point, isolated.point)
                host = self.locate_face(mid)
        leaving = self._halfedge_leaving_on_face(attached, host)
        prev = leaving.prev
        if prev is None:
            raise ValueError("attached vertex cycle is incomplete")
        half_edge, twin, edge = self._register_pair(
            attached, isolated, host, host
        )
        half_edge.prev = prev
        half_edge.next = twin
        twin.prev = half_edge
        twin.next = leaving
        prev.next = half_edge
        leaving.prev = twin
        isolated.half_edge = twin
        if attached.half_edge is None:
            attached.half_edge = half_edge
        return edge

    def _mef(
        self, face: Face, he_a: HalfEdge, he_b: HalfEdge
    ) -> SplitFaceResult:
        """Make edge and face: split one cycle of ``face`` (MEF)."""
        vertex_a = he_a.origin
        vertex_b = he_b.origin
        a_prev = he_a.prev
        b_prev = he_b.prev
        if a_prev is None or b_prev is None:
            raise ValueError("face cycle is incomplete")
        half_edge, twin, edge = self._register_pair(
            vertex_a, vertex_b, face, face
        )
        half_edge.prev = a_prev
        half_edge.next = he_b
        a_prev.next = half_edge
        he_b.prev = half_edge
        twin.prev = b_prev
        twin.next = he_a
        b_prev.next = twin
        he_a.prev = twin

        new_face = Face()
        new_face.mesh = self
        self._assign_cycle_face(half_edge, face)
        self._assign_cycle_face(twin, new_face)
        face.half_edge = half_edge
        new_face.half_edge = twin
        self._drop_inner(face, half_edge)
        self._drop_inner(face, twin)
        area_new = polygon_area(_cycle_points(twin))
        area_old = polygon_area(_cycle_points(half_edge))
        if face is self.outer_face:
            if area_new > 0:
                self.faces.append(new_face)
            elif area_old > 0:
                face.half_edge = twin
                new_face.half_edge = half_edge
                self._assign_cycle_face(twin, face)
                self._assign_cycle_face(half_edge, new_face)
                self.faces.append(new_face)
            else:
                self.faces.append(new_face)
        else:
            if area_new < 0 and area_old > 0:
                self._assign_cycle_face(twin, face)
                self._assign_cycle_face(half_edge, new_face)
                face.half_edge = twin
                new_face.half_edge = half_edge
                half_edge, twin = twin, half_edge
            self.faces.append(new_face)
        return SplitFaceResult(edge=edge, first_face=face, second_face=new_face)

    def _bridge_face_cycles(
        self, face: Face, he_a: HalfEdge, he_b: HalfEdge
    ) -> SplitFaceResult:
        """Connect two cycles of the same face (absorb a hole)."""
        vertex_a = he_a.origin
        vertex_b = he_b.origin
        a_prev = he_a.prev
        b_prev = he_b.prev
        if a_prev is None or b_prev is None:
            raise ValueError("face cycle is incomplete")
        half_edge, twin, edge = self._register_pair(
            vertex_a, vertex_b, face, face
        )
        half_edge.prev = a_prev
        half_edge.next = he_b
        a_prev.next = half_edge
        he_b.prev = half_edge
        twin.prev = b_prev
        twin.next = he_a
        b_prev.next = twin
        he_a.prev = twin
        self._assign_cycle_face(half_edge, face)
        if face.half_edge is None:
            face.half_edge = half_edge
        self._rebuild_inner_boundaries(face)
        return SplitFaceResult(edge=edge, first_face=face, second_face=face)

    def _kef(self, edge: Edge) -> Face:
        """Kill edge and face: merge the two faces of ``edge`` (KEF)."""
        half_edge = edge.half_edge
        twin = half_edge.twin
        if twin is None:
            raise ValueError("edge has no twin")
        face_a = half_edge.face
        face_b = twin.face
        if face_a is None or face_b is None:
            raise ValueError("edge faces are not set")
        if face_a is face_b:
            raise ValueError("remove_edge requires two distinct faces")
        h_prev = half_edge.prev
        h_next = half_edge.next
        t_prev = twin.prev
        t_next = twin.next
        if h_prev is None or h_next is None or t_prev is None or t_next is None:
            raise ValueError("edge cycle is incomplete")
        survivor = face_a
        dead = face_b
        if dead is self.outer_face:
            survivor, dead = face_b, face_a
        h_prev.next = t_next
        t_next.prev = h_prev
        t_prev.next = h_next
        h_next.prev = t_prev
        origin = half_edge.origin
        destination = half_edge.destination
        if origin is not None and origin.half_edge in (half_edge, twin):
            origin.half_edge = t_next
        if destination is not None and destination.half_edge in (
            half_edge,
            twin,
        ):
            destination.half_edge = h_next
        self._unlink_edge_pair(edge)
        self._assign_cycle_face(h_prev, survivor)
        if (
            survivor.half_edge is None
            or survivor.half_edge.face is not survivor
        ):
            survivor.half_edge = h_prev
        for start in dead.inner_boundaries:
            if (
                start.face is survivor
                and start not in survivor.inner_boundaries
            ):
                survivor.inner_boundaries.append(start)
        self.faces.remove(dead)
        self._rebuild_inner_boundaries(survivor)
        return survivor

    def _unlink_edge_pair(self, edge: Edge) -> None:
        """Remove an edge and its twins from the mesh lists."""
        half_edge = edge.half_edge
        twin = half_edge.twin
        self.edges.remove(edge)
        self.half_edges.remove(half_edge)
        if twin is not None:
            self.half_edges.remove(twin)

    def _splice_out_edge(self, edge: Edge) -> None:
        """Remove ``edge`` from face cycles without deleting the records yet."""
        half_edge = edge.half_edge
        twin = half_edge.twin
        if twin is None:
            raise ValueError("edge has no twin")
        if half_edge.next is twin and twin.next is half_edge:
            for face in (half_edge.face, twin.face):
                if face is None:
                    continue
                if face.half_edge is half_edge or face.half_edge is twin:
                    face.half_edge = None
                face.inner_boundaries = [
                    item
                    for item in face.inner_boundaries
                    if item is not half_edge and item is not twin
                ]
            return
        if half_edge.prev is twin:
            surviving_prev = twin.prev
            surviving_next = half_edge.next
        elif half_edge.next is twin:
            surviving_prev = half_edge.prev
            surviving_next = twin.next
        else:
            surviving_prev = None
            surviving_next = None
        if surviving_prev is not None or surviving_next is not None:
            if (
                surviving_prev is None
                or surviving_next is None
                or surviving_prev is half_edge
                or surviving_prev is twin
                or surviving_next is half_edge
                or surviving_next is twin
            ):
                raise ValueError("edge cycle is incomplete")
            surviving_prev.next = surviving_next
            surviving_next.prev = surviving_prev
            for face in (half_edge.face, twin.face):
                if face is None:
                    continue
                if face.half_edge is half_edge or face.half_edge is twin:
                    face.half_edge = surviving_next
            return
        h_prev = half_edge.prev
        h_next = half_edge.next
        t_prev = twin.prev
        t_next = twin.next
        if h_prev is None or h_next is None or t_prev is None or t_next is None:
            raise ValueError("edge cycle is incomplete")
        h_prev.next = t_next
        t_next.prev = h_prev
        t_prev.next = h_next
        h_next.prev = t_prev
        for face in (half_edge.face, twin.face):
            if face is None:
                continue
            if face.half_edge is half_edge or face.half_edge is twin:
                face.half_edge = h_prev

    def _remove_slit(self, edge: Edge) -> None:
        """Remove an edge whose twins bound the same face."""
        half_edge = edge.half_edge
        twin = half_edge.twin
        if twin is None:
            raise ValueError("edge has no twin")
        if half_edge.face is not twin.face:
            raise ValueError("edge is not a slit")
        face = half_edge.face
        origin = half_edge.origin
        destination = half_edge.destination
        self._splice_out_edge(edge)
        self._unlink_edge_pair(edge)
        for vertex in (origin, destination):
            if vertex is None:
                continue
            if (
                vertex.half_edge is not None
                and vertex.half_edge not in self.half_edges
            ):
                replacement = None
                for candidate in self.half_edges:
                    if candidate.origin is vertex:
                        replacement = candidate
                        break
                vertex.half_edge = replacement
        if face is not None:
            self._rebuild_inner_boundaries(face)

    def _assign_cycle_face(self, start: HalfEdge, face: Face) -> None:
        """Set ``face`` on every half-edge of the cycle at ``start``."""
        for half_edge in _cycle_halfedges(start):
            half_edge.face = face
            half_edge._load_segment()
        face._load_boundary(start)

    def _drop_inner(self, face: Face, start: HalfEdge) -> None:
        """Remove a cycle start from ``face.inner_boundaries`` if present."""
        start_ids = {id(half_edge) for half_edge in _cycle_halfedges(start)}
        face.inner_boundaries = [
            item for item in face.inner_boundaries if id(item) not in start_ids
        ]

    def _rebuild_inner_boundaries(self, face: Face) -> None:
        """Recompute hole starts for ``face`` from half-edges labeled with it."""
        starts: list[HalfEdge] = []
        seen: set[int] = set()
        for half_edge in self.half_edges:
            if half_edge.face is not face or id(half_edge) in seen:
                continue
            cycle = list(_cycle_halfedges(half_edge))
            for item in cycle:
                seen.add(id(item))
            starts.append(cycle[0])
        if not starts:
            face.half_edge = None
            face.inner_boundaries = []
            face._load_boundary()
            return
        outer = starts[0]
        outer_area = abs(polygon_area(_cycle_points(outer)))
        for start in starts[1:]:
            area = abs(polygon_area(_cycle_points(start)))
            if area > outer_area:
                outer = start
                outer_area = area
        face.half_edge = outer
        face.inner_boundaries = [
            start for start in starts if start is not outer
        ]
        face._load_boundary()

    def _reject_collapse(
        self, edge: Edge, target: Vertex, other: Vertex
    ) -> None:
        """Raise if collapsing ``edge`` would produce invalid topology."""
        if target is other:
            raise ValueError("cannot collapse a self-loop")
        neighbors = set()
        for half_edge in other.outgoing_halfedges():
            dest = half_edge.destination
            if dest is target:
                continue
            if dest in neighbors:
                raise ValueError("collapse would create a duplicate edge")
            neighbors.add(dest)
            if self.edge_between(target, dest) is not None:
                raise ValueError("collapse would create a duplicate edge")
        for half_edge in target.outgoing_halfedges():
            dest = half_edge.destination
            if dest is other:
                continue
            if dest in neighbors:
                raise ValueError("collapse would create a duplicate edge")
        for face in (edge.half_edge.face, edge.half_edge.twin.face):
            if face is None or face.half_edge is None:
                continue
            n_edges = len(list(_cycle_halfedges(face.half_edge)))
            if n_edges < 4 and face is not self.outer_face:
                raise ValueError("collapse would produce an invalid face cycle")
        for half_edge in other.outgoing_halfedges():
            if half_edge is edge.half_edge or half_edge is edge.half_edge.twin:
                continue
            dest = half_edge.destination
            new_segment = (target.point, dest.point)
            for existing in self.edges:
                if existing is edge:
                    continue
                ends = {existing.origin, existing.destination}
                if other in ends:
                    continue
                kind, point = segment_intersection(
                    new_segment, existing.segment
                )
                if kind != Connection.INTERSECT or point is None:
                    continue
                if _same_point(point, target.point) or _same_point(
                    point, dest.point
                ):
                    continue
                existing_ends = (
                    existing.origin.point,
                    existing.destination.point,
                )
                if any(_same_point(point, end) for end in existing_ends):
                    continue
                raise ValueError(
                    "collapse would create a geometric intersection"
                )

    def _segment_hits(
        self, start: PointType, end: PointType
    ) -> list[tuple[float, tuple[float, float], Edge | None, Vertex | None]]:
        """Return sorted (parameter, point, edge, vertex) hits along a segment."""
        hits: list[
            tuple[float, tuple[float, float], Edge | None, Vertex | None]
        ] = []
        hits.append(
            (
                0.0,
                (float(start[0]), float(start[1])),
                None,
                self._vertex_at(start),
            )
        )
        hits.append(
            (1.0, (float(end[0]), float(end[1])), None, self._vertex_at(end))
        )
        hits.extend(
            (
                _segment_param(start, end, vertex.point),
                vertex.point,
                None,
                vertex,
            )
            for vertex in self.vertices
            if on_segment(start, end, vertex.point)
            or _same_point(vertex.point, start)
            or _same_point(vertex.point, end)
        )
        for edge in list(self.edges):
            kind, point = segment_intersection((start, end), edge.segment)
            if kind != Connection.INTERSECT or point is None:
                continue
            hits.append(
                (
                    _segment_param(start, end, point),
                    point,
                    edge,
                    self._vertex_at(point),
                )
            )
        hits.sort(key=lambda item: item[0])
        unique: list[
            tuple[float, tuple[float, float], Edge | None, Vertex | None]
        ] = []
        for item in hits:
            if unique and _same_point(unique[-1][1], item[1]):
                param, point, hit_edge, vertex = unique[-1]
                if vertex is None:
                    vertex = item[3]
                if hit_edge is None:
                    hit_edge = item[2]
                unique[-1] = (param, point, hit_edge, vertex)
                continue
            unique.append(item)
        return unique

    def _segment_hits_face(self, face: Face, segment: LineType) -> bool:
        """Return True if ``segment`` meets a boundary edge of ``face``."""
        start = segment[0][:2]
        end = segment[1][:2]
        for half_edge in self._face_boundary_halfedges(face):
            origin = half_edge.origin
            destination = half_edge.destination
            if origin is None or destination is None:
                raise ValueError("half-edge endpoints are not set")
            kind, point = segment_intersection(
                (start, end), (origin.point, destination.point)
            )
            if kind == Connection.INTERSECT and point is not None:
                return True
        return False

    def _boolean(self, other: DCEL, keep: Callable[[bool, bool], bool]) -> DCEL:
        """Overlay then keep faces according to ``keep(in_a, in_b)``."""
        overlaid = overlay(self, other)
        keep_flags: dict[int, bool] = {}
        for face in overlaid.bounded_faces:
            sample = overlaid.face_centroid(face)
            in_a = not self.is_outer_face(self.locate_face(sample))
            in_b = not other.is_outer_face(other.locate_face(sample))
            keep_flags[id(face)] = keep(in_a, in_b)
        changed = True
        while changed:
            changed = False
            for face in list(overlaid.bounded_faces):
                if id(face) not in keep_flags or not keep_flags[id(face)]:
                    continue
                for neighbor in overlaid.adjacent_faces(face):
                    if neighbor is overlaid.outer_face:
                        continue
                    if id(neighbor) not in keep_flags:
                        continue
                    if not keep_flags[id(neighbor)]:
                        continue
                    if not overlaid.are_adjacent(face, neighbor):
                        continue
                    survivor = overlaid.merge_faces(face, neighbor)
                    keep_flags[id(survivor)] = True
                    changed = True
                    break
                if changed:
                    break
        rings = []
        for face in overlaid.bounded_faces:
            if id(face) not in keep_flags or not keep_flags[id(face)]:
                continue
            if face.half_edge is None:
                raise ValueError("kept face has no boundary")
            rings.append(_cycle_points(face.half_edge))
            rings.extend(_cycle_points(hole) for hole in face.inner_boundaries)
        result = DCEL()
        if rings:
            result.build_from_polygons(rings)
        return result


def _closed_ring(
    polygon: Sequence[PointType], abs_tol: float | None = None
) -> list[tuple[float, float]]:
    """Return a ring without a repeated closing vertex."""
    ring = [(float(point[0]), float(point[1])) for point in polygon]
    if len(ring) >= 2 and _same_point(ring[0], ring[-1], abs_tol=abs_tol):
        ring = ring[:-1]
    if not ring:
        raise ValueError("polygon has no vertices")
    return ring


def _rings_match(
    ring_a: Sequence[PointType],
    ring_b: Sequence[PointType],
    abs_tol: float | None = None,
) -> bool:
    """Return True if two rings match up to rotation and direction."""
    if len(ring_a) != len(ring_b):
        return False
    if not ring_a:
        return True
    n = len(ring_a)
    for direction in (ring_b, list(reversed(ring_b))):
        for offset in range(n):
            if all(
                _same_point(ring_a[i], direction[(offset + i) % n], abs_tol)
                for i in range(n)
            ):
                return True
    return False


def _polygon_rings(
    polygons: Shape | Group | Sequence[Shape] | Sequence[Sequence[PointType]],
) -> list[Sequence[PointType]]:
    """Normalize shapes, groups, or rings to a list of vertex rings."""
    if isinstance(polygons, Group):
        return list(polygons.all_polygons())
    if isinstance(polygons, Shape):
        return [polygons.vertices]
    if not isinstance(polygons, Sequence):
        raise TypeError(
            "polygons must be a Shape, Group, sequence of shapes, "
            "or sequence of vertex rings"
        )
    if len(polygons) == 0:
        return []
    first = polygons[0]
    if isinstance(first, Shape):
        return [item.vertices for item in polygons]
    if isinstance(first, Group):
        rings: list[Sequence[PointType]] = []
        for item in polygons:
            if isinstance(item, Group):
                rings.extend(item.all_polygons())
            elif isinstance(item, Shape):
                rings.append(item.vertices)
            else:
                rings.append(item)
        return rings
    sample = first[0]
    if isinstance(sample, Sequence) and not isinstance(sample, (str, bytes)):
        return list(polygons)
    return [polygons]


def _same_cycle(he_a: HalfEdge, he_b: HalfEdge) -> bool:
    """Return True if ``he_b`` lies on the cycle of ``he_a``."""
    for half_edge in _cycle_halfedges(he_a):
        if half_edge is he_b:
            return True
    return False


def _edge_with_points(
    mesh: DCEL, point_a: PointType, point_b: PointType
) -> Edge:
    """Return the edge whose endpoints match ``point_a`` and ``point_b``."""
    target = {tuple(point_a[:2]), tuple(point_b[:2])}
    for edge in mesh.edges:
        ends = {edge.origin.point, edge.destination.point}
        if ends == target:
            return edge
    raise ValueError("no edge with those endpoints")


def overlay(dcel_a: DCEL, dcel_b: DCEL) -> DCEL:
    """Return the overlay of two subdivisions.

    Args:
        dcel_a: First mesh.
        dcel_b: Second mesh.

    Returns:
        DCEL: Combined arrangement.

    Examples:
        >>> from simetri.geom.polygons.dcel import DCEL, overlay
        >>> a = DCEL()
        >>> _ = a.build_from_polygons([[(0, 0), (80, 0), (80, 80), (0, 80)]])
        >>> b = DCEL()
        >>> _ = b.build_from_polygons([[(40, 40), (60, 40), (60, 60), (40, 60)]])
        >>> combined = overlay(a, b)
        >>> combined.validate()
        True
    """
    return dcel_a.overlay(dcel_b)


def union(dcel_a: DCEL, dcel_b: DCEL) -> DCEL:
    """Return the union of two meshes' bounded regions."""
    return dcel_a.union(dcel_b)


def difference(dcel_a: DCEL, dcel_b: DCEL) -> DCEL:
    """Return ``dcel_a`` minus ``dcel_b``."""
    return dcel_a.difference(dcel_b)


def symmetric_difference(dcel_a: DCEL, dcel_b: DCEL) -> DCEL:
    """Return the symmetric difference of two meshes."""
    return dcel_a.symmetric_difference(dcel_b)


def intersection(dcel_a: DCEL, dcel_b: DCEL) -> DCEL:
    """Return the intersection of two meshes' bounded regions."""
    return dcel_a.intersection(dcel_b)

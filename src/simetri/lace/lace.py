"""Simetri library's interlace objects."""

import logging
from itertools import combinations
from typing import Iterator, List, Any, Union
from collections import OrderedDict

import networkx as nx
import numpy as np

from numpy import isclose

from ..graphics.shape import Shape, custom_attributes
from ..graphics.batch import Batch

from ..colors import colors
from ..helpers.geometry import (
    connected_pairs,
    polygon_area,
    distance,
    double_offset_polygons,
    get_polygons,
    double_offset_polylines,
    intersection2,
    right_handed,
    polygon_cg,
    round_point,
    offset_polygon,
    offset_polygon_points,
    convex_hull,
    close_points2,
    polygon_center,
)
from ..helpers.graph import get_cycles
from ..graphics.common import get_defaults, common_properties
from ..graphics.all_enums import Types, Connection
from ..canvas.style_map import shape_style_map, ShapeStyle
from ..canvas.canvas import Canvas
from ..settings.settings import defaults
from ..helpers.utilities import flatten, group_into_bins
from ..helpers.validation import validate_args


def _set_style(obj: Any, attribs):
    for attr in attribs:
        setattr(obj, attr, getattr(defaults["style"], attr))


array = np.array


# Lace (Batch)
#     parallel_polyline_list (list)
#         parallel_polyline1 (ParallelPolyline-Batch-PARALLELPOLYLINE)
#             polyline_list (list)
#             |    polyline1 (Polyline-Shape)
#             |        divisions (list)
#             |            division1 (Division-Shape)
#             |                p1 (tuple)
#             |                p2 (tuple)
#             |                sections (list)
#             |                    section1(Section-Shape)
#             |                        p1 (tuple)
#             |                        p2
#             |                        is_overlap (bool)
#             |                        overlap (Overlap-Batch)
#             |overlaps (list)
#             |   overlap1(Overlap-Batch)
#             |       divisions (list)
#             |           division1(Division-Shape)
#             |               p1 (tuple)
#             |               p2 (tuple)
#             |               sections (list)
#             |                   section1(Section-Shape)
#             |                       start (Intersection-Shape)
#             |                       end (Intersection-Shape)
#             |                       overlap
#             |
#             |
#             |fragments
#                 fragment1(Shape)
#                     divisions
#                         division1(Shape)
#                             p1
#                             p2
#                             sections
#                                 section1

#             plaits (list)
#                 plait1 (Plait-Shape)

# All objects in this module is a subclass of the Shape or Batch class.
# They are used to compute the interlacing patterns.

#  Example of a Lace object.

#     /\
#    //\\ /\
#   //  \//\\
#  //   /\\ \\
# //   // \\ \\
# \\   \\ //  //
#  \\   \//  //
#   \\  //\\//
#    \\//  \/
#     \/

# Example of a ParallelPolylines object. The lace object above has two
# ParallelPolylines objects. Main polylines are not shown, they are
# located in the middle of the offset polylines.

#         /\
#        //\\
#       //  \\
#      //    \\
#      \\    //
#       \\  //
#        \\//
#         \/

# Example of a Fragment object.
# They are polygons or polylines.
# The lace object above has three fragments.
# This is the middle fragment.

#         /\
#        /  \
#        \  /
#         \/

# Example of a plait object. They are polylines.
# Used for drawing under/over interlacing.

#         /\
#         \ \
#          \ \
#          / /
#         / /
#         \/

# Example of an Overlap object.
# The lace object above has two overlap regions.

#         /\
#         \/

# Example of a Polyline object.

#          /\
#         /  \
#        /    \
#       /      \
#      /        \
#      \        /
#       \      /
#        \    /
#         \  /
#          \/

# Example of an Division object.
# Each polyline is made up of one or more "Division" objects.
# Each division is divided into sections
# Maybe "Divisions" would be a better name instead of "Division"?
#            *
#             \
#              *
#               \
#                *
#                 \
#                  *
# Example of a Section object.
# Each division is made up of one or more sections.
# Sections have intersection points at their ends.
#                 *
#                  \
#                   *
# * intersections have a point, division1, division2 attributes.
# """

# class Intersection(Shape):
#     """
#    *     *
#     \   /
#      \ /    *
#       *    /     Example: Three inersecting divisions.
#      /  \ /      One division has three sections and four intersections.
#     /    *       The other two divisions have two sections and three
#    *    / \      intersections.
#        /   \
#       /     *
#      *

#     Intersection of two divisions. They are at the endpoints of Section
#     objects. A division can have multiple sections and multiple
#     intersections. They can be located at the end of a division.

#     If the division is connected to another division, that division is the division2
#     attribute of the intersection object.

#     If the division is not connected to another division, division2 is None.

#     If the intersection is at the end of a division, "endpoint" is True.

#     Intersection objects are mostly used internally but they can be used
#     for custom purposes. They inherit from Shape class, all attributes
#     of Shape objects are applicable to Intersection objects. Their
#     subtype is Types.INTERSECTION. They also take part in the Overlap
#     objects. Overlap objects are formed by four connected sections, thus
#     each overlap object have four intersection objects attached to these
#     sections. Like all other objects in the Simetri library,
#     intersection objects have unique ID's (intersection.id) and
#     "visible" attributes that can be set to True or False for each
#     individual instance. If the "visible" attribute is False then the
#     object is not drawn. The default value for intersection.visible is False.

#     Arguments:
#     ----------
#         point: (tuple) (x, y) coordinates of the intersection point.
#         division1: (Division) First division. division2: (Division) Second division. If the
#         intersection is at the end of a division, then division2 is None.
#         endpoint: (bool) If the intersection is at the end of a division,
#         then endpoint is True, False otherwise.
#           **kwargs:
#         (dict) Additional attributes for cosmetic/drawing purposes.

#     Example:
#     --------
#         x_point = Intersection((200.2, -10.51), division1, division2)

#


class Intersection(Shape):
    """Intersection of two divisions. They are at the endpoints of Section
    objects. A division can have multiple sections and multiple
    intersections. They can be located at the end of a division."""

    def __init__(
        self,
        point: tuple,
        division1: "Division",
        division2: "Division" = None,
        endpoint: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            [point], xform_matrix=None, subtype=Types.INTERSECTION, **kwargs
        )
        self._point = point
        self.division1 = division1
        self.division2 = division2
        self.overlap = None
        self.endpoint = endpoint
        self.division = None  # used for fragment divisions' DCEL structure

        common_properties(self, id_only=True)

    def _update(self, xform_matrix: array, reps=0):
        if reps == 0:
            self.xform_matrix = self.xform_matrix @ xform_matrix
            res = self
        else:
            res = []
            for _ in range(reps):
                shape = self.copy()
                shape._update(xform_matrix)
                res.append(shape)

        return res

    def copy(self):
        intersection = Intersection(self.point, self.division1, self.division2)
        for attrib in shape_style_map:
            setattr(intersection, attrib, getattr(self, attrib))
        custom_attribs = custom_attributes(self)
        for attrib in custom_attribs:
            setattr(intersection, attrib, getattr(self, attrib))
        return intersection

    @property
    def point(self):
        """Return the intersection point."""
        return list(np.array([*self._point, 1.0]) @ self.xform_matrix)[:2]

    def __str__(self):
        return (
            f"Intersection({[round(x, defaults['n_round']) for x in self.point]}, "
            f"{tuple(list([self.division1, self.division2]))}"
        )

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return close_points2(self.point, other.point, dist2=defaults["dist_tol"] ** 2)


class Partition(Shape):
    """These are the polygons of the non-interlaced geometry.
    Fragments and partitions are scaled versions of each other."""

    def __init__(self, points, **kwargs):
        super().__init__(points, **kwargs)
        self.subtype = Types.PART
        self.area = polygon_area(self.vertices)
        self.CG = polygon_cg(self.vertices)
        common_properties(self)

    def __str__(self):
        return f"Part({self.vertices})"

    def __repr__(self):
        return self.__str__()


class Fragment(Shape):
    """A Fragment is a collection of section objects that are connected
    to each other. These sections are already defined. They belong to
    the polyline objects in a lace. Fragements can be open or closed.
    They are created by the lace object.
    """

    # To do: we should create the fragments similar to the overlaps
    # each fragment should have a list of sections
    def __init__(self, points, **kwargs):
        super().__init__(points, **kwargs)
        self.subtype = Types.FRAGMENT
        self.area = polygon_area(self.vertices)
        self.sections = []
        self.intersections = []
        self.inner_lines = []
        self._divisions = []
        self.CG = polygon_cg(self.vertices)
        common_properties(self)

    def __str__(self):
        return f"Fragment({self.vertices})"

    def __repr__(self):
        return self.__str__()

    @property
    def divisions(self):
        """Return the divisions of the fragment."""
        return self._divisions

    @property
    def center(self):
        """Return the center of the fragment."""
        return self.CG

    def _set_divisions(self, dist_tol=None):
        if dist_tol is None:
            dist_tol = defaults["dist_tol"]
        dist_tol2 = dist_tol * dist_tol  # squared distance tolerance
        d_points__section = {}
        for section in self.sections:
            start = section.start.point
            end = section.end.point
            start = round(start[0], 2), round(start[1], 2)
            end = round(end[0], 2), round(end[1], 2)
            d_points__section[(start, end)] = section
            d_points__section[(end, start)] = section

        coord_pairs = connected_pairs(self.vertices)
        self._divisions = []
        for pair in coord_pairs:
            x1, y1 = round_point(pair[0])
            x2, y2 = round_point(pair[1])
            division = Division((x1, y1), (x2, y2))
            division.section = d_points__section[((x1, y1), (x2, y2))]
            division.fragment = self
            start_point = round_point(division.section.start.point)
            end_point = round_point(division.section.end.point)
            if close_points2(start_point, (x1, y1), dist2=dist_tol2):
                division.intersections = [division.section.start, division.section.end]
                division.section.start.division = division
            elif close_points2(end_point, (x1, y1), dist2=dist_tol2):
                division.intersections = [division.section.end, division.section.start]
                division.section.end.division = division
            else:
                raise ValueError("Division does not match section")
            self._divisions.append(division)
        n = len(self._divisions)
        for i, division in enumerate(self._divisions):
            division.prev = self._divisions[i - 1]
            division.next = self._divisions[(i + 1) % n]

    def _set_twin_divisions(self):
        for division in self.divisions:
            section = division.section
            if section.twin and section.twin.fragment:
                twin_fragment = section.twin.fragment
                distances = []
                for _, division2 in enumerate(twin_fragment.divisions):
                    dist = distance(
                        division.section.mid_point, division2.section.mid_point
                    )
                    distances.append((dist, division2))
                distances.sort(key=lambda x: x[0])
                division.twin = distances[0][1]


class Section(Shape):
    """A section is a line segment between two intersections.
    A division can have multiple sections. Sections are used to
    draw the over/under plaits.
    """

    def __init__(
        self,
        start: Intersection = None,
        end: Intersection = None,
        is_overlap: bool = False,
        overlap: "Overlap" = None,
        is_over: bool = False,
        twin: "Section" = None,
        fragment: "Fragment" = None,
        **kwargs,
    ):
        super().__init__([start.point, end.point], subtype=Types.SECTION, **kwargs)
        self.start = start
        self.end = end
        self.is_overlap = is_overlap
        self.overlap = overlap
        self.is_over = is_over
        self.twin = twin
        self.fragment = fragment
        super().__init__([start.point, end.point], subtype=Types.SECTION, **kwargs)
        self.length = distance(self.start.point, self.end.point)
        self.mid_point = [
            (self.start.point[0] + self.end.point[0]) / 2,
            (self.start.point[1] + self.end.point[1]) / 2,
        ]
        common_properties(self)

    def copy(self):
        overlap = self.overlap.copy() if self.overlap else None
        start = self.start.copy()
        end = self.end.copy()
        section = Section(start, end, self.is_overlap, overlap, self.is_over)

        return section

    def end_point(self):
        """Return the end point of the section."""
        if self.start.endpoint:
            res = self.start
        elif self.end.endpoint:
            res = self.end
        else:
            res = None

        return res

    def __str__(self):
        return f"Section({self.start}, {self.end})"

    def __repr__(self):
        return self.__str__()

    @property
    def is_endpoint(self):
        """Return True if the section is an endpoint."""
        return self.start.endpoint or self.end.endpoint


class Overlap(Batch):
    """An overlap is a collection of four connected sections."""

    def __init__(
        self,
        intersections: list[Intersection] = None,
        sections: list[Section] = None,
        visited=False,
        drawable=True,
        **kwargs,
    ):
        self.intersections = intersections
        self.sections = sections
        super().__init__(sections, **kwargs)
        self.subtype = Types.OVERLAP
        self.visited = visited
        self.drawable = drawable
        common_properties(self)

    def __str__(self):
        return f"Overlap({self.id})"

    def __repr__(self):
        return f"Overlap({self.id})"


class Division(Shape):
    """A division is a line segment between two intersections."""

    def __init__(self, p1, p2, xform_matrix=None, **kwargs):
        super().__init__([p1, p2], subtype=Types.DIVISION, **kwargs)
        self.p1 = p1
        self.p2 = p2
        self.intersections = []
        self.twin = None  # used for fragment divisions only
        self.section = None  # used for fragment divisions only
        self.fragment = None  # used for fragment divisions only
        self.next = None  # used for fragment divisions only
        self.prev = None  # used for fragment divisions only
        self.sections = []
        super().__init__(
            [p1, p2], subtype=Types.DIVISION, xform_matrix=xform_matrix, **kwargs
        )
        common_properties(self)

    def _update(self, xform_matrix, reps=0):
        if reps == 0:
            self.xform_matrix = self.xform_matrix @ xform_matrix
            res = self
        else:
            res = []
            for _ in range(reps):
                shape = self.copy()
                shape._update(xform_matrix)
                res.append(shape)

        return res

    def __str__(self):
        return (
            f"Division(({self.p1[0]:.2f}, {self.p1[1]:.2f}), "
            f"({self.p2[0]:.2f}, {self.p2[1]:.2f}))"
        )

    def __repr__(self):
        return (
            f"Division(({self.p1[0]:.2f}, {self.p1[1]:.2f}), "
            f"({self.p2[0]:.2f}, {self.p2[1]:.2f}))"
        )

    def copy(
        self,
        section: Section = None,
        twin: Section = None,
    ):
        division = Division(self.p1[:], self.p2[:], np.copy(self.xform_matrix))
        for attrib in shape_style_map:
            setattr(division, attrib, getattr(self, attrib))
        division.twin = twin
        division.section = section
        division.fragment = self.fragment
        division.next = self.next
        division.prev = self.prev
        division.sections = [x.copy() for x in self.sections]
        custom_attributes_ = custom_attributes(self)
        for attrib in custom_attributes_:
            setattr(division, attrib, getattr(self, attrib))
        return division

    def _merged_sections(self):
        chains = []
        chain = [self.intersections[0]]
        sections = self.sections[:]
        for section in sections:
            if not section.is_over:
                if section.start.id == chain[-1].id:
                    chain.append(section.end)
                else:
                    chains.append(chain)
                    chain = [section.start, section.end]
        if chain not in chains:
            chains.append(chain)
        return chains

    def _sort_intersections(self) -> None:
        self.intersections.sort(key=lambda x: distance(self.p1, x.point))

    def is_connected(self, other: "Division") -> bool:
        """Return True if the division is connected to another division."""
        return self.p1 in other.end_points or self.p2 in other.end_points

    @property
    def end_points(self):
        """Return the end points of the division."""
        return [self.p1, self.p2]

    @property
    def start(self) -> Intersection:
        """Return the start intersection of the division."""
        return self.intersections[0]

    @property
    def end(self) -> Intersection:
        """Return the end intersection of the division."""
        return self.intersections[-1]


class Polyline(Shape):
    """
    Connected points, similar to Shape objects.
    They can be closed or open.
    They are defined by a sequence of points.
    They have divisions, sections, and intersections.
    """

    def __init__(self, points, closed=True, xform_matrix=None, **kwargs):
        self.__dict__["style"] = ShapeStyle()
        self.__dict__["_style_map"] = shape_style_map
        self._set_aliases()
        self.closed = closed
        kwargs["subtype"] = Types.POLYLINE
        super().__init__(points, closed=closed, xform_matrix=xform_matrix, **kwargs)
        self._set_divisions()
        if not self.closed:
            self._set_intersections()
        common_properties(self)

    def _update(self, xform_matrix, reps=0):
        if reps == 0:
            self.xform_matrix = self.xform_matrix @ xform_matrix
            for division in self.divisions:
                division._update(xform_matrix, reps=reps)
            res = self
        else:
            res = []
            for _ in range(reps):
                shape = self.copy()
                shape._update(xform_matrix)
                res.append(shape)

        return res

    def __str__(self):
        return f"Polyline({self.final_coords[:, :2]})"

    def __repr__(self):
        return self.__str__()

    def iter_sections(self) -> Iterator:
        for division in self.divisions:
            yield from division.sections

    def iter_intersections(self):
        for division in self.divisions:
            yield from division.intersections

    @property
    def intersections(self):
        res = []
        for division in self.divisions:
            res.extend(division.intersections)
        return res

    @property
    def area(self):
        """Return the area of the polygon."""
        return polygon_area(self.vertices)

    @property
    def sections(self):
        sections = []
        for division in self.divisions:
            sections.extend(division.sections)
        return sections

    @property
    def divisions(self):
        return self.__dict__["divisions"]

    def _set_divisions(self):
        vertices = self.vertices
        if self.closed:
            vertices = list(vertices) + [vertices[0]]
        pairs = connected_pairs(vertices)
        divisions = [Division(p1, p2) for p1, p2 in pairs]
        self.__dict__["divisions"] = divisions

    def _set_intersections(self):
        """fake intersections for open lines"""
        division1 = self.divisions[0]
        division2 = self.divisions[-1]
        x1 = Intersection(division1.p1, division1, None, True)
        division1.intersections = [x1]
        x2 = Intersection(division2.p2, division2, None, True)
        if division1.id == division2.id:
            division1.intersections.append(x2)
        else:
            division2.intersections = [x2]


class ParallelPolyline(Batch):
    """A ParallelPolylines is a collection of parallel Polylines.
    They are defined by a main polyline and a list of offset
    values (that can be negative or positive).
    """

    def __init__(
        self,
        polyline,
        offset,
        lace,
        under=False,
        closed=True,
        dist_tol=None,
        **kwargs,
    ):
        if dist_tol is None:
            dist_tol = defaults["dist_tol"]
        dist_tol2 = dist_tol * dist_tol
        self.polyline = polyline
        self.offset = offset
        self.dist_tol = dist_tol
        self.dist_tol2 = dist_tol2
        self.closed = closed
        if "canvas" in kwargs:
            self._canvas = kwargs["canvas"]
        else:
            self._canvas = None
        if "canvas" in kwargs:
            self._canvas = kwargs["canvas"]
            kwargs.pop("canvas")
        self._set_offset_polylines()
        self.polyline_list = [self.polyline] + self.offset_poly_list
        super().__init__(self.polyline_list, **kwargs)
        self.subtype = Types.PARALLELPOLYLINE
        self.overlaps = None
        self.under = under
        common_properties(self)

    @property
    def sections(self) -> List[Section]:
        sects = []
        for polyline in self.polyline_list:
            sects.extend(polyline.sections)
        return sects

    def _set_offset_polylines(self):
        polyline = self.polyline
        if self.closed:
            vertices = list(polyline.vertices)
            vertices = vertices + [vertices[0]]
            offset_polygons = double_offset_polygons(
                vertices, self.offset, dist_tol=self.dist_tol, canvas=self._canvas
            )
        else:
            offset_polylines = double_offset_polylines(polyline.vertices, self.offset)
        polylines = []
        if self.closed:
            for polygon in offset_polygons:
                polylines.append(Polyline(polygon, closed=self.closed))
        else:
            for polyline in offset_polylines:
                polylines.append(Polyline(polyline, closed=self.closed))

        self.offset_poly_list = polylines


class Lace(Batch):
    """
    A Lace is a collection of ParallelPolylines objects.
    They are used to create interlace patterns.
    """

    # @timing
    def __init__(
        self,
        polygon_shapes: Union[Batch, list[Shape]] = None,
        polyline_shapes: Union[Batch, list[Shape]] = None,
        offset: float = 2,
        rtol: float = None,
        swatch: list = None,
        breakpoints: list = None,
        plait_color: colors.Color = None,
        draw_fragments: bool = True,
        palette: list = None,
        color_step: int = 1,
        with_plaits: bool = True,
        area_threshold: float = None,
        radius_threshold: float = None,
        **kwargs,
    ) -> None:
        validate_args(kwargs, shape_style_map)
        (
            rtol,
            swatch,
            plait_color,
            draw_fragments,
            area_threshold,
            radius_threshold,
        ) = get_defaults(
            [
                "rtol",
                "swatch",
                "plait_color",
                "draw_fragments",
                "area_threshold",
                "radius_threshold",
            ],
            [
                rtol,
                swatch,
                plait_color,
                draw_fragments,
                area_threshold,
                radius_threshold,
            ],
        )
        if polygon_shapes:
            polygon_shapes = polygon_shapes.merge_shapes()
            self.polygon_shapes = self._check_polygons(polygon_shapes)
        else:
            self.polygon_shapes = []
        if polyline_shapes:
            polyline_shapes = polyline_shapes.merge_shapes()
            self.polyline_shapes = self._check_polylines(polyline_shapes)
        else:
            self.polyline_shapes = []
        if not self.polygon_shapes and not self.polyline_shapes:
            msg = "Lace.__init__ : No polygons or polylines found."
            logging.error(msg)
            raise ValueError(msg)
        self.polyline_shapes = polyline_shapes
        self.offset = offset
        self.main_intersections = None
        self.offset_intersections = None
        self.xform_matrix = np.eye(3)
        self.rtol = rtol
        self.swatch = swatch
        self.breakpoints = breakpoints
        self.plait_color = plait_color
        self.draw_fragments = draw_fragments
        self.palette = palette
        self.color_step = color_step
        self.with_plaits = with_plaits
        self.d_intersections = {}  # key, value:intersection.id, intersection
        self.d_connections = {}
        self.plaits = []
        self.overlaps = []
        self._groups = None
        self.area_threshold = area_threshold
        self.radius_threshold = radius_threshold
        if kwargs and "debug" in kwargs:
            if kwargs["debug"]:
                self._canvas = Canvas()
        else:
            self._canvas = None
        if kwargs and "_copy" in kwargs:
            # pass the pre-computed values
            for k, v in kwargs:
                if k == "_copy":
                    continue
                else:
                    setattr(self, k, v)
        else:
            self._set_polyline_list()  # main divisions are set here along with polylines
            # polyline.divisions is the list of Division objects
            if self._canvas:
                for polyline in self.polyline_list:
                    self._canvas.draw(polyline, fill=False)
            self._set_parallel_poly_list()
            if self._canvas:
                self._canvas.new_page()
                for polyline in self.parallel_poly_list:
                    self._canvas.draw(polyline, fill=False)
            # start_time2 = time.perf_counter()
            self._set_intersections()
            # end_time2 = time.perf_counter()
            # print(
            #     f"Lace.__init__ intersections computed in {end_time2 - start_time2:0.4f} seconds"
            # )

            self._set_overlaps()
            self._set_twin_sections()
            self._set_fragments()
            if not self.polyline_shapes:
                self._set_outline()
            # self._set_partitions()
            self._set_over_under()
            if self.with_plaits:
                self.set_plaits()
            # self._set_convex_hull()
            # self._set_concave_hull()
            # self._set_fragment_groups()
            # self._set_partition_groups()
            self._b_box = None
        elements = [polyline for polyline in self.parallel_poly_list] + self.fragments
        if self._canvas:
            self._canvas.save("c:/tmp/lace_debug.pdf")
        if "debug" in kwargs:
            kwargs.pop("debug")
        super().__init__(elements, **kwargs)
        if kwargs and "_copy" not in kwargs:
            for k, v in kwargs.items():
                if k in shape_style_map:
                    setattr(self, k, v)  # todo: we should check for valid values here
                else:
                    raise AttributeError(f"{k}. Invalid attribute!")
        self.subtype = Types.LACE
        common_properties(self)

    @property
    def center(self):
        return self.outline.CG

    @property
    def fragment_groups(self):
        center = self.center
        radius_frag = []
        for fragment in self.fragments:
            radius = int(distance(center, fragment.CG))
            for rad, frag in radius_frag:
                if abs(radius - rad) <= 2:
                    radius = rad
                    break
            radius_frag.append((radius, fragment))

        radius_frag.sort(key=lambda x: x[0])

        d_groups = {}
        for i, (radius, fragment) in enumerate(radius_frag):
            if i == 0:
                d_groups[radius] = [fragment]
            else:
                if radius in d_groups:
                    d_groups[radius].append(fragment)
                else:
                    d_groups[radius] = [fragment]

        return d_groups

    def _check_polygons(self, polygon_shapes):
        if isinstance(polygon_shapes, Batch):
            polygon_shapes = polygon_shapes.all_shapes
        for polygon in polygon_shapes:
            if len(polygon.primary_points) < 3:
                msg = "Lace.__init__ found polygon with less than 3 points."
                logging.error(msg)
                raise ValueError(msg)
            if not polygon.closed:
                msg = "Lace.__init__ : Invalid polygons"
                logging.error(msg)
                raise ValueError(msg)
            if polygon.primary_points[0] != polygon.primary_points[-1]:
                polygon.primary_points.append(polygon.primary_points[0])
        # check if the polygons are clockwise
        for polygon in polygon_shapes:
            if not right_handed(polygon.vertices):
                polygon.primary_points.reverse()
                # msg = "Lace._check_polygons reversed clockwise points."
                # logging.info(msg)

        return polygon_shapes

    def _check_polylines(self, polyline_shapes):
        if isinstance(polyline_shapes, Batch):
            polyline_shapes = polyline_shapes.all_shapes
        for polyline in polyline_shapes:
            if len(polyline.primary_points) < 2:
                msg = "Lace.__init__ found polyline with less than 2 points."
                logging.error(msg)
                raise ValueError(msg)

        return polyline_shapes

    def _update(self, xform_matrix, reps=0):
        if reps == 0:
            self.xform_matrix = self.xform_matrix @ xform_matrix
            for polygon in self.polygon_shapes:
                polygon._update(xform_matrix)

            if self.polyline_shapes:
                for polyline in self.polyline_shapes:
                    polyline._update(xform_matrix)

            for polyline in self.parallel_poly_list:
                polyline._update(xform_matrix)

            for fragment in self.fragments:
                fragment._update(xform_matrix)

            for intersection in self.main_intersections:
                intersection._update(xform_matrix)

            for intersection in self.offset_intersections:
                intersection._update(xform_matrix)

            for overlap in self.overlaps:
                overlap._update(xform_matrix)

            for plait in self.plaits:
                plait._update(xform_matrix)

            return self
        else:
            res = []
            for _ in range(reps):
                shape = self.copy()
                shape._update(xform_matrix)
                res.append(shape)
            return res

    # @timing
    def _set_twin_sections(self):
        for par_poly in self.parallel_poly_list:
            poly1, poly2 = par_poly.offset_poly_list
            for i, sec in enumerate(poly1.iter_sections()):
                sec1 = sec
                sec2 = poly2.sections[i]
                sec1.twin = sec2
                sec2.twin = sec1

    # @timing
    def _set_partitions(self):
        for fragment in self.fragments:
            self.partitions = []
            for fragment in self.fragments:
                partition = Shape(offset_polygon(fragment.vertices, self.offset))
                self.partitions.append(partition)

    # To do: This doesn't work if we have polyline shapes!
    def _set_outline(self):
        # outline is a special fragment that covers the whole lace
        areas = []
        for fragment in self.fragments:
            areas.append((fragment.area, fragment))
        areas.sort(reverse=True, key=lambda x: x[0])
        self.outline = areas[0][1]
        self.fragments.remove(self.outline)
        # perimenter is the outline of the partitions
        self.perimeter = Shape(offset_polygon(self.outline.vertices, -self.offset))
        # skeleton is the input polylines that the lace is based on
        self.skeleton = Batch(self.polyline_list)
        if self._canvas:
            self._canvas.new_page()
            for fragment in self.fragments:
                self._canvas.draw(fragment)

    def set_fragment_groups(self):
        # to do : handle repeated code. same in _set_partition_groups
        areas = []
        for i, fragment in enumerate(self.fragments):
            areas.append((fragment.area, i))
        areas.sort()
        bins = group_into_bins(areas, self.area_threshold)
        self.fragments_by_area = OrderedDict()
        for i, bin in enumerate(bins):
            area_values = [x[0] for x in bin]
            key = sum([x[0] for x in bin]) / len(bin)
            fragments = []
            for area, ind in areas:
                if area in area_values:
                    fragments.append(self.fragments[ind])
            self.fragments_by_area[key] = fragments

        radii = []
        for i, fragment in enumerate(self.fragments):
            radii.append((distance(self.center, fragment.CG), i))
        radii.sort()
        bins = group_into_bins(radii, self.radius_threshold)
        self.fragments_by_radius = OrderedDict()
        for i, bin in enumerate(bins):
            radius_values = [x[0] for x in bin]
            key = sum([x[0] for x in bin]) / len(bin)
            fragments = []
            for radius, ind in radii:
                if radius in radius_values:
                    fragments.append(self.fragments[ind])
            self.fragments_by_radius[key] = fragments
        if self._canvas:
            self._canvas.new_page()
            for fragment in fragments:
                self._canvas.draw(fragment)

    # @timing
    def _set_partition_groups(self):
        # to do : handle repeated code. same in set_fragment_groups
        areas = []
        for i, partition in enumerate(self.partitions):
            areas.append((partition.area, i))
        areas.sort()
        bins = group_into_bins(areas, self.area_threshold)
        self.partitions_by_area = OrderedDict()
        for i, bin_ in enumerate(bins):
            area_values = [x[0] for x in bin_]
            key = sum([x[0] for x in bin_]) / len(bin_)
            partitions = []
            for area, ind in areas:
                if area in area_values:
                    partitions.append(self.partitions[ind])
            self.partitions_by_area[key] = partitions

        radii = []
        for i, partition in enumerate(self.partitions):
            CG = polygon_center(partition.vertices)
            radii.append((distance(self.center, CG), i))
        radii.sort()
        bins = group_into_bins(radii, self.radius_threshold)
        self.partitions_by_radius = OrderedDict()
        for i, bin_ in enumerate(bins):
            radius_values = [x[0] for x in bin_]
            key = sum([x[0] for x in bin_]) / len(bin_)
            partitions = []
            for radius, ind in radii:
                if radius in radius_values:
                    partitions.append(self.partitions[ind])
            self.partitions_by_radius[key] = partitions

    # @timing
    def _set_fragments(self):
        G = nx.Graph()
        for section in self.iter_offset_sections():
            if section.is_overlap:
                continue
            G.add_edge(section.start.id, section.end.id, section=section)

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

        self.fragments = fragments

    def _set_concave_hull(self):
        self.concave_hull = self.outline.vertices

    def _set_convex_hull(self):
        self.convex_hull = convex_hull(self.outline.vertices)

    def copy(self):
        class Dummy(Lace):
            pass

        # we need to copy the polyline_list and parallel_poly_list
        for polyline in self.polyline_list:
            polyline.copy()

    def get_sketch(self):
        """
        Create and return a Sketch object. Sketch is a Batch object
        with Shape elements corresponding to the vertices of the plaits
        and fragments of the Lace instance. They have 'plaits' and
        'fragments' attributes to hold lists of Shape objects populated
        with plait and fragment vertices of the Lace instance
        respectively. They are used for drawing multiple copies of the
        original lace pattern. They are light-weight compared to the
        Lace objects since they only contain sufficient data to draw the
        lace objects. Hundreds of these objects can be used to create
        wallpaper patterns or other patterns without having to contain
        unnecessary data. They do not share points with the original
        Lace object.

        Arguments:
        ----------
            None

        Prerequisites:
        --------------
            * A lace object to be copied.

        Side effects:
        -------------
            None

        Return:
        --------
            A Sketch object.

        """
        fragments = []
        for fragment in self.fragments:
            polygon = Shape(fragment.vertices)
            polygon.fill = True
            polygon.subtype = Types.FRAGMENT
            fragments.append(polygon)

        plaits = []
        for plait in self.plaits:
            polygon = Shape(plait.vertices)
            polygon.fill = True
            polygon.subtype = Types.PLAIT
            plaits.append(polygon)

        sketch = Batch((fragments + plaits))
        sketch.fragments = fragments
        sketch.plaits = plaits
        sketch.outline = self.outline
        sketch.subtype = Types.SKETCH

        sketch.draw_plaits = True
        sketch.draw_fragments = True
        return sketch

    def group_fragments(self, tol=None):
        """_group the fragments by the number of vertices and the area."""
        if tol is None:
            tol = defaults["tol"]
        frags = self.fragments
        vert_groups = [
            [frag for frag in frags if len(frag.vertices) == n]
            for n in set([len(f.vertices) for f in frags])
        ]
        groups = []
        for group in vert_groups:
            areas = [
                [f for f in group if isclose(f.area, area, rtol=tol)]
                for area in set([frag.area for frag in group])
            ]
            areas.sort(key=lambda x: x[0].area, reverse=True)
            groups.append(areas)
        groups.sort(key=lambda x: x[0][0].area, reverse=True)

        return groups

    def get_fragment_cycles(self):
        """
        Iterate over the offset sections and create a graph of the
        intersections (start and end of the sections). Then find the
        cycles in the graph. self.d_intersections is used to map
        the graph nodes to the actual intersection points.
        """
        graph_edges = []
        for section in self.iter_offset_sections():
            if section.is_overlap:
                continue
            graph_edges.append((section.start.id, section.end.id))

        return get_cycles(graph_edges)

    def _set_inner_lines(self, item, n, offset, line_color=colors.blue, line_width=1):
        for i in range(n):
            vertices = item.vertices
            dist_tol = defaults["dist_tol"]
            offset_poly = offset_polygon_points(
                vertices, offset * (i + 1), dist_tol=dist_tol
            )
            shape = Shape(offset_poly)
            shape.fill = False
            shape.line_width = line_width
            shape.line_color = line_color
            item.inner_lines.append(shape)

    def set_plait_lines(self, n, offset, line_color=colors.blue, line_width=1):
        """Create offset lines inside the plaits of the lace."""
        for plait in self.plaits:
            plait.inner_lines = []
            self._set_inner_lines(plait, n, offset, line_color, line_width)

    def set_fragment_lines(
        self,
        n: int,
        offset: float,
        line_color: colors.Color = colors.blue,
        line_width=1,
    ) -> None:
        """
        Create offset lines inside the fragments of the lace.
        """
        for fragment in self.fragments:
            fragment.inner_lines = []
            self._set_inner_lines(fragment, n, offset, line_color, line_width)

    @property
    def all_divisions(self) -> List:
        """
        Return a list of all the divisions (both main and offset) in the lace.
        """
        res = []
        for parallel_polyline in self.parallel_poly_list:
            for polyline in parallel_polyline.polyline_list:
                res.extend(polyline.divisions)
        return res

    def iter_main_intersections(self) -> Iterator:
        """'
        Iterate over the main intersections.
        """
        for ppoly in self.parallel_poly_list:
            for division in ppoly.polyline.divisions:
                yield from division.intersections

    def iter_offset_intersections(self) -> Iterator:
        """
        Iterate over the offset intersections.
        """
        for ppoly in self.parallel_poly_list:
            for poly in ppoly.offset_poly_list:
                for division in poly.divisions:
                    yield from division.intersections

    def iter_offset_sections(self) -> Iterator:
        """
        Iterate over the offset sections.
        """
        for ppoly in self.parallel_poly_list:
            for poly in ppoly.offset_poly_list:
                for division in poly.divisions:
                    yield from division.sections

    def iter_main_sections(self) -> Iterator:
        """Iterate over the main sections."""
        for ppoly in self.parallel_poly_list:
            for division in ppoly.polyline.divisions:
                yield from division.sections

    def iter_offset_divisions(self) -> Iterator:
        """
        Iterate over the offset divisions.
        """
        for ppoly in self.parallel_poly_list:
            for poly in ppoly.offset_poly_list:
                yield from poly.divisions

    def iter_main_divisions(self) -> Iterator:
        """
        Iterate over the main divisions.
        """
        for ppoly in self.parallel_poly_list:
            yield from ppoly.polyline.divisions

    @property
    def main_divisions(self) -> List[Division]:
        """Main divisions are the divisions of the main polyline."""
        res = []
        for parallel_polyline in self.parallel_poly_list:
            res.extend(parallel_polyline.polyline.divisions)
        return res

    @property
    def offset_divisions(self) -> List[Division]:
        """Offset divisions are the divisions of the offset polylines."""
        res = []
        for parallel_polyline in self.parallel_poly_list:
            for polyline in parallel_polyline.offset_poly_list:
                res.extend(polyline.divisions)
        return res

    @property
    def intersections(self) -> List[Intersection]:
        """Return all the intersections in the parallel_poly_list."""
        res = []
        for parallel_polyline in self.parallel_poly_list:
            for polyline in parallel_polyline.polyline_list:
                res.extend(polyline.intersections)
        return res

    # @timing
    def _set_polyline_list(self):
        """
        Populate the self.polyline_list list with Polyline objects.

        * Internal use only.

        Arguments:
        ----------
            None

        Return:
        --------
            None

        Prerequisites:
        --------------

            * self.polygon_shapes and/or self.polyline_shapes must be
              established.
        """

        self.polyline_list = []
        if self.polygon_shapes:
            for polygon in self.polygon_shapes:
                self.polyline_list.append(Polyline(polygon.vertices, closed=True))
        if self.polyline_shapes:
            for polyline in self.polyline_shapes:
                self.polyline_list.append(Polyline(polyline.vertices, closed=False))

    # @timing
    def _set_parallel_poly_list(self):
        """
        Populate the self.parallel_poly_list list with ParallelPolyline
        objects.

        Arguments:
        ----------
            None

        Return:
        --------
            None

        Prerequisites:
        --------------
            * self.polygon_shapes and/or self.polyline_shapes must be
              established prior to this.
            * Parallel polylines are created by offsetting the original
              polygon and polyline shapes in two directions using the
              self.offset value.

        Notes:
        ------
            This method is called by the Lace constructor.  It is not
            for users to call directly. Without this method, the Lace
            object cannot be created.
        """

        self.parallel_poly_list = []
        if self.polyline_list:
            for _, polyline in enumerate(self.polyline_list):
                if self._canvas:
                    self._canvas.new_page()
                    self._canvas.draw(polyline, fill=False)
                self.parallel_poly_list.append(
                    ParallelPolyline(
                        polyline,
                        self.offset,
                        lace=self,
                        closed=polyline.closed,
                        dist_tol=defaults["dist_tol"],
                        canvas=self._canvas,
                    )
                )

    # @timing
    def _set_overlaps(self):
        """
        Populate the self.overlaps list with Overlap objects. Side
        effects listed below.

        Arguments:
        ----------
            None

        Return:
        --------
            None

        Side Effects:
        -------------
            * self.overlaps is populated with Overlap objects.
            * Section objects' overlap attribute is populated with the
              corresponding Overlap object that they are a part of. Not
              all sections will have an overlap.

        Prerequisites:
        --------------
            self.polyline and self.parallel_poly_list must be populated.
            self.main_intersections, self.offset_sections and
            self.d_intersections must be populated prior to creating
            the overlaps.

        Notes:
        ------
            This method is called by the Lace constructor.  It is not
            for users to call directly.
            Without this method, the Lace object cannot be created.

        """
        G = nx.Graph()
        for section in self.iter_offset_sections():
            if section.is_overlap:
                G.add_edge(section.start.id, section.end.id, section=section)
        cycles = nx.cycle_basis(G)
        for cycle in cycles:
            cycle.append(cycle[0])
            edges = connected_pairs(cycle)
            sections = [G.edges[edge]["section"] for edge in edges]
            s_intersections = set()
            for section in sections:
                s_intersections.add(section.start.id)
                s_intersections.add(section.end.id)
            intersections = [self.d_intersections[i] for i in s_intersections]
            overlap = Overlap(intersections=intersections, sections=sections)
            for section in sections:
                section.overlap = overlap
            for edge in edges:
                section = G.edges[edge]["section"]
                section.start.overlap = overlap
                section.end.overlap = overlap
            self.overlaps.append(overlap)
        if self._canvas:
            self._canvas.new_page()
        for overlap in self.overlaps:
            for section in overlap.sections:
                if section.is_over:
                    line_width = 3
                else:
                    line_width = 1
                line = Shape(
                    [section.start.point, section.end.point], line_width=line_width
                )
                if self._canvas:
                    self._canvas.draw(line, line_width=3)
        for division in self.offset_divisions:
            p1 = division.start.point
            p2 = division.end.point
            if self._canvas:
                self._canvas.draw(Shape([p1, p2]))

    def set_plaits(self):
        """
        Populate the self.plaits list with Plait objects. Plaits are
        optional for drawing. They form the under/over interlacing. They
        are created if the "with_plaits" argument is set to be True in
        the constructor. with_plaits is True by default but this can be
        changed by setting the auto_plaits value to False in the
        settings.py This method can be called by the user to create the
        plaits after the creation of the Lace object if they were not
        created initally.

        * Can be called by users.

        Arguments:
        ----------
            None

        Return:
        --------
            None

        Side Effects:
        -------------
            * self.plaits is populated with Plait objects.

        Prerequisites:
        --------------
            self.polyline and self.parallel_poly_list must be populated.
            self.divisions and self.intersections must be populated.
            self.overlaps must be populated.

        Where used:
        ----------
            Lace.__init__

        Notes:
        ------
            This method is called by the Lace constructor.  It is not
            for users to call directly. Without this method, the Lace
            object cannot be created.
        """
        if self.plaits:
            logging.warning("Plaits already exist. set lace.plaits = []")
            return
        plait_sections = []
        if self._canvas:
            self._canvas.new_page()
        for division in self.iter_offset_divisions():
            if self._canvas:
                start = division.start.point
                end = division.end.point
                self._canvas.draw(Shape([start, end]))
            merged_sections = division._merged_sections()
            for merged in merged_sections:
                plait_sections.append((merged[0], merged[-1]))

        # connect the open ends of the polyline_shapes
        for ppoly in self.parallel_poly_list:
            if not ppoly.closed:
                polyline1 = ppoly.offset_poly_list[0]
                polyline2 = ppoly.offset_poly_list[1]
                p1_start_x = polyline1.intersections[0]
                p1_end_x = polyline1.intersections[-1]
                p2_start_x = polyline2.intersections[0]
                p2_end_x = polyline2.intersections[-1]

                plait_sections.append((p1_start_x, p2_start_x))
                plait_sections.append((p1_end_x, p2_end_x))
        if self._canvas:
            self._canvas.new_page()
        for sec in self.iter_offset_sections():
            if not sec.is_over and sec.is_overlap:
                plait_sections.append((sec.start, sec.end))
                if self._canvas:
                    self._canvas.draw(
                        Shape([sec.start.point, sec.end.point]), line_width=2
                    )
            if sec.is_over and sec.is_overlap:
                if self._canvas:
                    self._canvas.draw(
                        Shape([sec.start.point, sec.end.point]), line_width=4
                    )
        if self._canvas:
            self._canvas.new_page()
            for section in plait_sections:
                self._canvas.draw(
                    Shape([section[0].point, section[1].point]), line_width=2
                )
        graph_edges = [(r[0].id, r[1].id) for r in plait_sections]
        cycles = get_cycles(graph_edges)
        plaits = []
        count = 0
        for cycle in cycles:
            cycle = connected_pairs(cycle)
            dup = cycle[1:]
            plait = [cycle[0][0], cycle[0][1]]
            for _ in range(len(cycle) - 1):
                last = plait[-1]
                for edge in dup:
                    if edge[0] == last:
                        plait.append(edge[1])
                        dup.remove(edge)
                        break
                    if edge[1] == last:
                        plait.append(edge[0])
                        dup.remove(edge)
                        break
            plaits.append(plait)
            count += 1
        d_x = self.d_intersections
        for plait in plaits:
            intersections = [d_x[x] for x in plait]
            vertices = [x.point for x in intersections]
            if not right_handed(vertices):
                vertices.reverse()
                plait.reverse()
            shape = Shape(vertices)
            shape.intersections = intersections
            shape.fill_color = colors.gold
            shape.inner_lines = None
            shape.subtype = Types.PLAIT
            self.plaits.append(shape)
        if self._canvas:
            self._canvas.new_page()
            for plait in self.plaits:
                self._canvas.draw(plait, fill=True)

    # @timing
    def _set_intersections(self):
        """
        Compute all intersection points (by calling all_intersections)
        among the divisions of the polylines (both main and offset).
        Populate the self.main_intersections and
        self.offset_intersections lists with Intersection objects. This
        method is called by the Lace constructor and customized to be
        used with Lace objects only. Without this method, the Lace
        object cannot be created.

        * Internal use only!

        Arguments:
        ----------
            None

        Return:
        --------
            None

        Side Effects:
        -------------
            * self.main_intersections are populated.
            * self.offset_intersections are populated.
            * "sections" attribute of the divisions are populated.
            * "is_overlap" attribute of the sections are populated.
            * "intersections" attribute of the divisions are populated.
            * "endpoint" attribute of the intersections are populated.

        Where used:
        -----------
            Lace.__init__

        Prerequisites:
        --------------
            * self.main_divisions must be populated.
            * self.offset_divisions must be populated.
            * Two endpoint intersections of the divisions must be set.

        Notes:
        ------
            This method is called by the Lace constructor.  It is not
            for users to call directly. Without this method, the Lace
            object cannot be created.
            Works only for regular under/over interlacing.
        """
        # set intersections for the main polylines
        main_divisions = self.main_divisions
        offset_divisions = self.offset_divisions
        self.main_intersections = all_intersections(
            main_divisions, self.d_intersections, self.d_connections
        )

        # set sections for the main divisions
        self.main_sections = []
        for division in main_divisions:
            division.intersections[0].endpoint = True
            division.intersections[-1].endpoint = True
            segments = connected_pairs(division.intersections)
            division.sections = []
            for i, segment in enumerate(segments):
                section = Section(*segment)
                if i % 2 == 1:
                    section.is_overlap = True
                division.sections.append(section)
                self.main_sections.append(section)
        if self._canvas:
            for section in self.main_sections:
                self._canvas.draw(
                    Shape([section.start.point, section.end.point]), line_width=2
                )
        # set intersections for the offset polylines
        self.offset_intersections = all_intersections(
            offset_divisions, self.d_intersections, self.d_connections
        )
        if self._canvas:
            self._canvas.new_page()
            for intersection in self.offset_intersections:
                self._canvas.circle(intersection.point, 2)
            for division in offset_divisions:
                p1 = division.start.point
                p2 = division.end.point
                self._canvas.draw(Shape([p1, p2]))
        # set sections for the offset divisions
        self.offset_sections = []
        for i, division in enumerate(offset_divisions):
            division.intersections[0].endpoint = True
            division.intersections[-1].endpoint = True
            lines = connected_pairs(division.intersections)
            division.sections = []
            for j, line in enumerate(lines):
                section = Section(*line)
                if j % 2 == 1:
                    section.is_overlap = True
                division.sections.append(section)
                self.offset_sections.append(section)
        if self._canvas:
            self._canvas.new_page()
            for section in self.offset_sections:
                self._canvas.draw(
                    Shape([section.start.point, section.end.point]), line_width=2
                )
        # to do: intersect main divisions with the offset divisions

    def _all_polygons(self, polylines, rtol=None):
        """Return a list of polygons from a list of lists of points.
        polylines: [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)], ...]
        return [[(x1, y1), (x2, y2), (x3, y3), ...], ...]
        """
        if rtol is None:
            rtol = self.rtol
        return get_polygons(polylines, rtol)

    # @timing
    def _set_over_under(self):
        def next_poly(exclude):
            for ppoly in self.parallel_poly_list:
                poly1, poly2 = ppoly.offset_poly_list
                if poly1 in exclude:
                    continue
                ind = 0
                for sec in poly1.iter_sections():
                    if sec.is_overlap:
                        if sec.overlap.drawable and sec.overlap.visited:
                            even_odd = ind % 2 == 1
                            return (poly1, poly2, even_odd)
                        ind += 1
            return (None, None, None)

        for ppoly in self.parallel_poly_list:
            poly1, poly2 = ppoly.offset_poly_list
            exclude = []
            even_odd = 0
            while poly1:
                ind = 0
                for i, division in enumerate(poly1.divisions):
                    for j, section in enumerate(division.sections):
                        if self._canvas:
                            if j == 0:
                                rad = 1
                            else:
                                rad = 2
                            self._canvas.circle(section.mid_point, rad)
                        if section.is_overlap:
                            if section.overlap is None:
                                msg = (
                                    "Overlap section in the lace has no "
                                    "overlap object.\n"
                                    "Try different offset value and/or "
                                    "tolerance."
                                )
                                logging.error(msg)
                                raise RuntimeError(msg)
                            section.overlap.visited = True
                            if ind % 2 == even_odd:
                                if section.overlap.drawable:
                                    section.overlap.drawable = False
                                    section.is_over = True
                                    section2 = poly2.divisions[i].sections[j]
                                    section2.is_over = True
                            ind += 1
                exclude.extend([poly1, poly2])
                poly1, poly2, even_odd = next_poly(exclude)

    def fragment_edge_graph(self) -> nx.Graph:
        """
        Return a networkx graph of the connected fragments.
        If two fragments have a "common" division then they are connected.
        """
        G = nx.Graph()
        fragments = [(f.area, f) for f in self.fragments]
        fragments.sort(key=lambda x: x[0], reverse=True)
        for fragment in [f[1] for f in fragments[1:]]:
            for division in fragment.divisions:
                if division.twin and division.twin.fragment:
                    fragment2 = division.twin.fragment
                    G.add_node(fragment.id, fragment=fragment)
                    G.add_node(fragment2.id, fragment=fragment2)
                    G.add_edge(fragment.id, fragment2.id, edge=division)
        return G

    def fragment_vertex_graph(self) -> nx.Graph:
        """
        Return a networkx graph of the connected fragments.
        If two fragments have a "common" vertex then they are connected.
        """

        def get_neighbours(intersection):
            division = intersection.division
            if not division.next.twin:
                return None
            fragments = [division.fragment]
            start_division_id = division.id
            if division.next.twin:
                twin_id = division.next.twin.id
            else:
                return None
            while twin_id != start_division_id:
                if division.next.twin:
                    if division.fragment.id not in [x.id for x in fragments]:
                        fragments.append(division.fragment)
                    twin_id = division.next.twin.id
                    division = division.next.twin
                else:
                    if division.next.fragment.id not in [x.id for x in fragments]:
                        fragments.append(division.next.fragment)
                    break

            return fragments

        neighbours = []
        for fragment in self.fragments:
            for intersection in fragment.intersections:
                neighbours.append(get_neighbours(intersection))
        G = self.fragment_edge_graph()
        G2 = nx.Graph()
        for n in neighbours:
            if n:
                if len(n) > 2:
                    for pair in combinations(n, 2):
                        ids = tuple([x.id for x in pair])
                        if ids not in G.edges:
                            G2.add_node(ids[0], fragment=pair[0])
                            G2.add_node(ids[1], fragment=pair[1])
                            G2.add_edge(*ids)
        return G2


def all_intersections(
    division_list: list[Division],
    d_intersections: dict[int, Intersection],
    d_connections: dict[frozenset, Intersection],
    loom=False,
) -> list[Intersection]:
    """
    Find all intersections of the given divisions. Sweep-line algorithm
    without a self-balancing tree. Instead of a self-balancing tree,
    it uses a numpy array to sort and filter the divisions. For the
    number of divisions that are commonly needed in a lace, this is
    sufficiently fast. It is also more robust and much easier to
    understand and debug. Tested with tens of thousands of divisions but
    not millions. The book has a section on this algorithm.
    simetri.geometry.py has another version (called
    all_intersections) for finding intersections among a given
    list of divisions.

    Arguments:
    ----------
        division_list: list of Division objects.

    Side Effects:
    -------------
        * Modifies the given division objects (in the division_list) in place
            by adding the intersections to the divisions' "intersections"
            attribute.
        * Updates the d_intersections
        * Updates the d_connections


    Return:
    --------
        A list of all intersection objects among the given division
        list.
    """
    # register fake intersections at the endpoints of the open lines
    for division in division_list:
        if division.intersections:
            for x in division.intersections:
                d_intersections[x.id] = x

    # All objects are assigned an integer id attribute when they are created
    division_array = array(
        [flatten(division.vertices) + [division.id] for division in division_list]
    )
    n_divisions = division_array.shape[0]  # number of divisions
    # precompute the min and max x and y values for each division
    # these will be used with the sweep line algorithm
    xmin = np.minimum(division_array[:, 0], division_array[:, 2]).reshape(
        n_divisions, 1
    )
    xmax = np.maximum(division_array[:, 0], division_array[:, 2]).reshape(
        n_divisions, 1
    )
    ymin = np.minimum(division_array[:, 1], division_array[:, 3]).reshape(
        n_divisions, 1
    )
    ymax = np.maximum(division_array[:, 1], division_array[:, 3]).reshape(
        n_divisions, 1
    )
    division_array = np.concatenate((division_array, xmin, ymin, xmax, ymax), 1)
    d_divisions = {}
    for division in division_list:
        d_divisions[division.id] = division
    i_id, i_xmin, i_ymin, i_xmax, i_ymax = range(4, 9)  # column indices
    # sort by xmin values
    division_array = division_array[division_array[:, i_xmin].argsort()]
    intersections = []
    for i in range(n_divisions):
        x1, y1, x2, y2, id1, sl_xmin, sl_ymin, sl_xmax, sl_ymax = division_array[i, :]
        division1_vertices = [x1, y1, x2, y2]
        start = i + 1  # search should start from the next division
        # filter the array by checking if the divisions' bounding-boxes are
        # overlapping with the bounding-box of the current division
        candidates = division_array[start:, :][
            (
                (
                    (division_array[start:, i_xmax] >= sl_xmin)
                    & (division_array[start:, i_xmin] <= sl_xmax)
                )
                & (
                    (division_array[start:, i_ymax] >= sl_ymin)
                    & (division_array[start:, i_ymin] <= sl_ymax)
                )
            )
        ]
        for candid in candidates:
            id2 = candid[i_id]
            division2_vertices = candid[:4]
            if loom:
                x1, y1 = division1_vertices[:2]
                x2, y2 = division2_vertices[:2]
                if x1 == x2 or y1 == y2:
                    continue
            connection_type, x_point = intersection2(
                *division1_vertices, *division2_vertices
            )
            if connection_type not in [
                Connection.DISJOINT,
                Connection.NONE,
                Connection.PARALLEL,
            ]:
                division1, division2 = d_divisions[int(id1)], d_divisions[int(id2)]
                x_point__e1_2 = frozenset((division1.id, division2.id))
                if x_point__e1_2 not in d_connections:
                    inters_obj = Intersection(x_point, division1, division2)
                    d_intersections[inters_obj.id] = inters_obj
                    d_connections[x_point__e1_2] = inters_obj
                    division1.intersections.append(inters_obj)
                    division2.intersections.append(inters_obj)
                    intersections.append(inters_obj)

    for division in division_list:
        division._sort_intersections()
    return intersections


def merge_nodes(
    division_list: list[Division],
    d_intersections: dict[int, Intersection],
    d_connections: dict[frozenset, Intersection],
    loom=False,
) -> list[Intersection]:
    """
    Find all intersections of the given divisions. Sweep-line algorithm
    without a self-balancing tree. Instead of a self-balancing tree,
    it uses a numpy array to sort and filter the divisions. For the
    number of divisions that are commonly needed in a lace, this is
    sufficiently fast. It is also more robust and much easier to
    understand and debug. Tested with tens of thousands of divisions but
    not millions. The book has a section on this algorithm.
    simetri.geometry.py has another version (called
    all_intersections) for finding intersections among a given
    list of divisions.

    Arguments:
    ----------
        division_list: list of division objects.

    Side Effects:
    -------------
        * Modifies the given division objects (in the division_list) in place
            by adding the intersections to the divisions' "intersections"
            attribute.
        * Updates the d_intersections
        * Updates the d_connections


    Return:
    --------
        A list of all intersection objects among the given division
        list.
    """
    # register fake intersections at the endpoints of the open lines
    for division in division_list:
        if division.intersections:
            for x in division.intersections:
                d_intersections[x.id] = x

    # All objects are assigned an integer id attribute when they are created
    division_array = array(
        [flatten(division.vertices) + [division.id] for division in division_list]
    )
    n_divisions = division_array.shape[0]  # number of divisions
    # precompute the min and max x and y values for each division
    # these will be used with the sweep line algorithm
    xmin = np.minimum(division_array[:, 0], division_array[:, 2]).reshape(
        n_divisions, 1
    )
    xmax = np.maximum(division_array[:, 0], division_array[:, 2]).reshape(
        n_divisions, 1
    )
    ymin = np.minimum(division_array[:, 1], division_array[:, 3]).reshape(
        n_divisions, 1
    )
    ymax = np.maximum(division_array[:, 1], division_array[:, 3]).reshape(
        n_divisions, 1
    )
    division_array = np.concatenate((division_array, xmin, ymin, xmax, ymax), 1)
    d_divisions = {}
    for division in division_list:
        d_divisions[division.id] = division
    i_id, i_xmin, i_ymin, i_xmax, i_ymax = range(4, 9)  # column indices
    # sort by xmin values
    division_array = division_array[division_array[:, i_xmin].argsort()]
    intersections = []
    for i in range(n_divisions):
        x1, y1, x2, y2, id1, sl_xmin, sl_ymin, sl_xmax, sl_ymax = division_array[i, :]
        division1_vertices = [x1, y1, x2, y2]
        start = i + 1  # search should start from the next division
        # filter the array by checking if the divisions' bounding-boxes are
        # overlapping with the bounding-box of the current division
        candidates = division_array[start:, :][
            (
                (
                    (division_array[start:, i_xmax] >= sl_xmin)
                    & (division_array[start:, i_xmin] <= sl_xmax)
                )
                & (
                    (division_array[start:, i_ymax] >= sl_ymin)
                    & (division_array[start:, i_ymin] <= sl_ymax)
                )
            )
        ]
        for candid in candidates:
            id2 = candid[i_id]
            division2_vertices = candid[:4]
            if loom:
                x1, y1 = division1_vertices[:2]
                x2, y2 = division2_vertices[:2]
                if x1 == x2 or y1 == y2:
                    continue
            connection_type, x_point = intersection2(
                *division1_vertices, *division2_vertices
            )
            if connection_type not in [
                Connection.DISJOINT,
                Connection.NONE,
                Connection.PARALLEL,
            ]:
                division1, division2 = d_divisions[int(id1)], d_divisions[int(id2)]
                x_point__e1_2 = frozenset((division1.id, division2.id))
                if x_point__e1_2 not in d_connections:
                    inters_obj = Intersection(x_point, division1, division2)
                    d_intersections[inters_obj.id] = inters_obj
                    d_connections[x_point__e1_2] = inters_obj
                    division1.intersections.append(inters_obj)
                    division2.intersections.append(inters_obj)
                    intersections.append(inters_obj)

    for division in division_list:
        division._sort_intersections()
    return intersections

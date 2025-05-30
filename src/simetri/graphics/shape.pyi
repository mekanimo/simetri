"""Let the IDEs know about the dynamically created attributes of the Shape class."""

from typing import Sequence, Union, List, Tuple

import numpy as np
from numpy import array
from typing_extensions import Self

from .all_enums import *
from .bbox import BoundingBox
from .common import Point, Line

from .points import Points
from .batch import Batch
from ..colors.colors import Color

class Shape:
    """The main class for all geometric entities in Simetri.

    A Shape is created by providing a sequence of points (a sequence of (x, y) coordinates).
    If a style argument (a ShapeStyle object) is provided, then its style attributes override
    the default values the Shape object would assign. Additional attributes (e.g. line_width, fill_color, line_style)
    may be provided.

    """
    def __init__(
        self,
        points: Sequence[Point] = None,
        closed: bool = False,
        xform_matrix: np.array = None,
        **kwargs,
    ) -> None:
        """Initialize a Shape object.

        Args:
            points (Sequence[Point], optional): The points that make up the shape.
            closed (bool, optional): Whether the shape is closed. Defaults to False.
            xform_matrix (np.array, optional): The transformation matrix. Defaults to None.
            **kwargs (dict): Additional attributes for the shape. Common kwargs are, style, line_width, fill, fill_color, line_color, fill_alpha, etc.
        Raises:
            ValueError: If the provided subtype is not valid.
        """
        self.line_alpha: float = 1,
        self.line_cap: LineCap = LineCap.BUTT,
        self.line_color: Color = Color(0.0, 0.0, 0.0),
        self.line_dash_array: Union[list, LineDashArray] = None,
        self.line_dash_phase: float = 0,
        self.double: None = None,
        self.double_distance: float = 2,
        self.draw_fillets: bool = False,
        self.draw_markers: bool = False,
        self.fillet_radius: float = None,
        self.line_join: LineJoin = LineJoin.MITER,
        self.marker_color: Color = Color(0.0, 0.0, 0.0),
        self.marker_type: MarkerType = MarkerType.FCIRCLE,
        self.marker_size: float = 3,
        self.markers_only: bool = False,
        self.line_miter_limit: float = 10,
        self.smooth: bool = False,
        self.stroke: bool = True,
        self.line_width: float = 1,
        self.fill_alpha: float = 1,
        self.back_style: BackStyle = BackStyle.COLOR,
        self.fill_color: Color = Color(0.0, 0.0, 0.0),
        self.fill: bool = True,
        self.grid_alpha: float = 0.5,
        self.grid_back_color: Color = Color(1.0, 1.0, 1.0),
        self.grid_line_color: Color = Color(0.573, 0.584, 0.569),
        self.grid_line_width: float = 0.5,
        self.fill_mode: FillMode = FillMode.EVENODD,
        self.pattern_angle: float = 0,
        self.pattern_color: Color = Color(0.0, 0.0, 0.0),
        self.pattern_distance: float = 3,
        self.pattern_line_width: float = 0,
        self.pattern_type: PatternType = PatternType.HORIZONTAL_LINES,
        self.pattern_points: int = 5,
        self.pattern_radius: float = 10,
        self.pattern_x_shift: float = 0,
        self.pattern_y_shift: float = 0,
        self.shade_bottom_color: Color = Color(1.0, 1.0, 1.0),
        self.shade_color_wheel: bool = False,
        self.shade_color_wheel_black: bool = False,
        self.shade_color_wheel_white: bool = False,
        self.shade_inner_color: Color = Color(1.0, 1.0, 1.0),
        self.shade_left_color: Color = Color(0.0, 0.0, 0.0),
        self.shade_lower_left_color: Color = Color(0.0, 0.0, 0.0),
        self.shade_lower_right_color: Color = Color(1.0, 1.0, 1.0),
        self.shade_middle_color: Color = Color(1.0, 1.0, 1.0),
        self.shade_middle_color: Color = Color(1.0, 1.0, 1.0),
        self.shade_outer_color: Color = Color(1.0, 1.0, 1.0),
        self.shade_middle_color: Color = Color(1.0, 1.0, 1.0),
        self.shade_middle_color: Color = Color(1.0, 1.0, 1.0),
        self.shade_middle_color: Color = Color(1.0, 1.0, 1.0),
        self.shade_outer_color: Color = Color(1.0, 1.0, 1.0),
        self.shade_middle_color: Color = Color(1.0, 1.0, 1.0),
        self.shade_middle_color: Color = Color(1.0, 1.0, 1.0),
        self.shade_outer_color: Color = Color(1.0, 1.0, 1.0),
        self.shade_middle_color: Color = Color(1.0, 1.0, 1.0),
        self.shade_middle_color: Color = Color(1.0, 1.0, 1.0),
        self.shade_outer_color: Color = Color(1.0, 1.0, 1.0),
        self.shade_right_color: Color = Color(1.0, 1.0, 1.0),
        self.shade_top_color: Color = Color(0.0, 0.0, 0.0),
        self.shade_upper_left_color: Color = Color(0.0, 0.0, 0.0),
        self.shade_upper_right_color: Color = Color(1.0, 1.0, 1.0)

        self.type: Types = Types.SHAPE
        self.subtype: Types = None
        self.primary_points: Points = None
        self.dist_tolerance: float = None
        self.dist_tolerance2: float = None
        self.primary_points: Points = None
        self.closed: bool = False
        self.b_box: BoundingBox = None
        self.xform_matrix: 'ndarray' = None

    def __setattr__(self, name, value):
        """Set an attribute of the shape.

        Args:
            name (str): The name of the attribute.
            value (Any): The value to set.
        """


    def __getattr__(self, name):
        """Retrieve an attribute of the shape.

        Args:
            name (str): The attribute name to return.

        Returns:
            Any: The value of the attribute.

        Raises:
            AttributeError: If the attribute cannot be found.
        """


    def _get_closed(self, points: Sequence[Point], closed: bool):
        """Determine whether the shape should be considered closed.

        Args:
            points (Sequence[Point]): The points that define the shape.
            closed (bool): The user-specified closed flag.

        Returns:
            tuple: A tuple consisting of:
                - bool: True if the shape is closed, False otherwise.
                - list: The (possibly modified) list of points.
        """


    def __len__(self):
        """Return the number of points in the shape.

        Returns:
            int: The number of primary points.
        """


    def __str__(self):
        """Return a string representation of the shape.

        Returns:
            str: A string representation of the shape.
        """


    def __repr__(self):
        """Return a string representation of the shape.

        Returns:
            str: A string representation of the shape.
        """


    def __getitem__(self, subscript: Union[int, slice]):
        """Retrieve point(s) from the shape by index or slice.

        Args:
            subscript (int or slice): The index or slice specifying the point(s) to retrieve.

        Returns:
            Point or list[Point]: The requested point or list of points (after applying the transformation).

        Raises:
            TypeError: If the subscript type is invalid.
        """


    def __setitem__(self, subscript, value):
        """Set the point(s) at the given subscript.

        Args:
            subscript (int or slice): The subscript to set the point(s) at.
            value (Point or list[Point]): The value to set the point(s) to.

        Raises:
            TypeError: If the subscript type is invalid.
        """


    def __delitem__(self, subscript) -> Self:
        """Delete the point(s) at the given subscript.

        Args:
            subscript (int or slice): The subscript to delete the point(s) from.
        """


    def index(self, point: Point, atol=None) -> int:
        """Return the index of the given point.

        Args:
            point (Point): The point to find the index of.

        Returns:
            int: The index of the point.
        """



    def remove(self, point: Point) -> Self:
        """Remove a point from the shape.

        Args:
            point (Point): The point to remove.
        """


    def append(self, point: Point) -> Self:
        """Append a point to the shape.

        Args:
            point (Point): The point to append.
        """


    def insert(self, index: int, point: Point) -> Self:
        """Insert a point at a given index.

        Args:
            index (int): The index to insert the point at.
            point (Point): The point to insert.
        """


    def extend(self, points: Sequence[Point]) -> Self:
        """Extend the shape with a list of points.

        Args:
            points (list[Point]): The points to extend the shape with.
        """


    def pop(self, index: int = -1) -> Point:
        """Pop a point from the shape.

        Args:
            index (int, optional): The index to pop the point from, defaults to -1.

        Returns:
            Point: The popped point.
        """


    def __iter__(self):
        """Return an iterator over the vertices of the shape.

        Returns:
            Iterator[Point]: An iterator over the vertices of the shape.
        """


    def _update(self, xform_matrix: array, reps: int = 0) -> Batch:
        """Used internally. Update the shape with a transformation matrix.

        Args:
            xform_matrix (array): The transformation matrix.
            reps (int, optional): The number of repetitions, defaults to 0.

        Returns:
            Batch: The updated shape or a batch of shapes.
        """


    def __eq__(self, other):
        """Check if the shape is equal to another shape.

        Args:
            other (Shape): The other shape to compare to.

        Returns:
            bool: True if the shapes are equal, False otherwise.
        """


    def __bool__(self):
        """Return whether the shape has any points.

        Returns:
            bool: True if the shape has points, False otherwise.
        """


    def topology(self) -> Topology:
        """Return info about the topology of the shape.

        Returns:
            set: A set of topology values.
        """


    def merge(self, other, dist_tol: float = None) -> Union[Self, None]:
        """Merge two shapes if they are connected. Does not work for polygons.
        Only polyline shapes can be merged together.

        Args:
            other (Shape): The other shape to merge with.
            dist_tol (float, optional): The distance tolerance for merging, defaults to None.

        Returns:
            Shape or None: The merged shape or None if the shapes cannot be merged.
        """


    def connect(self, other) -> Self:
        """Connect two shapes by adding the other shape's vertices to self.

        Args:
            other (Shape): The other shape to connect.
        """


    def _chain_vertices(
        self, verts1: Sequence[Point], verts2: Sequence[Point], dist_tol: float = None
    ) -> Union[List[Point], None]:
        """Chain two sets of vertices if they are connected.

        Args:
            verts1 (list[Point]): The first set of vertices.
            verts2 (list[Point]): The second set of vertices.
            dist_tol (float, optional): The distance tolerance for chaining, defaults to None.

        Returns:
            list[Point] or None: The chained vertices or None if the vertices cannot be chained.
        """


    def _is_polygon(self, vertices: Sequence[Point]) -> bool:
        """Return True if the vertices form a polygon.

        Args:
            vertices (list[Point]): The vertices to check.

        Returns:
            bool: True if the vertices form a polygon, False otherwise.
        """


    def as_graph(self, directed=False, weighted=False, n_round=None) -> 'nx.Graph':
        """Return the shape as a graph object.

        Args:
            directed (bool, optional): Whether the graph is directed, defaults to False.
            weighted (bool, optional): Whether the graph is weighted, defaults to False.
            n_round (int, optional): The number of decimal places to round to, defaults to None.

        Returns:
            Graph: The graph object.
        """


    def as_array(self, homogeneous=False) -> 'ndarray':
        """Return the vertices as an array.

        Args:
            homogeneous (bool, optional): Whether to return homogeneous coordinates, defaults to False.

        Returns:
            ndarray: The vertices as an array.
        """


    def as_list(self) -> List[Point]:
        """Return the vertices as a list of tuples.

        Returns:
            list[tuple]: The vertices as a list of tuples.
        """


    @property
    def final_coords(self) -> 'ndarray':
        """The final coordinates of the shape. primary_points @ xform_matrix.

        Returns:
            ndarray: The final coordinates of the shape.
        """


    @property
    def vertices(self) -> Tuple[Point]:
        """The final coordinates of the shape.

        Returns:
            tuple: The final coordinates of the shape.
        """


    @property
    def vertex_pairs(self) -> List[Tuple[Point, Point]]:
        """Return a list of connected pairs of vertices.

        Returns:
            list[tuple[Point, Point]]: A list of connected pairs of vertices.
        """


    @property
    def orig_coords(self) -> 'ndarray':
        """The primary points in homogeneous coordinates.

        Returns:
            ndarray: The primary points in homogeneous coordinates.
        """


    @property
    def b_box(self) -> BoundingBox:
        """Return the bounding box of the shape.

        Returns:
            BoundingBox: The bounding box of the shape.
        """


    @property
    def area(self) -> float:
        """Return the area of the shape.

        Returns:
            float: The area of the shape.
        """


    @property
    def total_length(self) -> float:
        """Return the total length of the shape.

        Returns:
            float: The total length of the shape.
        """


    @property
    def is_polygon(self) -> bool:
        """Return True if 'closed'.

        Returns:
            bool: True if the shape is closed, False otherwise.
        """


    def clear(self) -> Self:
        """Clear all points and reset the style attributes.

        Returns:
            None
        """


    def count(self, point: Point) -> int:
        """Return the number of times the point is found in the shape.

        Args:
            point (Point): The point to count.

        Returns:
            int: The number of times the point is found in the shape.
        """


    def copy(self) -> 'Shape':
        """Return a copy of the shape.

        Returns:
            Shape: A copy of the shape.
        """


    @property
    def edges(self) -> List[Line]:
        """Return a list of edges.

        Edges are represented as tuples of points:
        edge: ((x1, y1), (x2, y2))
        edges: [((x1, y1), (x2, y2)), ((x2, y2), (x3, y3)), ...]

        Returns:
            list[tuple[Point, Point]]: A list of edges.
        """


    @property
    def segments(self) -> List[Line]:
        """Return a list of edges.

        Edges are represented as tuples of points:
        edge: ((x1, y1), (x2, y2))
        edges: [((x1, y1), (x2, y2)), ((x2, y2), (x3, y3)), ...]

        Returns:
            list[tuple[Point, Point]]: A list of edges.
        """

    def translate(self, dx: float = 0, dy: float = 0, reps: int = 0) -> Self:
        """
        Translates the object by dx and dy.

        Args:
            dx (float): The translation distance along the x-axis.
            dy (float): The translation distance along the y-axis.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The transformed object.
        """

    def translate_along(
        self,
        path: Sequence[Point],
        step: int = 1,
        align_tangent: bool = False,
        scale: float = 1,  # scale factor
        rotate: float = 0,  # angle in radians
    ) -> Self:
        """
        Translates the object along the given curve.
        Every n-th point is used to calculate the translation vector.
        If align_tangent is True, the object is rotated to align with the tangent at each point.
        scale is the scale factor applied at each point.
        rotate is the angle in radians applied at each point.

        Args:
            path (Sequence[Point]): The path to translate along.
            step (int, optional): The step size. Defaults to 1.
            align_tangent (bool, optional): Whether to align the object with the tangent. Defaults to False.
            scale (float, optional): The scale factor. Defaults to 1.
            rotate (float, optional): The rotation angle in radians. Defaults to 0.

        Returns:
            Self: The transformed object.
        """

    def rotate(self, angle: float, about: Point = (0, 0), reps: int = 0) -> Self:
        """
        Rotates the object by the given angle (in radians) about the given point.

        Args:
            angle (float): The rotation angle in radians.
            about (Point, optional): The point to rotate about. Defaults to (0, 0).
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The rotated object.
        """

    def mirror(self, about: Union[Line, Point], reps: int = 0) -> Self:
        """
        Mirrors the object about the given line or point.

        Args:
            about (Union[Line, Point]): The line or point to mirror about.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The mirrored object.
        """

    def glide(self, glide_line: Line, glide_dist: float, reps: int = 0) -> Self:
        """
        Glides (first mirror then translate) the object along the given line
        by the given glide_dist.

        Args:
            glide_line (Line): The line to glide along.
            glide_dist (float): The distance to glide.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The glided object.
        """

    def scale(
        self,
        scale_x: float,
        scale_y: Union[float, None] = None,
        about: Point = (0, 0),
        reps: int = 0,
    ) -> Self:
        """
        Scales the object by the given scale factors about the given point.

        Args:
            scale_x (float): The scale factor in the x direction.
            scale_y (float, optional): The scale factor in the y direction. Defaults to None.
            about (Point, optional): The point to scale about. Defaults to (0, 0).
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The scaled object.
        """

    def shear(self, theta_x: float, theta_y: float, reps: int = 0) -> Self:
        """
        Shears the object by the given angles.

        Args:
            theta_x (float): The shear angle in the x direction.
            theta_y (float): The shear angle in the y direction.
            reps (int, optional): The number of repetitions. Defaults to 0.

        Returns:
            Self: The sheared object.
        """

    def move_to(self, pos: Point, anchor: Anchor = Anchor.CENTER) -> Self:
        """
        Moves the object to the given position by using its center point.

        Args:
            pos (Point): The position to move to.
            anchor (Anchor, optional): The anchor point. Defaults to Anchor.CENTER.

        Returns:
            Self: The moved object.
        """

"""Canvas class for drawing shapes and text on a page. All drawing
operations are handled by the Canvas class. Canvas class can draw all
graphics objects and text objects. It also provides methods for
drawing basic shapes like lines, circles, and polygons.
"""

import os
import webbrowser
import subprocess
from typing import Optional, Any, Tuple, Sequence
from pathlib import Path
from dataclasses import dataclass

from typing_extensions import Self, Union
import networkx as nx
import fitz

from simetri.graphics.affine import (
    rotation_matrix,
    translation_matrix,
    scale_matrix,
    identity_matrix,
)
from simetri.graphics.common import common_properties, _set_Nones, VOID, Point, Vec2
from simetri.graphics.all_enums import Types, Drawable, Result, Anchor
from simetri.settings.settings import defaults
from simetri.graphics.bbox import bounding_box
from simetri.graphics.batch import Batch
from simetri.graphics.shape import Shape
from simetri.colors import colors
from simetri.canvas import draw
from simetri.helpers.utilities import wait_for_file_availability
from simetri.helpers.illustration import logo
from simetri.tikz.tikz import Tex, get_tex_code
from simetri.helpers.validation import validate_args
from simetri.canvas.style_map import canvas_args
from simetri.notebook import display

Color = colors.Color


class Canvas:
    """Canvas class for drawing shapes and text on a page. All drawing
    operations are handled by the Canvas class. Canvas class can draw all
    graphics objects and text objects. It also provides methods for
    drawing basic shapes like lines, circles, and polygons.
    """

    def __init__(
        self,
        size: Vec2 = None,
        back_color: Optional[Color] = None,
        border=None,
        **kwargs,
    ):
        """
        Initialize the Canvas.

        Args:
            size (Vec2, optional): The size of the canvas with canvas.origin at (0, 0).
            back_color (Optional[Color], optional): The background color of the canvas.
            border (Any, optional): The border of the canvas.
            kwargs (dict): Additional keyword arguments.
        """
        validate_args(kwargs, canvas_args)
        _set_Nones(self, ["back_color", "border"], [back_color, border])
        self._size = size
        self.border = border
        self.type = Types.CANVAS
        self.subtype = Types.CANVAS
        self._code = []
        self._font_list = []
        self._pos = [0, 0]
        self._angle = 0
        self._scale = [1, 1]
        self.preamble = defaults["preamble"]
        self.back_color = back_color
        self.pages = [Page(self.size, self.back_color, self.border)]
        self.active_page = self.pages[0]
        self._all_vertices = []
        self.blend_mode = None
        self.blend_group = False
        self.transparency_group = False
        self.alpha = None
        self.line_alpha = None
        self.fill_alpha = None
        self.text_alpha = None
        self.clip = None  # if True then clip the canvas to the mask
        self.mask = None  # Mask object
        self.even_odd_rule = None  # True or False
        self.draw_grid = False
        self._origin = [0, 0]
        common_properties(self)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._xform_matrix = identity_matrix()
        self.tex: Tex = Tex()
        self.render = defaults["render"]
        if self._size is not None:
            x, y = self.origin[:2]
            self._limits  = [x, y, x+self.size[0], y+self.size[1]]
        else:
            self._limits = None

    def __setattr__(self, name, value):
        if hasattr(self, "active_page") and name in ["back_color", "border"]:
            self.active_page.__setattr__(name, value)
            self.__dict__[name] = value
        elif name in ["size", "origin", "limits"]:
            if name == "size":
                type(self).size.fset(self, value)
            elif name == "origin":
                type(self).origin.fset(self, value)
            elif name == "limits":
                type(self).limits.fset(self, value)
        else:
            self.__dict__[name] = value

    def display(self) -> Self:
        """Show the canvas in a notebook cell."""
        display(self)

    @property
    def size(self) -> Vec2:
        """
        The size of the canvas.

        Returns:
            Vec2: The size of the canvas.
        """
        return self._size

    @size.setter
    def size(self, value: Vec2) -> None:
        """
        Set the size of the canvas.

        Args:
            value (Vec2): The size of the canvas.
        """
        if len(value) == 2:
            self._size = value
            x, y = self.origin[:2]
            w, h = value
            self._limits = (x, y, x + w, y + h)
        else:
            raise ValueError("Size must be a tuple of 2 values.")

    @ property
    def origin(self) -> Vec2:
        """
        The origin of the canvas.

        Returns:
            Vec2: The origin of the canvas.
        """
        return self._origin[:2]

    @origin.setter
    def origin(self, value: Vec2) -> None:
        """
        Set the origin of the canvas.

        Args:
            value (Vec2): The origin of the canvas.
        """
        if len(value) == 2:
            self._origin = value
        else:
            raise ValueError("Origin must be a tuple of 2 values.")

    @property
    def limits(self) -> Vec2:
        """
        The limits of the canvas.

        Returns:
            Vec2: The limits of the canvas.
        """
        if self.size is None:
            res = None
        else:
            x, y = self.origin[:2]
            w, h = self.size
            res = (x, y, x + w, y + h)

        return res


    @limits.setter
    def limits(self, value: Vec2) -> None:
        """
        Set the limits of the canvas.

        Args:
            value (Vec2): The limits of the canvas.
        """
        if len(value) == 4:
            x1, y1, x2, y2 = value
            self._size = (x2 - x1, y2 - y1)
            self._origin = (x1, y1)
        else:
            raise ValueError("Limits must be a tuple of 4 values.")

    def arc(
        self,
        center: Point,
        start_angle: float,
        span_angle: float,
        radius_x: float,
        radius_y: float = None,
        rot_angle: float = 0,
        **kwargs,
    ) -> Self:
        """
        Draw an arc with the given center, radius, start angle and end angle.

        Args:
            center (Point): The center of the arc.
            start_angle (float): The start angle of the arc.
            end_angle (float): The end angle of the arc.
            radius_x (float): The radius of the arc.
            radius_y (float, optional): The second radius of the arc, defaults to None.
            rot_angle (float, optional): The rotation angle of the arc, defaults to 0.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        if radius_x is None:
            radius_y = radius
        draw.arc(
            self, center, start_angle, span_angle, radius_x, radius_y, rot_angle, **kwargs
        )
        return self

    def bezier(self, control_points: Sequence[Point], **kwargs) -> Self:
        """
        Draw a bezier curve.

        Args:
            control_points (Sequence[Point]): The control points of the bezier curve.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.bezier(self, control_points, **kwargs)
        return self

    def circle(self, center: Point, radius: float, **kwargs) -> Self:
        """
        Draw a circle with the given center and radius.

        Args:
            center (Point): The center of the circle.
            radius (float): The radius of the circle.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.circle(self, center, radius, **kwargs)
        return self

    def ellipse(
        self, center: Point, width: float, height: float, angle: float = 0, **kwargs
    ) -> Self:
        """
        Draw an ellipse with the given center and radius.

        Args:
            center (Point): The center of the ellipse.
            width (float): The width of the ellipse.
            height (float): The height of the ellipse.
            angle (float, optional): The angle of the ellipse, defaults to 0.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.ellipse(self, center, width, height, angle, **kwargs)
        return self

    def text(
        self,
        text: str,
        pos: Point,
        font_family: str = None,
        font_size: int = None,
        anchor: Anchor = None,
        **kwargs,
    ) -> Self:
        """
        Draw text at the given point.

        Args:
            text (str): The text to draw.
            pos (Point): The position to draw the text.
            font_family (str, optional): The font family of the text, defaults to None.
            font_size (int, optional): The font size of the text, defaults to None.
            anchor (Anchor, optional): The anchor of the text, defaults to None.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        pos = [pos[0], pos[1], 1]
        pos = pos @ self._xform_matrix
        draw.text(
            self,
            txt=text,
            pos=pos,
            font_family=font_family,
            font_size=font_size,
            anchor=anchor,
            **kwargs,
        )
        return self

    def help_lines(
        self,
        pos=(-100, -100),
        width: float = 400,
        height: float = 400,
        spacing=25,
        cs_size: float = 25,
        **kwargs,
    ) -> Self:
        """
        Draw help lines on the canvas.

        Args:
            pos (tuple): The position to start drawing the help lines.
            width (float): The length of the help lines along the x-axis.
            height (float): The length of the help lines along the y-axis.
            spacing (int): The spacing between the help lines.
            cs_size (float): The size of the coordinate system.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.help_lines(self, pos, width, height, spacing, cs_size, **kwargs)
        return self

    def grid(
        self, pos: Point, width: float, height: float, spacing: float, **kwargs
    ) -> Self:
        """
        Draw a grid with the given size and spacing.

        Args:
            pos (Point): The position to start drawing the grid.
            width (float): The length of the grid along the x-axis.
            height (float): The length of the grid along the y-axis.
            spacing (float): The spacing between the grid lines.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.grid(self, pos, width, height, spacing, **kwargs)
        return self

    def line(self, start: Point, end: Point, **kwargs) -> Self:
        """
        Draw a line from start to end.

        Args:
            start (Point): The starting point of the line.
            end (Point): The ending point of the line.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.line(self, start, end, **kwargs)
        return self

    def rectangle(
        self,
        center: Point = (0, 0),
        width: float = 100,
        height: float = 100,
        angle: float = 0,
        **kwargs,
    ) -> Self:
        """
        Draw a rectangle.

        Args:
            center (Point): The center of the rectangle.
            width (float): The width of the rectangle.
            height (float): The height of the rectangle.
            angle (float, optional): The angle of the rectangle, defaults to 0.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.rectangle(self, center, width, height, angle, **kwargs)
        return self

    def square(
        self, center: Point = (0, 0), size: float = 100, angle: float = 0, **kwargs
    ) -> Self:
        """
        Draw a square with the given center and size.

        Args:
            center (Point): The center of the square.
            size (float): The size of the square.
            angle (float, optional): The angle of the square, defaults to 0.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.rectangle(self, center, size, size, angle, **kwargs)
        return self

    def lines(self, points: Sequence[Point], **kwargs) -> Self:
        """
        Draw a polyline through the given points.

        Args:
            points (Sequence[Point]): The points to draw the polyline through.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.lines(self, points, **kwargs)
        return self

    def draw_lace(self, lace: Batch, **kwargs) -> Self:
        """
        Draw the lace.

        Args:
            lace (Batch): The lace to draw.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.draw_lace(self, lace, **kwargs)
        return self

    def draw_dimension(self, dim: Shape, **kwargs) -> Self:
        """
        Draw the dimension.

        Args:
            dim (Shape): The dimension to draw.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.draw_dimension(self, dim, **kwargs)
        return self

    def draw(self, item_s: Union[Drawable, list, tuple], **kwargs) -> Self:
        """
        Draw the item_s. item_s can be a single item or a list of items.

        Args:
            item_s (Union[Drawable, list, tuple]): The item(s) to draw.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        if isinstance(item_s, (list, tuple)):
            for item in item_s:
                draw.draw(self, item, **kwargs)
        else:
            draw.draw(self, item_s, **kwargs)
        return self

    def draw_CS(self, size: float = None, **kwargs) -> Self:
        """
        Draw the Canvas coordinate system.

        Args:
            size (float, optional): The size of the coordinate system, defaults to None.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.draw_CS(self, size, **kwargs)
        return self

    def draw_frame(
        self, margin: Union[float, Sequence] = None, width=None, **kwargs
    ) -> Self:
        """
        Draw a frame around the canvas.

        Args:
            margins (Union[float, Sequence]): The margins of the frame.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        # to do: add shadow and frame color, shadow width. Check canvas.clip.
        if margin is None:
            margin = defaults["canvas_frame_margin"]
        if width is None:
            width = defaults["canvas_frame_width"]
        b_box = bounding_box(self._all_vertices)
        box2 = b_box.get_inflated_b_box(margin)
        box3 = box2.get_inflated_b_box(15)
        shadow = Shape([box3.northwest, box3.southwest, box3.southeast])
        self.draw(shadow, line_color=colors.light_gray,line_width=width)
        self.draw(Shape(box2.corners, closed=True), fill=False, line_width=width)

        return self

    def reset(self) -> Self:
        """
        Reset the canvas to its initial state.

        Returns:
            Self: The canvas object.
        """
        self._code = []
        self._pos = [0, 0]
        self._angle = 0
        self._scale = [1, 1]
        self.preamble = defaults["preamble"]
        self.pages = [Page(self.size, self.back_color, self.border)]
        self.active_page = self.pages[0]
        self._all_vertices = []
        self.tex: Tex = Tex()
        self.clip: bool = False  # if true then clip the canvas to the mask
        self._xform_matrix = identity_matrix()
        self.active_page = self.pages[0]
        self._all_vertices = []
        return self

    def __str__(self) -> str:
        """
        Return a string representation of the canvas.

        Returns:
            str: The string representation of the canvas.
        """
        return "Canvas()"

    def __repr__(self) -> str:
        """
        Return a string representation of the canvas.

        Returns:
            str: The string representation of the canvas.
        """
        return "Canvas()"

    @property
    def xform_matrix(self) -> 'ndarray':
        """
        The transformation matrix of the canvas.

        Returns:
            ndarray: The transformation matrix of the canvas.
        """
        return self._xform_matrix.copy()

    def reset_transform(self) -> Self:
        """
        Reset the transformation matrix of the canvas.
        The canvas origin is at (0, 0) and the orientation angle is 0.
        Transformation matrix is the identity matrix.

        Returns:
            Self: The canvas object.
        """
        self._xform_matrix = identity_matrix()
        self._pos = [0, 0]
        self._angle = 0
        self._scale = [1, 1]

        return self

    def translate(self, x: float, y: float) -> Self:
        """
        Translate the canvas by x and y.

        Args:
            x (float): The translation distance along the x-axis.
            y (float): The translation distance along the y-axis.

        Returns:
            Self: The canvas object.
        """
        self._pos[0] += x
        self._pos[1] += y
        self._xform_matrix = translation_matrix(x, y) @ self._xform_matrix

        return self

    def rotate(self, angle: float) -> Self:
        """
        Rotate the canvas by angle in radians about the origin.

        Args:
            angle (float): The rotation angle in radians.

        Returns:
            Self: The canvas object.
        """
        self._angle += angle
        about = (self.x, self.y)
        self._xform_matrix = rotation_matrix(angle, about) @ self._xform_matrix

        return self

    def _flip(self, axis: str) -> Self:
        """
        Flip the canvas along the specified axis.

        Args:
            axis (str): The axis to flip the canvas along ('x' or 'y').

        Returns:
            Self: The canvas object.
        """
        if axis == "x":
            self._scale[0] *= -1
        elif axis == "y":
            self._scale[1] *= -1

        sx, sy = self._scale
        self._xform_matrix = scale_matrix(sx, sy) @ self._xform_matrix

        return self

    def flip_x_axis(self) -> Self:
        """
        Flip the x-axis direction.

        Returns:
            Self: The canvas object.
        """
        return self._flip("x")

    def flip_y_axis(self) -> Self:
        """
        Flip the y-axis direction.

        Returns:
            Self: The canvas object.
        """
        return self._flip("y")

    def scale(self, sx: float, sy: float = None) -> Self:
        """
        Scale the canvas by sx and sy about the Canvas origin.

        Args:
            sx (float): The scale factor along the x-axis.
            sy (float, optional): The scale factor along the y-axis, defaults to None.

        Returns:
            Self: The canvas object.
        """
        if sy is None:
            sy = sx
        self._scale[0] *= sx
        self._scale[1] *= sy
        self._xform_matrix = scale_matrix(sx, sy) @ self._xform_matrix

        return self

    @property
    def x(self) -> float:
        """
        The x coordinate of the canvas origin.

        Returns:
            float: The x coordinate of the canvas origin.
        """
        return self._pos[0]

    @property
    def y(self) -> float:
        """
        The y coordinate of the canvas origin.

        Returns:
            float: The y coordinate of the canvas origin.
        """
        return self._pos[1]

    @property
    def angle(self) -> float:
        """
        The orientation angle in radians.

        Returns:
            float: The orientation angle in radians.
        """
        return self._angle

    @property
    def scale_factors(self) -> Vec2:
        """
        The scale factors.

        Returns:
            Vec2: The scale factors.
        """
        return self._scale

    def batch_graph(self, batch: "Batch") -> nx.DiGraph:
        """
        Return a directed graph of the batch and its elements.
        Canvas is the root of the graph.
        Graph nodes are the ids of the elements.

        Args:
            batch (Batch): The batch to create the graph from.

        Returns:
            nx.DiGraph: The directed graph of the batch and its elements.
        """

        def add_batch(batch, graph):
            graph.add_node(batch.id)
            for item in batch.elements:
                graph.add_edge(batch.id, item.id)
                if item.subtype == Types.BATCH:
                    add_batch(item, graph)
            return graph

        di_graph = nx.DiGraph()
        di_graph.add_edge(self.id, batch.id)
        for item in batch.elements:
            if item.subtype == Types.BATCH:
                di_graph.add_edge(batch.id, item.id)
                add_batch(item, di_graph)
            else:
                di_graph.add_edge(batch.id, item.id)

        return di_graph

    def _resolve_property(self, item: Drawable, property_name: str) -> Any:
        """
        Handles None values for properties.
        try item.property_name first,
        then try canvas.property_name,
        finally use the default value.

        Args:
            item (Drawable): The item to resolve the property for.
            property_name (str): The name of the property to resolve.

        Returns:
            Any: The resolved property value.
        """
        value = getattr(item, property_name)
        if value is None:
            value = self.__dict__.get(property_name, None)
            if value is None:
                value = defaults.get(property_name, VOID)
            if value == VOID:
                print(f"Property {property_name} is not in defaults.")
                value = None
        return value

    def get_fonts_list(self) -> list[str]:
        """
        Get the list of fonts used in the canvas.

        Returns:
            list[str]: The list of fonts used in the canvas.
        """
        user_fonts = set(self._font_list)

        latex_fonts = set(
            [
                defaults["main_font"],
                defaults["sans_font"],
                defaults["mono_font"],
                "serif",
                "sansserif",
                "monospace",
            ]
        )
        for sketch in self.active_page.sketches:
            if sketch.subtype == Types.TAG_SKETCH:
                name = sketch.font_family
                if name is not None and name not in latex_fonts:
                    user_fonts.add(name)
        return list(user_fonts.difference(latex_fonts))

    def _calculate_size(self, border=None, b_box=None) -> Tuple[float, float]:
        """
        Calculate the size of the canvas based on the bounding box and border.

        Args:
            border (float, optional): The border of the canvas, defaults to None.
            b_box (Any, optional): The bounding box of the canvas, defaults to None.

        Returns:
            Tuple[float, float]: The size of the canvas.
        """
        vertices = self._all_vertices
        if vertices:
            if b_box is None:
                b_box = bounding_box(vertices)

            if border is None:
                if self.border is None:
                    border = defaults["border"]
                else:
                    border = self.border
            w = b_box.width + 2 * border
            h = b_box.height + 2 * border
            offset_x, offset_y = b_box.southwest
            res = w, h, offset_x - border, offset_y - border
        else:
            res = None
        return res

    def _show_browser(
        self, file_path: Path, show_browser: bool, multi_page_svg: bool
    ) -> None:
        """
        Show the file in the browser.

        Args:
            file_path (Path): The path to the file.
            show_browser (bool): Whether to show the file in the browser.
            multi_page_svg (bool): Whether the file is a multi-page SVG.
        """
        if show_browser is None:
            show_browser = defaults["show_browser"]
        if show_browser:
            file_path = 'file:///' + file_path
            if multi_page_svg:
                for i, _ in enumerate(self.pages):
                    f_path = file_path.replace(".svg", f"_page{i + 1}.svg")
                    webbrowser.open(f_path)
            else:
                webbrowser.open(file_path)

    def save(
        self,
        file_path: Path = None,
        overwrite: bool = None,
        show: bool = None,
        print_output=True,
    ) -> Self:
        """
        Save the canvas to a file.

        Args:
            file_path (Path, optional): The path to save the file.
            overwrite (bool, optional): Whether to overwrite the file if it exists.
            show (bool, optional): Whether to show the file in the browser.
            print_output (bool, optional): Whether to print the output of the compilation.

        Returns:
            Self: The canvas object.
        """

        def validate_file_path(file_path: Path, overwrite: bool) -> Result:
            """
            Validate the file path.

            Args:
                file_path (Path): The path to the file.
                overwrite (bool): Whether to overwrite the file if it exists.

            Returns:
                Result: The parent directory, file name, and extension.
            """
            path_exists = os.path.exists(file_path)
            if path_exists and not overwrite:
                raise FileExistsError(
                    f"File {file_path} already exists. \n"
                    "Use canvas.save(file_path, overwrite=True) to overwrite the file."
                )
            parent_dir, file_name = os.path.split(file_path)
            file_name, extension = os.path.splitext(file_name)
            if extension not in [".pdf", ".eps", ".ps", ".svg", ".png", ".tex"]:
                raise RuntimeError("File type is not supported.")
            if not os.path.exists(parent_dir):
                raise NotADirectoryError(f"Directory {parent_dir} does not exist.")
            if not os.access(parent_dir, os.W_OK):
                raise PermissionError(f"Directory {parent_dir} is not writable.")

            return parent_dir, file_name, extension

        def compile_tex(cmd):
            """
            Compile the TeX file.

            Args:
                cmd (str): The command to compile the TeX file.

            Returns:
                str: The output of the compilation.
            """
            os.chdir(parent_dir)
            with subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, text=True
            ) as p:
                output = p.communicate("_s\n_l\n")[0]
            if print_output:
                print(output.split("\n")[-3:])
            return output

        def remove_aux_files(file_path):
            """
            Remove auxiliary files generated during compilation.

            Args:
                file_path (Path): The path to the file.
            """
            time_out = 1  # seconds
            parent_dir, file_name = os.path.split(file_path)
            file_name, extension = os.path.splitext(file_name)
            aux_file = os.path.join(parent_dir, file_name + ".aux")
            if os.path.exists(aux_file):
                if not wait_for_file_availability(aux_file, time_out):
                    print(
                        (
                            f"File '{aux_file}' is not available after waiting for "
                            f"{time_out} seconds."
                        )
                    )
                else:
                    os.remove(aux_file)
            log_file = os.path.join(parent_dir, file_name + ".log")
            if os.path.exists(log_file):
                if not wait_for_file_availability(log_file, time_out):
                    print(
                        (
                            f"File '{log_file}' is not available after waiting for "
                            f"{time_out} seconds."
                        )
                    )
                else:
                    if not defaults["keep_log_files"]:
                        os.remove(log_file)
            tex_file = os.path.join(parent_dir, file_name + ".tex")
            if os.path.exists(tex_file):
                if not wait_for_file_availability(tex_file, time_out):
                    print(
                        (
                            f"File '{tex_file}' is not available after waiting for "
                            f"{time_out} seconds."
                        )
                    )
                else:
                    os.remove(tex_file)
            file_name, extension = os.path.splitext(file_name)
            if extension not in [".pdf", ".tex"]:
                pdf_file = os.path.join(parent_dir, file_name + ".pdf")
                if os.path.exists(pdf_file):
                    if not wait_for_file_availability(pdf_file, time_out):
                        print(
                            (
                                f"File '{pdf_file}' is not available after waiting for "
                                f"{time_out} seconds."
                            )
                        )
                    else:
                        # os.remove(pdf_file)
                        pass
            log_file = os.path.join(parent_dir, "simetri.log")
            if os.path.exists(log_file):
                try:
                    os.remove(log_file)
                except PermissionError:
                    # to do: log the error
                    pass

        def run_job():
            """
            Run the job to compile and save the file.

            Returns:
                None
            """
            output_path = os.path.join(parent_dir, file_name + extension)
            cmd = "xelatex " + tex_path + " --output-directory " + parent_dir
            res = compile_tex(cmd)
            if "No pages of output" in res:
                raise RuntimeError("Failed to compile the tex file.")
            pdf_path = os.path.join(parent_dir, file_name + ".pdf")
            if not os.path.exists(pdf_path):
                raise RuntimeError("Failed to compile the tex file.")

            if extension in [".eps", ".ps"]:
                ps_path = os.path.join(parent_dir, file_name + extension)
                os.chdir(parent_dir)
                cmd = f"pdf2ps {pdf_path} {ps_path}"
                res = subprocess.run(cmd, shell=True, check=False)
                if res.returncode != 0:
                    raise RuntimeError("Failed to convert pdf to ps.")
            elif extension == ".svg":
                doc = fitz.open(pdf_path)
                page = doc.load_page(0)
                svg = page.get_svg_image()
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(svg)
            elif extension == ".png":
                pdf_file = fitz.open(pdf_path)
                page = pdf_file[0]
                pix = page.get_pixmap()
                pix.save(output_path)
                pdf_file.close()

        parent_dir, file_name, extension = validate_file_path(file_path, overwrite)

        tex_code = get_tex_code(self)
        tex_path = os.path.join(parent_dir, file_name + ".tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_code)
        if extension == ".tex":
            return self

        run_job()
        remove_aux_files(file_path)

        self._show_browser(file_path=file_path, show_browser=show, multi_page_svg=False)
        return self

    def new_page(self, **kwargs) -> Self:
        """
        Create a new page and add it to the canvas.pages.

        Args:
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        page = Page()
        self.pages.append(page)
        self.active_page = page
        for k, v in kwargs.items():
            setattr(page, k, v)
        return self


@dataclass
class PageGrid:
    """
    Grid class for drawing grids on a page.

    Args:
        spacing (float, optional): The spacing between grid lines.
        back_color (Color, optional): The background color of the grid.
        line_color (Color, optional): The color of the grid lines.
        line_width (float, optional): The width of the grid lines.
        line_dash_array (Sequence[float], optional): The dash array for the grid lines.
        x_shift (float, optional): The x-axis shift of the grid.
        y_shift (float, optional): The y-axis shift of the grid.
    """

    spacing: float = None
    back_color: Color = None
    line_color: Color = None
    line_width: float = None
    line_dash_array: Sequence[float] = None
    x_shift: float = None
    y_shift: float = None

    def __post_init__(self):
        self.type = Types.PAGE_GRID
        self.subtype = Types.RECTANGULAR
        self.spacing = defaults["page_grid_spacing"]
        self.back_color = defaults["page_grid_back_color"]
        self.line_color = defaults["page_grid_line_color"]
        self.line_width = defaults["page_grid_line_width"]
        self.line_dash_array = defaults["page_grid_line_dash_array"]
        self.x_shift = defaults["page_grid_x_shift"]
        self.y_shift = defaults["page_grid_y_shift"]
        common_properties(self)


@dataclass
class Page:
    """
    Page class for drawing sketches and text on a page. All drawing
    operations result as sketches on the canvas.active_page.

    Args:
        size (Vec2, optional): The size of the page.
        back_color (Color, optional): The background color of the page.
        mask (Any, optional): The mask of the page.
        margins (Any, optional): The margins of the page (left, bottom, right, top).
        recto (bool, optional): Whether the page is recto (True) or verso (False).
        grid (PageGrid, optional): The grid of the page.
        kwargs (dict, optional): Additional keyword arguments.
    """

    size: Vec2 = None
    back_color: Color = None
    mask = None
    margins = None  # left, bottom, right, top
    recto: bool = True  # True if page is recto, False if verso
    grid: PageGrid = None
    kwargs: dict = None

    def __post_init__(self):
        self.type = Types.PAGE
        self.sketches = []
        if self.grid is None:
            self.grid = PageGrid()
        if self.kwargs:
            for k, v in self.kwargs.items():
                setattr(self, k, v)
        common_properties(self)


def hello() -> None:
    """
    Show a hello message.
    Used for testing an installation of simetri.
    """
    canvas = Canvas()

    canvas.text("Hello from simetri.graphics!", (0, -130), bold=True, font_size=20)
    canvas.draw(logo())

    d_path = os.path.dirname(os.path.abspath(__file__))
    f_path = os.path.join(d_path, "hello.pdf")

    canvas.save(f_path, overwrite=True)

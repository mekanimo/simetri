"""Canvas class for drawing shapes and text on a page. All drawing
operations are handled by the Canvas class. Canvas class can draw all
graphics objects and text objects. It also provides methods for
drawing basic shapes like lines, circles, and polygons.
"""

import os
import time
import tempfile
import shutil
import webbrowser
import warnings
from typing import Optional, Any, Tuple, Sequence
from pathlib import Path
from dataclasses import dataclass
from math import pi

from typing_extensions import Self, Union
import numpy as np
import networkx as nx

from simetri.graphics.affine import (
    rotation_matrix,
    translation_matrix,
    scale_matrix,
    scale_in_place_matrix,
    identity_matrix,
)
from simetri.graphics.common import common_properties, _set_Nones, VOID, Point, Vec2
from simetri.graphics.all_enums import (
    Types,
    Drawable,
    Result,
    Anchor,
    TexLoc,
    Align,
    Axis,
    FilterType
)
from simetri.settings.settings import defaults
from simetri.graphics.bbox import bounding_box
from simetri.graphics.batch import Batch
from simetri.graphics.shape import Shape
from simetri.graphics.sketch import TexSketch
from simetri.colors.colors import Color, light_gray
from simetri.canvas import draw
from simetri.helpers.utilities import wait_for_file_availability
from simetri.helpers.illustration import logo
from simetri.helpers.file_operations import validate_filepath
from simetri.tikz.tikz import get_tex_code
from simetri.helpers.validation import validate_args
from simetri.canvas.style_map import canvas_args
from simetri.notebook import display
from simetri.image.image import Image, create_image_from_data
from simetri.tex.tex import remove_aux_files, run_job, Tex


class Canvas:
    """Canvas class for drawing shapes and text on a page. All drawing
    operations are handled by the Canvas class. Canvas class can draw all
    graphics objects and text objects. It also provides methods for
    drawing basic shapes like lines, circles, and polygons.
    """

    def __init__(
        self,
        back_color: Optional["Color"] = None,
        border: Optional[float] = None,
        size: Optional[Vec2] = None,
        **kwargs,
    ):
        """
        Initialize the Canvas.

        Args:
            back_color (Optional[Color]): The background color of the canvas.
            border (Optional[float]): The border width of the canvas.
            size (Vec2, optional): The size of the canvas with canvas.origin at (0, 0).
            kwargs (dict): Additional keyword arguments. Rarely used.

            You should not need to specify "size" since it is calculated.
        """
        validate_args(kwargs, canvas_args)
        _set_Nones(self, ["back_color", "border"], [back_color, border])
        self._size = size
        self.border = border
        """This value is added to the bounding box of the canvas to expand the output size."""
        self.type = Types.CANVAS
        """Used internally to identify the type of object. Do not change it!.
        Most objects in simetri has a type and subtype attribute. See ... for
        using this for user defined objects.
        """
        self.subtype = Types.CANVAS
        """Used internally to identify the type of object. Do not change it!.
        Most objects in simetri has a type and subtype attribute. See ... for
        using this for user defined objects.
        """
        self._code = []
        self._font_list = []
        self.preamble = defaults["preamble"]
        """Used for generating TikZ code."""
        self.back_color = back_color
        """Background color of the canvas."""
        self.pages = [Page(self.size, self.back_color, self.border)]
        """Each page of the canvas is a Page object.
        The canvas can have multiple pages that result in multi-page PDF output,
        or multiple images with image_name_1.svg, image_name_2.svg, etc."""
        self.active_page = self.pages[0]
        """canvas.draw() draws on the active page. If there are multiple pages,
        the active page is the last page created."""
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
        self.inset = 0
        self.page_size = None
        common_properties(self)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._xform_matrix = identity_matrix()
        self._sketch_xform_matrix = identity_matrix()
        self.tex: Tex = Tex()
        self.render = defaults["render"]
        if self._size is not None:
            x, y = self.origin[:2]
            self._limits = [x, y, x + self.size[0], y + self.size[1]]
        else:
            self._limits = None
        self.overlay = False  # used for inserting pdf pictures


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
        elif name == "scale":
            if isinstance(value, (list, tuple)):
                type(self).scale.fset(self, value[0], value[1])
            else:
                type(self).scale.fset(self, value)
        elif name == "pos":
            if isinstance(value, (list, tuple, np.ndarray)):
                type(self).pos.fset(self, value)
            else:
                raise ValueError("pos must be a list, tuple or np.ndarray.")
        elif name == "angle":
            if isinstance(value, (int, float)):
                type(self).angle.fset(self, value)
            else:
                raise ValueError("angle must be a number.")

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

    @property
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
        [min_x, min_y, max_x, max_y]

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
        [min_x, min_y, max_x, max_y]

        Args:
            value (Vec2): The limits of the canvas.
        """
        if len(value) == 4:
            x1, y1, x2, y2 = value
            self._size = (x2 - x1, y2 - y1)
            self._origin = (x1, y1)
        else:
            raise ValueError("Limits must be a tuple of 4 values.")

    def capture(self, **kwargs) -> Image:
        """
        Create an image from the canvas.

        Returns:
            Image: The Image object.
        """
        tmpdirname = tempfile.TemporaryDirectory().name
        os.makedirs(tmpdirname, exist_ok=True)
        file_name = next(tempfile._get_candidate_names())
        filepath = os.path.join(tmpdirname, file_name + ".ps")

        self.save(filepath, show=False, print_output=False, remove_aux=False)
        wait_for_file_availability(filepath, timeout=5)
        temp_img = create_image_from_data(filepath)

        time.sleep(1)
        shutil.rmtree(tmpdirname, ignore_errors=True)
        return temp_img

    def insert_code(self, code, loc: TexLoc = TexLoc.PICTURE) -> Self:
        """
        Insert code into the canvas.

        Args:
            code (str): The code to insert.
            loc (TexLoc): The location to insert the code.

        Returns:
            Self: The canvas object.
        """
        draw.insert_code(self, code, loc)
        return self

    def arc(
        self,
        center: Point,
        radius_x: float,
        radius_y: float = None,
        start_angle: float = 0,
        span_angle: float = pi / 2,
        rot_angle: float = 0,
        **kwargs,
    ) -> Self:
        """
        Draw an arc with the given center, radius, start angle and end angle.

        Args:
            center (Point): The center of the arc.
            radius_x (float): The radius of the arc.
            radius_y (float, optional): The second radius of the arc, defaults to None.
            start_angle (float): The start angle of the arc.
            end_angle (float): The end angle of the arc.
            rot_angle (float, optional): The rotation angle of the arc, defaults to 0.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        if radius_y is None:
            radius_y = radius_x
        draw.arc(
            self,
            center,
            radius_x,
            radius_y,
            start_angle,
            span_angle,
            rot_angle,
            **kwargs,
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

    def circle(self, radius: float, center: Point = (0, 0), **kwargs) -> Self:
        """
        Draw a circle with the given center and radius.

        Args:
            center (Point): The center of the circle.
            radius (float): The radius of the circle.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        draw.circle(self, radius, center, **kwargs)
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

    def draw_fragments(self, lace, palette=None, **kwargs):
        draw.draw_fragments(self, lace, palette, **kwargs)
        return self

    def text(
        self,
        text: str,
        pos: Point,
        font_family: str = None,
        font_size: int = None,
        font_color: Color = None,
        anchor: Anchor = None,
        align: Align = None,
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
            anchor options: BASE, BASE_EAST, BASE_WEST, BOTTOM, CENTER, EAST, NORTH,
            NORTHEAST, NORTHWEST, SOUTH, SOUTHEAST, SOUTHWEST, WEST, MIDEAST, MIDWEST, RIGHT,
            LEFT, TOP
            align (Align, optional): The alignment of the text, defaults to Align.CENTER.
            align options: CENTER, FLUSH_CENTER, FLUSH_LEFT, FLUSH_RIGHT, JUSTIFY, LEFT, RIGHT
            kwargs (dict): Additional keyword arguments.
            common kwargs: fill_color, line_color, line_width, fill, line, alpha, font_color

        Returns:
            Self: The canvas object.
        """
        draw.text(
            self,
            txt=text,
            pos=pos,
            font_family=font_family,
            font_size=font_size,
            font_color=font_color,
            anchor=anchor,
            align=align,
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
            pos (tuple): The lower-left corner of the grid.
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

    def rectangle2(
        self,
        corner1: Point = (0, 0),
        corner2: Point = (0, 0),
        angle: float = 0,
        **kwargs,
    ) -> Self:
        """
        Draw a rectangle.

        Args:
            corner1 (Point): The first corner of the rectangle.
            corner2 (Point): The diagonally opposing corner.
            angle (float, optional): The angle of the rectangle, defaults to 0.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        x1, y1 = corner1
        x2, y2 = corner2
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        draw.rectangle(self, center, width, height, angle, **kwargs)
        return self

    def rectangle3(
        self,
        upper_left: Point,
        width: float = 100,
        height: float = 100,
        angle: float = 0,
        **kwargs,
    ) -> Self:
        """
        Draw a rectangle.

        Args:
            upper_left (Point): The upper_left corner of the rectangle.
            width (float): The width of the rectangle.
            height (float): The height of the rectangle.
            angle (float, optional): The angle of the rectangle, defaults to 0.
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        x1, y1 = upper_left[:2]
        x2, y2 = x1 + width, y1 - height
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

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

    def begin_clip(
        self,
        item: Union[Shape, Batch] = None,
        corners: Sequence[Point] = None,
        margins: Sequence[float] = None,
        fade: Anchor = None,
        fade_margins: Sequence[float] = None,
        fillet_radius: float = None,
    ) -> Self:
        """Insert a TeX code into the canvas.

        Args:
            item (Shape or Batch): The item to clip.
            margins: [left, bottom, right, top] margins to apply to the clipping area.
            fade (Anchor, optional): The direction of the fade effect. Defaults to None.
            fade_margins (Sequence[float], optional): Margins for the fade effect.
            fade_margins are applied after the clipping margins.

        Returns:
            Self: The canvas object.
        """
        d_directions = {
            Anchor.NORTH: "north",
            Anchor.SOUTH: "south",
            Anchor.EAST: "east",
            Anchor.WEST: "west",
            Anchor.SOUTHWEST: "south west",
            Anchor.SOUTHEAST: "south east",
            Anchor.NORTHEAST: "north east",
            Anchor.NORTHWEST: "north west",
        }

        active_sketches = self.active_page.sketches
        if item is not None:
            corners = item.corners
            x1, y1 = corners[1]
            x2, y2 = corners[3]
            left, bottom, right, top = margins
            x1 += left
            y1 += bottom
            x2 -= right
            y2 -= top
        else:
            x1, y1, x2, y2 = corners
        clip_code = ""
        if fillet_radius is not None:
            if fillet_radius < 0:
                raise ValueError("Fillet radius must be a positive number.")
            clip_code = f"[rounded corners={fillet_radius}]"
        code = f"""\\begin{{scope}}
\\clip{clip_code}({x1}, {y1}) rectangle ({x2}, {y2});\n"""
        if fade is not None:
            if fade_margins is not None:
                left, bottom, right, top = fade_margins
                x1 += left
                y1 += bottom
                x2 -= right
                y2 -= top
            code += f"\\path [scope fading={d_directions[fade]}] ({x1}, {y1}) rectangle ({x2}, {y2});\n"

        sketch = TexSketch(code)
        if fade is not None:
            sketch.library = "fadings"
        active_sketches.append(sketch)
        return self

    def begin_style(self, style: str):
        # code = rf'\begin{{scope}}[every path/.append style={{dashed, draw=green}}]'
        code = rf"\begin{{scope}}[every path/.append style={{ {style} }}]"
        code += "\n"
        sketch = TexSketch(code)
        self.active_page.sketches.append(sketch)

        return self

    def end_clip(self):
        return self._end_scope()

    def end_style(self):
        return self._end_scope()

    def _end_scope(self):
        sketch = TexSketch("\\end{scope}\n")
        self.active_page.sketches.append(sketch)

        return self

    def draw(
        self,
        item_s: Union[Shape, Batch, Sequence],
        pos: Point = None,
        angle: float = 0,
        rotocenter: Point = (0, 0),
        scale=(1, 1),
        about=(0, 0),
        show: bool = False,
        filter: FilterType = None,
        **kwargs,
    ) -> Self:
        """
        Draw the item_s. item_s can be a single item or a list of items.

        Args:
            item_s (Union[Batch, Shape, Sequence]): The item(s) to draw.
            pos (Point, optional): The position to draw the item(s), defaults to None.
            angle (float, optional): The angle to rotate the item(s), defaults to 0.
            rotocenter (Point, optional): The point about which to rotate, defaults to (0, 0).
            scale (tuple, optional): The scale factors for the x and y axes, defaults to (1, 1).
            about (tuple, optional): The point about which to scale, defaults to (0, 0).
            show (bool, optional): If True, draws the canvas in a Jupyter cell.
            filter (FilterType, optional): SVG filter type to apply to the drawn item(s).
            kwargs (dict): Additional keyword arguments.

        Returns:
            Self: The canvas object.
        """
        sketch_xform = self._sketch_xform_matrix

        swatch = kwargs.get("swatch", None)
        if swatch:
            n_swatch = len(swatch)

        color = kwargs.get("color", None)
        if color:
            kwargs["line_color"] = color
            kwargs["fill_color"] = color

        if filter is not None:
            kwargs["filter"] = filter

        if pos is not None:
            sketch_xform = translation_matrix(*pos[:2]) @ sketch_xform
        if scale[0] != 1 or scale[1] != 1:
            if pos is None:
                pos = (0, 0)
            sketch_xform = scale_in_place_matrix(*pos[:2], about) @ sketch_xform
        if angle != 0:
            sketch_xform = rotation_matrix(angle, rotocenter) @ sketch_xform
        self._sketch_xform_matrix = sketch_xform @ self._xform_matrix

        if isinstance(item_s, (list, tuple)):
            for i, item in enumerate(item_s):
                if swatch:
                    kwargs["fill_color"] = Color(*swatch[i % n_swatch])
                draw.draw(self, item, **kwargs)
        else:
            draw.draw(self, item_s, **kwargs)

        self._sketch_xform_matrix = identity_matrix()
        if show:
            self.display()
        else:
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

    def draw_pdf(
        self, pdf, pos: Point, size=None, scale=None, angle=0, **kwargs
    ) -> Self:
        """
        Draw a PDF on the canvas.

        Args:
            pdf (PDF): The PDF object to draw or file path.
            pos (Point): Upper-left position to draw the PDF at.

        Returns:
            Self: The canvas object.
        """
        draw.draw_pdf(self, pdf, pos, size, scale, angle, **kwargs)
        return self

    def draw_image(self, image: Image, pos: Point, **kwargs) -> Self:
        """
        Draw an image on the canvas.

        Args:
            image (Image): The image to draw.
            pos (Point): The position to draw the image at.

        Returns:
            Self: The canvas object.
        """
        draw.draw_image(self, image, pos, **kwargs)
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
        self.draw(shadow, line_color=light_gray, line_width=width, style=style)
        self.draw(
            Shape(box2.corners, closed=True), fill=False, line_width=width, style=style
        )

        return self

    def reset(self) -> Self:
        """
        Reset the canvas to its initial state.

        Returns:
            Self: The canvas object.
        """
        self._code = []
        self.preamble = defaults["preamble"]
        self.pages = [Page(self.size, self.back_color, self.border)]
        self.active_page = self.pages[0]
        self._all_vertices = []
        self.tex: Tex = Tex()
        self.clip: bool = False  # if true then clip the canvas to the mask
        self._xform_matrix = identity_matrix()
        self._sketch_xform_matrix = identity_matrix()
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
    def pos(self) -> Point:
        """
        The position of the canvas.

        Args:
            point (Point, optional): The point to set the position to.

        Returns:
            Point: The position of the canvas.
        """

        return self._xform_matrix[2, :2].tolist()[:2]

    @pos.setter
    def pos(self, point: Point) -> None:
        """
        Set the position of the canvas.

        Args:
            point (Point): The point to set the position to.
        """
        self._xform_matrix[2, :2] = point[:2]

    @property
    def angle(self) -> float:
        """
        The angle of the canvas.

        Returns:
            float: The angle of the canvas.
        """
        xform = self._xform_matrix

        return np.arctan2(xform[0, 1], xform[0, 0])

    @angle.setter
    def angle(self, angle: float) -> None:
        """
        Set the angle of the canvas.

        Args:
            angle (float): The angle to set the canvas to.
        """
        self._xform_matrix = rotation_matrix(angle) @ self._xform_matrix

    @property
    def scale(self) -> Vec2:
        """
        The scale of the canvas.

        Returns:
            Vec2: The scale of the canvas.
        """
        xform = self._xform_matrix

        return np.linalg.norm(xform[:2, 0]), np.linalg.norm(xform[:2, 1])

    @scale.setter
    def scale(
        self, scale_x: float = 1, scale_y: float = None, about: Point = (0, 0)
    ) -> None:
        """
        Set the scale of the canvas.

        Args:
            scale_x (float): The x-scale to set the canvas to.
            scale_y (float): The y-scale to set the canvas to.
            about (Point): The point about which to scale the canvas.
        """
        if scale_y is None:
            scale_y = scale_x

        self._xform_matrix = self._xform_matrix @ scale_in_place_matrix(
            scale_x, scale_y, about=about
        )

    @property
    def xform_matrix(self) -> "np.ndarray":
        """
        The transformation matrix of the canvas.

        Returns:
            np.ndarray: The transformation matrix of the canvas.
        """
        return self._xform_matrix.copy()

    def transform(self, transform_matrix: "np.ndarray") -> Self:
        """
        Transforms the canvas by the given transformation matrix.

        Args:
            transform_matrix (np.ndarray): The transformation matrix.

        Returns:
            Self: The Canvas object.
        """
        self._xform_matrix = transform_matrix @ self._xform_matrix

        return self

    def reset_transform(self) -> Self:
        """
        Reset the transformation matrix of the canvas.
        The canvas origin is at (0, 0) and the orientation angle is 0.
        Transformation matrix is the identity matrix.

        Returns:
            Self: The canvas object.
        """
        self._xform_matrix = identity_matrix()

        return self

    def translate(self, dx: float, dy: float) -> Self:
        """
        Translate the canvas by dx and dy.

        Args:
            dx (float): The translation distance along the x-axis.
            dy (float): The translation distance along the y-axis.

        Returns:
            Self: The canvas object.
        """

        self._xform_matrix = translation_matrix(dx, dy) @ self._xform_matrix

        return self

    def rotate(self, angle: float, about=(0, 0)) -> Self:
        """
        Rotate the canvas by angle in radians about the given point.

        Args:
            angle (float): The rotation angle in radians.
            about (tuple): The point about which to rotate the canvas.

        Returns:
            Self: The canvas object.
        """

        self._xform_matrix = rotation_matrix(angle, about) @ self._xform_matrix

        return self

    def _flip(self, axis: Axis) -> Self:
        """
        Flip the canvas along the specified axis.

        Args:
            axis (str): The axis to flip the canvas along ('x' or 'y').

        Returns:
            Self: The canvas object.
        """
        if axis == Axis.X:
            sx = -self.scale[0]
            sy = 1
        elif axis == Axis.Y:
            sx = 1
            sy = -self.scale[1]

        self._xform_matrix = scale_matrix(sx, sy) @ self._xform_matrix

        return self

    def flip_x_axis(self) -> Self:
        """
        Flip the x-axis direction. Warning: This will reverse the positive rotation direction.

        Returns:
            Self: The canvas object.
        """
        warnings.warn(
            "Flipping the x-axis will change the positive rotation direction."
        )
        return self._flip(Axis.X)

    def flip_y_axis(self) -> Self:
        """
        Flip the y-axis direction.

        Returns:
            Self: The canvas object.
        """
        warnings.warn(
            "Flipping the y-axis will reverse the positive rotation direction."
        )

        return self._flip(Axis.Y)

    @property
    def x(self) -> float:
        """
        The x coordinate of the canvas origin.

        Returns:
            float: The x coordinate of the canvas origin.
        """
        return self.pos[0]

    @x.setter
    def x(self, value: float) -> None:
        """
        Set the x coordinate of the canvas origin.

        Args:
            value (float): The x coordinate to set.
        """
        self.pos = [value, self.pos[1]]

    @property
    def y(self) -> float:
        """
        The y coordinate of the canvas origin.

        Returns:
            float: The y coordinate of the canvas origin.
        """
        return self.pos[1]

    @y.setter
    def y(self, value: float) -> None:
        """
        Set the y coordinate of the canvas origin.

        Args:
            value (float): The y coordinate to set.
        """
        self.pos = [self.pos[0], value]

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

    def resolve_property(self, item: Drawable, property_name: str) -> Any:
        """
        Handles None values for properties.
        try item.property_name first,
        then try style-object,
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

    def draw_all_segments(
        self, item: Union[Shape, Batch], vert_indices=False, **kwargs
    ) -> Self:
        """
        Using intersections, splits edges of the item into separate segments and
        draws them with their indices. This is usually used for the "get_loop"
        function.

        Args:
            item: A shape or a batch.
            vert_indices: If True, vertex indices are shown.
                          Default is False, edge indices are shown.
        Returns:
            The canvas object.
        """

        return draw.draw_all_segments(self, item, vert_indices, **kwargs)

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

    def clip(
        self,
        x_min: float = None,
        x_max: float = None,
        y_min: float = None,
        y_max: float = None,
    ) -> Self:
        """
        Clip the canvas to the specified bounding box.

        Args:
            x_min (float): The minimum x coordinate of the bounding box.
            x_max (float): The maximum x coordinate of the bounding box.
            y_min (float): The minimum y coordinate of the bounding box.
            y_max (float): The maximum y coordinate of the bounding box.

        Returns:
            Self: The canvas object.
        """

    def set_page_size(self, width, height):
        self.page_size = (width, height)

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
        self, filepath: Path, show_browser: bool, multi_page_svg: bool
    ) -> None:
        """
        Show the file in the browser.

        Args:
            filepath (Path): The path to the file.
            show_browser (bool): Whether to show the file in the browser.
            multi_page_svg (bool): Whether the file is a multi-page SVG.
        """
        if show_browser is None:
            show_browser = defaults["show_browser"]
        if show_browser:
            filepath = "file:///" + filepath
            if multi_page_svg:
                for i, _ in enumerate(self.pages):
                    f_path = filepath.replace(".svg", f"_page{i + 1}.svg")
                    webbrowser.open(f_path)
            else:
                webbrowser.open(filepath)

    def save(
        self,
        filepath: Path,
        overwrite: bool = None,
        show: bool = None,
        print_output=False,
        remove_aux=True,
        inset=None,
        display=False,
    ) -> Self:
        """
        Save the canvas to a file.

        Args:
            filepath (Path, optional): The path to save the file.
            overwrite (bool, optional): Whether to overwrite the file if it exists.
            show (bool, optional): Whether to show the file in the browser.
            inset (float, optional): The inset value will be clipped from all sides, defaults to None.
        Returns:
            Self: The canvas object.
        """

        if inset is not None:
            self.inset = inset
        parent_dir, file_name, extension = validate_filepath(filepath, overwrite)
        if extension == ".svg":
            from simetri.svg.svg import get_svg_code
            svg_code = get_svg_code(self)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(svg_code)
        else:
            tex_code = get_tex_code(self)
            tex_path = os.path.join(parent_dir, file_name + ".tex")
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(tex_code)
            if extension == ".tex":
                return self

            run_job(parent_dir, file_name, extension, tex_path)
            if remove_aux:
                remove_aux_files(filepath)

        self._show_browser(filepath=filepath, show_browser=show, multi_page_svg=False)
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
    back_color: "Color" = None
    line_color: "Color" = None
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
    back_color: "Color" = None
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

"""Settings and default values for the Simetri library.
Do not modify these values here.
If you are going to share your code with others, you should set these values in your code.
"""

__all__ = ["defaults", "set_defaults", "tikz_defaults", "set_tikz_defaults"]

import sys
from collections import defaultdict
from dataclasses import dataclass

from math import pi

import numpy as np

from ..graphics.all_enums import FontFamily

VOID = 'VOID'

# This is the alpha testing stage for the Simetri library.
# These default values may change in the future.

@dataclass
class Default:
    """A class to represent a default value.

    Attributes:
        name (str): The name of the default value.
        value (any): The default value.
        type (type): The type of the default value.
        help (str): A description of the default value.
        user_value (any): The user-defined value for the default value.

    """
    name: str
    simetri_value: any
    type: type
    help: str
    user_value: any = None

    @property
    def value(self):
        '''
        Returns the simetri_value if user_value is not set.
        '''
        res = self.simetri_value
        if self.user_value is not None:
            res = self.user_value

        return res


class _Defaults:
    """A singleton class that behaves like a dictionary.

    It is used to store default values for the Simetri library.
    It should not be modified directly.
    """

    _instance = None

    def __init__(self):
        """Initializes the _Defaults singleton instance."""
        if _Defaults._instance is not None:
            raise Exception("This class is a singleton!")
        self.defaults = {}
        self.log = set()

    def __getitem__(self, key):
        """Gets the value associated with the key.

        Args:
            key: The key to look up.

        Returns:
            The value associated with the key.
        """
        value = self.defaults[key]
        str_value = str(value)
        self.log.add((key, str_value))
        return value

    def __setitem__(self, key, value):
        """Sets the value for the given key.

        Args:
            key: The key to set.
            value: The value to associate with the key.
        """
        self.defaults[key] = value

    def get(self, key, default=None):
        """Gets the value of a key. If the key does not exist, return the default value.

        Args:
            key: The key to look up.
            default: The default value to return if the key does not exist.

        Returns:
            The value associated with the key, or the default value.
        """
        if key in self.defaults:
            res = self.defaults[key]
        else:
            res = default

        return res

    def keys(self):
        """Returns the keys of the dictionary.

        Returns:
            A view object that displays a list of all the keys.
        """
        return self.defaults.keys()

    def items(self):
        """Returns the items of the dictionary.

        Returns:
            A view object that displays a list of dictionary's key-value tuple pairs.
        """
        return self.defaults.items()

    def values(self):
        """Returns the values of the dictionary.

        Returns:
            A view object that displays a list of all the values.
        """
        return self.defaults.values()


defaults = _Defaults()
default_types = {}
defaults_help = {}

def set_defaults():
    """Sets the default values for the Simetri library."""
    from ..graphics.all_enums import (
        Anchor,
        BackStyle,
        BlendMode,
        DocumentClass,
        FillMode,
        FrameShape,
        LineCap,
        LineJoin,
        MarkerType,
        PageMargins,
        PageNumberPosition,
        PageNumbering,
        PageSize,
        Compiler,
        PageOrientation,
        PatternType,
        ShadeType,
        Align,
    )
    from ..canvas.style_map import (
        ShapeStyle,
        TagStyle,
        LineStyle,
        FillStyle,
        FrameStyle,
        MarkerStyle,
    )

    from ..colors.palettes import seq_MATTER_256
    from ..colors import colors

    global defaults
    global default_types
    global defaults_help

    # tol, rtol, and rtol are used for comparing floats
    # These are used in numpy.isclose and numpy.allclose
    # If you are not careful you may get unexpected results
    # They do not mean that the difference in the compared numbers are within these values
    # numpy isclose returns np.absolute(a - b) <= (atol + rtol * np.absolute(b))
    # if you set atol=.1 and rtol=.1, it means that the difference between a and b
    # is within .1 and .1 * b
    # np.isclose(721, 800, rtol=.1) returns True
    # np.isclose(800, 721, rtol=.1) returns False
    # atol makes a bigger difference when comparing values close to zero
    # if this surrprises you, please read the numpy documentation
    defaults["BB_EPSILON"] = 0.01
    default_types["BB_EPSILON"] = float
    defaults_help["BB_EPSILON"] = (
        "Bounding box epsilon. "
        "Positive float. Length in <points>. "
        "This is a small value used for line/point bounding boxes."
    )

    defaults["INF"] = np.inf
    default_types["INF"] = float
    defaults_help["INF"] = (
        "Infinity. Positive integer. "
        "Used for representing very large numbers. "
        "Maybe usefull for zero division or comparisons."
    )

    defaults["PRINTTEXOUTPUT"] = True  # Print output from the TeX compiler
    default_types["PRINTTEXOUTPUT"] = bool

    defaults["active"] = True  # active objects are drawn
    default_types["active"] = bool
    defaults_help["active"] = (
        "Boolean property for drawable objects. "
        "Currently only used for drawing objects. "
        "If False, the object is not drawn."
        "In the future it may be used with modifiers and transformations."
        "All drawable objects have this property set to True by default."
        "Example: shape.active = False"
    )

    defaults["all_caps"] = False  # use all caps for text
    default_types["all_caps"] = bool
    defaults_help["all_caps"] = (
        "Boolean property for text objects. "
        "If True, the text is displayed in all caps."
    )

    defaults["allow_consec_dup_points"] = False  # use all caps for text
    default_types["allow_consec_dup_points"] = bool
    defaults_help["allow_consec_dup_points"] = (
        "Boolean property for allowing consecutive duplicate points in Shape objects. "
        "Do not change this unless you absolutely have to. Likely to cause problems."
    )

    defaults["alpha"] = 1.0  # used for transparency
    default_types["alpha"] = float
    defaults_help["alpha"] = (
        "Alpha value for transparency. "
        "Float between 0 and 1. "
        "0 is fully transparent, 1 is fully opaque."
        "Both line opacity and fill opacity are set with this value."
    )

    defaults["anchor"] = Anchor.CENTER  # used for text alignment
    default_types["anchor"] = Anchor
    defaults_help["anchor"] = (
        "Specifies text object location. "
        "Anchor.CENTER, Anchor.NORTH, Anchor.SOUTH, "
        "Anchor.EAST, Anchor.WEST, Anchor.NORTHEAST, "
        "Anchor.NORTHWEST, Anchor.SOUTHEAST, Anchor.SOUTHWEST"
        "Example: text.anchor = Anchor.NORTH"
    )

    defaults["angle_atol"] = 0.001  # used for comparing angles
    default_types["angle_atol"] = float
    defaults_help["angle_atol"] = (
        "Angle absolute tolerance. "
        "Positive float. Angle in radians. "
        "Used for comparing angles."
    )

    defaults["angle_rtol"] = 0.001  # used for comparing angles
    default_types["angle_rtol"] = float
    defaults_help["angle_rtol"] = (
        "Angle relative tolerance. "
        "Positive float. Angle in radians. "
        "Used for comparing angles."
    )

    defaults["angle_tol"] = (
        0.001  # used for comparing angles in radians .001 rad = .057 degrees
    )
    default_types["angle_tol"] = float
    defaults_help["angle_tol"] = (
        "Angle tolerance. "
        "Positive float. Angle in radians. "
        "Used for comparing angles."
    )

    defaults["area_atol"] = 0.001  # used for comparing areas
    default_types["area_atol"] = float
    defaults_help["area_atol"] = (
        "Area absolute tolerance. "
        "Positive float. Length in <points>. "
        "Used for comparing areas."
    )

    defaults["area_rtol"] = 0.001  # used for comparing areas
    default_types["area_rtol"] = float
    defaults_help["area_rtol"] = (
        "Area relative tolerance. "
        "Positive float. Length in <points>. "
        "Used for comparing areas."
    )

    defaults["area_threshold"] = 1  # used for grouping fragments in a lace object
    default_types["area_threshold"] = float
    defaults_help["area_threshold"] = (
        "Area threshold. "
        "Positive float. Length in <points>. "
        "Used for grouping fragments in a lace object."
    )

    defaults["arrow_head_length"] = 8
    default_types["arrow_head_length"] = float
    defaults_help["arrow_head_length"] = (
        "Arrow head length. "
        "Positive float. Length in <points>. "
        "Length of the arrow head."
    )

    defaults["arrow_head_width"] = 3
    default_types["arrow_head_width"] = float
    defaults_help["arrow_head_width"] = (
        "Arrow head width. "
        "Positive float. Length in <points>. "
        "Width of the arrow head."
    )

    defaults["atol"] = 0.05  # used for comparing floats
    default_types["atol"] = float
    defaults_help["atol"] = (
        "Absolute tolerance. "
        "Positive float. Length in <points>. "
        "1in = 72pt."
        "Used for comparing floats."
    )

    defaults["back_color"] = colors.white  # canvas background color
    default_types["back_color"] = colors.Color
    defaults_help["back_color"] = (
        "Background color. Color object. Background color for the canvas."
    )

    defaults["back_style"] = BackStyle.COLOR  # EMPTY, COLOR, SHADING, PATTERN, GRIDLINES
    default_types["back_style"] = BackStyle
    defaults_help["back_style"] = (
        "Background style for the Canvas. "
        "BackStyle.EMPTY, BackStyle.COLOR, BackStyle.SHADING, "
        "BackStyle.PATTERN, BackStyle.GRIDLINES."
    )

    defaults["begin_doc"] = "\\begin{document}\n"
    default_types["begin_doc"] = str
    defaults_help["begin_doc"] = "Used with the generated .tex file."

    defaults["begin_tikz"] = "\\begin{tikzpicture}[x=1pt, y=1pt, scale=1]\n"
    default_types["begin_tikz"] = str
    defaults_help["begin_tikz"] = "Used with the generated .tex file."

    defaults["blend_mode"] = BlendMode.NORMAL
    default_types["blend_mode"] = BlendMode
    defaults_help["blend_mode"] = (
        "Blend mode. This can be set with the Canvas or Batch objects. "
        "BlendMode.NORMAL, BlendMode.MULTIPLY, BlendMode.SCREEN, "
        "BlendMode.OVERLAY, BlendMode.DARKEN, BlendMode.LIGHTEN, "
        "BlendMode.COLOR_DODGE, BlendMode.COLOR_BURN, "
        "BlendMode.HARD_LIGHT, BlendMode.SOFT_LIGHT, BlendMode.DIFFERENCE, "
        "BlendMode.EXCLUSION, BlendMode.HUE, BlendMode.SATURATION, "
        "BlendMode.COLOR, BlendMode.LUMINOSITY."
    )

    defaults["bold"] = False  # use bold font if True
    default_types["bold"] = bool
    defaults_help["bold"] = (
        "Boolean property for text objects. If True, the text is displayed in bold."
    )

    defaults["border"] = 25  # border around canvas
    default_types["border"] = float
    defaults_help["border"] = (
        "Border around the canvas. "
        "Positive float. Length in <points>. "
        "Border size for the canvas."
    )

    defaults["border_size"] = 4  # border size for the canvas
    default_types["border_size"] = float
    defaults_help["border_size"] = (
        "Border size for the canvas. "
        "Positive float. Length in <points>. "
        "Border size for the canvas."
    )

    defaults["canvas_back_style"] = BackStyle.EMPTY
    default_types["canvas_back_style"] = BackStyle
    defaults_help["canvas_back_style"] = (
        "Canvas background style. "
        "BackStyle.EMPTY, BackStyle.COLOR, BackStyle.SHADING, "
        "BackStyle.PATTERN, BackStyle.GRIDLINES."
    )

    defaults["canvas_frame_color"] = colors.black # frame color for the canvas
    default_types["canvas_frame_color"] = colors.Color
    defaults_help["canvas_frame_color"] = (
        "Frame color for the canvas. Color object."
    )

    defaults["canvas_frame_margin"] = 15  # margin around the canvas frame
    default_types["canvas_frame_margin"] = float
    defaults_help["canvas_frame_margin"] = (
        "Margin around the canvas frame. "
    )

    defaults["canvas_frame_shadow_width"] = 5  # shadow width for the canvas frame
    default_types["canvas_frame_shadow_width"] = float
    defaults_help["canvas_frame_shadow_width"] = (
        "Shadow width for the canvas frame. "
    )

    defaults["canvas_frame_width"] = 45  # frame width for the canvas
    default_types["canvas_frame_width"] = float
    defaults_help["canvas_frame_width"] = (
        "Frame width for the canvas. Positive float. Length in <points>."
    )

    defaults["canvas_size"] = None  # (width, height) canvas size in points
    default_types["canvas_size"] = tuple
    defaults_help["canvas_size"] = (
        "Canvas size. "
        "Tuple of two positive floats. Length in <points>. "
        "Canvas size in points."
    )

    defaults["circle_radius"] = 20
    default_types["circle_radius"] = float
    defaults_help["circle_radius"] = (
        "Circle radius. Positive float. Length in <points>. Radius of the circle."
    )

    defaults["clip"] = False  # clip the outside of the clip_path to the canvas
    default_types["clip"] = bool
    defaults_help["clip"] = (
        "Boolean property for the canvas and Batch objects. "
        "If true, clip the outside of the canvas.mask to the canvas"
        "or Batch elemetns."
    )

    defaults["color"] = colors.black
    default_types["color"] = colors.Color
    defaults_help["color"] = "Color. Color object."

    defaults["shade_color_wheel"] = False
    default_types["shade_color_wheel"] = bool
    defaults_help["shade_color_wheel"] = (
        "Boolean property for the shape objects. "
        "If True, use the color wheel for shading."
    )

    defaults["shade_color_wheel_black"] = False
    default_types["shade_color_wheel_black"] = bool
    defaults_help["shade_color_wheel_black"] = (
        "Boolean property for the shape object. "
        "If True, use the color wheel for shading."
    )

    defaults["shade_color_wheel_white"] = False
    default_types["shade_color_wheel_white"] = bool
    defaults_help["shade_color_wheel_white"] = (
        "Boolean property for the shape object. "
        "If True, use the color wheel for shading."
    )

    defaults["CS_origin_size"] = (
        2  # size of the circle at the origin of the coordinate system
    )
    default_types["CS_origin_size"] = float
    defaults_help["CS_origin_size"] = (
        "Size of the circle at the origin of the coordinate system. "
        "Positive float. Length in <points>."
    )

    defaults["CS_origin_color"] = colors.gray
    default_types["CS_origin_color"] = colors.Color
    defaults_help["CS_origin_color"] = (
        "Color of the circle at the origin of the coordinate "
        "system. "
        "Color object."
    )

    defaults["CS_size"] = (
        25  # size of the coordinate system axes. Used with canvas.draw_CS
    )
    default_types["CS_size"] = float
    defaults_help["CS_size"] = (
        "Size of the coordinate system axes. Positive float. Length in <points>."
    )

    defaults["CS_line_width"] = 2
    default_types["CS_line_width"] = float
    defaults_help["CS_line_width"] = (
        "Line width of the coordinate system axes. "
        "Positive float. Length in <points>."
    )

    defaults["CS_x_color"] = colors.red
    default_types["CS_x_color"] = colors.Color
    defaults_help["CS_x_color"] = (
        "Color of the x-axis in the coordinate system. Color object."
    )

    defaults["CS_y_color"] = colors.green
    default_types["CS_y_color"] = colors.Color
    defaults_help["CS_y_color"] = (
        "Color of the y-axis in the coordinate system. Color object."
    )

    defaults["debug_mode"] = False
    default_types["debug_mode"] = bool
    defaults_help["debug_mode"] = (
        "Boolean property for enabling debug mode. "
        "If True, debug information is printed."
    )

    defaults["dist_tol"] = (
        0.05  # used for comparing two points to check if they are the
    )
    default_types["dist_tol"] = float
    defaults_help["dist_tol"] = (
        "Distance tolerance for comparing two points. "
        "Positive float. Length in <points>."
    )

    defaults["document_class"] = DocumentClass.STANDALONE  # STANDALONE, ARTICLE, BOOK,
    # REPORT, LETTER, SLIDES, BEAMER,
    # MINIMAL
    default_types["document_class"] = DocumentClass
    defaults_help["document_class"] = (
        "Document class for the LaTeX document. DocumentClass enum."
    )

    defaults["document_options"] = ["12pt", "border=25pt"]
    default_types["document_options"] = list
    defaults_help["document_options"] = (
        "Options for the LaTeX document class. List of strings."
    )

    defaults["dot_color"] = colors.black  # for Dot objects
    default_types["dot_color"] = colors.Color
    defaults_help["dot_color"] = "Color for Dot objects. Color object."

    defaults["double_lines"] = False
    default_types["double_lines"] = bool
    defaults_help["double_lines"] = (
        "Boolean property for using double lines. If True, double lines are used."
    )

    defaults["double_distance"] = 2
    default_types["double_distance"] = float
    defaults_help["double_distance"] = (
        "Distance between double lines. Positive float. Length in <points>."
    )

    defaults["draw_fillets"] = False  # draw rounded corners for shapes
    default_types["draw_fillets"] = bool
    defaults_help["draw_fillets"] = (
        "Boolean property for drawing rounded corners for shapes. "
        "If True, rounded corners are drawn."
    )

    defaults["draw_frame"] = False  # draw a frame around the Tag objects
    default_types["draw_frame"] = bool
    defaults_help["draw_frame"] = (
        "Boolean property for drawing a frame around Tag objects. "
        "If True, a frame is drawn."
    )

    defaults["draw_markers"] = False  # draw markers at each vertex of a Shape object
    default_types["draw_markers"] = bool
    defaults_help["draw_markers"] = (
        "Boolean property for drawing markers at each vertex "
        "of a Shape object. "
        "If True, markers are drawn."
    )

    defaults["ellipse_width_height"] = (40, 20)  # width and height of the ellipse
    default_types["ellipse_width_height"] = tuple
    defaults_help["ellipse_width_height"] = (
        "Width and height of the ellipse. "
        "Tuple of two positive floats. Length in <points>."
    )

    defaults["end_doc"] = "\\end{document}\n"
    default_types["end_doc"] = str
    defaults_help["end_doc"] = "End document string for the generated .tex file."

    defaults["end_tikz"] = "\\end{tikzpicture}\n"
    default_types["end_tikz"] = str
    defaults_help["end_tikz"] = "End TikZ picture string for the generated .tex file."

    defaults["even_odd"] = True  # use even-odd rule for filling shapes
    default_types["even_odd"] = bool
    defaults_help["even_odd"] = (
        "Boolean property for using the even-odd rule for filling shapes. "
        "If True, the even-odd rule is used."
    )

    defaults["ext_length2"] = 25  # dimension extra extension length
    default_types["ext_length2"] = float
    defaults_help["ext_length2"] = (
        "Dimension extra extension length. Positive float. Length in <points>."
    )

    defaults["fill"] = True
    default_types["fill"] = bool
    defaults_help["fill"] = (
        "Boolean property for filling shapes. If True, shapes are filled."
    )

    defaults["fill_alpha"] = 1
    default_types["fill_alpha"] = float
    defaults_help["fill_alpha"] = (
        "Alpha value for fill transparency. Float between 0 and 1."
    )

    defaults["fill_color"] = colors.black
    default_types["fill_color"] = colors.Color
    defaults_help["fill_color"] = "Fill color for shapes. Color object."

    defaults["fill_mode"] = FillMode.EVENODD
    default_types["fill_mode"] = FillMode
    defaults_help["fill_mode"] = "Fill mode for shapes. FillMode enum."

    defaults["fill_blend_mode"] = BlendMode.NORMAL
    default_types["fill_blend_mode"] = BlendMode
    defaults_help["fill_blend_mode"] = "Blend mode for fill. BlendMode enum."

    defaults["fillet_radius"] = None
    default_types["fillet_radius"] = float
    defaults_help["fillet_radius"] = (
        "Radius for rounded corners (fillets). Positive float. Length in <points>."
    )

    defaults["font_blend_mode"] = BlendMode.NORMAL
    default_types["font_blend_mode"] = BlendMode
    defaults_help["font_blend_mode"] = "Blend mode for font. BlendMode enum."

    defaults["font_alpha"] = 1
    default_types["font_alpha"] = float
    defaults_help["font_alpha"] = (
        "Alpha value for font transparency. Float between 0 and 1."
    )

    defaults["font_color"] = colors.black  # use the default font color in LaTeX engine
    default_types["font_color"] = colors.Color
    defaults_help["font_color"] = (
        "Font color. Color object. Font color for the text objects."
    )

    defaults["font_family"] = (
        FontFamily.SERIF  # use the default font family in LaTeX engine
    )
    default_types["font_family"] = str
    defaults_help["font_family"] = (
        "Font family. String. Font family for the text objects."
    )

    defaults["font_size"] = 12
    default_types["font_size"] = float
    defaults_help["font_size"] = (
        "Font size. "
        "Positive float. Length in <points>. "
        "Font size for the text objects."
    )

    defaults["font_style"] = ""
    default_types["font_style"] = str
    defaults_help["font_style"] = "Font style. String. Font style for the text objects."

    defaults["frame_active"] = True
    default_types["frame_active"] = bool
    defaults_help["frame_active"] = (
        "Boolean property for active frames. If True, frames are drawn."
    )

    defaults["frame_alpha"] = 1
    default_types["frame_alpha"] = float
    defaults_help["frame_alpha"] = (
        "Alpha value for frame transparency. Float between 0 and 1."
    )

    defaults["frame_back_alpha"] = 1
    default_types["frame_back_alpha"] = float
    defaults_help["frame_back_alpha"] = (
        "Alpha value for frame background transparency. Float between 0 and 1."
    )

    defaults["frame_back_color"] = colors.white
    default_types["frame_back_color"] = colors.Color
    defaults_help["frame_back_color"] = "Frame background color. Color object."

    defaults["frame_blend_mode"] = BlendMode.NORMAL
    default_types["frame_blend_mode"] = BlendMode
    defaults_help["frame_blend_mode"] = "Blend mode for frame. BlendMode enum."

    defaults["frame_color"] = colors.black
    default_types["frame_color"] = colors.Color
    defaults_help["frame_color"] = "Frame color. Color object."

    defaults["frame_draw_fillets"] = False
    default_types["frame_draw_fillets"] = bool
    defaults_help["frame_draw_fillets"] = (
        "Boolean property for drawing fillets for frames. "
        "If True, fillets are drawn."
    )

    defaults["frame_fill"] = True
    default_types["frame_fill"] = bool
    defaults_help["frame_fill"] = (
        "Boolean property for filling frames. If True, frames are filled."
    )

    defaults["frame_fillet_radius"] = 3
    default_types["frame_fillet_radius"] = float
    defaults_help["frame_fillet_radius"] = (
        "Fillet radius for frames. Positive float. Length in <points>."
    )

    defaults["frame_gradient"] = None
    default_types["frame_gradient"] = object  # Assuming gradient is a custom object
    defaults_help["frame_gradient"] = "Frame gradient. Gradient object."

    defaults["frame_inner_sep"] = 3
    default_types["frame_inner_sep"] = float
    defaults_help["frame_inner_sep"] = (
        "Frame inner separation. Positive float. Length in <points>."
    )

    defaults["frame_inner_xsep"] = None
    default_types["frame_inner_xsep"] = float
    defaults_help["frame_inner_xsep"] = (
        "Frame inner x separation. Positive float. Length in <points>."
    )

    defaults["frame_inner_ysep"] = None
    default_types["frame_inner_ysep"] = float
    defaults_help["frame_inner_ysep"] = (
        "Frame inner y separation. Positive float. Length in <points>."
    )

    defaults["frame_outer_sep"] = 0
    default_types["frame_outer_sep"] = float
    defaults_help["frame_outer_sep"] = (
        "Frame outer separation. Positive float. Length in <points>."
    )

    defaults["frame_line_cap"] = LineCap.BUTT
    default_types["frame_line_cap"] = LineCap
    defaults_help["frame_line_cap"] = "Line cap for frames. LineCap enum."

    defaults["frame_line_dash_array"] = []
    default_types["frame_line_dash_array"] = list
    defaults_help["frame_line_dash_array"] = (
        "Line dash array for frames. List of floats."
    )

    defaults["frame_line_join"] = LineJoin.MITER
    default_types["frame_line_join"] = LineJoin
    defaults_help["frame_line_join"] = "Line join for frames. LineJoin enum."

    defaults["frame_line_width"] = 1
    default_types["frame_line_width"] = float
    defaults_help["frame_line_width"] = (
        "Line width for frames. Positive float. Length in <points>."
    )

    defaults["frame_min_height"] = 50
    default_types["frame_min_height"] = float
    defaults_help["frame_min_height"] = (
        "Minimum height for frames. Positive float. Length in <points>."
    )

    defaults["frame_min_width"] = 50
    default_types["frame_min_width"] = float
    defaults_help["frame_min_width"] = (
        "Minimum width for frames. Positive float. Length in <points>."
    )

    defaults["frame_min_size"] = 50
    default_types["frame_min_size"] = float
    defaults_help["frame_min_size"] = (
        "Minimum size for frames. Positive float. Length in <points>."
    )

    defaults["frame_pattern"] = None
    default_types["frame_pattern"] = object  # Assuming pattern is a custom object
    defaults_help["frame_pattern"] = "Frame pattern. Pattern object."

    defaults["frame_rounded_corners"] = False
    default_types["frame_rounded_corners"] = bool
    defaults_help["frame_rounded_corners"] = (
        "Boolean property for rounded corners for frames. "
        "If True, rounded corners are drawn."
    )

    defaults["frame_shape"] = FrameShape.RECTANGLE
    default_types["frame_shape"] = FrameShape
    defaults_help["frame_shape"] = "Frame shape. FrameShape enum."

    defaults["frame_smooth"] = True
    default_types["frame_smooth"] = bool
    defaults_help["frame_smooth"] = (
        "Boolean property for smooth frames. If True, frames are smooth."
    )

    defaults["frame_stroke"] = True
    default_types["frame_stroke"] = bool
    defaults_help["frame_stroke"] = (
        "Boolean property for stroke frames. If True, frames are stroked."
    )

    defaults["frame_visible"] = True
    default_types["frame_visible"] = bool
    defaults_help["frame_visible"] = (
        "Boolean property for visible frames. If True, frames are visible."
    )

    defaults["gap"] = 5  # dimension extension gap
    default_types["gap"] = float
    defaults_help["gap"] = (
        "Dimension extension gap. Positive float. Length in <points>."
    )

    defaults["graph_palette"] = seq_MATTER_256  # this needs to be a 256 color palette
    default_types["graph_palette"] = list
    defaults_help["graph_palette"] = "Graph palette. List of colors."

    defaults["grid_back_color"] = colors.white
    default_types["grid_back_color"] = colors.Color
    defaults_help["grid_back_color"] = "Grid background color. Color object."

    defaults["grid_line_color"] = colors.gray
    default_types["grid_line_color"] = colors.Color
    defaults_help["grid_line_color"] = "Grid line color. Color object."

    defaults["grid_line_width"] = 0.5
    default_types["grid_line_width"] = float
    defaults_help["grid_line_width"] = (
        "Grid line width. Positive float. Length in <points>."
    )

    defaults["grid_alpha"] = 0.5
    default_types["grid_alpha"] = float
    defaults_help["grid_alpha"] = "Grid alpha value. Float between 0 and 1."

    defaults["grid_line_dash_array"] = [2, 2]
    default_types["grid_line_dash_array"] = list
    defaults_help["grid_line_dash_array"] = "Grid line dash array. List of floats."

    defaults["indices_font_family"] = "ttfamily"  # ttfamily, rmfamily, sffamily
    default_types["indices_font_family"] = str
    defaults_help["indices_font_family"] = "Indices font family. String."

    defaults["indices_font_size"] = "tiny"  # tiny, scriptsize, footnotesize, small,
    # normalsize, large, Large, LARGE, huge, Huge
    default_types["indices_font_size"] = str
    defaults_help["indices_font_size"] = "Indices font size. String."

    defaults["image_align"] = Align.CENTER
    default_types["image_align"] = Align
    defaults_help["image_align"] = "image alignment. Align enum."

    defaults["image_alpha"] = 1
    default_types["image_alpha"] = float
    defaults_help["image_alpha"] = (
        "Alpha value for image transparency. Float between 0 and 1."
    )

    defaults["image_blend_mode"] = BlendMode.NORMAL
    default_types["image_blend_mode"] = BlendMode
    defaults_help["image_blend_mode"] = "Blend mode for image. BlendMode enum."

    defaults["italic"] = False
    default_types["italic"] = bool
    defaults_help["italic"] = (
        "Boolean property for italic font. If True, the font is displayed in italic."
    )

    defaults["job_dir"] = None
    default_types["job_dir"] = str
    defaults_help["job_dir"] = "Job directory. String. Directory for the job files."

    defaults["keep_aux_files"] = False
    default_types["keep_aux_files"] = bool
    defaults_help["keep_aux_files"] = (
        "Boolean property for keeping auxiliary files. "
        "If True, auxiliary files are kept."
    )

    defaults["keep_tex_files"] = False
    default_types["keep_tex_files"] = bool
    defaults_help["keep_tex_files"] = (
        "Boolean property for keeping TeX files. If True, TeX files are kept."
    )

    defaults["keep_log_files"] = False
    default_types["keep_log_files"] = bool
    defaults_help["keep_log_files"] = (
        "Boolean property for keeping log files. If True, log files are kept."
    )

    defaults["lace_offset"] = 4
    default_types["lace_offset"] = float
    defaults_help["lace_offset"] = "Lace offset. Positive float. Length in <points>."

    defaults["latex_compiler"] = Compiler.XELATEX  # PDFLATEX, XELATEX, LUALATEX
    default_types["latex_compiler"] = Compiler
    defaults_help["latex_compiler"] = "LaTeX compiler. Compiler enum."

    defaults["line_alpha"] = 1
    default_types["line_alpha"] = float
    defaults_help["line_alpha"] = (
        "Alpha value for line transparency. Float between 0 and 1."
    )

    defaults["line_blend_mode"] = BlendMode.NORMAL
    default_types["line_blend_mode"] = BlendMode
    defaults_help["line_blend_mode"] = "Blend mode for line. BlendMode enum."

    defaults["line_cap"] = LineCap.BUTT
    default_types["line_cap"] = LineCap
    defaults_help["line_cap"] = "Line cap for line. LineCap enum."

    defaults["line_color"] = colors.black
    default_types["line_color"] = colors.Color
    defaults_help["line_color"] = "Line color. Color object."

    defaults["line_dash_array"] = None
    default_types["line_dash_array"] = list
    defaults_help["line_dash_array"] = "Line dash array. List of floats."

    defaults["line_dash_phase"] = 0
    default_types["line_dash_phase"] = float
    defaults_help["line_dash_phase"] = (
        "Line dash phase. Positive float. Length in <points>."
    )

    defaults["line_join"] = LineJoin.MITER
    default_types["line_join"] = LineJoin
    defaults_help["line_join"] = "Line join for line. LineJoin enum."

    defaults["line_miter_limit"] = 10
    default_types["line_miter_limit"] = float
    defaults_help["line_miter_limit"] = "Line miter limit. Positive float."

    defaults["line_width"] = 1
    default_types["line_width"] = float
    defaults_help["line_width"] = "Line width. Positive float. Length in <points>."

    defaults["lualatex_run_options"] = None
    default_types["lualatex_run_options"] = str
    defaults_help["lualatex_run_options"] = "LuaLaTeX run options. String."

    defaults["main_font"] = "Times New Roman"
    default_types["main_font"] = str
    defaults_help["main_font"] = "Main font. String."

    defaults["margin"] = 1
    default_types["margin"] = float
    defaults_help["margin"] = "Margin. Positive float. Length in <points>."

    defaults["margin_bottom"] = 1
    default_types["margin_bottom"] = float
    defaults_help["margin_bottom"] = (
        "Bottom margin. Positive float. Length in <points>."
    )

    defaults["margin_left"] = 1
    default_types["margin_left"] = float
    defaults_help["margin_left"] = "Left margin. Positive float. Length in <points>."

    defaults["margin_right"] = 1
    default_types["margin_right"] = float
    defaults_help["margin_right"] = "Right margin. Positive float. Length in <points>."

    defaults["margin_top"] = 1  # to do! change these to point units
    default_types["margin_top"] = float
    defaults_help["margin_top"] = "Top margin. Positive float. Length in <points>."

    defaults["marker"] = None
    default_types["marker"] = object  # Assuming marker is a custom object
    defaults_help["marker"] = "Marker. Marker object."

    defaults["marker_color"] = colors.black
    default_types["marker_color"] = colors.Color
    defaults_help["marker_color"] = "Marker color. Color object."

    defaults["marker_line_style"] = "solid"
    default_types["marker_line_style"] = str
    defaults_help["marker_line_style"] = "Marker line style. String."

    defaults["marker_line_width"] = 1
    default_types["marker_line_width"] = float
    defaults_help["marker_line_width"] = (
        "Marker line width. Positive float. Length in <points>."
    )

    defaults["marker_palette"] = seq_MATTER_256  # this needs to be a 256 color
    # palette
    default_types["marker_palette"] = list
    defaults_help["marker_palette"] = "Marker palette. List of colors."

    defaults["marker_radius"] = 3  # Used for MarkerType.CIRCLE, MarkerType.STAR
    default_types["marker_radius"] = float
    defaults_help["marker_radius"] = (
        "Marker radius. Positive float. Length in <points>."
    )

    defaults["marker_size"] = 3  # To do: find out what the default is
    default_types["marker_size"] = float
    defaults_help["marker_size"] = "Marker size. Positive float. Length in <points>."

    defaults["marker_type"] = MarkerType.FCIRCLE
    default_types["marker_type"] = MarkerType
    defaults_help["marker_type"] = "Marker type. MarkerType enum."

    defaults["markers_only"] = False
    default_types["markers_only"] = bool
    defaults_help["markers_only"] = (
        "Boolean property for drawing markers only. If True, only markers are drawn."
    )

    defaults["mask"] = None
    default_types["mask"] = object  # Assuming mask is a custom object
    defaults_help["mask"] = "Mask. Mask object."

    defaults["merge"] = True  # merge transformations with reps > 0
    default_types["merge"] = bool
    defaults_help["merge"] = (
        "Boolean property for merging transformations. "
        "If True, transformations with reps > 0 are merged."
    )

    defaults["merge_tol"] = 0.01  # if the distance between two nodes is less
    # than this value,
    default_types["merge_tol"] = float
    defaults_help["merge_tol"] = "Merge tolerance. Positive float. Length in <points>."
    # defaults['min_height'] = 10
    # defaults['min_width'] = 20
    # defaults['min_size'] = 50

    defaults["mono_font"] = "Courier New"
    default_types["mono_font"] = str
    defaults_help["mono_font"] = "Monospace font. String."

    defaults['n_arc_points'] = 40  # number of proportional points for arcs
    default_types['n_arc_points'] = int
    defaults_help['n_arc_points'] = 'Number of points for arcs. Positive integer.'

    defaults['n_circle_points'] = 30  # number of points for circles
    default_types['n_circle_points'] = int
    defaults_help['n_circle_points'] = 'Number of points for circles. Positive integer.'

    defaults['n_bezier_points'] = 40  # number of points for Bezier curves
    default_types['n_bezier_points'] = int
    defaults_help['n_bezier_points'] = 'Number of points for Bezier curves. Positive integer.'

    defaults['n_ellipse_points'] = 40  # number of points for ellipses
    default_types['n_ellipse_points'] = int
    defaults_help['n_ellipse_points'] = 'Number of points for ellipses. Positive integer.'

    defaults['n_hobby_points'] = 40  # number of points for Hobby curves
    default_types['n_hobby_points'] = int
    defaults_help['n_hobby_points'] = 'Number of points for Hobby curves. Positive integer.'

    defaults['n_q_bezier_points'] = 30  # number of points for quadratic Bezier curves
    default_types['n_q_bezier_points'] = int
    defaults_help['n_q_bezier_points'] = 'Number of points for quadratic Bezier curves. Positive integer.'

    defaults["n_round"] = 2  # used for rounding floats
    default_types["n_round"] = int
    defaults_help["n_round"] = (
        "Number of decimal places to round floats. Positive integer."
    )

    defaults["old_style_nums"] = False
    default_types["old_style_nums"] = bool
    defaults_help["old_style_nums"] = (
        "Boolean property for old style numbers. "
        "If True, old style numbers are used."
    )

    defaults["orientation"] = PageOrientation.PORTRAIT  # PORTRAIT, LANDSCAPE
    default_types["orientation"] = PageOrientation
    defaults_help["orientation"] = "Page orientation. PageOrientation enum."

    defaults["output_dir"] = None  # output directory for TeX files if None, use
    # the current directory
    default_types["output_dir"] = str
    defaults_help["output_dir"] = "Output directory for TeX files. String."

    defaults["overline"] = False
    default_types["overline"] = bool
    defaults_help["overline"] = (
        "Boolean property for overline. If True, overline is used."
    )

    defaults["overwrite_files"] = False
    default_types["overwrite_files"] = bool
    defaults_help["overwrite_files"] = (
        "Boolean property for overwriting files. If True, files are overwritten."
    )

    defaults["packages"] = ["tikz", "pgf"]
    default_types["packages"] = list
    defaults_help["packages"] = "Packages. List of strings."

    defaults["page_grid_back_color"] = colors.white
    default_types["page_grid_back_color"] = colors.Color
    defaults_help["page_grid_back_color"] = "Page grid background color. Color object."

    defaults["page_grid_line_color"] = colors.gray
    default_types["page_grid_line_color"] = colors.Color
    defaults_help["page_grid_line_color"] = "Page grid line color. Color object."

    defaults["page_grid_line_dash_array"] = [2, 2]
    default_types["page_grid_line_dash_array"] = list
    defaults_help["page_grid_line_dash_array"] = (
        "Page grid line dash array. List of floats."
    )

    defaults["page_grid_line_width"] = 0.5
    default_types["page_grid_line_width"] = float
    defaults_help["page_grid_line_width"] = (
        "Page grid line width. Positive float. Length in <points>."
    )

    defaults["page_grid_spacing"] = 18
    default_types["page_grid_spacing"] = float
    defaults_help["page_grid_spacing"] = (
        "Page grid spacing. Positive float. Length in <points>."
    )

    defaults["page_grid_x_shift"] = 0
    default_types["page_grid_x_shift"] = float
    defaults_help["page_grid_x_shift"] = (
        "Page grid x shift. Positive float. Length in <points>."
    )

    defaults["page_grid_y_shift"] = 0
    default_types["page_grid_y_shift"] = float
    defaults_help["page_grid_y_shift"] = (
        "Page grid y shift. Positive float. Length in <points>."
    )

    defaults["page_margins"] = PageMargins.CUSTOM
    default_types["page_margins"] = PageMargins
    defaults_help["page_margins"] = "Page margins. PageMargins enum."

    defaults["page_number_position"] = PageNumberPosition.BOTTOM_CENTER
    default_types["page_number_position"] = PageNumberPosition
    defaults_help["page_number_position"] = (
        "Page number position. PageNumberPosition enum."
    )

    defaults["page_numbering"] = PageNumbering.NONE
    default_types["page_numbering"] = PageNumbering
    defaults_help["page_numbering"] = "Page numbering. PageNumbering enum."

    defaults["page_size"] = PageSize.A4  #  A0, A1, A2, A3, A4, A5, A6, B0, B1, B2,
    # B3, B4, B5, B6, LETTER, LEGAL,
    # EXECUTIVE, 11X17
    default_types["page_size"] = PageSize
    defaults_help["page_size"] = "Page size. PageSize enum."

    defaults["pattern_style"] = None
    default_types["pattern_style"] = object  # Assuming pattern style is a custom object
    defaults_help["pattern_style"] = "Pattern style. PatternStyle object."

    defaults["pattern_type"] = PatternType.HORIZONTAL_LINES  #  DOTS, HATCH, STARS
    default_types["pattern_type"] = PatternType
    defaults_help["pattern_type"] = "Pattern type. PatternType enum."

    defaults["pattern_color"] = colors.black
    default_types["pattern_color"] = colors.Color
    defaults_help["pattern_color"] = "Pattern color. Color object."

    defaults["pattern_distance"] = 3  # distance between items
    default_types["pattern_distance"] = float
    defaults_help["pattern_distance"] = (
        "Pattern distance. Positive float. Length in <points>."
    )

    defaults["pattern_angle"] = 0  # angle of the pattern in radians
    default_types["pattern_angle"] = float
    defaults_help["pattern_angle"] = "Pattern angle. Float. Angle in radians."

    defaults["pattern_x_shift"] = 0  # shift in the x direction
    default_types["pattern_x_shift"] = float
    defaults_help["pattern_x_shift"] = (
        "Pattern x shift. Positive float. Length in <points>."
    )

    defaults["pattern_y_shift"] = 0  # shift in the y direction
    default_types["pattern_y_shift"] = float
    defaults_help["pattern_y_shift"] = (
        "Pattern y shift. Positive float. Length in <points>."
    )

    defaults["pattern_line_width"] = 0  # line width for LINES and HATCH
    default_types["pattern_line_width"] = float
    defaults_help["pattern_line_width"] = (
        "Pattern line width. Positive float. Length in <points>."
    )

    defaults["pattern_radius"] = 10  # radius of the circle for STARS
    default_types["pattern_radius"] = float
    defaults_help["pattern_radius"] = (
        "Pattern radius. Positive float. Length in <points>."
    )

    defaults["pattern_points"] = 5  # number of points for STAR
    default_types["pattern_points"] = int
    defaults_help["pattern_points"] = "Pattern points. Positive integer."

    defaults["pdflatex_run_options"] = None
    default_types["pdflatex_run_options"] = str
    defaults_help["pdflatex_run_options"] = "PDFLaTeX run options. String."

    defaults["plait_color"] = colors.bluegreen
    default_types["plait_color"] = colors.Color
    defaults_help["plait_color"] = "Plait color. Color object."

    defaults["preamble"] = ""
    default_types["preamble"] = str
    defaults_help["preamble"] = "Preamble. String."

    defaults["radius_threshold"] = 1  # used for grouping fragments in a lace object
    default_types["radius_threshold"] = float
    defaults_help["radius_threshold"] = (
        "Radius threshold. Positive float. Length in <points>. "
    )
    defaults["random_marker_colors"] = True
    default_types["random_marker_colors"] = bool
    defaults_help["random_marker_colors"] = (
        "Boolean property for random marker colors. "
        "If True, random marker colors are used."
    )

    defaults["random_node_colors"] = True
    default_types["random_node_colors"] = bool
    defaults_help["random_node_colors"] = (
        "Boolean property for random node colors. "
        "If True, random node colors are used."
    )

    defaults["rectangle_width_height"] = (40, 20)  # width and height of the rectangle
    default_types["rectangle_width_height"] = tuple
    defaults_help["rectangle_width_height"] = (
        "Width and height of the rectangle. "
        "Tuple of two positive floats. Length in <points>."
    )

    defaults["render"] = "TEX"  # Render.TEX, Render.SVG, Render.PNG use string values
    default_types["render"] = str
    defaults_help["render"] = "Render. Render enum."

    defaults["rev_arrow_length"] = 20  # length of reverse arrow
    default_types["rev_arrow_length"] = float
    defaults_help["rev_arrow_length"] = (
        "Length of reverse arrow. Positive float. Length in <points>."
    )

    defaults["rtol"] = (
        0  # used for comparing floats. If this is 0 then only atol is used
    )
    default_types["rtol"] = float
    defaults_help["rtol"] = "Relative tolerance. Positive float. Length in <points>. "

    defaults["sans_font"] = str
    defaults_help["sans_font"] = "Sans font. String."

    defaults["save_with_versions"] = (
        False  # if the file exists, save with a version number
    )
    default_types["save_with_versions"] = bool
    defaults_help["save_with_versions"] = (
        "Boolean property for saving with versions. "
        "If True, files are saved with a version number."
    )

    defaults["section_color"] = colors.black
    default_types["section_color"] = colors.Color
    defaults_help["section_color"] = "Section color. Color object."

    defaults["section_dash_array"] = None
    default_types["section_dash_array"] = list
    defaults_help["section_dash_array"] = "Section dash array. List of floats."

    defaults["section_line_cap"] = LineCap.BUTT.value
    default_types["section_line_cap"] = LineCap
    defaults_help["section_line_cap"] = "Section line cap. LineCap enum."

    defaults["section_line_join"] = LineJoin.MITER.value
    default_types["section_line_join"] = LineJoin
    defaults_help["section_line_join"] = "Section line join. LineJoin enum."

    defaults["section_width"] = 1
    default_types["section_width"] = float
    defaults_help["section_width"] = (
        "Section width. Positive float. Length in <points>."
    )

    defaults["shade_axis_angle"] = (
        pi / 4
    )  # angle from the x-axis for the shading in radians
    default_types["shade_axis_angle"] = float
    defaults_help["shade_axis_angle"] = (
        "Axis angle for shading. Float. Angle in radians."
    )

    defaults["shade_ball_color"] = colors.black
    default_types["shade_ball_color"] = colors.Color
    defaults_help["shade_ball_color"] = "Ball color for shading. Color object."

    defaults["shade_bottom_color"] = colors.white
    default_types["shade_bottom_color"] = colors.Color
    defaults_help["shade_bottom_color"] = "Bottom color for shading. Color object."

    defaults["shade_inner_color"] = colors.white
    default_types["shade_inner_color"] = colors.Color
    defaults_help["shade_inner_color"] = "Inner color for shading. Color object."

    defaults["shade_middle_color"] = colors.white
    default_types["shade_middle_color"] = colors.Color
    defaults_help["shade_middle_color"] = "Middle color for shading. Color object."

    defaults["shade_outer_color"] = colors.white
    default_types["shade_outer_color"] = colors.Color
    defaults_help["shade_outer_color"] = "Outer color for shading. Color object."

    defaults["shade_left_color"] = colors.black
    default_types["shade_left_color"] = colors.Color
    defaults_help["shade_left_color"] = "Left color for shading. Color object."

    defaults["shade_lower_left_color"] = colors.black
    default_types["shade_lower_left_color"] = colors.Color
    defaults_help["shade_lower_left_color"] = (
        "Lower left color for shading. Color object."
    )

    defaults["shade_lower_right_color"] = colors.white
    default_types["shade_lower_right_color"] = colors.Color
    defaults_help["shade_lower_right_color"] = (
        "Lower right color for shading. Color object."
    )

    defaults["shade_right_color"] = colors.white
    default_types["shade_right_color"] = colors.Color
    defaults_help["shade_right_color"] = "Right color for shading. Color object."

    defaults["shade_top_color"] = colors.black
    default_types["shade_top_color"] = colors.Color
    defaults_help["shade_top_color"] = "Top color for shading. Color object."

    defaults["shade_type"] = ShadeType.AXIS_TOP_BOTTOM
    default_types["shade_type"] = ShadeType
    defaults_help["shade_type"] = "Shade type. ShadeType enum."

    defaults["shade_upper_left_color"] = colors.black
    default_types["shade_upper_left_color"] = colors.Color
    defaults_help["shade_upper_left_color"] = (
        "Upper left color for shading. Color object."
    )

    defaults["shade_upper_right_color"] = colors.white
    default_types["shade_upper_right_color"] = colors.Color
    defaults_help["shade_upper_right_color"] = (
        "Upper right color for shading. Color object."
    )

    defaults["show_browser"] = True
    default_types["show_browser"] = bool
    defaults_help["show_browser"] = (
        "Boolean property for showing the browser. If True, the browser is shown."
    )

    defaults["show_log_on_console"] = True  # show log messages on console
    default_types["show_log_on_console"] = bool
    defaults_help["show_log_on_console"] = (
        "Boolean property for showing LateX log messages on console. "
        "If True, log messages are shown on console."
    )

    defaults["slanted"] = False
    default_types["slanted"] = bool
    defaults_help["slanted"] = (
        "Boolean property for slanted font. "
        "If True, the font is displayed in slanted."
    )

    defaults["small_caps"] = False
    default_types["small_caps"] = bool
    defaults_help["small_caps"] = (
        "Boolean property for small caps font. "
        "If True, the font is displayed in small caps."
    )

    defaults["smooth"] = False
    default_types["smooth"] = bool
    defaults_help["smooth"] = (
        "Boolean property for smooth lines. If True, lines are smooth."
    )

    defaults["strike_through"] = False
    default_types["strike_through"] = bool
    defaults_help["strike_through"] = (
        "Boolean property for strike through. If True, strike through is used."
    )

    defaults["stroke"] = True
    default_types["stroke"] = bool
    defaults_help["stroke"] = "Boolean property for stroke. If True, stroke is used."

    defaults["swatch"] = seq_MATTER_256
    default_types["swatch"] = list
    defaults_help["swatch"] = "Swatch. List of colors."

    defaults["tag_align"] = Align.LEFT
    default_types["tag_align"] = Align
    defaults_help["tag_align"] = "Tag text alignment. Align enum."

    defaults["tag_alpha"] = 1
    default_types["tag_alpha"] = float
    defaults_help["tag_alpha"] = (
        "Alpha value for tag transparency. Float between 0 and 1."
    )

    defaults["tag_blend_mode"] = BlendMode.NORMAL
    default_types["tag_blend_mode"] = BlendMode
    defaults_help["tag_blend_mode"] = "Blend mode for tag. BlendMode enum."

    defaults["temp_dir"] = "sytem_temp_dir"
    default_types["temp_dir"] = str
    defaults_help["temp_dir"] = "Temporary directory. String."

    defaults["text_offset"] = 5  # gap between text and dimension line
    default_types["text_offset"] = float
    defaults_help["text_offset"] = "Text offset. Positive float. Length in <points>."

    defaults["text_width"] = None  # width of the text box
    default_types["text_width"] = float
    defaults_help["text_width"] = "Text width. Positive float. Length in <points>."

    defaults["tikz_libraries"] = [
        "plotmarks",
        "calc",
        "shapes.multipart",
        "arrows",
        "decorations.pathmorphing",
        "decorations.markings",
        "backgrounds",
        "patterns",
        "patterns.meta",
        "shapes",
        "shadings",
    ]
    default_types["tikz_libraries"] = list
    defaults_help["tikz_libraries"] = "TikZ libraries. List of strings."

    defaults["tikz_nround"] = 3
    default_types["tikz_nround"] = int
    defaults_help["tikz_nround"] = (
        "Number of decimal places to round floats in TikZ. Positive integer."
    )

    defaults["tikz_scale"] = 1
    default_types["tikz_scale"] = float
    defaults_help["tikz_scale"] = "TikZ scale. Positive float."

    defaults["tol"] = 0.005  # used for comparing angles and collinearity
    default_types["tol"] = float
    defaults_help["tol"] = "Tolerance. Positive float. Length in <points>."

    defaults["underline"] = False
    default_types["underline"] = bool
    defaults_help["underline"] = (
        "Boolean property for underline. If True, underline is used."
    )

    defaults["use_packages"] = ["tikz", "pgf"]
    default_types["use_packages"] = list
    defaults_help["use_packages"] = "Use packages. List of strings."

    defaults["validate"] = False
    default_types["validate"] = bool
    defaults_help["validate"] = (
        "Boolean property for validating. If True, validation is used."
    )

    defaults["visible"] = True
    default_types["visible"] = bool
    defaults_help["visible"] = "Boolean property for visible. If True, visible is used."

    defaults["xelatex_run_options"] = None
    default_types["xelatex_run_options"] = str
    defaults_help["xelatex_run_options"] = "XeLaTeX run options. String."

    defaults["x_marker"] = (
        2  # a circle with radius=2 will be drawn at each intersection
    )
    default_types["x_marker"] = float
    defaults_help["x_marker"] = (
        "Marker for intersection points. Positive float. Length in <points>."
    )

    defaults["x_visible"] = False  # do not show intersection points by default
    default_types["x_visible"] = bool
    defaults_help["x_visible"] = (
        "Boolean property for visible intersection points. "
        "If True, intersection points are visible."
    )

    # styles need to be set after the defaults are set
    defaults["marker_style"] = MarkerStyle()
    defaults["line_style"] = LineStyle()
    defaults["fill_style"] = FillStyle()
    defaults["circle_style"] = ShapeStyle()
    defaults["edge_style"] = LineStyle()
    defaults["plait_style"] = ShapeStyle()
    defaults["section_style"] = LineStyle()
    defaults["shape_style"] = ShapeStyle()
    defaults["tag_frame_style"] = FrameStyle()
    defaults["tag_style"] = TagStyle()



tikz_defaults = defaultdict(str)


def set_tikz_defaults():
    """Sets the default values for the TikZ objects."""
    from ..colors import colors
    from ..graphics.all_enums import LineCap, LineJoin, BlendMode

    tikz_defaults.update(
        {
            "color": colors.black,
            "line width": 1,
            "line cap": LineCap.BUTT,
            "line join": LineJoin.MITER,
            "fill": colors.black,
            "fill opacity": 1,
            "draw": colors.black,
            "miter limit": 10,
            "dash pattern": [],
            "dash phase": 0,
            "blend mode": BlendMode.NORMAL,
            "font": "",
            "font size": 12,
            "font color": colors.black,
            "font opacity": 1,
            "text opacity": 1,
            "rotate": 0,
        }
    )

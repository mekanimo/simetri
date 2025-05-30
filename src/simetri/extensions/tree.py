"""Create a tree structure and draw it."""
import os
from typing import Sequence, Any

import simetri.graphics as sg


diamond = sg.Shape([(0, 5), (3, 0), (0, -5), (-3, 0)], closed=True)
diamond.fill_color = sg.black
diamond.stroke = False
star2 = sg.regular_star_polygon(5, 2, 7)
star2.set_attribs('fill_color', sg.red)
star2.set_attribs('stroke', False)
square = sg.Shape([(0, 5), (5, 5), (5, 0), (0, 0)], closed=True)
star = sg.regular_star_polygon(8, 3, 7)
star.set_attribs('fill_color', sg.red)
star.set_attribs('stroke', False)
star3 = sg.regular_star_polygon(8, 2, 5)
star3.set_attribs('fill_color', sg.blue)
star3.set_attribs('stroke', False)
circle = sg.Circle((0, 0), 1.5, fill_color=sg.white, stroke=False)
hexagon = sg.Batch([sg.reg_poly_shape((0, 0), 6, 4, fill_color=sg.teal,
                                                    stroke=False), circle])


def next_id():
    """Generates a unique ID for each node."""
    next_id.counter += 1
    return next_id.counter


next_id.counter = 0


class TreeNode:
    """TreeNode object representing a tree structure.
    Each node has a tag, an id, a list of children, and optional extra attributes.

    Args:
        tag (str): The tag of the node.
        children (list): A list of child nodes. Defaults to an empty list.
        extra (str): Optional extra attribute for the node.
        **kwargs: Additional keyword arguments for the node. (font_size, font_color, bold)

    Returns:
        None
    """

    def __init__(
        self,
        tag: str = "",
        children: Sequence["TreeNode"] = None,
        extra: Any = None,
        font_size = 12,
        font_color = sg.black,
        bold = False
    ):
        self.tag = tag
        self.id = next_id()
        self.children = children if children is not None else []
        self.extra = extra
        self.font_size = font_size
        self.font_color = font_color
        self.bold = bold

    def add_child(self, child):
        """Adds a child node to the current node.
            Args:
                child (TreeNode): The child node to add.

            Returns:
                None
        """
        if child.id not in [c.id for c in self.children]:
            self.children.append(child)

    def num_all_children(self):
        """Counts the number of children and grandchildren of the node.
            Args:
                None

            Returns:
                int: The number of children and grandchildren.
        """
        return len(self.children) + sum(
            child.num_all_children() for child in self.children
        )

    def depth(self):
        """Calculates the depth of the node in the tree.
            Args:
                None


            Returns:
                int: The depth of the node.
        """
        if not self.children:
            return 0
        n = self.num_all_children()
        m = self.children[-1].num_all_children()
        return n - m


def make_tree(
    node,
    canvas: "Canvas" = None,
    file_path: str = None,
    overwrite: bool = False,
    dx: float = 10,
    dy: float = 18,
    icons = None,
    line1_color = sg.gray,
    line1_width = 1.75,
    line1_cap = sg.LineCap.ROUND,
    line2_color = sg.gray,
    line2_width = 1,
    line2_cap = sg.LineCap.ROUND,
    scale=1
):
    """Creates a tree structure and draws it on the canvas.
    Args:
        node: The root node of the tree.
        canvas: The canvas to draw the tree structure.
        file_path: The file path to save the tree structure.
        overwrite: Whether to overwrite the existing file.
        dx: The horizontal distance between nodes.
        dy: The vertical distance between nodes.
        icons: A list of icons to use for the nodes.
        line1_color: The color of the first line.
        line1_width: The width of the first line.
        line1_cap: The cap style of the first line.
        line2_color: The color of the second line.
        line2_width: The width of the second line.
        line2_cap: The cap style of the second line.

    Returns:
        None
"""
    count = 0
    if icons is None:
        icons = [star, diamond, star3, hexagon]

    icon1, icon2, icon3, icon4 = icons

    def draw_tree(node, indent: int = 0, canvas: "Canvas" = None):
        """Prints the tree structure of the node and its children.
        Args:
            node: The node to print.
            indent: The indentation level for the current node.
            canvas: The canvas to draw the tree structure.
            file_path: The file path to save the tree structure.
            overwrite: Whether to overwrite the existing file.

        Returns:
            None
        """
        nonlocal count
        count += 1
        x = indent * dx
        y = -count * dy
        x1 = x
        y1 = y - node.depth() * dy
        cx, cy = x, y
        if node.depth() > 0:
            canvas.line(
                (x, y),
                (x1, y1),
                line_color=line1_color,
                line_width=line1_width,
                line_cap=line1_cap,
            )


        x2 = x1 + indent * dx
        y2 = y
        if indent > 0:
            x -= dx
        p1 = (x, y)
        p2 = (x2, y2)
        if not sg.close_points2(p1, p2):
            canvas.line(
                p1,
                p2,
                line_color=line2_color,
                line_width=line2_width,
                line_cap=line2_cap,
            )
        if node.depth() > 0 and indent > 0:
            icon3.move_to((cx, cy))
            canvas.draw(icon3)
        font_size = node.font_size
        font_color = node.font_color
        bold = node.bold

        if indent > 1:
            #
            icon4.move_to((x, y))
            canvas.draw(icon4)
        elif indent == 1:
            icon2.move_to((x, y))
            canvas.draw(icon2)
        else:
            icon1.move_to((x, y))
            canvas.draw(icon1)
        canvas.text(
            node.tag,
            (x2, y2),
            font_family=sg.FontFamily.MONOSPACE,
            font_size=font_size,
            color=sg.red,
            anchor=sg.Anchor.WEST,
            fill=False,
            bold=bold,
            font_color=font_color,
        )
        for child in node.children:
            draw_tree(child, indent + 1, canvas=canvas)

    draw_tree(node, canvas=canvas)
    if scale != 1:
        canvas.scale = (scale, scale)
    canvas.save(file_path, overwrite=overwrite)


isdir = os.path.isdir
join = os.path.join

def print_tree(root_dir, prefix=""):
    """ Recursively generates a file tree using Unicode characters """
    entries = sorted(os.listdir(root_dir))
    entries = [e for e in entries if not e.startswith('.')]  # Hide hidden files

    for i, entry in enumerate(entries):
        path = os.path.join(root_dir, entry)
        is_last = i == len(entries) - 1

        # Unicode branch characters
        if not os.path.isdir(path):
            print(f"{prefix}{'└──' if is_last else '├──'}🗎 {entry}")
        else:
            print(f"{prefix}{'└──' if is_last else '├──'}📁 {entry}")

        if os.path.isdir(path):
            new_prefix = f"{prefix}{'    ' if is_last else '│   '}"
            print_tree(path, new_prefix)

def list_directories(path):
    """List all directories in a given path.
    Args:
        path (str): The path to search for directories.
    Returns:
        list: A list of directories in the given path.
    """
    return [e for e in os.listdir(path) if isdir(join(path, e))]




# Example Usage
# root_directory = "c:/simetri-blog/"  # Change this to your desired directory path
# print("📁 File Structure:\n")
# print_tree(root_directory)

# canvas = sg.Canvas()
# root = TreeNode("{} Base", extra="root", font_color=sg.orange)
# methods = TreeNode("Methods", font_color=sg.blue)
# root.add_child(methods)
# transforms = [
#     "translate",
#     "rotate",
#     "mirror",
#     "glide",
#     "scale",
#     "shear",
#     "transform",
# ]
# args = [
#     "(dx: float, dy: float)",
#     "(angle: float, about: Point)",
#     "(about: Line)",
#     "(glide_line: Line, glide_dist: float)",
#     "(scale_x: float, scale_y: float)",
#     "(shear_x:float, shear_y:)",
#     "(transform_matrix: ndarray)",
# ]

# for i, trans in enumerate(transforms):
#     methods.add_child(TreeNode(f"{trans}{args[i][:-1]}, reps: int=0) -> Self"))

# make_tree(
#     root, canvas=canvas, file_path="c:/tmp/tree_generator4.pdf", overwrite=True
# )

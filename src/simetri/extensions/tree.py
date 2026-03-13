"""Create and render tree structures."""

import enum
import os
from typing import Sequence, Any
import inspect
import re

import simetri.graphics as sg


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
        font_size=12,
        font_color=sg.black,
        bold=False,
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
    canvas: Any = None,
    file_path: str = None,
    overwrite: bool = False,
    dx: float = 10,
    dy: float = 18,
    icons=None,
    line1_color=sg.gray,
    line1_width=.5,
    line1_cap=sg.LineCap.ROUND,
    line2_color=sg.gray,
    line2_width=.75,
    line2_cap=sg.LineCap.ROUND,
    scale=1,
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
    diamond = sg.Shape([(0, 5), (3, 0), (0, -5), (-3, 0)], closed=True)
    diamond.fill_color = sg.black
    diamond.stroke = False

    star_polygon = sg.reg_star_polygon

    star2 = star_polygon(5, 2, 5, fill_color=sg.blue, stroke=False)
    star2.rotate(sg.pi/10)
    d = 6
    square = sg.Shape([(0, d), (d, d), (d, 0), (0, 0)], closed=True)
    square.fill_color = sg.blue
    square.stroke = False
    star = star_polygon(8, 3, 7, fill_color=sg.red, stroke=False)
    circle = sg.Circle(1.5, fill_color=sg.white, stroke=False)
    hexagon = sg.Batch(
        [sg.reg_poly_shape(6, 4, (0, 0), fill_color=sg.teal, stroke=False), circle]
    )

    # Single red diamond for enum values
    red_diamond = sg.Shape([(0, 4), (3, 0), (0, -4), (-3, 0)], closed=True)
    red_diamond.fill_color = sg.red
    red_diamond.stroke = False
    # Two red diamonds side-by-side for enum classes
    _rd1 = sg.Shape([(-3, 4), (0, 0), (-3, -4), (-6, 0)], closed=True)
    _rd2 = sg.Shape([(3, 4), (6, 0), (3, -4), (0, 0)], closed=True)
    _rd1.fill_color = _rd2.fill_color = sg.red
    _rd1.stroke = _rd2.stroke = False
    double_red_diamond = sg.Batch([_rd1, _rd2])

    count = 0
    if icons is None:
        icons = [star, diamond, star2, hexagon]

    icon1, icon2, icon3, icon4 = icons
    icon5 = red_diamond        # enum value / enum class
    icon6 = double_red_diamond  # (unused, kept for reference)
    icon7 = sg.Batch(
        [sg.reg_poly_shape(6, 4, (0, 0), fill_color=sg.teal, stroke=False),
         sg.Circle(1.5, fill_color=sg.white, stroke=False)]
    )  # method

    def node_icon(node):
        if node.extra == "enum_value":
            return icon5
        if node.extra == "enum_class":
            return icon5
        if node.extra == "method":
            return icon7
        if node.extra == "property":
            return icon4
        if node.extra == "class":
            return icon3
        return icon2

    def draw_tree(node, indent: int = 0, canvas: Any = None):
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

        # Shorter horizontal connectors from level 1 and deeper
        horiz_span = 1 if indent >= 1 else 0
        x2 = x1 + horiz_span * dx
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
            icon = node_icon(node)
            icon.move_to((cx, cy))
            canvas.draw(icon)
        font_size = node.font_size
        font_color = node.font_color
        bold = node.bold

        if indent > 0:
            icon = node_icon(node)
            icon.move_to((x, y))
            canvas.draw(icon)
        else:
            icon1.move_to((x, y))
            canvas.draw(icon1)

        title_x = x2 + 8 if indent == 0 else x2
        label = str(node.tag)
        canvas.text(
            label,
            (title_x, y2),
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
    if file_path:
        canvas.save(file_path, overwrite=overwrite)


isdir = os.path.isdir
join = os.path.join


def _print_file_tree(root_dir, prefix=""):
    """Recursively generate a file tree using Unicode characters."""
    entries = sorted(os.listdir(root_dir))
    entries = [e for e in entries if not e.startswith(".")]  # Hide hidden files

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
            _print_file_tree(path, new_prefix)


def tree_from_class(
    canvas,
    class_obj,
    exclude=None,
    stubs=None,
    draw=True,
    file_path=None,
    overwrite=False,
    show_enum_values=False,
    **kwargs,
):
    """Create a TreeNode hierarchy from a class and optionally draw it on a canvas.

    Args:
        canvas: Target canvas used for drawing when draw=True.
        class_obj: Root class to introspect.
        exclude: Iterable of attribute/class names to skip.
        stubs: Iterable of class names that should appear as single nodes (no subnodes).
        draw: Whether to draw to canvas.
        file_path: Optional output file path.
        overwrite: Overwrite output file when saving.
        show_enum_values: If True, enum members are listed as child nodes.
        **kwargs: Extra args forwarded to make_tree.
    """
    from typing import Union, get_args, get_origin

    def _type_to_str(tp):
        if tp is inspect._empty:
            return "Any"
        if isinstance(tp, str):
            return tp

        origin = get_origin(tp)
        args = get_args(tp)
        if origin is not None:
            if origin is Union:
                inner = ", ".join(_type_to_str(arg) for arg in args)
                return f"Union[{inner}]"

            origin_name = getattr(origin, "__name__", str(origin).replace("typing.", ""))
            if args:
                inner = ", ".join(_type_to_str(arg) for arg in args)
                return f"{origin_name}[{inner}]"
            return origin_name

        if hasattr(tp, "__name__"):
            return tp.__name__
        return str(tp).replace("typing.", "")

    def _iter_referenced_types(tp):
        origin = get_origin(tp)
        args = get_args(tp)
        if origin is None:
            if tp in {list, tuple, set, Sequence}:
                return
            if inspect.isclass(tp):
                yield tp
            return
        for arg in args:
            if inspect.isclass(arg):
                yield arg
            else:
                yield from _iter_referenced_types(arg)

    def _resolve_type_name_in_context(type_name: str, cls):
        module = inspect.getmodule(cls)
        if module is not None:
            candidate = getattr(module, type_name, None)
            if inspect.isclass(candidate):
                return candidate

        candidate = getattr(sg, type_name, None)
        if inspect.isclass(candidate):
            return candidate
        return None

    def _infer_sequence_item_types(cls, field_name, ann):
        inferred = []
        origin = get_origin(ann)
        args = get_args(ann)

        is_sequence = ann is list or origin in {list, tuple, set, Sequence}
        if not is_sequence or args:
            return inferred

        doc = inspect.getdoc(cls) or ""
        if doc:
            field_pattern = re.compile(
                rf"^\s*{re.escape(field_name)}\s*\([^)]*\)\s*:\s*(.+)$",
                flags=re.IGNORECASE | re.MULTILINE,
            )
            field_match = field_pattern.search(doc)
            if field_match:
                desc = field_match.group(1)
                names = []

                for match in re.finditer(r"\bList\[([A-Za-z_][A-Za-z0-9_]*)\]", desc):
                    names.append(match.group(1))

                for match in re.finditer(
                    r"\b(?:list|sequence|tuple|set)\s+of\s+([A-Za-z_][A-Za-z0-9_]*)",
                    desc,
                    flags=re.IGNORECASE,
                ):
                    names.append(match.group(1))

                for type_name in names:
                    ref_type = _resolve_type_name_in_context(type_name, cls)
                    if ref_type is not None:
                        inferred.append(ref_type)

        return inferred

    def _method_signature(member):
        sig = inspect.signature(member)

        params = []
        for i, (name, param) in enumerate(sig.parameters.items()):
            if i == 0 and name in {"self", "cls"}:
                continue
            params.append(str(param))

        ret_text = _type_to_str(sig.return_annotation)
        return f"({', '.join(params)}) -> {ret_text}"

    def _add_class_tree(node, cls, visited, depth=0, max_depth=2):
        if cls in visited or depth > max_depth:
            return
        if cls.__name__ in stubs:
            return
        visited.add(cls)

        # If the class is an Enum and enum values are requested, list members and stop.
        if show_enum_values and issubclass(cls, enum.Enum):
            for member in cls:
                value_node = TreeNode(f"{member.name} = {member.value!r}", extra="enum_value")
                node.add_child(value_node)
            return

        sig = inspect.signature(cls.__init__)
        properties = {
            name: member
            for name, member in inspect.getmembers(cls)
            if isinstance(member, property)
            and not name.startswith("_")
            and name not in exclude
        }

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                continue
            if name.startswith("_") or name in exclude:
                continue
            if name in properties:
                # Prefer the property representation when both exist.
                continue

            ann = param.annotation
            ref_types = list(_iter_referenced_types(ann))
            if not ref_types:
                ref_types = _infer_sequence_item_types(cls, name, ann)

            ann_text = _type_to_str(ann)
            origin = get_origin(ann)
            args = get_args(ann)
            is_untyped_sequence = (ann is list or origin in {list, tuple, set, Sequence}) and not args
            if is_untyped_sequence and len(ref_types) == 1:
                seq_name = "list" if ann is list or origin is list else (
                    "tuple" if origin is tuple else ("set" if origin is set else "Sequence")
                )
                ann_text = f"{seq_name}[{ref_types[0].__name__}]"

            child = TreeNode(f"{name}: {ann_text}")
            node.add_child(child)

            for ref_type in ref_types:
                if ref_type.__module__.startswith("simetri") and ref_type is not cls:
                    if ref_type.__name__ in exclude:
                        if issubclass(ref_type, enum.Enum) and child.extra is None:
                            child.extra = "enum_class"
                        continue
                    sub_extra = "enum_class" if issubclass(ref_type, enum.Enum) else "class"
                    sub = TreeNode(ref_type.__name__, extra=sub_extra)
                    child.add_child(sub)
                    if ref_type.__name__ not in stubs:
                        _add_class_tree(sub, ref_type, visited.copy(), depth + 1, max_depth)

        for name, member in properties.items():
            ret = inspect.signature(member.fget).return_annotation
            if ret is inspect._empty and name in sig.parameters:
                ret = sig.parameters[name].annotation
            prop_node = TreeNode(f"{name} -> {_type_to_str(ret)}", extra="property")
            node.add_child(prop_node)

            for ref_type in _iter_referenced_types(ret):
                if ref_type.__module__.startswith("simetri") and ref_type is not cls:
                    if ref_type.__name__ in exclude:
                        if issubclass(ref_type, enum.Enum) and prop_node.extra == "property":
                            prop_node.extra = "enum_class"
                        continue
                    sub_extra = "enum_class" if issubclass(ref_type, enum.Enum) else "class"
                    sub = TreeNode(ref_type.__name__, extra=sub_extra)
                    prop_node.add_child(sub)
                    if ref_type.__name__ not in stubs:
                        _add_class_tree(sub, ref_type, visited.copy(), depth + 1, max_depth)

        methods = {
            name: member
            for name, member in inspect.getmembers(cls)
            if inspect.isroutine(member)
            and not name.startswith("_")
            and name not in exclude
        }

        for name, member in methods.items():
            try:
                ret = inspect.signature(member).return_annotation
                ret_str = _type_to_str(ret)
            except (ValueError, TypeError):
                ret_str = "Any"
            method_node = TreeNode(f"{name} -> {ret_str}", extra="method")
            node.add_child(method_node)

    if exclude is None:
        exclude = set()
    exclude = set(exclude)
    if stubs is None:
        stubs = set()
    stubs = set(stubs)

    bases = [base.__name__ for base in class_obj.__bases__ if base is not object]
    title = f"{class_obj.__name__}({', '.join(bases)})" if bases else class_obj.__name__
    root = TreeNode(title, extra="root", font_color=sg.orange)
    _add_class_tree(root, class_obj, visited=set(), depth=0, max_depth=2)

    if draw:
        if canvas is None:
            canvas = sg.Canvas()
        make_tree(root, canvas=canvas, file_path=file_path, overwrite=overwrite, **kwargs)

    return root


def print_tree(
    root,
    prefix="",
    file_tree=True,
    canvas=None,
    file_path=None,
    overwrite=False,
    exclude=None,
    stubs=None,
    **kwargs,
):
    """Print or draw a tree.

    Args:
        root: Root directory (for file trees), a class object, or a TreeNode.
        prefix: Prefix used for console file tree formatting.
        file_tree: If True, use filesystem tree printer (existing behavior).
        canvas: Optional canvas for graphic tree rendering.
        file_path: Optional output path for graphic tree rendering.
        overwrite: Overwrite output file when saving.
        exclude: Optional exclude list used when root is a class.
        stubs: Optional class-name list to render as single non-expanded nodes.
        **kwargs: Extra arguments forwarded to make_tree.
    """
    if file_tree:
        _print_file_tree(root, prefix)
        return None

    if canvas is None:
        canvas = sg.Canvas()

    if inspect.isclass(root):
        node = tree_from_class(canvas, root, exclude=exclude, stubs=stubs, draw=False)
    else:
        node = root

    make_tree(node, canvas=canvas, file_path=file_path, overwrite=overwrite, **kwargs)
    return node


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
#     "(angle: float, about: PointType)",
#     "(about: Line)",
#     "(glide_line: Line, glide_dist: float)",
#     "(scale_x: float, scale_y: float)",
#     "(shear_x:float, shear_y:)",
#     "(transform_matrix: ndarray)",
# ]

# for i, trans in enumerate(transforms):
#     methods.add_child(TreeNode(f"{trans}{args[i][:-1]}, reps: int=0) -> Self"))

# make_tree(
#     root, canvas=canvas, file_path="c:/tmp/tree_generator_test.pdf", overwrite=True
# )

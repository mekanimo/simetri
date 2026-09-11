"""A table object for displaying items in a grid."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Number

from simetri.base.all_enums import Align, Anchor, Types
from simetri.base.common import PointType
from simetri.coloring.colors import Color, check_color, white
from simetri.group.batch import Group
from simetri.shapes.geom_items import Rectangle, Rectangle2
from simetri.shapes.shape import Clipping, Shape


def _inset_values(value, name: str) -> tuple[float, float, float, float]:
    if value is None:
        return 0.0, 0.0, 0.0, 0.0
    if isinstance(value, Number):
        margin = float(value)
        return margin, margin, margin, margin
    if isinstance(value, (list, tuple)) and len(value) == 4:
        left, bottom, right, top = value
        return float(left), float(bottom), float(right), float(top)
    raise ValueError(f"{name} must be a scalar or 4-value sequence")


def _expand_axis_values(values, count: int, name: str) -> list[float]:
    if values is None:
        return [0.0] * count
    if isinstance(values, Number):
        return [float(values)] * count
    if isinstance(values, (list, tuple)) and len(values) == count:
        return [float(value) for value in values]
    raise ValueError(
        f"{name} must be a scalar or a sequence with {count} values"
    )


@dataclass
class Column:
    """Column layout metadata for a ``Table``.

    Attributes:
        width: Column width in points.
        style: ``(border_style_dict, back_color)`` pair.
        x: Left x position of the column.
    """

    width: float
    style: tuple[dict, Color]
    x: float  # x position


@dataclass
class Row:
    """Row layout metadata for a ``Table``.

    Attributes:
        height: Row height in points.
        style: ``(border_style_dict, back_color)`` pair.
        y: Bottom y position of the row.
    """

    height: float
    style: tuple[dict, Color]
    y: float  # y position


@dataclass
class Cell:
    """A single table cell holding an item, size, and optional clipping.

    Attributes:
        item: Drawable content (copied on construction).
        size: ``(width, height)`` of the cell.
        pos: Cell origin position.
        border: Border shape for the cell.
        inset: ``(left, bottom, right, top)`` inset values.
        clip_margins: Optional clip margins inside the cell.
    """

    def __init__(
        self,
        item: Shape | Group,
        size: PointType,
        pos: PointType,
        border: Shape,
        inset: float | list[float] | None = None,
        clip_margins: float | list[float] | None = None,
    ):
        """Create a cell and optionally clip its item.

        Args:
            item: Drawable content to place in the cell (copied).
            size: ``(width, height)`` of the cell.
            pos: Cell origin; the item is moved here.
            border: Border shape for the cell.
            inset: Scalar or 4-value inset ``[left, bottom, right, top]``.
            clip_margins: Optional clip margins inside the cell.
        """
        self.item = item.copy()
        self.size = size
        self.width, self.height = size
        self.pos = pos
        self.border = border
        self.inset = _inset_values(inset, "cell inset")
        self.clip_margins = None
        self.item.move(self.pos)
        if clip_margins is not None:
            left, bottom, right, top = _inset_values(
                clip_margins, "clip_margins"
            )
            self.clip_margins = [left, bottom, right, top]
            width = self.width - (left + right)
            height = self.height - (top + bottom)
            self.clipper = Rectangle(self.pos, width, height)
            self.clipping = Clipping(self.item, self.clipper)


@dataclass
class Table:
    """A Table for drawing objects in a grid."""

    def __init__(
        self,
        items: list | list[list],
        n_rows: int,
        n_columns: int,
        pos: PointType = (0, 0),
        table_size: PointType = None,
        table_back_color: Color = white,
        row_heights: float | list[float] | None = None,
        column_widths: float | list[float] | None = None,
        border_style: dict | None = None,
        inset: float | Sequence[float] | None = None,
        cell_border_style: dict | None = None,
        cell_back_color: Color | list[Color] | None = None,
        row_gaps: float | list[float] | None = None,
        column_gaps: float | list[float] | None = None,
        cell_inset: float | list[float] | None = None,
        scale_cells: bool = False,
        row_style: list[tuple[dict, Color]] | None = None,
        column_style: list[tuple[dict, Color]] | None = None,
        clip_margins: float | list[float] | None = None,
        page_height: float | None = None,
    ):
        """
        items: List of Shape/Group objects. If an item in the list is None then that cell will be empty.

        All units are in points.

        n_rows, n_columns: Number of rows and columns

        pos: If page_height is given, the upper-left offset from the (0, page_height) spot. Otherwise it is the upper_left corner position.

        table_size: (table.width, table.height) If None then it will be compute by using n_rows, n_cols, cell_size, row_gaps, column_gaps, and the inset value/s.

        table_back_color: If the table border_style is None then table_back_color is ignored. If there is a border then it becomes the fill_color of the table_border rectangle.

        row_heights: It can be a single value for all rows or a list of values for the corresponding rows.

        column_widths: It can be a single value for all columns or a list of values for the corresponding columns.

        border_style: If None then there is no visible border. If specified then the border rectangle will be drawn with the given style.

        inset: inset is the gap between the border and the cells. It could be specified as a single value or [left_inset, bottom_inset, right_inset, top_inset] values.

        cell_border_style: If None then there will be no border drawn for the cells. If given then the cell-rectangle will be drawn with this style.

        cell_back_color: If None there will be no back_color. If a single color is given then all cells will have the same background color. If a list or list of lists is given then the corresponding colors will be used.

        row_gaps: Gaps betwen rows. Could be a single value or a list.

        column_gaps: Gaps betwen columns. Could be a single value or a list.

        cell_inset: inset values for the cells. Similar to the table-inset.

        scale_cells: If this is True then the corresponding item will be scaled to fit into the cell (honoring the cell-inset as well)

        row_style: (border-style, back-color) If given then the corresponding cells in the rows will use the border-style and back-color values.

        column_style: (border-style, back-color) If given then the corresponding cells in the columns will use the border-style and back-color values.

        clip_margins: If given then the corresponding items will be clipped by a rectangle sized by using the cell-size and clip-margins (inset values will be ignored if given)


        """
        self.type = Types.GROUP
        self.subtype = Types.TABLE
        self.items = items
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.pos = pos
        self.table_size = table_size
        self.border_style = border_style
        self.table_back_color = table_back_color
        self.cell_border_style = cell_border_style
        self.scale_cells = scale_cells
        self.cell_back_color = cell_back_color
        self.inset = _inset_values(inset, "inset")
        if clip_margins is None:
            self.clip_margins = None
        else:
            self.clip_margins = _inset_values(clip_margins, "clip_margins")

        self.row_gaps = _expand_axis_values(
            row_gaps, max(n_rows - 1, 0), "row_gaps"
        )
        self.column_gaps = _expand_axis_values(
            column_gaps, max(n_columns - 1, 0), "column_gaps"
        )

        self.page_height = page_height
        self.column_widths = _expand_axis_values(
            column_widths, n_columns, "columns"
        )
        self.row_heights = _expand_axis_values(row_heights, n_rows, "rows")
        self.column_styles = self._expand_styles(column_style)
        self.row_styles = self._expand_styles(row_style, "rows")
        self._set_columns()
        self._set_cells()
        self._set_table_size()

    def _expand_styles(self, style, type="column"):
        styles = []
        if isinstance(style, (tuple, List)):
            if type == "column":
                for i in range(self.n_columns):
                    styles.append(style[i])
            else:
                for i in range(self.n_rows):
                    styles.append(style[i])
        else:
            if type == "column":
                for _ in range(self.n_columns):
                    styles.append(style)
            else:
                for _ in range(self.n_rows):
                    styles.append(style)
        return styles

    def _set_columns(self):
        self.columns = []
        left_inset = self.inset[0]
        x0 = left_inset
        for ind in range(self.n_columns):
            width = self.column_widths[ind]
            style = self.column_styles[ind]
            gap = 0
            if ind > 0:
                gap = self.column_gaps[ind - 1]
            x0 += gap
            x0 += width
            column = Column(width, style, x0 - width / 2)
            if self.column_styles:
                column.style = self.column_styles[ind]
            self.columns.append(column)

    def _set_rows(self):
        self.rows = []
        top_inset = self.inset[3]
        y0 = top_inset
        for ind in range(self.n_rows):
            height = self.row_heights[ind]
            style = self.row_styles[ind]
            gap = 0
            if ind > 0:
                gap = self.row_gaps[ind - 1]
            y0 += gap
            y0 += height
            row = Row(height, style, y0 - height / 2)
            if self.row_styles:
                row.style = self.row_styles[ind]
            self.rows.append(row)

    def _set_cells(self):
        self._set_columns()
        self._set_rows()
        self.cells = []
        self.cell_array = []
        count = 0
        for i, row in enumerate(self.rows):
            cell_row = []
            for j, column in enumerate(self.columns):
                center = (column.x, row.y)
                width = column.width
                height = row.height
                size = (width, height)
                style = False
                if row.style is None:
                    style = column.style
                else:
                    style = row.style
                if self.cell_back_color:
                    # back_color = next(self.cell_back_color)
                    back_color = self.cell_back_color
                    if style:
                        style["fill_color"] = back_color
                    else:
                        style = {"fill_color": back_color}
                if style:
                    border = Rectangle(center, width, height, **style)
                else:
                    border = Rectangle(center, width, height)
                cell = Cell(
                    self.items[count],
                    size,
                    center,
                    border,
                    self.inset,
                    self.clip_margins,
                )
                self.cells.append(cell)
                cell_row.append(cell)
                count += 1
            self.cell_array.append(cell_row)

    def _set_table_size(self):
        if self.table_size is None:
            tot_h_insets = self.inset[0] + self.inset[2]
            tot_col_width = sum([col.width for col in self.columns])
            tot_h_gap = sum(self.column_gaps)
            width = tot_h_insets + tot_col_width + tot_h_gap

            tot_v_insets = self.inset[1] + self.inset[3]
            tot_row_height = sum([row.height for row in self.rows])
            tot_v_gap = sum(self.row_gaps)
            height = tot_v_insets + tot_row_height + tot_v_gap

            self.table_size = (width, height)

    def _table_border(self):
        if self.border_style is None:
            return None

        table_left, table_bottom, table_right, table_top = self.table_bounds
        border = Rectangle2(
            (table_left, table_bottom),
            (table_right, table_top),
        )
        if self.table_back_color is None:
            border.fill = False
        else:
            border.fill = True
            border.fill_color = self.table_back_color
        self._apply_style(self.border_style, border)

        return border

    def _fit_item_into_bounds(self, cell, left, bottom, right, top):
        if cell.item is None:
            return
        available_width = right - left
        available_height = top - bottom
        if available_width <= 0 or available_height <= 0:
            raise ValueError("Available cell area must be positive")
        item = cell.item
        item_width = item.width
        item_height = item.height
        if item_width <= 0 or item_height <= 0:
            raise ValueError(
                "Item width and height must be positive for scale_cells"
            )

        scale_factor = min(
            available_width / item_width, available_height / item_height
        )
        item.scale(scale_factor, about=cell.pos)

    def draw_list(self) -> Group:
        """Returns a list of the graphics items that form the grid."""

        table_group = Group(subtype=Types.TABLE)

        table_border = self._table_border()
        if table_border is not None:
            table_group.append(table_border)
        table_group.extend([cell.border for cell in self.cells])
        if self.clip_margins is not None:
            table_group.extend([cell.clipping for cell in self.cells])
        else:
            table_group.extend([cell.item for cell in self.cells])

        return table_group

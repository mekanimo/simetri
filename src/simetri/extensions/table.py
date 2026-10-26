"""Simetri ``Table`` for documentation layouts.

Cells may hold plain text or drawable objects (``Shape``, ``Group``, ``Lace``,
and similar). Plain text is rendered as ``Tag`` sketches for SVG/PDF export;
drawables are sketched inside their cell bounds.

Styling merges in order: table, column, row, cell (later wins). Every
cell uses that same merge. ``header_format`` is a format. ``show_header``
applies it with ``format_row(0)``. There is no header row, header
column, or header cell.

Notebook preview uses :class:`rich.table.Table` (Jupyter integration).

Most layout defaults for this extension live in this module. Cell **fill**
defaults use ``runtime_defaults['table_fill']`` in ``config/settings.py``
(merged before table / column / row / cell layers). Every row uses that
fill.

Cell indexing (read this before addressing cells):

- All indices are **0-based** Python integers. There are **no** letter column
  labels (no ``A``, ``B``, ``C`` coordinates).
- **Columns** are always numbered ``0``, ``1``, ``2``, … left to right.
- **Rows** use one index for :attr:`Table.cells`, :meth:`Table.range`,
  :meth:`Table.cell`, :meth:`Table.row`, :meth:`Table.populate`, and
  ``table.rows[i]``. Every cell is ``row, column`` the same way.
  ``from_bottom_left`` is the only exception: it numbers rows from the
  bottom. A label row from ``show_header`` is an ordinary row, not a
  separate index. ``show_header`` also applies the default header format
  to row ``0``.
- :attr:`Table.columns` uses **column indices** ``0``, ``1``, … on the full
  grid.
"""

from __future__ import annotations

import simetri.graphics  # populate defaults before illustration import chain

from collections.abc import Iterator, Sequence
from typing import Any, Literal, Self

from rich.console import Console
from rich.style import Style as RichStyle
from rich.table import Table as RichTable

from ..base.all_enums import Align, Anchor, Border, BorderLine, Types
from ..base.common import PointType
from ..coloring.colors import Color, black, gray, white
from ..geom.bbox import BoundingBox, bounding_box
from ..geom.affine import translation_matrix
from ..helpers.illustration import Tag
from ..render.draw import (
    _measure_latex_formula,
    _tag_latex_formula,
    get_sketches,
)
from ..render.sketch import TableSketch
from ..shapes.shape import Shape

# Extension defaults (keep in this file).
_TABLE_DEFAULT_STROKE = False
_TABLE_DEFAULT_CELL_PAD_X = 6.0
_TABLE_DEFAULT_CELL_PAD_Y = 4.0
_TABLE_DEFAULT_COLUMN_ALIGN = Align.CENTER
_TABLE_DEFAULT_FONT_SIZE = 10.0
_TABLE_DEFAULT_FORMAT: dict[str, Any] = {
    "align": _TABLE_DEFAULT_COLUMN_ALIGN,
    "font_size": _TABLE_DEFAULT_FONT_SIZE,
}
_TABLE_DEFAULT_GRID_LINE_WIDTH = 0.5
_TABLE_BORDER_THIN = 1.0
_TABLE_BORDER_THICK = 2.25
_TABLE_BORDER_DOUBLE_GAP = 1.5
_BORDER_EDGE_TOP = "top"
_BORDER_EDGE_BOTTOM = "bottom"
_BORDER_EDGE_LEFT = "left"
_BORDER_EDGE_RIGHT = "right"
_BORDER_EDGE_DIAGONAL_DOWN = "diagonal_down"
_BORDER_EDGE_DIAGONAL_UP = "diagonal_up"
_TABLE_DEFAULT_MIN_WIDTH: int | None = None
_TABLE_DEFAULT_PADDING = (0, 1)
_TABLE_DEFAULT_POS: PointType = (0.0, 0.0)
_TABLE_DEFAULT_ROW_ALIGN = Align.VERT_CENTER
_TABLE_DEFAULT_SHOW_HEADER = True
_TABLE_DEFAULT_SHOW_LINES = True
_TABLE_DEFAULT_CAPTION: Tag | str | None = None
_TABLE_DEFAULT_TITLE: Tag | str | None = None
_TABLE_DEFAULT_TITLE_PAD = 8.0


def _merge_style(*layers: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for layer in layers:
        merged.update(layer)
    return merged


def _align_to_rich_justify(align: Align | str) -> str:
    """Map Simetri ``Align`` values to Rich column ``justify`` strings."""
    if isinstance(align, Align):
        key = align
    else:
        key = Align(align)
    rich_map = {
        Align.BOTTOM: "left",
        Align.CENTER: "center",
        Align.FLUSH_CENTER: "center",
        Align.FLUSH_LEFT: "left",
        Align.FLUSH_RIGHT: "right",
        Align.HORIZ_CENTER: "center",
        Align.JUSTIFIED: "full",
        Align.LEFT: "left",
        Align.NONE: "left",
        Align.RIGHT: "right",
        Align.TOP: "left",
        Align.VERT_CENTER: "center",
    }
    return rich_map[key]


def _cell_display_text(content: Any) -> str:
    """Notebook cell text until drawable cells render as graphics."""
    if isinstance(content, SplitCell):
        return f"{content.first} {content.kind} {content.second}"
    if isinstance(content, str):
        return content
    if isinstance(content, (int, float, bool)):
        return str(content)
    try:
        cell_type = content.type
    except AttributeError:
        return str(content)
    return f"[{cell_type}]"


def _is_drawable_content(content: Any) -> bool:
    try:
        content.type
        content.subtype
    except AttributeError:
        return False
    return True


def _resolve_align(align: Align | str) -> Align:
    if isinstance(align, Align):
        return align
    return Align(align)


class SplitCell:
    """Two labels in one table cell, separated by a divider.

    This is cell *content*, not an unmerge operation. ``kind`` is a divider
    character or a ``SplitCell`` class attribute (same strings).

    ``"-"`` / ``HORIZONTAL``: ``first`` on top, ``second`` on bottom.
    ``"|"`` / ``VERTICAL``: ``first`` on the left, ``second`` on the right.
    ``"\\\\"`` / ``DIAGONAL_DOWN``: ``first`` lower-left, ``second`` upper-right.
    ``"/"`` / ``DIAGONAL_UP``: ``first`` upper-left, ``second`` lower-right.

    ``offset1`` and ``offset2`` are ``(dx, dy)`` added after that default
    placement.

    Examples:
        >>> split = SplitCell(SplitCell.HORIZONTAL, "top", "bottom")
        >>> split.kind
        '-'
        >>> split.second
        'bottom'
    """

    HORIZONTAL = "-"
    VERTICAL = "|"
    DIAGONAL_UP = "/"
    DIAGONAL_DOWN = "\\"

    def __init__(
        self,
        kind: str,
        first: str,
        second: str,
        offset1: PointType = (0.0, 0.0),
        offset2: PointType = (0.0, 0.0),
    ) -> None:
        if kind not in _SPLIT_CELL_KINDS:
            raise ValueError(
                f"SplitCell kind must be one of {sorted(_SPLIT_CELL_KINDS)}"
            )
        self.kind = kind
        self.first = first
        self.second = second
        self.offset1 = offset1
        self.offset2 = offset2


_SPLIT_CELL_KINDS = frozenset(
    {
        SplitCell.DIAGONAL_DOWN,
        SplitCell.DIAGONAL_UP,
        SplitCell.HORIZONTAL,
        SplitCell.VERTICAL,
    }
)


class Cell:
    """One table cell: content plus local format and background overrides.

    Examples:
        >>> cell = Cell("value")
        >>> cell.content
        'value'
    """

    def __init__(self, content: Any = "") -> None:
        """Create a cell.

        Args:
            content: Text, number, or drawable object.

        Examples:
            >>> Cell(42).content
            42
        """
        object.__setattr__(self, "_layout_revision", 0)
        self.content = content
        self._format: dict[str, Any] = {}
        self._background: dict[str, Any] = {}
        self._borders: dict[str, dict[str, Any]] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name == "content" and "_layout_revision" in vars(self):
            object.__setattr__(
                self, "_layout_revision", self._layout_revision + 1
            )

    def set_format(self, **kwargs: Any) -> Self:
        """Set cell styling (merged on top of row/column/table).

        Content keys go to ``_format``. Background keys (``fill``,
        ``fill_color``, ``fill_alpha``, ``stroke``, ``line_color``,
        ``line_alpha``, ``line_width``) go to ``_background``, same as
        ``cell.fill_color = ...``.

        Examples:
            >>> cell = Cell("x")
            >>> cell.set_format(bold=True) is cell
            True
            >>> cell._format["bold"]
            True
            >>> _ = cell.set_format(fill=True, fill_color="gold")
            >>> cell._background["fill"]
            True
        """
        format_kwargs, background_kwargs = _split_style_kwargs(kwargs)
        self._format.update(format_kwargs)
        self._background.update(background_kwargs)
        if format_kwargs:
            object.__setattr__(
                self, "_layout_revision", self._layout_revision + 1
            )
        return self

    def set_background(self, **kwargs: Any) -> Self:
        """Set background styling for this cell.

        Examples:
            >>> cell = Cell("x")
            >>> _ = cell.set_background(fill=True)
            >>> cell._background["fill"]
            True
        """
        self._background.update(kwargs)
        return self


class Column:
    """Table column: header label and column-wide styling.

    Examples:
        >>> table = Table(columns=["H"])
        >>> table._columns[0].header
        'H'
    """

    def __init__(self, table: Table, index: int, header: str = "") -> None:
        self._table = table
        self.index = index
        self._header = header
        self._format: dict[str, Any] = {}
        self._background: dict[str, Any] = {}
        self._width: float | None = None
        self.header_cell = Cell(header)

    @property
    def header(self) -> str:
        """Column header label.

        Examples:
            >>> table = Table(columns=["Name"])
            >>> table.column(0).header
            'Name'
        """
        return self._header

    @header.setter
    def header(self, value: str) -> None:
        """Set the column header text.

        Examples:
            >>> table = Table(columns=["A"])
            >>> table.column(0).header = "B"
            >>> table.column(0).header
            'B'
        """
        self._header = value
        self.header_cell.content = value

    def set_format(self, **kwargs: Any) -> Self:
        """Apply format to this column (all body cells and the header cell).

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row("x")
            >>> _ = table.column(0).set_format(bold=True)
            >>> table.cell(0, 0)._format["bold"]
            True
        """
        self._format.update(kwargs)
        self.header_cell.set_format(**kwargs)
        for row in self._table._rows:
            if self.index < len(row.cells):
                row.cells[self.index].set_format(**kwargs)
        return self

    def set_background(self, **kwargs: Any) -> Self:
        """Apply background to this column (all body cells and the header cell).

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.column(0).set_background(fill=True)
            >>> table._columns[0]._background["fill"]
            True
        """
        self._background.update(kwargs)
        self.header_cell.set_background(**kwargs)
        for row in self._table._rows:
            if self.index < len(row.cells):
                row.cells[self.index].set_background(**kwargs)
        return self

    @property
    def b_box(self) -> BoundingBox:
        """Axis-aligned box enclosing this column (header and body cells).

        Examples:
            >>> table = Table(columns=["A", "B"])
            >>> _ = table.add_row(1, 2)
            >>> table.column(0).b_box.width > 0
            True
        """
        return self._table._column_bbox(self.index)


class Row:
    """Table data row: cells and row-wide styling.

    Examples:
        >>> table = Table(columns=["A"])
        >>> row = table.add_row("v")
        >>> row.cells[0].content
        'v'
    """

    def __init__(self, table: Table, index: int, cells: Sequence[Any]) -> None:
        self._table = table
        self.index = index
        self.cells: list[Cell] = [
            cell if isinstance(cell, Cell) else Cell(cell) for cell in cells
        ]
        self._format: dict[str, Any] = {}
        self._background: dict[str, Any] = {}
        self._height: float | None = None

    def set_format(self, **kwargs: Any) -> Self:
        """Apply format to every cell in this row.

        Examples:
            >>> table = Table(columns=["A"])
            >>> row = table.add_row("x")
            >>> _ = row.set_format(italic=True)
            >>> row.cells[0]._format["italic"]
            True
        """
        self._format.update(kwargs)
        for cell in self.cells:
            cell.set_format(**kwargs)
        return self

    def set_background(self, **kwargs: Any) -> Self:
        """Apply background to every cell in this row.

        Examples:
            >>> table = Table(columns=["A"])
            >>> row = table.add_row("x")
            >>> _ = row.set_background(fill=True)
            >>> row._background["fill"]
            True
        """
        self._background.update(kwargs)
        for cell in self.cells:
            cell.set_background(**kwargs)
        return self

    @property
    def b_box(self) -> BoundingBox:
        """Axis-aligned box enclosing this data row's cells.

        Examples:
            >>> table = Table(columns=["A", "B"])
            >>> _ = table.add_row(1, 2)
            >>> table.row(0).b_box.width > 0
            True
        """
        return self._table._row_bbox(self.index)


def _is_contiguous_indices(indices: list[int]) -> bool:
    if not indices:
        return True
    first, last = indices[0], indices[-1]
    return indices == list(range(first, last + 1))


def _slice_pick_indices(
    start: int,
    end: int,
    key: int | slice,
    *,
    base: list[int] | None = None,
) -> list[int]:
    """Map an index or slice onto a list of grid indices (supports step)."""
    if base is None:
        if start > end:
            raise IndexError("range index out of range")
        indices = list(range(start, end + 1))
    else:
        indices = base
    picked = indices[key]
    if isinstance(picked, int):
        return [picked]
    return list(picked)


def _explicit_indices(picked: list[int]) -> tuple[int, ...] | None:
    if not picked or _is_contiguous_indices(picked):
        return None
    return tuple(picked)


_RANGE_BACKGROUND_KEYS = frozenset(
    {
        "fill",
        "fill_alpha",
        "fill_color",
        "line_alpha",
        "line_color",
        "line_width",
        "stroke",
    }
)
_RANGE_FORMAT_KEYS = frozenset(
    {
        "align",
        "bold",
        "font_alpha",
        "font_color",
        "font_family",
        "font_size",
        "italic",
        "old_style_nums",
        "overline",
        "small_caps",
        "strike_through",
        "underline",
        "v_align",
    }
)
_RANGE_STYLE_ALIASES = {
    "crossed": "strike_through",
    "horiz_alignment": "align",
    "underlined": "underline",
    "vert_alignment": "v_align",
}
_TAG_FORMAT_ATTRS = frozenset(
    {
        "bold",
        "font_alpha",
        "font_color",
        "font_family",
        "font_size",
        "italic",
        "old_style_nums",
        "overline",
        "small_caps",
        "strike_through",
        "underline",
    }
)


def _range_style_target(name: str) -> tuple[str, str] | None:
    """Return ``("format"|"background", storage_name)`` or ``None``."""
    if name in _RANGE_STYLE_ALIASES:
        return "format", _RANGE_STYLE_ALIASES[name]
    if name in _RANGE_FORMAT_KEYS:
        return "format", name
    if name in _RANGE_BACKGROUND_KEYS:
        return "background", name
    return None


def _split_style_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split mixed style kwargs into format vs background stores."""
    format_kwargs: dict[str, Any] = {}
    background_kwargs: dict[str, Any] = {}
    for name, value in kwargs.items():
        target = _range_style_target(name)
        if target is None:
            format_kwargs[name] = value
            continue
        kind, storage_name = target
        if storage_name in ("align", "v_align"):
            value = _resolve_align(value)
        if kind == "background":
            background_kwargs[storage_name] = value
        else:
            format_kwargs[storage_name] = value
    return format_kwargs, background_kwargs


class Range:
    """Rectangular grid region (inclusive bounds, 0-based grid indices).

    **Grid coordinates:** ``row`` and ``col`` are integers starting at ``0``.
    Column ``0`` is the leftmost column. Row numbering follows the table
    origin: top-down by default, bottom-up when ``table.from_bottom_left`` is
    True. The header row is included only when explicit headers are shown. This
    matches :attr:`Table.cells`, :meth:`Table.range`, and :meth:`Table.cell`.

    Single cell: ``table.range(r, c, r, c).cell`` or ``table.cells[r, c]``.

    ``table.columns[i]`` / ``table.columns[i:j]`` slice **column indices**
    ``0``, ``1``, … ``table.rows[i]`` uses the same row index as
    ``table.cells``; ``table.rows[0]`` is the first grid row.

    ``table.cells[row, col]`` is NumPy-style (grid row first). A lone index or
    slice on ``table.cells`` selects grid rows and keeps all columns.

    Assign format or background attributes on a range to apply them to every
    cell in the range (column/row ranges also store the value on the
    ``Column`` / ``Row`` so later rows inherit it).

    Examples:
        >>> table = Table(columns=["A", "B"])
        >>> _ = table.add_row(1, 2)
        >>> table.cells.font_size = 12
        >>> table.cell(1, 0).content
        1
        >>> table.column(0).header
        'A'
"""

    def __init__(
        self,
        table: Table,
        row_start: int,
        col_start: int,
        row_end: int,
        col_end: int,
        *,
        size_axis: Literal["columns", "rows"] | None = None,
        explicit_grid_rows: tuple[int, ...] | None = None,
        explicit_grid_cols: tuple[int, ...] | None = None,
    ) -> None:
        if row_start < 0 or row_end < 0 or col_start < 0 or col_end < 0:
            object.__setattr__(self, "_table", table)
            object.__setattr__(self, "_row_start", row_start)
            object.__setattr__(self, "_row_end", row_end)
            object.__setattr__(self, "_col_start", col_start)
            object.__setattr__(self, "_col_end", col_end)
            object.__setattr__(self, "_size_axis", size_axis)
            object.__setattr__(self, "_explicit_grid_rows", None)
            object.__setattr__(self, "_explicit_grid_cols", None)
            return
        object.__setattr__(self, "_table", table)
        object.__setattr__(self, "_row_start", min(row_start, row_end))
        object.__setattr__(self, "_row_end", max(row_start, row_end))
        object.__setattr__(self, "_col_start", min(col_start, col_end))
        object.__setattr__(self, "_col_end", max(col_start, col_end))
        object.__setattr__(self, "_size_axis", size_axis)
        object.__setattr__(self, "_explicit_grid_rows", explicit_grid_rows)
        object.__setattr__(self, "_explicit_grid_cols", explicit_grid_cols)

    def _empty_subrange(self) -> Range:
        return Range(
            self._table,
            0,
            0,
            -1,
            -1,
            size_axis=self._size_axis,
        )

    def __getitem__(
        self,
        key: int | slice | tuple[int | slice, int | slice],
    ) -> Range:
        if self._size_axis == "columns":
            if not isinstance(key, (int, slice)):
                raise TypeError("columns range indices must be an int or slice")
            col_base = (
                list(self._explicit_grid_cols)
                if self._explicit_grid_cols is not None
                else None
            )
            picked_cols = _slice_pick_indices(
                self._col_start,
                self._col_end,
                key,
                base=col_base,
            )
            if not picked_cols:
                return self._empty_subrange()
            return Range(
                self._table,
                self._row_start,
                picked_cols[0],
                self._row_end,
                picked_cols[-1],
                size_axis="columns",
                explicit_grid_rows=self._explicit_grid_rows,
                explicit_grid_cols=_explicit_indices(picked_cols),
            )
        if self._size_axis == "rows":
            if not isinstance(key, (int, slice)):
                raise TypeError("rows range indices must be an int or slice")
            row_base = (
                list(self._explicit_grid_rows)
                if self._explicit_grid_rows is not None
                else None
            )
            picked_rows = _slice_pick_indices(
                self._row_start,
                self._row_end,
                key,
                base=row_base,
            )
            if not picked_rows:
                return self._empty_subrange()
            return Range(
                self._table,
                picked_rows[0],
                self._col_start,
                picked_rows[-1],
                self._col_end,
                size_axis="rows",
                explicit_grid_rows=_explicit_indices(picked_rows),
                explicit_grid_cols=self._explicit_grid_cols,
            )
        if isinstance(key, (int, slice)):
            row_key: int | slice = key
            col_key: int | slice = slice(None)
        elif isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
        else:
            raise TypeError(
                "cell range indices must be a row index/slice or a "
                "(row, column) pair of ints or slices"
            )
        if not isinstance(row_key, (int, slice)) or not isinstance(
            col_key, (int, slice)
        ):
            raise TypeError(
                "cell range indices must be a row index/slice or a "
                "(row, column) pair of ints or slices"
            )
        row_base = (
            list(self._explicit_grid_rows)
            if self._explicit_grid_rows is not None
            else None
        )
        col_base = (
            list(self._explicit_grid_cols)
            if self._explicit_grid_cols is not None
            else None
        )
        picked_rows = _slice_pick_indices(
            self._row_start,
            self._row_end,
            row_key,
            base=row_base,
        )
        picked_cols = _slice_pick_indices(
            self._col_start,
            self._col_end,
            col_key,
            base=col_base,
        )
        if not picked_rows or not picked_cols:
            return Range(self._table, 0, 0, -1, -1)
        return Range(
            self._table,
            picked_rows[0],
            picked_cols[0],
            picked_rows[-1],
            picked_cols[-1],
            explicit_grid_rows=_explicit_indices(picked_rows),
            explicit_grid_cols=_explicit_indices(picked_cols),
        )

    def _is_empty(self) -> bool:
        return (
            self._row_end < self._row_start or self._col_end < self._col_start
        )

    def _grid_row_indices(self) -> list[int]:
        if self._is_empty():
            return []
        if self._explicit_grid_rows is not None:
            return list(self._explicit_grid_rows)
        return list(range(self._row_start, self._row_end + 1))

    def _grid_col_indices(self) -> list[int]:
        if self._is_empty():
            return []
        if self._explicit_grid_cols is not None:
            return list(self._explicit_grid_cols)
        return list(range(self._col_start, self._col_end + 1))

    def __getattr__(self, name: str) -> Any:
        if name == "width":
            return self._common_column_width()
        if name == "height":
            return self._common_row_height()
        if name == "size":
            return (self.width, self.height)
        target = _range_style_target(name)
        if target is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        _kind, storage_name = target
        return self._style_value(storage_name, _kind)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name == "content":
            self.cell.content = value
            return
        if name == "width":
            self._assign_width(value)
            return
        if name == "height":
            self._assign_height(value)
            return
        if name == "size":
            width, height = value[:2]
            self._assign_width(width)
            self._assign_height(height)
            return
        target = _range_style_target(name)
        if target is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        kind, storage_name = target
        if storage_name in ("align", "v_align"):
            value = _resolve_align(value)
        self._apply_style(kind, storage_name, value)

    def _style_store(self, cell: Cell, kind: str) -> dict[str, Any]:
        if kind == "format":
            return cell._format
        return cell._background

    def _style_value(self, storage_name: str, kind: str) -> Any:
        first = None
        saw_value = False
        for cell in self:
            store = self._style_store(cell, kind)
            if storage_name not in store:
                return None
            current = store[storage_name]
            if not saw_value:
                first = current
                saw_value = True
            elif current != first:
                return None
        if not saw_value:
            return None
        return first

    def _apply_style(self, kind: str, storage_name: str, value: Any) -> None:
        if self._is_empty():
            raise ValueError("assignment on an empty range")
        kwargs = {storage_name: value}
        if self._size_axis == "columns":
            for col_index in self._grid_col_indices():
                column = self._table._columns[col_index]
                if kind == "format":
                    column.set_format(**kwargs)
                else:
                    column.set_background(**kwargs)
            return
        if self._size_axis == "rows":
            for grid_row in self._grid_row_indices():
                row = self._table._row_object_for_grid(grid_row)
                if kind == "format":
                    row.set_format(**kwargs)
                else:
                    row.set_background(**kwargs)
            return
        if kind == "format":
            self.set_format(**kwargs)
        else:
            self.set_background(**kwargs)

    @property
    def cell(self) -> Cell:
        """The sole cell when this range is one grid cell.

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row("v")
            >>> table.range(1, 0, 1, 0).cell.content
            'v'
        """
        if self._row_start != self._row_end or self._col_start != self._col_end:
            raise ValueError("cell is only defined for a single-cell range")
        return self._table._cell_at_grid(self._row_start, self._col_start)

    @property
    def content(self) -> Any:
        """Content of a single-cell range.

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row(7)
            >>> table.range(1, 0, 1, 0).content
            7
        """
        return self.cell.content

    def __iter__(self) -> Iterator[Cell]:
        for grid_row in self._grid_row_indices():
            for col_index in self._grid_col_indices():
                yield self._table._cell_at_grid(grid_row, col_index)

    @property
    def cells(self) -> list[list[Cell]]:
        """Cell matrix for this range (rows, then columns).

        Examples:
            >>> table = Table(columns=["A", "B"])
            >>> _ = table.add_row(1, 2)
            >>> len(table.cells.cells)
            2
        """
        matrix: list[list[Cell]] = []
        for grid_row in self._grid_row_indices():
            row_cells: list[Cell] = []
            for col_index in self._grid_col_indices():
                row_cells.append(  # noqa: PERF401
                    self._table._cell_at_grid(grid_row, col_index)
                )
            matrix.append(row_cells)
        return matrix

    def set_format(self, **kwargs: Any) -> Self:
        """Apply format to every cell in this range.

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row("x")
            >>> _ = table.rows.set_format(bold=True)
            >>> table.cell(1, 0)._format["bold"]
            True
        """
        for cell in self:
            cell.set_format(**kwargs)
        return self

    def set_background(self, **kwargs: Any) -> Self:
        """Apply background to every cell in this range.

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row("x")
            >>> _ = table.rows.set_background(fill=True)
            >>> table.cell(1, 0)._background["fill"]
            True
        """
        for cell in self:
            cell.set_background(**kwargs)
        return self

    def set_borders(
        self,
        kind: Border | str,
        *,
        line_color: Color | None = None,
        line_style: BorderLine | str | None = None,
    ) -> Self:
        """Apply an Excel border command to every cell in this range.

        ``line_style`` is the weight for edges the command does not already
        name as thick or double. ``NONE`` and ``NO_BORDER`` clear every
        edge on those cells. Draw and erase are this method and ``NO_BORDER``.

        Args:
            kind: Border command.
            line_color: Border color. ``None`` uses black.
            line_style: ``THIN``, ``THICK``, or ``DOUBLE``. ``None`` is thin.

        Returns:
            Self: This range.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(size=(2, 2), show_header=False)
            >>> block = table.range(0, 0, 1, 1)
            >>> _ = block.set_borders(
            ...     sg.Border.OUTSIDE,
            ...     line_color=sg.black,
            ...     line_style=sg.BorderLine.THIN,
            ... )
            >>> "top" in table.cell(0, 0)._borders
            True
            >>> "right" in table.cell(0, 0)._borders
            False
            >>> "right" in table.cell(0, 1)._borders
            True
        """
        if self._table is not None and self._is_empty():
            return self
        color = black if line_color is None else line_color
        _apply_border_kind(
            self._table,
            [
                (row_index, col_index)
                for row_index in self._grid_row_indices()
                for col_index in self._grid_col_indices()
            ],
            _as_border(kind),
            _as_border_line(line_style),
            color,
        )
        self._table._invalidate_layout()
        return self

    def _assign_width(self, width: float | Sequence[float]) -> None:
        if self._is_empty():
            raise ValueError("width assignment on an empty range")
        self._table._apply_column_widths(
            self._col_start, self._col_end, width
        )

    def _assign_height(self, height: float | Sequence[float]) -> None:
        if self._is_empty():
            raise ValueError("height assignment on an empty range")
        grid_rows = self._grid_row_indices()
        if (
            self._explicit_grid_rows is None
            and len(grid_rows) == self._row_end - self._row_start + 1
        ):
            self._table._apply_grid_row_heights(
                self._row_start, self._row_end, height
            )
            return
        for grid_row, row_height in zip(
            grid_rows,
            self._table._sizes_for_span(len(grid_rows), height, kind="row"),
            strict=True,
        ):
            self._table._rows[grid_row]._height = row_height
        self._table._invalidate_layout()

    def _common_column_width(self) -> float | None:
        first = None
        saw_value = False
        for col_index in self._grid_col_indices():
            current = self._table._columns[col_index]._width
            if not saw_value:
                first = current
                saw_value = True
            elif current != first:
                return None
        if not saw_value:
            return None
        return first

    def _stored_grid_row_height(self, grid_row: int) -> float | None:
        return self._table._rows[grid_row]._height

    def _common_row_height(self) -> float | None:
        first = None
        saw_value = False
        for grid_row in self._grid_row_indices():
            current = self._stored_grid_row_height(grid_row)
            if not saw_value:
                first = current
                saw_value = True
            elif current != first:
                return None
        if not saw_value:
            return None
        return first

    @property
    def b_box(self) -> BoundingBox:
        """Axis-aligned box enclosing this range.

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row("x")
            >>> table.cells.b_box.width > 0
            True
        """
        boxes = [
            table_cell_bbox(self._table, grid_row, col_index)
            for grid_row in self._grid_row_indices()
            for col_index in self._grid_col_indices()
        ]
        return _union_bounding_boxes(boxes)


def _rich_color(color: Color | str) -> str:
    """Return a color string Rich can parse.

    Examples:
        >>> _rich_color(white)
        'rgb(255, 255, 255)'
        >>> _rich_color("blue")
        'blue'
    """
    if isinstance(color, str):
        return color
    red = int(round(color.red * 255))
    green = int(round(color.green * 255))
    blue = int(round(color.blue * 255))
    return f"rgb({red}, {green}, {blue})"


def _resolve_cell_format(
    table: Table,
    grid_row: int,
    col_index: int,
) -> dict[str, Any]:
    row = table._rows[grid_row]
    layers = [
        _TABLE_DEFAULT_FORMAT,
        table._format,
    ]
    if col_index < len(table._columns):
        layers.append(table._columns[col_index]._format)
    layers.append(row._format)
    if col_index < len(row.cells):
        layers.append(row.cells[col_index]._format)
    return _merge_style(*layers)


def _apply_row_format_to_cell(row: Row, cell: Cell) -> None:
    """Copy ``row``'s format onto ``cell``. Every row uses this."""
    if row._format:
        cell.set_format(**row._format)
    if row._background:
        cell.set_background(**row._background)


def _default_header_format() -> dict[str, Any]:
    """Format applied to row 0 when ``show_header`` is True.

    ``header_row_fill`` turns the row fill on. The default fill color is
    black, so the text color is white when that fill is on.
    """
    from ..config.settings import runtime_defaults

    header_format: dict[str, Any] = {"bold": True}
    if runtime_defaults["header_row_fill"]:
        header_format["fill"] = True
        header_format["font_color"] = white
    return header_format


def _default_cell_background(table: Table) -> dict[str, Any]:
    from ..config.settings import runtime_defaults

    return {
        "fill": runtime_defaults["table_fill"],
        "stroke": _TABLE_DEFAULT_STROKE,
    }


def _resolve_cell_background(
    table: Table,
    grid_row: int,
    col_index: int,
) -> dict[str, Any]:
    row = table._rows[grid_row]
    layers = [
        _default_cell_background(table),
        table._background,
    ]
    if col_index < len(table._columns):
        layers.append(table._columns[col_index]._background)
    layers.append(row._background)
    if col_index < len(row.cells):
        layers.append(row.cells[col_index]._background)
    return _merge_style(*layers)


def _grid_dimensions(table: Table) -> tuple[int, int]:
    return len(table._columns), len(table._rows)


def _layout_content_cache_key(content: Any) -> tuple[Any, ...]:
    if isinstance(content, Cell):
        return (
            Cell,
            content._layout_revision,
            _layout_content_cache_key(content.content),
        )
    if isinstance(content, SplitCell):
        return (
            SplitCell,
            content.kind,
            content.first,
            content.second,
            tuple(content.offset1),
            tuple(content.offset2),
        )
    if _is_drawable_content(content):
        box = content.b_box
        return type(content), id(content), box.width, box.height
    if content is None or isinstance(content, (str, int, float, bool)):
        return type(content), content
    return type(content), str(content)


def _table_layout_cache_key(table: Table) -> tuple[Any, ...]:
    cell_keys: list[tuple[Any, ...]] = []
    for row in table._rows:
        cell_keys.extend(
            (
                cell._layout_revision,
                _layout_content_cache_key(cell.content),
            )
            for cell in row.cells
        )
    return table._layout_revision, tuple(cell_keys)


def _public_full_grid_rows(table: Table) -> list[int]:
    rows = list(range(len(table._rows)))
    if table.from_bottom_left:
        rows.reverse()
    return rows


def _storage_row_index(table: Table, row_index: int) -> int:
    if row_index < 0 or row_index >= len(table._rows):
        raise IndexError(f"row index {row_index} out of range")
    if table.from_bottom_left:
        return len(table._rows) - 1 - row_index
    return row_index


def _empty_grid_range(table: Table) -> Range:
    """Range that contains no cells (empty table or zero columns)."""
    return Range(table, 0, 0, -1, -1)


def _cell_content_size(
    content: Any, fmt: dict[str, Any]
) -> tuple[float, float]:
    if isinstance(content, Cell):
        content = content.content
    if isinstance(content, SplitCell):
        first_w, first_h = _cell_content_size(content.first, fmt)
        second_w, second_h = _cell_content_size(content.second, fmt)
        if content.kind == SplitCell.VERTICAL:
            return first_w + second_w, max(first_h, second_h)
        if content.kind == SplitCell.HORIZONTAL:
            return max(first_w, second_w), first_h + second_h
        return first_w + second_w, first_h + second_h
    if isinstance(content, (str, int, float, bool)):
        text = str(content)
        formula = _tag_latex_formula(text)
        if formula is not None:
            font_family = (
                fmt["font_family"] if "font_family" in fmt else None
            )
            bold = fmt["bold"] if "bold" in fmt else False
            return _measure_latex_formula(
                formula, fmt["font_size"], font_family, bold
            )
        tag = Tag(text, (0.0, 0.0), font_size=fmt["font_size"])
        box = tag.b_box
        return box.width, box.height
    if _is_drawable_content(content):
        box = content.b_box
        return box.width, box.height
    text = str(content)
    tag = Tag(text, (0.0, 0.0), font_size=fmt["font_size"])
    box = tag.b_box
    return box.width, box.height


def _table_padding_xy(table: Table) -> tuple[float, float]:
    padding = table.padding
    if type(padding) in (int, float):
        if padding < 0:
            raise ValueError("padding must be non-negative")
        value = float(padding)
        return value, value
    if not isinstance(padding, Sequence) or isinstance(padding, (str, bytes)):
        raise ValueError(
            "padding must be a non-negative number or a (horizontal, vertical) pair"
        )
    values = list(padding)
    if len(values) != 2:
        raise ValueError(
            "padding must be a non-negative number or a (horizontal, vertical) pair"
        )
    horizontal, vertical = values
    if (
        type(horizontal) not in (int, float)
        or type(vertical) not in (int, float)
        or horizontal < 0
        or vertical < 0
    ):
        raise ValueError(
            "padding must be a non-negative number or a (horizontal, vertical) pair"
        )
    return float(horizontal), float(vertical)


def compute_table_layout(
    table: Table,
) -> tuple[list[float], list[float], int]:
    """Return column widths, row heights, and column count for ``table``.

    Examples:
        >>> table = Table(columns=["A", "B"])
        >>> _ = table.add_row(1, 22)
        >>> widths, heights, n_cols = compute_table_layout(table)
        >>> n_cols
        2
        >>> len(widths)
        2
        >>> len(heights)
        2
        >>> square = Table(size=(2, 2), column_width=30, row_height=30, padding=0)
        >>> compute_table_layout(square)
        ([30, 30], [30, 30], 2)
    """
    cache_key = _table_layout_cache_key(table)
    if (
        table._layout_cache_key == cache_key
        and table._layout_cache is not None
    ):
        cached_widths, cached_heights, cached_column_count = (
            table._layout_cache
        )
        return (
            list(cached_widths),
            list(cached_heights),
            cached_column_count,
        )

    column_count, grid_row_count = _grid_dimensions(table)
    if column_count == 0:
        return [], [], 0

    col_widths = [0.0] * column_count
    row_heights = [0.0] * grid_row_count

    padding_x, padding_y = _table_padding_xy(table)
    pad_x = 2 * padding_x
    pad_y = 2 * padding_y

    for grid_row in range(grid_row_count):
        for col_index in range(column_count):
            fmt = _resolve_cell_format(table, grid_row, col_index)
            content = table._rows[grid_row].cells[col_index].content
            content_w, content_h = _cell_content_size(content, fmt)
            col_widths[col_index] = max(
                col_widths[col_index], content_w + pad_x
            )
            row_heights[grid_row] = max(
                row_heights[grid_row], content_h + pad_y
            )

    for col_index in range(column_count):
        width = table._columns[col_index]._width
        if width is not None:
            col_widths[col_index] = max(col_widths[col_index], width)

    for grid_row in range(grid_row_count):
        height = table._rows[grid_row]._height
        if height is not None:
            row_heights[grid_row] = max(row_heights[grid_row], height)

    if table.min_width is not None:
        total = sum(col_widths)
        if total < table.min_width and column_count:
            extra = (table.min_width - total) / column_count
            col_widths = [width + extra for width in col_widths]

    table._layout_cache_key = cache_key
    table._layout_cache = (
        tuple(col_widths),
        tuple(row_heights),
        column_count,
    )

    return col_widths, row_heights, column_count


def _bbox_from_cell_bounds(
    x_left: float,
    y_top: float,
    x_right: float,
    y_bottom: float,
) -> BoundingBox:
    southwest = (x_left, y_bottom)
    northeast = (x_right, y_top)
    return BoundingBox(southwest, northeast)


def _union_bounding_boxes(boxes: Sequence[BoundingBox]) -> BoundingBox:
    points: list[PointType] = []
    for box in boxes:
        if box.southwest is None:
            continue
        points.extend(box.corners)
    if not points:
        return BoundingBox()
    return bounding_box(points)


def _table_label_tag(value: Tag | str | None) -> Tag | None:
    """Return a title or caption tag.

    ``None`` and ``""`` mean no label. A string becomes a ``Tag`` with
    default tag values. A ``Tag`` is kept as given.

    Raises:
        TypeError: If ``value`` is not a ``Tag``, a string, or ``None``.
    """
    if value is None or value == "":
        return None
    if isinstance(value, Tag):
        return value
    if isinstance(value, str):
        return Tag(value, _TABLE_DEFAULT_POS)
    raise TypeError("title and caption must be a Tag, a string, or None")


def _label_half_height(tag: Tag) -> float:
    """Half the tag's bounding-box height."""
    box = tag.b_box
    ymin = box.southwest[1]
    ymax = box.northeast[1]
    return (ymax - ymin) / 2


def _move_tag_to(tag: Tag, x: float, y: float) -> None:
    """Move ``tag`` so its position is ``(x, y)`` (mutated)."""
    current_x, current_y = tag.pos[:2]
    tag.translate(x - current_x, y - current_y)


def _grid_origin(table: Table) -> tuple[float, float]:
    """Return the grid's northwest corner.

    When ``title`` is set, the tag is placed above that corner (mutated)
    and the grid moves down to leave ``_TABLE_DEFAULT_TITLE_PAD`` between
    the title and the grid.
    """
    x0, y0 = table.pos[:2]
    if table.title is None:
        return x0, y0
    half = _label_half_height(table.title)
    _move_tag_to(table.title, x0, y0 + _TABLE_DEFAULT_TITLE_PAD)
    return x0, y0 - half


def _place_caption(table: Table, row_heights: Sequence[float]) -> None:
    """Place ``caption`` below the grid (mutated)."""
    if table.caption is None:
        return
    x0, grid_top = _grid_origin(table)
    grid_bottom = grid_top - sum(row_heights)
    half = _label_half_height(table.caption)
    _move_tag_to(
        table.caption,
        x0,
        grid_bottom - _TABLE_DEFAULT_TITLE_PAD - half,
    )


def table_cell_bbox(
    table: Table,
    grid_row: int,
    col_index: int,
) -> BoundingBox:
    """Bounding box of one grid cell.

    Examples:
        >>> table = Table(columns=["A"])
        >>> _ = table.add_row("x")
        >>> box = table_cell_bbox(table, 1, 0)
        >>> box.width > 0
        True
    """
    col_widths, row_heights, column_count = compute_table_layout(table)
    if col_index >= column_count or grid_row >= len(row_heights):
        return BoundingBox()
    pos = _grid_origin(table)
    x_left, y_top, x_right, y_bottom = _cell_box(
        pos,
        col_widths,
        row_heights,
        grid_row,
        col_index,
    )
    return _bbox_from_cell_bounds(x_left, y_top, x_right, y_bottom)


def table_grid_bbox(table: Table) -> BoundingBox:
    """Bounding box of the full cell grid.

    Examples:
        >>> table = Table(columns=["A", "B"])
        >>> _ = table.add_row(1, 2)
        >>> table_grid_bbox(table).width > 0
        True
    """
    col_widths, row_heights, _ = compute_table_layout(table)
    if not col_widths or not row_heights:
        return BoundingBox()
    x0, y0 = _grid_origin(table)
    return _bbox_from_cell_bounds(
        x0,
        y0,
        x0 + sum(col_widths),
        y0 - sum(row_heights),
    )


def table_title_bbox(table: Table) -> BoundingBox | None:
    """Bounding box of the title tag, or ``None`` when ``title`` is ``None``.

    The title tag's position is updated.

    Examples:
        >>> table = Table(title="", columns=["A"])
        >>> table_title_bbox(table) is None
        True
        >>> table.title = "Summary"
        >>> table_title_bbox(table).width > 0
        True
    """
    if table.title is None:
        return None
    _grid_origin(table)
    return table.title.b_box


def _cell_box(
    pos: PointType,
    col_widths: list[float],
    row_heights: list[float],
    row_index: int,
    col_index: int,
) -> tuple[float, float, float, float]:
    x0, y0 = pos[:2]
    x_left = x0 + sum(col_widths[:col_index])
    x_right = x_left + col_widths[col_index]
    y_top = y0 - sum(row_heights[:row_index])
    y_bottom = y_top - row_heights[row_index]
    return x_left, y_top, x_right, y_bottom


def _cell_anchor_pos(
    x_left: float,
    y_top: float,
    x_right: float,
    y_bottom: float,
    h_align: Align,
    v_align: Align,
    padding_x: float,
    padding_y: float,
) -> PointType:
    if h_align in (Align.LEFT, Align.FLUSH_LEFT):
        x = x_left + padding_x
    elif h_align in (Align.RIGHT, Align.FLUSH_RIGHT):
        x = x_right - padding_x
    else:
        x = (x_left + x_right) / 2
    if v_align in (Align.TOP,):
        y = y_top - padding_y
    elif v_align in (Align.BOTTOM,):
        y = y_bottom + padding_y
    else:
        y = (y_top + y_bottom) / 2
    return (x, y)


def _background_rect_sketch(
    canvas: Any,
    x_left: float,
    y_top: float,
    x_right: float,
    y_bottom: float,
    background: dict[str, Any],
) -> list:
    fill = background["fill"]
    if not fill:
        return []
    rect = Shape(
        [
            (x_left, y_top),
            (x_right, y_top),
            (x_right, y_bottom),
            (x_left, y_bottom),
        ],
        closed=True,
    )
    sketch_kwargs: dict[str, Any] = {
        "fill": fill,
        "stroke": background["stroke"],
    }
    if "fill_color" in background:
        sketch_kwargs["fill_color"] = background["fill_color"]
    if "fill_alpha" in background:
        sketch_kwargs["fill_alpha"] = background["fill_alpha"]
    if "line_color" in background:
        sketch_kwargs["line_color"] = background["line_color"]
    if "line_alpha" in background:
        sketch_kwargs["line_alpha"] = background["line_alpha"]
    if "line_width" in background:
        sketch_kwargs["line_width"] = background["line_width"]
    return get_sketches(rect, canvas, **sketch_kwargs)


def _apply_tag_text_format(tag: Tag, fmt: dict[str, Any]) -> None:
    """Copy cell text-format keys onto ``tag``."""
    for name in _TAG_FORMAT_ATTRS:
        if name in fmt:
            setattr(tag, name, fmt[name])


def _tag_for_split_label(
    text: str,
    pos: PointType,
    fmt: dict[str, Any],
    align: Align,
    anchor: Anchor,
) -> Tag:
    tag = Tag(
        text,
        pos,
        font_size=fmt["font_size"],
        align=align,
    )
    _apply_tag_text_format(tag, fmt)
    tag.anchor = anchor
    return tag


def _split_cell_sketches(
    content: SplitCell,
    canvas: Any,
    x_left: float,
    y_top: float,
    x_right: float,
    y_bottom: float,
    fmt: dict[str, Any],
    padding_x: float,
    padding_y: float,
    **kwargs: Any,
) -> list:
    x_mid = (x_left + x_right) / 2
    y_mid = (y_top + y_bottom) / 2
    if content.kind == SplitCell.VERTICAL:
        divider = Shape([(x_mid, y_top), (x_mid, y_bottom)])
        first_pos = (x_left + padding_x, y_mid)
        second_pos = (x_right - padding_x, y_mid)
        first_align = Align.LEFT
        first_anchor = Anchor.WEST
        second_align = Align.RIGHT
        second_anchor = Anchor.EAST
    elif content.kind == SplitCell.HORIZONTAL:
        divider = Shape([(x_left, y_mid), (x_right, y_mid)])
        first_pos = (x_mid, y_top - padding_y)
        second_pos = (x_mid, y_bottom + padding_y)
        first_align = Align.CENTER
        first_anchor = Anchor.NORTH
        second_align = Align.CENTER
        second_anchor = Anchor.SOUTH
    elif content.kind == SplitCell.DIAGONAL_DOWN:
        divider = Shape([(x_left, y_top), (x_right, y_bottom)])
        first_pos = (x_left + padding_x, y_bottom + padding_y)
        second_pos = (x_right - padding_x, y_top - padding_y)
        first_align = Align.LEFT
        first_anchor = Anchor.SOUTHWEST
        second_align = Align.RIGHT
        second_anchor = Anchor.NORTHEAST
    else:
        divider = Shape([(x_left, y_bottom), (x_right, y_top)])
        first_pos = (x_left + padding_x, y_top - padding_y)
        second_pos = (x_right - padding_x, y_bottom + padding_y)
        first_align = Align.LEFT
        first_anchor = Anchor.NORTHWEST
        second_align = Align.RIGHT
        second_anchor = Anchor.SOUTHEAST
    first_dx, first_dy = content.offset1[:2]
    second_dx, second_dy = content.offset2[:2]
    first_pos = (first_pos[0] + first_dx, first_pos[1] + first_dy)
    second_pos = (second_pos[0] + second_dx, second_pos[1] + second_dy)
    first_tag = _tag_for_split_label(
        content.first, first_pos, fmt, first_align, first_anchor
    )
    second_tag = _tag_for_split_label(
        content.second, second_pos, fmt, second_align, second_anchor
    )
    sketches = get_sketches(
        divider,
        canvas,
        stroke=True,
        fill=False,
        line_width=_TABLE_DEFAULT_GRID_LINE_WIDTH,
        line_color=gray,
    )
    sketches.extend(get_sketches(first_tag, canvas, **kwargs))
    sketches.extend(get_sketches(second_tag, canvas, **kwargs))
    return sketches


def _tag_for_content(
    content: Any,
    pos: PointType,
    fmt: dict[str, Any],
) -> Tag:
    h_align = _resolve_align(fmt["align"])
    tag = Tag(
        _cell_display_text(content),
        pos,
        font_size=fmt["font_size"],
        align=h_align,
    )
    _apply_tag_text_format(tag, fmt)
    if h_align in (Align.LEFT, Align.FLUSH_LEFT):
        tag.anchor = Anchor.WEST
    elif h_align in (Align.RIGHT, Align.FLUSH_RIGHT):
        tag.anchor = Anchor.EAST
    else:
        tag.anchor = Anchor.CENTER
    return tag


def _as_border(kind: Border | str) -> Border:
    if isinstance(kind, Border):
        return kind
    return Border[kind.upper()]


def _as_border_line(style: BorderLine | str | None) -> BorderLine | None:
    if style is None:
        return None
    if isinstance(style, BorderLine):
        return style
    return BorderLine[style.upper()]


def _border_default_style(line_style: BorderLine | None) -> BorderLine:
    if line_style is None:
        return BorderLine.THIN
    return line_style


def _set_border_edge(
    cell: Cell,
    edge: str,
    style: BorderLine,
    line_color: Color,
) -> None:
    cell._borders[edge] = {"color": line_color, "style": style}


def _apply_border_kind(
    table: Table,
    positions: Sequence[tuple[int, int]],
    kind: Border,
    line_style: BorderLine | None,
    line_color: Color,
) -> None:
    """Write ``kind`` onto the cells at storage ``positions``."""
    selected = set(positions)
    default_style = _border_default_style(line_style)

    def neighbor(
        row_index: int,
        col_index: int,
        row_step: int,
        col_step: int,
    ) -> bool:
        return (row_index + row_step, col_index + col_step) in selected

    for row_index, col_index in positions:
        cell = table._rows[row_index].cells[col_index]
        if kind in (Border.NONE, Border.NO_BORDER):
            cell._borders = {}
            continue
        if kind in (Border.OUTSIDE, Border.OUTLINE, Border.THICK_OUTSIDE):
            style = (
                BorderLine.THICK
                if kind is Border.THICK_OUTSIDE
                else default_style
            )
            if not neighbor(row_index, col_index, -1, 0):
                _set_border_edge(cell, _BORDER_EDGE_TOP, style, line_color)
            if not neighbor(row_index, col_index, 1, 0):
                _set_border_edge(cell, _BORDER_EDGE_BOTTOM, style, line_color)
            if not neighbor(row_index, col_index, 0, -1):
                _set_border_edge(cell, _BORDER_EDGE_LEFT, style, line_color)
            if not neighbor(row_index, col_index, 0, 1):
                _set_border_edge(cell, _BORDER_EDGE_RIGHT, style, line_color)
            continue
        if kind is Border.INSIDE:
            if neighbor(row_index, col_index, -1, 0):
                _set_border_edge(
                    cell, _BORDER_EDGE_TOP, default_style, line_color
                )
            if neighbor(row_index, col_index, 1, 0):
                _set_border_edge(
                    cell, _BORDER_EDGE_BOTTOM, default_style, line_color
                )
            if neighbor(row_index, col_index, 0, -1):
                _set_border_edge(
                    cell, _BORDER_EDGE_LEFT, default_style, line_color
                )
            if neighbor(row_index, col_index, 0, 1):
                _set_border_edge(
                    cell, _BORDER_EDGE_RIGHT, default_style, line_color
                )
            continue
        if kind is Border.INSIDE_VERTICAL:
            if neighbor(row_index, col_index, 0, -1):
                _set_border_edge(
                    cell, _BORDER_EDGE_LEFT, default_style, line_color
                )
            if neighbor(row_index, col_index, 0, 1):
                _set_border_edge(
                    cell, _BORDER_EDGE_RIGHT, default_style, line_color
                )
            continue
        if kind is Border.INSIDE_HORIZONTAL:
            if neighbor(row_index, col_index, -1, 0):
                _set_border_edge(
                    cell, _BORDER_EDGE_TOP, default_style, line_color
                )
            if neighbor(row_index, col_index, 1, 0):
                _set_border_edge(
                    cell, _BORDER_EDGE_BOTTOM, default_style, line_color
                )
            continue
        if kind is Border.ALL:
            for edge in (
                _BORDER_EDGE_TOP,
                _BORDER_EDGE_BOTTOM,
                _BORDER_EDGE_LEFT,
                _BORDER_EDGE_RIGHT,
            ):
                _set_border_edge(cell, edge, default_style, line_color)
            continue
        if kind is Border.TOP:
            _set_border_edge(cell, _BORDER_EDGE_TOP, default_style, line_color)
        elif kind is Border.BOTTOM:
            _set_border_edge(
                cell, _BORDER_EDGE_BOTTOM, default_style, line_color
            )
        elif kind is Border.LEFT:
            _set_border_edge(cell, _BORDER_EDGE_LEFT, default_style, line_color)
        elif kind is Border.RIGHT:
            _set_border_edge(
                cell, _BORDER_EDGE_RIGHT, default_style, line_color
            )
        elif kind is Border.BOTTOM_DOUBLE:
            _set_border_edge(
                cell, _BORDER_EDGE_BOTTOM, BorderLine.DOUBLE, line_color
            )
        elif kind is Border.THICK_BOTTOM:
            _set_border_edge(
                cell, _BORDER_EDGE_BOTTOM, BorderLine.THICK, line_color
            )
        elif kind is Border.TOP_AND_BOTTOM:
            _set_border_edge(cell, _BORDER_EDGE_TOP, default_style, line_color)
            _set_border_edge(
                cell, _BORDER_EDGE_BOTTOM, default_style, line_color
            )
        elif kind is Border.TOP_AND_THICK_BOTTOM:
            _set_border_edge(cell, _BORDER_EDGE_TOP, default_style, line_color)
            _set_border_edge(
                cell, _BORDER_EDGE_BOTTOM, BorderLine.THICK, line_color
            )
        elif kind is Border.TOP_AND_DOUBLE_BOTTOM:
            _set_border_edge(cell, _BORDER_EDGE_TOP, default_style, line_color)
            _set_border_edge(
                cell, _BORDER_EDGE_BOTTOM, BorderLine.DOUBLE, line_color
            )
        elif kind is Border.DIAGONAL_DOWN:
            _set_border_edge(
                cell, _BORDER_EDGE_DIAGONAL_DOWN, default_style, line_color
            )
        elif kind is Border.DIAGONAL_UP:
            _set_border_edge(
                cell, _BORDER_EDGE_DIAGONAL_UP, default_style, line_color
            )
        else:
            raise ValueError(f"unhandled border {kind!r}")


def _offset_segment(
    start: PointType,
    end: PointType,
    distance: float,
) -> tuple[PointType, PointType] | None:
    x1, y1 = start[:2]
    x2, y2 = end[:2]
    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return None
    normal_x = -dy / length
    normal_y = dx / length
    return (
        (x1 + normal_x * distance, y1 + normal_y * distance),
        (x2 + normal_x * distance, y2 + normal_y * distance),
    )


def _border_line_sketches(
    canvas: Any,
    start: PointType,
    end: PointType,
    style: BorderLine,
    line_color: Color,
) -> list:
    if style is BorderLine.THICK:
        line_width = _TABLE_BORDER_THICK
    else:
        line_width = _TABLE_BORDER_THIN
    segments = [(start, end)]
    if style is BorderLine.DOUBLE:
        gap = _TABLE_BORDER_DOUBLE_GAP / 2
        above = _offset_segment(start, end, gap)
        below = _offset_segment(start, end, -gap)
        if above is None or below is None:
            return []
        segments = [above, below]
    sketches: list = []
    for segment_start, segment_end in segments:
        line = Shape([segment_start, segment_end])
        sketches.extend(
            get_sketches(
                line,
                canvas,
                stroke=True,
                fill=False,
                line_width=line_width,
                line_color=line_color,
            )
        )
    return sketches


def _border_sketches(
    table: Table,
    canvas: Any,
    pos: PointType,
    col_widths: list[float],
    row_heights: list[float],
) -> list:
    sketches: list = []
    column_count = len(col_widths)
    for grid_row in range(len(row_heights)):
        if grid_row >= len(table._rows):
            continue
        row = table._rows[grid_row]
        for col_index in range(column_count):
            if col_index >= len(row.cells):
                continue
            borders = row.cells[col_index]._borders
            if not borders:
                continue
            x_left, y_top, x_right, y_bottom = _cell_box(
                pos, col_widths, row_heights, grid_row, col_index
            )
            edge_segments = {
                _BORDER_EDGE_TOP: ((x_left, y_top), (x_right, y_top)),
                _BORDER_EDGE_BOTTOM: ((x_left, y_bottom), (x_right, y_bottom)),
                _BORDER_EDGE_LEFT: ((x_left, y_top), (x_left, y_bottom)),
                _BORDER_EDGE_RIGHT: ((x_right, y_top), (x_right, y_bottom)),
                _BORDER_EDGE_DIAGONAL_DOWN: (
                    (x_left, y_top),
                    (x_right, y_bottom),
                ),
                _BORDER_EDGE_DIAGONAL_UP: (
                    (x_left, y_bottom),
                    (x_right, y_top),
                ),
            }
            for edge, (segment_start, segment_end) in edge_segments.items():
                if edge not in borders:
                    continue
                record = borders[edge]
                sketches.extend(
                    _border_line_sketches(
                        canvas,
                        segment_start,
                        segment_end,
                        record["style"],
                        record["color"],
                    )
                )
    return sketches


def _grid_line_sketches(
    table: Table,
    canvas: Any,
    pos: PointType,
    col_widths: list[float],
    row_heights: list[float],
) -> list:
    sketches: list = []
    x0, y0 = pos[:2]
    total_w = sum(col_widths)
    y = y0
    for row_index in range(len(row_heights) + 1):
        segment = Shape([(x0, y), (x0 + total_w, y)])
        sketches.extend(
            get_sketches(
                segment,
                canvas,
                stroke=True,
                fill=False,
                line_width=_TABLE_DEFAULT_GRID_LINE_WIDTH,
                line_color=gray,
            )
        )
        if row_index < len(row_heights):
            y -= row_heights[row_index]
    x = x0
    y_bottom = y0 - sum(row_heights)
    for col_index in range(len(col_widths) + 1):
        segment = Shape([(x, y0), (x, y_bottom)])
        sketches.extend(
            get_sketches(
                segment,
                canvas,
                stroke=True,
                fill=False,
                line_width=_TABLE_DEFAULT_GRID_LINE_WIDTH,
                line_color=gray,
            )
        )
        if col_index < len(col_widths):
            x += col_widths[col_index]
    return sketches


def _flatten_sketches(items: Sequence) -> list:
    """Flatten nested lists from ``get_sketches`` of groups."""
    flattened: list = []
    for item in items:
        if isinstance(item, list):
            flattened.extend(_flatten_sketches(item))
        else:
            flattened.append(item)
    return flattened


def build_table_sketch(table: Table, canvas: Any, **kwargs: Any) -> TableSketch:
    """Lay out ``table`` and return a ``TableSketch`` for SVG/PDF export.

    Examples:
        >>> table = Table(columns=["A"])
        >>> _ = table.add_row("x")
        >>> build_table_sketch.__name__
        'build_table_sketch'
    """
    col_widths, row_heights, column_count = compute_table_layout(table)
    pos = _grid_origin(table)
    sketches: list = []
    grid_row_count = len(row_heights)
    padding_x, padding_y = _table_padding_xy(table)

    if table.title is not None:
        sketches.extend(get_sketches(table.title, canvas, **kwargs))

    for grid_row in range(grid_row_count):
        for col_index in range(column_count):
            background = _resolve_cell_background(table, grid_row, col_index)
            x_left, y_top, x_right, y_bottom = _cell_box(
                pos, col_widths, row_heights, grid_row, col_index
            )
            sketches.extend(
                _background_rect_sketch(
                    canvas, x_left, y_top, x_right, y_bottom, background
                )
            )

    if table.show_lines and column_count and row_heights:
        sketches.extend(
            _grid_line_sketches(table, canvas, pos, col_widths, row_heights)
        )
    if column_count and row_heights:
        sketches.extend(
            _border_sketches(table, canvas, pos, col_widths, row_heights)
        )

    for grid_row in range(grid_row_count):
        for col_index in range(column_count):
            fmt = _resolve_cell_format(table, grid_row, col_index)
            h_align = _resolve_align(fmt["align"])
            v_align = _resolve_align(
                fmt.get("v_align", _TABLE_DEFAULT_ROW_ALIGN)
            )
            x_left, y_top, x_right, y_bottom = _cell_box(
                pos, col_widths, row_heights, grid_row, col_index
            )
            cell_pos = _cell_anchor_pos(
                x_left,
                y_top,
                x_right,
                y_bottom,
                h_align,
                v_align,
                padding_x,
                padding_y,
            )
            content = table._rows[grid_row].cells[col_index].content
            if isinstance(content, Cell):
                content = content.content
            if isinstance(content, SplitCell):
                sketches.extend(
                    _split_cell_sketches(
                        content,
                        canvas,
                        x_left,
                        y_top,
                        x_right,
                        y_bottom,
                        fmt,
                        padding_x,
                        padding_y,
                        **kwargs,
                    )
                )
            elif _is_drawable_content(content):
                midpoint_x, midpoint_y = content.midpoint[:2]
                dest_x, dest_y = cell_pos[:2]
                shift_x = dest_x - midpoint_x
                shift_y = dest_y - midpoint_y
                base_sketch_xform = canvas._sketch_xform_matrix
                canvas._sketch_xform_matrix = (
                    translation_matrix(shift_x, shift_y) @ base_sketch_xform
                )
                try:
                    cell_sketches = get_sketches(content, canvas, **kwargs)
                finally:
                    canvas._sketch_xform_matrix = base_sketch_xform

                sketches.extend(_flatten_sketches(cell_sketches))
            else:
                tag = _tag_for_content(content, cell_pos, fmt)
                sketches.extend(get_sketches(tag, canvas, **kwargs))

    _place_caption(table, row_heights)
    if table.caption is not None:
        sketches.extend(get_sketches(table.caption, canvas, **kwargs))

    return TableSketch(
        pos=pos,
        column_widths=col_widths,
        row_heights=row_heights,
        sketches=sketches,
        show_lines=table.show_lines,
        xform_matrix=canvas._sketch_xform_matrix,
    )


class Table:
    """Documentation table with Rich notebook and vector export.

    **Cell indexing:** 0-based column indices ``0``, ``1``, … everywhere.
    :attr:`cells`, :meth:`range`, :meth:`cell`, :meth:`row`, :meth:`populate`,
    and ``rows`` share one row index. Row ``0`` is the first grid row (the
    header, when a header is shown). Header strings in ``columns=["A", "B"]``
    are the contents of that first row — not column coordinates.

    Examples:
        >>> table = Table(
        ...     title="Parameters",
        ...     columns=["Name", "Value"],
        ...     size=(2, 2),
        ...     column_align=[Align.LEFT, Align.RIGHT],
        ... )
        >>> table.cell(1, 0).content = "n_sides"
        >>> table.cell(1, 1).content = 7
        >>> table.cell(2, 0).content = "skip"
        >>> table.cell(2, 1).content = 3
        >>> table.build_rich_table()  # doctest: +ELLIPSIS
        <rich.table.Table object at ...>
"""

    def __init__(
        self,
        title: Tag | str | None = _TABLE_DEFAULT_TITLE,
        columns: Sequence[str] | None = None,
        *,
        caption: Tag | str | None = _TABLE_DEFAULT_CAPTION,
        size: tuple[int, int] | None = None,
        from_bottom_left: bool = False,
        pos: PointType = _TABLE_DEFAULT_POS,
        show_header: bool = _TABLE_DEFAULT_SHOW_HEADER,
        show_lines: bool = _TABLE_DEFAULT_SHOW_LINES,
        column_align: Sequence[Align | str] | None = None,
        column_width: float | None = None,
        min_width: int | None = _TABLE_DEFAULT_MIN_WIDTH,
        padding: tuple[int, int] = _TABLE_DEFAULT_PADDING,
        font_size: float = _TABLE_DEFAULT_FONT_SIZE,
        row_height: float | None = None,
    ) -> None:
        """Create an empty table with optional column headers and size.

        Args:
            title: Title above the grid. A string becomes a ``Tag`` with
                default tag values. ``None`` draws no title. Drawing and
                ``b_box`` set the tag position (mutated).
            columns: Optional column header labels.
            caption: Caption below the grid. A string becomes a ``Tag``
                with default tag values. ``None`` draws no caption.
                Drawing and ``b_box`` set the tag position (mutated).
            size: Optional ``(rows, columns)`` body size to pre-create.
            from_bottom_left: Whether row indexing starts at the bottom.
            pos: Northwest table position.
            show_header: Whether to show the header row. When True, row 0
                receives the default header format.
            show_lines: Whether to stroke grid lines.
            column_align: Optional per-column alignment list.
            column_width: Default width applied to created columns.
            min_width: Minimum total table width after layout.
            padding: Cell padding for vector layout and Rich preview.
            font_size: Base font size.
            row_height: Default height applied to created rows.

        Raises:
            ValueError: If ``size`` is not two non-negative integers, or if
                ``columns`` has more labels than ``size[1]``.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(size=(2, 3))
            >>> len(table._rows), len(table._columns)
            (2, 3)
            >>> bottom = sg.Table(size=(2, 1), from_bottom_left=True)
            >>> bottom.cell(0, 0).content = "south"
            >>> bottom.cell(1, 0).content = "north"
            >>> bottom.row(0).cells[0].content
            'south'
            >>> no_headers = sg.Table(size=(1, 1))
            >>> len(no_headers.cells.cells)
            1
            >>> sized = sg.Table(columns=["A", "B"], size=(1, 3))
            >>> sized.column_headers
            ['A', 'B', '']
            >>> header = sg.Table(columns=["A"], size=(1, 1))
            >>> header.cell(0, 0)._format["bold"]
            True
            >>> header.cell(0, 0)._format["font_color"] == sg.white
            True
            >>> header.cell(1, 0)._format
            {}
            >>> plain = sg.Table(columns=["A"], size=(1, 1), show_header=False)
            >>> plain.cell(0, 0)._format
            {}
            >>> unlabeled = sg.Table()
            >>> unlabeled.title is None
            True
            >>> unlabeled.caption is None
            True
            >>> unlabeled.title = "Summary"
            >>> isinstance(unlabeled.title, sg.Tag)
            True
            >>> unlabeled.title.bold
            False
            >>> unlabeled.title.anchor == sg.Anchor.CENTER
            True
            >>> unlabeled.caption = sg.Tag("Note", (0, 0))
            >>> unlabeled.caption.text
            'Note'
        """
        self._layout_cache: (
            tuple[tuple[float, ...], tuple[float, ...], int] | None
        ) = None
        self._layout_cache_key: tuple[Any, ...] | None = None
        self._layout_revision = 0
        self.title = title
        self.caption = caption
        self.pos = pos
        self.type = Types.TABLE
        self.subtype = Types.TABLE
        self.visible = True
        self.from_bottom_left = from_bottom_left
        self.show_header = show_header
        self.show_lines = show_lines
        self._explicit_headers = False
        self.min_width = min_width
        self.padding = padding
        self.font_size = font_size
        self._format: dict[str, Any] = {"font_size": font_size}
        self._background: dict[str, Any] = {}
        self._columns: list[Column] = []
        self._rows: list[Row] = []
        self._header_row: Row | None = None
        self._default_column_width = column_width
        self._default_row_height = row_height

        row_count = 0
        target_column_count = 0
        if size is not None:
            if len(size) != 2:
                raise ValueError("size must be a (rows, columns) tuple")
            row_count, target_column_count = size
            if (
                type(row_count) is not int
                or type(target_column_count) is not int
                or row_count < 0
                or target_column_count < 0
            ):
                raise ValueError(
                    "size must contain two non-negative integers"
                )

        if (
            size is not None
            and columns is not None
            and len(columns) > target_column_count
        ):
            raise ValueError(
                "columns has more labels than the requested size columns"
            )

        explicit_headers = list(columns) if columns is not None else []
        headers = list(explicit_headers)
        if target_column_count > len(headers):
            headers.extend([""] * (target_column_count - len(headers)))

        if explicit_headers:
            self._explicit_headers = True
        if headers:
            aligns = (
                list(column_align)
                if column_align is not None
                else [_TABLE_DEFAULT_COLUMN_ALIGN] * len(headers)
            )
            for index, header in enumerate(headers):
                align = (
                    aligns[index]
                    if index < len(aligns)
                    else _TABLE_DEFAULT_COLUMN_ALIGN
                )
                column = self._append_column(header)
                column.set_format(align=align)

        if self.show_header and self._explicit_headers and self._columns:
            self._header_row = self.add_row(
                *[column.header_cell for column in self._columns]
            )

        for _ in range(row_count):
            self.add_row(*([""] * len(self._columns)))
        self._apply_header_format()

    def __repr__(self) -> str:
        """Return a Table string from this table's title.

        Examples:
            >>> import simetri.graphics as sg
            >>> repr(sg.Table(title="Parameters"))
            "Table(title='Parameters')"
        """
        title = None if self.title is None else self.title.text
        return f"Table(title={title!r})"

    @property
    def columns(self) -> Range:
        """``Range`` over the grid; ``width`` sets column widths.

        Examples:
            >>> table = Table(columns=["A", "B"])
            >>> table.columns._size_axis
            'columns'
        """
        column_count, grid_row_count = _grid_dimensions(self)
        if column_count == 0 or grid_row_count == 0:
            return Range(self, 0, 0, -1, -1, size_axis="columns")
        public_rows = tuple(_public_full_grid_rows(self))
        return Range(
            self,
            0,
            0,
            grid_row_count - 1,
            column_count - 1,
            size_axis="columns",
            explicit_grid_rows=public_rows,
        )

    @property
    def column_headers(self) -> list[str]:
        """Column header labels.

        Examples:
            >>> Table(columns=["X", "Y"]).column_headers
            ['X', 'Y']
        """
        return [column.header for column in self._columns]

    @property
    def rows(self) -> Range:
        """``Range`` over every grid row; ``height`` sets row heights.

        ``table.rows[i]`` uses the same row index as ``table.cells[i]``.
        Row ``0`` is the first grid row.

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row(1)
            >>> table.rows._size_axis
            'rows'
            >>> table.rows[0].content
            'A'
        """
        column_count, grid_row_count = _grid_dimensions(self)
        if column_count == 0 or grid_row_count == 0:
            return Range(self, 0, 0, -1, -1, size_axis="rows")
        public_rows = tuple(_public_full_grid_rows(self))
        return Range(
            self,
            public_rows[0],
            0,
            public_rows[-1],
            column_count - 1,
            size_axis="rows",
            explicit_grid_rows=public_rows,
        )

    @property
    def cells(self) -> Range:
        """``Range`` over the full grid (header row included when present).

        Index as ``table.cells[row, col]`` — both axes 0-based. Row numbering
        follows the table origin and matches :meth:`cell`, :meth:`row`, and
        :meth:`populate`. With a header, row ``0`` is that header when
        ``from_bottom_left`` is False, and the last row when it is True.

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row("z")
            >>> table.cells[0, 0].content
            'A'
            >>> table.cells[1, 0].content
            'z'
            >>> table.cell(1, 0).content
            'z'
        """
        column_count, grid_row_count = _grid_dimensions(self)
        if column_count == 0 or grid_row_count == 0:
            return _empty_grid_range(self)
        public_rows = tuple(_public_full_grid_rows(self))
        return Range(
            self,
            0,
            0,
            grid_row_count - 1,
            column_count - 1,
            explicit_grid_rows=public_rows,
        )

    def range(
        self,
        row_start: int,
        col_start: int,
        row_end: int,
        col_end: int,
    ) -> Range:
        """Return an inclusive rectangular region (0-based grid indices).

        Same row/column convention as :attr:`cells`: grid row ``0`` is the
        first row at the current origin. The header is included only when
        explicit headers exist and ``show_header`` is True. Column indices start
        at ``0``.

        Args:
            row_start: Top grid row of the range.
            col_start: Left column index.
            row_end: Bottom grid row of the range.
            col_end: Right column index.

        Examples:
            >>> table = Table(columns=["A", "B"])
            >>> _ = table.add_row(1, 2)
            >>> table.range(1, 0, 1, 0).content
            1
        """
        public_rows = _public_full_grid_rows(self)
        start = min(row_start, row_end)
        end = max(row_start, row_end)
        picked_rows = public_rows[start : end + 1]
        return Range(
            self,
            picked_rows[0],
            col_start,
            picked_rows[-1],
            col_end,
            explicit_grid_rows=tuple(picked_rows),
        )

    def _sizes_for_span(
        self,
        span: int,
        size: float | Sequence[float],
        *,
        kind: str,
    ) -> list[float]:
        if isinstance(size, Sequence) and not isinstance(size, (str, bytes)):
            values = list(size)
            if len(values) != span:
                raise ValueError(
                    f"{kind} size sequence length {len(values)} "
                    f"does not match range span {span}"
                )
            return values
        return [float(size)] * span

    def _apply_column_widths(
        self,
        col_start: int,
        col_end: int,
        size: float | Sequence[float],
    ) -> None:
        span = col_end - col_start + 1
        if span <= 0:
            raise ValueError("width assignment on an empty range")
        for col_index, width in zip(
            range(col_start, col_end + 1),
            self._sizes_for_span(span, size, kind="column"),
            strict=True,
        ):
            self._columns[col_index]._width = width
        self._invalidate_layout()

    def _apply_grid_row_heights(
        self,
        row_start: int,
        row_end: int,
        size: float | Sequence[float],
    ) -> None:
        span = row_end - row_start + 1
        if span <= 0:
            raise ValueError("height assignment on an empty range")
        for grid_row, height in zip(
            range(row_start, row_end + 1),
            self._sizes_for_span(span, size, kind="row"),
            strict=True,
        ):
            self._rows[grid_row]._height = height
        self._invalidate_layout()

    def _cell_at_grid(self, grid_row: int, col_index: int) -> Cell:
        column_count, grid_row_count = _grid_dimensions(self)
        if col_index < 0 or col_index >= column_count:
            raise IndexError(f"column index {col_index} out of range")
        if grid_row < 0 or grid_row >= grid_row_count:
            raise IndexError(f"grid row {grid_row} out of range")
        return self._rows[grid_row].cells[col_index]

    def _public_row_grid(self, row_index: int) -> int:
        """Map a public row index to a storage row.

        ``from_bottom_left`` is the only reversal. Every column uses this
        same row.
        """
        return _storage_row_index(self, row_index)

    def _row_object_for_grid(self, grid_row: int) -> Row:
        """Return the stored row at ``grid_row``."""
        _, grid_row_count = _grid_dimensions(self)
        if grid_row < 0 or grid_row >= grid_row_count:
            raise IndexError(f"row index {grid_row} out of range")
        return self._rows[grid_row]

    def cell(self, row_index: int, col_index: int) -> Cell:
        """Return the cell at ``(row_index, col_index)``.

        Row ``0`` is the first row. ``from_bottom_left`` numbers from the
        bottom instead. ``col_index`` ``0`` is the leftmost column. Every
        column uses that same row index.

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row("v")
            >>> table.cell(0, 0).content
            'A'
            >>> table.cell(1, 0).content
            'v'
        """
        grid_row = self._public_row_grid(row_index)
        return self._cell_at_grid(grid_row, col_index)

    def column(self, col_index: int) -> Column:
        """Return the column at ``col_index``.

        Examples:
            >>> table = Table(columns=["H"])
            >>> table.column(0).header
            'H'
        """
        return self._columns[col_index]

    def row(self, row_index: int) -> Row:
        """Return the row at ``row_index``.

        Same index as :meth:`cell` and :meth:`populate`. Row ``0`` is the
        first grid row.

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row(9)
            >>> table.row(0).cells[0].content
            'A'
            >>> table.row(1).cells[0].content
            9
        """
        grid_row = self._public_row_grid(row_index)
        return self._row_object_for_grid(grid_row)

    def set_format(self, **kwargs: Any) -> Self:
        """Apply format to the table and every cell (including headers).

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.set_format(font_size=14)
            >>> table.font_size
            14
        """
        self._format.update(kwargs)
        if "font_size" in kwargs:
            object.__setattr__(self, "font_size", kwargs["font_size"])
        for column in self._columns:
            column.set_format(**kwargs)
        for row in self._rows:
            row.set_format(**kwargs)
        self._invalidate_layout()
        return self

    def format_column(self, column: int, **kwargs: Any) -> Self:
        """Apply format to every cell in ``column``.

        Args:
            column: Column index, 0-based.
            **kwargs: Format and background keys, same as :meth:`set_format`.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(size=(1, 2), show_header=False)
            >>> _ = table.format_column(1, italic=True)
            >>> table.cell(0, 1)._format["italic"]
            True
            >>> "italic" in table.cell(0, 0)._format
            False
        """
        self.column(column).set_format(**kwargs)
        return self

    def format_row(self, row: int, **kwargs: Any) -> Self:
        """Apply format to every cell in ``row``.

        Args:
            row: Row index, 0-based. ``from_bottom_left`` numbers from the
                bottom.
            **kwargs: Format and background keys, same as :meth:`set_format`.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(size=(2, 1), show_header=False)
            >>> _ = table.format_row(1, italic=True)
            >>> table.cell(1, 0)._format["italic"]
            True
            >>> "italic" in table.cell(0, 0)._format
            False
        """
        self.row(row).set_format(**kwargs)
        return self

    def format_range(
        self,
        row_start: int,
        column_start: int,
        row_end: int,
        column_end: int,
        **kwargs: Any,
    ) -> Self:
        """Apply format to every cell in an inclusive rectangle.

        Args:
            row_start: First row index, 0-based.
            column_start: First column index, 0-based.
            row_end: Last row index, 0-based.
            column_end: Last column index, 0-based.
            **kwargs: Format and background keys, same as :meth:`set_format`.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(size=(2, 2), show_header=False)
            >>> _ = table.format_range(0, 1, 1, 1, italic=True)
            >>> table.cell(0, 1)._format["italic"]
            True
            >>> table.cell(1, 1)._format["italic"]
            True
            >>> "italic" in table.cell(0, 0)._format
            False
        """
        self.range(row_start, column_start, row_end, column_end).set_format(
            **kwargs
        )
        return self

    def set_borders(
        self,
        kind: Border | str,
        *,
        line_color: Color | None = None,
        line_style: BorderLine | str | None = None,
    ) -> Self:
        """Apply an Excel border command to every cell.

        ``line_style`` is the weight for edges the command does not already
        name as thick or double. ``NONE`` and ``NO_BORDER`` clear every edge.
        ``line_color`` ``None`` uses black.

        Args:
            kind: Border command.
            line_color: Border color. ``None`` uses black.
            line_style: ``THIN``, ``THICK``, or ``DOUBLE``. ``None`` is thin.

        Returns:
            Self: This table.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(size=(1, 1), show_header=False)
            >>> _ = table.set_borders(
            ...     sg.Border.BOTTOM,
            ...     line_color=sg.red,
            ...     line_style=sg.BorderLine.THICK,
            ... )
            >>> edge = table.cell(0, 0)._borders["bottom"]
            >>> edge["style"] == sg.BorderLine.THICK
            True
            >>> edge["color"] is sg.red
            True
        """
        color = black if line_color is None else line_color
        positions = [
            (row_index, col_index)
            for row_index in range(len(self._rows))
            for col_index in range(len(self._columns))
            if col_index < len(self._rows[row_index].cells)
        ]
        _apply_border_kind(
            self,
            positions,
            _as_border(kind),
            _as_border_line(line_style),
            color,
        )
        self._invalidate_layout()
        return self

    def _apply_header_format(self) -> None:
        """Apply ``header_format`` to row 0 when ``show_header`` is set.

        This is ``format_row(0)``. It does not record a header row.
        """
        if not self.show_header or not self._rows:
            return
        self.format_row(0, **_default_header_format())

    def set_background(self, **kwargs: Any) -> Self:
        """Apply background to the table and every cell (including headers).

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.set_background(fill=True)
            >>> table._background["fill"]
            True
        """
        self._background.update(kwargs)
        for column in self._columns:
            column.set_background(**kwargs)
        for row in self._rows:
            row.set_background(**kwargs)
        return self

    def reset(self) -> Self:
        """Empty all cells and clear table, row, column, and cell styling.

        The table keeps its current row and column counts, title, caption,
        position, and visibility settings. Header labels are cleared because
        header cells
        are part of the grid.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(columns=["A"], size=(1, 1))
            >>> table.cell(0, 0).content = "x"
            >>> _ = table.column(0).set_format(align=sg.Align.RIGHT)
            >>> table.rows.height = 24
            >>> table.reset() is table
            True
            >>> table.column_headers
            ['']
            >>> table.cell(0, 0).content
            ''
            >>> table.column(0)._format
            {}
            >>> table.row(0)._height is None
            True
        """
        self.font_size = _TABLE_DEFAULT_FONT_SIZE
        self._format = {}
        self._background = {}

        for column in self._columns:
            column._format = {}
            column._background = {}
            column._width = self._default_column_width
            column.header = ""
            column.header_cell._format = {}
            column.header_cell._background = {}

        for row in self._rows:
            row._format = {}
            row._background = {}
            row._height = self._default_row_height
            for cell in row.cells:
                cell.content = ""
                cell._format = {}
                cell._background = {}
                cell._borders = {}

        self._apply_header_format()
        self._invalidate_layout()
        return self

    def _invalidate_layout(self) -> None:
        object.__setattr__(self, "_layout_revision", self._layout_revision + 1)
        object.__setattr__(self, "_layout_cache_key", None)
        object.__setattr__(self, "_layout_cache", None)

    def __getattr__(self, name: str) -> Any:
        target = _range_style_target(name)
        if target is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        kind, storage_name = target
        store = self._format if kind == "format" else self._background
        if storage_name not in store:
            return None
        return store[storage_name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("caption", "title"):
            object.__setattr__(self, name, _table_label_tag(value))
            return
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if "_format" not in vars(self):
            object.__setattr__(self, name, value)
            return
        target = _range_style_target(name)
        if target is None:
            object.__setattr__(self, name, value)
            if name == "show_header" and "_rows" in vars(self):
                self._sync_header_row()
                self._apply_header_format()
            if name in {"min_width", "padding", "show_header"}:
                self._invalidate_layout()
            return
        kind, storage_name = target
        if storage_name in ("align", "v_align"):
            value = _resolve_align(value)
        if kind == "format":
            self.set_format(**{storage_name: value})
            return
        self.set_background(**{storage_name: value})

    @property
    def b_box(self) -> BoundingBox:
        """Axis-aligned box enclosing the grid, title, and caption.

        Title and caption positions are updated.

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row("x")
            >>> table.b_box.width > 0
            True
            >>> import simetri.graphics as sg
            >>> labeled = sg.Table(size=(1, 1), show_header=False)
            >>> labeled.title = "Title"
            >>> labeled.caption = "Caption"
            >>> _ = labeled.b_box
            >>> grid = table_grid_bbox(labeled)
            >>> labeled.title.b_box.southwest[1] > grid.northeast[1]
            True
            >>> labeled.caption.b_box.northeast[1] < grid.southwest[1]
            True
        """
        _, row_heights, _ = compute_table_layout(self)
        _place_caption(self, row_heights)
        boxes = [table_grid_bbox(self)]
        if self.title is not None:
            boxes.append(self.title.b_box)
        if self.caption is not None:
            boxes.append(self.caption.b_box)
        return _union_bounding_boxes(boxes)

    def _column_bbox(self, col_index: int) -> BoundingBox:
        _, row_heights, column_count = compute_table_layout(self)
        if col_index >= column_count:
            return BoundingBox()
        boxes = [
            table_cell_bbox(self, grid_row, col_index)
            for grid_row in range(len(row_heights))
        ]
        return _union_bounding_boxes(boxes)

    def _row_bbox(self, row_index: int) -> BoundingBox:
        _, row_heights, column_count = compute_table_layout(self)
        if row_index < 0 or row_index >= len(row_heights):
            return BoundingBox()
        boxes = [
            table_cell_bbox(self, row_index, col_index)
            for col_index in range(column_count)
        ]
        return _union_bounding_boxes(boxes)

    @property
    def corners(self) -> list[PointType]:
        """Bounding quadrilateral of the full table (northwest first).

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row("x")
            >>> len(table.corners)
            4
        """
        box = self.b_box
        if box.southwest is None:
            x, y = self.pos[:2]
            return [(x, y), (x, y), (x, y), (x, y)]
        return [
            box.northwest,
            box.northeast,
            box.southeast,
            box.southwest,
        ]

    @property
    def all_vertices(self) -> list[PointType]:
        """Flattened corner vertices for bounding-box updates.

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row("x")
            >>> len(table.all_vertices) >= 4
            True
        """
        box = self.b_box
        if box.southwest is None:
            x, y = self.pos[:2]
            return [(x, y)]
        return list(box.corners)

    def _append_column(self, header: str) -> Column:
        column = Column(self, len(self._columns), header)
        column._width = self._default_column_width
        self._columns.append(column)
        for row in self._rows:
            if len(row.cells) >= len(self._columns):
                continue
            if row is self._header_row:
                cell = column.header_cell
            else:
                cell = Cell("")
            _apply_row_format_to_cell(row, cell)
            row.cells.append(cell)
        self._invalidate_layout()
        return column

    def _sync_header_row(self) -> None:
        """Include or drop the label row using the same row list as every cell."""
        visible = (
            self.show_header
            and self._explicit_headers
            and bool(self._columns)
        )
        if visible:
            if self._header_row is None or self._header_row not in self._rows:
                self._header_row = self.add_row(
                    *[column.header_cell for column in self._columns]
                )
            return
        if self._header_row is not None and self._header_row in self._rows:
            self._rows.remove(self._header_row)
            for index, row in enumerate(self._rows):
                row.index = index
            self._invalidate_layout()

    def add_column(
        self,
        header: str,
        align: Align | str = _TABLE_DEFAULT_COLUMN_ALIGN,
    ) -> Column:
        """Append a column and return its ``Column`` object.

        Examples:
            >>> table = Table()
            >>> col = table.add_column("New")
            >>> col.header
            'New'
        """
        self._explicit_headers = True
        column = self._append_column(header)
        column.set_format(align=align)
        self._sync_header_row()
        self._apply_header_format()
        return column

    def insert_column(self, index: int) -> Column:
        """Insert an empty column at ``index``.

        Existing columns at that index and after move right by one.
        ``index`` equal to the column count appends.

        Args:
            index: Column index, 0-based.

        Returns:
            Column: The new empty column.

        Raises:
            IndexError: If ``index`` is negative or greater than the
                column count.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(columns=["A", "B"], show_header=False)
            >>> _ = table.add_row("x", "y")
            >>> column = table.insert_column(1)
            >>> column.header
            ''
            >>> table.cell(0, 1).content
            ''
            >>> table.cell(0, 2).content
            'y'
            >>> both = sg.Table(columns=["A", "B"])
            >>> _ = both.insert_row(0)
            >>> _ = both.insert_column(0)
            >>> "bold" in both.cell(0, 0)._format
            False
            >>> both.cell(1, 0)._format["bold"]
            True
            >>> both.cell(1, 0)._background["fill"]
            True
            >>> both.cell(1, 1).content
            'A'
        """
        column_count = len(self._columns)
        if index < 0 or index > column_count:
            raise IndexError(f"column index {index} out of range")
        column = Column(self, index, "")
        column._width = self._default_column_width
        self._columns.insert(index, column)
        for column_index, current_column in enumerate(self._columns):
            current_column.index = column_index
        for row in self._rows:
            cell = Cell("")
            _apply_row_format_to_cell(row, cell)
            row.cells.insert(index, cell)
        self._invalidate_layout()
        return column

    def insert_columns(self, index: int, n_columns: int) -> list[Column]:
        """Insert ``n_columns`` empty columns at ``index``.

        Same index as :meth:`insert_column`. The new columns occupy
        ``index`` through ``index + n_columns - 1``.

        Args:
            index: Column index, 0-based.
            n_columns: How many empty columns to insert.

        Returns:
            list[Column]: The new columns, in index order.

        Raises:
            IndexError: If ``index`` is negative or greater than the
                column count.
            ValueError: If ``n_columns`` is less than 1.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(columns=["A"], show_header=False)
            >>> _ = table.add_row("x")
            >>> columns = table.insert_columns(0, 2)
            >>> [column.header for column in columns]
            ['', '']
            >>> table.cell(0, 2).content
            'x'
        """
        if n_columns < 1:
            raise ValueError("n_columns must be at least 1")
        return [
            self.insert_column(index + offset)
            for offset in range(n_columns)
        ]

    def remove_column(self, index: int) -> Column:
        """Remove the column at ``index``.

        Columns after ``index`` move left by one. Every row loses the
        cell in that column.

        Args:
            index: Column index, 0-based.

        Returns:
            Column: The removed column.

        Raises:
            IndexError: If ``index`` is negative or not a current column.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(columns=["A", "B", "C"], show_header=False)
            >>> _ = table.add_row("x", "y", "z")
            >>> column = table.remove_column(1)
            >>> column.header
            'B'
            >>> table.cell(0, 1).content
            'z'
        """
        column_count = len(self._columns)
        if index < 0 or index >= column_count:
            raise IndexError(f"column index {index} out of range")
        column = self._columns.pop(index)
        for column_index, current_column in enumerate(self._columns):
            current_column.index = column_index
        for row in self._rows:
            row.cells.pop(index)
        self._invalidate_layout()
        return column

    def remove_columns(self, index: int, n_columns: int) -> list[Column]:
        """Remove ``n_columns`` columns starting at ``index``.

        Same index as :meth:`remove_column`. Columns ``index`` through
        ``index + n_columns - 1`` are removed.

        Args:
            index: First column index, 0-based.
            n_columns: How many columns to remove.

        Returns:
            list[Column]: The removed columns, in index order.

        Raises:
            IndexError: If the column span is not inside the table.
            ValueError: If ``n_columns`` is less than 1.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(columns=["A", "B", "C"], show_header=False)
            >>> _ = table.add_row("x", "y", "z")
            >>> columns = table.remove_columns(0, 2)
            >>> [column.header for column in columns]
            ['A', 'B']
            >>> table.cell(0, 0).content
            'z'
        """
        if n_columns < 1:
            raise ValueError("n_columns must be at least 1")
        column_count = len(self._columns)
        if index < 0 or index + n_columns > column_count:
            raise IndexError(f"column index {index} out of range")
        return [self.remove_column(index) for _ in range(n_columns)]

    def add_row(self, *cells: Any) -> Row:
        """Append one data row and return its ``Row`` object.

        Examples:
            >>> table = Table(columns=["A", "B"])
            >>> row = table.add_row(1, 2)
            >>> row.cells[1].content
            2
        """
        if self._columns and len(cells) != len(self._columns):
            raise ValueError(
                f"Row has {len(cells)} cells but table has "
                f"{len(self._columns)} columns."
            )
        if not self._columns and cells:
            for _ in cells:
                self._append_column("")
        row = Row(self, len(self._rows), cells)
        row._height = self._default_row_height
        while len(row.cells) < len(self._columns):
            row.cells.append(Cell(""))
        if self.from_bottom_left:
            self._rows.insert(0, row)
        else:
            self._rows.append(row)
        for index, current_row in enumerate(self._rows):
            current_row.index = index
        self._invalidate_layout()
        return row

    def insert_row(self, index: int) -> Row:
        """Insert an empty row at ``index``.

        Same index as :meth:`cell`. Existing rows at that index and after
        move up by one. ``from_bottom_left`` numbers from the bottom.
        ``index`` equal to the row count appends in that numbering.

        Args:
            index: Row index, 0-based.

        Returns:
            Row: The new empty row.

        Raises:
            IndexError: If ``index`` is negative or greater than the row
                count.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(columns=["A"], show_header=False)
            >>> _ = table.add_row("x")
            >>> row = table.insert_row(0)
            >>> row.cells[0].content
            ''
            >>> table.cell(1, 0).content
            'x'
            >>> bottom = sg.Table(
            ...     size=(2, 1), from_bottom_left=True, show_header=False
            ... )
            >>> bottom.cell(0, 0).content = "south"
            >>> _ = bottom.insert_row(0)
            >>> bottom.cell(0, 0).content
            ''
            >>> bottom.cell(1, 0).content
            'south'
        """
        row_count = len(self._rows)
        if index < 0 or index > row_count:
            raise IndexError(f"row index {index} out of range")
        if self.from_bottom_left:
            storage_index = row_count - index
        else:
            storage_index = index
        row = Row(self, storage_index, [""] * len(self._columns))
        row._height = self._default_row_height
        self._rows.insert(storage_index, row)
        for row_index, current_row in enumerate(self._rows):
            current_row.index = row_index
        self._invalidate_layout()
        return row

    def insert_rows(self, index: int, n_rows: int) -> list[Row]:
        """Insert ``n_rows`` empty rows at ``index``.

        Same index as :meth:`insert_row`. The new rows occupy ``index``
        through ``index + n_rows - 1``. ``from_bottom_left`` numbers from
        the bottom.

        Args:
            index: Row index, 0-based.
            n_rows: How many empty rows to insert.

        Returns:
            list[Row]: The new rows, in index order.

        Raises:
            IndexError: If ``index`` is negative or greater than the row
                count.
            ValueError: If ``n_rows`` is less than 1.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(columns=["A"], show_header=False)
            >>> _ = table.add_row("x")
            >>> rows = table.insert_rows(0, 2)
            >>> len(rows)
            2
            >>> table.cell(0, 0).content
            ''
            >>> table.cell(2, 0).content
            'x'
        """
        if n_rows < 1:
            raise ValueError("n_rows must be at least 1")
        return [
            self.insert_row(index + offset) for offset in range(n_rows)
        ]

    def remove_row(self, index: int) -> Row:
        """Remove the row at ``index``.

        Same index as :meth:`cell`. Rows after ``index`` move up by one.
        ``from_bottom_left`` numbers from the bottom.

        Args:
            index: Row index, 0-based.

        Returns:
            Row: The removed row.

        Raises:
            IndexError: If ``index`` is negative or not a current row.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(columns=["A"], show_header=False)
            >>> _ = table.add_row("x")
            >>> _ = table.add_row("y")
            >>> row = table.remove_row(0)
            >>> row.cells[0].content
            'x'
            >>> table.cell(0, 0).content
            'y'
            >>> bottom = sg.Table(
            ...     size=(2, 1), from_bottom_left=True, show_header=False
            ... )
            >>> bottom.cell(0, 0).content = "south"
            >>> bottom.cell(1, 0).content = "north"
            >>> removed = bottom.remove_row(0)
            >>> removed.cells[0].content
            'south'
            >>> bottom.cell(0, 0).content
            'north'
        """
        row_count = len(self._rows)
        if index < 0 or index >= row_count:
            raise IndexError(f"row index {index} out of range")
        if self.from_bottom_left:
            storage_index = row_count - 1 - index
        else:
            storage_index = index
        row = self._rows.pop(storage_index)
        for row_index, current_row in enumerate(self._rows):
            current_row.index = row_index
        if row is self._header_row:
            self._header_row = None
        self._invalidate_layout()
        return row

    def remove_rows(self, index: int, n_rows: int) -> list[Row]:
        """Remove ``n_rows`` rows starting at ``index``.

        Same index as :meth:`remove_row`. Rows ``index`` through
        ``index + n_rows - 1`` are removed. ``from_bottom_left`` numbers
        from the bottom.

        Args:
            index: First row index, 0-based.
            n_rows: How many rows to remove.

        Returns:
            list[Row]: The removed rows, in index order.

        Raises:
            IndexError: If the row span is not inside the table.
            ValueError: If ``n_rows`` is less than 1.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(size=(3, 1), show_header=False)
            >>> table.cell(0, 0).content = "a"
            >>> table.cell(2, 0).content = "c"
            >>> rows = table.remove_rows(0, 2)
            >>> len(rows)
            2
            >>> table.cell(0, 0).content
            'c'
        """
        if n_rows < 1:
            raise ValueError("n_rows must be at least 1")
        row_count = len(self._rows)
        if index < 0 or index + n_rows > row_count:
            raise IndexError(f"row index {index} out of range")
        return [self.remove_row(index) for _ in range(n_rows)]

    def _clear_cell_content(self, cell: Cell) -> None:
        """Remove ``cell`` content. Formatting stays."""
        cell.content = ""
        for column in self._columns:
            if cell is column.header_cell:
                column._header = ""

    def clear_cell(self, row: int, column: int) -> Self:
        """Remove the content of the cell at ``(row, column)``.

        Same indexes as :meth:`cell`. Formatting on the cell stays.

        Args:
            row: Row index, 0-based.
            column: Column index, 0-based.

        Returns:
            Self: This table.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(columns=["A"], show_header=False)
            >>> _ = table.add_row("x")
            >>> _ = table.cell(0, 0).set_format(bold=True)
            >>> table.clear_cell(0, 0) is table
            True
            >>> table.cell(0, 0).content
            ''
            >>> table.cell(0, 0)._format["bold"]
            True
        """
        self._clear_cell_content(self.cell(row, column))
        self._invalidate_layout()
        return self

    def clear_row(self, index: int) -> Self:
        """Remove the content of every cell in the row at ``index``.

        Same index as :meth:`cell`. Formatting stays.

        Args:
            index: Row index, 0-based.

        Returns:
            Self: This table.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(columns=["A", "B"], show_header=False)
            >>> _ = table.add_row("x", "y")
            >>> _ = table.add_row("p", "q")
            >>> table.clear_row(0) is table
            True
            >>> table.cell(0, 0).content
            ''
            >>> table.cell(0, 1).content
            ''
            >>> table.cell(1, 0).content
            'p'
        """
        for cell in self.row(index).cells:
            self._clear_cell_content(cell)
        self._invalidate_layout()
        return self

    def clear_column(self, index: int) -> Self:
        """Remove the content of every cell in the column at ``index``.

        Formatting stays.

        Args:
            index: Column index, 0-based.

        Returns:
            Self: This table.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(columns=["A", "B"], show_header=False)
            >>> _ = table.add_row("x", "y")
            >>> table.clear_column(0) is table
            True
            >>> table.cell(0, 0).content
            ''
            >>> table.cell(0, 1).content
            'y'
        """
        column_count = len(self._columns)
        if index < 0 or index >= column_count:
            raise IndexError(f"column index {index} out of range")
        for row in self._rows:
            self._clear_cell_content(row.cells[index])
        self._invalidate_layout()
        return self

    def clear_cells(self, range: Range) -> Self:
        """Remove the content of every cell in ``range``.

        Formatting stays.

        Args:
            range: A :class:`Range` from this table.

        Returns:
            Self: This table.

        Raises:
            ValueError: If ``range`` belongs to another table.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(columns=["A", "B"], show_header=False)
            >>> _ = table.add_row("x", "y")
            >>> _ = table.add_row("p", "q")
            >>> table.clear_cells(table.range(0, 1, 1, 1)) is table
            True
            >>> table.cell(0, 1).content
            ''
            >>> table.cell(1, 1).content
            ''
            >>> table.cell(0, 0).content
            'x'
        """
        if range._table is not self:
            raise ValueError("range is not from this table")
        for cell in range:
            self._clear_cell_content(cell)
        self._invalidate_layout()
        return self

    def clear_table(self) -> Self:
        """Remove the content of every cell.

        The table keeps its rows, columns, and formatting.

        Returns:
            Self: This table.

        Examples:
            >>> import simetri.graphics as sg
            >>> table = sg.Table(columns=["A", "B"], show_header=False)
            >>> _ = table.add_row("x", "y")
            >>> _ = table.cell(0, 0).set_format(italic=True)
            >>> table.clear_table() is table
            True
            >>> table.cell(0, 0).content
            ''
            >>> table.cell(0, 1).content
            ''
            >>> table.cell(0, 0)._format["italic"]
            True
        """
        for row in self._rows:
            for cell in row.cells:
                self._clear_cell_content(cell)
        self._invalidate_layout()
        return self

    def _ensure_grid_covers(
        self,
        min_grid_row: int,
        max_grid_row: int,
        max_grid_col: int,
    ) -> None:
        if min_grid_row < 0 or max_grid_col < 0:
            raise IndexError("grid row or column index out of range")
        while len(self._columns) <= max_grid_col:
            self._append_column("")
        _, grid_row_count = _grid_dimensions(self)
        while grid_row_count <= max_grid_row:
            self.add_row(*([""] * len(self._columns)))
            _, grid_row_count = _grid_dimensions(self)

    def populate(
        self,
        row: int,
        column: int,
        columns: int,
        items: Sequence[Any],
        from_bottom_left: bool = False,
    ) -> Self:
        """Fill a rectangle of cells starting at ``(row, column)``.

        All indices are 0-based and match :meth:`cell`. Row ``0`` is the first
        grid row, including a header row when one is shown.
        Items are placed in row-major order with ``columns`` cells per row—the
        same layout as :func:`~simetri.helpers.utilities.get_cell_pos`. When
        ``from_bottom_left`` is False, filling continues at ``row + 1``,
        ``row + 2``, …. When True, it continues toward ``row - 1``,
        ``row - 2``, ….

        Args:
            row: Starting row index (0-based, same as :meth:`cell`).
            column: Starting column index (0-based).
            columns: Number of columns in each filled row (band width, >= 1).
            items: Cell contents in fill order.
            from_bottom_left: If True, advance toward smaller row indices;
                if False, advance toward larger ones.

        Returns:
            Self: This table.

        Raises:
            ValueError: If ``columns`` is less than 1.
            IndexError: If a target row or column index is negative.

        Examples:
            >>> table = Table(columns=["A", "B"])
            >>> table.populate(0, 0, 2, [1, 2, 3]) is table
            True
            >>> table.cell(0, 0).content
            1
            >>> table.cell(0, 1).content
            2
            >>> table.cell(1, 0).content
            3
            >>> table2 = Table(columns=["A", "B"])
            >>> _ = table2.add_row("", "")
            >>> _ = table2.add_row("", "")
            >>> _ = table2.add_row("", "")
            >>> _ = table2.populate(2, 0, 2, ["a", "b", "c"], from_bottom_left=True)
            >>> table2.cell(2, 0).content
            'a'
            >>> table2.cell(1, 0).content
            'c'
        """
        if columns < 1:
            raise ValueError("columns must be at least 1")
        if row < 0 or column < 0:
            raise IndexError("row or column index out of range")
        if not items:
            return self

        item_count = len(items)
        local_rows = (item_count + columns - 1) // columns
        if from_bottom_left:
            min_row = row - (local_rows - 1)
            max_row = row
        else:
            min_row = row
            max_row = row + (local_rows - 1)
        if min_row < 0:
            raise IndexError("row index out of range")
        max_grid_col = column + columns - 1
        self._ensure_grid_covers(0, max_row, max_grid_col)

        for index, item in enumerate(items):
            local_row = index // columns
            local_col = index % columns
            public_row = (
                row - local_row if from_bottom_left else row + local_row
            )
            grid_row = self._public_row_grid(public_row)
            self._cell_at_grid(grid_row, column + local_col).content = item
        return self

    def _rich_style_from_cell(
        self,
        grid_row: int,
        col_index: int,
    ) -> RichStyle | None:
        fmt = _resolve_cell_format(self, grid_row, col_index)
        bg = _resolve_cell_background(self, grid_row, col_index)
        style_kwargs: dict[str, Any] = {}
        if fmt.get("bold"):
            style_kwargs["bold"] = True
        if fmt.get("italic"):
            style_kwargs["italic"] = True
        if "font_color" in fmt:
            style_kwargs["color"] = _rich_color(fmt["font_color"])
        if bg.get("fill") and "fill_color" in bg:
            style_kwargs["bgcolor"] = _rich_color(bg["fill_color"])
        if not style_kwargs:
            return None
        return RichStyle(**style_kwargs)

    def build_rich_table(self) -> RichTable:
        """Build a Rich ``Table`` for notebook display.

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row("x")
            >>> table.build_rich_table()  # doctest: +ELLIPSIS
            <rich.table.Table object at ...>
        """
        title = None if self.title is None else self.title.text
        caption = None if self.caption is None else self.caption.text
        rich_table = RichTable(
            title=title,
            caption=caption,
            show_header=False,
            show_lines=self.show_lines,
            min_width=self.min_width,
            padding=self.padding,
        )
        column_count = len(self._columns)
        for col_index, column in enumerate(self._columns):
            if self._rows:
                fmt = _resolve_cell_format(self, 0, col_index)
            else:
                fmt = _merge_style(
                    _TABLE_DEFAULT_FORMAT,
                    self._format,
                    column._format,
                )
            justify = _align_to_rich_justify(fmt["align"])
            rich_table.add_column(column.header, justify=justify)
        for grid_row, row in enumerate(self._rows):
            rich_cells = []
            from rich.text import Text

            for col_index in range(column_count):
                cell = row.cells[col_index]
                text = _cell_display_text(cell.content)
                style = self._rich_style_from_cell(grid_row, col_index)
                if style is not None:
                    rich_cells.append(Text(text, style=style))
                else:
                    rich_cells.append(text)
            rich_table.add_row(*rich_cells)
        return rich_table

    def display(self) -> None:
        """Show this table in a Jupyter cell using Rich.

        Examples:
            >>> table = Table(columns=["A"])
            >>> table.display()  # doctest: +SKIP
        """
        console = Console(force_jupyter=True)
        console.print(self.build_rich_table())

    def _ipython_display_(self) -> None:
        """IPython hook: Rich table in notebook output."""
        self.display()

    def drawable_cells(self) -> list[Any]:
        """Return drawable objects stored in cells (row-major).

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row("text")
            >>> table.drawable_cells()
            []
        """
        drawables: list[Any] = []
        column_count = len(self._columns)
        for grid_row in _public_full_grid_rows(self):
            for col_index in range(column_count):
                content = self._cell_at_grid(grid_row, col_index).content
                if _is_drawable_content(content):
                    drawables.append(content)
        return drawables

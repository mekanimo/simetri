"""Simetri ``Table`` for documentation layouts.

Cells may hold plain text or drawable objects (``Shape``, ``Group``, ``Lace``,
and similar). Plain text is rendered as ``Tag`` sketches for SVG/PDF export;
drawables are sketched inside their cell bounds.

Styling merges in order: table, column, row, cell (later wins). Header cells
use table, column, and the column header ``Cell`` only.

Notebook preview uses :class:`rich.table.Table` (Jupyter integration).

Most layout defaults for this extension live in this module. Cell **fill**
defaults use ``runtime_defaults['table_fill']`` and ``runtime_defaults['header_row_fill']`` in
``config/settings.py`` (merged before table / column / row / cell layers).

Cell indexing (read this before addressing cells):

- All indices are **0-based** Python integers. There are **no** letter column
  labels (no ``A``, ``B``, ``C`` coordinates).
- **Columns** are always numbered ``0``, ``1``, ``2``, … left to right.
- Two row conventions exist:
  - **Grid row** — used by :attr:`Table.cells`, :meth:`Table.range`,
    :meth:`Table.cell`, and ``table.cells[row, col]``. Row ``0`` is the
    **header** when a header row is shown; the next row is ``1``. With no
    header, row ``0`` is the first body row.
  - **Data row** — used by :meth:`Table.row`, :meth:`Table.populate`, and
    ``table.rows[i]``. Row ``0`` is the first **body** row; the header is
    not counted.
- :attr:`Table.columns` uses **column indices** ``0``, ``1``, … on the full
  grid (including the header row in the underlying ``Range``).
"""

from __future__ import annotations

import simetri.graphics  # populate defaults before illustration import chain

from collections.abc import Iterator, Sequence
from typing import Any, Literal, Self

from rich.console import Console
from rich.style import Style as RichStyle
from rich.table import Table as RichTable

from ..base.all_enums import Align, Anchor, Types
from ..base.common import PointType
from ..coloring.colors import Color, gray, white
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
_TABLE_DEFAULT_HEADER_BOLD = True
_TABLE_DEFAULT_MIN_WIDTH: int | None = None
_TABLE_DEFAULT_PADDING = (0, 1)
_TABLE_DEFAULT_POS: PointType = (0.0, 0.0)
_TABLE_DEFAULT_ROW_ALIGN = Align.VERT_CENTER
_TABLE_DEFAULT_SHOW_HEADER = True
_TABLE_DEFAULT_SHOW_LINES = True
_TABLE_DEFAULT_TITLE = ""
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
    ``0``, ``1``, … ``table.rows[i]`` slices **data rows** only (header
    excluded); ``table.rows[0]`` is the first body row.

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
            header_rows = 1 if _has_header_row(self._table) else 0
            for grid_row in self._grid_row_indices():
                data_row = grid_row - header_rows
                row = self._table._rows[data_row]
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
            if _has_header_row(self._table) and grid_row == 0:
                self._table._header_row_height = row_height
            else:
                data_row = grid_row - (1 if _has_header_row(self._table) else 0)
                self._table._rows[data_row]._height = row_height
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
        table = self._table
        if _has_header_row(table) and grid_row == 0:
            return table._header_row_height
        data_row = grid_row - (1 if _has_header_row(table) else 0)
        return table._rows[data_row]._height

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
    has_header = _has_header_row(table)
    is_header = has_header and grid_row == 0
    data_row_index = grid_row - (1 if has_header else 0)
    layers = [
        _TABLE_DEFAULT_FORMAT,
        table._format,
    ]
    if col_index < len(table._columns):
        column = table._columns[col_index]
        layers.append(column._format)
    if is_header:
        if col_index < len(table._columns):
            layers.append(table._columns[col_index].header_cell._format)
    else:
        row = table._rows[data_row_index]
        layers.append(row._format)
        if col_index < len(row.cells):
            layers.append(row.cells[col_index]._format)
    fmt = _merge_style(*layers)
    if is_header and _TABLE_DEFAULT_HEADER_BOLD and "bold" not in fmt:
        fmt = dict(fmt)
        fmt["bold"] = True
    if is_header and "font_color" not in fmt:
        background = _resolve_cell_background(table, grid_row, col_index)
        if background["fill"]:
            fmt = dict(fmt)
            fmt["font_color"] = white
    return fmt


def _default_cell_background(table: Table, *, is_header: bool) -> dict[str, Any]:
    from ..config.settings import runtime_defaults

    fill = runtime_defaults["table_fill"]
    if is_header:
        fill = runtime_defaults["header_row_fill"]
    return {"fill": fill, "stroke": _TABLE_DEFAULT_STROKE}


def _resolve_cell_background(
    table: Table,
    grid_row: int,
    col_index: int,
) -> dict[str, Any]:
    has_header = _has_header_row(table)
    is_header = has_header and grid_row == 0
    data_row_index = grid_row - (1 if has_header else 0)
    layers = [
        _default_cell_background(table, is_header=is_header),
        table._background,
    ]
    if col_index < len(table._columns):
        column = table._columns[col_index]
        layers.append(column._background)
    if is_header:
        if col_index < len(table._columns):
            layers.append(table._columns[col_index].header_cell._background)
    else:
        row = table._rows[data_row_index]
        layers.append(row._background)
        if col_index < len(row.cells):
            layers.append(row.cells[col_index]._background)
    return _merge_style(*layers)


def _grid_dimensions(table: Table) -> tuple[int, int]:
    column_count = len(table._columns)
    row_count = len(table._rows) + (1 if _has_header_row(table) else 0)
    return column_count, row_count


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
    for column in table._columns:
        header_cell = column.header_cell
        cell_keys.append(
            (
                header_cell._layout_revision,
                _layout_content_cache_key(header_cell.content),
            )
        )
    for row in table._rows:
        cell_keys.extend(
            (
                cell._layout_revision,
                _layout_content_cache_key(cell.content),
            )
            for cell in row.cells
        )
    return table._layout_revision, tuple(cell_keys)


def _has_header_row(table: Table) -> bool:
    return table.show_header and table._explicit_headers and bool(table._columns)


def _public_body_grid_rows(table: Table) -> list[int]:
    header_offset = 1 if _has_header_row(table) else 0
    rows = list(range(header_offset, header_offset + len(table._rows)))
    if table.from_bottom_left:
        rows.reverse()
    return rows


def _public_full_grid_rows(table: Table) -> list[int]:
    body_rows = _public_body_grid_rows(table)
    if not _has_header_row(table):
        return body_rows
    if table.from_bottom_left:
        return body_rows + [0]
    return [0] + body_rows


def _storage_row_index(table: Table, row_index: int) -> int:
    if row_index < 0 or row_index >= len(table._rows):
        raise IndexError(f"row index {row_index} out of range")
    if table.from_bottom_left:
        return len(table._rows) - 1 - row_index
    return row_index


def _grid_row_from_data_row(table: Table, row_index: int) -> int:
    header_offset = 1 if _has_header_row(table) else 0
    return header_offset + _storage_row_index(table, row_index)


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
    has_header = _has_header_row(table)

    for grid_row in range(grid_row_count):
        for col_index in range(column_count):
            fmt = _resolve_cell_format(table, grid_row, col_index)
            if has_header and grid_row == 0:
                content = table._columns[col_index].header_cell.content
            else:
                data_row = grid_row - (1 if has_header else 0)
                content = table._rows[data_row].cells[col_index].content
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
        if has_header and grid_row == 0:
            header_height = table._header_row_height
            if header_height is not None:
                row_heights[grid_row] = max(
                    row_heights[grid_row], header_height
                )
        else:
            data_row = grid_row - (1 if has_header else 0)
            height = table._rows[data_row]._height
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
    pos = table.pos[:2]
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
    pos = table.pos[:2]
    x0, y0 = pos
    return _bbox_from_cell_bounds(
        x0,
        y0,
        x0 + sum(col_widths),
        y0 - sum(row_heights),
    )


def table_title_bbox(table: Table) -> BoundingBox | None:
    """Bounding box of the title tag, or ``None`` when ``title`` is empty.

    Examples:
        >>> table = Table(title="", columns=["A"])
        >>> table_title_bbox(table) is None
        True
        >>> table.title = "Summary"
        >>> table_title_bbox(table).width > 0
        True
    """
    if not table.title:
        return None
    pos = table.pos[:2]
    title_y = pos[1] + _TABLE_DEFAULT_TITLE_PAD
    title_tag = Tag(
        table.title,
        (pos[0], title_y),
        font_size=table.font_size + 2,
        bold=True,
        anchor=Anchor.WEST,
        align=Align.LEFT,
    )
    return title_tag.b_box


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
    pos = table.pos[:2]
    sketches: list = []
    grid_row_count = len(row_heights)
    padding_x, padding_y = _table_padding_xy(table)

    if table.title:
        title_y = pos[1] + _TABLE_DEFAULT_TITLE_PAD
        title_tag = Tag(
            table.title,
            (pos[0], title_y),
            font_size=table.font_size + 2,
            bold=True,
            anchor=Anchor.WEST,
            align=Align.LEFT,
        )
        sketches.extend(get_sketches(title_tag, canvas, **kwargs))
        pos = (pos[0], pos[1] - _TABLE_DEFAULT_TITLE_PAD - table.font_size)

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

    has_header = _has_header_row(table)
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
            if has_header and grid_row == 0:
                content = table._columns[col_index].header_cell.content
            else:
                data_row = grid_row - (1 if has_header else 0)
                content = table._rows[data_row].cells[col_index].content
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
    Use **grid rows** for :attr:`cells`, :meth:`range`, and :meth:`cell`
    (row ``0`` is the header when a header is shown; the next row is ``1``).
    Use **data rows** for :meth:`row` and :meth:`populate` (row ``0`` is the
    first body row). See the module docstring for the full convention.
    Header strings in ``columns=["A", "B"]`` are display labels only — not
    column coordinates.

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
        title: str = _TABLE_DEFAULT_TITLE,
        columns: Sequence[str] | None = None,
        *,
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
            title: Table title for notebook and vector output.
            columns: Optional column header labels.
            size: Optional ``(rows, columns)`` body size to pre-create.
            from_bottom_left: Whether row indexing starts at the bottom.
            pos: Northwest table position.
            show_header: Whether to show the header row.
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
        """
        self._layout_cache: (
            tuple[tuple[float, ...], tuple[float, ...], int] | None
        ) = None
        self._layout_cache_key: tuple[Any, ...] | None = None
        self._layout_revision = 0
        self.title = title
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
        self._default_column_width = column_width
        self._default_row_height = row_height
        self._header_row_height: float | None = row_height

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

        for _ in range(row_count):
            self.add_row(*([""] * len(self._columns)))

    def __repr__(self) -> str:
        """Return a Table string from this table's title.

        Examples:
            >>> import simetri.graphics as sg
            >>> repr(sg.Table(title="Parameters"))
            "Table(title='Parameters')"
        """
        return f"Table(title={self.title!r})"

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
        """``Range`` over data rows; ``height`` sets row heights.

        ``table.rows[i]`` uses **data-row** indices: ``0`` is the first body
        row at the current origin (top by default, bottom when
        ``from_bottom_left`` is True). This differs from ``table.cells[i, …]``,
        which uses **grid-row** indices.

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row(1)
            >>> table.rows._size_axis
            'rows'
        """
        column_count, grid_row_count = _grid_dimensions(self)
        if column_count == 0:
            return Range(self, 0, 0, -1, -1, size_axis="rows")
        public_rows = _public_body_grid_rows(self)
        if not public_rows:
            return Range(self, 0, 0, -1, -1, size_axis="rows")
        return Range(
            self,
            public_rows[0],
            0,
            public_rows[-1],
            column_count - 1,
            size_axis="rows",
            explicit_grid_rows=tuple(public_rows),
        )

    @property
    def cells(self) -> Range:
        """``Range`` over the full grid (header row included when present).

        Index as ``table.cells[grid_row, col]`` — both axes 0-based **grid**
        indices. Row numbering follows the table origin. With explicit headers
        and ``show_header=True``, the header is row ``0`` only when
        ``from_bottom_left`` is False; with ``from_bottom_left=True`` the header
        is the last grid row. For body-only row numbers use :meth:`cell` or
        :meth:`populate` instead.

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
            if _has_header_row(self) and grid_row == 0:
                self._header_row_height = height
            else:
                data_row = grid_row - (1 if _has_header_row(self) else 0)
                self._rows[data_row]._height = height
        self._invalidate_layout()

    def _cell_at_grid(self, grid_row: int, col_index: int) -> Cell:
        column_count, grid_row_count = _grid_dimensions(self)
        if col_index < 0 or col_index >= column_count:
            raise IndexError(f"column index {col_index} out of range")
        if grid_row < 0 or grid_row >= grid_row_count:
            raise IndexError(f"grid row {grid_row} out of range")
        if _has_header_row(self) and grid_row == 0:
            return self._columns[col_index].header_cell
        data_row = grid_row - (1 if _has_header_row(self) else 0)
        return self._rows[data_row].cells[col_index]

    def cell(self, row_index: int, col_index: int) -> Cell:
        """Return the cell at ``(row_index, col_index)``.

        Row ``0`` is the header when a header row is shown. The next row is
        ``1``. With no header, row ``0`` is the first body row.
        ``col_index`` ``0`` is the leftmost column.

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row("v")
            >>> table.cell(0, 0).content
            'A'
            >>> table.cell(1, 0).content
            'v'
        """
        if _has_header_row(self):
            return self._cell_at_grid(row_index, col_index)
        storage_index = _storage_row_index(self, row_index)
        return self._rows[storage_index].cells[col_index]

    def column(self, col_index: int) -> Column:
        """Return the column at ``col_index``.

        Examples:
            >>> table = Table(columns=["H"])
            >>> table.column(0).header
            'H'
        """
        return self._columns[col_index]

    def row(self, row_index: int) -> Row:
        """Return the data row at ``row_index`` (0-based, header excluded).

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row(9)
            >>> table.row(0).cells[0].content
            9
        """
        storage_index = _storage_row_index(self, row_index)
        return self._rows[storage_index]

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

        The table keeps its current row and column counts, title, position,
        and visibility settings. Header labels are cleared because header cells
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
        self._header_row_height = self._default_row_height

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
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if "_format" not in vars(self):
            object.__setattr__(self, name, value)
            return
        target = _range_style_target(name)
        if target is None:
            object.__setattr__(self, name, value)
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
        """Axis-aligned box enclosing the grid and optional title.

        Examples:
            >>> table = Table(columns=["A"])
            >>> _ = table.add_row("x")
            >>> table.b_box.width > 0
            True
        """
        boxes = [table_grid_bbox(self)]
        title_box = table_title_bbox(self)
        if title_box is not None:
            boxes.append(title_box)
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
        grid_row = _grid_row_from_data_row(self, row_index)
        if grid_row >= len(row_heights):
            return BoundingBox()
        boxes = [
            table_cell_bbox(self, grid_row, col_index)
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
            while len(row.cells) < len(self._columns):
                row.cells.append(Cell(""))
        self._invalidate_layout()
        return column

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
        return column

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
        """Fill a rectangle of body cells starting at ``(row, column)``.

        All indices are 0-based. ``row`` and ``column`` are **data-row**
        indices (the header is not part of ``row``). :meth:`cell` counts the
        header as row ``0`` when a header is shown.
        Items are placed in row-major order with ``columns`` cells per row—the
        same layout as :func:`~simetri.helpers.utilities.get_cell_pos`. When
        ``from_bottom_left`` is False, filling continues at ``row + 1``,
        ``row + 2``, … in increasing data-row order. When True, it continues
        toward ``row - 1``, ``row - 2``, … instead.

        Args:
            row: Starting data row index (0-based).
            column: Starting column index (0-based).
            columns: Number of columns in each filled row (band width, >= 1).
            items: Cell contents in fill order.
            from_bottom_left: If True, advance toward smaller data-row indices;
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
            >>> table.cell(1, 0).content
            1
            >>> table.cell(1, 1).content
            2
            >>> table.cell(2, 0).content
            3
            >>> table2 = Table(columns=["A", "B"])
            >>> _ = table2.add_row("", "")
            >>> _ = table2.add_row("", "")
            >>> _ = table2.add_row("", "")
            >>> _ = table2.populate(2, 0, 2, ["a", "b", "c"], from_bottom_left=True)
            >>> table2.cell(3, 0).content
            'a'
            >>> table2.cell(2, 0).content
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
            min_data_row = row - (local_rows - 1)
            max_data_row = row
        else:
            min_data_row = row
            max_data_row = row + (local_rows - 1)
        if min_data_row < 0:
            raise IndexError("row index out of range")
        max_grid_col = column + columns - 1
        while len(self._columns) <= max_grid_col:
            self._append_column("")
        while len(self._rows) <= max_data_row:
            self.add_row(*([""] * len(self._columns)))

        for index, item in enumerate(items):
            local_row = index // columns
            local_col = index % columns
            grid_col = column + local_col
            if from_bottom_left:
                data_row = row - local_row
            else:
                data_row = row + local_row
            grid_row = _grid_row_from_data_row(self, data_row)
            self._cell_at_grid(grid_row, grid_col).content = item
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
        rich_table = RichTable(
            title=self.title or None,
            show_header=False,
            show_lines=self.show_lines,
            min_width=self.min_width,
            padding=self.padding,
        )
        column_count = len(self._columns)
        has_header = _has_header_row(self)
        for col_index, column in enumerate(self._columns):
            fmt = _resolve_cell_format(self, 0, col_index)
            justify = _align_to_rich_justify(fmt["align"])
            rich_table.add_column(column.header, justify=justify)
        if has_header:
            from rich.text import Text

            header_cells = []
            for col_index in range(column_count):
                style = self._rich_style_from_cell(0, col_index)
                text = self._columns[col_index].header
                if style is not None:
                    header_cells.append(Text(text, style=style))
                else:
                    header_cells.append(text)
            rich_table.add_row(*header_cells)
        grid_header_rows = 1 if has_header else 0
        for row_index, row in enumerate(self._rows):
            grid_row = row_index + grid_header_rows
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
        for row in self._rows:
            for cell in row.cells:
                content = cell.content
                if _is_drawable_content(content):
                    drawables.append(content)
        return drawables

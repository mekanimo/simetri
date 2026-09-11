"""Creates a where-used dictionary."""

from __future__ import annotations

import ast
import asyncio
import html
import json
import pickle
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import networkx as nx
from multilspy import LanguageServer
from multilspy.multilspy_config import MultilspyConfig
from multilspy.multilspy_logger import MultilspyLogger
from pyvis.network import Network

DEFAULT_START_PATH = "c:/uv_simetri_3.9/simetri/src/simetri/"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "where_used_cache.pkl"
DEFAULT_EXCLUDE = ("gui", "tests")
_REPR_PATH_PREFIX = "c:/uv_simetri_3.9/simetri/src/"

DRY_RUN_START_PATH = "D:/simetri_backup_Sep_6/src"
DRY_RUN_CACHE_PATH = Path("D:/simetri_backup_Sep_6/where_used_cache.pkl")
DRY_RUN_GRAPH_PATH = Path("D:/simetri_backup_Sep_6/where_used_graph.html")
DRY_RUN_HTML_DIR = Path("D:/simetri_backup_Sep_6/where_used_html")
_DRY_RUN_PROBE_RELATIVE = "simetri/_where_used_dry_run_probe.py"
_DRY_RUN_PROBE_CALLEE = "double_area3"
_REAL_REPO_MARKER = "uv_simetri_3.9"

WHERE_USED_LINK_PREFIX = "# Where used: "

# Graph / HTML colors for caller kinds.
COLOR_EXACT = "#2e7d32"
COLOR_GUESS = "#ef6c00"
COLOR_FOCUS = "#1565c0"
COLOR_ENUM_MEMBER = "#1565c0"
COLOR_ENUM_USER = "#f9a825"


def _as_posix_dir(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("\\", "/").rstrip("/") + "/"


def _use_dry_run_paths() -> None:
    """Point short-path stripping at the backup tree; refuse the real repo."""
    global _REPR_PATH_PREFIX
    start = Path(DRY_RUN_START_PATH).resolve()
    start_text = str(start).replace("\\", "/").lower()
    if _REAL_REPO_MARKER in start_text:
        raise RuntimeError(
            f"dry_run start_path looks like the real repo: {start}"
        )
    if not start.is_dir():
        raise FileNotFoundError(f"dry_run start_path missing: {start}")
    _REPR_PATH_PREFIX = _as_posix_dir(start)


class CallableEntity:
    def __init__(
        self,
        filename: str,
        class_name: str | None = None,
        function_name: str | None = None,
    ):
        self.filename = filename
        self.class_name = class_name
        self.function_name = function_name

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CallableEntity):
            return NotImplemented
        return (
            self.filename == other.filename
            and self.class_name == other.class_name
            and self.function_name == other.function_name
        )

    def __hash__(self) -> int:
        return hash((self.filename, self.class_name, self.function_name))

    def __repr__(self) -> str:
        if self.class_name and self.function_name:
            qual = f"{self.class_name}.{self.function_name}"
        elif self.function_name:
            qual = self.function_name
        elif self.class_name:
            qual = self.class_name
        else:
            qual = "<module>"
        display_path = self.filename.replace("\\", "/")
        display_path = display_path.removeprefix(_REPR_PATH_PREFIX)
        return f"CallableEntity({display_path!r}, {qual})"


class WhereUsedMaps:
    """Exact callers (calls / resolved property reads) and guessed property reads."""

    def __init__(
        self,
        exact: dict[CallableEntity, list[CallableEntity]] | None = None,
        guess: dict[CallableEntity, list[CallableEntity]] | None = None,
    ):
        self.exact: dict[CallableEntity, list[CallableEntity]] = (
            exact if exact is not None else {}
        )
        self.guess: dict[CallableEntity, list[CallableEntity]] = (
            guess if guess is not None else {}
        )

    def has_users(self, entity: CallableEntity) -> bool:
        return bool(self.exact[entity]) or bool(self.guess[entity])

    def ensure_entity(self, entity: CallableEntity) -> None:
        if entity not in self.exact:
            self.exact[entity] = []
        if entity not in self.guess:
            self.guess[entity] = []

    def sort_all(self) -> None:
        for users in self.exact.values():
            users.sort(key=_user_sort_key)
        for users in self.guess.values():
            users.sort(key=_user_sort_key)

    def __iter__(self):
        return iter(self.exact)

    def __contains__(self, entity: object) -> bool:
        return entity in self.exact

    def __len__(self) -> int:
        return len(self.exact)


def _iter_python_files(start_path: Path, exclude: tuple[str, ...] | list[str]):
    exclude_names = set(exclude)
    for path in sorted(start_path.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if exclude_names.intersection(path.parts):
            continue
        yield path


def _collect_entities(
    tree: ast.AST, filename: str
) -> tuple[list[CallableEntity], set[CallableEntity]]:
    """Return all callable entities and the subset that are ``@property`` methods."""
    entities: list[CallableEntity] = []
    property_entities: set[CallableEntity] = set()

    class Collector(ast.NodeVisitor):
        def __init__(self):
            self.class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef):
            entities.append(
                CallableEntity(
                    filename, class_name=node.name, function_name=None
                )
            )
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            self._add_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self._add_function(node)

        def _add_function(self, node):
            class_name = self.class_stack[-1] if self.class_stack else None
            entity = CallableEntity(
                filename, class_name=class_name, function_name=node.name
            )
            entities.append(entity)
            if _function_is_property(node):
                property_entities.add(entity)
            self.generic_visit(node)

    Collector().visit(tree)
    return entities, property_entities


def _function_is_property(node: ast.AST) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "property":
            return True
        if (
            isinstance(decorator, ast.Attribute)
            and decorator.attr == "property"
        ):
            return True
    return False


def _call_spec(node: ast.Call) -> tuple[str | None, str] | None:
    """Return ``(receiver, name)`` for a call, or ``None`` if unsupported.

    Examples:
        ``foo()`` → ``(None, "foo")``
        ``self.bar()`` → ``("self", "bar")``
        ``Dots.__init__(...)`` → ``("Dots", "__init__")``
        ``super().__init__()`` → ``("super", "__init__")``
    """
    func = node.func
    if isinstance(func, ast.Name):
        return (None, func.id)
    if isinstance(func, ast.Attribute):
        receiver: str | None = None
        if isinstance(func.value, ast.Name):
            receiver = func.value.id
        elif (
            isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Name)
            and func.value.func.id == "super"
        ):
            receiver = "super"
        return (receiver, func.attr)
    return None


def _callee_matches_call(
    callee: CallableEntity,
    receiver: str | None,
    name: str,
    caller_filename: str,
    caller_class: str | None,
) -> bool:
    """Whether a call site should count as using ``callee``.

    Module-level functions match only bare ``name()`` calls.
    Methods match ``self.name`` / ``cls.name`` in the same class+file, or
    explicit ``ClassName.name``; ``super().name`` is skipped (needs MRO).
    """
    if callee.function_name != name:
        return False
    if callee.class_name is None:
        return receiver is None
    if receiver is None:
        return False
    if receiver in ("self", "cls"):
        return (
            caller_class == callee.class_name
            and caller_filename == callee.filename
        )
    if receiver == "super":
        return False
    return receiver == callee.class_name


def _find_callers(
    tree: ast.AST,
    filename: str,
    by_function_name: dict[str, list[CallableEntity]],
    by_class_name: dict[str, list[CallableEntity]],
    maps: WhereUsedMaps,
) -> None:
    """Record call sites into ``maps.exact``.

    Bare ``ClassName(...)`` is an exact use of the class, not of ``__init__``.
    """

    class CallFinder(ast.NodeVisitor):
        def __init__(self):
            self.class_stack: list[str] = []
            self.func_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef):
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            self.func_stack.append(node.name)
            self.generic_visit(node)
            self.func_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self.func_stack.append(node.name)
            self.generic_visit(node)
            self.func_stack.pop()

        def visit_Call(self, node: ast.Call):
            if self.func_stack:
                class_name = self.class_stack[-1] if self.class_stack else None
                caller = CallableEntity(
                    filename,
                    class_name=class_name,
                    function_name=self.func_stack[-1],
                )
                spec = _call_spec(node)
                if spec is not None:
                    receiver, name = spec
                    if name in by_function_name:
                        for callee in by_function_name[name]:
                            if not _callee_matches_call(
                                callee,
                                receiver,
                                name,
                                filename,
                                class_name,
                            ):
                                continue
                            users = maps.exact[callee]
                            if caller not in users:
                                users.append(caller)
                    # Division(...) / Fragment(...), not Class.__init__(...).
                    if receiver is None and name in by_class_name:
                        for callee in by_class_name[name]:
                            users = maps.exact[callee]
                            if caller not in users:
                                users.append(caller)
            self.generic_visit(node)

    CallFinder().visit(tree)


def _attribute_spec(node: ast.Attribute) -> tuple[str | None, str] | None:
    """Return ``(receiver, attr)`` for a Load attribute, else ``None``."""
    if not isinstance(node.ctx, ast.Load):
        return None
    receiver: str | None = None
    if isinstance(node.value, ast.Name):
        receiver = node.value.id
    return (receiver, node.attr)


def _find_property_users(
    tree: ast.AST,
    filename: str,
    by_property_name: dict[str, list[CallableEntity]],
    maps: WhereUsedMaps,
) -> None:
    """Record ``obj.prop`` loads against ``@property`` methods (exact or guess)."""

    class AttrFinder(ast.NodeVisitor):
        def __init__(self):
            self.class_stack: list[str] = []
            self.func_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef):
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            self.func_stack.append(node.name)
            self.generic_visit(node)
            self.func_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self.func_stack.append(node.name)
            self.generic_visit(node)
            self.func_stack.pop()

        def visit_Attribute(self, node: ast.Attribute):
            if self.func_stack:
                spec = _attribute_spec(node)
                if spec is not None:
                    receiver, name = spec
                    if name in by_property_name:
                        class_name = (
                            self.class_stack[-1] if self.class_stack else None
                        )
                        caller = CallableEntity(
                            filename,
                            class_name=class_name,
                            function_name=self.func_stack[-1],
                        )
                        for callee in by_property_name[name]:
                            if caller == callee:
                                continue
                            if _callee_matches_call(
                                callee,
                                receiver,
                                name,
                                filename,
                                class_name,
                            ):
                                exact_users = maps.exact[callee]
                                if caller not in exact_users:
                                    exact_users.append(caller)
                                guess_users = maps.guess[callee]
                                if caller in guess_users:
                                    guess_users.remove(caller)
                            else:
                                # Name-only attribute load → guessed use.
                                if caller in maps.exact[callee]:
                                    continue
                                guess_users = maps.guess[callee]
                                if caller not in guess_users:
                                    guess_users.append(caller)
            self.generic_visit(node)

    AttrFinder().visit(tree)


def _path_key(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("\\", "/").casefold()


def _uri_to_path(uri: str) -> Path:
    path = unquote(urlparse(uri).path)
    if path.startswith("/") and len(path) > 2 and path[2] == ":":
        path = path[1:]
    return Path(path)


def _identifier_column(line_text: str, col_offset: int, name: str) -> int:
    """Column of ``name`` as an identifier, not a prefix of a longer name."""
    index = col_offset
    while True:
        column = line_text.find(name, index)
        if column < 0:
            raise ValueError(
                f"identifier {name!r} not found on definition line: {line_text!r}"
            )
        end = column + len(name)
        before = "" if column == 0 else line_text[column - 1]
        after = "" if end == len(line_text) else line_text[end]
        before_ok = not (before.isalnum() or before == "_")
        after_ok = not (after.isalnum() or after == "_")
        if before_ok and after_ok:
            return column
        index = column + 1


def _index_file_for_references(
    tree: ast.AST, filename: str, source: str
) -> tuple[
    list[tuple[CallableEntity, int, int]],
    list[tuple[int, int, CallableEntity]],
]:
    """Definition positions (LSP line/column) and function spans (1-based)."""
    lines = source.splitlines()
    sites: list[tuple[CallableEntity, int, int]] = []
    spans: list[tuple[int, int, CallableEntity]] = []

    class Indexer(ast.NodeVisitor):
        def __init__(self):
            self.class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef):
            entity = CallableEntity(
                filename, class_name=node.name, function_name=None
            )
            sites.append(
                (
                    entity,
                    node.lineno - 1,
                    _identifier_column(
                        lines[node.lineno - 1], node.col_offset, node.name
                    ),
                )
            )
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            self._add_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self._add_function(node)

        def _add_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
            class_name = self.class_stack[-1] if self.class_stack else None
            entity = CallableEntity(
                filename, class_name=class_name, function_name=node.name
            )
            sites.append(
                (
                    entity,
                    node.lineno - 1,
                    _identifier_column(
                        lines[node.lineno - 1], node.col_offset, node.name
                    ),
                )
            )
            if node.end_lineno is None:
                raise ValueError(
                    f"definition has no end_lineno: {filename}:{node.lineno} {node.name}"
                )
            spans.append((node.lineno, node.end_lineno, entity))
            self.generic_visit(node)

    Indexer().visit(tree)
    return sites, spans


def _enclosing_function(
    spans: list[tuple[int, int, CallableEntity]], lineno: int
) -> CallableEntity | None:
    """Innermost function whose span contains the 1-based ``lineno``."""
    best: CallableEntity | None = None
    best_width: int | None = None
    for start, end, entity in spans:
        if start <= lineno <= end:
            width = end - start
            if best_width is None or width < best_width:
                best = entity
                best_width = width
    return best


def _record_reference(
    ref: dict,
    entity: CallableEntity,
    definition_line: int,
    spans_by_key: dict[str, list[tuple[int, int, CallableEntity]]],
    path_key_to_filename: dict[str, str],
    maps: WhereUsedMaps,
) -> None:
    if "uri" not in ref or "range" not in ref:
        raise KeyError(f"reference missing uri/range: {ref!r}")
    ref_key = _path_key(_uri_to_path(ref["uri"]))
    if ref_key not in path_key_to_filename:
        return
    start = ref["range"]["start"]
    if "line" not in start:
        raise KeyError(f"reference range missing start line: {ref!r}")
    # The definition itself is not a user. Jedi reports it even when
    # includeDeclaration is false.
    if (
        ref_key == _path_key(entity.filename)
        and start["line"] == definition_line
    ):
        return
    caller = _enclosing_function(spans_by_key[ref_key], start["line"] + 1)
    if caller is None:
        return
    users = maps.exact[entity]
    if caller not in users:
        users.append(caller)


async def _fill_references_with_multilspy(
    root: Path,
    sites_by_relpath: dict[str, list[tuple[CallableEntity, int, int]]],
    spans_by_key: dict[str, list[tuple[int, int, CallableEntity]]],
    path_key_to_filename: dict[str, str],
    maps: WhereUsedMaps,
) -> None:
    """Ask jedi-language-server for references of each collected definition."""
    config = MultilspyConfig.from_dict({"code_language": "python"})
    logger = MultilspyLogger()
    repository_root = str(root.resolve())
    server = LanguageServer.create(config, logger, repository_root)
    total = sum(len(sites) for sites in sites_by_relpath.values())
    done = 0
    print(f"where_used: querying {total} symbols with multilspy")
    async with server.start_server():
        for rel_path, sites in sites_by_relpath.items():
            uri = (root.resolve() / rel_path).as_uri()
            with server.open_file(rel_path):
                for entity, line, column in sites:
                    response = await server.server.send.references(
                        {
                            "context": {"includeDeclaration": False},
                            "textDocument": {"uri": uri},
                            "position": {"line": line, "character": column},
                        }
                    )
                    done += 1
                    if done == total or done % 50 == 0:
                        print(f"where_used: references {done}/{total}")
                    if response is None:
                        continue
                    if not isinstance(response, list):
                        raise TypeError(
                            f"unexpected references response: {response!r}"
                        )
                    for ref in response:
                        _record_reference(
                            ref,
                            entity,
                            line,
                            spans_by_key,
                            path_key_to_filename,
                            maps,
                        )


def get_where_used(
    start_path: str = DEFAULT_START_PATH,
    exclude: tuple[str, ...] | list[str] = DEFAULT_EXCLUDE,
) -> WhereUsedMaps:
    """Scan ``start_path`` and return where-used maps from multilspy references.

    Exact users are functions or methods that reference the entity. The
    definition itself is not a user. References outside a function, and
    references in excluded files, are ignored. Guessed users are left empty.
    """
    root = Path(start_path)
    maps = WhereUsedMaps()
    sites_by_relpath: dict[str, list[tuple[CallableEntity, int, int]]] = (
        defaultdict(list)
    )
    spans_by_key: dict[str, list[tuple[int, int, CallableEntity]]] = {}
    path_key_to_filename: dict[str, str] = {}
    parse_errors: list[tuple[str, SyntaxError]] = []
    root_resolved = root.resolve()

    for path in _iter_python_files(root, exclude):
        filename = str(path).replace("\\", "/")
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as exc:
            parse_errors.append((filename, exc))
            continue
        for entity in _collect_entities(tree, filename)[0]:
            maps.ensure_entity(entity)
        sites, spans = _index_file_for_references(tree, filename, source)
        rel_path = (
            Path(filename).resolve().relative_to(root_resolved).as_posix()
        )
        sites_by_relpath[rel_path].extend(sites)
        spans_by_key[_path_key(filename)] = spans
        path_key_to_filename[_path_key(filename)] = filename

    if parse_errors:
        for filename, exc in parse_errors:
            print(f"where_used: skip (SyntaxError): {filename}: {exc}")

    asyncio.run(
        _fill_references_with_multilspy(
            root_resolved,
            sites_by_relpath,
            spans_by_key,
            path_key_to_filename,
            maps,
        )
    )
    maps.sort_all()
    return maps


def save_where_used(
    maps: WhereUsedMaps,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> Path:
    """Pickle ``WhereUsedMaps`` for later ``load_where_used``."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(maps, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_where_used(
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> WhereUsedMaps:
    """Load a previously saved ``WhereUsedMaps`` from pickle."""
    path = Path(cache_path)
    with path.open("rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, WhereUsedMaps):
        raise TypeError(
            f"where_used cache must be WhereUsedMaps, got {type(loaded)!r}"
        )
    return loaded


def get_key(
    function_name: str,
    d_where_used: WhereUsedMaps | None = None,
) -> CallableEntity | list[CallableEntity]:
    """Return the ``CallableEntity`` key(s) for ``function_name``.

    If ``d_where_used`` is omitted, loads from ``DEFAULT_CACHE_PATH``.
    """
    maps = load_where_used() if d_where_used is None else d_where_used
    matches = [
        entity for entity in maps.exact if entity.function_name == function_name
    ]
    if not matches:
        raise KeyError(
            f"No CallableEntity with function_name={function_name!r}"
        )
    if len(matches) == 1:
        return matches[0]
    return matches


def _short_path(filename: str) -> str:
    display_path = filename.replace("\\", "/")
    display_path = display_path.removeprefix(_REPR_PATH_PREFIX)
    return display_path


def _entity_label(entity: CallableEntity) -> str:
    if entity.class_name and entity.function_name:
        return f"{entity.class_name}.{entity.function_name}"
    if entity.function_name:
        return entity.function_name
    if entity.class_name:
        return entity.class_name
    return "<module>"


def _user_sort_key(user: CallableEntity) -> tuple[str, str]:
    return (_short_path(user.filename), _entity_label(user))


def _sorted_user_keys(users: list[CallableEntity]) -> list[tuple[str, str]]:
    """Alphabetical ``(module, entity)`` keys; list compare can exit early."""
    return sorted(_user_sort_key(user) for user in users)


def _users_changed(
    old_users: list[CallableEntity], new_users: list[CallableEntity]
) -> bool:
    if len(old_users) != len(new_users):
        return True
    old_keys = _sorted_user_keys(old_users)
    new_keys = _sorted_user_keys(new_users)
    for old_key, new_key in zip(old_keys, new_keys):
        if old_key != new_key:
            return True
    return False


def _is_where_used_link_line(line: str) -> bool:
    return line.lstrip().startswith(WHERE_USED_LINK_PREFIX)


def _strip_where_used_blocks(text: str) -> str:
    """Remove where-used comment lines."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    index = 0
    while index < len(lines):
        if _is_where_used_link_line(lines[index]):
            if out and out[-1].strip() == "":
                out.pop()
            index += 1
            if index < len(lines) and lines[index].strip() == "":
                index += 1
            continue
        out.append(lines[index])
        index += 1
    return "".join(out)


def _module_html_stem(short_path: str) -> str:
    stem = short_path.replace("\\", "/")
    stem = stem.removesuffix(".py")
    return stem.replace("/", "__")


def _entity_anchor(entity: CallableEntity) -> str:
    label = _entity_label(entity)
    return "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in label
    )


def _module_html_path(entity: CallableEntity, html_dir: Path) -> Path:
    return html_dir / f"{_module_html_stem(_short_path(entity.filename))}.html"


def _entity_html_href(entity: CallableEntity, html_dir: Path) -> str:
    page = _module_html_path(entity, html_dir).resolve()
    return f"{page.as_uri()}#{_entity_anchor(entity)}"


def _definition_insert_at(node: ast.AST) -> int:
    """0-based line index just before decorators (if any) or the class."""
    if node.decorator_list:
        return min(decorator.lineno for decorator in node.decorator_list) - 1
    return node.lineno - 1


def _module_page_href(filename: str, html_dir: Path) -> str:
    page = html_dir / f"{_module_html_stem(_short_path(filename))}.html"
    return page.resolve().as_uri()


def _where_used_comment_block(indent: str, href: str) -> list[str]:
    """Blank line, one ``# Where used:`` comment, blank line."""
    return ["\n", f"{indent}{WHERE_USED_LINK_PREFIX}{href}\n", "\n"]


def _module_docstring_end_index(tree: ast.Module) -> int | None:
    """0-based index of the line after the module docstring, or ``None``."""
    if not tree.body:
        return None
    first = tree.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.end_lineno
    return None


def _file_has_users(filename: str, maps: WhereUsedMaps) -> bool:
    return any(
        entity.filename == filename and maps.has_users(entity)
        for entity in maps.exact
    )


def _collect_insertions(
    tree: ast.AST,
    filename: str,
    lines: list[str],
    maps: WhereUsedMaps,
    html_dir: Path,
) -> list[tuple[int, list[str]]]:
    """Insert links after the module docstring and before used classes only."""
    insertions: list[tuple[int, list[str]]] = []

    class Finder(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef):
            entity = CallableEntity(
                filename, class_name=node.name, function_name=None
            )
            if entity in maps and maps.has_users(entity):
                insert_at = _definition_insert_at(node)
                class_line = lines[node.lineno - 1]
                indent = class_line[
                    : len(class_line) - len(class_line.lstrip())
                ]
                insertions.append(
                    (
                        insert_at,
                        _where_used_comment_block(
                            indent, _entity_html_href(entity, html_dir)
                        ),
                    )
                )
            self.generic_visit(node)

    Finder().visit(tree)

    if _file_has_users(filename, maps):
        docstring_end = _module_docstring_end_index(tree)
        if docstring_end is not None:
            insertions.append(
                (
                    docstring_end,
                    _where_used_comment_block(
                        "", _module_page_href(filename, html_dir)
                    ),
                )
            )
    return insertions


def _build_file_where_used(
    filename: str,
    maps: WhereUsedMaps,
    html_dir: Path,
) -> str | None:
    """Return rewritten file text, or ``None`` if the file cannot be parsed."""
    path = Path(filename)
    original = path.read_text(encoding="utf-8")
    cleaned = _strip_where_used_blocks(original)
    try:
        tree = ast.parse(cleaned, filename=filename)
    except SyntaxError as exc:
        print(f"where_used comments: skip (SyntaxError): {filename}: {exc}")
        return None
    lines = cleaned.splitlines(keepends=True)
    insertions = _collect_insertions(tree, filename, lines, maps, html_dir)
    if not insertions:
        return cleaned
    insertions.sort(key=lambda item: item[0], reverse=True)
    for insert_at, block in insertions:
        lines[insert_at:insert_at] = block
    return "".join(lines)


def _rewrite_file_where_used(
    filename: str,
    maps: WhereUsedMaps,
    html_dir: Path,
) -> bool:
    """Strip and reinsert where-used link comments. True if changed."""
    path = Path(filename)
    original = path.read_text(encoding="utf-8")
    updated = _build_file_where_used(filename, maps, html_dir)
    if updated is None:
        return False
    if updated != original:
        path.write_text(updated, encoding="utf-8")
        return True
    return False


def _files_with_users(maps: WhereUsedMaps) -> list[str]:
    by_file: dict[str, list[CallableEntity]] = defaultdict(list)
    for entity in maps.exact:
        by_file[entity.filename].append(entity)
    return sorted(
        filename
        for filename, entities in by_file.items()
        if any(maps.has_users(entity) for entity in entities)
    )


def _caller_list_html(
    users: list[CallableEntity], css_class: str, heading: str
) -> str:
    if not users:
        return ""
    ordered = sorted(users, key=_user_sort_key)
    items = "".join(
        f"<li class='{css_class}'><code>{_short_path(user.filename)}</code>, "
        f"<code>{_entity_label(user)}</code></li>\n"
        for user in ordered
    )
    return (
        f"<h3 class='{css_class}'>{heading} ({len(ordered)})</h3>\n"
        f"<ul>\n{items}</ul>\n"
    )


def write_where_used_pages(
    maps: WhereUsedMaps,
    html_dir: str | Path = DRY_RUN_HTML_DIR,
    write_pyvis: bool = True,
    pyvis_max_nodes: int = 400,
) -> Path:
    """Write one HTML page per module (anchors per entity) plus index.html.

    Exact callers are green; guessed property readers are amber.
    """
    out_dir = Path(html_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_module: dict[str, list[CallableEntity]] = defaultdict(list)
    for entity in maps.exact:
        if not maps.has_users(entity):
            continue
        by_module[_short_path(entity.filename)].append(entity)

    index_items: list[str] = []
    for module_path in sorted(by_module):
        entities = sorted(
            by_module[module_path],
            key=lambda entity: _entity_label(entity),
        )
        page_name = f"{_module_html_stem(module_path)}.html"
        page_path = out_dir / page_name
        stem = _module_html_stem(module_path)
        graph_name = f"{stem}_graph.html"
        graph_link = ""
        if write_pyvis:
            write_where_used_pyvis(
                maps,
                output_path=out_dir / graph_name,
                path_contains=module_path,
                direction="up",
                max_nodes=pyvis_max_nodes,
                use_dry_run_prefix=False,
            )
            graph_link = (
                f"<p><a href='{graph_name}'>full module graph</a> "
                f"(green=exact, amber=guess). "
                f"Click an entity title for a focused callers graph.</p>\n"
            )
        sections: list[str] = []
        for entity in entities:
            exact_users = maps.exact[entity]
            guess_users = maps.guess[entity]
            label = _entity_label(entity)
            if write_pyvis:
                focus_graph_name = (
                    f"{stem}__{_entity_anchor(entity)}_graph.html"
                )
                write_where_used_pyvis(
                    maps,
                    output_path=out_dir / focus_graph_name,
                    seed_entity=entity,
                    direction="up",
                    max_nodes=pyvis_max_nodes,
                    use_dry_run_prefix=False,
                )
                focus_href = (
                    f"{focus_graph_name}#focus="
                    f"{quote(_entity_node_id(entity), safe='')}"
                )
                heading = (
                    f'<h2><a class="focus-link" href="{focus_href}">'
                    f"{label}</a></h2>\n"
                )
            else:
                heading = f"<h2>{label}</h2>\n"
            sections.append(
                f'<section id="{_entity_anchor(entity)}">\n'
                + heading
                + _caller_list_html(exact_users, "exact", "Exact callers")
                + _caller_list_html(
                    guess_users, "guess", "Guessed property readers"
                )
                + "</section>\n"
            )
        page_html = (
            "<!DOCTYPE html>\n<html><head><meta charset='utf-8'/>"
            f"<title>where-used: {module_path}</title>\n"
            "<style>"
            "body{font-family:sans-serif;max-width:900px;margin:24px auto;}"
            "code{font-size:0.95em} section{border-top:1px solid #ddd;padding:12px 0}"
            "h1{font-size:1.2rem}"
            "a.focus-link{color:#111;text-decoration:none;border-bottom:1px dotted #1565c0}"
            "a.focus-link:hover{color:#1565c0;border-bottom-style:solid}"
            f".exact{{color:{COLOR_EXACT}}} .guess{{color:{COLOR_GUESS}}}"
            "li.exact{{color:#111}} li.guess{{color:#111}}"
            "li.exact code{{background:#e8f5e9}} li.guess code{{background:#fff3e0}}"
            "</style>\n</head><body>\n"
            f"<p><a href='index.html'>index</a></p>\n"
            f"<h1>{module_path}</h1>\n"
            f"<p><span class='exact'>Exact</span> = proven call / "
            f"<code>self.prop</code>; "
            f"<span class='guess'>Guess</span> = other "
            f"<code>.prop</code> reads sharing the property name.</p>\n"
            + graph_link
            + "".join(sections)
            + "</body></html>\n"
        )
        page_path.write_text(page_html, encoding="utf-8")
        index_items.append(
            f"<li><a href='{page_name}'><code>{module_path}</code></a> "
            f"({len(entities)})"
            + (f" · <a href='{graph_name}'>graph</a>" if write_pyvis else "")
            + "</li>\n"
        )

    index_path = out_dir / "index.html"
    index_path.write_text(
        "<!DOCTYPE html>\n<html><head><meta charset='utf-8'/>"
        "<title>where-used index</title>\n"
        "<style>body{font-family:sans-serif;max-width:900px;margin:24px auto}"
        f".exact{{color:{COLOR_EXACT}}} .guess{{color:{COLOR_GUESS}}}"
        "</style></head><body>\n"
        "<h1>where-used</h1>\n"
        f"<p><span class='exact'>Exact</span> / "
        f"<span class='guess'>Guessed</span> property readers</p>\n"
        "<ul>\n" + "".join(index_items) + "</ul>\n</body></html>\n",
        encoding="utf-8",
    )
    print(f"wrote {len(by_module)} module page(s) under {out_dir}")
    return out_dir


def _all_enums_path(start_path: str | Path) -> Path:
    return Path(start_path) / "simetri" / "base" / "all_enums.py"


def _is_strenum_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "StrEnum":
            return True
    return False


def _collect_strenum_members(all_enums_path: Path) -> dict[str, list[str]]:
    """Definition-order member names for each ``StrEnum`` class."""
    tree = ast.parse(
        all_enums_path.read_text(encoding="utf-8"), filename=str(all_enums_path)
    )
    members_by_class: dict[str, list[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or not _is_strenum_class(node):
            continue
        members: list[str] = []
        for statement in node.body:
            target = None
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
            ):
                target = statement.targets[0]
            elif isinstance(statement, ast.AnnAssign):
                target = statement.target
            if isinstance(target, ast.Name) and not (
                target.id.startswith("__") and target.id.endswith("__")
            ):
                members.append(target.id)
        members_by_class[node.name] = members
    return members_by_class


def _enum_user_label(class_name: str | None, function_name: str | None) -> str:
    if class_name and function_name:
        return f"{class_name}.{function_name}"
    if function_name:
        return function_name
    if class_name:
        return class_name
    return "<module>"


def _find_enum_member_uses(
    start_path: str | Path,
    exclude: tuple[str, ...] | list[str],
    members_by_class: dict[str, list[str]],
) -> dict[tuple[str, str], list[tuple[str, str, int]]]:
    """Map ``(enum class, member)`` to ``(short path, user label, line)`` uses.

    A use is a bare ``Class.MEMBER`` load, not the member assignment itself.
    """
    member_names = {
        class_name: set(members)
        for class_name, members in members_by_class.items()
    }
    uses: dict[tuple[str, str], list[tuple[str, str, int]]] = {
        (class_name, member): []
        for class_name, members in members_by_class.items()
        for member in members
    }

    class Finder(ast.NodeVisitor):
        def __init__(self, filename: str):
            self.filename = filename
            self.class_stack: list[str] = []
            self.func_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef):
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef):
            self.func_stack.append(node.name)
            self.generic_visit(node)
            self.func_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            self.func_stack.append(node.name)
            self.generic_visit(node)
            self.func_stack.pop()

        def visit_Attribute(self, node: ast.Attribute):
            if (
                isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name)
                and node.value.id in member_names
                and node.attr in member_names[node.value.id]
            ):
                class_name = self.class_stack[-1] if self.class_stack else None
                function_name = self.func_stack[-1] if self.func_stack else None
                user = (
                    _short_path(self.filename),
                    _enum_user_label(class_name, function_name),
                    node.lineno,
                )
                key = (node.value.id, node.attr)
                if user not in uses[key]:
                    uses[key].append(user)
            self.generic_visit(node)

    root = Path(start_path)
    for path in _iter_python_files(root, exclude):
        filename = str(path).replace("\\", "/")
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as exc:
            print(f"enum where-used: skip (SyntaxError): {filename}: {exc}")
            continue
        Finder(filename).visit(tree)
    for users in uses.values():
        users.sort(key=lambda item: (item[0], item[1], item[2]))
    return uses


def _enum_member_graph(
    class_name: str,
    members: list[str],
    uses: dict[tuple[str, str], list[tuple[str, str, int]]],
) -> nx.DiGraph:
    """User → member edges for one enum class."""
    graph = nx.DiGraph()
    for member in members:
        member_id = f"enum::{class_name}.{member}"
        graph.add_node(
            member_id,
            label=member,
            title=f"{class_name}.{member}",
            group="enum-member",
            color=COLOR_ENUM_MEMBER,
        )
        for path, label, lineno in uses[(class_name, member)]:
            user_id = f"user::{path}::{label}"
            if user_id not in graph:
                graph.add_node(
                    user_id,
                    label=label,
                    title=f"{label}\n{path}:{lineno}",
                    group="enum-user",
                    color=COLOR_ENUM_USER,
                )
            graph.add_edge(
                user_id,
                member_id,
                kind="exact",
                color=COLOR_EXACT,
            )
    return graph


def _enum_page_style() -> str:
    return (
        "<style>"
        "body{font-family:sans-serif;max-width:900px;margin:24px auto;}"
        "code{font-size:0.95em} section{border-top:1px solid #ddd;padding:12px 0}"
        "h1{font-size:1.2rem}"
        f".exact{{color:{COLOR_EXACT}}}"
        "li code{background:#e8f5e9}"
        f".swatch-member{{background:{COLOR_ENUM_MEMBER};color:#fff;padding:1px 8px}}"
        f".swatch-user{{background:{COLOR_ENUM_USER};padding:1px 8px}}"
        "</style>\n"
    )


def _enum_color_legend() -> str:
    return (
        "<p><span class='swatch-member'>blue</span> enum member. "
        "<span class='swatch-user'>yellow</span> code that uses "
        "<code>Class.MEMBER</code>. "
        "The arrow points at the member.</p>\n"
    )


def write_enum_where_used_pages(
    start_path: str | Path = DRY_RUN_START_PATH,
    exclude: tuple[str, ...] | list[str] = DEFAULT_EXCLUDE,
    html_dir: str | Path = DRY_RUN_HTML_DIR,
) -> Path:
    """Write enum class list, member where-used pages, and one graph per class.

    ``enums.html`` lists classes only. Each class page lists every member and
    the ``Class.MEMBER`` loads that use it.
    """
    out_dir = Path(html_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_enums_path = _all_enums_path(start_path)
    members_by_class = _collect_strenum_members(all_enums_path)
    uses = _find_enum_member_uses(start_path, exclude, members_by_class)

    class_items: list[str] = []
    for class_name in sorted(members_by_class):
        members = members_by_class[class_name]
        page_name = f"enum_{class_name}.html"
        graph_name = f"enum_{class_name}_graph.html"
        graph = _enum_member_graph(class_name, members, uses)
        graph_path = out_dir / graph_name
        _write_pinned_column_pyvis(graph, graph_path)
        _insert_html_after_body(
            graph_path,
            "<div style='font-family:sans-serif;padding:8px 12px;"
            "background:#f5f5f5;border-bottom:1px solid #ddd'>"
            f"<span class='swatch-member' style='background:{COLOR_ENUM_MEMBER};"
            "color:#fff;padding:1px 8px'>blue</span> enum member. "
            f"<span class='swatch-user' style='background:{COLOR_ENUM_USER};"
            "padding:1px 8px'>yellow</span> code that uses "
            "<code>Class.MEMBER</code>. The arrow points at the member."
            "</div>\n",
        )
        print(
            f"wrote enum graph {graph_name} "
            f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
        )
        sections: list[str] = []
        for member in members:
            member_uses = uses[(class_name, member)]
            if member_uses:
                items = "".join(
                    f"<li class='exact'><code>{html.escape(path)}</code>, "
                    f"<code>{html.escape(label)}</code> "
                    f"(line {lineno})</li>\n"
                    for path, label, lineno in member_uses
                )
                body = (
                    f"<h3 class='exact'>Uses ({len(member_uses)})</h3>\n"
                    f"<ul>\n{items}</ul>\n"
                )
            else:
                body = "<p>No uses.</p>\n"
            sections.append(
                f'<section id="{html.escape(member)}">\n'
                f"<h2><code>{html.escape(class_name)}.{html.escape(member)}</code></h2>\n"
                + body
                + "</section>\n"
            )
        page_html = (
            "<!DOCTYPE html>\n<html><head><meta charset='utf-8'/>"
            f"<title>where-used: {html.escape(class_name)}</title>\n"
            + _enum_page_style()
            + "</head><body>\n"
            "<p><a href='index.html'>index</a> · "
            "<a href='enums.html'>enums</a></p>\n"
            f"<h1>{html.escape(class_name)}</h1>\n"
            f"<p><a href='{graph_name}'>full enum graph</a></p>\n"
            + _enum_color_legend()
            + "".join(sections)
            + "</body></html>\n"
        )
        (out_dir / page_name).write_text(page_html, encoding="utf-8")
        class_items.append(
            f"<li><a href='{page_name}'><code>{html.escape(class_name)}</code></a> "
            f"({len(members)})</li>\n"
        )

    enums_html = (
        "<!DOCTYPE html>\n<html><head><meta charset='utf-8'/>"
        "<title>where-used: enums</title>\n"
        + _enum_page_style()
        + "</head><body>\n"
        "<p><a href='index.html'>index</a></p>\n"
        "<h1>enums</h1>\n"
        "<p>Classes only. Open a class for member where-used.</p>\n"
        + _enum_color_legend()
        + "<ul>\n"
        + "".join(class_items)
        + "</ul>\n</body></html>\n"
    )
    (out_dir / "enums.html").write_text(enums_html, encoding="utf-8")

    index_path = out_dir / "index.html"
    index_text = index_path.read_text(encoding="utf-8")
    marker = "<h1>where-used</h1>\n"
    link = (
        "<p><a href='enums.html'>enums</a> "
        "(classes only; open a class for member where-used)</p>\n"
    )
    if "href='enums.html'" not in index_text:
        if marker not in index_text:
            raise RuntimeError(
                f"where-used index missing heading: {index_path}"
            )
        index_text = index_text.replace(marker, marker + link, 1)
        index_path.write_text(index_text, encoding="utf-8")
    print(f"wrote {len(members_by_class)} enum page(s) under {out_dir}")
    return out_dir


def dry_run() -> WhereUsedMaps:
    """Scan the D: backup, write HTML/graphs, and insert module/class links.

    Comments are only ``# Where used:`` after a module docstring and before
    a class that has users. Functions and methods are not annotated.
    ``gui`` and ``tests`` are excluded. Enum member pages are written too.
    """
    _use_dry_run_paths()
    maps = get_where_used(
        start_path=DRY_RUN_START_PATH, exclude=DEFAULT_EXCLUDE
    )
    save_where_used(maps, DRY_RUN_CACHE_PATH)
    populate_comments(
        start_path=DRY_RUN_START_PATH,
        exclude=DEFAULT_EXCLUDE,
        d_where_used=maps,
        html_dir=DRY_RUN_HTML_DIR,
    )
    write_enum_where_used_pages(
        start_path=DRY_RUN_START_PATH,
        exclude=DEFAULT_EXCLUDE,
        html_dir=DRY_RUN_HTML_DIR,
    )
    return maps


def check_dry_run(maps: WhereUsedMaps) -> str:
    """Compare backup-tree link comments to ``maps``. Return Pass/Fail."""
    _use_dry_run_paths()
    failures: list[str] = []
    for filename in _files_with_users(maps):
        expected = _build_file_where_used(filename, maps, DRY_RUN_HTML_DIR)
        if expected is None:
            failures.append(filename)
            continue
        actual = Path(filename).read_text(encoding="utf-8")
        if actual != expected:
            failures.append(filename)
    if failures:
        print(f"check_dry_run Fail: {len(failures)} file(s) mismatch")
        for filename in failures[:20]:
            print(f"  {filename}")
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} more")
        return "Fail"
    print("check_dry_run Pass")
    return "Pass"


def check_updates() -> str:
    """Add a dummy caller, run ``update_comments``, verify, then clean up.

    Requires ``dry_run()`` to have been run first (cache under the backup).
    """
    _use_dry_run_paths()
    probe_path = Path(DRY_RUN_START_PATH) / _DRY_RUN_PROBE_RELATIVE
    probe_filename = str(probe_path).replace("\\", "/")
    probe_source = (
        "def where_used_dry_run_probe_caller():\n"
        f"    {_DRY_RUN_PROBE_CALLEE}(0, 0, 0)\n"
    )
    try:
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        probe_path.write_text(probe_source, encoding="utf-8")
        update_comments(
            start_path=DRY_RUN_START_PATH,
            exclude=DEFAULT_EXCLUDE,
            cache_path=DRY_RUN_CACHE_PATH,
            html_dir=DRY_RUN_HTML_DIR,
        )
        maps = load_where_used(DRY_RUN_CACHE_PATH)
        callee_keys = [
            entity
            for entity in maps.exact
            if entity.function_name == _DRY_RUN_PROBE_CALLEE
            and entity.filename.replace("\\", "/").endswith("geom/geometry.py")
        ]
        if len(callee_keys) != 1:
            print(
                "check_updates Fail: "
                f"expected one {_DRY_RUN_PROBE_CALLEE} in geometry.py, "
                f"got {len(callee_keys)}"
            )
            return "Fail"
        callee = callee_keys[0]
        probe_user = CallableEntity(
            probe_filename,
            class_name=None,
            function_name="where_used_dry_run_probe_caller",
        )
        if probe_user not in maps.exact[callee]:
            print(
                "check_updates Fail: probe caller missing from "
                f"{_DRY_RUN_PROBE_CALLEE} exact users"
            )
            return "Fail"
        expected_href = _module_page_href(callee.filename, DRY_RUN_HTML_DIR)
        expected_line = f"{WHERE_USED_LINK_PREFIX}{expected_href}"
        geometry_text = Path(callee.filename).read_text(encoding="utf-8")
        if expected_line not in geometry_text:
            print(
                "check_updates Fail: geometry.py missing module link "
                f"{expected_line!r}"
            )
            return "Fail"
        page_text = _module_html_path(callee, DRY_RUN_HTML_DIR).read_text(
            encoding="utf-8"
        )
        if "where_used_dry_run_probe_caller" not in page_text:
            print("check_updates Fail: HTML page missing probe caller")
            return "Fail"
        if check_dry_run(maps) != "Pass":
            print("check_updates Fail: comments do not match updated maps")
            return "Fail"
        print("check_updates Pass")
        return "Pass"
    finally:
        if probe_path.is_file():
            probe_path.unlink()
        update_comments(
            start_path=DRY_RUN_START_PATH,
            exclude=DEFAULT_EXCLUDE,
            cache_path=DRY_RUN_CACHE_PATH,
            html_dir=DRY_RUN_HTML_DIR,
        )


def populate_comments(
    start_path: str = DEFAULT_START_PATH,
    exclude: tuple[str, ...] | list[str] = DEFAULT_EXCLUDE,
    d_where_used: WhereUsedMaps | None = None,
    html_dir: str | Path = DRY_RUN_HTML_DIR,
) -> int:
    """Insert ``# Where used:`` after the module docstring and before used classes.

    No comments on functions or methods. Writes/refreshes HTML pages first.
    """
    html_path = Path(html_dir)
    if d_where_used is None:
        maps = get_where_used(start_path=start_path, exclude=exclude)
    else:
        maps = d_where_used
    write_where_used_pages(maps, html_path)

    by_file: dict[str, list[CallableEntity]] = defaultdict(list)
    for entity in maps.exact:
        by_file[entity.filename].append(entity)

    files_changed = 0
    for filename, entities in sorted(by_file.items()):
        if not any(maps.has_users(entity) for entity in entities):
            continue
        if _rewrite_file_where_used(filename, maps, html_path):
            files_changed += 1

    return files_changed


def update_comments(
    start_path: str = DEFAULT_START_PATH,
    exclude: tuple[str, ...] | list[str] = DEFAULT_EXCLUDE,
    d_where_used: WhereUsedMaps | None = None,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    html_dir: str | Path = DRY_RUN_HTML_DIR,
) -> int:
    """Rerun get_where_used, refresh HTML, update link comments in changed files."""
    html_path = Path(html_dir)
    if d_where_used is None:
        new_maps = get_where_used(start_path=start_path, exclude=exclude)
    else:
        new_maps = d_where_used
    new_maps.sort_all()

    old_maps = WhereUsedMaps()
    cache_file = Path(cache_path)
    if cache_file.is_file():
        old_maps = load_where_used(cache_file)

    changed_files: set[str] = set()
    for entity in new_maps.exact:
        old_exact = []
        if entity in old_maps.exact:
            old_exact = old_maps.exact[entity]
        old_guess = []
        if entity in old_maps.guess:
            old_guess = old_maps.guess[entity]
        if _users_changed(old_exact, new_maps.exact[entity]) or _users_changed(
            old_guess, new_maps.guess[entity]
        ):
            changed_files.add(entity.filename)

    for entity in old_maps.exact:
        if entity not in new_maps.exact and (
            old_maps.exact[entity] or old_maps.guess[entity]
        ):
            changed_files.add(entity.filename)

    write_where_used_pages(new_maps, html_path)

    files_changed = 0
    for filename in sorted(changed_files):
        if _rewrite_file_where_used(filename, new_maps, html_path):
            files_changed += 1

    save_where_used(new_maps, cache_path)
    return files_changed


def _is_dunder_method(entity: CallableEntity) -> bool:
    """True for ``__init__`` / ``__str__`` and other dunder methods."""
    name = entity.function_name
    return (
        name is not None
        and len(name) > 4
        and name.startswith("__")
        and name.endswith("__")
    )


def _entity_node_id(entity: CallableEntity) -> str:
    return f"{_short_path(entity.filename)}::{_entity_label(entity)}"


def _entity_node_title(entity: CallableEntity) -> str:
    return f"{_entity_label(entity)}\n{_short_path(entity.filename)}"


def build_where_used_digraph(
    maps: WhereUsedMaps,
    function_name: str | None = None,
    path_contains: str | None = None,
    seed_entity: CallableEntity | None = None,
    direction: str = "both",
    max_nodes: int = 400,
) -> nx.DiGraph:
    """Build a caller→callee digraph from exact + guessed where-used maps.

    Edge attributes: ``kind`` (``\"exact\"`` / ``\"guess\"``) and ``color``.
    If ``seed_entity`` is set, that entity alone is the seed (for focus graphs).
    Dunder methods are omitted, except as the focused seed.
    """
    if direction not in ("up", "down", "both"):
        raise ValueError(f"direction must be up/down/both, got {direction!r}")

    seeds: list[CallableEntity] = []
    if seed_entity is not None:
        seeds = [seed_entity]
    else:
        for entity in maps.exact:
            if (
                function_name is not None
                and entity.function_name != function_name
            ):
                continue
            if path_contains is not None and path_contains not in _short_path(
                entity.filename
            ):
                continue
            if function_name is None and path_contains is None:
                continue
            seeds.append(entity)

        if function_name is None and path_contains is None:
            seeds = [entity for entity in maps.exact if maps.has_users(entity)]
        if seed_entity is None:
            seeds = [
                entity for entity in seeds if not _is_dunder_method(entity)
            ]

    callers_of: dict[CallableEntity, list[CallableEntity]] = defaultdict(list)
    callees_of: dict[CallableEntity, list[CallableEntity]] = defaultdict(list)
    edge_kind: dict[tuple[CallableEntity, CallableEntity], str] = {}

    for callee, callers in maps.exact.items():
        for caller in callers:
            callers_of[callee].append(caller)
            callees_of[caller].append(callee)
            edge_kind[(caller, callee)] = "exact"
    for callee, callers in maps.guess.items():
        for caller in callers:
            if (caller, callee) in edge_kind:
                continue
            callers_of[callee].append(caller)
            callees_of[caller].append(callee)
            edge_kind[(caller, callee)] = "guess"

    include: set[CallableEntity] = set(seeds)
    for seed in seeds:
        if direction in ("up", "both"):
            include.update(callers_of[seed])
        if direction in ("down", "both"):
            include.update(callees_of[seed])
    include = {
        entity
        for entity in include
        if not _is_dunder_method(entity) or entity == seed_entity
    }

    if len(include) > max_nodes:
        print(
            f"where_used graph: truncating {len(include)} nodes to {max_nodes}"
        )
        # Prefer seeds, then their callers, so focus graphs stay useful.
        seed_set = set(seeds)
        neighbor_set: set[CallableEntity] = set()
        for seed in seeds:
            neighbor_set.update(callers_of[seed])
            neighbor_set.update(callees_of[seed])
        neighbor_set -= seed_set
        kept: list[CallableEntity] = sorted(seed_set, key=_entity_node_id)
        if len(kept) > max_nodes:
            kept = kept[:max_nodes]
        else:
            kept.extend(
                sorted(neighbor_set, key=_entity_node_id)[
                    : max_nodes - len(kept)
                ]
            )
            if len(kept) < max_nodes:
                rest = sorted(include - set(kept), key=_entity_node_id)
                kept.extend(rest[: max_nodes - len(kept)])
        include = set(kept)

    graph = nx.DiGraph()
    for entity in include:
        graph.add_node(
            _entity_node_id(entity),
            label=_entity_label(entity),
            title=_entity_node_title(entity),
            path=_short_path(entity.filename),
            group=_short_path(entity.filename).split("/")[0]
            if "/" in _short_path(entity.filename)
            else _short_path(entity.filename),
        )

    for (caller, callee), kind in edge_kind.items():
        if caller not in include or callee not in include:
            continue
        color = COLOR_EXACT if kind == "exact" else COLOR_GUESS
        graph.add_edge(
            _entity_node_id(caller),
            _entity_node_id(callee),
            kind=kind,
            color=color,
            title=kind,
        )

    return graph


def _center_out_order(nodes: list[str]) -> list[str]:
    """Top-to-bottom order with the busiest node in the vertical middle."""
    count = len(nodes)
    if count == 0:
        return []
    slots: list[str | None] = [None] * count
    middle = count // 2
    slots[middle] = nodes[0]
    above = middle - 1
    below = middle + 1
    place_above = True
    for node in nodes[1:]:
        if place_above and above >= 0:
            slots[above] = node
            above -= 1
            place_above = False
        elif below < count:
            slots[below] = node
            below += 1
            place_above = True
        else:
            slots[above] = node
            above -= 1
    return [node for node in slots if node is not None]


def _column_layout_positions(
    graph: nx.DiGraph,
) -> dict[str, tuple[float, float]]:
    """Place nodes in columns by link count, busiest in the center.

    Horizontal: the 2–3 highest-degree nodes form the center column. The next
    busiest fill columns of five immediately left and right, then further out.
    Degree-zero nodes sit to the right, alphabetically, at most ten per column.

    Vertical: within each linked column the busiest node is in the middle;
    quieter nodes sit toward the top and bottom. Gaps grow with the busier
    neighbor, so busy nodes have more room than quiet ones. Columns an odd
    number of steps from the center are shifted up by about 1.5 node heights.
    Degree-zero nodes use a fixed pitch of about twice a node height.

    Coordinates are in inches. Pyvis scales them.
    """
    degree = {node_id: graph.degree(node_id) for node_id in graph.nodes}
    isolated = sorted(
        (node_id for node_id, links in degree.items() if links == 0),
        key=lambda node_id: graph.nodes[node_id]["label"].casefold(),
    )
    linked = sorted(
        (node_id for node_id, links in degree.items() if links > 0),
        key=lambda node_id: (-degree[node_id], graph.nodes[node_id]["label"]),
    )
    if len(linked) >= 8:
        center_count = 3
    elif len(linked) >= 2:
        center_count = 2
    else:
        center_count = len(linked)
    center = linked[:center_count]
    rest = linked[center_count:]
    side_size = 5
    left_columns: list[list[str]] = []
    right_columns: list[list[str]] = []
    rest_index = 0
    place_left = True
    while rest_index < len(rest):
        chunk = rest[rest_index : rest_index + side_size]
        rest_index += side_size
        if place_left:
            left_columns.append(chunk)
            place_left = False
        else:
            right_columns.append(chunk)
            place_left = True
    columns: list[list[str]] = list(reversed(left_columns))
    center_index = len(columns) if center else None
    if center:
        columns.append(center)
    columns.extend(right_columns)
    isolated_size = 10
    for index in range(0, len(isolated), isolated_size):
        columns.append(isolated[index : index + isolated_size])

    max_links = max(degree.values()) if degree else 0

    def _gap(links: int, quiet: float, busy: float) -> float:
        if max_links <= 0:
            return quiet
        return quiet + (busy - quiet) * (links / max_links)

    column_peaks = [
        max((degree[node_id] for node_id in column_nodes), default=0)
        for column_nodes in columns
    ]
    column_x = [0.0]
    for column_index in range(1, len(columns)):
        both_isolated = (
            column_peaks[column_index - 1] == 0
            and column_peaks[column_index] == 0
        )
        if both_isolated:
            # Wide enough for the boxed labels, still tighter than linked columns.
            step = 1.9
        else:
            pair_links = max(
                column_peaks[column_index - 1], column_peaks[column_index]
            )
            # Tighter than before; wider next to busy columns.
            step = _gap(pair_links, 1.6, 2.5)
        column_x.append(column_x[-1] + step)

    # Degree-zero boxes sit about one node-height apart (pitch ≈ 2× height).
    node_height = 0.35
    isolated_y = 2 * node_height
    stagger_y = 1.5 * node_height
    positions: dict[str, tuple[float, float]] = {}
    for column_index, column_nodes in enumerate(columns):
        is_isolated = all(degree[node_id] == 0 for node_id in column_nodes)
        # Keep alphabetical order in the no-link columns.
        ordered = (
            list(column_nodes)
            if is_isolated
            else _center_out_order(column_nodes)
        )
        if not ordered:
            continue
        steps: list[float] = []
        for row in range(1, len(ordered)):
            if is_isolated:
                steps.append(isolated_y)
                continue
            pair_links = max(degree[ordered[row - 1]], degree[ordered[row]])
            # Larger than before; still larger beside a busy node.
            steps.append(_gap(pair_links, 1.8, 3.0))
        y = sum(steps) / 2
        if (
            center_index is not None
            and abs(column_index - center_index) % 2 == 1
        ):
            y += stagger_y
        x = column_x[column_index]
        positions[ordered[0]] = (x, y)
        for row, step in enumerate(steps, start=1):
            y -= step
            positions[ordered[row]] = (x, y)
    return positions


def _insert_html_after_body(html_path: Path, snippet: str) -> None:
    text = html_path.read_text(encoding="utf-8")
    needle = "<body>"
    if needle not in text:
        raise RuntimeError(f"graph html missing body tag: {html_path}")
    html_path.write_text(
        text.replace(needle, needle + "\n" + snippet, 1), encoding="utf-8"
    )


def _write_pinned_column_pyvis(graph: nx.DiGraph, output_path: Path) -> Path:
    """Write a pinned-column pyvis page. Nodes stay put until dragged."""
    network = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#111111",
        cdn_resources="remote",
    )
    network.set_options(
        """
var options = {
  "layout": { "improvedLayout": false },
  "physics": { "enabled": false },
  "nodes": {
    "shape": "box",
    "margin": 8,
    "font": { "size": 14, "face": "arial" },
    "widthConstraint": { "maximum": 140 }
  },
  "edges": {
    "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } },
    "smooth": { "type": "continuous", "roundness": 0.2 }
  },
  "interaction": {
    "hover": true,
    "dragNodes": true,
    "dragView": true,
    "tooltipDelay": 60,
    "navigationButtons": true,
    "keyboard": true
  }
}
"""
    )
    positions = _column_layout_positions(graph)
    for node_id, data in graph.nodes(data=True):
        x_inch, y_inch = positions[node_id]
        node_fields = {
            "label": data["label"],
            "title": data["title"],
            "group": data["group"],
            "shape": "box",
            "x": x_inch * 96,
            "y": -y_inch * 96,
            "physics": False,
        }
        if "color" in data:
            node_fields["color"] = data["color"]
        network.add_node(node_id, **node_fields)
    for source, target, data in graph.edges(data=True):
        network.add_edge(
            source,
            target,
            length=40,
            color=data["color"],
            title=data["kind"],
            kind=data["kind"],
        )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    network.write_html(str(out), notebook=False, open_browser=False)
    _freeze_vis_physics_after_stabilize(out)
    return out


def write_where_used_pyvis(
    d_where_used: WhereUsedMaps | None = None,
    cache_path: str | Path = DRY_RUN_CACHE_PATH,
    output_path: str | Path = DRY_RUN_GRAPH_PATH,
    function_name: str | None = None,
    path_contains: str | None = None,
    seed_entity: CallableEntity | None = None,
    direction: str = "both",
    max_nodes: int = 400,
    use_dry_run_prefix: bool = True,
) -> Path:
    """Write an interactive pyvis HTML call graph from ``WhereUsedMaps``.

    Does not rescan sources. Edges are caller → callee (green=exact, amber=guess).
    """
    if use_dry_run_prefix:
        _use_dry_run_paths()
    maps = load_where_used(cache_path) if d_where_used is None else d_where_used

    graph = build_where_used_digraph(
        maps,
        function_name=function_name,
        path_contains=path_contains,
        seed_entity=seed_entity,
        direction=direction,
        max_nodes=max_nodes,
    )

    out = _write_pinned_column_pyvis(graph, Path(output_path))
    if seed_entity is None:
        print(
            f"wrote pyvis {out} "
            f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
        )
    return out


def _freeze_vis_physics_after_stabilize(html_path: Path) -> None:
    """Disable physics after layout and add double-click focus behavior."""
    text = html_path.read_text(encoding="utf-8")
    if "whereUsedFocusOnDoubleClick" in text:
        return
    needle = "network = new vis.Network(container, data, options);"
    hook = (
        "network = new vis.Network(container, data, options);\n"
        "                  network.once("
        "'stabilizationIterationsDone', function () {\n"
        "                    network.setOptions("
        "{ physics: { enabled: false } });\n"
        "                  });\n"
        "                  whereUsedFocusOnDoubleClick(network, nodes, edges);"
    )
    if needle not in text:
        return
    text = text.replace(needle, hook, 1)
    draw_marker = "function drawGraph() {"
    if draw_marker in text:
        text = text.replace(
            draw_marker,
            _WHERE_USED_FOCUS_JS + "\n              " + draw_marker,
            1,
        )
    html_path.write_text(text, encoding="utf-8")


# Injected into graph HTML: double-click a node (or open #focus=nodeId) to
# center it with callers around it. Exact=green inner ring; guess=amber outer.
_WHERE_USED_FOCUS_JS = f"""
function whereUsedFocusOnDoubleClick(network, nodes, edges) {{
  var COLOR_EXACT = "{COLOR_EXACT}";
  var COLOR_GUESS = "{COLOR_GUESS}";
  var COLOR_FOCUS = "{COLOR_FOCUS}";
  var originalNodeColor = {{}};
  nodes.forEach(function (node) {{
    originalNodeColor[node.id] = node.color;
  }});

  function restoreAll() {{
    var restoreNodes = [];
    nodes.forEach(function (node) {{
      restoreNodes.push({{
        id: node.id,
        hidden: false,
        color: originalNodeColor[node.id]
      }});
    }});
    nodes.update(restoreNodes);
    network.fit({{ animation: true }});
  }}

  function applyFocus(focusId) {{
    if (!nodes.get(focusId)) {{
      window.alert("Focus node not in this graph: " + focusId);
      return;
    }}
    var incoming = edges.get({{
      filter: function (edge) {{
        return edge.to === focusId;
      }}
    }});
    var exactCallers = [];
    var guessCallers = [];
    for (var i = 0; i < incoming.length; i++) {{
      var edge = incoming[i];
      if (edge.kind === "guess" || edge.color === COLOR_GUESS) {{
        guessCallers.push(edge.from);
      }} else {{
        exactCallers.push(edge.from);
      }}
    }}
    var keep = {{}};
    keep[focusId] = true;
    for (var ei = 0; ei < exactCallers.length; ei++) {{
      keep[exactCallers[ei]] = true;
    }}
    for (var gi = 0; gi < guessCallers.length; gi++) {{
      keep[guessCallers[gi]] = true;
    }}
    var updates = [];
    nodes.forEach(function (node) {{
      var color = originalNodeColor[node.id];
      if (node.id === focusId) {{
        color = COLOR_FOCUS;
      }} else if (keep[node.id]) {{
        if (exactCallers.indexOf(node.id) >= 0) {{
          color = COLOR_EXACT;
        }} else if (guessCallers.indexOf(node.id) >= 0) {{
          color = COLOR_GUESS;
        }}
      }}
      updates.push({{ id: node.id, hidden: !keep[node.id], color: color }});
    }});
    nodes.update(updates);
    network.setOptions({{ physics: {{ enabled: false }} }});
    network.moveNode(focusId, 0, 0);
    function placeRing(callers, radius) {{
      var count = callers.length;
      for (var j = 0; j < count; j++) {{
        var angle = (2 * Math.PI * j) / Math.max(count, 1) - Math.PI / 2;
        network.moveNode(
          callers[j],
          radius * Math.cos(angle),
          radius * Math.sin(angle)
        );
      }}
    }}
    var exactCount = exactCallers.length;
    var guessCount = guessCallers.length;
    var innerRadius = Math.max(140, 50 + exactCount * 14);
    placeRing(exactCallers, innerRadius);
    if (guessCount) {{
      var outerRadius = innerRadius + Math.max(100, 40 + guessCount * 10);
      placeRing(guessCallers, outerRadius);
    }}
    network.fit({{
      nodes: Object.keys(keep),
      animation: true
    }});
  }}

  function focusIdFromHash() {{
    var hash = window.location.hash || "";
    if (hash.indexOf("#focus=") !== 0) {{
      return null;
    }}
    return decodeURIComponent(hash.slice("#focus=".length));
  }}

  function tryFocusFromHash() {{
    var focusId = focusIdFromHash();
    if (focusId) {{
      applyFocus(focusId);
    }}
  }}

  network.on("doubleClick", function (params) {{
    if (!params.nodes.length) {{
      restoreAll();
      return;
    }}
    applyFocus(params.nodes[0]);
  }});

  network.once("stabilizationIterationsDone", function () {{
    tryFocusFromHash();
  }});
  window.addEventListener("hashchange", function () {{
    var focusId = focusIdFromHash();
    if (focusId) {{
      applyFocus(focusId);
    }} else {{
      restoreAll();
    }}
  }});
}}
"""


def write_where_used_html(
    d_where_used: WhereUsedMaps | None = None,
    cache_path: str | Path = DRY_RUN_CACHE_PATH,
    output_path: str | Path = DRY_RUN_GRAPH_PATH,
    function_name: str | None = None,
    path_contains: str | None = None,
    direction: str = "both",
    max_nodes: int = 400,
    use_dry_run_prefix: bool = True,
) -> Path:
    """Write an interactive HTML call graph (vis-network) from the pickle.

    Does not rescan sources. Defaults to the D: dry-run cache/output paths.
    Edges are caller → callee (green=exact, amber=guess).
    """
    if use_dry_run_prefix:
        _use_dry_run_paths()
    maps = load_where_used(cache_path) if d_where_used is None else d_where_used

    graph = build_where_used_digraph(
        maps,
        function_name=function_name,
        path_contains=path_contains,
        direction=direction,
        max_nodes=max_nodes,
    )

    nodes = []
    for node_id, data in graph.nodes(data=True):
        nodes.append(
            {
                "id": node_id,
                "label": data["label"],
                "title": data["title"],
                "group": data["group"],
            }
        )
    edges = [
        {
            "from": source,
            "to": target,
            "arrows": "to",
            "color": data["color"],
            "kind": data["kind"],
            "title": data["kind"],
        }
        for source, target, data in graph.edges(data=True)
    ]

    title_bits = []
    if function_name is not None:
        title_bits.append(function_name)
    if path_contains is not None:
        title_bits.append(path_contains)
    page_title = "where_used: " + (
        ", ".join(title_bits) if title_bits else "graph"
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{page_title}</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    html, body {{ margin: 0; height: 100%; font-family: sans-serif; }}
    #header {{ padding: 8px 12px; background: #111; color: #eee; }}
    #network {{ width: 100%; height: calc(100% - 40px); }}
  </style>
</head>
<body>
  <div id="header">{page_title} — {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges (green=exact, amber=guess)</div>
  <div id="network"></div>
  <script>
    const nodes = new vis.DataSet({json.dumps(nodes)});
    const edges = new vis.DataSet({json.dumps(edges)});
    const container = document.getElementById("network");
    const data = {{ nodes, edges }};
    const options = {{
      physics: {{
        enabled: true,
        forceAtlas2Based: {{
          gravitationalConstant: -25,
          centralGravity: 0.05,
          springLength: 40,
          springConstant: 0.2,
          damping: 0.5,
          avoidOverlap: 1
        }},
        solver: "forceAtlas2Based",
        stabilization: {{ enabled: true, iterations: 400 }}
      }},
      nodes: {{
        shape: "box",
        margin: 6,
        font: {{ size: 14 }},
        widthConstraint: {{ maximum: 140 }}
      }},
      edges: {{
        arrows: {{ to: {{ enabled: true, scaleFactor: 0.5 }} }},
        smooth: {{ type: "continuous", roundness: 0.3 }},
        length: 40
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 60,
        navigationButtons: true,
        keyboard: true
      }}
    }};
    const network = new vis.Network(container, data, options);
    network.once("stabilizationIterationsDone", function () {{
      network.setOptions({{ physics: {{ enabled: false }} }});
    }});
    whereUsedFocusOnDoubleClick(network, nodes, edges);
  </script>
  <script>
{_WHERE_USED_FOCUS_JS}
  </script>
</body>
</html>
"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(
        f"wrote {out} ({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
    )
    return out


def test():
    maps = dry_run()
    check_dry_run(maps)
    check_updates()


dry_run()
# test()
# if __name__ == "__main__":
#     where_used = get_where_used()
#     cache_file = save_where_used(where_used)
#     print(f"saved {cache_file} ({cache_file.stat().st_size} bytes)")
#     # Spot-check: callers of geometry.double_area3
#     for key in where_used.exact:
#         if key.function_name == "double_area3" and key.filename.endswith(
#             "geom/geometry.py"
#         ):
#             print(key)
#             for user in where_used.exact[key]:
#                 print("  exact <-", user)
#             for user in where_used.guess[key]:
#                 print("  guess <-", user)
#             break
#     print(f"entities: {len(where_used)}")
#     loaded = load_where_used(cache_file)
#     print(f"reloaded entities: {len(loaded)}")

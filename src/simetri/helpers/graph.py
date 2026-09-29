"""Graph related functions and classes. Uses NetworkX for graph operations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from ..base.all_enums import Types
from ..config.settings import runtime_defaults
from ..geom.geom_utils import close_points_square
from ..geom.points.point_utils import distance


@dataclass
class Node:
    """A 2D point with ``x`` and ``y`` coordinates for graph use.

    Attributes:
        x (float): X coordinate.
        y (float): Y coordinate.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.helpers.graph import Node
        >>> Node(3, 4).pos
        (3, 4)
        >>> Node(0, 0) == Node(0, 0)
        True
        >>> Node(0, 0) == Node(1, 0)
        False
    """

    x: float
    y: float

    @property
    def pos(self) -> tuple[float, float]:
        """Return the position of the node.

        Returns:
            tuple[float, float]: ``(x, y)``.

        Examples:
            >>> import simetri.graphics as sg
            >>> from simetri.helpers.graph import Node
            >>> Node(3, 4).pos
            (3, 4)
        """
        return (self.x, self.y)

    def __eq__(self, other: object) -> bool:
        """Return True when ``other`` is within ``abs_tol`` of this node.

        Args:
            other (object): Another node with a ``pos`` attribute.

        Returns:
            bool: True if the nodes are equal, False otherwise.

        Examples:
            >>> import simetri.graphics as sg
            >>> from simetri.helpers.graph import Node
            >>> Node(0, 0) == Node(0, 0)
            True
            >>> Node(0, 0) == Node(10, 0)
            False
        """
        return close_points_square(
            self.pos, other.pos, dist2=runtime_defaults["abs_tol"] ** 2
        )


@dataclass
class GraphEdge:
    """Edge in a graph with start and end nodes.

    ``start`` and ``end`` must be ``Node`` objects (they supply ``pos``).

    Attributes:
        start (Node): Start node.
        end (Node): End node.
        length (float): Edge length computed in ``__post_init__``.

    Examples:
        >>> import simetri.graphics as sg
        >>> from simetri.helpers.graph import GraphEdge, Node
        >>> edge = GraphEdge(Node(0, 0), Node(3, 4))
        >>> edge.length
        5.0
        >>> tuple(node.pos for node in edge.nodes)
        ((0, 0), (3, 4))
    """

    start: Node
    end: Node
    length: float = field(init=False)

    def __post_init__(self) -> None:
        """Compute ``length`` from the start and end node positions."""
        self.length = distance(self.start.pos, self.end.pos)

    @property
    def nodes(self) -> tuple[Node, Node]:
        """Return the start and end nodes of the edge.

        Returns:
            tuple[Node, Node]: ``(start, end)``.

        Examples:
            >>> import simetri.graphics as sg
            >>> from simetri.helpers.graph import GraphEdge, Node
            >>> edge = GraphEdge(Node(0, 0), Node(1, 0))
            >>> tuple(node.pos for node in edge.nodes)
            ((0, 0), (1, 0))
        """
        return (self.start, self.end)


def edges_to_nodes(edges: Sequence[Sequence[Any]]) -> list[Any]:
    """Return the node sequence of a connected edge list.

    Args:
        edges: List of two-endpoint edges.

    Returns:
        list: Nodes in chain order. A closed chain repeats the start node
        at the end.

    Examples:
        >>> from simetri.helpers.graph import edges_to_nodes
        >>> edges_to_nodes([(1, 2), (0, 1), (2, 3)])
        [0, 1, 2, 3]
        >>> edges_to_nodes([(0, 1), (1, 2), (2, 0)])
        [0, 1, 2, 0]
    """
    chain = longest_chain(edges)
    closed = chain[0][0] == chain[-1][-1]
    if closed:
        nodes = [x[0] for x in chain[:-1]]
    else:
        nodes = [x[0] for x in chain] + [chain[-1][1]]
    if closed:
        last_edge = chain[-1]
        if last_edge[1] == nodes[-1]:
            nodes.extend(reversed(last_edge))
        elif last_edge[0] == nodes[-1]:
            nodes.extend(last_edge)
        elif last_edge[0] == nodes[0]:
            nodes.extend(reversed(last_edge))
        elif last_edge[1] == nodes[0]:
            nodes.extend(last_edge)

    return nodes


def get_cycles(edges: Sequence[Sequence[Any]]) -> list[list[Any]] | None:
    """Return every cycle in a graph given as an edge list.

    Each cycle is a node list with the first node repeated at the end.
    Returns ``None`` when the graph has no cycle.

    Args:
        edges: Two-endpoint edges (hashable nodes).

    Returns:
        list[list] | None: Closed cycles, or ``None`` if there is none.

    Examples:
        >>> from simetri.helpers.graph import get_cycles
        >>> get_cycles([(0, 1), (1, 2), (2, 0)])
        [[1, 0, 2, 1]]
        >>> get_cycles([(0, 1), (1, 2)])
        >>> get_cycles([])
    """
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(edges)
    cycles = nx.cycle_basis(nx_graph)
    res = None
    if cycles:
        for cycle in cycles:
            cycle.append(cycle[0])

        res = cycles
    return res


# find all open paths starting from a given node
def find_all_paths(graph: nx.Graph, node: Any) -> list[list[Any]]:
    """Find all simple paths of length at least 1 starting from ``node``.

    Args:
        graph: The graph.
        node: The starting node.

    Returns:
        list[list]: Paths from ``node`` to every other reachable node.

    Examples:
        >>> import networkx as nx
        >>> from simetri.helpers.graph import find_all_paths
        >>> path = nx.path_graph(3)
        >>> find_all_paths(path, 0)
        [[0, 1], [0, 1, 2]]
        >>> find_all_paths(path, 1)
        [[1, 0], [1, 2]]
    """
    paths = []
    for node_ in graph.nodes():
        paths.extend(
            path
            for path in nx.all_simple_paths(graph, node, node_)
            if len(path) > 1
        )
    return paths


def is_open_walk2(graph: nx.Graph, island: Sequence[Any]) -> bool:
    """Return True when ``island`` is an open walk in ``graph``.

    Unlike ``is_open_walk``, a two-node island is not treated as an
    open walk unless the degree condition also holds (two degree-1
    nodes and the rest degree 2).

    Args:
        graph: The graph.
        island: Nodes of one connected component.

    Returns:
        bool: True if the island is an open walk, False otherwise.

    Examples:
        >>> import networkx as nx
        >>> from simetri.helpers.graph import is_open_walk2
        >>> path = nx.path_graph(3)
        >>> is_open_walk2(path, [0, 1, 2])
        True
        >>> is_open_walk2(path, [0, 1])
        False
    """
    degrees = [graph.degree(node) for node in island]
    return set(degrees) == {1, 2} and degrees.count(1) == 2


def longest_chain(edges: Sequence[Sequence[Any]]) -> list[tuple[Any, Any]]:
    """Order edges into one connected chain.

    Starts at ``edges[0]`` and grows from either end. Edges in the
    result are oriented so consecutive edges share the connecting node.

    Args:
        edges: Two-endpoint edges.

    Returns:
        list[tuple]: Oriented edges in chain order, or ``[]`` if
        ``edges`` is empty.

    Examples:
        >>> from simetri.helpers.graph import longest_chain
        >>> longest_chain([(1, 2), (0, 1), (2, 3)])
        [(0, 1), (1, 2), (2, 3)]
        >>> longest_chain([(0, 1), (1, 2), (2, 0)])
        [(0, 1), (1, 2), (2, 0)]
        >>> longest_chain([])
        []
    """
    if not edges:
        return []

    endpoint_edges = {}
    for edge_index, edge in enumerate(edges):
        start, end = edge
        endpoint_edges.setdefault(start, []).append(edge_index)
        endpoint_edges.setdefault(end, []).append(edge_index)

    chain = [tuple(edges[0])]
    processed_indices = {0}

    while True:
        extended = False
        for edge_index in endpoint_edges[chain[-1][1]]:
            if edge_index not in processed_indices:
                edge = edges[edge_index]
                if edge[0] == chain[-1][1]:
                    chain.append(tuple(edge))
                else:
                    chain.append((edge[1], edge[0]))
                processed_indices.add(edge_index)
                extended = True
                break
        if extended:
            continue

        for edge_index in endpoint_edges[chain[0][0]]:
            if edge_index not in processed_indices:
                edge = edges[edge_index]
                if edge[1] == chain[0][0]:
                    chain.insert(0, tuple(edge))
                else:
                    chain.insert(0, (edge[1], edge[0]))
                processed_indices.add(edge_index)
                extended = True
                break
        if not extended:
            break

    return chain


def is_cycle(graph: nx.Graph, island: Sequence[Any]) -> bool:
    """Return True when every node in ``island`` has degree 2.

    Args:
        graph: The graph.
        island: Nodes of one connected component.

    Returns:
        bool: True if the island is a cycle, False otherwise.

    Examples:
        >>> import networkx as nx
        >>> from simetri.helpers.graph import is_cycle
        >>> is_cycle(nx.cycle_graph(3), [0, 1, 2])
        True
        >>> is_cycle(nx.path_graph(3), [0, 1, 2])
        False
    """
    degrees = [graph.degree(node) for node in island]
    return set(degrees) == {2}


def is_open_walk(graph: nx.Graph, island: Sequence[Any]) -> bool:
    """Return True when ``island`` is an open walk in ``graph``.

    A two-node island is always True. Longer islands need exactly two
    degree-1 nodes and the rest degree 2.

    Args:
        graph: The graph.
        island: Nodes of one connected component.

    Returns:
        bool: True if the island is an open walk, False otherwise.

    Examples:
        >>> import networkx as nx
        >>> from simetri.helpers.graph import is_open_walk
        >>> path = nx.path_graph(3)
        >>> is_open_walk(path, [0, 1, 2])
        True
        >>> is_open_walk(path, [0, 1])
        True
        >>> is_open_walk(nx.cycle_graph(3), [0, 1, 2])
        False
    """
    if len(island) == 2:
        return True
    degrees = [graph.degree(node) for node in island]
    return set(degrees) == {1, 2} and degrees.count(1) == 2


def graph_summary(graph: nx.Graph) -> str:
    """Return a summary of cycles, open walks, and degenerate nodes.

    Args:
        graph: The graph.

    Returns:
        str: One block per connected component.

    Examples:
        >>> import networkx as nx
        >>> from simetri.helpers.graph import graph_summary
        >>> print(graph_summary(nx.path_graph(3)))
        Island: {0, 1, 2}
        Open Walk: 3 nodes
        ----------------------------------------
        >>> print(graph_summary(nx.cycle_graph(3)))
        Island: {0, 1, 2}
        Cycle: 3 nodes
        ----------------------------------------
        >>> print(graph_summary(nx.star_graph(3)))
        Island: {0, 1, 2, 3}
        Degenerate: 4 nodes
        (Node, Degree): [(0, 3)]
        ----------------------------------------
    """
    lines = []
    for island in nx.connected_components(graph):
        if len(island) > 8:
            island = list(island)
            lines.append(f"Island: {island[:4]}, ... , {island[-4:]}")
        else:
            lines.append(f"Island: {island}")
        if is_cycle(graph, island):
            lines.append(f"Cycle: {len(island)} nodes")
        elif is_open_walk(graph, island):
            lines.append(f"Open Walk: {len(island)} nodes")
        else:
            degenerates = [node for node in island if graph.degree(node) > 2]
            degrees = f"{[(node, graph.degree(node)) for node in degenerates]}"
            lines.append(f"Degenerate: {len(island)} nodes")
            lines.append(f"(Node, Degree): {degrees}")
        lines.append("-" * 40)
    return "\n".join(lines)


@dataclass
class Graph:
    """A collection of nodes and edges backed by NetworkX.

    Attributes:
        type: Graph kind. Defaults to ``"undirected"``.
        subtype: Graph subtype. Defaults to ``"none"``.
        nx_graph: The NetworkX graph. Defaults to None.

    Examples:
        >>> import networkx as nx
        >>> from simetri.helpers.graph import Graph
        >>> wrapped = Graph(nx_graph=nx.cycle_graph(3))
        >>> wrapped.cycles
        [[1, 0, 2]]
        >>> list(Graph(nx_graph=nx.path_graph(2)).edges)
        [(0, 1)]
        >>> list(Graph(nx_graph=nx.path_graph(2)).nodes)
        [0, 1]
    """

    type: str | Types = "undirected"
    subtype: str | Types = "none"  # this can be Types.WEIGHTED
    nx_graph: nx.Graph | None = None

    @property
    def islands(self) -> list[list[Any]]:
        """Return a list of all islands both cyclic and acyclic.

        Returns:
            list[list]: Node lists, one per connected component.
        """
        return [
            list(island)
            for island in self.nx_graph.connected_components(self.nx_graph)
        ]

    @property
    def cycles(self) -> list[list[Any]]:
        """Return a list of cycles.

        Cycles are not closed (the start node is not repeated).

        Returns:
            list[list]: Cycles from ``networkx.cycle_basis``.

        Examples:
            >>> import networkx as nx
            >>> from simetri.helpers.graph import Graph
            >>> Graph(nx_graph=nx.cycle_graph(3)).cycles
            [[1, 0, 2]]
            >>> Graph(nx_graph=nx.path_graph(3)).cycles
            []
        """
        return nx.cycle_basis(self.nx_graph)

    @property
    def open_walks(self) -> list[list[Any]]:
        """Return a list of open walks (aka open chains).

        Returns:
            list[list]: Islands that ``is_open_walk`` accepts.
        """
        return [
            island
            for island in self.islands
            if is_open_walk(self.nx_graph, island)
        ]

    @property
    def edges(self) -> Any:
        """Return the edges of the graph.

        Returns:
            EdgeView: Edges of ``nx_graph``.

        Examples:
            >>> import networkx as nx
            >>> from simetri.helpers.graph import Graph
            >>> list(Graph(nx_graph=nx.path_graph(2)).edges)
            [(0, 1)]
        """
        return self.nx_graph.edges

    @property
    def nodes(self) -> Any:
        """Return the nodes of the graph.

        Returns:
            NodeView: Nodes of ``nx_graph``.

        Examples:
            >>> import networkx as nx
            >>> from simetri.helpers.graph import Graph
            >>> list(Graph(nx_graph=nx.path_graph(2)).nodes)
            [0, 1]
        """
        return self.nx_graph.nodes


def sanitize_weighted_graph_edges(
    edges: Sequence[tuple[Any, Any, Any]],
) -> list[tuple[Any, Any, Any]]:
    """Drop duplicate undirected pairs, keeping the first weight.

    Args:
        edges: Weighted edges ``(u, v, weight)``.

    Returns:
        list[tuple]: Deduplicated weighted edges, sorted.

    Examples:
        >>> from simetri.helpers.graph import sanitize_weighted_graph_edges
        >>> sanitize_weighted_graph_edges([(1, 0, 5), (0, 1, 9), (2, 3, 1)])
        [(1, 0, 5), (2, 3, 1)]
    """
    clean_edges = []
    s_seen = set()
    for edge in edges:
        e1, e2, _ = edge
        frozen_edge = frozenset((e1, e2))
        if frozen_edge in s_seen:
            continue
        s_seen.add(frozen_edge)
        clean_edges.append(edge)
    clean_edges.sort()
    return clean_edges


def sanitize_graph_edges(
    edges: Sequence[tuple[Any, Any]],
) -> list[tuple[Any, Any]]:
    """Drop duplicate undirected pairs and order each pair ``(min, max)``.

    Args:
        edges: Two-endpoint edges.

    Returns:
        list[tuple]: Deduplicated edges with ordered endpoints, sorted.

    Examples:
        >>> from simetri.helpers.graph import sanitize_graph_edges
        >>> sanitize_graph_edges([(1, 0), (0, 1), (2, 3), (3, 2)])
        [(0, 1), (2, 3)]
    """
    s_edge_set = set()
    for edge in edges:
        s_edge_set.add(frozenset(edge))
    edges = [tuple(x) for x in s_edge_set]
    edges = [(min(x), max(x)) for x in edges]
    edges.sort()
    return edges

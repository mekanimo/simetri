"""Graph related functions and classes. Uses NetworkX for graph operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from dataclasses import dataclass

import networkx as nx

from ..base.all_enums import Types
from ..base.common import PointType
from ..config.settings import runtime_defaults as defaults
from ..geom.geom_utils import close_points_square
from ..geom.points.point_utils import distance


@dataclass
class GraphEdge:
    """Edge in a graph with start and end nodes.

    Examples:
    Attributes:
        start (PointType): Start node.
        end (PointType): End node.
        length (float): Edge length computed in ``__post_init__``.
    """

    start: PointType
    end: PointType

    def __post_init__(self) -> None:
        """Compute ``length`` from the start and end node positions."""
        self.length = distance(self.start.pos, self.end.pos)

    @property
    def nodes(self) -> tuple[PointType, PointType]:
        """Return the start and end nodes of the edge.

        Returns:
            tuple: ``(start, end)`` nodes.

        Examples:
        """
        return (self.start, self.end)


def edges_to_nodes(edges: Sequence[Sequence]) -> Sequence:
    """Given a list of edges, return a connected list of nodes.

    Args:
        edges (Sequence[Sequence]): List of edges.

    Returns:
        Sequence: Connected list of nodes.

    Examples:
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


def get_cycles(edges: Sequence[GraphEdge]) -> Sequence[GraphEdge]:
    """Computes all the cycles in a given graph of edges.

    Args:
        edges (Sequence[GraphEdge]): List of graph edges.

    Returns:
        Sequence[GraphEdge]: List of cycles if any cycle is found, None otherwise.

    Examples:
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
    """Find all paths starting from a given node.

    Args:
        graph (nx.Graph): The graph.
        node: The starting node.

    Returns:
        List: All paths starting from the given node.

    Examples:
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

    Args:
        graph (nx.Graph): The graph.
        island: The island.

    Returns:
        bool: True if the island is an open walk, False otherwise.

    Examples:
    """
    degrees = [graph.degree(node) for node in island]
    return set(degrees) == {1, 2} and degrees.count(1) == 2


def longest_chain(edges: Sequence[Sequence]) -> Sequence:
    """Given a list of graph edges, return a list of connected nodes.

    Args:
        edges (Sequence[Sequence]): List of graph edges.

    Returns:
        Sequence: List of connected nodes.

    Examples:
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


def is_cycle(graph: nx.Graph, island: Sequence) -> bool:
    """Return True when ``island`` is a cycle in ``graph``.

    Args:
        graph (nx.Graph): The graph.
        island (Sequence): The island.

    Returns:
        bool: True if the island is a cycle, False otherwise.

    Examples:
    """
    degrees = [graph.degree(node) for node in island]
    return set(degrees) == {2}


def is_open_walk(graph: nx.Graph, island: Sequence) -> bool:
    """Return True when ``island`` is an open walk in ``graph``.

    Args:
        graph (nx.Graph): The graph.
        island (Sequence): The island.

    Returns:
        bool: True if the island is an open walk, False otherwise.

    Examples:
    """
    if len(island) == 2:
        return True
    degrees = [graph.degree(node) for node in island]
    return set(degrees) == {1, 2} and degrees.count(1) == 2


def graph_summary(graph: nx.Graph) -> str:
    """Return a summary of cycles, open walks, and degenerate nodes.

    Args:
        graph (nx.Graph): The graph.

    Returns:
        str: Summary of the graph.

    Examples:
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
class Node:
    """A 2D point with ``x`` and ``y`` coordinates for graph use.

    Examples:
    Attributes:
        x (float): X coordinate.
        y (float): Y coordinate.
    """

    x: float
    y: float

    @property
    def pos(self) -> tuple[float, float]:
        """Return the position of the node.

        Examples:
        """
        return (self.x, self.y)

    def __eq__(self, other: object) -> bool:
        """Check if two nodes are equal.

        Args:
            other (object): The other node.

        Returns:
            bool: True if the nodes are equal, False otherwise.

        Examples:
        """
        return close_points_square(
            self.pos, other.pos, dist2=defaults["abs_tol"] ** 2
        )


@dataclass
class Graph:
    """A collection of nodes and edges backed by NetworkX.

    Examples:
    Attributes:
        type (Types): The type of the graph.
        subtype (Types): The subtype of the graph.
        nx_graph (nx.Graph): The NetworkX graph object.
    """

    type: Types = "undirected"
    subtype: Types = "none"  # this can be Types.WEIGHTED
    nx_graph: "nx.Graph" = None

    @property
    def islands(self) -> list[list[Any]]:
        """Return a list of all islands both cyclic and acyclic.

        Returns:
            List: List of all islands.

        Examples:
        """
        return [
            list(island)
            for island in self.nx_graph.connected_components(self.nx_graph)
        ]

    @property
    def cycles(self) -> list[list[Any]]:
        """Return a list of cycles.

        Returns:
            List: List of cycles.

        Examples:
        """
        return nx.cycle_basis(self.nx_graph)

    @property
    def open_walks(self) -> list[list[Any]]:
        """Return a list of open walks (aka open chains).

        Returns:
            List: List of open walks.

        Examples:
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
            EdgeView: Edges of the graph.

        Examples:
        """
        return self.nx_graph.edges

    @property
    def nodes(self) -> Any:
        """Return the nodes of the graph.

        Returns:
            NodeView: Nodes of the graph.

        Examples:
        """
        return self.nx_graph.nodes


def sanitize_weighted_graph_edges(
    edges: Sequence[tuple[Any, Any, Any]],
) -> list[tuple[Any, Any, Any]]:
    """Sanitize weighted graph edges.

    Args:
        edges: A list of weighted graph edges.

    Returns:
        list[tuple[Any, Any, Any]]: Deduplicated weighted edges, sorted.

    Examples:
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
    """Sanitize graph edges.

    Args:
        edges: A list of graph edges.

    Returns:
        list[tuple[Any, Any]]: Deduplicated edges with ordered endpoints, sorted.

    Examples:
    """
    s_edge_set = set()
    for edge in edges:
        s_edge_set.add(frozenset(edge))
    edges = [tuple(x) for x in s_edge_set]
    edges = [(min(x), max(x)) for x in edges]
    edges.sort()
    return edges

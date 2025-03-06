"""Graph related functions and classes. Uses NetworkX for graph operations."""

from dataclasses import dataclass
from typing import Sequence
import logging

import networkx as nx

from ..graphics.common import common_properties, Point
from ..graphics.all_enums import Types
from ..settings.settings import defaults
from ..geometry.geometry import distance, close_points2


@dataclass
class GraphEdge:
    """Edge in a graph. It has a start and end point as nodes."""

    start: Point
    end: Point

    def __post_init__(self):
        """Initialize the GraphEdge with start and end points."""
        self.length = distance(self.start.pos, self.end.pos)
        common_properties(self)

    @property
    def nodes(self):
        """Return the start and end nodes of the edge."""
        return (self.start, self.end)


def edges2nodes(edges: Sequence[Sequence]) -> Sequence:
    """
    Given a list of edges, return a connected list of nodes.

    Args:
        edges (Sequence[Sequence]): List of edges.

    Returns:
        Sequence: Connected list of nodes.
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
        else:
            logging.warning("edges2nodes: last_edge not connected to nodes")

    return nodes


def get_cycles(edges: Sequence[GraphEdge]) -> Sequence[GraphEdge]:
    """
    Computes all the cycles in a given graph of edges.

    Args:
        edges (Sequence[GraphEdge]): List of graph edges.

    Returns:
        Sequence[GraphEdge]: List of cycles if any cycle is found, None otherwise.
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
def find_all_paths(graph, node):
    """
    Find all paths starting from a given node.

    Args:
        graph (nx.Graph): The graph.
        node: The starting node.

    Returns:
        List: All paths starting from the given node.
    """
    paths = []
    for node_ in graph.nodes():
        for path in nx.all_simple_paths(graph, node, node_):
            if len(path) > 1:
                paths.append(path)
    return paths


def is_open_walk2(graph, island):
    """
    Given a NetworkX Graph and an island, return True if the given island is an open walk.

    Args:
        graph (nx.Graph): The graph.
        island: The island.

    Returns:
        bool: True if the island is an open walk, False otherwise.
    """
    degrees = [graph.degree(node) for node in island]
    return set(degrees) == {1, 2} and degrees.count(1) == 2


def longest_chain(edges: Sequence[Sequence]) -> Sequence:
    """
    Given a list of graph edges, return a list of connected nodes.

    Args:
        edges (Sequence[Sequence]): List of graph edges.

    Returns:
        Sequence: List of connected nodes.
    """

    def add_edge(edge, chain, index, processed):
        nonlocal no_change
        if index == 0:
            chain.insert(0, edge)
        elif index == -1:
            chain.append(edge)
        processed.append(set(edge))
        no_change = False

    chain = []
    processed = []
    while len(chain) < len(edges):
        no_change = True
        for edge in edges:
            if set(edge) in processed:
                continue
            if not chain:
                add_edge(edge, chain, -1, processed)
            else:
                if edge[0] == chain[-1][1]:
                    add_edge(edge, chain, -1, processed)
                elif edge[0] == chain[0][0]:
                    add_edge((edge[1], edge[0]), chain, 0, processed)
                elif edge[1] == chain[0][0]:
                    add_edge(edge, chain, 0, processed)
                elif edge[1] == chain[-1][1]:
                    add_edge((edge[1], edge[0]), chain, -1, processed)
        if no_change:
            break
    return chain


def is_cycle(graph: nx.Graph, island: Sequence) -> bool:
    """
    Given a NetworkX Graph and an island, return True if the given island is a cycle.

    Args:
        graph (nx.Graph): The graph.
        island (Sequence): The island.

    Returns:
        bool: True if the island is a cycle, False otherwise.
    """
    degrees = [graph.degree(node) for node in island]
    return set(degrees) == {2}


def is_open_walk(graph: nx.Graph, island: Sequence) -> bool:
    """
    Given a NetworkX Graph and an island, return True if the given island is an open walk.

    Args:
        graph (nx.Graph): The graph.
        island (Sequence): The island.

    Returns:
        bool: True if the island is an open walk, False otherwise.
    """
    if len(island) == 2:
        return True
    degrees = [graph.degree(node) for node in island]
    return set(degrees) == {1, 2} and degrees.count(1) == 2


def graph_summary(graph: nx.Graph) -> str:
    """
    Return a summary of a graph including cycles, open walks and degenerate nodes.

    Args:
        graph (nx.Graph): The graph.

    Returns:
        str: Summary of the graph.
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
    """
    A Node object is a 2D point with x and y coordinates.
    Used in graphs corresponding to shapes and batches.

    Attributes:
        x (float): X coordinate.
        y (float): Y coordinate.
    """

    x: float
    y: float

    def __post_init__(self):
        """Initialize the Node with x and y coordinates."""
        common_properties(self)

    @property
    def pos(self):
        """Return the position of the node."""
        return (self.x, self.y)

    def __eq__(self, other: object) -> bool:
        """
        Check if two nodes are equal.

        Args:
            other (object): The other node.

        Returns:
            bool: True if the nodes are equal, False otherwise.
        """
        return close_points2(self.pos, other.pos, dist2=defaults["dist_tol"] ** 2)


@dataclass
class Graph:
    """
    A Graph object is a collection of nodes and edges.

    Attributes:
        type (Types): The type of the graph.
        subtype (Types): The subtype of the graph.
        nx_graph (nx.Graph): The NetworkX graph object.
    """

    type: Types = "undirected"
    subtype: Types = "none"  # this can be Types.WEIGHTED
    nx_graph: "nx.Graph" = None

    def __post_init__(self):
        """Initialize the Graph with type and subtype."""
        common_properties(self)

    @property
    def islands(self):
        """
        Return a list of all islands both cyclic and acyclic.

        Returns:
            List: List of all islands.
        """
        return [
            list(island) for island in self.nx_graph.connected_components(self.nx_graph)
        ]

    @property
    def cycles(self):
        """
        Return a list of cycles.

        Returns:
            List: List of cycles.
        """
        return nx.cycle_basis(self.nx_graph)

    @property
    def open_walks(self):
        """
        Return a list of open walks (aka open chains).

        Returns:
            List: List of open walks.
        """
        res = []
        for island in self.islands:
            if is_open_walk(self.nx_graph, island):
                res.append(island)
        return res

    @property
    def edges(self):
        """
        Return the edges of the graph.

        Returns:
            EdgeView: Edges of the graph.
        """
        return self.nx_graph.edges

    @property
    def nodes(self):
        """
        Return the nodes of the graph.

        Returns:
            NodeView: Nodes of the graph.
        """
        return self.nx_graph.nodes

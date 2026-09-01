from __future__ import annotations

from math import ceil, degrees, log10, pi, sqrt
from typing import TYPE_CHECKING

import networkx as nx

from ..geometry.geom_utils import (
    inclination_angle,
    right_handed,
)
from ..helpers.graph import edges_to_nodes, get_cycles, is_cycle, is_open_walk
from ..settings.settings import defaults
from .common import LineType

if TYPE_CHECKING:
    from .batch import Group


def _merge_shapes(
    self,
    dist_tol: float | None = None,
    merge_angle_tol: float = 0.1,
    debug: bool = False,
    **kwargs,
) -> Group:
    """
    Tries to merge the shapes in the group. Returns a new group
    with the merged shapes as well as the shapes that could not be merged.

    Args:
        dist_tol (float, optional): Distance tolerance for merging vertices.
            Defaults to None.
        merge_angle_tol (float, optional): Angle tolerance for merging collinear
            edges. Defaults to 0.1.
        debug (bool, optional): Print point and angle diagnostics.
            Defaults to False.
        **kwargs: Additional keyword arguments passed to the returned group.

    Returns:
        Group: A new group with the merged shapes.
    """
    from .batch import Group
    from .shape import Shape

    if len(self) < 2:
        return self
    if dist_tol is None:
        dist_tol = defaults["dist_tol"]
    n_round = max(0, ceil(log10(sqrt(2) / dist_tol)))
    if debug:
        print("Merge diagnostics:")
    self._set_node_dictionaries(
        self.all_vertices, dist_tol=dist_tol, debug=debug
    )
    edges, segments = self._get_edges_and_segments(n_round=n_round)
    segments = self.merge_collinears(
        edges, merge_angle_tol=merge_angle_tol, debug=debug
    )
    d_coord_node = self.d_coord_node
    d_node_coord = self.d_node_coord
    edges = [[d_coord_node[coord] for coord in seg] for seg in segments]
    nx_graph = nx.Graph()
    nx_graph.update(edges)
    cycles = get_cycles(edges)
    new_shapes = []
    if cycles:
        for cycle in cycles:
            if len(cycle) < 3:
                continue
            nodes = cycle
            vertices = [d_node_coord[node] for node in nodes]
            if not right_handed(vertices):
                vertices.reverse()
            vertices = [self.d_rounded_coord[vert] for vert in vertices]
            shape = Shape(vertices, closed=True)
            new_shapes.append(shape)
    islands = list(nx.connected_components(nx_graph))
    if islands:
        for island in islands:
            if is_cycle(nx_graph, island):
                continue
            if is_open_walk(nx_graph, island):
                island = list(island)
                edges = [
                    edge
                    for edge in list(nx_graph.edges)
                    if edge[0] in island and edge[1] in island
                ]
                nodes = edges_to_nodes(edges)
                vertices = [d_node_coord[node] for node in nodes]
                if not right_handed(vertices):
                    vertices.reverse()
                vertices = [self.d_rounded_coord[vert] for vert in vertices]
                shape = Shape(vertices)
                new_shapes.append(shape)

    group = Group(new_shapes)
    for k, v in kwargs.items():
        group.set_attribs(k, v)

    return group


def _merge_bin(_bin: list, d_node_coord: dict, d_coord_node: dict):
    """Merge collinear edges in an angle bin."""
    incl_angle = degrees(_bin[0][0])
    node_adjacency = {}
    for _, edge in _bin:
        start, end = edge
        node_adjacency.setdefault(start, set()).add(end)
        node_adjacency.setdefault(end, set()).add(start)

    res = []
    unvisited_nodes = set(node_adjacency)
    while unvisited_nodes:
        start_node = unvisited_nodes.pop()
        component_nodes = {start_node}
        nodes_to_visit = [start_node]
        while nodes_to_visit:
            node = nodes_to_visit.pop()
            for neighbor in node_adjacency[node]:
                if neighbor in unvisited_nodes:
                    unvisited_nodes.remove(neighbor)
                    component_nodes.add(neighbor)
                    nodes_to_visit.append(neighbor)

        if len(component_nodes) == 1:
            node = component_nodes.pop()
            res.append(d_node_coord[node])
        elif 45 < incl_angle < 135:
            start_node = min(
                component_nodes, key=lambda node: d_node_coord[node][1]
            )
            end_node = max(
                component_nodes, key=lambda node: d_node_coord[node][1]
            )
            res.append((d_node_coord[start_node], d_node_coord[end_node]))
        else:
            start_node = min(
                component_nodes, key=lambda node: d_node_coord[node]
            )
            end_node = max(component_nodes, key=lambda node: d_node_coord[node])
            res.append((d_node_coord[start_node], d_node_coord[end_node]))

    return res


def _merge_collinears(
    self,
    edges: list[LineType],
    merge_angle_tol: float = 0.1,
    debug: bool = False,
) -> list[LineType]:
    """Merge connected collinear edges."""
    d_node_coord = self.d_node_coord
    d_coord_node = self.d_coord_node
    if len(edges) < 2:
        return edges

    angles_edges = []
    for edge in edges:
        edge = list(edge)
        start = d_node_coord[edge[0]]
        end = d_node_coord[edge[1]]
        angle = inclination_angle(start, end)
        if abs(angle - pi) < merge_angle_tol:
            angle = 0
        angles_edges.append((angle, edge))

    angles_edges.sort()
    bins = []
    current_bin = [angles_edges[0]]
    smallest_rejected_angle_difference = None
    for angle, edge in angles_edges[1:]:
        current_angle = current_bin[0][0]
        angle_difference = abs(angle - current_angle)
        if angle_difference <= merge_angle_tol:
            current_bin.append((angle, edge))
        else:
            if (
                smallest_rejected_angle_difference is None
                or angle_difference < smallest_rejected_angle_difference
            ):
                smallest_rejected_angle_difference = angle_difference
            bins.append(current_bin)
            current_bin = [(angle, edge)]
    bins.append(current_bin)

    if debug:
        angle_difference_degrees = (
            None
            if smallest_rejected_angle_difference is None
            else degrees(smallest_rejected_angle_difference)
        )
        print(
            "  Smallest rejected angle difference "
            f"(merge_angle_tol={merge_angle_tol}): "
            f"{smallest_rejected_angle_difference} radians; "
            f"{angle_difference_degrees} degrees"
        )
        print(
            "  Recommended angle setting: "
            f"merge_angle_tol>={smallest_rejected_angle_difference}"
        )

    res = []
    for angle_bin in bins:
        res.extend(_merge_bin(angle_bin, d_node_coord, d_coord_node))

    return res

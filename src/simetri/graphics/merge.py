import logging
from typing import List, Dict

from numpy import isclose
import networkx as nx

from .common import get_defaults, Line, Point
from ..settings.settings import defaults
from ..geometry.geometry import (
    right_handed,
    fix_degen_points,
    inclination_angle,
    all_intersections
)
from ..helpers.graph import get_cycles, is_cycle, is_open_walk, edges2nodes


def _merge_shapes(
    self,
    tol: float = None,
    rtol: float = None,
    atol: float = None,
    dist_tol: float = None,
    n_round: int = None,
    **kwargs,
) -> "Batch":
    """
    Tries to merge the shapes in the batch. Returns a new batch
    with the merged shapes as well as the shapes that could not be merged.

    Args:
        tol (float, optional): Tolerance for merging shapes. Defaults to None.
        rtol (float, optional): Relative tolerance for merging shapes. Defaults to None.
        atol (float, optional): Absolute tolerance for merging shapes. Defaults to None.
        dist_tol (float, optional): Distance tolerance for merging shapes. Defaults to None.
        n_round (int, optional): Number of rounding digits for merging shapes. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Batch: A new batch with the merged shapes.
    """
    from .batch import Batch
    from .shape import Shape

    tol, rtol, atol, dist_tol, n_round = get_defaults(
        ["tol", "rtol", "atol", "dist_tol", "n_round"],
        [tol, rtol, atol, dist_tol, n_round],
    )
    if all((len(shape) == 1 for shape in self.all_shapes)):
        edges = Shape([shape.vertices[0] for shape in self.all_shapes])
        batch = Batch(edges)
        return batch

    d_node_id_coords, edges = self._get_graph_nodes_and_edges(
        dist_tol=dist_tol, n_round=defaults["n_round"]
    )
    edges = self._merge_collinears(
        d_node_id_coords, edges, tol=tol, rtol=rtol, atol=atol
    )

    vertices = []
    for edge in edges:
        vertices.extend(edge)
    s_vertices = set(vertices)
    d_node_id_coords = {}
    d_cooords__node_id = {}
    for i, vertex in enumerate(s_vertices):
        d_node_id_coords[i] = vertex
        d_cooords__node_id[vertex] = i

    edges = [[d_cooords__node_id[x] for x in edge] for edge in edges]
    nx_graph = nx.Graph()
    nx_graph.update(edges)
    cycles = get_cycles(edges)
    new_shapes = []
    if cycles:
        for cycle in cycles:
            if len(cycle) < 3:
                continue
            nodes = cycle
            vertices = [d_node_id_coords[node_id] for node_id in nodes]
            if not right_handed(vertices):
                vertices.reverse()
            vertices = fix_degen_points(vertices, closed=True, dist_tol=dist_tol)
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
                nodes = edges2nodes(edges)
                vertices = [d_node_id_coords[node] for node in nodes]
                if not right_handed(vertices):
                    vertices.reverse()
                vertices = fix_degen_points(vertices, closed=False, dist_tol=dist_tol)
                shape = Shape(vertices)
                new_shapes.append(shape)
            else:
                msg = "Batch.merge_shapes: Degenerate points found!"
                logging.warning(msg)
    batch = Batch(new_shapes)
    for k, v in kwargs.items():
        batch.set_attribs(k, v)

    return batch


def _merge_collinears(
    self,
    d_node_id_coord: Dict[int, Point],
    edges: List[Line],
    angle_bin_size: float = 0.1,
    tol: float = None,
    rtol: float = None,
    atol: float = None,
) -> List[Line]:
    """
    Merge collinear edges.

    Args:
        d_node_id_coord (Dict[int, Point]): Dictionary of node id to coordinates.
        edges (List[Line]): List of edges.
        angle_bin_size (float, optional): Bin size for grouping angles. Defaults to 0.1.
        tol (float, optional): Tolerance for merging edges. Defaults to None.
        rtol (float, optional): Relative tolerance for merging edges. Defaults to None.
        atol (float, optional): Absolute tolerance for merging edges. Defaults to None.

    Returns:
        List[Line]: List of merged edges.
    """
    tol, rtol, atol = get_defaults(["tol", "rtol", "atol"], [tol, rtol, atol])

    def merge_multiple_edges(collinear_edges):
        """
        Merge multiple collinear edges in a list of edges.

        Args:
            collinear_edges (list): List of collinear edges.

        Returns:
            list: Merged edge.
        """
        x_coords = []
        y_coords = []
        points = []
        for edge in collinear_edges:
            x_coords.extend([edge[0][0], edge[1][0]])
            y_coords.extend([edge[0][1], edge[1][1]])
            points.extend(edge)

        xmin = min(x_coords)
        xmax = max(x_coords)
        ymin = min(y_coords)
        ymax = max(y_coords)
        rtol = defaults["rtol"]
        atol = defaults["atol"]
        if isclose(xmin, xmax, rtol=rtol, atol=atol):
            p1 = [
                p
                for p in points
                if isclose(p[1], ymin, rtol=rtol, atol=atol)
            ][0]
            p2 = [
                p
                for p in points
                if isclose(p[1], ymax, rtol=rtol, atol=atol)
            ][0]
        else:
            p1 = [
                p
                for p in points
                if isclose(p[0], xmin, rtol=rtol, atol=atol)
            ][0]
            p2 = [
                p
                for p in points
                if isclose(p[0], xmax, rtol=rtol, atol=atol)
            ][0]

        return [p1, p2]

    def process_islands(islands, res, merged):
        """
        Process islands of collinear edges.

        Args:
            islands (list): List of islands.
            res (list): List of merged edges.
            merged (set): Set of merged edges.
        """
        for island in islands:
            collinear_edges = []
            collinear_edge_indices = []
            for i in island:
                edge1_indices = bin_[i][i_edge]
                edge1 = [d_node_id_coord[x] for x in edge1_indices]
                for conn_type_x_res_ind2 in d_ind1_conn_type_x_res_ind2[i]:
                    _, _, ind2 = conn_type_x_res_ind2
                edge2_indices = bin_[ind2][i_edge]
                edge2 = [d_node_id_coord[x] for x in edge2_indices]
                if set((i, ind2)) in s_processed_edge_indices:
                    continue
                collinear_edges.extend([edge1, edge2])
                collinear_edge_indices.append(edge1_indices)
                collinear_edge_indices.append(edge2_indices)
                s_processed_edge_indices.add(frozenset((i, ind2)))
            if collinear_edges:
                for edge in collinear_edge_indices:
                    merged.append(frozenset(edge))
                res.append(merge_multiple_edges(collinear_edges))

    if len(edges) < 2:
        return edges

    angles_edges = []
    i_angle, i_edge = 0, 1
    for edge in edges:
        edge = list(edge)
        p1 = d_node_id_coord[edge[0]]
        p2 = d_node_id_coord[edge[1]]
        angle = inclination_angle(p1, p2)
        angles_edges.append((angle, edge))

    # group angles into bins
    angles_edges.sort()

    bins = []
    bin_ = [angles_edges[0]]
    for angle, edge in angles_edges[1:]:
        angle1 = bin_[0][i_angle]
        if abs(angle - angle1) <= angle_bin_size:
            bin_.append((angle, edge))
        else:
            bins.append(bin_)
            bin_ = [(angle, edge)]
    bins.append(bin_)
    merged = []
    res = []
    # x_res can be a point or segment(overlapping edges)
    i_ind2 = 2  # indices for intersection results
    for bin_ in bins:
        segments = [[d_node_id_coord[node] for node in x[i_edge]] for x in bin_]

        d_ind1_conn_type_x_res_ind2 = all_intersections(
            segments, rtol=rtol, use_intersection3=True
        )
        connections = {}
        for i in range(len(bin_)):
            connections[i] = set()
        for k, values in d_ind1_conn_type_x_res_ind2.items():
            for v in values:
                connections[k].add(v[i_ind2])
        # create a graph of connections
        g_connections = nx.Graph()
        for k, v in connections.items():
            for x in v:
                g_connections.add_edge(k, x)
        islands = list(nx.connected_components(g_connections))
        s_processed_edge_indices = set()
        process_islands(islands, res, merged)
    for edge in edges:
        if set(edge) not in merged:
            res.append([d_node_id_coord[x] for x in edge])

    return res
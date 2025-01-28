import logging

import networkx as nx

from .common import get_defaults
from ..settings.settings import defaults
from ..helpers.geometry import right_handed, fix_degen_points
from ..helpers.graph import get_cycles, is_cycle, is_open_walk, edges2nodes


def merge_shapes_(
    self,
    tol: float = None,
    rtol: float = None,
    atol: float = None,
    dist_tol: float = None,
    n_round: int = None,
    **kwargs,
) -> "Batch":
    from .batch import Batch
    from .shape import Shape

    """Tries to merge the shapes in the batch. Returns a new batch
    with the merged shapes as well as the shapes that could not be merged."""
    # if all shapes have only one point
    # connect them all
    # To do: we should check if the points are close
    tol, rtol, atol, dist_tol, n_round = get_defaults(
        ["tol", "rtol", "atol", "dist_tol", "n_round"],
        [tol, rtol, atol, dist_tol, n_round],
    )
    if all((len(shape) == 1 for shape in self.all_shapes)):
        edges = Shape([shape.vertices[0] for shape in self.all_shapes])
        batch = Batch(edges)
        return batch
    # To do: Check if the edges are duplicates if remove_duplicates is True
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

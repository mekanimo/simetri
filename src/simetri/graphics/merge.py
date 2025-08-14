from typing import List, Dict
from collections import OrderedDict
from math import degrees

from numpy import isclose
import networkx as nx

from .common import Line
from ..geometry.geometry import (
    right_handed,
    fix_degen_points,
    inclination_angle,
    round_segment,
    round_point,
    pi
)
from ..helpers.graph import get_cycles, is_cycle, is_open_walk, edges2nodes
from ..settings.settings import defaults

def _merge_shapes(
    self,
    n_round: int = None, **kwargs) -> "Batch":
    """
    Tries to merge the shapes in the batch. Returns a new batch
    with the merged shapes as well as the shapes that could not be merged.

    Args:
        n_round (int, optional): Number of rounding digits for merging shapes. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Batch: A new batch with the merged shapes.
    """
    from .batch import Batch
    from .shape import Shape
    n_round = defaults['n_round'] if n_round is None else n_round
    self._set_node_dictionaries(self.all_vertices, n_round=n_round)
    edges, segments = self._get_edges_and_segments(n_round=n_round)
    segments = self._merge_collinears(edges, n_round=n_round)
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
            vertices =  [self.d_rounded_coord[vert] for vert in vertices]
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
                vertices = [d_node_coord[node] for node in nodes]
                if not right_handed(vertices):
                    vertices.reverse()
                vertices =  [self.d_rounded_coord[vert] for vert in vertices]
                shape = Shape(vertices)
                new_shapes.append(shape)

    batch = Batch(new_shapes)
    for k, v in kwargs.items():
        batch.set_attribs(k, v)

    return batch


def _merge_collinears(
    self,
    edges: List[Line],
    angle_bin_size: float = 0.1,
    n_round: int = 2,
) -> List[Line]:
    """
    Merge collinear edges.

    Args:
        edges (List[Line]): List of edges.
        angle_bin_size (float, optional): Bin size for grouping angles. Defaults to 0.1.
        n_round (int, optional): Number of rounding digits. Defaults to 2.
    Returns:
        List[Line]: List of merged edges.
    """


    def merge_bin(_bin:list, d_node_coord:dict, d_coord_node: dict, n_round:int=2):
        '''Merge collinear edges in a bin.

        Args:
            _bin (list): List of edges in a bin.
            d_node_coord (dict): Dictionary of node id to coordinates.
            n_round (int, optional): Number of rounding digits. Defaults to 2.

        Returns:
            list: List of merged edges.
        '''
        segs = [[d_node_coord[node] for node in x[i_edge]] for x in bin_]
        incl_angle = degrees(_bin[0][0])
        if 45 < incl_angle < 135:
            # sort by y coordinates
            segs.sort(key=lambda x: x[0][1])
        else:
            # sort by x coordinates
            segs.sort(key=lambda x: x[0][0])

        seg_ids = []
        for seg in segs:
            p1, p2 = seg
            seg_ids.append([d_coord_node[p1], d_coord_node[p2]])

        graph = nx.Graph()
        graph.add_edges_from(seg_ids)
        islands = list(nx.connected_components(graph))
        res = []
        for island in islands:
            island = list(island)
            if len(island) > 1:

                island = [d_node_coord[x] for x in island]
                if 45 < incl_angle < 135:
                    # sort by y coordinates
                    island.sort(key=lambda x: x[1])
                else:
                    # sort by x coordinates
                    segs.sort(key=lambda x: x[0][0])
                    island.sort()
                res.append((island[0], island[-1]))
            else:
                res.append(d_node_coord[island[0]])

        return res

    d_node_coord = self.d_node_coord
    d_coord_node = self.d_coord_node
    if len(edges) < 2:
        return edges

    angles_edges = []
    i_angle, i_edge = 0, 1
    for edge in edges:
        edge = list(edge)
        p1 = d_node_coord[edge[0]]
        p2 = d_node_coord[edge[1]]
        angle = inclination_angle(p1, p2)
        if abs(angle - pi) < angle_bin_size:
            angle = 0
        angles_edges.append((angle, edge))

    # group angles into bins
    angles_edges.sort()

    bins = []
    bin_ = [angles_edges[0]]
    for angle, edge in angles_edges[1:]:
        if abs(angle - pi) < angle_bin_size:
            angle = 0
        angle1 = bin_[0][i_angle]
        if abs(angle1 - pi) < angle_bin_size:
            angle1 = 0
        diff = abs(angle - angle1)
        if diff <= angle_bin_size:
            bin_.append((angle, edge))
        else:
            bins.append(bin_)
            bin_ = [(angle, edge)]
    bins.append(bin_)

    res = []
    for bin_ in bins:
        res.extend(merge_bin(bin_, d_node_coord, d_coord_node, n_round=n_round))

    return res

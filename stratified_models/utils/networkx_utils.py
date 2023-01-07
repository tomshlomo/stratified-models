from typing import Iterable

import networkx as nx
from networkx import Graph


def cartesian_product(graphs: Iterable[Graph]) -> Graph:
    """Performs a cartesian product between a list of networkx graphs."""
    graphs = list(graphs)
    if len(graphs) == 1:
        return graphs[0]
    G = nx.cartesian_product(graphs[0], graphs[1])
    for i in range(2, len(graphs)):
        G = nx.cartesian_product(G, graphs[i])
    mapping = {}
    for node in G.nodes():
        mapping[node] = tuple(flatten(node))
    return nx.relabel_nodes(G, mapping)


def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

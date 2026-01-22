from functools import reduce
from typing import Any, Iterable

import networkx as nx
from networkx import Graph


def cartesian_product(graphs: Iterable[Graph]) -> Graph:
    g = reduce(nx.cartesian_product, graphs)

    def flatten(container: Any) -> Iterable[Any]:
        if not isinstance(container, tuple):
            yield container
            return
        for i in container:
            if isinstance(i, (list, tuple)):
                yield from flatten(i)
            else:
                yield i

    mapping = {}
    for node in g.nodes():
        mapping[node] = tuple(flatten(node))
    return nx.relabel_nodes(g, mapping)

from functools import reduce
from typing import Iterable

import networkx as nx
from networkx import Graph


def cartesian_product(graphs: Iterable[Graph]) -> Graph:
    return reduce(nx.cartesian_product, graphs)

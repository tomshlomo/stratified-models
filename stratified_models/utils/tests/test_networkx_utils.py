from itertools import product

import networkx as nx

from stratified_models.utils.networkx_utils import cartesian_product


def test_cartesian_product() -> None:
    g1 = nx.cycle_graph(2)
    g2 = nx.path_graph(3)
    g3 = nx.path_graph(4)

    g = cartesian_product([g1, g2])
    assert list(g.nodes) == list(product(range(2), range(3)))
    assert list(g.edges) == [
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1)),
        ((0, 1), (1, 1)),
        ((0, 1), (0, 2)),
        ((0, 2), (1, 2)),
        ((1, 0), (1, 1)),
        ((1, 1), (1, 2)),
    ]

    g = cartesian_product([g1])
    assert list(g.nodes) == [(0,), (1,)]

    g = cartesian_product([g1, g2, g3])
    assert list(g.nodes) == list(product(range(2), range(3), range(4)))

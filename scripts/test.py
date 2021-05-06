import networkx as nx
G = nx.Graph()
G.add_nodes_from({1, 2})
G.nodes[1]['type'] = 'alpha'
import numpy as np
import networkx as nx
import copy
import json


# utils
def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path.strip(".json").split("/")[-1]
    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"])

    if "features" in data.keys():
        features = data["features"]
    else:
        features = nx.degree(graph)

    features = {int(k): v for k, v in features.items()}
    nx.set_node_attributes(graph, features, "label")

    return graph


def node_match_with_type(node1, node2):
    return node1['label'] == node2['label']


def HasVertexInCommon(G, p, G_p):
    if (check_G1_in_G2(p, G) or check_G1_in_G2(G, p)) and check_G1_in_G2(p, G_p):
        return True
    else:
        return False


def union_subgraphs(C, G_p):
    nodes_common = []
    for c in C:
        is_c_in_G_p, matches = check_G1_in_G2(c, G_p)
        for match in matches:
            nodes_common.extend(list(match.keys()))

    G_C = G_p.subgraph(nodes_common)
    return G_C


def check_G1_in_G2(G1, G2):
    GM = nx.algorithms.isomorphism.GraphMatcher(G2, G1, node_match=node_match_with_type)
    is_G1_in_G2 = GM.subgraph_is_isomorphic()
    match = []
    for subgraph in GM.subgraph_isomorphisms_iter():
        match.append(subgraph)

    return is_G1_in_G2, match


def check_G1_equal_G2(G1, G2):
    is_G1_in_G2, _ = check_G1_in_G2(G1, G2)
    is_G2_in_G1, _ = check_G1_in_G2(G2, G1)
    is_G1_equal_G2 = (is_G1_in_G2 and is_G2_in_G1)
    return is_G1_equal_G2


def GetComponents(G):
    nodeset_list = list(nx.connected_components(G))
    C = []
    for nodeset in nodeset_list:
        C.append(G.subgraph(nodeset))
    return C


def Get_phi(x, G, D):
    S = D['subgraphs']
    W = D['where']
    N = [len(w) for w in W]

    phi = np.zeros((len(S)))

    for i_x, (x, n) in enumerate(zip(S, N)):
        is_G_in_x, matches = check_G1_in_G2(G, x)
        if is_G_in_x and len(list(x.nodes())) > len(V_G):
            phi[i_x] = n


def Get_Gdots(G, v, N_C):
    Gdots = []
    V_G = list(G.nodes())
    max_v = max(V_G)
    #     for v in V_G:
    for c in range(1, N_C + 1):
        Gdot = G.copy()
        Gdot.add_nodes_from([(max_v + 1, {"label": str(c)})])
        Gdot.add_edges_from([(v, max_v + 1)])
        Gdots.append(Gdot)
    return Gdots


# algorithm1 : Graph splitting
def Graph_Split(G_p, S):
    # <input>  G_p : thecurrent partial graph
    # <output> C : theoverlapping subgraphs of the partial graph

    P = []
    for s in S:
        is_s_in_G_p, match = check_G1_in_G2(s, G_p)
        if is_s_in_G_p == True:
            flag_s_is_largest_subgraph = True
            for sd in S:
                is_sd_in_G_p, _ = check_G1_in_G2(sd, G_p)
                is_s_in_sd, _ = check_G1_in_G2(s, sd)

                is_s_same_sd = nx.is_isomorphic(s, sd, node_match=node_match_with_type)
                if is_sd_in_G_p and is_s_in_sd and is_s_same_sd == False:
                    flag_s_is_largest_subgraph = False

            if flag_s_is_largest_subgraph == True:
                P.append(s)

    # sort in descending order
    size_P = []
    for p in P:
        num_nodes = len(p.nodes())
        size_P.append(num_nodes)
    sorted_idx = np.argsort(size_P)
    P = [P[i] for i in sorted_idx[::-1]]

    C = []
    P, p = FCFS(P, G_p, [], G_p)
    if p != 0:
        C.append(p)

    while not check_G1_equal_G2(G_p, union_subgraphs(C, G_p)):
        Found = False
        for c in C:
            P, cdot = FCFS(P, c, C, G_p)
            if cdot != 0:
                C.append(cdot)
                Found = True
                break

        if Found == False:
            D_g = copy.deepcopy(G_p)
            D_g.remove_nodes_from(list(union_subgraphs(C, G_p).nodes()))

            V_D_g = list(D_g.nodes())
            N = []
            for v in V_D_g:
                n = list(nx.all_neighbors(G_p, v))
                N.extend(n)

            N.extend(list(D_g.nodes()))
            D_g = G_p.subgraph(N)

            C.extend(GetComponents(D_g))

    return C


# algorithm2 : FindCommonFreqSubgraph
def FCFS(P, G, C, G_p):
    # <input>
    # P : the sortedsequence of frequent subgraphs that are present in the partial graph
    # G : agraph which theresultshould have somevertex in common with, this is always some Ci except for the initial execution
    # C : the thus far addedoverlapping subgraphs of the partial graph
    # <output>
    # p : the largest frequent subgraph present inthe partial graph that has at least one vertex in common with G (if found). p is also removed from the set P.

    for p in P:
        if len(C) > 0:
            is_p_in_C, _ = check_G1_in_G2(p, union_subgraphs(C, G_p))
        else:
            is_p_in_C = True

        if HasVertexInCommon(G, p, G_p) and is_p_in_C:
            P.remove(p)
            _, match = check_G1_in_G2(p, G_p)
            p = G_p.subgraph(list(match[0].keys()))
            return P, p
    return P, 0


def MostLikeEditGraph(G, D, v, N_C):
    # <input>
    # G : a “small”graph, one subgraph fromthe output of the graph splitting
    # D : the graph database
    # N_C : # of categoriex
    # v : interested node

    S = D['subgraphs']
    W = D['where']
    N = [len(w) for w in W]

    V_G = list(G.nodes())
    phi = np.zeros((len(V_G), N_C))

    Gdots = Get_Gdots(G, v, N_C)
    phi = np.zeros((len(Gdots)))
    for i_x, (x, n) in enumerate(zip(S, N)):
        is_G_in_x, matches = check_G1_in_G2(G, x)
        if is_G_in_x:
            for i_Gdot, Gdot in enumerate(Gdots):
                is_Gdot_in_x, _ = check_G1_in_G2(Gdot, x)
                if is_Gdot_in_x:
                    phi[i_Gdot] = phi[i_Gdot] + n

    max_i_Gdot = np.argmax(phi)
    max_Gdot = Gdots[max_i_Gdot]

    return max_Gdot, phi


def node_prediction(G, v, D, D_s, N_C, reg):
    # G : partial observed graph
    # v : interested node
    # D : database
    # N_C : # of categories
    # reg : regularization for existing hypothetical place ([0, 1]])

    S = D['subgraphs']
    C = Graph_Split(G, D_s['subgraphs'])
    print('=====')
    print(G.nodes())
    for c in C:
        print(c.nodes())

    C_v = []
    for c in C:
        #         print(c.nodes(data=True))
        if v in list(c.nodes()):
            C_v.append(c)

    V_exist = list(nx.neighbors(G, v))
    L_G = nx.get_node_attributes(G, 'label')
    L_exist = [L_G[v_e] for v_e in V_exist]
    L_exist = list(np.unique(L_exist))
    P_H_list = []
    for c in C_v:
        x, phi = MostLikeEditGraph(c, D, v, N_C)
        if sum(phi) == 0:
            continue
        P_H = [p / sum(phi) for p in phi]
        for c_H in range(1, N_C + 1):
            if str(c_H) in L_exist:
                P_H[c_H - 1] = P_H[c_H - 1] * reg
        P_H = [p / sum(phi) for p in phi]
        P_H_list.append(P_H)

    if len(P_H_list) == 0:
        return 0, [1] + [0 for i in range(N_C)]

    P_H_sum = [0 for i in range(N_C)]
    for P_H in P_H_list:
        P_H_sum = [P_H_sum[i] + P_H[i] for i in range(N_C)]
    P_H_avg = [p / len(P_H_list) for p in P_H_sum]

    c_pred = np.argmax(P_H_avg) + 1
    P_c_pred = P_H_avg[c_pred - 1]
    #     print(P_H_avg)
    return c_pred, [0] + P_H_avg

def relabel_network(network):
    #remap network just in case there is empty index
    mapping = {}
    index = 0
    for n in network.nodes():
        mapping[n] = index
        index += 1
    network = nx.relabel_nodes(network, mapping, copy=True)
    return network, mapping

print('done')
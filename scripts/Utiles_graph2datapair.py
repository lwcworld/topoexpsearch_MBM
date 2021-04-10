import networkx as nx
import json
import numpy as np
import copy

np.random.seed(101)

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
    return graph, features, name


def make_random_walk_from_G(input_G, NUM_WALKS=10, WALK_LENGTH=5, RETURN_PARAMS=0.0):
    """
    - G로부터 무작위의 랜덤 워크를 만든다.
    - 기본적으로 weight를 고려하며, edge의 weight가 클수록 해당 edge를 많이 지나가도록 선택된다.
    - 맨 처음 선택되는 시작 노드 또한, node의 weight에 따라서 선택된다.
        - 사실 degree centrality등 다양한 뱡식으로 선택해서 처리할 수 있지만, 일단은 그냥 무작위.
    - RETURN_Param: 이전의 노드 시퀀스가 ('a', 'b')였고, 지금이 'b'인 상태에서 다음 스텝을 선택할 때, 'a'로 돌아갈 확률을 의미함.
        - 예를 들어서, RETURN_Param가 0.3이라면, 'a'로 돌아갈 확률이 0.3이고, 나머지가 0.7에서 선택되는 것임.
        - 다만 여기서, 나머지가 없다면(terminal node라면) 무조건 원래대로 돌아가게 되는 것이고.
    """

    def find_next_node(input_G, previous_node, current_node, RETURN_PARAMS):
        """
        input_G의 current_node에서 weight를 고려하여 다음 노드를 선택함.
        - 이 과정에서 RETURN_params를 고려함.
        - 이 값은 previous_node로 돌아가는가 돌아가지 않는가를 정하게 됨.
        """
        select_probabilities = {}
        for node in input_G.neighbors(current_node):
            select_probabilities[node] = 1

        select_probabilities_sum = sum(select_probabilities.values())
        if previous_node is not None:
            if len([n for n in input_G.neighbors(current_node)]) == 1:
                select_probabilities = {k: 1 for k, v in select_probabilities.items()}
            else:
                select_probabilities = {k: v / (select_probabilities_sum - 1) * (1 - RETURN_PARAMS) for k, v in
                                        select_probabilities.items()}
                select_probabilities[previous_node] = RETURN_PARAMS  # 이 노드는 RETURN_PARAMS에 의해 결정됨.
        else:
            select_probabilities = {k: v / (select_probabilities_sum) for k, v in select_probabilities.items()}

        # print(select_probabilities)

        selected_node = np.random.choice(
            a=[k for k in select_probabilities.keys()],
            p=[v for v in select_probabilities.values()]
        )
        return selected_node

    ####################################
    path_lst = []
    for i in range(0, NUM_WALKS):
        path = [np.random.choice(input_G.nodes())]  # start node
        # 처음에는 previous node가 없으므로, None으로 세팅하고 진행함.
        next_node = find_next_node(input_G, None, path[-1], RETURN_PARAMS)
        path.append(next_node)
        # 이미 2 노드가 나왔으므로 그만큼 제외하고 새로운 노드를 찾고 넣어줌.
        for j in range(2, WALK_LENGTH):
            next_node = find_next_node(input_G, path[-2], path[-1], RETURN_PARAMS)
            path.append(next_node)
        # 새로운 패스가 만들어졌으므로 넣어줌.
        path_lst.append(path)
    return path_lst


def get_seq_nodesets(G, walk):
    # nodeset should be connected in G
    current_nodeset = list(set(walk))
    prev_nodesets = []
    for n in current_nodeset:
        prev_nodeset_c = copy.deepcopy(current_nodeset)
        prev_nodeset_c.remove(n)
        subG = G.subgraph(prev_nodeset_c)
        is_connected_subgraph = nx.is_connected(subG)
        if is_connected_subgraph == True:
            prev_nodesets.append(prev_nodeset_c)
    return prev_nodesets, current_nodeset


def get_bfstree(graph, feature, root, depth_limit):
    graph_c = copy.deepcopy(graph)
    graph_bfs = nx.bfs_tree(graph_c, root, depth_limit=depth_limit)
    nodes_bfs = graph_bfs.nodes
    feature_bfs = {key: feature[key] for key in nodes_bfs}
    return graph_bfs, feature_bfs


def decompose_tree_graph(graph, feature):
    # G is directed tree graph
    root = [n for n, d in graph.in_degree if d == 0][0]
    ends = [n for n, d in graph.out_degree if d == 0]
    G_d_list = []
    feature_d_list = []
    for e in ends:
        d = list(nx.all_simple_paths(graph, root, e))[0]
        G_d = graph.subgraph(d)
        nodes_d = G_d.nodes
        feature_d = {key: feature[key] for key in nodes_d}
        G_d_list.append(G_d)
        feature_d_list.append(feature_d)
    return G_d_list, feature_d_list


def generate_dataset(graph, feature, walks, i_o):
    datas_prev = []
    datas_current = []
    for i_w, walk in enumerate(walks):
        prev_nodesets, current_nodeset = get_seq_nodesets(graph, walk)

        graph_current = graph.subgraph(current_nodeset)
        feature_current = {key: feature[key] for key in current_nodeset}

        for prev_nodeset in prev_nodesets:
            unknown_node = list(set(current_nodeset) - set(prev_nodeset))[0]

            graph_current_tree, feature_current_tree = get_bfstree(graph_current, feature_current, root=unknown_node,
                                                                   depth_limit=3)
            graph_current_decom_list, feature_current_decom_list = decompose_tree_graph(graph_current_tree,
                                                                                        feature_current_tree)
            for graph_current_decom_i, feature_current_decom_i in zip(graph_current_decom_list,
                                                                      feature_current_decom_list):
                data_current = dict_edge_feature(graph_current_decom_i, feature_current_decom_i, i_o, unknown_node)
                datas_current.append(data_current)

                graph_prev_decom_i = copy.deepcopy(graph_current_decom_i)
                feature_prev_decom_i = copy.deepcopy(feature_current_decom_i)
                feature_prev_decom_i[unknown_node] = '0'
                data_prev = dict_edge_feature(graph_prev_decom_i, feature_prev_decom_i, i_o, unknown_node)
                datas_prev.append(data_prev)

            i_o = i_o + 1
    return datas_prev, datas_current, i_o


def dict_edge_feature(G, feature, i_o, std_node):
    edges_tuple = G.edges()
    edges_list = []
    v_start = std_node
    for i in range(0, len(edges_tuple)):
        edge = [[int(e[0]), int(e[1])] for e in edges_tuple if e[0] == v_start][0]
        edges_list.append(edge)
        v_start = edge[1]

    feature = {str(k): str(v) for k, v in feature.items()}

    out = {'idx': i_o, 'edges': edges_list, 'features': feature}
    return out


def save_data(file_name, dir, data):
    with open(dir + file_name, 'w') as outfile:
        json.dump(data, outfile)
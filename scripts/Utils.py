import networkx as nx
from gensim.models.doc2vec import Doc2Vec
from keras.models import model_from_json
import copy
import keras.backend as K
import numpy as np
import hashlib
from gensim.models.doc2vec import TaggedDocument
from std_msgs.msg import String

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """

    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])] + sorted([str(deg) for deg in degs])
            features = "_".join(features)
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing
        self.extracted_features = self.extracted_features + list(new_features.values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()


def exp_advanced(x):
    return K.minimum(K.exp(x), 5.0)


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


def normalize_feature(X, bound):
    X_bound = np.zeros((len(X)))
    for i_f, x in enumerate(X):
        min_F = np.min(bound[0, i_f])
        max_F = np.max(bound[1, i_f])
        X_bound[i_f] = 2. * ((x - min_F) / (max_F - min_F) - 0.5)
    return X_bound


def get_predicted_prob(model, X, Tau_binary, subset_tau, N_class):
    P_tau, P_tau_with_0 = get_probs_tau(model, X, subset_tau)
    marginal_probs_with_0 = get_marginal_prob(P_tau_with_0, Tau_binary, N_class)

    return marginal_probs_with_0


def get_probs_tau(model, X, subset_tau):
    n_s = len(X)
    n_tau = np.shape(subset_tau)[0]
    Exp_f__tau = model.predict(X)
    P_tau = np.zeros((n_s, n_tau))
    P_tau_with_0 = np.zeros((n_s, n_tau + 1))
    for i_s in range(0, n_s):
        for i_tau in range(0, n_tau):
            subset_tau_i = subset_tau[i_tau]
            idx_subset_tau = [i for i in range(0, n_tau) if subset_tau_i[i] == 1]
            p_tau = np.prod(Exp_f__tau[i_s][idx_subset_tau])
            P_tau[i_s][i_tau] = p_tau
        np.sum(P_tau[i_s])
        P_tau_with_0[i_s] = np.concatenate(
            (np.array([1]) / (1 + np.sum(P_tau[i_s])), P_tau[i_s] / (1 + np.sum(P_tau[i_s]))))
    return P_tau, P_tau_with_0


def get_marginal_prob(P_tau_with_0, Tau_binary, N_class):
    n = np.shape(P_tau_with_0)[0]
    n_tau = np.shape(Tau_binary)[0]
    marginal_probs_with_0 = np.zeros((n, N_class + 1))
    for i_s in range(0, n):
        for c in range(1, N_class + 1):
            idxs_tau = [i for i in range(0, n_tau) if Tau_binary[i][c - 1] == 1]
            idxs_tau_with_0 = [x + 1 for x in idxs_tau]
            marginal_probs_with_0[i_s, c] = np.sum(P_tau_with_0[i_s][idxs_tau_with_0])
        marginal_probs_with_0[i_s, 0] = 1 - np.sum(marginal_probs_with_0[i_s][1:])
    return marginal_probs_with_0


def get_models(path_embedding_model, path_MBM_architecture, path_MBM_weights):
    # load graph2vec model
    model_embedding = Doc2Vec.load(path_embedding_model)

    # load MBM model
    with open(path_MBM_architecture, 'r') as json_file:
        model_MBM = model_from_json(json_file.read(), custom_objects={'exp_advanced': exp_advanced})
    model_MBM.load_weights(path_MBM_weights)

    return model_embedding, model_MBM


def get_feature(G):
    feature = nx.get_node_attributes(G, 'type')
    for k, v in feature.items():
        feature[k] = str(v)
    return feature


def HGC(G, s):
    h = len(G.nodes())  # hypothesis node
    G_H = copy.deepcopy(G)
    G_H.add_nodes_from([(h, {'type': 0})])
    G_H.add_edges_from([(s, h)])
    return G_H


def predict_G2V(model_G2V, dG_list, df_list, wl_iterations, dim):
    X_d_list = np.zeros((len(dG_list), dim))
    for iter, (dG, df) in enumerate(zip(dG_list, df_list)):
        machine = WeisfeilerLehmanMachine(dG, df, wl_iterations)
        wl_feature = TaggedDocument(words=machine.extracted_features, tags=["g_" + '0'])
        X_d = model_G2V.infer_vector(wl_feature[0])
        X_d_list[iter] = X_d
    return X_d_list

def convert_G_to_str(G):
    # convert NN to dict
    dict_G = {}
    E = [[v1, v2] for (v1, v2) in list(nx.edges(G))]
    C = {str(n): str(c) for n, c in nx.get_node_attributes(G, 'type').items()}
    dict_G["edges"] = E
    dict_G["features"] = C

    G_str = String()
    G_str.data = str(dict_G)

    return G_str
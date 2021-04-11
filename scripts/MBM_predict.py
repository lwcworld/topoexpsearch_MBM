from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from keras.models import model_from_json
from Utils_graph2vec import *
from Utils_MBM import *
from Utils_graph2datapair import *
import copy
from matplotlib import pyplot as plt

# param
dim = 8
N_class = 5
level = 4

# path
path_embedding_model = '../dataset/model/graph_embedding/model_embedding_dim' + str(dim)
dir_MBM = '../dataset/model/MBM/'
path_MBM_architecture = dir_MBM + 'arc_lv4_iter0.json'
dir_param = '../dataset/param_c5/'

# load graph2vec model
model_embedding = Doc2Vec.load(path_embedding_model)

# load MBM model
with open(path_MBM_architecture, 'r') as json_file:
    model_MBM = model_from_json(json_file.read(), custom_objects={'exp_advanced': exp_advanced})

Tau_binary = np.load(dir_param + 'Tau_binary_lv' + str(level) + '.npy')
subset_tau = np.load(dir_param + 'subset_tau_lv' + str(level) + '.npy')

# load toy graph
NN = nx.Graph()
NN.add_nodes_from([(0, {'pos': (-28., 0.), 'type': 3, 'is_robot': True, 'to_go': False, 'value': 0}),
                   (1, {'pos': (-35., 0.), 'type': 2, 'is_robot': False, 'to_go': True, 'value': 0.1}),
                   (2, {'pos': (-28., 5.), 'type': 1, 'is_robot': False, 'to_go': True, 'value': 0.9}),
                   (3, {'pos': (-28., -5.), 'type': 2, 'is_robot': False, 'to_go': True, 'value': 0.1}),
                   (4, {'pos': (-21., 0.), 'type': 1, 'is_robot': False, 'to_go': True, 'value': 0.1})])
NN.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])

# preprocess networkx NN
# add hypothetical node
s = 1 # source ( interest ) node
h = len(NN.nodes()) # hypothesis node
NN_H = copy.deepcopy(NN)
NN_H.add_nodes_from([(h, {'type':0})])
NN_H.add_edges_from([(s,5)])

# get node feature (type)
feature = nx.get_node_attributes(NN_H, 'type')
for k, v in feature.items():
    feature[k] = str(v)

# get tree graph & feature
tree_graph, feature = get_bfstree(NN_H, feature, root=h, depth_limit=3)

# decompose
dG_list, df_list = decompose_tree_graph(tree_graph, feature)

# get wl_feature
wl_iterations = 2
vec_array = np.zeros((len(dG_list), dim))
for iter, (dG, df) in enumerate(zip(dG_list, df_list)):
    machine = WeisfeilerLehmanMachine(dG, df, wl_iterations)
    wl_feature = TaggedDocument(words=machine.extracted_features, tags=["g_" + '0'])
    vec = model_embedding.infer_vector(wl_feature[0])
    vec_array[iter] = vec

# get marginal probabilities of decomposed graphs
mp_d = get_predicted_prob(model_MBM, vec_array, Tau_binary, subset_tau, N_class)
o = np.mean(mp_d, 0)

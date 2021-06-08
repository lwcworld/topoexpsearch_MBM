#!/usr/bin/env python3
import os
print(os.getcwd())
import rospy
from topoexpsearch_MBM.srv import pred_MBM
from gensim.models.doc2vec import Doc2Vec
from keras.models import model_from_json
from Utils_MBM import *
from Utils_graph2datapair import *
from Utils_graph2vec import *
from std_msgs.msg import String
from ast import literal_eval

class cb_srv():
    def __init__(self):
        # param
        self.dim = 8
        self.N_class = 5
        self.level = 4

        # path
        dir_package = '/home/lwcubuntu/workspaces/topoexpsearch/src/topoexpsearch_MBM/'
        path_embedding_model = dir_package + 'dataset/model/graph_embedding/model_embedding_dim' + str(self.dim)
        path_MBM_architecture = dir_package + 'dataset/model/MBM/arc_lv4_iter2.json'
        dir_param = dir_package + 'dataset/param_c5/'

        # load graph2vec model
        self.model_embedding = Doc2Vec.load(path_embedding_model)

        # load MBM model
        with open(path_MBM_architecture, 'r') as json_file:
            self.model_MBM = model_from_json(json_file.read(), custom_objects={'exp_advanced': exp_advanced})

        # load MBM model params
        self.Tau_binary = np.load(dir_param + 'Tau_binary_lv' + str(self.level) + '.npy')
        self.subset_tau = np.load(dir_param + 'subset_tau_lv' + str(self.level) + '.npy')

    def handle_predMBM(self, msg):
        NN_jsonstr = msg.NN_jsonstr.data # navigation network (json string type)
        s = msg.node # interset node

        NN_json = literal_eval(NN_jsonstr)
        NN = nx.from_edgelist(NN_json["edges"])
        if "features" in NN_json.keys():
            features = NN_json["features"]
        # else:
        #     features = nx.degree(NN)
        features = {int(k): v for k, v in features.items()}

        for k, v in features.items():
            NN.nodes[k]['type'] = v

        # preprocess networkx NN
        # add hypothetical node
        h = len(NN.nodes())  # hypothesis node
        NN_H = copy.deepcopy(NN)
        NN_H.add_nodes_from([(h, {'type': 0})])
        NN_H.add_edges_from([(s, h)])

        # get node feature (type)
        feature = nx.get_node_attributes(NN_H, 'type')
        for k, v in feature.items():
            feature[k] = str(v)

        # get tree graph & feature
        tree_graph, feature = get_bfstree(NN_H, feature, root=h, depth_limit=3)

        # decompose
        dG_list, df_list = decompose_tree_graph(tree_graph, feature)
        print('--------')
        print(tree_graph.nodes())
        for dG, df in zip(dG_list,df_list):
            print(dG.nodes())
            print(df)

        # graph2vec predict
        wl_iterations = 2
        vec_array = np.zeros((len(dG_list), self.dim))
        for iter, (dG, df) in enumerate(zip(dG_list, df_list)):
            machine = WeisfeilerLehmanMachine(dG, df, wl_iterations)
            wl_feature = TaggedDocument(words=machine.extracted_features, tags=["g_" + '0'])
            vec = self.model_embedding.infer_vector(wl_feature[0])
            vec_array[iter] = vec

        mp_d = get_predicted_prob(self.model_MBM, vec_array, self.Tau_binary, self.subset_tau, self.N_class)
        mp_o = list(np.mean(mp_d, 0))

        output = String()
        output.data = str(mp_o)

        return output

if __name__ == '__main__':
    rospy.init_node('MBM_server')
    cb_srv = cb_srv()

    s = rospy.Service('pred_MBM', pred_MBM, cb_srv.handle_predMBM)
    rospy.spin()
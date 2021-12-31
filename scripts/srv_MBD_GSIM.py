#!/usr/bin/env python3
import os
print(os.getcwd())
import rospy
from topoexpsearch_MBM.srv import MBD_GSIM
from Utils_MBD_GSIM import *
from std_msgs.msg import String
from ast import literal_eval

class cb_srv():
    def __init__(self):
        # param
        self.dim = 32
        self.N_class = 5
        self.level = 4
        self.wl_iterations = 10
        self.place_category = {1: 'office', 2: 'corridor', 3: 'share', 4: 'maintenance', 5: 'toilet'}

        # path
        dir_package = '/home/lwcubuntu/workspaces/topoexpsearch/src/topoexpsearch_MBM/'
        dir_model = dir_package + 'model/MBD_GSIM/'
        dir_model_G2V = dir_model + 'G2V/'
        dir_model_MBM = dir_model + 'MBM/'
        dir_MBM_param = dir_model + 'MBM/param/'

        path_embedding_model =  dir_model_G2V + 'model_embedding_dim' + str(self.dim)
        path_X_bound =  dir_model_G2V + 'X_bound_lv' + str(self.level) + '_dim' + str(self.dim) + '.npy'
        path_MBM_architecture =  dir_model_MBM + 'arc_lv4_iter1.json'
        path_MBM_weights =  dir_model_MBM + 'weight_lv4_iter1.h5'
        path_param_Tau_binary =  dir_MBM_param + 'Tau_binary_lv' + str(self.level) + '.npy'
        path_param_subset_tau =  dir_MBM_param + 'subset_tau_lv' + str(self.level) + '.npy'

        self.feature_bound = np.load(path_X_bound)

        self.model_embedding, self.model_MBM = get_models(path_embedding_model, path_MBM_architecture, path_MBM_weights)

        self.Tau_binary = np.load(path_param_Tau_binary)
        self.subset_tau = np.load(path_param_subset_tau)

    def handle_predMBM(self, msg):
        NN_jsonstr = msg.NN_jsonstr.data # navigation network (json string type)
        s = msg.node # interset node

        NN_json = literal_eval(NN_jsonstr)
        NN = nx.from_edgelist(NN_json["edges"])
        if "features" in NN_json.keys():
            features = NN_json["features"]
        features = {int(k): v for k, v in features.items()}

        for k, v in features.items():
            NN.nodes[k]['type'] = v

        edges = list(NN.edges())
        for edge in edges:
            if edge[0]==edge[1]:
                NN.remove_edge(edge[0], edge[1])

        # HGC
        NN_H = HGC(NN, s)

        # bfs tree
        feature = get_feature(NN_H)
        h = len(NN_H.nodes()) - 1
        tree_graph, feature = get_bfstree(NN_H, feature, root=h, depth_limit=3)

        # decompose
        dG_list, df_list = decompose_tree_graph(tree_graph, feature)

        # graph2vec predict
        X_d_list = predict_G2V(self.model_embedding, dG_list, df_list, self.wl_iterations, self.dim)

        # normalize feature vector
        X_d_norm_list = np.zeros(np.shape(X_d_list))
        for i_X, X_d in enumerate(X_d_list):
            X_d_norm_list[i_X] = normalize_feature(X_d, self.feature_bound)

        # MBM prediction
        mp_d = get_predicted_prob(self.model_MBM, X_d_norm_list, self.Tau_binary, self.subset_tau, self.N_class)
        mp_o = np.mean(mp_d, 0)

        output = String()
        MP_str = str(mp_o)
        while '  ' in MP_str:
            MP_str = str(MP_str).replace('  ', ' ')
        MP_str = str(MP_str).replace(' ', ',')
        output.data = MP_str

        return output

if __name__ == '__main__':
    rospy.init_node('MBM_server')
    cb_srv = cb_srv()

    s = rospy.Service('MBD_GSIM', MBD_GSIM, cb_srv.handle_predMBM)
    rospy.spin()
#!/usr/bin/env python3
import os
print(os.getcwd())
import rospy
from topoexpsearch_MBM.srv import FSE
from Utils_FSE import *
from std_msgs.msg import String
from ast import literal_eval

class cb_srv():
    def __init__(self):
        # param
        self.N_class = 5
        self.place_category = {1: 'office', 2: 'corridor', 3: 'share', 4: 'maintenance', 5: 'toilet'}

        dir_package = '/home/lwcubuntu/workspaces/topoexpsearch/src/topoexpsearch_MBM/'
        dir_data_floorplans_subgraphs = dir_package + 'model/FSE/subgraphs'

        # subgraphs database
        n_list = [2, 3, 4]
        for i, n in enumerate(n_list):
            if i == 0:
                self.D = np.load(dir_data_floorplans_subgraphs + '_n' + str(n) + '.npy', allow_pickle=True).item()
            else:
                d = np.load(dir_data_floorplans_subgraphs + '_n' + str(n) + '.npy', allow_pickle=True).item()
                self.D['subgraphs'].extend(d['subgraphs'])
                self.D['where'].extend(d['where'])

        # subgraphs for C
        n_list = [2, 3]
        for i, n in enumerate(n_list):
            if i == 0:
                self.D_s = np.load(dir_data_floorplans_subgraphs + '_n' + str(n) + '.npy', allow_pickle=True).item()
            else:
                d = np.load(dir_data_floorplans_subgraphs + '_n' + str(n) + '.npy', allow_pickle=True).item()
                self.D_s['subgraphs'].extend(d['subgraphs'])
                self.D_s['where'].extend(d['where'])

    def handle_predMBM(self, msg):
        NN_jsonstr = msg.NN_jsonstr.data # navigation network (json string type)
        s = msg.node # interset node

        NN_json = literal_eval(NN_jsonstr)
        NN = nx.from_edgelist(NN_json["edges"])
        if "features" in NN_json.keys():
            features = NN_json["features"]
        features = {int(k): v for k, v in features.items()}

        for k, v in features.items():
            NN.nodes[k]['label'] = str(v)

        edges = list(NN.edges())
        for edge in edges:
            if edge[0]==edge[1]:
                NN.remove_edge(edge[0], edge[1])
            else:
                if edge[0]>edge[1]:
                    NN.remove_edge(edge[0], edge[1])
                    NN.add_edge(edge[1], edge[0])
                    NN[edge[1]][edge[0]]['label'] = '0'
                else:
                    NN[edge[0]][edge[1]]['label'] = '0'


        # NN = nx.ego_graph(NN, s, radius=2)
        # NN, mapping = relabel_network(NN)
        #
        # print('======')
        # print(NN.edges(data=True))
        # print(NN.nodes(data=True))

        # print(self.D_s['subgraphs'][300].edges(data=True))
        # print(self.D_s['subgraphs'][300].nodes(data=True))

        c_pred, P_H_pred = node_prediction(NN, s, self.D, self.D_s, self.N_class, 0.8)
        # c_pred, P_H_pred = node_prediction(self.D_s['subgraphs'][300], 2, self.D, self.D_s, self.N_class, 0.8)

        # print(P_H_pred)
        output = String()
        MP_str = str(P_H_pred)

        output.data = MP_str

        return output

if __name__ == '__main__':
    rospy.init_node('FSE_server')
    cb_srv = cb_srv()

    s = rospy.Service('FSE', FSE, cb_srv.handle_predMBM)
    rospy.spin()



import networkx as nx
import rospy
from topoexpsearch_MBM.srv import pred_MBM
from std_msgs.msg import String
import ast
import numpy as np

NN = nx.Graph()
NN.add_nodes_from([(0, {'pos': (-28., 0.), 'type': 4, 'is_robot': True, 'to_go': False, 'value': 0}),
                   (1, {'pos': (-35., 0.), 'type': 4, 'is_robot': False, 'to_go': True, 'value': 0.1}),
                   (2, {'pos': (-28., 5.), 'type': 1, 'is_robot': False, 'to_go': True, 'value': 0.9}),
                   (3, {'pos': (-28., -5.), 'type': 4, 'is_robot': False, 'to_go': True, 'value': 0.1}),
                   (4, {'pos': (-21., 0.), 'type': 4, 'is_robot': False, 'to_go': True, 'value': 0.1}),
                   (5, {'pos': (-21., 0.), 'type': 2, 'is_robot': False, 'to_go': True, 'value': 0.1}),
                   (6, {'pos': (-21., 0.), 'type': 1, 'is_robot': False, 'to_go': True, 'value': 0.1}),
                   (7, {'pos': (-21., 0.), 'type': 1, 'is_robot': False, 'to_go': True, 'value': 0.1})])
NN.add_edges_from([(0, 1), (0,2), (0, 3), (0, 4), (4, 5), (5, 6), (6,7)])

# convert NN to dict
dict_G = {}
E = [[v1,v2] for (v1,v2) in list(nx.edges(NN))]
C = {str(n):str(c) for n, c in nx.get_node_attributes(NN, 'type').items()}
dict_G["edges"] = E
dict_G["features"] = C

msg_NN_jsonstr = String()
msg_NN_jsonstr.data = str(dict_G)
node = 5

rospy.wait_for_service('pred_MBM')
for i in range(0,3):
    srv_pred_MBM = rospy.ServiceProxy('pred_MBM', pred_MBM)
    output = srv_pred_MBM(msg_NN_jsonstr, node)
    MP = ast.literal_eval(output.marginal_probs.data)
    print(MP)
    print(np.argmax(MP))

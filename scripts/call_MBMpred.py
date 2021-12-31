import networkx as nx
import rospy
from topoexpsearch_MBM.srv import pred_MBM
import ast
from Utils_MBD_GSIM import *


# query
NN = nx.Graph()
NN.add_nodes_from([(0, {'pos': (-28., 0.), 'type': 2, 'is_robot': True, 'to_go': False, 'value': 0}),
                   (1, {'pos': (-35., 0.), 'type': 3, 'is_robot': False, 'to_go': True, 'value': 0.1}),
                   (2, {'pos': (-28., 5.), 'type': 3, 'is_robot': False, 'to_go': True, 'value': 0.9}),
                   (3, {'pos': (-28., -5.), 'type': 2, 'is_robot': False, 'to_go': True, 'value': 0.1}),
                   (4, {'pos': (-28., -5.), 'type': 4, 'is_robot': False, 'to_go': True, 'value': 0.1})])
NN.add_edges_from([(0,1), (0,2), (0,3), (0,4)])
s = 0 # source ( interest ) node

# convert NN to dict
msg_NN_jsonstr = convert_G_to_str(NN)

rospy.wait_for_service('pred_MBM')
for i in range(0,3):
    srv_pred_MBM = rospy.ServiceProxy('pred_MBM', pred_MBM)
    output = srv_pred_MBM(msg_NN_jsonstr, s)
    MP_str = output.marginal_probs.data

    MP = ast.literal_eval(MP_str)


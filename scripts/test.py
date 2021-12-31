#!/usr/bin/env python3
import os
print(os.getcwd())
import rospy
from topoexpsearch_MBM.srv import FSE
from Utils_FSE import *
from std_msgs.msg import String
from ast import literal_eval

dir_package = '/home/lwcubuntu/workspaces/topoexpsearch/src/topoexpsearch_MBM/'
dir_data_floorplans_subgraphs = dir_package + 'model/FSE/subgraphs'

# subgraphs database
n_list = [2, 3, 4]
for i, n in enumerate(n_list):
    if i == 0:
        D = np.load(dir_data_floorplans_subgraphs + '_n' + str(n) + '.npy', allow_pickle=True).item()
    else:
        d = np.load(dir_data_floorplans_subgraphs + '_n' + str(n) + '.npy', allow_pickle=True).item()
        D['subgraphs'].extend(d['subgraphs'])
        D['where'].extend(d['where'])

print(D['subgraphs'][0].nodes(data=True))
print(len(D['subgraphs']))
De = D['subgraphs'][300]

print(De.nodes(data=True))
print(De.edges(data=True))
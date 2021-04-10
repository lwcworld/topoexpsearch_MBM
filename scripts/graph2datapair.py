from Utiles_graph2datapair import *

np.random.seed(101)

# set dir
dir_in = '../dataset/dataset_graph_floorplan_json/'
dir_out_PO = '../dataset/dataset_PO_floorplan_c5/'
dir_out_label = '../dataset/dataset_label_floorplan_c5/'


# load one graph and feature from NCI1 data
list_data = list(range(0,20))
i_s = 0
i_o = 0
max_class = 5
for i_d in list_data:
    print('-- data # :' + str(i_d))
    # read data
    path = dir_in + str(list_data[i_d]) + '.json'
    graph, feature, name = dataset_reader(path)
    class_list = list(map(int, list(feature.values())))

    if any(c > max_class for c in class_list):
        print('pass!!')
        continue

    # generate parital data
    # 1. generate random walks
    walks = make_random_walk_from_G(graph, NUM_WALKS=40, WALK_LENGTH=10, RETURN_PARAMS = 0.0)
    # 2. generate data pairs (previous partially observed state & current state)
    datas_prev, datas_current, i_o = generate_dataset(graph, feature, walks, i_o)

    for i_p in range(0,len(datas_prev)):
        print('---- partial # :' + str(i_p) + ' || ' + 'save # :' + str(i_s))
        save_data(file_name=str(i_s)+'.json', dir=dir_out_PO, data=datas_prev[i_p])
        save_data(file_name=str(i_s)+'.json', dir=dir_out_label, data=datas_current[i_p])
        i_s = i_s + 1

print('finished')
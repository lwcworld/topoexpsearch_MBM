from Utils_MBM import *
import keras.backend as K
from pyDOE2 import *
from keras.layers import Input, Dense
from keras.models import Sequential
from keras.models import model_from_json
from keras import regularizers
from keras.layers import Dropout
import tensorflow as tf
import pickle
from tabulate import tabulate

# do toggle
do_construct_Tau = False
do_preprocess = True
do_train_model = False
do_eval = True
do_show_eval_result = True

# data parameters
N_class = 5
level = 4
dim = 8
N_data = 3000
i_mc = 100 # only has mean when MC simulation is performed

# directories of dataset pair
path_feature = '../dataset/feature_PO_floorplan_c5/dim'+str(dim)+'.csv'
dir_label = '../dataset/dataset_label_floorplan_c5/'
dir_MBM = '../dataset/model/MBM/'
path_MBM_architecture = dir_MBM + 'arc_lv4_iter0.json'
path_MBM_weight = dir_MBM + 'weight_lv4_iter0.h5'
dir_eval = '../dataset/evaluation/'


# Tau construction (don't need to run for each simulation)
if do_construct_Tau == True:
    Tau_binary = set_indice_binary(K=N_class*level, N_class=N_class, N_level=level, max_per_level=1)
    n_tau = np.shape(Tau_binary)[0]
    subset_tau = np.zeros((n_tau,n_tau))
    print('------')
    print(level, n_tau)
    for i_tau in range(0,n_tau):
        print(i_tau)
        tau_binary = Tau_binary[i_tau]
        subset_tau_idxs = [i for i in range(0,n_tau) if K.min(tau_binary-Tau_binary,axis=1)[i]>=0]
        subset_tau[i_tau][subset_tau_idxs] = 1
    np.save('../dataset/param' + '_c' + str(N_class) +'/Tau_binary_lv'+str(level), Tau_binary)
    np.save('../dataset/param' + '_c' + str(N_class) +'/subset_tau_lv'+str(level), subset_tau)
    print('done')
else:
    Tau_binary = np.load('../dataset/param' + '_c' + str(N_class) + '/Tau_binary_lv' + str(level) + '.npy')
    subset_tau = np.load('../dataset/param' + '_c' + str(N_class) + '/subset_tau_lv' + str(level) + '.npy')

if do_preprocess == True:
    X, Y_max, C, d2o = dataset_generator(path_feature, dir_label, N_class=N_class, N_level=level, N_data=N_data)
    Y = Y_max[:,0:level*N_class]
    np.save('../dataset/dataset_preprocess_floorplan_c5/X_lv'+str(level)+'_dim'+str(dim), X)
    np.save('../dataset/dataset_preprocess_floorplan_c5/Y_lv'+str(level), Y)
    np.save('../dataset/dataset_preprocess_floorplan_c5/C_lv'+str(level), C)
    np.save('../dataset/dataset_preprocess_floorplan_c5/d2o_lv'+str(level), d2o)
else:
    X = np.load('../dataset/dataset_preprocess_floorplan_c5/X_lv' + str(level) + '_dim' + str(dim) + '.npy')
    Y = np.load('../dataset/dataset_preprocess_floorplan_c5/Y_lv' + str(level) + '.npy')
    C = np.load('../dataset/dataset_preprocess_floorplan_c5/C_lv' + str(level) + '.npy')
    d2o = np.load('../dataset/dataset_preprocess_floorplan_c5/d2o_lv' + str(level) + '.npy')


# learning param
per_train = 0.9
data_q = 100
batch_size_train = 100
batch_size_test = 100
regul = 1e-3

tf.debugging.set_log_device_placement(False)

with tf.device("/GPU:0"):
    n_tau = np.shape(Tau_binary)[0]

    # sample train & test dataset
    X_train, Y_train, C_train, d2o_train, X_test, Y_test, C_test, d2o_test = uniform_class_sample_ver2(X, Y, C, d2o,
                                                                                                       margin=10,
                                                                                                       N_class=5,
                                                                                                       per_train=per_train)
    N_train = np.shape(X_train)[0]
    N_test = np.shape(X_test)[0]

    # data cut for batch learning
    X_train, Y_train, C_train, d2o_train = quotient_dataset(X_train, Y_train, C_train, d2o_train, data_q)
    X_test, Y_test, C_test, d2o_test = quotient_dataset(X_test, Y_test, C_test, d2o_test, data_q)

    if do_train_model == True:
        # build model
        F = len(X_train[0])
        model = Sequential()
        model.add(Input(shape=(F,)))
        model.add(Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(regul), activity_regularizer=regularizers.l2(regul)))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(regul), activity_regularizer=regularizers.l2(regul)))
        model.add(Dropout(0.5))
        model.add(Dense(n_tau, activation='tanh', kernel_regularizer=regularizers.l2(regul), activity_regularizer=regularizers.l2(regul)))
        model.add(Dropout(0.5))
        model.add(Dense(n_tau, activation=exp_advanced))

        model.compile(loss=mb_nll(Tau_binary=Tau_binary, subset_tau=subset_tau), optimizer='Adam')
        model.summary()

        hist = model.fit(x=X_train, y=Y_train, batch_size = batch_size_train, validation_data=(X_test, Y_test), validation_batch_size=batch_size_test, epochs = 500, shuffle=True, verbose=1)
        # save model
        save_model_weight(dir_MBM, model, level, i_mc)

        with open(dir_MBM + 'hist_lv' + str(level) + '_iter' + str(i_mc), 'wb') as f:
            pickle.dump(hist.history, f)
    else:
        with open(path_MBM_architecture, 'r') as json_file:
            model = model_from_json(json_file.read(), custom_objects={'exp_advanced':exp_advanced})
        model.load_weights(path_MBM_weight)

    if do_eval==True:
        print('evaluation start')
        acc_l_d_train, acc_mp_d_train, acc_l_o_train, acc_mp_o_train = eval(model,
                                                                            X_train[0:10],
                                                                            Y_train[0:10],
                                                                            C_train[0:10],
                                                                            d2o_train[0:10],
                                                                            Tau_binary, subset_tau, N_class)
        print('train data evaluation done')
        acc_l_d_test, acc_mp_d_test, acc_l_o_test, acc_mp_o_test = eval(model,
                                                                        X_test[0:10],
                                                                        Y_test[0:10],
                                                                        C_test[0:10],
                                                                        d2o_test[0:10],
                                                                        Tau_binary, subset_tau, N_class)
        print('test data evaluation done')
        np.save(dir_eval + 'acc_l_d_train_lv' + str(level) + '_iter' + str(i_mc), acc_l_d_train)
        np.save(dir_eval + 'acc_mp_d_train_lv' + str(level) + '_iter' + str(i_mc), acc_mp_d_train)
        np.save(dir_eval + 'acc_l_d_test_lv' + str(level) + '_iter' + str(i_mc), acc_l_d_test)
        np.save(dir_eval + 'acc_mp_d_test_lv' + str(level) + '_iter' + str(i_mc), acc_mp_d_test)
        np.save(dir_eval + 'acc_l_o_train_lv' + str(level) + '_iter' + str(i_mc), acc_l_o_train)
        np.save(dir_eval + 'acc_mp_o_train_lv' + str(level) + '_iter' + str(i_mc), acc_mp_o_train)
        np.save(dir_eval + 'acc_l_o_test_lv' + str(level) + '_iter' + str(i_mc), acc_l_o_test)
        np.save(dir_eval + 'acc_mp_o_test_lv' + str(level) + '_iter' + str(i_mc), acc_mp_o_test)

        print('           ----- decomposed data result -----')
        table = [[acc_l_d_train, acc_mp_d_train, acc_l_d_test, acc_mp_d_test]]
        print(tabulate(table, headers=['label (train)', 'M.P. (train)', 'label (test)', 'M.P. (test)'],
                       colalign=("center", "center", "center", "center")))

        print('           ----- original data result -----')
        table = [[acc_l_o_train, acc_mp_o_train, acc_l_o_test, acc_mp_o_test]]
        print(tabulate(table, headers=['label (train)', 'M.P. (train)', 'label (test)', 'M.P. (test)'],
                       colalign=("center", "center", "center", "center")))















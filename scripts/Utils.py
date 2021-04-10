import keras.backend as K
from pyDOE2 import *
import random
import json
import csv

def tanh_advanced(x):
    return K.tanh(x)

def exp_advanced(x):
    return K.minimum(K.exp(x), 5.0)

def uniform_class_sample(X, Y, C, d2o, margin, N_class):
    N_data = np.shape(X)[0]
    idx_class = []
    min_N = 100000
    for i_c in range(0, N_class):
        idx_c = [i for i in range(0, N_data) if C[i] == (i_c + 1)]
        idx_class.append(idx_c)
        if len(idx_c) < min_N:
            min_N = len(idx_c)

    idx_sample = []
    for i_c in range(0, N_class):
        idx_sample = idx_sample + random.sample(idx_class[i_c], min(min_N + margin, len(idx_class[i_c])))
    random.shuffle(idx_sample)

    X_s = X[idx_sample]
    Y_s = Y[idx_sample]
    C_s = C[idx_sample]
    d2o_s = d2o[idx_sample]

    N_s = np.shape(X_s)[0]

    return X_s, Y_s, C_s, d2o_s


def uniform_class_sample_ver2(X, Y, C_d, d2o, margin, N_class, per_train):
    N_d = np.shape(d2o)[0]
    N_o = len(np.unique(d2o))

    # get original data to dicompose data matching
    o2d = [[] for i in range(0, N_o)]
    for i_d, i_o in enumerate(d2o):
        i_o = int(i_o)
        if i_d not in o2d[i_o]:
            o2d[i_o].append(i_d)

    # get original data category
    C_o = np.zeros(N_o)
    for i_o in range(0, N_o):
        C_o[i_o] = C_d[o2d[i_o][0]]

    # number of classes
    N_C = np.zeros(N_class)
    for c in C_o:
        N_C[int(c) - 1] = N_C[int(c) - 1] + 1

    # get number of samples for each classes
    N_s = np.min(N_C) + margin

    # for each class, sample original train and test dataset indexes
    idx_o_list_train = []
    idx_o_list_test = []
    for c in range(1, N_class + 1):
        idx_c = list(np.where(C_o == c)[0])
        idx_o = random.sample(idx_c, int(min(N_s, N_C[c - 1])))
        N_o = len(idx_o)
        idx_o_train = idx_o[0:int(N_o * per_train)]
        idx_o_test = idx_o[int(N_o * per_train):]
        idx_o_list_train.extend(idx_o_train)
        idx_o_list_test.extend(idx_o_test)
    idx_o_list_train.sort()
    idx_o_list_test.sort()

    # get decomposed dataset indexes from original indexes
    idx_d_list_train = []
    idx_d_list_test = []
    for idx in idx_o_list_train:
        idx_d_list_train.extend(o2d[idx])
    for idx in idx_o_list_test:
        idx_d_list_test.extend(o2d[idx])

    X_train = X[idx_d_list_train]
    Y_train = Y[idx_d_list_train]
    C_train = C_d[idx_d_list_train]
    d2o_train = d2o[idx_d_list_train]
    X_test = X[idx_d_list_test]
    Y_test = Y[idx_d_list_test]
    C_test = C_d[idx_d_list_test]
    d2o_test = d2o[idx_d_list_test]

    return X_train, Y_train, C_train, d2o_train, X_test, Y_test, C_test, d2o_test


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


def get_class_true(Y, N_class):
    n_s = np.shape(Y)[0]
    classes = np.zeros((n_s))
    for i_s in range(0, n_s):
        if all(Y[i_s][0:N_class] == 0):
            classes[i_s] = 0
        else:
            classes[i_s] = np.argmax(Y[i_s][0:N_class]) + 1
    return classes


def get_predicted_prob(model, X, Tau_binary, subset_tau, N_class):
    P_tau, P_tau_with_0 = get_probs_tau(model, X, subset_tau)
    marginal_probs_with_0 = get_marginal_prob(P_tau_with_0, Tau_binary, N_class)

    return marginal_probs_with_0


def dataset_generator(dir_feature, dir_label, N_class, N_level, N_data):
    # input data load
    dat = open(dir_feature)
    reader = csv.reader(dat)
    lines = list(reader)
    F = len(lines[1][1:])

    # input data memory
    X = np.zeros((N_data, F))

    # output data memory
    Y = np.zeros((N_data, N_class * N_level))

    i_cand = 0  # candidate data id
    i_d = 0  # id to save
    C = np.zeros((N_data))

    d2o = np.zeros((N_data))
    while i_d < N_data:
        # output data parsing
        path = dir_label + str(i_cand) + '.json'
        name = path.strip(".json").split("/")[-1]
        data = json.load(open(path))

        Y_2d = np.zeros((N_level, N_class))
        n_nodes = len(data['edges']) + 1

        d2o[i_d] = data['idx']

        # flag_except = False
        for i_l in range(0, min(N_level, n_nodes)):
            if i_l < n_nodes - 1:
                node = data['edges'][i_l][0]
            else:
                node = data['edges'][i_l - 1][1]

            c = int(data['features'][str(node)])
            if i_l == 0:
                c_d = c

            Y_2d[i_l, c - 1] = 1

        X[i_d] = lines[i_cand + 1][1:]
        X = 2. * ((X - np.min(X)) / (np.max(X) - np.min(X)) - 0.5)  # normalize
        Y[i_d] = np.reshape(Y_2d, (-1,))
        C[i_d] = c_d

        i_d = i_d + 1
        i_cand = i_cand + 1
    return X, Y, C, d2o


def dataset_divide(X, Y, C, d2o, idx_train):
    N_data = np.shape(X)[0]
    idx_data = list(range(0, N_data))
    idx_test = np.setdiff1d(idx_data, idx_train)

    X_train = X[idx_train]
    Y_train = Y[idx_train]
    C_train = C[idx_train]
    d2o_train = d2o[idx_train]

    X_test = X[idx_test]
    Y_test = Y[idx_test]
    C_test = C[idx_test]
    d2o_test = d2o[idx_test]
    return X_train, Y_train, C_train, d2o_train, X_test, Y_test, C_test, d2o_test


def set_indice_binary(K, N_class=4, N_level=4, max_per_level=1):
    set = ff2n(K)
    set[set == -1] = 0
    set = np.delete(set, (0), axis=0)

    idxs = [i for i in range(0, np.shape(set)[0]) if sum(set[i, 0:N_class]) <= 1]
    for i_l in range(1, N_level):
        idxs_i_l = [i for i in range(0, np.shape(set)[0]) if sum(set[i, N_class * i_l:N_class * (i_l + 1)]) <= 1]
        idxs = np.intersect1d(idxs, idxs_i_l)
    idxs = np.sort(idxs)
    set = set[idxs]
    return set


def mb_nll(Tau_binary, subset_tau):
    def loss(ytrues, ypreds):
        n_s = int(ytrues.shape[0])
        n_tau = int(ypreds.shape[1])
        dim = int(ytrues.shape[1])

        Tau_binary_conc = K.concatenate([Tau_binary] * n_s, axis=1)
        subset_tau_conc = K.concatenate([subset_tau] * n_s, axis=1)

        f__tau = K.flatten(K.transpose(K.log(ypreds)))
        f__tau = K.cast(f__tau, 'double')
        ytrues_flat = K.reshape(ytrues, (-1,))
        ytrues_flat = K.cast(ytrues_flat, 'double')
        B__tau = K.equal(K.sum(K.reshape(ytrues_flat * Tau_binary_conc, (-1, dim)), axis=1),
                         K.sum(K.reshape(Tau_binary_conc, (-1, dim)), axis=1))
        B__tau = K.cast(B__tau, 'double')
        sumfB = K.sum(f__tau * B__tau)

        f__tau = K.flatten(K.log(ypreds))

        f__tau = K.cast(f__tau, 'double')
        S__tau = K.sum(K.reshape(f__tau * subset_tau_conc, (-1, n_tau)), 1)
        ExpS__tau = K.exp(K.reshape(K.transpose(S__tau), (-1, n_s)))
        SumExpS = K.sum(ExpS__tau, 0)
        b = K.sum(K.log(1.0 + SumExpS))

        NLL_sum = - sumfB + b

        return NLL_sum / n_s

    return loss


def accuracy_label(class_true, class_pred):
    n_s = len(class_true)
    return sum(class_true == class_pred) / n_s


def accuracy_marginal_prob(class_true, marginal_probs_with_0):
    n_s = len(class_true)
    class_prob = np.zeros((n_s))
    for i_s in range(0, n_s):
        class_prob[i_s] = marginal_probs_with_0[i_s][int(class_true[i_s])]
    return np.mean(class_prob)


def eval(model, X, Y, C, d2o, Tau_binary, subset_tau, N_class):
    N_d = np.shape(X)[0]

    mp_w0_d = get_predicted_prob(model, X, Tau_binary, subset_tau, N_class)
    C_pred_d = np.argmax(mp_w0_d, axis=1)

    acc_l_d = accuracy_label(C, C_pred_d)
    acc_mp_d = accuracy_marginal_prob(C, mp_w0_d)

    o_list = np.unique(d2o)
    N_o = len(o_list)

    mp_w0_o = np.zeros((N_o, N_class + 1))
    C_o = np.zeros((N_o))
    for i, o in enumerate(o_list):
        mp_w0_o[i] = np.mean(mp_w0_d[d2o == o], 0)
        C_o[i] = C[np.where(d2o == o)[0][0]]
    C_pred_o = np.argmax(mp_w0_o, axis=1)

    acc_l_o = accuracy_label(C_o, C_pred_o)
    acc_mp_o = accuracy_marginal_prob(C_o, mp_w0_o)

    return acc_l_d, acc_mp_d, acc_l_o, acc_mp_o


def save_model_weight(dir, model, lv, i_mc):
    model_json = model.to_json()
    with open(dir + "arc_lv" + str(lv) + '_iter' + str(i_mc) + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(dir + "weight_lv" + str(lv) + '_iter' + str(i_mc) + ".h5")


def quotient_dataset(X_train, Y_train, C_train, d2o_train, divide):
    N = np.shape(X_train)[0]
    r = N % divide
    q = N - r

    return X_train[0:q], Y_train[0:q], C_train[0:q], d2o_train[0:q]
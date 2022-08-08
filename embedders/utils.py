import numpy as np
import networkx as nx
import scipy.sparse

import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l1_l2

def create_alias_table(area_ratio):
    """
    :param area_ratio: sum(area_ratio) = 1
    :return: accept, alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1:
            small.append(i)
        else:
            large.append(i)
    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] -= 1 - area_ratio_[small_idx]
        if area_ratio_[large_idx] < 1:
            small.append(large_idx)
        else:
            large.append(large_idx)
    while large:
        accept[large.pop()] = 1
    while small:
        accept[small.pop()] = 1
    return accept, alias

def alias_sample(accept, alias):
    """
    :param accept:
    :param alias:
    :return: sample index
    """
    i = int(np.random.random() * len(accept))
    if np.random.random() < accept[i]:
        return i
    else:
        return alias[i]

def preprocess_nxgraph(graph):
    """
    :param graph:
    :return: idx2node, node2idx
    """
    node2idx, idx2node = {}, []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx

def partition_dict(vertices, workers):
    """
    :param vertices:
    :param workers:
    :return: part_list
    """
    batch_size = (len(vertices) - 1) // workers + 1
    part_list, part = [], []
    count = 0
    for v1, nbs in vertices.items():
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list

def partition_list(vertices, workers):
    """
    :param vertices:
    :param workers:
    :return: part_list
    """
    batch_size = (len(vertices) - 1) // workers + 1
    part_list, part = [], []
    count = 0
    for v1, nbs in enumerate(vertices):
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list

def partition_num(num, workers):
    """
    :param num:
    :param workers:
    :return: partition
    """
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]

def chooseNeighbor(v, graphs, layers_alias, layers_accept, layer):
    """
    :param v:
    :param graphs:
    :param layers_alias:
    :param layers_accept:
    :param layer:
    :return node"""
    v_list = graphs[layer][v]
    idx = alias_sample(layers_accept[layer][v], layers_alias[layer][v])
    return v_list[idx]

def l_2nd(beta):
    def loss_2nd(y_true, y_pred):
    #    b_ = K.ones_like(y_true)
    #    b_[y_true != 0] = beta
    #    x = K.square((y_true - y_pred) * b_)
    #    t = K.sum(x, axis=-1, )
    #    return K.mean(t)
        b_ = K.ones_like(y_true)
        x = K.square((y_true - tf.cast(y_pred, tf.int32)) * b_)
        t = K.sum(x, axis=-1)
        return K.mean(t)
    return loss_2nd


def l_1st(alpha):
    def loss_1st(y_true, y_pred):
        L = y_true
        Y = y_pred
        batch_size = tf.cast(K.shape(L)[0], tf.float32)
        return alpha * 2 * tf.linalg.trace(tf.matmul(tf.matmul(Y, L, transpose_a=True), Y)) / batch_size
    return loss_1st

def create_model(node_size, hidden_size=[256, 128], l1=1e-5, l2=1e-4):
    """
    :param node_size:
    :param hidden_size DEFAULT [256, 128]:
    :param l1 DEFAULT 1e-5:
    :param l2 DEFAULT 1e-4:
    :return: model, embedding
    """
    A = Input(shape=(node_size,))
    L = Input(shape=(None,))
    fc = A
    for i in range(len(hidden_size)):
        if i == len(hidden_size) - 1:
            fc = Dense(hidden_size[i], activation="relu", kernel_regularizer=l1_l2(l1, l2), name="1st")(fc)
        else:
            fc = Dense(hidden_size[i], activation="relu", kernel_regularizer=l1_l2(l1, l2))(fc)
    Y = fc
    for i in reversed(range(len(hidden_size) - 1)):
        fc = Dense(hidden_size[i], activation="relu", kernel_regularizer=l1_l2(l1, l2))(fc)
    A_ = Dense(node_size, "relu", name="2nd")(fc)
    model = Model(inputs=[A, L], outputs=[A_, Y])
    emb = Model(inputs=A, outputs=Y)
    return model, emb

def distance_matrix(graph):
    """
    :param graph:
    :return: a sparse matrix that is the distance matrix of the graph
    """
    nodecount = graph.number_of_nodes()
    dists = []
    i = 0
    for source in graph.nodes():
        j = 0
        for target in graph.nodes():
            if i == j:
                dists.append(0)
            elif i > j:
                dists.append(dists[j * nodecount + i])
            else:
                try:
                    dists.append(nx.shortest_path_length(graph, source, target))
                except:
                    dists.append(0)
            j += 1
        i += 1
    return scipy.sparse.csr_matrix((dists, (list(range(nodecount)), list(range(nodecount)))))

def unnormalized_laplacian_matrix(A):
    """
    :param A: adjacency matrix of a graph (can be weighted), should be a sparse matrix
    :return L: the unnormalized Laplacian matrix of the graph corresponding to A, which is a sparse matrix
    -------------------------------------------------------------------------------
    Explanation: L = D - W, where W is the adjacency matrix (weight matrix) and D is s.t. $D_{ii} = \sum_j W_{ji}$

    """
    nodecount = A.shape[0]
    D = scipy.sparse.lil_matrix((nodecount, nodecount))
    D.setdiag(A.todense().sum(axis=1))
    return D - A
import numpy as np

import time

import scipy.sparse as sp

from tensorflow.python.keras.callbacks import History

from embedders.utils import preprocess_nxgraph
from embedders.utils import create_model
from embedders.utils import l_2nd, l_1st

### ========== ========== ========= ========== ========== ###
### CLASS SDNE ###
### ========== ========== ========= ========== ========== ###

class SDNE(object):

    def __init__(self, graph, hidden_size=[32, 16], alpha=1e-6, beta=5., nu1=1e-5, nu2=1e-4):
        """
        :param self:
        :param graph:
        :param hidden_size:
        :param alpha:
        :param beta:
        :param nu1:
        :param nu2:
        :return: None
        """
        self.graph = graph
        self.idx2node, self.node2idx = preprocess_nxgraph(self.graph)
        self.node_size = self.graph.number_of_nodes()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2
        self.A, self.L = _create_A_L(self.graph, self.node2idx)
        self.reset_model()
        self.inputs = [self.A, self.L]
        self._embeddings = {}
    
    def reset_model(self, opt="adam"):
        """
        :param self:
        :param opt DEFAULT "adam":
        :return: None
        """
        self.model, self.emb_model = create_model(self.node_size, hidden_size=self.hidden_size, l1=self.nu1, l2=self.nu2)
        self.model.compile(opt, [l_2nd(self.beta), l_1st(self.alpha)], run_eagerly=True)
        self.get_embeddings()

    def train(self, batch_size=1024, epochs=1, initial_epoch=0, verbose=2):
        """
        :param self:
        :param batch_size:
        :param epochs:
        :param initial_epoch:
        :param verbose:
        :return: hist
        """
        if batch_size >= self.node_size:
            if batch_size > self.node_size:
                print("batch_size({0}) > node_size({1}), set batch_size = {1}".format(batch_size, self.node_size))
                batch_size = self.node_size
            return self.model.fit([self.A.todense(), self.L.todense()], [self.A.todense(), self.L.todense()],
                                  batch_size=batch_size, epochs=epochs, initial_epoch=initial_epoch, verbose=verbose, shuffle=False)
        else:
            steps_per_epoch = (self.node_size - 1) // batch_size + 1
            hist = History()
            hist.on_train_begin()
            logs = {}
            for epoch in range(initial_epoch, epochs):
                start_time = time.time()
                losses = np.zeros(3)
                for i in range(steps_per_epoch):
                    index = np.arange(i * batch_size, min((i + 1) * batch_size, self.node_size))
                    A_train = self.A[index, :].todense()
                    L_mat_train = self.L[index][:, index].todense()
                    inp = [A_train, L_mat_train]
                    batch_losses = self.model.train_on_batch(inp, inp)
                    losses += batch_losses
                losses /= steps_per_epoch
                logs["loss"] = losses[0]
                logs["2nd_loss"] = losses[1]
                logs["1st_loss"] = losses[2]
                epoch_time = int(time.time() - start_time)
                hist.on_epoch_end(epoch, logs)
                if verbose > 0:
                    print("Epoch {}/{}: {}s - loss: {:.4f}; 2nd_loss: {:.4f}; 1st_loss: {:.4f}".format(epoch + 1, epochs, epoch_time, losses[0], losses[1], losses[2]))
            return hist

    def evaluate(self):
        """
        :param self:
        :return: evaluation
        """
        return self.model.evaluate(x=self.inputs, y=self.inputs, batch_size=self.node_size)

    def get_embeddings(self):
        """
        :param self:
        :return: embeddings
        """
        self._embeddings = {}
        embeddings = self.emb_model.predict(self.A.todense(), batch_size=self.node_size)
        look_back = self.idx2node
        for i, embedding in enumerate(embeddings):
            self._embeddings[look_back[i]] = embedding
        return self._embeddings

def _create_A_L(graph, node2idx):
    """
    :param graph:
    :param node2idx:
    :return: adjacency matrix, L matrix
    """
    node_size = graph.number_of_nodes()
    A_data, A_row_index, A_col_index = [], [], []
    for edge in graph.edges():
        v1, v2 = edge
        edge_weight = graph[v1][v2].get("weight", 1)
        A_data.append(edge_weight)
        A_row_index.append(node2idx[v1])
        A_col_index.append(node2idx[v2])
    A = sp.csr_matrix((A_data, (A_row_index, A_col_index)), shape=(node_size, node_size))
    A_ = sp.csr_matrix((A_data + A_data, (A_row_index + A_col_index, A_col_index + A_row_index)), shape=(node_size, node_size))
    D = sp.diags(A_.sum(axis=1).flatten().tolist()[0])
    return A, D - A_
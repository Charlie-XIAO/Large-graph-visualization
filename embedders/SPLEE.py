import scipy.sparse.linalg as la

from embedders.utils import RBF_distance_metric, distance_matrix, unnormalized_laplacian_matrix

### ========== ========== ========= ========== ========== ###
### SHORTEST PATH LAPLACIAN EIGENMAP ###
### ========== ========== ========= ========== ========== ###
class SPLEE:

    def __init__(self, graph):
        """
        :param self:
        :param graph:
        :return: None
        """
        self.graph = graph
        self._embeddings = {}
    
    def get_embeddings(self, embed_size=128, iter=100):
        """
        :param self:
        :param embed_size DEFAULT 128:
        :param iter DEFAULT 100:
        :return: embedding
        """
        import time
        t0 = time.time()
        dist = RBF_distance_metric(distance_matrix(self.graph), sigma2=1.0)
        t1 = time.time()
        print(t1-t0)
        laplacian = unnormalized_laplacian_matrix(dist)
        t2 = time.time()
        print(t2-t1)
        node_embedding = la.eigsh(laplacian, k=embed_size, which="SM", maxiter=iter*self.graph.number_of_nodes(), return_eigenvectors=True)[1]
        t3 = time.time()
        print(t3-t2)
        self._embeddings = {}
        i = 0
        for node in self.graph.nodes():
            self._embeddings[node] = node_embedding[i]
            i += 1
        return self._embeddings

# python main.py --embed splee --data hr2 --description 128-rbf
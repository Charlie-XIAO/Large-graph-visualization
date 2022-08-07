import scipy.sparse.linalg as la

from embedders.utils import distance_matrix, unnormalized_laplacian_matrix

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
        dist = distance_matrix(self.graph)
        laplacian = unnormalized_laplacian_matrix(dist)
        node_embedding = la.eigsh(laplacian, k=embed_size, which="SM", maxiter=iter*self.graph.number_of_nodes(), return_eigenvectors=True)[1]
        self._embeddings = {}
        i = 0
        for node in self.graph.nodes():
            self._embeddings[node] = node_embedding[i]
            i += 1
        return self._embeddings
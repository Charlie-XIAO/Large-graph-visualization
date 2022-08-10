import scipy.sparse.linalg as la
import networkx as nx

### ========== ========== ========= ========== ========== ###
### LAPLACIAN EIGENMAP ###
### ========== ========== ========= ========== ========== ###
class LEE:

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
        Laplacian = nx.normalized_laplacian_matrix(self.graph)
        node_embedding = la.eigsh(Laplacian, k=embed_size, which="SM", maxiter=iter*self.graph.number_of_nodes(), return_eigenvectors=True)[1]
        self._embeddings = {}
        i = 0
        for node in self.graph.nodes():
            self._embeddings[node] = node_embedding[i]
            i += 1
        return self._embeddings
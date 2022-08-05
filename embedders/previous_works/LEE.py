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
        self.model = None
        self._embeddings = {}

    def train(self, embed_size=128, iter=100):
        Laplacian = nx.normalized_laplacian_matrix(self.graph)
        model = la.eigsh(Laplacian, k=embed_size, which="SM", maxiter=iter*self.graph.number_of_nodes(), return_eigenvectors=True)
        self.model = model
        return model
    
    def get_embeddings(self):
        """
        :param self:
        :return: embedding
        """
        if self.model is None:
            print("Error: model not trained.")
            return {}
        self._embeddings = {}
        i = 0
        for node in self.graph.nodes():
            self._embeddings[node] = self.model[1][i]
            i += 1
        return self._embeddings
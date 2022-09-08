"""
t-distributed Stochastic Graph Neighbor Embedding (TSGNE)

The KNN of each point is directly selected from the corresponding graph (i.e. use the graph neighbors).
"""

# from sklearn.manifold import TSNE
from time import time
from visualizing_tests.manifold import TSGNE

from visualizing_tests.AbstractVisTest import AbstractVisTest

from tests.utils import construct_knn_from_graph

class TSGNETest(AbstractVisTest):

    def __init__(self, graph, embeddings, has_feature, location, perplexity=30, n_components=2, verbose=1, random_state=0, mode="distance"):
        """
        :param self:
        :param graph: graph of the dataset
        :param embeddings: high-dimensional embeddings from cls.embeddings of any class that implements AbstractEmbedTest
        :param has_feature: cls.has_feature of any class that implements AbstractEmbedTest
        :param location: absolute path to save the plot image in .jpg format, need to create folder in advance if not exist
        :param n_components:
        :param verbose:
        :param random_state:
        :return: None
        """
        super().__init__(embeddings, has_feature, location)
        self.graph = graph
        self.perplexity = perplexity
        self.n_components = n_components
        self.verbose = verbose
        self.random_state = random_state
        self.mode = mode
        self.savePlot()
    
    def getProjection(self):
        t0 = time()
        self.n_neighbors = min(len(self.graph) - 1, int(3.0 * self.perplexity + 1))
        print(f"Using k={self.n_neighbors} for KNN in tsne")

        t1 = time()
        print("[t-sgne] Computing nearest neighbors for the embedding using a given KNN sparse matrix")

        self.knn_matrix = construct_knn_from_graph(self.graph, k=self.n_neighbors, sparse=True)
        
        print(
            "[t-sgne] Computed neighbors for {} samples in {:.3f}s...".format(
                len(self.graph), time() - t1
            )
        )

        model = TSGNE(
            perplexity=self.perplexity,
            n_components=self.n_components, 
            verbose=self.verbose, 
            random_state=self.random_state,
            knn_matrix = self.knn_matrix,
            mode=self.mode,
        )
        self.projections = model.fit_transform(self.X)
        self.duration = time() - t0
"""
t-distributed Graph Stochastic Neighbor Embedding (TGSNE)

The KNN of each point is directly selected from the corresponding graph (i.e. use the graph neighbors).
"""

from sklearn.manifold import TSNE

from visualizing_tests.AbstractVisTest import AbstractVisTest

class TGSNETest(AbstractVisTest):

    def __init__(self, graph, embeddings, has_feature, location, n_components=2, verbose=1, random_state=0):
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
        self.n_components = n_components
        self.verbose = verbose
        self.random_state = random_state
        self.savePlot()
    
    def getProjection(self):
        model = TSNE(n_components=self.n_components, verbose=self.verbose, random_state=self.random_state)
        self.projections = model.fit_transform(self.X)
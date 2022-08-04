import pandas as pd

from embedders.utils import *
from embedders.ShortestPath import ShortestPath

from embedding_tests.AbstractEmbedTest import AbstractEmbedTest

class ShortestPathTest(AbstractEmbedTest):

    def __init__(self, edgeset, featureset=None):
        """
        :param self:
        :param edgeset: absolute path of the node-node edgeset in .txt format
        :param featureset: absolute path of the node-feature featureset in .txt format
        :param walk_length:
        :param num_walks:
        :param workers:
        :param window_size:
        :param iter:
        :return: None
        """
        super().__init__(edgeset, featureset)
        self.embed()
    
    def getEmbeddings(self):
        model = ShortestPath(self.graph)
        embeddings = pd.DataFrame.from_dict(model.get_embeddings())
        self.embeddings = embeddings.T
import pandas as pd

from embedders.utils import *
from embedders.ShortestPath import ShortestPath

from embedding_tests.AbstractEmbedTest import AbstractEmbedTest

class ShortestPathTest(AbstractEmbedTest):

    def __init__(self, edgeset, embed_size, featureset=None):
        """
        :param self:
        :param edgeset: absolute path of the node-node edgeset in .txt format
        :param featureset: absolute path of the node-feature featureset in .txt format
        :return: None
        """
        super().__init__(edgeset, embed_size, featureset)
        self.embed()
    
    def getEmbeddings(self):
        model = ShortestPath(self.graph)
        embeddings = pd.DataFrame.from_dict(model.get_embeddings(embed_size=self.embed_size))
        self.embeddings = embeddings.T
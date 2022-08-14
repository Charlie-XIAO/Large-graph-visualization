import pandas as pd

from embedders.utils import *

from embedding_tests.AbstractEmbedTest import AbstractEmbedTest

class RandomEmbedTest(AbstractEmbedTest):

    def __init__(self, edgeset, embed_size=128, featureset=None, seed=1):
        """
        :param self:
        :param edgeset: absolute path of the node-node edgeset in .txt format
        :param featureset: absolute path of the node-feature featureset in .txt format
        """
        super().__init__(edgeset, embed_size, featureset)
        self.embed()
    
    def getEmbeddings(self):
        self.embeddings = pd.DataFrame(
            np.random.rand(self.graph.number_of_nodes(), self.embed_size),
            columns=[str(num) for num in range(self.embed_size)]
            )
        
        self.embeddings.index = self.embeddings.index.map(str)
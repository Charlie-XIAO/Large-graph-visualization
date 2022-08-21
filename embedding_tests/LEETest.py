from time import time
import pandas as pd

from embedders.utils import *
from embedders.previous_works.LEE import LEE

from embedding_tests.AbstractEmbedTest import AbstractEmbedTest

class LEETest(AbstractEmbedTest):

    def __init__(self, edgeset, embed_size=128, featureset=None, iter=100):
        """
        :param self:
        :param edgeset: absolute path of the node-node edgeset in .txt format
        :param featureset: absolute path of the node-feature featureset in .txt format
        :param iter:
        :return: None
        """
        super().__init__(edgeset, embed_size, featureset)
        self.iter = iter
        self.embed()
    
    def getEmbeddings(self):
        t0 = time()
        model = LEE(self.graph)
        embeddings = pd.DataFrame.from_dict(model.get_embeddings(embed_size=self.embed_size, iter=self.iter))
        self.embeddings = embeddings.T
        self.duration = time() - t0
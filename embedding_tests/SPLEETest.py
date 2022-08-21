from time import time
import pandas as pd

from embedders.utils import *
from embedders.SPLEE import SPLEE

from embedding_tests.AbstractEmbedTest import AbstractEmbedTest

class SPLEETest(AbstractEmbedTest):

    def __init__(self, edgeset, embed_size=128, featureset=None, iter=100, shape="gaussian", epsilon=7, threshold=8):
        """
        :param self:
        :param edgeset: absolute path of the node-node edgeset in .txt format
        :param featureset: absolute path of the node-feature featureset in .txt format
        :param iter:
        :param shape: can be "gaussian", "invquad", "invmultiquad", "bump"
        :param epsilon: the paramter for the kernel function
        :return: None
        """
        super().__init__(edgeset, embed_size, featureset)
        self.iter = iter
        self.shape = shape
        self.epsilon = epsilon
        self.threshold = threshold
        self.embed()
    
    def getEmbeddings(self):
        t0 = time()
        model = SPLEE(self.graph)
        embeddings = pd.DataFrame.from_dict(model.get_embeddings(embed_size=self.embed_size, iter=self.iter, shape=self.shape, epsilon=self.epsilon, threshold=self.threshold))
        self.embeddings = embeddings.T
        self.duration = time() - t0
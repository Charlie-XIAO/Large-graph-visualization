import pandas as pd
from time import time

from embedders.utils import *
from embedders.previous_works.DeepWalk import DeepWalk

from embedding_tests.AbstractEmbedTest import AbstractEmbedTest

class DeepWalkTest(AbstractEmbedTest):

    def __init__(self, edgeset, embed_size=128, featureset=None, walk_length=10, num_walks=80, workers=1, window_size=5, iter=3):
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
        super().__init__(edgeset, embed_size, featureset)
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.window_size = window_size
        self.iter = iter
        self.embed()
    
    def getEmbeddings(self):
        t0 = time()
        model = DeepWalk(self.graph, walk_length=self.walk_length, num_walks=self.num_walks, workers=self.workers)
        model.train(embed_size=self.embed_size, window_size=self.window_size, iter=self.iter)
        embeddings = pd.DataFrame.from_dict(model.get_embeddings())
        self.embeddings = embeddings.T
        self.duration = time() - t0
import pandas as pd

from embedders.utils import *
from embedders.previous_works.Node2Vec import Node2Vec

from embedding_tests.AbstractEmbedTest import AbstractEmbedTest

class Node2VecTest(AbstractEmbedTest):

    def __init__(self, edgeset, embed_size, featureset=None, walk_length=10, num_walks=80, p=0.25, q=4, workers=1, window_size=5, iter=3):
        """
        :param self:
        :param edgeset: absolute path of the node-node edgeset in .txt format
        :param featureset: absolute path of the node-feature featureset in .txt format
        :param walk_length:
        :param num_walks:
        :param p:
        :param q:
        :param workers:
        :param window_size:
        :param iter:
        :return: None
        """
        super().__init__(edgeset, embed_size, featureset)
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers
        self.window_size = window_size
        self.iter = iter
        self.embed()
    
    def getEmbeddings(self):
        model = Node2Vec(self.graph, walk_length=self.walk_length, num_walks=self.num_walks, p=self.p, q=self.q, workers=self.workers)
        model.train(embed_size=self.embed_size, window_size=self.window_size, iter=self.iter)
        embeddings = pd.DataFrame.from_dict(model.get_embeddings())
        self.embeddings = embeddings.T
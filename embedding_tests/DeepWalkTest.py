import pandas as pd

from embedders.utils import *
from embedders.previous_works.DeepWalk import DeepWalk

from embedding_tests.AbstractEmbedTest import AbstractEmbedTest

class DeepWalkTest(AbstractEmbedTest):

    def __init__(self, edgeset, featureset=None, walk_length=10, num_walks=80, workers=1, window_size=5, iter=3):
        super().__init__(edgeset, featureset)
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.window_size = window_size
        self.iter = iter
    
    def getEmbeddings(self):
        model = DeepWalk(self.graph, walk_length=self.walk_length, num_walks=self.num_walks, workers=self.workers)
        model.train(window_size=self.window_size, iter=self.iter)
        embeddings = pd.DataFrame.from_dict(model.get_embeddings())
        self.embeddings = embeddings.T
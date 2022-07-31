import pandas as pd

from embedders.utils import *
from embedders.previous_works.SDNE import SDNE

from embedding_tests.AbstractEmbedTest import AbstractEmbedTest

class SDNETest(AbstractEmbedTest):

    def __init__(self, edgeset, featureset=None, hidden_size=[256, 128], batch_size=3000, epochs=40, verbose=2):
        super().__init__(edgeset, featureset)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
    
    def getEmbeddings(self):
        model = SDNE(self.graph, hidden_size=self.hidden_size)
        model.train(batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose)
        embeddings = pd.DataFrame.from_dict(model.get_embeddings())
        self.embeddings = embeddings.T
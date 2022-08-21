
from time import time

from sklearn.decomposition import PCA

from visualizing_tests.AbstractVisTest import AbstractVisTest

class PCATest(AbstractVisTest):

    def __init__(self, embeddings, has_feature, location, n_components=2, random_state=0):
        """
        :param self:
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
        self.random_state = random_state
        self.savePlot()
    
    def getProjection(self):
        t0 = time()
        model = PCA(n_components=self.n_components, random_state=self.random_state)
        self.projections = model.fit_transform(self.X)
        self.duration = time() - t0
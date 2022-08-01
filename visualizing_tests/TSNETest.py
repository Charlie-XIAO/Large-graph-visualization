from sklearn.manifold import TSNE

from visualizing_tests.AbstractVisTest import AbstractVisTest

class TSNETest(AbstractVisTest):

    def __init__(self, embeddings, has_feature, location, n_components=2, verbose=1, random_state=0):
        super().__init__(embeddings, has_feature, location)
        self.n_components = n_components
        self.verbose = verbose
        self.random_state = random_state
    
    def getProjection(self):
        model = TSNE(n_components=self.n_components, verbose=self.verbose, random_state=self.random_state)
        print("TSNE initialized, processing", end="... ")
        self.projections = model.fit_transform(self.X)
        print("Projections done.")
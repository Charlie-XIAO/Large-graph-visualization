from tests.utils import *

from embedding_tests.ShortestPathTest import ShortestPathTest
from visualizing_tests.PCATest import PCATest

def ShortestPath_PCA_test(config):

    dim, edgeset, featureset, location = setup(config)

    shortestpath = ShortestPathTest(edgeset, embed_size=dim, featureset=featureset)
    pca = PCATest(shortestpath.embeddings, shortestpath.has_feature, location, n_components=2, random_state=0)
    show_evaluation_results(config, shortestpath, pca, k=10)
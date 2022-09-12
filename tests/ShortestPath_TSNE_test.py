from tests.utils import *

from embedding_tests.ShortestPathTest import ShortestPathTest
from visualizing_tests.TSNETest import TSNETest

def ShortestPath_TSNE_test(config):

    dim, edgeset, featureset, location = setup(config)

    shortestpath = ShortestPathTest(edgeset, embed_size=dim, featureset=featureset)
    tsne = TSNETest(shortestpath.embeddings, shortestpath.has_feature, location, n_components=2, verbose=2, random_state=0)
    show_evaluation_results(config, shortestpath, tsne, k=10)
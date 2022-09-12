from tests.utils import *

from embedding_tests.ShortestPathTest import ShortestPathTest
from visualizing_tests.TSGNETest import TSGNETest

def ShortestPath_TSGNE_test(config):

    dim, edgeset, featureset, location = setup(config)

    shortestpath = ShortestPathTest(edgeset, embed_size=dim, featureset=featureset)
    TSGNE = TSGNETest(shortestpath.graph, shortestpath.embeddings, shortestpath.has_feature, location, n_components=2, verbose=2, random_state=0)
    show_evaluation_results(config, shortestpath, TSGNE, k=10)
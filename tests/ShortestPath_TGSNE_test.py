from re import M
from tests.utils import *

from embedding_tests.ShortestPathTest import ShortestPathTest
from visualizing_tests.TGSNETest import TGSNETest

def ShortestPath_TGSNE_test(config):

    dim, edgeset, featureset, location = setup(config)

    shortestpath = ShortestPathTest(edgeset, embed_size=dim, featureset=featureset)
    tgsne = TGSNETest(
        shortestpath.graph, 
        shortestpath.embeddings, 
        shortestpath.has_feature, 
        location, 
        n_components=2, 
        verbose=1, 
        random_state=0)
    show_evaluation_results(config, shortestpath, tgsne, k=10)
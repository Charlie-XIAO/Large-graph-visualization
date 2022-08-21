from tests.utils import *

from embedding_tests.SPLEETest import SPLEETest
from visualizing_tests.TSGNETest import TSGNETest

def SPLEE_TSGNE_test(config):

    dim, edgeset, featureset, location = setup(config)

    splee = SPLEETest(edgeset, embed_size=dim, featureset=featureset, iter=100, shape="gaussian", epsilon=7.0, threshold=None)
    TSGNE = TSGNETest(splee.graph, splee.embeddings, splee.has_feature, location, n_components=2, verbose=1, random_state=0)
    show_evaluation_results(config, splee, TSGNE, k=10)
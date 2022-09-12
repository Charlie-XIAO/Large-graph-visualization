from tests.utils import *

from embedding_tests.SDNETest import SDNETest
from visualizing_tests.TSGNETest import TSGNETest

def SDNE_TSGNE_test(config):

    dim, edgeset, featureset, location = setup(config)

    sdne = SDNETest(edgeset, featureset=featureset, hidden_size=[dim * 2, dim], batch_size=3000, epochs=40, verbose=2)
    TSGNE = TSGNETest(sdne.graph, sdne.embeddings, sdne.has_feature, location, n_components=2, verbose=2, random_state=0)
    show_evaluation_results(config, sdne, TSGNE, k=10)
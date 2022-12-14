from tests.utils import *

from embedding_tests.LEETest import LEETest
from visualizing_tests.TSGNETest import TSGNETest

def LEE_TSGNE_test(config):

    dim, edgeset, featureset, location = setup(config)

    lee = LEETest(edgeset, embed_size=dim, featureset=featureset, iter=100)
    TSGNE = TSGNETest(lee.graph, lee.embeddings, lee.has_feature, location, n_components=2, verbose=2, random_state=0)
    show_evaluation_results(config, lee, TSGNE, k=10)
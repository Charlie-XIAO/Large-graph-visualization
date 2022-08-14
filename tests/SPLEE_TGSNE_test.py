from tests.utils import *

from embedding_tests.SPLEETest import SPLEETest
from visualizing_tests.TGSNETest import TGSNETest

def SPLEE_TGSNE_test(config):

    dim, edgeset, featureset, location = setup(config)

    splee = SPLEETest(edgeset, embed_size=dim, featureset=featureset, iter=100, shape="gaussian", epsilon=7.0, threshold=None)
    tgsne = TGSNETest(splee.graph, splee.embeddings, splee.has_feature, location, n_components=2, verbose=1, random_state=0)
    show_evaluation_results(config, splee, tgsne, k=10)
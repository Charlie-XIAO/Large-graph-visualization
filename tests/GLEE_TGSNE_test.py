from tests.utils import *

from embedding_tests.GLEETest import GLEETest
from visualizing_tests.TGSNETest import TGSNETest

def GLEE_TGSNE_test(config):

    dim, edgeset, featureset, location = setup(config)

    glee = GLEETest(edgeset, embed_size=dim, featureset=featureset)
    tgsne = TGSNETest(glee.graph, glee.embeddings, glee.has_feature, location, n_components=2, verbose=1, random_state=0)
    show_evaluation_results(config, glee, tgsne, k=10)
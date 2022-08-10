from tests.utils import *

from embedding_tests.GLEETest import GLEETest
from visualizing_tests.TSNETest import TSNETest

def GLEE_TSNE_test(config):

    dim, edgeset, featureset, location = setup(config)

    glee = GLEETest(edgeset, embed_size=dim, featureset=featureset)
    tsne = TSNETest(glee.embeddings, glee.has_feature, location, n_components=2, verbose=1, random_state=0)
    show_evaluation_results(config, glee, tsne, k=10)
from tests.utils import *

from embedding_tests.SDNETest import SDNETest
from visualizing_tests.TSNETest import TSNETest

def SDNE_TSNE_test(config):

    dim, edgeset, featureset, location = setup(config)

    sdne = SDNETest(edgeset, featureset=featureset, hidden_size=[dim * 2, dim], batch_size=3000, epochs=40, verbose=2)
    tsne = TSNETest(sdne.embeddings, sdne.has_feature, location, n_components=2, verbose=2, random_state=0)
    show_evaluation_results(config, sdne, tsne, k=10)
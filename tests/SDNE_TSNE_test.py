from tests.utils import *

from embedding_tests.SDNETest import SDNETest
from visualizing_tests.TSNETest import TSNETest

def SDNE_TSNE_test(config):

    edgeset, featureset, location = setup(config)

    sdne = SDNETest(edgeset, featureset=featureset, hidden_size=[256, 128], batch_size=3000, epochs=40, verbose=2)
    tsne = TSNETest(sdne.embeddings, sdne.has_feature, location, n_components=2, verbose=1, random_state=0)
    show_evaluation_results(sdne, tsne)
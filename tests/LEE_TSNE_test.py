from tests.utils import *

from embedding_tests.LEETest import LEETest
from visualizing_tests.TSNETest import TSNETest

def LEE_TSNE_test(config):

    edgeset, featureset, location = setup(config)

    lee = LEETest(edgeset, featureset=featureset, iter=100)
    tsne = TSNETest(lee.embeddings, lee.has_feature, location, n_components=2, verbose=1, random_state=0)
    show_evaluation_results(config, lee, tsne, k=10)
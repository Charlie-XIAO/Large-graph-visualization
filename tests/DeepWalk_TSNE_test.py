from tests.utils import *

from embedding_tests.DeepWalkTest import DeepWalkTest
from visualizing_tests.TSNETest import TSNETest

def DeepWalk_TSNE_test(config):

    edgeset, featureset, location = setup(config)

    deepwalk = DeepWalkTest(edgeset, featureset=featureset, walk_length=10, num_walks=80, workers=1, window_size=5, iter=3)
    tsne = TSNETest(deepwalk.embeddings, deepwalk.has_feature, location, n_components=2, verbose=1, random_state=0)
    show_evaluation_results(config, deepwalk, tsne, k=10)
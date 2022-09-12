from tests.utils import *

from embedding_tests.DeepWalkTest import DeepWalkTest
from visualizing_tests.TSGNETest import TSGNETest

def DeepWalk_TSGNE_test(config):

    dim, edgeset, featureset, location = setup(config)

    deepwalk = DeepWalkTest(edgeset, embed_size=dim, featureset=featureset, walk_length=10, num_walks=80, workers=1, window_size=5, iter=3)
    tsgne = TSGNETest(
        deepwalk.graph, 
        deepwalk.embeddings, 
        deepwalk.has_feature, 
        location,
        n_components=2, 
        verbose=2, 
        random_state=0,
        mode=config["knn_mode"]
        )
    show_evaluation_results(config, deepwalk, tsgne, k=10)
    
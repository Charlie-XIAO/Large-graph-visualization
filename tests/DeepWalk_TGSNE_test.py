from tests.utils import *

from embedding_tests.DeepWalkTest import DeepWalkTest
from visualizing_tests.TGSNETest import TGSNETest

def DeepWalk_TGSNE_test(config):

    dim, edgeset, featureset, location = setup(config)

    deepwalk = DeepWalkTest(edgeset, embed_size=dim, featureset=featureset, walk_length=10, num_walks=80, workers=1, window_size=5, iter=3)
    tgsne = TGSNETest(deepwalk.graph, deepwalk.embeddings, deepwalk.has_feature, location,n_components=2, verbose=1, random_state=0)
    show_evaluation_results(config, deepwalk, tgsne, k=10)
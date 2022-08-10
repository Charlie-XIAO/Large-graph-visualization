from tests.utils import *

from embedding_tests.DeepWalkTest import DeepWalkTest
from visualizing_tests.PCATest import PCATest

def DeepWalk_PCA_test(config):

    dim, edgeset, featureset, location = setup(config)

    deepwalk = DeepWalkTest(edgeset, embed_size=dim, featureset=featureset, walk_length=10, num_walks=80, workers=1, window_size=5, iter=3)
    pca = PCATest(deepwalk.embeddings, deepwalk.has_feature, location, n_components=2, random_state=0)
    show_evaluation_results(config, deepwalk, pca, k=10)
from tests.utils import *

from embedding_tests.GLEETest import GLEETest
from visualizing_tests.PCATest import PCATest

def GLEE_PCA_test(config):

    dim, edgeset, featureset, location = setup(config)

    glee = GLEETest(edgeset, embed_size=dim, featureset=featureset)
    pca = PCATest(glee.embeddings, glee.has_feature, location, n_components=2, random_state=0)
    show_evaluation_results(config, glee, pca, k=10)
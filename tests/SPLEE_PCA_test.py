from tests.utils import *

from embedding_tests.SPLEETest import SPLEETest
from visualizing_tests.PCATest import PCATest

def SPLEE_PCA_test(config):

    dim, edgeset, featureset, location = setup(config)

    splee = SPLEETest(edgeset, embed_size=dim, featureset=featureset, iter=100)
    pca = PCATest(splee.embeddings, splee.has_feature, location, n_components=2, random_state=0)
    show_evaluation_results(config, splee, pca, k=10)
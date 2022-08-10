from tests.utils import *

from embedding_tests.LEETest import LEETest
from visualizing_tests.PCATest import PCATest

def LEE_PCA_test(config):

    dim, edgeset, featureset, location = setup(config)

    lee = LEETest(edgeset, embed_size=dim, featureset=featureset, iter=100)
    pca = PCATest(lee.embeddings, lee.has_feature, location, n_components=2, random_state=0)
    show_evaluation_results(config, lee, pca, k=10)
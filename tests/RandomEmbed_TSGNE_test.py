from tests.utils import *

from embedding_tests.RandomEmbedTest import RandomEmbedTest
from visualizing_tests.TSGNETest import TSGNETest

def RandomEmbed_TSGNE_test(config):

    dim, edgeset, featureset, location = setup(config)

    randomembed = RandomEmbedTest(edgeset, embed_size=dim, featureset=featureset)
    
    TSGNE = TSGNETest(
        randomembed.graph, 
        randomembed.embeddings, 
        randomembed.has_feature, 
        location,
        n_components=2, 
        verbose=1, 
        random_state=0,
        mode=config["knn_mode"],)
    show_evaluation_results(config, randomembed, TSGNE, k=10)
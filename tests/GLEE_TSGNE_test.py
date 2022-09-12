from tests.utils import *

from embedding_tests.GLEETest import GLEETest
from visualizing_tests.TSGNETest import TSGNETest

def GLEE_TSGNE_test(config):

    dim, edgeset, featureset, location = setup(config)

    glee = GLEETest(edgeset, embed_size=dim, featureset=featureset)
    TSGNE = TSGNETest(
        glee.graph, 
        glee.embeddings, 
        glee.has_feature, 
        location, 
        n_components=2, 
        verbose=2, 
        random_state=0,
        mode=config["knn_mode"]
    )
        
    show_evaluation_results(config, glee, TSGNE, k=10)
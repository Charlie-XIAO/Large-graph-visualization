from tests.utils import *

from embedding_tests.Node2VecTest import Node2VecTest
from visualizing_tests.TSGNETest import TSGNETest

def Node2Vec_TSGNE_test(config):

    dim, edgeset, featureset, location = setup(config)
    
    node2vec = Node2VecTest(edgeset, embed_size=dim, featureset=featureset, walk_length=10, num_walks=80, p=0.25, q=4, workers=1, window_size=5, iter=3)
    TSGNE = TSGNETest(node2vec.graph, node2vec.embeddings, node2vec.has_feature, location, n_components=2, verbose=2, random_state=0)
    show_evaluation_results(config, node2vec, TSGNE, k=10)
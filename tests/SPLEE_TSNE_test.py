from tests.utils import *

from embedding_tests.SPLEETest import SPLEETest
from visualizing_tests.TSNETest import TSNETest

def SPLEE_TSNE_test(config):

    dim, edgeset, featureset, location = setup(config)

    splee = SPLEETest(edgeset, embed_size=dim, featureset=featureset, iter=100, shape="gaussian", epsilon=6.0)
    tsne = TSNETest(splee.embeddings, splee.has_feature, location, n_components=2, verbose=1, random_state=0)
    show_evaluation_results(config, splee, tsne, k=10)

"""
python main.py --embed shortestpath --data lfr_3000_easy
python main.py --embed splee --data lastfm --description 128-gaussian6.0
python main.py --embed shortestpath --data random

"""
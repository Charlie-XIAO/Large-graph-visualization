from tests.utils import *

from embedding_tests.SDNETest import SDNETest
from visualizing_tests.TGSNETest import TGSNETest

def SDNE_TGSNE_test(config):

    dim, edgeset, featureset, location = setup(config)

    sdne = SDNETest(edgeset, featureset=featureset, hidden_size=[dim * 2, dim], batch_size=3000, epochs=40, verbose=2)
    tgsne = TGSNETest(sdne.graph, sdne.embeddings, sdne.has_feature, location, n_components=2, verbose=1, random_state=0)
    show_evaluation_results(config, sdne, tgsne, k=10)
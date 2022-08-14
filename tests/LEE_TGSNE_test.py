from tests.utils import *

from embedding_tests.LEETest import LEETest
from visualizing_tests.TGSNETest import TGSNETest

def LEE_TGSNE_test(config):

    dim, edgeset, featureset, location = setup(config)

    lee = LEETest(edgeset, embed_size=dim, featureset=featureset, iter=100)
    tgsne = TGSNETest(lee.graph, lee.embeddings, lee.has_feature, location, n_components=2, verbose=1, random_state=0)
    show_evaluation_results(config, lee, tgsne, k=10)
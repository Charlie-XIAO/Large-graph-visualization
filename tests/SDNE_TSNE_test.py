from tests.utils import *

from embedding_tests.SDNETest import SDNETest
from visualizing_tests.TSNETest import TSNETest

def SDNE_TSNE_test():

    i = get_index()

    if i == 1:
        print_block("Test 1: wiki edgeset")
        edgeset = "./datasets/wiki/wiki_edgelist.txt"
        featureset = "./datasets/wiki/wiki_labels.txt"
        location = "./images/wiki/wiki_SDNE_TSNE_1.jpg"
    
    elif i == 2:
        print_block("Test 2: hr2 edgeset")
        edgeset = "./datasets/hr2/hr2_edgelist.txt"
        featureset = "./datasets/hr2/hr2_labels.txt"
        location = "./images/hr2/hr2_SDNE_TSNE_1.jpg"

    #elif i == 3:
    
    else:
        print("Test of index {} currently unavailable.".format(i))
        return

    sdne = SDNETest(edgeset, featureset=featureset, hidden_size=[256, 128], batch_size=3000, epochs=40, verbose=2)
    sdne.getEmbeddings()
    sdne.addFeature()  # optional
    tsne = TSNETest(sdne.embeddings, sdne.has_feature, location, n_components=2, verbose=1, random_state=0)
    tsne.savePlot(edgeset)
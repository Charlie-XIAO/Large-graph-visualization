from tests.utils import *

from embedding_tests.DeepWalkTest import DeepWalkTest
from visualizing_tests.TSNETest import TSNETest

def DeepWalk_TSNE_test():

    i = get_index()

    if i == 1:
        print_block("Test 1: wiki edgeset")
        edgeset = "./datasets/wiki/wiki_edgelist.txt"
        featureset = "./datasets/wiki/wiki_labels.txt"
        location = "./images/wiki/wiki_DeepWalk_TSNE_1.jpg"
    
    elif i == 2:
        print_block("Test 2: hr2 edgeset")
        edgeset = "./datasets/hr2/hr2_edgelist.txt"
        featureset = "./datasets/hr2/hr2_labels.txt"
        location = "./images/hr2/hr2_DeepWalk_TSNE_1.jpg"

    elif i == 3:
        print_block("Test 3: lock edgeset")
        edgeset = "./datasets/lock/lock_edgelist.txt"
        featureset = "./datasets/lock/hr2_labels.txt"
        location = "./images/lock/lock_DeepWalk_TSNE_1.jpg"
        deepwalk = DeepWalkTest(edgeset, featureset=featureset, walk_length=10, num_walks=80, workers=1, window_size=5, iter=3)
        deepwalk.getEmbeddings()
        deepwalk.addFeature()  # optional
        tsne = TSNETest(deepwalk.embeddings, deepwalk.has_feature, location, n_components=2, verbose=1, random_state=0)
        tsne.savePlot(edgeset)
    
    else:
        print("Test of index {} currently unavailable.".format(i))
        return

    deepwalk = DeepWalkTest(edgeset, featureset=featureset, walk_length=10, num_walks=80, workers=1, window_size=5, iter=3)
    deepwalk.getEmbeddings()
    deepwalk.addFeature()  # optional
    tsne = TSNETest(deepwalk.embeddings, deepwalk.has_feature, location, n_components=2, verbose=1, random_state=0)
    tsne.savePlot(edgeset)
from tests.utils import *

from embedding_tests.DeepWalkTest import DeepWalkTest
from visualizing_tests.TSNETest import TSNETest

if __name__ == "__main__":

    i = get_index()

    if i == 1:
        print_block("Test 1: wiki edgeset")
        deepwalk = DeepWalkTest("./datasets/wiki/wiki_edgelist.txt", featureset="./datasets/wiki/wiki_labels.txt",
            walk_length=10, num_walks=80, workers=1, window_size=5, iter=3)
        deepwalk.addFeature()  # optional
        deepwalk.getEmbeddings()
        tsne = TSNETest(deepwalk.embeddings, deepwalk.has_feature, "./images/wiki/1.jpg",
            n_components=2, verbose=1, random_state=0)
        tsne.savePlot()

    #elif i == 2:
    
    else:
        print("Test of this index currently unavailable.")
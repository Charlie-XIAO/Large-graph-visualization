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
        featureset = "./datasets/lock/lock_labels.txt"
        location = "./images/lock/lock_DeepWalk_TSNE_1.jpg"
    
    else:
        print("Test of index {} currently unavailable.".format(i))
        return

    deepwalk = DeepWalkTest(edgeset, featureset=featureset, walk_length=10, num_walks=80, workers=1, window_size=5, iter=3)
    deepwalk.embed(k=20)
    tsne = TSNETest(deepwalk.embeddings, deepwalk.has_feature, location, n_components=2, verbose=1, random_state=0)
    tsne.savePlot()

    graph = deepwalk.readGraph()
    highDimEmbed = deepwalk.embeddings
    lowDimEmbed = tsne.projections
    randomHighDimEmbed = randomEmbeddings(deepwalk.embeddings)
    randomLowDimEmbed = randomEmbeddings(tsne.projections)

    k = 10
    # compared with d(graph, random_embedding)
    print(f"high dim embedding: {compare_KNN(graph, highDimEmbed, k):.2f}")
    print(f"random high dim embedding: {compare_KNN(graph, randomHighDimEmbed, k):.2f}")
    print(f"low dim embedding: {compare_KNN(graph, lowDimEmbed, k):.2f}")
    print(f"random low dim embedding: {compare_KNN(graph, randomLowDimEmbed, k):.2f}")
    high_v_low = np.average(compare_KNN_matrix(construct_knn_from_embeddings(highDimEmbed), construct_knn_from_embeddings(lowDimEmbed)))
    print(f"high dim vs low dim: {high_v_low:.2f}")
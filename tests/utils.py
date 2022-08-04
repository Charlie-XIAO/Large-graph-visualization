import numpy as np
import networkx as nx
import pandas as pd
import scipy
from sklearn.neighbors import NearestNeighbors
import time


def randomEmbeddings(embeddings, distribution="uniform"):
    """
    Generate a random embedding of the same size as the input embedding,
    following Uniform distribution $Unif(min(embeddings), max(embeddings))$
    """
    if isinstance(embeddings, pd.DataFrame):
        embeddings = embeddings.to_numpy()[:,:-1]
    if distribution == "uniform":       # Unif(min(embeddings), max(embeddings))
        min = embeddings.min()
        max = embeddings.max()
        length = embeddings.shape[0]*embeddings.shape[1]
        return np.random.rand(length).reshape(embeddings.shape) * (int(max) - int(min)) + min
    elif distribution == "normal":      # Norm(mean(embeddings), std(embeddings))
        return np.random.normal(embeddings.mean(), embeddings.std(), embeddings.shape)
    else:
        raise ValueError(f"'{distribution}' distribution not implemented.")


def construct_knn_from_embeddings(embeddings, k):
    # convert embeddings to adjacency matrix of KNN Graph
    if(isinstance(embeddings, pd.DataFrame)):
        embeddings = np.array(embeddings.sort_index())      #   sort embeddings based on row index, so that node i is at row i
    neigh = NearestNeighbors(n_neighbors=k)     # note that KNN includes the point itself
    neigh.fit(embeddings)
    return neigh.kneighbors_graph(embeddings)

# TODO: computation bottleneck
def construct_knn_from_graph(graph, k, sampling_method="knn"):
    knn_of_graph = scipy.sparse.lil_matrix((len(graph), len(graph)), dtype=np.intc)  # initialize the knn matrix
    
    
    if sampling_method == "knn":        # based on knn
        for v in range(len(graph)):
            tree = nx.bfs_tree(nx.convert_node_labels_to_integers(graph), v)
            knn_indices = list(tree)[0: k]
            knn_of_graph[v, knn_indices] = 1
    elif sampling_method == "random walk":      # based on random walk
        # random walk over each node for k steps
        # for v in range(len(graph)):
        #     pass
        pass


    return scipy.sparse.csr_matrix(knn_of_graph)


def compare_KNN_matrix(A, B):
    """
    :param A, B: two 0-1 matrices representing two KNN graphs, where the ith row (equiv. column) represents the k neighbors of ith node

    Examples
    --------
        # usage of tests.utils.compare_KNN_matrix
        g1 = nx.from_edgelist([
        (0, 1), (0, 2), (0, 4), (2, 3), (2, 5), (2, 6), (3, 8), (7, 8)
        ])

        g2 = nx.from_edgelist([
            (0, 2), (0, 4), (2, 3), (2, 5), (2, 6), (3, 8), (7, 8), (1, 8)
        ])

        knn_g1 = construct_knn_from_graph(g1, 2)
        knn_g2 = construct_knn_from_graph(g2, 2)
        print(knn_g1)
        print(compare_KNN_matrix(knn_g1, knn_g2))
    
    """
    intersect_sizes = np.ravel(A.multiply(B).sum(axis=0))
    union_sizes = np.ravel(((A+B).astype('bool').sum(axis=0)))
    return np.array([intersect_sizes[v] / union_sizes[v] for v in range(len(intersect_sizes)) if union_sizes[v] != 0])


def compare_KNN(graph, embeddings, k, timing=False):
    """
    :param graph: the graph-structured data, of type nx.Graph
    :param embeddings: arbitrary embedding of the graph (in $R_n$ or $R_2$), of type pd.DataFrame 
    :param k: the size of neighborhood, the K in KNN
    :return: knn_accuracy, a number between 0 and 1 that measures how the KNN in graph_data and embeddings differ 
        Define distance between two sets to be $d(X, Y) = |X \cap Y| / (|X| \cup |Y|)$
        
            for node v in V:
                KNN_accuracy += d(KNN_graph, KNN_embed)
            KNN_accuracy /= |V|

        (e.g. 1 means two KNNs are exactly the same, 0 means exactly different)

    """
    
    t0 = time.time()
    knn_of_graph = construct_knn_from_graph(graph, k)
    t1 = time.time()
    if timing:
        print(f"Time to construct KNN of graph: {t1-t0}")

    t2 = time.time()
    knn_of_embed = construct_knn_from_embeddings(embeddings, k)
    t3 = time.time()
    if timing:
        print(f"Time to construct KNN of embeddings: {t3-t2}")

    t4 = time.time()
    knn_accuracies = compare_KNN_matrix(knn_of_graph, knn_of_embed)
    knn_accuracy = np.average(knn_accuracies)
    t5 = time.time()
    if timing:
        print(f"Time to compare KNNs: {t5-t4}")
    

    return knn_accuracy


### ========== ========== ========== ========== ========== ###
###                     FORMAT ISSUES                      ###
### ========== ========== ========== ========== ========== ###
def print_block(title):
    print()
    if (len(title) > 52):
        length = len(title)
        print("### " + "=" * length + " ###")
        print("### " + " " * length + " ###")
        print("### " + title + " ###")
        print("### " + " " * length + " ###")
        print("### " + "=" * length + " ###")
    else:
        print("### " + "========== " * 5 + "###")
        print("###" + " " * 56 + "###")
        left = (56 - len(title)) // 2
        right = 56 - len(title) - left
        print("###" + " " * left + title + " " * right + "###")
        print("###" + " " * 56 + "###")
        print("### " + "========== " * 5 + "###")
    print()


def get_index():
    print("### 1: wiki")
    print("### 2: hr2")
    print("### 3: lock")
    while True:
        try:
            i = int(input("Enter test index: "))
            break
        except:
            print("Invalid index.")
    return i


def show_evaluation_results(embed_obj, vis_obj, k=10, timing=False):
    graph = embed_obj.graph
    highDimEmbed = embed_obj.embeddings
    lowDimEmbed = vis_obj.projections
    randomHighDimEmbed = randomEmbeddings(embed_obj.embeddings)
    randomLowDimEmbed = randomEmbeddings(vis_obj.projections)

    keepGoing = True
    high, high_base, low, low_base, high_v_low = -1, -1, -1, -1, -1
    while keepGoing:
        print()
        print("### 1: KNN embedding accuracy")
        print("### 2: KNN visualizing accuracy")
        print("### 3: KNN dimension reduction accuracy")
        print("### A/a: select all types")
        print("### Q/q: quit evaluation")
        check = input("Select evaluation benchmark: ")
        print()
        
        # compared with d(graph, random_embedding)
        if check == "1":
            if high == -1:
                high = compare_KNN(graph, highDimEmbed, k)
            if high_base == -1:
                high_base = compare_KNN(graph, randomHighDimEmbed, k)
            print("KNN embedding accuracy: {:.2f}, with baseline: {:.2f}".format(high, high_base))
        elif check == "2":
            if low == -1:
                low = compare_KNN(graph, lowDimEmbed, k)
            if low_base == -1:
                low_base = compare_KNN(graph, randomLowDimEmbed, k)
            print("KNN visualizing accuracy: {:.2f}, with baseline: {:.2f}".format(low, low_base))
        elif check == "3":
            if high_v_low == -1:
                high_v_low = np.average(compare_KNN_matrix(construct_knn_from_embeddings(highDimEmbed, k), construct_knn_from_embeddings(lowDimEmbed, k)))
            print("KNN dimension reduction accuracy: {:.2f}".format(high_v_low))
        elif check.upper() == "A":
            if high == -1:
                high = compare_KNN(graph, highDimEmbed, k, timing=timing)
            if high_base == -1:
                high_base = compare_KNN(graph, randomHighDimEmbed, k, timing=timing)
            if low == -1:
                low = compare_KNN(graph, lowDimEmbed, k, timing=timing)
            if low_base == -1:
                low_base = compare_KNN(graph, randomLowDimEmbed, k, timing=timing)
            if high_v_low == -1:
                t0 = time.time()
                high_v_low = np.average(compare_KNN_matrix(construct_knn_from_embeddings(highDimEmbed, k), construct_knn_from_embeddings(lowDimEmbed, k)))
                t1 = time.time()
                if timing:
                    print(f"Time to compare high-low KNNs: {t1-t0}")
            print("KNN embedding accuracy: {:.2f}, with baseline: {:.2f}".format(high, high_base))
            print("KNN visualizing accuracy: {:.2f}, with baseline: {:.2f}".format(low, low_base))
            print("KNN dimension reduction accuracy: {:.2f}".format(high_v_low))
        elif check.upper() == "Q":
            keepGoing = False
        else:
            print("Invalid evaluation type.")
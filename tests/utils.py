import numpy as np
import networkx as nx
import pandas as pd
#from scipy.sparse import *
from sklearn.neighbors import NearestNeighbors
import time


### ========== ========== ========== ========== ========== ###
###                      KNN ACCURACY                      ###
### ========== ========== ========== ========== ========== ###
def randomEmbeddings(embeddings, distribution="uniform"):
    """
    Generate a random embedding of the same size as the input embedding,
    following Uniform distribution $Unif(min(embeddings), max(embeddings))$

    :param embeddings: the input embedding, of type numpy.ndarray or pandas.DataFrame
    :param distribution: the distribution to use to generate the random embedding, of type str
    return: the random embedding, of type numpy.ndarray
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
    """
    Construct a KNN graph from a embedding.

    :param embeddings: the embedding of the graph, of type numpy.ndarray or pandas.DataFrame
    :param k: the size of neighborhood, the K in KNN
    return: the KNN matrix of the embeddings, of type numpy.ndaray
    """
    # convert embeddings to adjacency matrix of KNN Graph
    if(isinstance(embeddings, pd.DataFrame)):
        embeddings = np.array(embeddings.sort_index())      #   sort embeddings based on row index, so that node i is at row i
    neigh = NearestNeighbors(n_neighbors=k)     # note that KNN includes the point itself
    neigh.fit(embeddings)
    
    # return neigh.kneighbors_graph(embeddings)
    return neigh.kneighbors_graph(embeddings).toarray().astype(np.intc)


def construct_knn_from_graph(graph, k, neighbor_selection_method="knn", sparse=False):
    """
    Construct a KNN graph from a graph.

    :param graph: the graph-structured data, of type nx.Graph
    :param k: the size of neighborhood, the K in KNN
    :param neighbor_selection_method: the method to use to select the neighbors of a node.
    return: the KNN matrix of the graph
    """
    knn_of_graph = np.eye(len(graph), len(graph), dtype=np.bool)  # initialize the knn matrix
    # if sparse:
    #     knn_of_graph = scipy.sparse.lil_matrix(knn_of_graph)

    # For each node, perform BFS for k steps
    if neighbor_selection_method == "knn":        # based on knn
        for v in range(len(graph)):
            visited = [v]
            cur_at_visited_index = 0
            queue = [v]
            count = 1

            while knn_of_graph[v,:].sum() < k:
                if len(queue) == 0:
                    cur_at_visited_index += 1
                    queue = list(graph[cur_at_visited_index])
                cur_probe = queue.pop()
                visited.append(cur_probe)
                if len(graph[cur_probe]) + count < k:
                    knn_of_graph[v, graph[cur_probe]] = 1
                else:
                    knn_of_graph[v, list(graph[cur_probe])[0: k-count]] = 1

    elif neighbor_selection_method == "random walk":      # based on random walk
        # random walk over each node for k steps
        # for v in range(len(graph)):
        #     pass
        pass

    # if sparse:
    #     return scipy.sparse.csr_matrix(knn_of_graph)
    return knn_of_graph


def compare_KNN_matrix(A: np.ndarray, B: np.ndarray):
    """
    Compare two KNN matrices.

    :param A: the first KNN matrix
    :param B: the second KNN matrix
    return: 

    Explanation:
    KNN matrix is the adjacency matrix of a KNN graph, where a node is connected to its k nearest neighbors.

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
    intersect_sizes = np.ravel((A*B).sum(axis=0))
    union_sizes = np.ravel(((A+B).astype("bool").sum(axis=0)))
    return np.array([intersect_sizes[v] / union_sizes[v] for v in range(len(intersect_sizes)) if union_sizes[v] != 0])


def compare_KNN(graph, embeddings, k, timing=False):
    """
    :param graph: the graph-structured data, of type nx.Graph
    :param embeddings: arbitrary embedding of the graph (in $R_n$ or $R_2$), of type pd.DataFrame 
    :param k: the size of neighborhood, the K in KNN
    :return: an array of knn_accuracy for each node, of type numpy.ndarray
    
    Explanation
    -----------
    knn_accuracy:
        A number between 0 and 1 that measures how the KNN in graph_data and embeddings differ. It is computed as follow:
            Define distance between two sets to be $d(X, Y) = |X \cap Y| / (|X| \cup |Y|)$
        
            for node v in V:
                KNN_accuracy += d(KNN_graph, KNN_embed)
            KNN_accuracy /= |V|

        (e.g. 1 means two KNNs are exactly the same, 0 means exactly different)

    """
    
    t0 = time.time()
    knn_of_graph = construct_knn_from_graph(nx.convert_node_labels_to_integers(graph), k)
    t1 = time.time()
    if timing:
        print(f"Time to construct KNN of graph: {t1-t0:.2f}")

    t2 = time.time()
    knn_of_embed = construct_knn_from_embeddings(embeddings, k)
    t3 = time.time()
    if timing:
        print(f"Time to construct KNN of embeddings: {t3-t2:.2f}")

    t4 = time.time()
    knn_accuracies = compare_KNN_matrix(knn_of_graph, knn_of_embed)
    knn_accuracy = np.average(knn_accuracies)
    t5 = time.time()
    if timing:
        print(f"Time to compare KNNs: {t5-t4:.2f}")
    
    return knn_accuracy


### ========== ========== ========== ========== ========== ###
###                    CLUSTER QUALITY                     ###
### ========== ========== ========== ========== ========== ###
def density_check(projections, k=10, threshold=0.5):

    x_min = projections[0][0]
    for i in range(projections.shape[0]):
        if projections[i][0] < x_min:
            x_min = projections[i][0]
    y_min = projections[0][1]
    for i in range(projections.shape[0]):
        if projections[i][1] < y_min:
            y_min = projections[i][1]
    x_max = projections[0][0]
    for i in range(projections.shape[0]):
        if projections[i][0] > x_max:
            x_max = projections[i][0]

    y_max = projections[0][1]
    for i in range(projections.shape[0]):
        if projections[i][1] > y_max:
            y_max = projections[i][1]

    x_unit = (x_max - x_min) / k
    y_unit = (y_max - y_min) / k
    grid_dic = density_grid(k)
    grid_dic_input = density_grid_input(projections, x_min, y_min, x_unit, y_unit, grid_dic, k)
    portion = density_grid_cal(grid_dic_input, k, threshold)
    return portion

def density_grid(k):
    """
    :param k:
    :return: a dictionary of the grids for partitioning items and check density
    """
    grid_dic = {}
    for i in range(k):
        grid_dic[i] = {}
        for j in range(k):
            grid_dic[i][j] = [{}, 0]
    return grid_dic

def density_grid_input(projections, x_min, y_min, x_unit, y_unit, grid_dic, k):
    for i in range(projections.shape[0]):
        x = (projections[i][0] - x_min) // x_unit
        if x == k:
            x -= 1
        y = (projections[i][1] - y_min) // y_unit
        if y == k:
            y -= 1
        feature = int(projections[i][2])
        try:
            grid_dic[x][y][0][feature] += 1
        except:
            grid_dic[x][y][0][feature] = 1
        grid_dic[x][y][1] += 1
    return grid_dic

def density_grid_cal(grid_dic_input, k, threshold):
    satisfy_num = 0
    for i in range(k):
        for j in range(k):
            total = grid_dic_input[i][j][1]
            not_found, all_zero = True, True
            for n in grid_dic_input[i][j][0].values():
                if n != 0:
                    all_zero = False
                if not_found and n > total * threshold:
                    satisfy_num += 1
                    not_found = False
            if all_zero:
                satisfy_num += 1
    return satisfy_num / (k * k)


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
    print("1: wiki edgeset")
    print("2: hr2 edgeset")
    print("3: lock edgeset")
    while True:
        try:
            i = int(input("(Index) Select tested edgeset: "))
            break
        except:
            print("Invalid index.")
    return i


def show_evaluation_results(embed_obj, vis_obj, k=10, timing=False):

    if embed_obj.has_feature:
        featured_projection = np.insert(vis_obj.projections, 2, list(vis_obj.embeddings.feature), axis=1)
        print("Visualization quality: {}".format(density_check(featured_projection, k=10, threshold=0.5)))
    
    graph = embed_obj.graph
    highDimEmbed = embed_obj.embeddings
    lowDimEmbed = vis_obj.projections
    randomHighDimEmbed = randomEmbeddings(embed_obj.embeddings)
    randomLowDimEmbed = randomEmbeddings(vis_obj.projections)

    high = compare_KNN(graph, highDimEmbed, k, timing=timing)
    high_base = compare_KNN(graph, randomHighDimEmbed, k, timing=timing)
    low = compare_KNN(graph, lowDimEmbed, k, timing=timing)
    low_base = compare_KNN(graph, randomLowDimEmbed, k, timing=timing)
    high_v_low = np.average(compare_KNN_matrix(construct_knn_from_embeddings(highDimEmbed, k), construct_knn_from_embeddings(lowDimEmbed, k)))

    print("KNN embedding accuracy: {:.2f}, with baseline: {:.2f}".format(high, high_base))
    print("KNN visualizing accuracy: {:.2f}, with baseline: {:.2f}".format(low, low_base))
    print("KNN dimension reduction accuracy: {:.2f}".format(high_v_low))

    # while keepGoing:
    #     check = input("(Index) Select evaluation benchmark: ")
    #     # compared with d(graph, random_embedding)
    #     if check == "1":
    #         if high == -1:
    #             high = compare_KNN(graph, highDimEmbed, k, timing=timing)
    #         if high_base == -1:
    #             high_base = compare_KNN(graph, randomHighDimEmbed, k, timing=timing)
    #         print("KNN embedding accuracy: {:.2f}, with baseline: {:.2f}".format(high, high_base))
    #     elif check == "2":
    #         if low == -1:
    #             low = compare_KNN(graph, lowDimEmbed, k, timing=timing)
    #         if low_base == -1:
    #             low_base = compare_KNN(graph, randomLowDimEmbed, k, timing=timing)
    #         print("KNN visualizing accuracy: {:.2f}, with baseline: {:.2f}".format(low, low_base))
    #     elif check == "3":
    #         if high_v_low == -1:
    #             t0 = time.time()
    #             high_v_low = np.average(compare_KNN_matrix(construct_knn_from_embeddings(highDimEmbed, k), construct_knn_from_embeddings(lowDimEmbed, k)))
    #             t1 = time.time()
    #             if timing:
    #                 print("Time to compare high-low KNNs: {:.2f}".format(t1 - t0))
    #         print("KNN dimension reduction accuracy: {:.2f}".format(high_v_low))
    #     elif check.upper() == "A":
    #         if high == -1:
    #             high = compare_KNN(graph, highDimEmbed, k)
    #         if high_base == -1:
    #             high_base = compare_KNN(graph, randomHighDimEmbed, k)
    #         if low == -1:
    #             low = compare_KNN(graph, lowDimEmbed, k)
    #         if low_base == -1:
    #             low_base = compare_KNN(graph, randomLowDimEmbed, k)
    #         if high_v_low == -1:
    #             high_v_low = np.average(compare_KNN_matrix(construct_knn_from_embeddings(highDimEmbed, k), construct_knn_from_embeddings(lowDimEmbed, k)))
    #         print("KNN embedding accuracy: {:.2f}, with baseline: {:.2f}".format(high, high_base))
    #         print("KNN visualizing accuracy: {:.2f}, with baseline: {:.2f}".format(low, low_base))
    #         print("KNN dimension reduction accuracy: {:.2f}".format(high_v_low))
    #     elif check.upper() == "Q":
    #         keepGoing = False
    #     else:
    #         print("Invalid evaluation type.")

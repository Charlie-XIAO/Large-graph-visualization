import numpy as np
import networkx as nx
import pandas as pd
import scipy
from sklearn.neighbors import NearestNeighbors

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
    while True:
        try:
            i = int(input("Enter test index: "))
            break
        except:
            print("Invalid index.")
    return i


def randomEmbeddings(embeddings, distribution="uniform"):
    """
    Generate a random embedding of the same size as the input embedding,
    following Uniform distribution $Unif(min(embeddings), max(embeddings))$
    """
    if isinstance(embeddings, pd.DataFrame):
        embeddings = embeddings.to_numpy()[:,:-1]
    if distribution == "uniform":
        min = embeddings.min()
        max = embeddings.max()
        length = embeddings.shape[0]*embeddings.shape[1]
        return np.random.rand(length).reshape(embeddings.shape) * (int(max) - int(min)) + min
        
    elif distribution == "normal":
        pass
    else:
        raise ValueError(f"'{distribution}' distribution not implemented")


def construct_knn_from_embeddings(embeddings, k):
    # convert embeddings to adjacency matrix of KNN Graph
    if(isinstance(embeddings, pd.DataFrame)):
        embeddings = np.array(embeddings.sort_index())      #   sort embeddings based on row index, so that node i is at row i
    neigh = NearestNeighbors(n_neighbors=k)     # note that KNN includes the point itself
    neigh.fit(embeddings)
    return neigh.kneighbors_graph(embeddings)


def construct_knn_from_graph(graph, k):
    knn_of_graph = scipy.sparse.lil_matrix((len(graph), len(graph)), dtype=np.intc)  # initialize the knn matrix
    
    for v in range(len(graph)):
        tree = nx.bfs_tree(nx.convert_node_labels_to_integers(graph), v)
        knn_indices = list(tree)[0: k]
        knn_of_graph[v, knn_indices] = 1
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


def compare_KNN(graph, embeddings, k):
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
    
    knn_of_graph = construct_knn_from_graph(graph, k)
    knn_of_embed = construct_knn_from_embeddings(embeddings, k)

    knn_accuracies = compare_KNN_matrix(knn_of_graph, knn_of_embed)
    knn_accuracy = np.average(knn_accuracies)

    return knn_accuracy
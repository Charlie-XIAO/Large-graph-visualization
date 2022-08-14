import os
import numpy as np
import networkx as nx
from networkx import bfs_edges as bfs_edges_without_depth_now
from tests import bfs_edges as bfs_edges_with_depth_now
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scipy.sparse
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
    :param k: the size of neighborhood, the K in KNN. The point itself is not included.
    return: the KNN matrix of the embeddings, of type numpy.ndaray
    """
    # convert embeddings to adjacency matrix of KNN Graph
    if(isinstance(embeddings, pd.DataFrame)):
        embeddings = np.array(embeddings.sort_index())      #   sort embeddings based on row index, so that node i is at row i
    neigh = NearestNeighbors(n_neighbors=k)     # note that KNN includes the point itself
    neigh.fit(embeddings)
    
    # return neigh.kneighbors_graph(embeddings)
    return neigh.kneighbors_graph(mode="distance").toarray()

def graph2embeddingKnn(graph, k):
    pass


# size of int16, int32, int64 
MAXSIZE = {
    np.int16: 2**15,
    np.int32: 2**31,
    np.int64: 2**63
}

# TODO: Reasonable dtype for knn
# Intuitively, KNN shouldn't be that large. 2**15 ~= 32768 should be enough
KNN_DTYPE = np.int32

def construct_knn_from_graph(graph, k, mode="distance", sparse=True):
    graph = nx.convert_node_labels_to_integers(graph)

    if mode == "connectivity":

        t0 = time.time()
        # knn_of_graph = np.zeros((len(graph), len(graph)), dtype=KNN_DTYPE)
        knn_of_graph = scipy.sparse.lil_matrix((len(graph), len(graph)), dtype=KNN_DTYPE)
        
        for v in range(len(graph)):
            
            ### avoid duplicated visiting
            # method 1: set
            visited = {v}

            bfs_iter = bfs_edges_without_depth_now(graph, v)
            try:
                while len(visited) - 1 < k and bfs_iter:
                    e = next(bfs_iter)
                    knn_of_graph[v, e[1]] = 1
                    visited.add(e[1])
            except StopIteration:   # the last element is iterated
                print(f"End early at {v}")
                if len(visited) - 1 < k:
                    remainder = k - len(visited) + 1
                    # # sample implementation #1
                    # sampled_indices = np.random.choice(np.nonzero(knn_of_graph[v, :] == 0)[1], remainder, replace=False)
                    
                    # sample implementation #2 (lim_matrix == 0 is inefficient)
                    sampled_indices = np.random.choice(
                        np.nonzero(knn_of_graph[v, :].toarray() == 0)[1], 
                        remainder, 
                        replace=False)

                    knn_of_graph[v, sampled_indices] = np.min((len(graph), 2**15))      # set distance to large number

        t1 = time.time()
        print(f"Time to construct knn matrix of graph ({mode}): {t1 - t0}")

    elif mode == "distance":        # TODO: implement this mode
        t0 = time.time()
        # knn_of_graph = np.zeros((len(graph), len(graph)), dtype=KNN_DTYPE)
        knn_of_graph = scipy.sparse.lil_matrix((len(graph), len(graph)), dtype=KNN_DTYPE)
        
        for v in range(len(graph)):
            
            ### avoid duplicated visiting
            # method 1: set
            visited = {v}

            bfs_iter = bfs_edges_with_depth_now(graph, v)
            try:
                while len(visited) - 1 < k and bfs_iter:
                    e, depth_now = next(bfs_iter)
                    knn_of_graph[v, e[1]] = depth_now
                    visited.add(e[1])
            except StopIteration:   # the last element is iterated
                # print("End early")
                if len(visited) - 1 < k:
                    remainder = k - len(visited) + 1
                    # # sample implementation #1
                    # sampled_indices = np.random.choice(np.nonzero(knn_of_graph[v, :] == 0)[1], remainder, replace=False)
                    
                    # sample implementation #2 (lim_matrix == 0 is inefficient)
                    sampled_indices = np.random.choice(
                        np.nonzero(knn_of_graph[v, :].toarray() == 0)[1], 
                        remainder, 
                        replace=False)

                    knn_of_graph[v, sampled_indices] = np.min((len(graph), 2**15))      # set distance to large number

        t1 = time.time()
        print(f"Time to construct knn matrix of graph (distance): {t1 - t0}")

    else:
        raise ValueError(f"Mode {mode} is not supported. Use 'connectivity' or 'distance' instead")

    if sparse:
        return scipy.sparse.csr_matrix(knn_of_graph)
    return knn_of_graph.toarray()


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


def compare_KNN(graph, embeddings, k):
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
    # if isinstance(graph.nodes[0], str):
    graph = nx.convert_node_labels_to_integers(graph)
    knn_of_graph = construct_knn_from_graph(nx.convert_node_labels_to_integers(graph), k)
    knn_of_embed = construct_knn_from_embeddings(embeddings, k)

    knn_accuracies = compare_KNN_matrix(knn_of_graph, knn_of_embed)
    knn_accuracy = np.average(knn_accuracies)
    
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
    grid_dic_input, feature_dic = density_grid_input(projections, x_min, y_min, x_unit, y_unit, grid_dic, k)
    portion, feature_dic_final = density_grid_cal(grid_dic_input, k, threshold, feature_dic)
    distance_rate=feature_dis_cal(feature_dic_final, k)
    return portion, distance_rate

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
    feature_dic={}
    for i in range(projections.shape[0]):
        x = (projections[i][0] - x_min) // x_unit
        if x == k:
            x -= 1
        y = (projections[i][1] - y_min) // y_unit
        if y == k:
            y -= 1
        feature = int(projections[i][2])
        feature_dic[feature]=[]
        try: 
            grid_dic[x][y][0][feature] += 1
        except:
            grid_dic[x][y][0][feature] = 1
        grid_dic[x][y][1] += 1
    return grid_dic, feature_dic

def density_grid_cal(grid_dic_input, k, threshold, feature_dic):
    satisfy_num = 0
    for i in range(k):
        for j in range(k):
            total = grid_dic_input[i][j][1]
            not_found = True
            dominant, dominant_value = None, 0
            for feature, n in grid_dic_input[i][j][0].items():
                if not_found and n > total * threshold:
                    satisfy_num += 1
                    not_found = False
                    dominant, dominant_value = feature, n
                elif n > dominant_value:
                    dominant, dominant_value = feature, n
            if dominant is not None:
                feature_dic[dominant].append([i, j])
    return satisfy_num / (k * k), feature_dic

def feature_dis_cal(feature_dic_final,k):
    feature_count = 0
    x_total_sum, y_total_sum = 0, 0
    for _, grid_list in feature_dic_final.items():
        feature_count += 1
        grid_n = len(grid_list)
        x_dis_sum, y_dis_sum = 0, 0
        if grid_list:
            for i in range(grid_n):
                for j in range(grid_n):
                    if abs(grid_list[i][0] - grid_list[j][0]) > 1:
                        x_dis_sum += abs(grid_list[i][0] - grid_list[j][0])
                    if abs(grid_list[i][1] - grid_list[j][1]) > 1:
                        y_dis_sum += abs(grid_list[i][1] - grid_list[j][1])
        if len(grid_list) == 0:
            x_total_sum += 1
            y_total_sum += 1
        elif len(grid_list) > 1:
            x_avr = x_dis_sum / (grid_n * (grid_n - 1))
            y_avr = y_dis_sum / (grid_n * (grid_n - 1))
            x_total_sum += x_avr / k
            y_total_sum += y_avr / k
    return (x_total_sum / feature_count) * (y_total_sum / feature_count)


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

def show_evaluation_results(config, embed_obj, vis_obj, k=10):

    print_block("EVALUATION RESULTS on {} EDGELIST".format(config["data"]))

    print("Embedding method: {}".format(config["embed"]), end=" ( ")
    for x in vars(embed_obj):
        if x not in ["edgeset", "graph", "featureset", "embeddings", "has_feature"]:
            print("{}={}".format(x, vars(embed_obj)[x]), end=", ")
    print(")")
    print("Visualization method: {}".format(config["vis"]), end=" ( ")
    for x in vars(vis_obj):
        if x not in ["embeddings", "has_feature", "X", "location", "projections"]:
            print("{}={}".format(x, vars(vis_obj)[x]), end=", ")
    print(")\n")

    if embed_obj.has_feature:
        featured_projection = np.insert(vis_obj.projections, 2, list(vis_obj.embeddings.feature), axis=1)
        density, distance = density_check(featured_projection, k=10, threshold=0.6)
        print("Visualization quality (density): {:.4f}".format(density))
        print("Visualization quality (distance): {:.4f}".format(distance))
        print("Score (lower is better): {:.4f}\n".format(distance / (density ** 2)))
    
    graph = embed_obj.graph
    highDimEmbed = embed_obj.embeddings
    lowDimEmbed = vis_obj.projections
    randomHighDimEmbed = randomEmbeddings(embed_obj.embeddings)
    randomLowDimEmbed = randomEmbeddings(vis_obj.projections)

    high = compare_KNN(graph, highDimEmbed, k)
    print("k = {}, Embedding KNN accuracy: {:.4f}".format(k, high), end=", ")
    high_base = compare_KNN(graph, randomHighDimEmbed, k)
    print("with baseline {:.4f}".format(high_base))
    low = compare_KNN(graph, lowDimEmbed, k)
    print("k = {}, Visualization KNN accuracy: {:.4f}".format(k, low), end=", ")
    low_base = compare_KNN(graph, randomLowDimEmbed, k)
    print("with baseline {:.4f}".format(low_base))
    high_v_low = np.average(compare_KNN_matrix(construct_knn_from_embeddings(highDimEmbed, k), construct_knn_from_embeddings(lowDimEmbed, k)))
    print("k = {}, Dimension reduction KNN accuracy: {:.4f}".format(k, high_v_low))

def setup(config):
    """
    :param config:
    :return: dim, edgeset, featureset, location
    """
    print_block(f"Running {config['embed']} + {config['vis']} on {config['data']}")
    if config["description"] == "":
        config["location"] = os.path.join(config["image_folder"], f"{config['data']}_{config['embed']}_{config['vis']}.{config['image_format']}")
    else:
        config["location"] = os.path.join(config["image_folder"], f"{config['data']}_{config['embed']}_{config['vis']}_{config['description']}.{config['image_format']}")
    return config["dim"], config["edgeset"], config["featureset"], config["location"]


if __name__ == "__main__":
    graph =  nx.Graph(nx.gnm_random_graph(30, 50))
    highDimEmbed = np.random.rand(30,64)
    k = 6

    knn_of_graph = construct_knn_from_graph(nx.convert_node_labels_to_integers(graph), k)

    for n in range(30):
        print(np.sum(knn_of_graph[n,:]))

    print(compare_KNN(graph, highDimEmbed, k))

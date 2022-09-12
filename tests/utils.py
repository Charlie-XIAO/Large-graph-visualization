import os
import numpy as np
import networkx as nx
import pandas as pd
import scipy.sparse
from datetime import datetime
from prettytable import PrettyTable

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import normalized_mutual_info_score, rand_score, f1_score
from sknetwork.clustering import Louvain, PropagationClustering

from networkx import bfs_edges as bfs_edges_without_depth_now
from tests import bfs_edges as bfs_edges_with_depth_now


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

        # knn_of_graph = np.zeros((len(graph), len(graph)), dtype=KNN_DTYPE)
        knn_of_graph = scipy.sparse.lil_matrix((len(graph), len(graph)), dtype=KNN_DTYPE)
        
        count = 0
        total = len(graph)

        for v in range(len(graph)):
            
            if count % vis_step == 0:
                print(f"[t-SGNE] Finished computing the neighbors of {count}/{total} nodes")

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
            count += 1
        print(f"[t-SGNE] Finished computing the neighbors of {count}/{total} nodes, finished!")

    elif mode == "distance":

        # knn_of_graph = np.zeros((len(graph), len(graph)), dtype=KNN_DTYPE)
        knn_of_graph = scipy.sparse.lil_matrix((len(graph), len(graph)), dtype=KNN_DTYPE)
        
        count = 0
        total = len(graph)

        if total < 1000:
            vis_step = 100
        elif total >= 1000 and total < 100000:
            vis_step = 1000
        else:
            vis_step = 10000
        
        for v in range(len(graph)):
            
            if count % vis_step == 0:
                print(f"[t-SGNE] Finished computing the neighbors of {count}/{total} nodes")

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
            count += 1
        print(f"[t-SGNE] Finished computing the neighbors of {count}/{total} nodes, finished!")

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
###                  CLUSTERING ACCURACY                   ###
### ========== ========== ========== ========== ========== ###

def get_projection_clustering_labels(projections, n_clusters=8):
    """
    :param projections: a numpy ndarray, representing the 2D projections of the graph
    :param n_clusters: the number of clusters to form as well as the number of centroids to generate,
                       alternatively, can be a range of such numbers, then use silhouette scores to find out the optimal choice

    :return: opt_k, which is the optimal number of clusters for kMeans clustering, determined by Silhoutte score
    :return: opt_score, which is the Silhoutte score corresponding to opt_k, float of range [-1, 1]
    :return: proj_labels, which is a numpy array containing labels of each node as clustered in the 2D projections,
             where the labels are in the same node ordering as in graph.nodes()
    """
    if isinstance(n_clusters, range):
        opt_k, opt_score, proj_labels = 0, -2, None
        for k in n_clusters:
            kmeans = KMeans(n_clusters=k, verbose=0, random_state=0).fit(projections)
            score = silhouette_score(projections, kmeans.labels_)
            if score > opt_score:
                opt_k, opt_score, proj_labels = k, score, kmeans.labels_
        return opt_k, opt_score, proj_labels
    else:
        kmeans = KMeans(n_clusters=n_clusters, verbose=0, random_state=0).fit(projections)
        score = silhouette_score(projections, kmeans.labels_)
        return n_clusters, score, kmeans.labels_

def get_graph_clustering_labels(graph):
    """
    :param graph: the networkx graph to be clustered
    :return: k, which is the number of clusters
    :return: graph_labels
    """
    A = nx.to_scipy_sparse_matrix(graph)
    louvain = Louvain().fit_transform(A)
    # propagation = PropagationClustering().fit_transform(A)
    return len(set(louvain)), louvain

def clustering_accuracy(graph, projections, has_feature=False, features=None):
    """
    :param graph: the networkx graph to be clustered
    :param projections: a numpy ndarray, representing the 2D projections of the graph
    :param has_feature: if True, also return the evaluation between the graph labels and the projections clustering
    :param features: vis_obj.embeddings.feature, pd.DataFrame series object, containing the DEFAULT labels of the nodes

    :return: ks, a python list including the number of clusters when evaluating the graph clustering and the projections clustering
             if has_feature = True, also includes the number of clusters when evaluating the graph labels and the projections clustering
    :return: NMIscores, a python list including the NMI between the graph clustering and the projections clustering
             if has_feature = True, also includes the NMI between the graph labels and the projections clustering
    :return: RIscores, a python list including the RI between the graph clustering and the projections clustering
             if has_feature = True, also includes the RI between the graph labels and the projections clustering

    Explanation:
    ------------------------------------------------------------------------------------------------
    - NMIscore  Score between 0.0 and 1.0 in normalized nats (based on the natural logarithm).
                1.0 stands for perfectly complete labeling.
                
        Normalized Mutual Information between two clusterings. Normalized Mutual Information (NMI)
        is a normalization of the Mutual Information (MI) score to scale the results between 0
        (no mutual information) and 1 (perfect correlation). In this function, mutual information
        is normalized by some generalized mean of H(labels_true) and H(labels_pred)), defined by the
        average_method. This measure is not adjusted for chance. Therefore adjusted_mutual_info_score
        might be preferred. This metric is independent of the absolute values of the labels: a
        permutation of the class or cluster label values won't change the score value in any way.
        This metric is furthermore symmetric: switching label_true with label_pred will return the
        same score value. This can be useful to measure the agreement of two independent label
        assignments strategies on the same dataset when the real ground truth is not known.
    
    - RIscore   Similarity score between 0.0 and 1.0, inclusive, 1.0 stands for perfect match.

        The Rand Index (RI) computes a similarity measure between two clusterings by considering all
        pairs of samples and counting pairs that are assigned in the same or different clusters in
        the predicted and true clusterings. The raw RI score is: RI = (number of agreeing pairs) / (number of pairs)

    - Fmeasure  F1 score of the positive class in binary classification.

        Compute the F1 score, also known as balanced F-score or F-measure. The F1 score can be
        interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its
        best value at 1 and worst score at 0. The relative contribution of precision and recall to
        the F1 score are equal. The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    """
    k, graph_labels = get_graph_clustering_labels(graph)
    proj_labels = get_projection_clustering_labels(projections, n_clusters=k)[2]
    NMIscore = normalized_mutual_info_score(graph_labels, proj_labels)
    RIscore = rand_score(graph_labels, proj_labels)
    # Fmeasure = f1_score(graph_labels, proj_labels)
    ks, NMIscores, RIscores = [k], [NMIscore], [RIscore]
    # ks, NMIscores, RIscores, Fmeasures = [k], [NMIscore], [RIscore], [Fmeasure]
    if has_feature and features is not None:
        graph_labels = np.array(features)
        k = len(set(graph_labels))
        ks.append(k)
        proj_labels = get_projection_clustering_labels(projections, n_clusters=k)[2]
        NMIscores.append(normalized_mutual_info_score(graph_labels, proj_labels))
        RIscores.append(rand_score(graph_labels, proj_labels))
        # Fmeasures.append(f1_score(graph_labels, proj_labels))
    return ks, NMIscores, RIscores


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

def show_evaluation_results(config, embed_obj, vis_obj, k=10, write_to_log=False):

    print_block("EVALUATION RESULTS on {} EDGELIST".format(config["data"]))

    print(datetime.now().strftime("%m/%d/%Y %H:%M:%S"))
    print()

    embedding_table = PrettyTable()
    field_names, row_contents = ["Embedding"], [config["embed"]]
    assert embed_obj.duration is not None, "embed duration shouldn't be None"
    field_names.append("Embed Duration")
    row_contents.append(f"{embed_obj.duration:.3f}s")
    embedding_vars = vars(embed_obj)
    for x in embedding_vars:
        if x not in ["duration", "edgeset", "graph", "featureset", "embeddings", "has_feature"]:
            field_names.append(x)
            row_contents.append(embedding_vars[x])
    embedding_table.field_names = field_names
    embedding_table.add_row(row_contents)
    print(embedding_table.get_string())

    visualization_table = PrettyTable()
    field_names, row_contents = ["Visualization"], [config["vis"]]
    assert vis_obj.duration is not None, "vis duration shouldn't be None"
    field_names.append("Vis Duration")
    row_contents.append(f"{vis_obj.duration:.3f}s")
    visualization_vars = vars(vis_obj)
    for x in visualization_vars:
        if x not in ["duration", "embeddings", "has_feature", "X", "location", "projections", "graph", "knn_matrix"]:
            field_names.append(x)
            row_contents.append(visualization_vars[x])
    visualization_table.field_names = field_names
    visualization_table.add_row(row_contents)
    print(visualization_table.get_string())


    score_table = PrettyTable()
    field_names, row_contents = [], []

    field_names.append("Total Duration")
    total_duration = embed_obj.duration + vis_obj.duration
    row_contents.append(f"{total_duration:.3f}s")

    log_write_type = {
        0: "minimal",       
        1: "default",
    }[config["eval"]]

    if log_write_type == "minimal":
        score_table.field_names = field_names
        score_table.add_row(row_contents)
        print(score_table.get_string())

        # write log file for *each dataset*
        if not os.path.exists(os.path.join( os.getcwd(), "log",)):
            os.mkdir("log")
        os.chdir(os.path.join(os.getcwd(), "log",))
        if not os.path.exists(os.path.join( os.getcwd(), config['data'],)):
            os.mkdir(config['data'])
        os.chdir(config['data'])

        if not os.path.exists(f"log_{config['data']}.csv"):
            with open(f"log_{config['data']}.csv", "a") as log_file:
                    log_file.write(
                        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                            "Time",
                            "Dataset",
                            "Embedding method",
                            "Visualization method",
                            "Embedding size",
                            "Embedding duration",
                            "Visualization duration",
                            "Total duration",
                            "Density",
                            "Distance",
                            "k0",
                            "Clustering NMI",
                            "Clustering RI",
                            "k1",
                            "Labels NMI",
                            "Labels RI",
                        )
                    )

        with open(f"log_{config['data']}.csv", "a") as log_file:
            log_file.write(
                "{},{},{},{},{:.3f},{:.3f},{:.3f},{},{},{},{},{},{},{},{}\n".format(
                    datetime.now().__str__(),
                    config['data'],
                    config['embed'],
                    config['vis'],
                    config['dim'],
                    embed_obj.duration,
                    vis_obj.duration,
                    embed_obj.duration + vis_obj.duration,
                    -1,-1,-1,-1,-1,-1,-1,-1
                )
            )
    elif log_write_type == "default":
        features = None
        featured_projection = np.insert(vis_obj.projections, 2, list(get_graph_clustering_labels(embed_obj.graph)[1]), axis=1)
        density, distance = density_check(featured_projection, k=10, threshold=0.6)
        field_names.extend(["Density", "Distance"])
        row_contents.extend([format(density, ".4f"), format(distance, ".4f")])

        ks, NMIscores, RIscores = clustering_accuracy(embed_obj.graph, vis_obj.projections, embed_obj.has_feature, features)
        field_names.extend(["Clustering NMI", "Clustering RI"])
        row_contents.extend(["(k={}) {:.4f}".format(ks[0], NMIscores[0]), "(k={}) {:.4f}".format(ks[0], RIscores[0])])

        if False:
        # if embed_obj.has_feature:
            field_names.extend(["Labels NMI", "Labels RI"])
            row_contents.extend(["(k={}) {:.4f}".format(ks[1], NMIscores[1]), "(k={}) {:.4f}".format(ks[1], RIscores[1])])
        
        score_table.field_names = field_names
        score_table.add_row(row_contents)
        print(score_table.get_string())


        # write log file for *each dataset*
        if not os.path.exists(os.path.join( os.getcwd(), "log",)):
            os.mkdir("log")
        os.chdir(os.path.join(os.getcwd(), "log",))
        if not os.path.exists(os.path.join( os.getcwd(), config['data'],)):
            os.mkdir(config['data'])
        os.chdir(config['data'])

        if not os.path.exists(f"log_{config['data']}.csv"):
            with open(f"log_{config['data']}.csv", "a") as log_file:
                    log_file.write(
                        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                            "Time",
                            "Dataset",
                            "Embedding method",
                            "Visualization method",
                            "Embedding size",
                            "Embedding duration",
                            "Visualization duration",
                            "Total duration",
                            "Density",
                            "Distance",
                            "k0",
                            "Clustering NMI",
                            "Clustering RI",
                            "k1",
                            "Labels NMI",
                            "Labels RI",
                        )
                    )
        # check if indices exist.
        if 'density' not in locals():
            density = -1
        if 'distance' not in locals():
            distance = -1
        if len(ks) < 2:
            ks.append(-1)
            NMIscores.append(-1)
            RIscores.append(-1)

        with open(f"log_{config['data']}.csv", "a") as log_file:
            log_file.write(
                "{},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{},{:.3f},{:.3f},{},{:.3f},{:.3f}\n".format(
                    datetime.now().__str__(),
                    config['data'],
                    config['embed'],
                    config['vis'],
                    config['dim'],
                    embed_obj.duration,
                    vis_obj.duration,
                    embed_obj.duration + vis_obj.duration,
                    density,
                    distance,
                    ks[0],
                    NMIscores[0],
                    RIscores[0],
                    ks[1],
                    NMIscores[1],
                    RIscores[1],
                )
            )
    else:
        raise ValueError("Unknown log write type.")

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


### ========== ========== ========== ========== ========== ###
###      THE CODES BELOW ARE FOR TESING PURPOSE ONLY       ###
### ========== ========== ========== ========== ========== ###
if __name__ == "__main__":
    graph =  nx.Graph(nx.gnm_random_graph(30, 50))
    highDimEmbed = np.random.rand(30, 64)
    k = 6

    knn_of_graph = construct_knn_from_graph(nx.convert_node_labels_to_integers(graph), k)

    for n in range(30):
        print(np.sum(knn_of_graph[n,:]))

    print(compare_KNN(graph, highDimEmbed, k))

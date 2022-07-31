import os

import pandas as pd

import networkx as nx

import plotly.express as px
from sklearn.manifold import TSNE

from embedders.utils import *
from embedders.previous_works.DeepWalk import DeepWalk
from embedders.previous_works.SDNE import SDNE

### ========== ========= ========== ========== ========== ###
### ========== =========    TEST    ========== ========== ###
### ========== ========= ========== ========== ========== ###

def get_dataset(dataset, topic):
    return "./datasets/" + dataset + "/" + dataset + "_" + topic + ".txt"

def TSNE_plot(df, dataset, path, by_categories=False, by_labels=False, by_clusters=False, raw=True):
    """
    :param df: high-dimensional embedding with last three columns as special results
    :param dataset: dataset name
    :param path: directory to save plot from "./images/" folder, without ".jpg" suffix
    :param by_categories: set True to save plot colored by categories
    :param by_labels: set True to save plot colored by labels
    :param by_cluster: set True to save plot colored by "nx.clustering()" clusters
    :param raw: set False to disable saving uncolored raw plot
    :return: None
    """
    count = by_categories + by_labels + by_clusters
    print("### Dataframe generated as follows:")
    print(df)
    features = df.iloc[:, :-count]
    tsne = TSNE(n_components=2, verbose=1, random_state=0)
    print("### TSNE initialized, processing", end="... ")
    projections = tsne.fit_transform(features)
    print("2D projections done.")
    if not os.path.exists("./images/" + dataset):
        os.makedirs("./images/" + dataset)
        print("Assigned directory undetected, new folder {} created.".format("./images/" + dataset))
    if by_categories:
        fig = px.scatter(projections, x=0, y=1, color=df.categories)
        print("### Plot figure (colored by categories) created, saving", end="... ")
        location = "./images/" + path + "_categories.jpg"
        fig.write_image(location)
        print("Plot saved: [ {} ]".format(os.getcwd() + location))
    if by_labels:
        fig = px.scatter(projections, x=0, y=1, color=df.labels)
        print("### Plot figure (colored by labels) created, saving", end="... ")
        location = "./images/" + path + "_labels.jpg"
        fig.write_image(location)
        print("Plot saved: [ {} ]".format(os.getcwd() + location))
    if by_clusters:
        fig = px.scatter(projections, x=0, y=1, color=df.clusters)
        print("### Plot figure (colored by clusters) created, saving", end="... ")
        location = "./images/" + path + "_clusters.jpg"
        fig.write_image(location)
        print("Plot saved: [ {} ]".format(os.getcwd() + location))
    if raw:
        fig = px.scatter(projections, x=0, y=1)
        print("### Plot figure created, saving", end="... ")
        location = "./images/" + path + ".jpg"
        fig.write_image(location)
        print("Plot saved: [ {} ]".format(os.getcwd() + location))
    if not by_categories and not by_clusters and not by_labels and not raw:
        print("### No output plot, select at least one type of plot.")

def embed4plot(dataset, description, embeddings, clusters, by_categories=False, by_labels=False, by_clusters=False, raw=True):
    """
    :param dataset: dataset name
    :param description: description for plotted data
    :param embeddings: high-dimensional embedding with last three columns as special results
    :param by_categories: set True to save plot colored by categories
    :param by_labels: set True to save plot colored by labels
    :param by_cluster: set True to save plot colored by "nx.clustering()" clusters
    :param raw: set False to disable saving uncolored raw plot
    :return: None
    """
    if by_labels:
        labels = {}
        print("### Loading labels data", end="... ")
        try:
            with open(get_dataset(dataset, "labels")) as f:
                for line in f:
                    k, v = line.split()
                    labels[k] = v
            embeddings["labels"] = [labels[node] for node in embeddings.index]
            print("Done.")
        except Exception as e:
            print("Failed.")
            by_labels = False
            print(e)
    if by_categories:
        categories = {}
        print("### Loading categories data", end="... ")
        try:
            with open(get_dataset(dataset, "categories")) as f:
                for line in f:
                    k, v = line.split()
                    categories[k] = v
            embeddings["categories"] = [categories[node] for node in embeddings.index]
            print("Done.")
        except Exception as e:
            print("Failed.")
            by_categories = False
            print(e)
    if by_clusters:
        print("### Computing clustering data", end="... ")
        embeddings["clusters"] = [clusters[node] for node in embeddings.index]
        print("Done.")
    TSNE_plot(embeddings, dataset, dataset + "/" + description, by_categories=by_categories, by_labels=by_labels, by_clusters=by_clusters, raw=raw)


def test_DeepWalk(dataset, description, create_using=nx.Graph(), delimiter=" ", by_categories=False, by_labels=False, by_clusters=False, raw=True):
    """
    :param dataset: dataset name
    :param description: description for plotted data
    :param create_using: graph type DEFAULT nx.Graph() OPTION nx.DiGraph() etc.
    :param delimiter DEFAULT " ":
    :param by_categories: set True to save plot colored by categories
    :param by_labels: set True to save plot colored by labels
    :param by_cluster: set True to save plot colored by "nx.clustering()" clusters
    :param raw: set False to disable saving uncolored raw plot
    :return: None
    """
    G = nx.read_edgelist(get_dataset(dataset, "edgelist"), delimiter=delimiter, create_using=create_using, nodetype=None, data=[("weight", int)])
    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    model.train(window_size=5, iter=3)
    embeddings = pd.DataFrame.from_dict(model.get_embeddings())
    embed4plot(dataset, description, embeddings.T, nx.clustering(G), by_categories=by_categories, by_labels=by_labels, by_clusters=by_clusters, raw=raw)
    
def test_SDNE(dataset, description, create_using=nx.Graph(), delimiter=" ", by_categories=False, by_labels=False, by_clusters=False, raw=True):
    """
    :param dataset: dataset name
    :param description: description for plotted data
    :param create_using: graph type DEFAULT nx.Graph() OPTION nx.DiGraph() etc.
    :param delimiter DEFAULT " ":
    :param by_categories: set True to save plot colored by categories
    :param by_labels: set True to save plot colored by labels
    :param by_cluster: set True to save plot colored by "nx.clustering()" clusters
    :param raw: set False to disable saving uncolored raw plot
    :return: None
    """
    G = nx.read_edgelist(get_dataset(dataset, "edgelist"), delimiter=delimiter, create_using=create_using, nodetype=None, data=[("weight", int)])
    model = SDNE(G, hidden_size=[256, 128])
    model.train(batch_size=3000, epochs=40, verbose=2)
    embeddings = pd.DataFrame.from_dict(model.get_embeddings())
    embed4plot(dataset, description, embeddings.T, nx.clustering(G), by_categories=by_categories, by_labels=by_labels, by_clusters=by_clusters, raw=raw)

### ========== ========= ========== ========== ========== ###
### ========== =========    MAIN    ========== ========== ###
### ========== ========= ========== ========== ========== ###
def main():
    print("// ===== ===== ===== ===== DeepWalk ===== ===== ===== ===== //")
    test_DeepWalk("wiki", "DeepWalk_G", nx.Graph(), delimiter=" ", by_categories=True, by_labels=True, by_clusters=True, raw=True)
    print("// ===== ===== ===== ===== SDNE ===== ===== ===== ===== //")
    test_SDNE("wiki", "SDNE_G", nx.Graph(), delimiter=" ", by_categories=True, by_labels=True, by_clusters=True, raw=True)

if __name__ == "__main__":
    main()
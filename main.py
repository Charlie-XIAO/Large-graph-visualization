import argparse
import os
import sys

# DeepWalk + vis
from tests.DeepWalk_TSNE_test import DeepWalk_TSNE_test
# Node2Vec + vis
from tests.Node2Vec_TSNE_test import Node2Vec_TSNE_test
# SDNE + vis
from tests.SDNE_TSNE_test import SDNE_TSNE_test
# ShortestPath + vis
from tests.ShortestPath_TSNE_test import ShortestPath_TSNE_test


if __name__ == "__main__":

    ### default settings
    dataset_folder = os.path.join(os.path.dirname(__file__), "datasets")
    image_folder = os.path.join(os.path.dirname(__file__), "images")

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    ###  set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="name of the dataset to use", default="lock")
    parser.add_argument("--embed", help="name of the graph embedding method to use", default="deepwalk")
    parser.add_argument("--vis", help="name of the visualization method to use", default="tsne")
    
    # Below are some less used options. Feel free to tune them.
    parser.add_argument("--dim", help="dimension of the high-dimensional embedding", type=int, default=128)
    parser.add_argument("--k", help="k neighbors to use for the knn graph construction", type=int, default=10)
    parser.add_argument("--seed", help="random seed", type=int, default=20220804)             #TODO: fix a random seed for reproducibility
    parser.add_argument("--image_format", help="image format", default="png")
    parser.add_argument("--description", help="extra description of current test", default="")


    ###  parse arguments
    args = parser.parse_args()
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    config["dataset_folder"] = os.path.join(dataset_folder, config["data"])
    config["image_folder"] = os.path.join(image_folder, config["data"])
    config["edgeset"] = os.path.join(config["dataset_folder"], f"{config['data']}_edgelist.txt")
    config["featureset"] = os.path.join(config["dataset_folder"], f"{config['data']}_labels.txt")


    EMBED_METHODS = {
        "deepwalk": "DeepWalk", 
        "node2vec": "Node2Vec", 
        "sdne": "SDNE",
        "shortestpath": "ShortestPath",
        }
    VIS_METHODS = {
        "tsne": "TSNE",
        }
    
    if config["embed"].lower() not in EMBED_METHODS:
        print(f"{config['embed']} is not a valid embedding method. Valid methods are:", end=" ")
        for name in EMBED_METHODS.values():
            print(name, end="  ")
        print()
        sys.exit(1)
    else:
        config["embed"] = EMBED_METHODS[config["embed"]]
    
    if config["vis"].lower() not in VIS_METHODS:
        print(f"{config['vis']} is not a valid visualization method. Valid methods are:", end=" ")
        for name in VIS_METHODS.values():
            print(name, end="  ")
        print()
        sys.exit(1)
    else:
        config["vis"] = VIS_METHODS[config["vis"]]

    if config["image_format"] not in ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'eps', 'json']:
        print(f"{config['image_format']} is not a valid image format. Valid formats are: png, pdf")
        sys.exit(1)


    if config["embed"] == "DeepWalk":
        if config["vis"] == "TSNE":
            DeepWalk_TSNE_test(config)

    elif config["embed"] == "Node2Vec":
        if config["vis"] == "TSNE":
            Node2Vec_TSNE_test(config)
    
    elif config["embed"] == "SDNE":
        if config["vis"] == "TSNE":
            SDNE_TSNE_test(config)
    
    elif config["embed"] == "ShortestPath":
        if config["vis"] == "TSNE":
            ShortestPath_TSNE_test(config)
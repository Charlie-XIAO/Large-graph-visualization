import argparse
import os
import sys

# DeepWalk + vis
from tests.DeepWalk_TSNE_test import DeepWalk_TSNE_test
# Node2Vec + vis
from tests.Node2Vec_TSNE_test import Node2Vec_TSNE_test
# SDNE + vis
from tests.SDNE_TSNE_test import SDNE_TSNE_test


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
    parser.add_argument("--dataset", help="name of the dataset to use", default="lock")
    parser.add_argument("--embedding", help="name of the graph embedding method to use", default="deepwalk")
    parser.add_argument("--visualization", help="name of the visualization method to use", default="tsne")
    
    # Below are some less used options. Feel free to tune them.
    parser.add_argument("--dim", help="dimension of the high-dimensional embedding", type=int, default=128)
    parser.add_argument("--k", help="k neighbors to use for the knn graph construction", type=int, default=10)
    parser.add_argument("--verbose", help="verbose output", type=bool, default=False)
    parser.add_argument("--seed", help="random seed", type=int, default=20220804)             #TODO: fix a random seed for reproducibility
    parser.add_argument("--image_format", help="image format", default="png")
    parser.add_argument("--timing", help="show timing", type=bool, default=False)
    parser.add_argument("--description", help="extra description of current test", default="")


    ###  parse arguments
    args = parser.parse_args()
    config = dict()
    for arg in vars(args):
        config[arg] = getattr(args, arg)
    config["dataset_folder"] = os.path.join(dataset_folder, config["dataset"])
    config["image_folder"] = os.path.join(image_folder, config["dataset"])
    config["edgeset"] = os.path.join(config["dataset_folder"], f"{config['dataset']}_edgelist.txt")
    config["featureset"] = os.path.join(config["dataset_folder"], f"{config['dataset']}_labels.txt")


    EMBED_METHODS = {
        "deepwalk": "DeepWalk", 
        "node2vec": "Node2Vec", 
        "snde": "SDNE",
        }
    VIS_METHODS = {
        "tsne": "TSNE",
        }
    
    if config["embedding"].lower() not in EMBED_METHODS:
        print(f"{config['embedding']} is not a valid embedding method. Valid methods are: {EMBED_METHODS}")
        sys.exit(1)
    else:
        config["embedding"] = EMBED_METHODS[config["embedding"]]
    
    if config["visualization"].lower() not in VIS_METHODS:
        print(f"{config['visualization']} is not a valid visualization method. Valid methods are: {VIS_METHODS}")
        sys.exit(1)
    else:
        config["visualization"] = VIS_METHODS[config["visualization"]]

    if config["image_format"] not in ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'eps', 'json']:
        print(f"{config['image_format']} is not a valid image format. Valid formats are: png, pdf")
        sys.exit(1)


    if config["embedding"] == "DeepWalk":
        if config["visualization"] == "TSNE":
            DeepWalk_TSNE_test(config)

    elif config["embedding"] == "Node2Vec":
        if config["visualization"] == "TSNE":
            Node2Vec_TSNE_test(config)
    
    elif config["embedding"] == "SDNE":
        if config["visualization"] == "TSNE":
            SDNE_TSNE_test(config)
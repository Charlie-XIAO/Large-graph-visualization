import argparse
import os
import sys

# DeepWalk + vis
from tests.DeepWalk_TSNE_test import DeepWalk_TSNE_test
from tests.DeepWalk_TGSNE_test import DeepWalk_TGSNE_test
from tests.DeepWalk_PCA_test import DeepWalk_PCA_test
# Node2Vec + vis
from tests.Node2Vec_TSNE_test import Node2Vec_TSNE_test
from tests.Node2Vec_TGSNE_test import Node2Vec_TGSNE_test
from tests.Node2Vec_PCA_test import Node2Vec_PCA_test
# SDNE + vis
from tests.SDNE_TSNE_test import SDNE_TSNE_test
from tests.SDNE_TGSNE_test import SDNE_TGSNE_test
from tests.SDNE_PCA_test import SDNE_PCA_test
# ShortestPath + vis
from tests.ShortestPath_TSNE_test import ShortestPath_TSNE_test
from tests.ShortestPath_TGSNE_test import ShortestPath_TGSNE_test
from tests.ShortestPath_PCA_test import ShortestPath_PCA_test
# LEE + vis
from tests.LEE_TSNE_test import LEE_TSNE_test
from tests.LEE_TGSNE_test import LEE_TGSNE_test
from tests.LEE_PCA_test import LEE_PCA_test
# GLEE + vis
from tests.GLEE_TSNE_test import GLEE_TSNE_test
from tests.GLEE_TGSNE_test import GLEE_TGSNE_test
from tests.GLEE_PCA_test import GLEE_PCA_test
# SPLEE + vis
from tests.SPLEE_TSNE_test import SPLEE_TSNE_test
from tests.SPLEE_TGSNE_test import SPLEE_TGSNE_test
from tests.SPLEE_PCA_test import SPLEE_PCA_test


if __name__ == "__main__":
    
    ### default settings
    dataset_folder = os.path.join(os.path.dirname(__file__), "datasets")
    image_folder = os.path.join(os.path.dirname(__file__), "images")

    ###  set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="name of the dataset to use", default="wiki")
    parser.add_argument("--embed", help="name of the graph embedding method to use", default="deepwalk")
    parser.add_argument("--vis", help="name of the visualization method to use", default="tgsne")
    
    # Below are some less used options. Feel free to tune them.
    parser.add_argument("--dim", help="dimension of the high-dimensional embedding", type=int, default=128)
    # According to sklearn implemnetation, k should be calculated using perplexity (k = min(n_samples - 1, int(3.0 * self.perplexity + 1)))
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

    if not os.path.exists(config["image_folder"]):
        os.makedirs(config["image_folder"])
        print(f"Created folder: {config['image_folder']}")



    ### ========== ========== ========== ========== ========== ###
    ###           EMBEDDING AND VISUALIZING METHODS            ###
    ### ========== ========== ========== ========== ========== ###
    ### V V V V V V V V V V V V V V V V V V V V V V V V V V V  ###
    
    EMBED_METHODS = {
        "deepwalk": "DeepWalk", 
        "node2vec": "Node2Vec", 
        "sdne": "SDNE",
        "shortestpath": "ShortestPath",
        "lee": "LEE",
        "glee": "GLEE",
        "splee": "SPLEE",
        }
    VIS_METHODS = {
        "tsne": "TSNE",
        "tgsne": "TGSNE",
        "pca": "PCA",
        }

    
    ### A A A A A A A A A A A A A A A A A A A A A A A A A A A  ###
    ### ========== ========== ========== ========== ========== ###
    ###       END OF EMBEDDING AND VISUALIZING METHODS         ###
    ### ========== ========== ========== ========== ========== ###
    
    if config["embed"].lower() not in EMBED_METHODS:
        print(f"{config['embed']} is not a valid embedding method. Valid methods are:", end=" ")
        for name in EMBED_METHODS.values():
            print(name, end="  ")
        print()
        sys.exit(1)
    else:
        config["embed"] = EMBED_METHODS[config["embed"].lower()]
    
    if config["vis"].lower() not in VIS_METHODS:
        print(f"{config['vis']} is not a valid visualization method. Valid methods are:", end=" ")
        for name in VIS_METHODS.values():
            print(name, end="  ")
        print()
        sys.exit(1)
    else:
        config["vis"] = VIS_METHODS[config["vis"].lower()]

    if config["image_format"] not in ["png", "jpg", "jpeg", "webp", "svg", "pdf", "eps", "json"]:
        print(f"{config['image_format']} is not a valid image format. Valid formats are: png, pdf")
        sys.exit(1)


    ### ========== ========== ========== ========== ========== ###
    ###       CALLING EMBEDDING AND VISUALIZING METHODS        ###
    ### ========== ========== ========== ========== ========== ###
    ### V V V V V V V V V V V V V V V V V V V V V V V V V V V  ###

    if config["embed"] == "DeepWalk":
        if config["vis"] == "TSNE":
            DeepWalk_TSNE_test(config)
        elif config["vis"] == "TGSNE":
            DeepWalk_TGSNE_test(config)
        elif config["vis"] == "PCA":
            DeepWalk_PCA_test(config)

    elif config["embed"] == "Node2Vec":
        if config["vis"] == "TSNE":
            Node2Vec_TSNE_test(config)
        elif config["vis"] == "TGSNE":
            Node2Vec_TGSNE_test(config)
        elif config["vis"] == "PCA":
            Node2Vec_PCA_test(config)
    
    elif config["embed"] == "SDNE":
        if config["vis"] == "TSNE":
            SDNE_TSNE_test(config)
        elif config["vis"] == "TGSNE":
            SDNE_TGSNE_test(config)
        elif config["vis"] == "PCA":
            SDNE_PCA_test(config)
    
    elif config["embed"] == "ShortestPath":
        if config["vis"] == "TSNE":
            ShortestPath_TSNE_test(config)
        elif config["vis"] == "TGSNE":
            ShortestPath_TGSNE_test(config)
        elif config["vis"] == "PCA":
            ShortestPath_PCA_test(config)
    
    elif config["embed"] == "LEE":
        if config["vis"] == "TSNE":
            LEE_TSNE_test(config)
        elif config["vis"] == "TGSNE":
            LEE_TGSNE_test(config)
        elif config["vis"] == "PCA":
            LEE_PCA_test(config)
    
    elif config["embed"] == "GLEE":
        if config["vis"] == "TSNE":
            GLEE_TSNE_test(config)
        elif config["vis"] == "TGSNE":
            GLEE_TGSNE_test(config)
        elif config["vis"] == "PCA":
            GLEE_PCA_test(config)

    elif config["embed"] == "SPLEE":
        if config["vis"] == "TSNE":
            SPLEE_TSNE_test(config)
        elif config["vis"] == "TGSNE":
            SPLEE_TGSNE_test(config)
        elif config["vis"] == "PCA":
            SPLEE_PCA_test(config)

    ### A A A A A A A A A A A A A A A A A A A A A A A A A A A  ###
    ### ========== ========== ========== ========== ========== ###
    ###   END OF CALLING EMBEDDING AND VISUALIZING METHODS     ###
    ### ========== ========== ========== ========== ========== ###
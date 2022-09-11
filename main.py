import argparse
import os
import sys

# DeepWalk + vis
from tests.DeepWalk_TSNE_test import DeepWalk_TSNE_test
from tests.DeepWalk_TSGNE_test import DeepWalk_TSGNE_test
from tests.DeepWalk_PCA_test import DeepWalk_PCA_test
# Node2Vec + vis
from tests.Node2Vec_TSNE_test import Node2Vec_TSNE_test
from tests.Node2Vec_TSGNE_test import Node2Vec_TSGNE_test
from tests.Node2Vec_PCA_test import Node2Vec_PCA_test
# SDNE + vis
from tests.SDNE_TSNE_test import SDNE_TSNE_test
from tests.SDNE_TSGNE_test import SDNE_TSGNE_test
from tests.SDNE_PCA_test import SDNE_PCA_test
# ShortestPath + vis
from tests.ShortestPath_TSNE_test import ShortestPath_TSNE_test
from tests.ShortestPath_TSGNE_test import ShortestPath_TSGNE_test
from tests.ShortestPath_PCA_test import ShortestPath_PCA_test
# LEE + vis
from tests.LEE_TSNE_test import LEE_TSNE_test
from tests.LEE_TSGNE_test import LEE_TSGNE_test
from tests.LEE_PCA_test import LEE_PCA_test
# GLEE + vis
from tests.GLEE_TSNE_test import GLEE_TSNE_test
from tests.GLEE_TSGNE_test import GLEE_TSGNE_test
from tests.GLEE_PCA_test import GLEE_PCA_test
# SPLEE + vis
from tests.SPLEE_TSNE_test import SPLEE_TSNE_test
from tests.SPLEE_TSGNE_test import SPLEE_TSGNE_test
from tests.SPLEE_PCA_test import SPLEE_PCA_test
# RandomEmbed + vis (for testing purpose only)
from tests.RandomEmbed_TSGNE_test import RandomEmbed_TSGNE_test


if __name__ == "__main__":

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
        "randomembed": "RandomEmbed",
        }
    VIS_METHODS = {
        "t-sne": "t-SNE",
        "t-sgne": "t-SGNE",
        "pca": "PCA",
        }

    ### default settings
    dataset_folder = os.path.join(os.path.dirname(__file__), "datasets")
    image_folder = os.path.join(os.path.dirname(__file__), "images")

    ###  set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="name of the dataset to use", default="lastfm")
    parser.add_argument(
        "--embed", 
        help=f"name of the graph embedding method to use, choices are {list(EMBED_METHODS.keys())}", 
        default="splee")
    parser.add_argument(
        "--vis", 
        help=f"name of the visualization method to use, choices are {list(VIS_METHODS.keys())}", 
        default="t-sne")
    
    # Below are some less used options. Feel free to tune them.

    parser.add_argument("--dim", help="dimension of the high-dimensional embedding", type=int, default=128)
    # According to sklearn implemnetation, k should be calculated using perplexity (k = min(n_samples - 1, int(3.0 * self.perplexity + 1)))
    parser.add_argument("--k", help="k neighbors to use for the knn graph construction", type=int, default=10)
    parser.add_argument("--seed", help="random seed", type=int, default=20220804)             #TODO: fix a random seed for reproducibility
    parser.add_argument("--image_format", help="image format", default="png")
    parser.add_argument("--description", help="extra description of current test", default="")
    parser.add_argument("--knn_mode", help="The mode of knn matrix constructed from graph ('connectivity' or 'distance')", default="distance")
    # whether or not to apply NMI / RI metrics on the final result
    parser.add_argument("--eval", help="whether or not to evaluate the result (calculate density / NMI / RI)", type=int, default=1)

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
        if config["vis"] == "t-SNE":
            DeepWalk_TSNE_test(config)
        elif config["vis"] == "t-SGNE":
            DeepWalk_TSGNE_test(config)
        elif config["vis"] == "PCA":
            DeepWalk_PCA_test(config)

    elif config["embed"] == "Node2Vec":
        if config["vis"] == "t-SNE":
            Node2Vec_TSNE_test(config)
        elif config["vis"] == "t-SGNE":
            Node2Vec_TSGNE_test(config)
        elif config["vis"] == "PCA":
            Node2Vec_PCA_test(config)
    
    elif config["embed"] == "SDNE":
        if config["vis"] == "t-SNE":
            SDNE_TSNE_test(config)
        elif config["vis"] == "t-SGNE":
            SDNE_TSGNE_test(config)
        elif config["vis"] == "PCA":
            SDNE_PCA_test(config)
    
    elif config["embed"] == "ShortestPath":
        if config["vis"] == "t-SNE":
            ShortestPath_TSNE_test(config)
        elif config["vis"] == "t-SGNE":
            ShortestPath_TSGNE_test(config)
        elif config["vis"] == "PCA":
            ShortestPath_PCA_test(config)
    
    elif config["embed"] == "LEE":
        if config["vis"] == "t-SNE":
            LEE_TSNE_test(config)
        elif config["vis"] == "t-SGNE":
            LEE_TSGNE_test(config)
        elif config["vis"] == "PCA":
            LEE_PCA_test(config)
    
    elif config["embed"] == "GLEE":
        if config["vis"] == "t-SNE":
            GLEE_TSNE_test(config)
        elif config["vis"] == "t-SGNE":
            GLEE_TSGNE_test(config)
        elif config["vis"] == "PCA":
            GLEE_PCA_test(config)

    elif config["embed"] == "SPLEE":
        if config["vis"] == "t-SNE":
            SPLEE_TSNE_test(config)
        elif config["vis"] == "t-SGNE":
            SPLEE_TSGNE_test(config)
        elif config["vis"] == "PCA":
            SPLEE_PCA_test(config)
    
    elif config["embed"] == "RandomEmbed":
        if config["vis"] == "t-SGNE":
            RandomEmbed_TSGNE_test(config)
        else:
            NotImplementedError("Random embedding + other visualization is not implemented yet")

    ### A A A A A A A A A A A A A A A A A A A A A A A A A A A  ###
    ### ========== ========== ========== ========== ========== ###
    ###   END OF CALLING EMBEDDING AND VISUALIZING METHODS     ###
    ### ========== ========== ========== ========== ========== ###
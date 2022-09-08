import subprocess
import sys
import os
from tests.utils import print_block


def exhaustive_experiment(data):
    print_block("Running exhaustive_experiment")

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

    output_file = "exhaustive_experiment.log"

    cmd = "python main.py --data {} --embed {} --vis {} >> log\{}"

    if not os.path.exists("log"):
        os.mkdir("log")

    for vis in VIS_METHODS:
        for embed in EMBED_METHODS:  
            print(cmd.format(data, embed, vis, output_file))     
            subprocess.call(
                cmd.format(data, embed, vis, output_file), 
                stderr=sys.stdout.fileno(),
                shell=True)

def refined_experiment(data):   
    print_block("Running refined_experiment")

    EMBED_METHODS = ["deepwalk", "shortestpath", "splee"]
    VIS_METHODS = ["t-sne", "t-sgne"]
    output_file = "refined_experiment.log"

    cmd = "python main.py --data {} --embed {} --vis {} >> log\{}"

    if not os.path.exists("log"):
        os.mkdir("log")

    for vis in VIS_METHODS:
        for embed in EMBED_METHODS:  
            print(cmd.format(data, embed, vis, output_file))
            subprocess.call(
                cmd.format(data, embed, vis, output_file), 
                stderr=sys.stdout.fileno(),
                shell=True)


if __name__ == "__main__":
    data = "astro"
    refined_experiment(data)
    # exhaustive_experiment(data)
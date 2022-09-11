import subprocess
import sys
import os
from tests.utils import print_block

def exp_and_log(EMBED_METHODS, VIS_METHODS, data, output_file):
    # write stdout and stderr both to log file and to screen
    cmd = "(python main.py --data {} --embed {} --vis {} --eval 0 >> {}) 2>&1 | tee {}"

    if not os.path.exists("log"):
        os.mkdir("log")

    output_file = os.path.join('log', output_file)

    for embed in EMBED_METHODS:
        for vis in VIS_METHODS:

            fcmd = cmd.format(data, embed, vis, output_file, output_file)   
            print(fcmd)     
            subprocess.call(
                fcmd, 
                stderr=sys.stdout.fileno(),
                shell=True)

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

    exp_and_log(EMBED_METHODS, VIS_METHODS, data, output_file)
    
def refined_experiment(data):   
    print_block("Running refined_experiment")

    EMBED_METHODS = ["deepwalk", "shortestpath", "splee"]
    VIS_METHODS = ["t-sne", "t-sgne"]
    output_file = "refined_experiment.log"

    exp_and_log(EMBED_METHODS, VIS_METHODS, data, output_file)

def exp02(datasets):
    def exp02_per(data):
        EMBED_METHODS = ["shortestpath"]
        VIS_METHODS = ["t-sgne"]
        output_file = "experiment_02.log"

        exp_and_log(EMBED_METHODS, VIS_METHODS, data, output_file)

    print_block("Running Experiment 02:")

    for data in datasets:
        exp02_per(data) 


if __name__ == "__main__":
    # datasets = [
    #     "lfr_30000_0.18",
    #     "lfr_300000_0.18",
    #    "lfr_3000000_0.18"
    #    ]
    datasets = [
        "lock",
        "lock",
        "lock"
    ]
    
    exp02(datasets)
    

    # data = "lastfm"
    # refined_experiment(data)
    # exhaustive_experiment(data)


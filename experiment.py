import subprocess
import sys
import os
from tests.utils import print_block

def exp_and_log(EMBED_METHODS, VIS_METHODS, data, output_file):
    # write stdout and stderr both to log file and to screen
    cmd = "python -u main.py --data {} --embed {} --vis {} --eval 0 | tee -a {}"

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


def exp02(data):
    """ comparison between t-SNE and t-SGNE
    Compare t-SNE and t-SGNE for
    -   synthetic datasets (lfr) of increasing size
    -   increasing size of high dimensional embedding

    Due to neighbor graph construction, t-SNE is $O(d|V|^2)$ where $d$ is the dimension of the embedding. $d$ is
    involved because we need to compute the Euclidean between two embeddings.

    t-SGNE, on the other hand, is $O(K|V|)$. This is contributed by the $K$-step BFS for all nodes.

    """
    EMBED_METHODS = ["shortestpath"]
    VIS_METHODS = ["t-sne", "t-sgne"]
    output_file = "experiment_03.log"

    exp_and_log(EMBED_METHODS, VIS_METHODS, data, output_file)


def exp03(datasets):
    def exp03_per(data):
        EMBED_METHODS = ["shortestpath"]
        VIS_METHODS = ["t-sgne"]
        output_file = "experiment_03.log"

        exp_and_log(EMBED_METHODS, VIS_METHODS, data, output_file)

    print_block("Running Experiment 03:")

    for data in datasets:
        exp03_per(data) 



    

def toyexp():
    exp_and_log(["shortestpath"], ["t-sgne"], "lock", "toyexp.log")

if __name__ == "__main__":
    # datasets = [                # Nodes / Edges
    # synthetic datasests
        # "lfr_30000_0.18",       # Nodes 30,000 / Edges 75,643
        # "lfr_300000_0.18",      # Nodes 300,000 / Edges 656,664
        # "lfr_3000000_0.18",     # Nodes 3,000,000 / Edges 4,937,941

    # # real-world datasets
    #     "twitch_gamers",        # Nodes 168,114 / Edges 6,797,557
    #     "dblp",                 # Nodes	317,080 / Edges 1,049,866
    #     "youtube_community",    # Nodes	1,134,890 / Edges 2,987,624
    #     "livejournal",          # Nodes	3,997,962 / Edges 34,681,189
    #    ]

    # datasets = [
    #     "lock",
    #     "lock",
    #     "lock"
    # ]
    
    # exp03(datasets)

    dataset = "lfr_30000_0.18"
    exp02(dataset)

    # toyexp()
    
    # data = "lastfm"
    # refined_experiment(data)
    # exhaustive_experiment(data)


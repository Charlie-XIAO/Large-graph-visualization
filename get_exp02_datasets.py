from dataset_generator.lfr_generator import *
import os
import requests
import gzip

seed = 20220810

# artificial datasets
for n in [30000, 300000, 3000000]:
    mu = 0.18

    print(f"Generating lfr_{n}_{mu}")

    G = generate_lfr_graph(
        n=n, 
        mu=mu, 
        seed=seed,
    )
    G = nx.Graph(G)
    print(G)

    if not os.path.exists(f"./datasets/lfr_{n}_{mu}"):
        os.mkdir(f"./datasets/lfr_{n}_{mu}")
    save_graph_edgelist(G, f"./datasets/lfr_{n}_{mu}/lfr_{n}_{mu}_edgelist.txt")
    save_node_property(G, f"./datasets/lfr_{n}_{mu}/lfr_{n}_{mu}_labels.txt")


# real-world dataset
dsname2url = {
    # Youtube Communities: Node: 1134890, Edge: 2987624
    "youtube_communities": "http://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz",    



}
for dsname in dsname2url:
    if dsname in ["youtube_communities"]:
        pass
    elif dsname in []:
        pass
    else:
        raise ValueError(f"dataset:{dsname} is not the dataset for experiment 02")

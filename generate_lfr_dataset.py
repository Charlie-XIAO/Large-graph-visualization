from dataset_generator.lfr_generator import *
import os

seed = 20220810
n = 300000
mu = 0.18

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
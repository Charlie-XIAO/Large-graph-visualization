from dataset_generator.lfr_generator import *

seed = 20220810
n = 3000
mu = 0.15

G = generate_lfr_graph(
    n=n, 
    mu=mu, 
    seed=seed,
)
G = nx.Graph(G)
print(G)
save_graph_edgelist(G, f"lfr_{n}_{mu}_edgelist.txt")
save_node_property(G, f"lfr_{n}_{mu}_labels.txt")
from tkinter import N
import networkx as nx
from networkx import LFR_benchmark_graph
import numpy as np
from networkx.generators.community import LFR_benchmark_graph as lfr


def generate_lfr_graph(n=1000, mu=0.6, seed=10):
    # .\benchmark.exe -N 1000 -k 15 -maxk 20 -mu 0.1 -minc 20 -maxc 30
    # n = np.random.randint(n_low, n_high)
    tau1 = 4
    tau2 = 4
    # mu = np.random.uniform(0.03, 0.75)
    average_degree = 6
    max_degree = 20
    min_community = 100
    max_community = 500
    
    G = lfr(
        n,
        tau1,
        tau2,
        mu,
        average_degree=average_degree,
        min_community=min_community,
        max_community=max_community,
        max_degree=max_degree,
        seed=seed,
    )
    return G


def save_graph_edgelist(G, filename):
    with open(filename, "w") as file:
        #file.write("source target\n")
        for edge in G.edges():
            file.write(str(edge[0]) + " " + str(edge[1]) + "\n")
    print("Saving Completed")


def save_node_property(G, filename):
    # save it to a file

    labels = np.zeros(len(G), dtype=np.int32)
    communities = nx.get_node_attributes(G, "community")
    record = []
    count = 0

    for node in G:
        community = list(communities[node])
        # sign = communities[0]       # seelct first element of the community
        # if sign not in record:
        if community not in record:
            print(f"community #{count} detected, size = {len(community)}")
            record.append(community)
            labels[community] = count
            count += 1

    with open(filename, "w") as f:
        #f.write("node community\n")
        for node in G.nodes():
            f.write(str(node) + " " + str(labels[node]) + "\n")
    print("Saving Completed")
    

if __name__ == "__main__":

    seed = 20220810
    n = 3000
    mu = 0.3

    G = generate_lfr_graph(
        n=n, 
        mu=mu, 
        seed=seed,
    )
    G = nx.Graph(G)
    print(G)
    save_graph_edgelist(G, f"lfr_{n}_{mu}_edgelist.txt")
    save_node_property(G, f"lfr_{n}_{mu}_labels.txt")
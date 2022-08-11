import numpy as np
import networkx as nx
import heapq
import math

### ========== ========== ========= ========== ========== ###
### CLASS SHORTEST PATH ###
### ========== ========== ========= ========== ========== ###
class ShortestPath:

    def __init__(self, graph):
        """
        :param self:
        :param graph:
        :return: None
        """
        self.graph = graph
        self._embeddings = {}
    
    def get_embeddings(self, embed_size=128, sampling="random"):
        """
        :param self:
        :param embed_size DEFAULT 128:
        :param sampling DEFAULT "random": sampling method, can be "random", "hd" (highest degree), "ld" (lowest degree)
        :return: embeddings
        """
        edge_count = self.graph.number_of_edges()
        node_count=self.graph.number_of_nodes()
        #embed_size=node_count//int((10*(edge_count/node_count)**(1/2)))
        #embed_size=node_count//10
        embed_size=min(int(node_count*(edge_count/node_count)**(1/2))//10, 500)

        if sampling == "random":
            # Random node sampling
            X = np.random.choice(list(self.graph.nodes()), size=embed_size)
        elif sampling == "hd":
            # Select nodes of highest degrees as base nodes
            # Method: using priority queue STL library, runs O(nlogn) time and takes O(1) space
            pq = []
            heapq.heapify(pq)
            degrees = [(d[1], d[0]) for d in self.graph.degree]  # (degree, node)
            for i in range(self.graph.number_of_nodes()):
                heapq.heappush(pq, degrees[i])
                if len(pq) > embed_size:
                    heapq.heappop(pq)
            X = np.array([heapq.heappop(pq)[1] for _ in range(embed_size)])
        elif sampling == "ld":
            # Select nodes of lowest degrees as base nodes
            # Method: using priority queue STL library, runs O(nlogn) time and takes O(1) space
            pq = []
            heapq.heapify(pq)
            degrees = [(-d[1], d[0]) for d in self.graph.degree]  # (-degree, node)
            for i in range(self.graph.number_of_nodes()):
                heapq.heappush(pq, degrees[i])
                if len(pq) > embed_size:
                    heapq.heappop(pq)
            X = np.array([heapq.heappop(pq)[1] for _ in range(embed_size)])
        else:
            print("Unknown sampling method, switched to random sampling.")

        # Generate node embeddings
        self._embeddings = {}
        edge_count = self.graph.number_of_edges()
        node_count=self.graph.number_of_nodes()
        threshold=edge_count**(1/2)
        for source in self.graph.nodes():
            position = []
            for target in X:
                try:
                    l=nx.shortest_path_length(self.graph, source=source, target=target)
                    if l<=threshold:
                        position.append(l)
                    else:
                        position.append(threshold+1)
                except:
                    position.append((threshold)+2)
            self._embeddings[source] = position
        return self._embeddings
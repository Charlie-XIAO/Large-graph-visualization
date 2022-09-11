import numpy as np
import networkx as nx
import heapq
from embedders.shortest_paths.unweighted import multiple_to_all_shortest_path_length

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
        #embed_size=min(int(node_count*(edge_count/node_count)**(1/2))//10, 500)

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
            X = np.random.choice(list(self.graph.nodes()), size=embed_size)
        
        
        self._embeddings={}
        threshold = int(edge_count**(1/2))
        S = X.astype(int)
        max_node_num = max(np.max(S), len(self.graph))
        # space complexity of embedding_np = O( len(G) * embed_size ), probably no need for sparse matrix
        _embeddings_np = np.ones((max_node_num, embed_size), dtype=np.int32) * (threshold + 2) * embed_size

        # Generate node embeddings
        if len(X) < 1000:
            vis_step = 10
        elif len(X) >= 1000 and len(X) < 100000:
            vis_step = 100
        else:
            vis_step = 1000

        target_index=0

        node2idx = {int(node): i for i, node in enumerate(self.graph.nodes())}
        S = np.array([node2idx[int(x)] for x in X])     

        # it doesn't matter which node in X are we computing, because X = [3, 4, 5] and X = [4, 3, 5] should generate equivalent embeddings
        for _, sp_len_dict in multiple_to_all_shortest_path_length(
            # S = X.astype(np.int32), 
            # G = nx.convert_node_labels_to_integers(self.graph)):
            S = S, 
            G = nx.convert_node_labels_to_integers(self.graph)):
            # [Pitfall] by calling nx.convert_node_labels_to_integers on a graph of nodes ['1', '3', '4', '5', '7'],
            # the resulting nodes are [0, 1, 2, 3, 4].
            
            neighbors, lengths =  np.array(sp_len_dict.keys()), np.array(sp_len_dict.values())
            # _embeddings_np[neighbors, target_index] = lengths

            if target_index % vis_step == 0:
                print("[ShortestPath] {}/{} embeddings calculated".format(target_index, len(X)))
            target_index += 1
        
        _embeddings_np[_embeddings_np > threshold] = threshold

        print("[ShortestPath] {}/{} embeddings calculated, finished!".format(len(X), len(X)))

        for n in self.graph:
            self._embeddings[n] = _embeddings_np[node2idx[int(n)], :]

        return self._embeddings
        

if __name__ == "__main__":
    # simple shortestpath of networkx demonstration
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5), (4, 6), (5, 6)])
    sp_iter = nx.multiple_to_all_shortest_path_length([1,2], G)
    print(list(sp_iter))
    
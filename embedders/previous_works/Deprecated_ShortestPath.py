"""
This version of ShortestPath is deprecated due to the slow running speed. 
Please use the submodule `embedders.shortest_paths` instead.
"""



import numpy as np
import networkx as nx
import heapq
import math
import scipy.sparse

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

        # Generate node embeddings
        
        threshold=int(edge_count**(1/2))
        node2idx = {node: i for i, node in enumerate(self.graph.nodes())}
        # dists = np.zeros(shape=(node_count, node_count))
        dists = scipy.sparse.lil_matrix((node_count, node_count))
        self._embeddings={}
        for node in self.graph.nodes():
            self._embeddings[node] = [threshold+2]*embed_size
        target_index=0
        count = 0
        step = 100
        if len(X) < 1000:
            vis_step = 100
        elif len(X) >= 1000 and len(X) < 100000:
            vis_step = 1000
        else:
            vis_step = 10000

        for node in X:
            node_i = node2idx[node]
            queue = [node]
            visited = [0] * node_count
            visited[node_i] = 1
            while queue:
                curNode = queue.pop(0)
                curNode_i = node2idx[curNode]
                for neighbor in self.graph.neighbors(curNode):
                    neighbor_i = node2idx[neighbor]
                    if not visited[neighbor_i]:
                        visited[neighbor_i] = 1
                        queue.append(neighbor)
                        temp = dists[node_i, curNode_i] + 1
                        if temp < threshold:
                            dists[node_i, neighbor_i] = temp
                            self._embeddings[neighbor][target_index]=temp
                        else:
                            self._embeddings[neighbor][target_index]=threshold+1
                            queue = []
            target_index+=1
            count += 1
            if count % vis_step == 0:
                print("[ShortestPath] {}/{} embeddings calculated".format(count, len(X)))
        return self._embeddings
        
        '''
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
        '''
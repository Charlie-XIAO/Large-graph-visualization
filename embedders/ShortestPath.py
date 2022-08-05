import numpy as np
import networkx as nx

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
    
    def get_embeddings(self, embed_size=128):
        """
        :param self:
        :param embed_size DEFAULT 128:
        :return: embeddings
        """
        edge_count = self.graph.number_of_edges()
        node_count = self.graph.number_of_nodes()
        X = np.random.choice(list(self.graph.nodes()), size=embed_size)
        self._embeddings = {}
        for source in self.graph.nodes():
            position = []
            for target in X:
                try:
                    position.append(nx.shortest_path_length(self.graph, source=source, target=target))
                except:
                    position.append(edge_count)
            self._embeddings[source] = position
        return self._embeddings
import itertools
import math
import random

import pandas as pd
from joblib import Parallel, delayed

from embedders.utils import partition_num
from embedders.utils import chooseNeighbor

### CLASS BIASED WALKER ###
class BiasedWalker:

    def __init__(self, idx2node, temp_path):
        """
        :param self:
        :param idx2node:
        :param temp_path:
        :return: None
        """
        self.idx2node = idx2node
        self.idx = list(range(len(self.idx2node)))
        self.temp_path = temp_path
    
    def simulate_walks(self, num_walks, walk_length, stay_prob=0.3, workers=1, verbose=0):
        """
        :param self:
        :param num_walks:
        :param walk_length:
        :param stay_prob:
        :param workers:
        :param verbose:
        :return: walk
        """
        layers_adj = pd.read_pickle(self.temp_path + "layers_adj.pkl")
        layers_alias = pd.read_pickle(self.temp_path + "layers_alias.pkl")
        layers_accept = pd.read_pickle(self.temp_path + "layers_accept.pkl")
        gamma = pd.read_pickle(self.temp_path + "gamma.pkl")
        results = Parallel(n_jobs=workers, verbose=verbose)(
            delayed(self._simulate_walks)(self.idx, num, walk_length, stay_prob, layers_adj, layers_accept, layers_alias, gamma) for num in partition_num(num_walks, workers))
        return list(itertools.chain(*results))
    
    def _simulate_walks(self, nodes, num_walks, walk_length, stay_prob, layers_adj, layers_accept, layers_alias, gamma):
        """
        :param self:
        :param nodes:
        :param num_walks:
        :param walk_length:
        :param stay_prob:
        :param layers_adj:
        :param layers_accept:
        :param layers_alias:
        :param gamma:
        :return: walks
        """
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self._exec_random_walk(layers_adj, layers_accept, layers_alias, v, walk_length, gamma, stay_prob))
        return walks
    
    def _exec_random_walk(self, graphs, layers_accept, layers_alias, v, walk_length, gamma, stay_prob=0.3):
        """
        :param self:
        :param graphs:
        :param layers_accept:
        :param layers_alias:
        :param v:
        :param walk_length:
        :param gamma:
        :param stay_prob:
        :return: path
        """
        initialLayer, layer = 0, 0
        path = [self.idx2node[v]]
        while len(path) < walk_length:
            r = random.random()
            if r < stay_prob:
                v = chooseNeighbor(v, graphs, layers_alias, layers_accept, layer)
                path.append(self.idx2node[v])
            else:
                r = random.random()
                try:
                    x = math.log(gamma[layer][v] + math.e)
                    p_moveup = x / (x + 1)
                except:
                    print(layer, v)
                    raise ValueError()
                if r > p_moveup:
                    if layer > initialLayer:
                        layer -= 1
                elif (layer + 1) in graphs and v in graphs[layer + 1]:
                    layer += 1
        return path
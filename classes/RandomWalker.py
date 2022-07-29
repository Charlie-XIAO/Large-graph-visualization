import itertools
import random

from joblib import Parallel, delayed

from classes.utils import create_alias_table
from classes.utils import partition_num
from classes.utils import alias_sample

### CLASS RANDOM WALKER ###
class RandomWalker:
    
    def __init__(self, G, p=1, q=1, use_rejection_sampling=False):
        """
        :param self:
        :param G:
        :param p: return parameter, controls the likelihood of immediately revisiting a node in the walk
        :param q: in-out parameter, allows the search to differentiate between inward and outward nodes
        :param use_rejection_sampling: whether to use the rejection sampling strategy in node2vec
        :return: None
        """
        self.G = G
        self.p = p
        self.q = q
        self.use_rejection_sampling = use_rejection_sampling
    
    def deepwalk_walk(self, walk_length, start_node):
        """
        :param self:
        :param walk_length:
        :param start_node:
        :return: walk
        """
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def node2vec_walk(self, walk_length, start_node):
        """
        :param self:
        :param walk_length:
        :param start_node:
        :return: walk
        """
        alias_nodes, alias_edges = self.alias_nodes, self.alias_edges
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    edge = (walk[-2], cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0], alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break
        return walk

    """ Reference:
    KnightKing: A Fast Distributed Graph Random Walk Engine
    http://madsys.cs.tsinghua.edu.cn/publications/SOSP19-yang.pdf """
    def node2vec_walk2(self, walk_length, start_node):
        """
        :param self:
        :param walk_length:
        :param start_node:
        :return: walk
        """
        def rejection_sample(inv_p, inv_q, nbrs_num):
            upper_bound = max(1.0, max(inv_p, inv_q))
            lower_bound = min(1.0, min(inv_p, inv_q))
            shatter = 0
            second_upper_bound = max(1.0, inv_q)
            if (inv_p > second_upper_bound):
                shatter = second_upper_bound / nbrs_num
                upper_bound = second_upper_bound + shatter
            return upper_bound, lower_bound, shatter

        G = self.G
        alias_nodes = self.alias_nodes
        inv_p = 1.0 / self.p
        inv_q = 1.0 / self.q
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    upper_bound, lower_bound, shatter = rejection_sample(
                        inv_p, inv_q, len(cur_nbrs))
                    prev = walk[-2]
                    prev_nbrs = set(G.neighbors(prev))
                    while True:
                        prob = random.random() * upper_bound
                        if (prob + shatter >= upper_bound):
                            next_node = prev
                            break
                        next_node = cur_nbrs[alias_sample(
                            alias_nodes[cur][0], alias_nodes[cur][1])]
                        if (prob < lower_bound):
                            break
                        if (prob < inv_p and next_node == prev):
                            break
                        _prob = 1.0 if next_node in prev_nbrs else inv_q
                        if (prob < _prob):
                            break
                    walk.append(next_node)
            else:
                break
        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):
        """
        :param self:
        :param num_walks:
        :param walk_length:
        :param workers:
        :param verbose:
        :return: walks"""
        nodes = list(self.G.nodes())
        results = Parallel(n_jobs=workers, verbose=verbose)(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in partition_num(num_walks, workers))
        return list(itertools.chain(*results))

    def _simulate_walks(self, nodes, num_walks, walk_length):
        """
        :param self:
        :param nodes:
        :param num_walks:
        :param walk_length:
        :return: walks"""
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                if self.p == 1 and self.q == 1:
                    walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=v))
                elif self.use_rejection_sampling:
                    walks.append(self.node2vec_walk2(walk_length=walk_length, start_node=v))
                else:
                    walks.append(self.node2vec_walk(walk_length=walk_length, start_node=v))
        return walks

    def get_alias_edge(self, t, v):
        """
        Compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
        :param self:
        :param t:
        :param v:
        :return: accept, alias
        """
        unnormalized_probs = []
        for x in self.G.neighbors(v):
            weight = self.G[v][x].get("weight", 1.0)
            if x == t:
                unnormalized_probs.append(weight / self.p)
            elif self.G.has_edge(x, t):
                unnormalized_probs.append(weight)
            else:
                unnormalized_probs.append(weight / self.q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        return create_alias_table(normalized_probs)
    
    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        :param self:
        :return: None
        """
        alias_nodes = {}
        for node in self.G.nodes():
            unnormalized_probs = [self.G[node][nbr].get("weight", 1.0) for nbr in self.G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)
        if not self.use_rejection_sampling:
            alias_edges = {}
            for edge in self.G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                if not self.G.is_directed():
                    alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
                self.alias_edges = alias_edges
        self.alias_nodes = alias_nodes
        return
from abc import ABC, abstractmethod
from embedding_tests.utils import *

import networkx as nx

class AbstractEmbedTest(ABC):

    def __init__(self, edgeset, featureset=None):
        self.edgeset = edgeset
        self.graph = self.readGraph()
        self.featureset = featureset
        self.embeddings = None
        self.has_feature = False
        print(self.graph)
        print()

    def readGraph(self):
        print("Reading graph data", end="... ")
        G = nx.read_edgelist(self.edgeset, create_using=nx.Graph(), nodetype=None, data=[("weight", int)])
        print("Done.")
        return G

    @abstractmethod
    def getEmbeddings(self):
        """
        :param self:
        :return: None
        
        Set self.embeddings to the high-dimensional embeddings with one node per row.
        Need to get transpose of the default embeddings as: embeddings.T
        """
        pass

    def addFeature(self):
        if not self.featureset:
            return
        features = {}
        print("Reading features data", end="... ")
        try:
            with open(self.featureset) as f:
                for line in f:
                    k, v = line.split()
                    features[k] = v
        except Exception as e:
            print("Failed: {}".format(e))
        if features:
            self.embeddings["feature"] = [features[node] for node in self.embeddings.index]
            self.has_feature = True
            print("Done.")
    
    def embed(self):
        self.getEmbeddings()
        self.addFeature()
        print("Embeddings generated as follows:")
        print(self.embeddings)
        print()
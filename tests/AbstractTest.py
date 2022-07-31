from abc import ABC, abstractmethod

import networkx as nx

class AbstractTest(ABC):

    def __init__(self, edgeset, featureset=None):
        self.edgeset = edgeset
        self.graph = self.readGraph()
        self.featureset = featureset
        self.embeddings = None
        self.has_feature = False
        self.addFeature()

    def readGraph(self):
        print("Reading graph data", end="... ")
        G = nx.read_edgelist(self.edgeset, create_using=nx.Graph(), nodetype=None, data=[("weight", int)])
        print("Done.")
        return G

    @abstractmethod
    def getEmbeddings(self):
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
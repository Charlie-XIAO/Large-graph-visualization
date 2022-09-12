from gensim.models import Word2Vec

from embedders.previous_works.RandomWalker import RandomWalker

### ========== ========== ========= ========== ========== ###
### CLASS DEEP WALK ###
### ========== ========== ========= ========== ========== ###
class DeepWalk:

    def __init__(self, graph, walk_length, num_walks, workers=1):
        """
        :param self:
        :param graph:
        :param walk_length:
        :param num_walks:
        :param workers:
        :return: None
        """
        self.graph = graph
        self.w2v_nodel = None
        self._embeddings = {}
        self.walker = RandomWalker(graph, p=1, q=1)
        self.sentences = self.walker.simulate_walks(num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=2)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        """
        :param self:
        :param embed_size DEFAULT 128:
        :param window_size DEFAULT 5:
        :param workers DEFAULT 3:
        :param iter DEFAULT 5:
        :param **kwargs:
        :return: model
        """
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter
        print("Learning embedding vectors", end="... ")
        model = Word2Vec(**kwargs)
        print("Done.")
        self.w2v_nodel = model
        return model
    
    def get_embeddings(self):
        """
        :param self:
        :return: embedding
        """
        if self.w2v_nodel is None:
            print("Error: model not trained.")
            return {}
        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_nodel.wv[word]
        return self._embeddings
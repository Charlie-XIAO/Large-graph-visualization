from gensim.models import Word2Vec

from embedders.previous_works.RandomWalker import RandomWalker

### ========== ========== ========= ========== ========== ###
### CLASS NODE TO VEC ###
### ========== ========== ========= ========== ========== ###
class Node2Vec:

    def __init__(self, graph, walk_length, num_walks, p=1, q=1, workers=1, use_rejection_sampling=False):
        """
        :param self:
        :param graph:
        :param walk_length:
        :param num_walks:
        :param p:
        :param q:
        :param workers:
        :param use_rejection_sampling:
        :return: None
        """
        self.graph = graph
        self._embeddings = {}
        self.walker = RandomWalker(graph, p=p, q=q, use_rejection_sampling=use_rejection_sampling)
        print("Preprocessing transition probabilities...")
        self.walker.preprocess_transition_probs()
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
        kwargs["hs"] = 0  # node2vec does not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter
        print("Learning embedding vectors", end="... ")
        model = Word2Vec(**kwargs)
        print("Done.")
        self.w2v_model = model
        return model
    
    def get_embeddings(self):
        """
        :param self:
        :return: embedding
        """
        if self.w2v_model is None:
            print("Error: model not trained.")
            return {}
        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]
        return self._embeddings
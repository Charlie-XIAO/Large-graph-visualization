# Instructions

To be updated...

# Graph Embedding

## Method

|   Model   | Paper                                                                                                                      | Note                                                                                        |
| :-------: | :------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------ |
| DeepWalk  | [KDD 2014] [DeepWalk: Online Learning of Social Representations](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)   | [【Graph Embedding】DeepWalk：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56380812)  |
|   LINE    | [WWW 2015] [LINE: Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf)                          | [【Graph Embedding】LINE：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56478167)      |
| Node2Vec  | [KDD 2016] [node2vec: Scalable Feature Learning for Networks](https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf) | [【Graph Embedding】Node2Vec：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56542707)  |
|   SDNE    | [KDD 2016] [Structural Deep Network Embedding](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf)               | [【Graph Embedding】SDNE：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56637181)      |
| Struc2Vec | [KDD 2017] [struc2vec: Learning Node Representations from Structural Identity](https://arxiv.org/pdf/1704.03165.pdf)        | [【Graph Embedding】Struc2Vec：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56733145) |

## DeepWalk

```python
model = DeepWalk(self.graph, walk_length=10, num_walks=80, workers=1)
model.train(window_size=5, iter=3)
embeddings = pd.DataFrame.from_dict(model.get_embeddings())
self.embeddings = embeddings.T
```

## LINE

```python
model = LINE(self.graph, embedding_size=128, order="second")  # Order can be "first", "second", or "all"
model.train(batch_size=1024, epochs=50, verbose=2)
embeddings = pd.DataFrame.from_dict(model.get_embeddings())
self.embeddings = embeddings.T
```

## Node2Vec

```python
model = Node2Vec(self.graph, walk_length=10, num_walks=80, p=0.25, q=4, workers=1)
model.train(window_size=5, iter=3)
embeddings = pd.DataFrame.from_dict(model.get_embeddings())
self.embeddings = embeddings.T
```

## SDNE

```python
model = SDNE(self.graph, hidden_size=[256, 128])
model.train(batch_size=3000, epochs=40, verbose=2)
embeddings = pd.DataFrame.from_dict(model.get_embeddings())
self.embeddings = embeddings.T
```

# Graph Visualization

## Method

To be updated...

## TSNE

```python
model = TSNE(n_components=2, verbose=1, random_state=0)
self.projections = model.fit_transform(self.X)
```
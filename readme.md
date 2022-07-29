# GraphEmbedding

## Method

|   Model   | Paper                                                                                                                      | Note                                                                                        |
| :-------: | :------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------ |
| DeepWalk  | [KDD 2014] [DeepWalk: Online Learning of Social Representations](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)   | [【Graph Embedding】DeepWalk：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56380812)  |
|   LINE    | [WWW 2015] [LINE: Large-scale Information Network Embedding](https://arxiv.org/pdf/1503.03578.pdf)                          | [【Graph Embedding】LINE：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56478167)      |
| Node2Vec  | [KDD 2016] [node2vec: Scalable Feature Learning for Networks](https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf) | [【Graph Embedding】Node2Vec：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56542707)  |
|   SDNE    | [KDD 2016] [Structural Deep Network Embedding](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf)               | [【Graph Embedding】SDNE：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56637181)      |
| Struc2Vec | [KDD 2017] [struc2vec: Learning Node Representations from Structural Identity](https://arxiv.org/pdf/1704.03165.pdf)        | [【Graph Embedding】Struc2Vec：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56733145) |

## Instructions

```python
test_XXX("dataset_name", "description", type, delimiter=" ",
        by_categories=True, by_labels=True, by_clusters=True, raw=True)
```

`dataset_name` 数据集名称；

`description` 为对输出类型的描述，包括使用的embedding方法、图类型等；

`type` 为图类型，可选用 `nx.Graph()` 或 `nx.DiGraph()`；

`delimiter` 为数据集中 node-node 的分隔符；

`by_categories` 设置为 `True` 则尝试以 categories 为标准给输出的 plot 上色，需要 `./datasets/"dataset_name"/"dataset_name"_categories.txt` 文件，数据为 node-(float)category，分隔符为空格；

`by_labels` 设置为 `True` 则尝试以 labels 为标准给输出的 plot 上色，需要 `./datasets/"dataset_name"/"dataset_name"_labels.txt` 文件，数据为 node-(float)label，分隔符为空格；

`by_clusters` 设置为 `True` 则尝试以 clusters 为标准给输出的 plot 上色，cluster 判定使用 `nx.clustering` 方法；

`raw` 设置为 `True` 则输出未上色的散点 plot。

例如：

```python
test_DeepWalk("wiki", "DeepWalk_G", nx.Graph(), delimiter=" ",
            by_categories=True, by_labels=True, by_clusters=True, raw=True)
```

## DeepWalk

```python
# Read graph
G = nx.read_edgelist("filename", create_using=nx.Graph(), nodetype=None, data=[("weight", int)])
# Initialize DeepWalk model
model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
# Train DeepWalk model
model.train(window_size=5, iter=3)
# Get embedding vectors
embeddings = model.get_embeddings()
```

## LINE

```python
# Read graph
G = nx.read_edgelist("filename", create_using=nx.Graph(), nodetype=None, data=[("weight", int)])
# Initialize LINE model, order can be "first", "second", or "all"
model = LINE(G, embedding_size=128, order="second")
# Train LINE model
model.train(batch_size=1024, epochs=50, verbose=2)
# Get embedding vectors
embeddings = model.get_embeddings()
```

## Node2Vec

```python
# Read graph
G = nx.read_edgelist("filename", create_using=nx.Graph(), nodetype=None, data=[("weight", int)])
# Initialize Node2Vec model
model = Node2Vec(G, walk_length=10, num_walks=80, p=0.25, q=4, workers=1)
# Train Node2Vec model
model.train(window_size=5, iter=3)
# Get embedding vectors
embeddings = model.get_embeddings()
```

## SDNE

```python
# Read graph
G = nx.read_edgelist("filename", create_using=nx.Graph(), nodetype=None, data=[("weight", int)])
# Initialize SDNE model
model = SDNE(G, hidden_size=[256, 128])
# Train SDNE model
model.train(batch_size=3000, epochs=40, verbose=2)
# Get embedding vectors
embeddings = model.get_embeddings()
```

## Struc2Vec

```python
# Read graph
G = nx.read_edgelist("filename", create_using=nx.DiGraph(), nodetype=None, data=[("weight", int)])
# Initialize Struc2Vec model
model = Struc2Vec(G, 10, 80, workers=4, verbose=40)
# Train Struc2Vec model
model.train(window_size=5, iter=3)
# Get embedding vectors
embeddings = model.get_embeddings()
```
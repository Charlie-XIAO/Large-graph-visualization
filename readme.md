# Graph Embedding and Visualization

This is a repository for graph embedding and visualization. Various different graph embedding methods and dimension reduction methods are combined to produce 2D layouts for graph data.

## Instructions

0. Install Python 3.9 (other unprescribed versions of Python may work, but are not tested).

1. Clone the repository. Use `Clone Git Repository...` tab in an empty window of VSCode or use the following command line in Command Prompt:

```
git clone https://github.com/Charlie-XIAO/embedding-visualization-test.git
```

2. Set the Python virtual environment using the following command lines in Command Prompt:

```
python -m venv myvenv
(For Windows) myvenv\Scripts\activate
(For Mac/Linux) source myvenv/bin/activate
```

3. Install required packages in the Python virtual environment using the following command line in Command Prompt:

```
pip install -r requirements.txt
```

4. Run `main.py` using the following command line in Command Prompt:

```
python main.py --data <dataset_name> --embed <embedding_method> --vis <visualization_method>
(Example) python main.py --data wiki --embed deepwalk --vis t-sne
```

5. To run the program on large datasets mentioned in the experiment 2 of the essay, download zipped datasets from [this google drive link](https://drive.google.com/file/d/1n4sE6AfZZZU81IeqehnxYmQOHk7CqgAh/view?usp=sharing), and unzip the file in the `datasets` folder.

```

## Graph Embedding

### Previous Works

|   Method   | Paper                                                                                                                      | Note                                                                                        |
| :-------: | :------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------ |
| DeepWalk  | [KDD 2014] [DeepWalk: Online Learning of Social Representations](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)   | [【Graph Embedding】DeepWalk：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56380812)  |
| Node2Vec  | [KDD 2016] [Node2Vec: Scalable Feature Learning for Networks](https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf) | [【Graph Embedding】Node2Vec：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56542707)  |
|   LE     | [KDD 2001] [Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering](https://proceedings.neurips.cc/paper/2001/file/f106b7f99d2cb30c3db1c3cc0fde9ccb-Paper.pdf)        | [【Graph Embedding】LE（拉普拉斯映射）特征提取方法](https://zhuanlan.zhihu.com/p/100002630) |
|   GLEE     | [KDD 2019] [GLEE: Geometric Laplacian Eigenmap Embedding](https://arxiv.org/pdf/1905.09763.pdf)        |  |
|   SDNE    | [KDD 2016] [Structural Deep Network Embedding](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf)               | [【Graph Embedding】SDNE：算法原理，实现和应用](https://zhuanlan.zhihu.com/p/56637181)      |

#### DeepWalk

```python
model = DeepWalk(self.graph, walk_length=10, num_walks=80, workers=1)
model.train(embed_size=128, window_size=5, iter=3)
embeddings = pd.DataFrame.from_dict(model.get_embeddings())
self.embeddings = embeddings.T
```

#### Node2Vec

```python
model = Node2Vec(self.graph, walk_length=10, num_walks=80, p=0.25, q=4, workers=1)
model.train(embed_size=128, window_size=5, iter=3)
embeddings = pd.DataFrame.from_dict(model.get_embeddings())
self.embeddings = embeddings.T
```

#### LE

```python
model = LEE(self.graph)
embeddings = pd.DataFrame.from_dict(model.get_embeddings(embed_size=128, iter=100))
self.embeddings = embeddings.T
```

#### GLEE

```python
model = GLEE(self.graph)
embeddings = pd.DataFrame.from_dict(model.get_embeddings(embed_size=128, iter=100))
self.embeddings = embeddings.T
```

#### SDNE

```python
model = SDNE(self.graph, hidden_size=[256, 128])
model.train(batch_size=3000, epochs=40, verbose=2)
embeddings = pd.DataFrame.from_dict(model.get_embeddings())
self.embeddings = embeddings.T
```

### Our Contributions

#### SP

```python
model = ShortestPath(self.graph)
embeddings = pd.DataFrame.from_dict(model.get_embeddings(embed_size=128, sampling="random"))
self.embeddings = embeddings.T
```

#### SPLEE

```python
model = SPLEE(self.graph)
embeddings = pd.DataFrame.from_dict(model.get_embeddings(embed_size=128, iter=10, shape="gaussian", epsilon=6.0, threshold=5))
self.embeddings = embeddings.T
```

## Graph Visualization

### Previous Works

|   Method   | Paper                                                                                                                      | Note                                                                                        |
| :-------: | :------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------ |
|  PCA    | [WCS 2010] [Principal Component Analysis](https://wires.onlinelibrary.wiley.com/doi/epdf/10.1002/wics.101)   | [【Dimension Reduction】主成分分析（PCA）原理详解](https://zhuanlan.zhihu.com/p/37777074)  |
| t-SNE  | [KDD 2016] [Visualizing Data Using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf?fbcl) | [【Dimension Reduction】降维方法之 t-SNE](https://zhuanlan.zhihu.com/p/426068503)  |

#### PCA

```python
model = PCA(n_components=2, random_state=0)
self.projections = model.fit_transform(self.X)
```

#### t-SNE

```python
model = TSNE(n_components=2, verbose=1, random_state=0)
self.projections = model.fit_transform(self.X)
```
### Our Contributions

#### t-SGNE

```python
model = TSGNE(perplexity=30, n_components=2, verbose=1, random_state=0, knn_matrix=self.knn_matrix, mode="distance")
self.projections = model.fit_transform(self.X)
```

## Datasets

### Usage

In the `datasets` folder, create a folder with the name of the dataset. In this folder, put the edgelist file and the labels file (optional), and name them `<dataset_name>_edgelist.txt` and `<dataset_name>_labels.txt` respectively. The program automatically reads graph and label data from the correponding locations.

### Source

- [Index of Complex Networks (colorado.edu)](https://icon.colorado.edu/#!/networks)
  
- [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/)

- [Network Repository: An Interactive Scientific Network Data Repository](https://networkrepository.com/)
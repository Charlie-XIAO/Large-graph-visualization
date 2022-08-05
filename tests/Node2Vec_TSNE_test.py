from tests.utils import *

from embedding_tests.Node2VecTest import Node2VecTest
from visualizing_tests.TSNETest import TSNETest
import os

def Node2Vec_TSNE_test(config):

    print_block(f"Running {config['embedding']} + {config['visualization']} on {config['dataset']}")
    edgeset = config['edgeset']
    featureset = config['featureset']
    verbose = config['verbose']

    # specify some parameters, and determine the name of the image
    config["location"] = os.path.join(config["image_folder"], f"{config['dataset']}_{config['embedding']}_{config['visualization']}_{config['description']}.{config['image_format']}")
    location = config["location"]
      
    node2vec = Node2VecTest(edgeset, featureset=featureset, walk_length=10, num_walks=80, p=0.25, q=4, workers=1, window_size=5, iter=3)
    tsne = TSNETest(node2vec.embeddings, node2vec.has_feature, location, n_components=2, verbose=verbose, random_state=0)
    show_evaluation_results(config, node2vec, tsne)
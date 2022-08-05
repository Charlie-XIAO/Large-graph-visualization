from tests.utils import *

from embedding_tests.SDNETest import SDNETest
from visualizing_tests.TSNETest import TSNETest
import os

def SDNE_TSNE_test(config):

    print_block(f"Running {config['embedding']} + {config['visualization']} on {config['dataset']}")
    edgeset = config['edgeset']
    featureset = config['featureset']
    verbose = config['verbose']

    # specify some parameters, and determine the name of the image
    config["location"] = os.path.join(config["image_folder"], f"{config['dataset']}_{config['embedding']}_{config['visualization']}_{config['description']}.{config['image_format']}")
    location = config["location"]

    sdne = SDNETest(edgeset, featureset=featureset, hidden_size=[256, 128], batch_size=3000, epochs=40, verbose=2)
    tsne = TSNETest(sdne.embeddings, sdne.has_feature, location, n_components=2, verbose=verbose, random_state=0)
    show_evaluation_results(config, sdne, tsne)
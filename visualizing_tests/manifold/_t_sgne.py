from sklearn.manifold import TSNE
from sklearn.manifold._t_sne import _joint_probabilities, _joint_probabilities_nn

import warnings
from time import time
import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils.validation import check_non_negative
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

# mypy error: Module 'sklearn.manifold' has no attribute '_utils'
from sklearn.manifold import _utils  # type: ignore

# mypy error: Module 'sklearn.manifold' has no attribute '_barnes_hut_tsne'
from sklearn.manifold import _barnes_hut_tsne  # type: ignore


MACHINE_EPSILON = np.finfo(np.double).eps



class TSGNE(TSNE):

    def __init__(self, *args, **kwargs):
        if "knn_matrix" not in kwargs:
            raise ValueError("KNN matrix required for t-sgne")
        self.knn_matrix = kwargs.pop("knn_matrix")
        self.mode = kwargs.pop("mode", "connectivity")
        super().__init__(*args, **kwargs)
        
        
    
    def _fit(self, X, skip_num_points=0):
        """This is a modified version of TSNE._fit that takes in knn_matrix. Below is the origial docstring.
        Private function to fit the model using X as training data."""

        if isinstance(self.init, str) and self.init == "warn":
            # See issue #18018
            warnings.warn(
                "The default initialization in TSNE will change "
                "from 'random' to 'pca' in 1.2.",
                FutureWarning,
            )
            self._init = "random"
        else:
            self._init = self.init
        if self.learning_rate == "warn":
            # See issue #18018
            warnings.warn(
                "The default learning rate in TSNE will change "
                "from 200.0 to 'auto' in 1.2.",
                FutureWarning,
            )
            self._learning_rate = 200.0
        else:
            self._learning_rate = self.learning_rate

        if isinstance(self._init, str) and self._init == "pca" and issparse(X):
            raise TypeError(
                "PCA initialization is currently not supported "
                "with the sparse input matrix. Use "
                'init="random" instead.'
            )
        if self.method not in ["barnes_hut", "exact"]:
            raise ValueError("'method' must be 'barnes_hut' or 'exact'")
        if self.angle < 0.0 or self.angle > 1.0:
            raise ValueError("'angle' must be between 0.0 - 1.0")
        if self.square_distances != "deprecated":
            warnings.warn(
                "The parameter `square_distances` has not effect and will be "
                "removed in version 1.3.",
                FutureWarning,
            )
        if self._learning_rate == "auto":
            # See issue #18018
            self._learning_rate = X.shape[0] / self.early_exaggeration / 4
            self._learning_rate = np.maximum(self._learning_rate, 50)
        else:
            if not (self._learning_rate > 0):
                raise ValueError("'learning_rate' must be a positive number or 'auto'.")
        if self.method == "barnes_hut":
            X = self._validate_data(
                X,
                accept_sparse=["csr"],
                ensure_min_samples=2,
                dtype=[np.float32, np.float64],
            )
        else:
            X = self._validate_data(
                X, accept_sparse=["csr", "csc", "coo"], dtype=[np.float32, np.float64]
            )
        if self.metric == "precomputed":
            if isinstance(self._init, str) and self._init == "pca":
                raise ValueError(
                    'The parameter init="pca" cannot be used with metric="precomputed".'
                )
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix")

            check_non_negative(
                X,
                "TSNE.fit(). With metric='precomputed', X "
                "should contain positive distances.",
            )

            if self.method == "exact" and issparse(X):
                raise TypeError(
                    'TSNE with method="exact" does not accept sparse '
                    'precomputed distance matrix. Use method="barnes_hut" '
                    "or provide the dense distance matrix."
                )

        if self.method == "barnes_hut" and self.n_components > 3:
            raise ValueError(
                "'n_components' should be inferior to 4 for the "
                "barnes_hut algorithm as it relies on "
                "quad-tree or oct-tree."
            )
        random_state = check_random_state(self.random_state)

        if self.early_exaggeration < 1.0:
            raise ValueError(
                "early_exaggeration must be at least 1, but is {}".format(
                    self.early_exaggeration
                )
            )

        if self.n_iter < 250:
            raise ValueError("n_iter should be at least 250")

        n_samples = X.shape[0]

        neighbors_nn = None
        if self.method == "exact":
            # Retrieve the distance matrix, either using the precomputed one or
            # computing it.
            if self.metric == "precomputed":
                distances = X
            else:
                if self.verbose:
                    print("[t-sgne] Computing pairwise distances...")

                if self.metric == "euclidean":
                    # Euclidean is squared here, rather than using **= 2,
                    # because euclidean_distances already calculates
                    # squared distances, and returns np.sqrt(dist) for
                    # squared=False.
                    # Also, Euclidean is slower for n_jobs>1, so don't set here
                    distances = pairwise_distances(X, metric=self.metric, squared=True)
                else:
                    metric_params_ = self.metric_params or {}
                    distances = pairwise_distances(
                        X, metric=self.metric, n_jobs=self.n_jobs, **metric_params_
                    )

            if np.any(distances < 0):
                raise ValueError(
                    "All distances should be positive, the metric given is not correct"
                )

            if self.metric != "euclidean":
                distances **= 2

            # compute the joint probability distribution for the input space
            P = _joint_probabilities(distances, self.perplexity, self.verbose)
            assert np.all(np.isfinite(P)), "All probabilities should be finite"
            assert np.all(P >= 0), "All probabilities should be non-negative"
            assert np.all(
                P <= 1
            ), "All probabilities should be less or then equal to one"

        else:
            # Compute the number of nearest neighbors to find.
            # LvdM uses 3 * perplexity as the number of neighbors.
            # In the event that we have very small # of points
            # set the neighbors to n - 1.
            n_neighbors = min(n_samples - 1, int(3.0 * self.perplexity + 1))

            if self.verbose:
                print("[t-sgne] Computing {} nearest neighbors...".format(n_neighbors))

            # Find the nearest neighbors for every point
            # In t-SGNE, we use graph input to find knn



            # knn return the euclidean distance but we need it squared
            # to be consistent with the 'exact' method. Note that the
            # the method was derived using the euclidean method as in the
            # input space. Not sure of the implication of using a different
            # metric.
            # distances_nn.data **= 2

            print(f"[t-sgne] knn matrix is of size {self.knn_matrix.shape}. The first two rows and columns are: {self.knn_matrix[:2, :2].todense().tolist()}")

            # compute the joint probability distribution for the input space
            P = _joint_probabilities_nn(self.knn_matrix, self.perplexity, self.verbose)

        if isinstance(self._init, np.ndarray):
            X_embedded = self._init
        elif self._init == "pca":
            pca = PCA(
                n_components=self.n_components,
                svd_solver="randomized",
                random_state=random_state,
            )
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
            # TODO: Update in 1.2
            # PCA is rescaled so that PC1 has standard deviation 1e-4 which is
            # the default value for random initialization. See issue #18018.
            warnings.warn(
                "The PCA initialization in TSNE will change to "
                "have the standard deviation of PC1 equal to 1e-4 "
                "in 1.2. This will ensure better convergence.",
                FutureWarning,
            )
            # X_embedded = X_embedded / np.std(X_embedded[:, 0]) * 1e-4
        elif self._init == "random":
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            X_embedded = 1e-4 * random_state.standard_normal(
                size=(n_samples, self.n_components)
            ).astype(np.float32)
        else:
            raise ValueError("'init' must be 'pca', 'random', or a numpy array")

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1, 1)

        return self._tsne(
            P,
            degrees_of_freedom,
            n_samples,
            X_embedded=X_embedded,
            neighbors=neighbors_nn,
            skip_num_points=skip_num_points,
        )
    


    pass
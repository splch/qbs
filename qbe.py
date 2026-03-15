import numpy as np
from scipy.spatial import KDTree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class QuantileBalancedEnsemble(BaseEstimator, ClassifierMixin):
    """Ensemble classifier using QBS-inspired quantile-grid selection.

    For each base learner, samples points from the majority class's
    quantile grid and selects the nearest real samples — creating a
    balanced, distribution-preserving training subset. Different random
    seeds per learner ensure each sees a different representative slice
    of the majority class.

    Unlike single-pass QBS which discards most majority data, the
    ensemble collectively covers the full majority distribution while
    keeping each individual learner balanced.

    Parameters
    ----------
    n_estimators : int, default=50
        Number of base learners.
    q : int, default=10
        Number of quantile values per feature for grid construction.
    random_state : int or None, default=None
        Controls randomness of grid sampling and base learners.
    """

    def __init__(self, n_estimators=50, q=10, random_state=None):
        self.n_estimators = n_estimators
        self.q = q
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.sort(np.unique(y))
        n_classes = len(self.classes_)

        classes, counts = np.unique(y, return_counts=True)
        minority_label = classes[counts.argmin()]
        minority_mask = y == minority_label
        minority_X, minority_y = X[minority_mask], y[minority_mask]
        majority_X, majority_y = X[~minority_mask], y[~minority_mask]

        m = len(minority_y)
        f = X.shape[1]
        q = self.q

        # Precompute quantiles and KDTree (shared across all members)
        quantiles = np.percentile(majority_X, np.linspace(0, 100, q), axis=0)
        if quantiles.ndim == 1:
            quantiles = quantiles[:, np.newaxis]
        tree = KDTree(majority_X)

        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []

        # Grid oversample factor — query more grid points than target
        # to ensure good coverage after deduplication
        n_grid = m * 5

        for _ in range(self.n_estimators):
            # QBS-style: random grid points from quantile space
            idx = rng.integers(0, q, size=(n_grid, f))
            grid_points = quantiles[idx, np.arange(f)]

            # Find nearest real majority samples
            _, nn_idx = tree.query(grid_points, k=1, workers=-1)
            unique_idx, popularity = np.unique(nn_idx.ravel(), return_counts=True)

            # Keep the most representative samples (most grid points mapped to them)
            if len(unique_idx) > m:
                top = np.argsort(popularity)[-m:]
                selected = unique_idx[top]
            else:
                selected = unique_idx

            X_train = np.vstack((minority_X, majority_X[selected]))
            y_train = np.hstack((minority_y, majority_y[selected]))

            est = DecisionTreeClassifier(
                max_features='sqrt',
                random_state=int(rng.integers(0, 2**31)),
            )
            est.fit(X_train, y_train)
            self.estimators_.append(est)

        return self

    def predict_proba(self, X):
        check_is_fitted(self, 'estimators_')
        X = check_array(X)

        # Align probabilities to self.classes_ ordering
        avg = np.zeros((len(X), len(self.classes_)))
        for est in self.estimators_:
            p = est.predict_proba(X)
            for i, c in enumerate(est.classes_):
                j = np.searchsorted(self.classes_, c)
                avg[:, j] += p[:, i]
        avg /= self.n_estimators
        return avg

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

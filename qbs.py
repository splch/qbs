import numpy as np
from scipy.spatial import KDTree


def qbs(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantile-Based Balanced Sampling.

    Undersamples majority classes by selecting representatives nearest
    to quantile-derived grid points in feature space.
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return X.copy(), y.copy()

    minority_label = classes[counts.argmin()]
    m = counts.min()
    c = len(classes) - 1
    f = X.shape[1]
    target = c * m

    minority_mask = y == minority_label
    majority_X = X[~minority_mask]
    majority_y = y[~minority_mask]

    if target >= len(majority_X):
        return X.copy(), y.copy()

    # At least 5 quantiles for meaningful resolution — q=2 only gives
    # {min, max} per feature, which in high dimensions produces grid
    # points in empty corners of the bounding box.
    q = max(5, int(np.ceil(np.log(target) / np.log(f)))) if f > 1 else target

    quantiles = np.percentile(majority_X, np.linspace(0, 100, q), axis=0)
    if quantiles.ndim == 1:
        quantiles = quantiles[:, np.newaxis]

    # Enumerate when small; sample otherwise.
    log_grid = f * np.log(q) if q > 1 else 0
    if log_grid <= np.log(50_000):
        idx = np.indices((q,) * f).reshape(f, -1).T
    else:
        rng = np.random.default_rng(42)
        idx = rng.integers(0, q, size=(max(target * 50, 10_000), f))

    grid_points = quantiles[idx, np.arange(f)]

    tree = KDTree(majority_X)
    _, nn_idx = tree.query(grid_points, k=1, workers=-1)

    # Keep the most representative samples — those that the most grid
    # points mapped to sit closest to the quantile centers.
    unique_idx, popularity = np.unique(nn_idx.ravel(), return_counts=True)
    if len(unique_idx) > target:
        top = np.argsort(popularity)[-target:]
        selected = unique_idx[top]
    else:
        selected = unique_idx

    return (
        np.vstack((X[minority_mask], majority_X[selected])),
        np.hstack((y[minority_mask], majority_y[selected])),
    )

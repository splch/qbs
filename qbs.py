import numpy as np
from itertools import product
from scipy.spatial import KDTree

def qbs(X, y, version=1):
    # Identify unique classes and their counts
    unique_classes, class_counts = np.unique(y, return_counts=True)

    # Determine the minority class and its count
    minority_y = unique_classes[np.argmin(class_counts)]
    minority_y_count = np.min(class_counts)

    # Split the dataset into minority and majority class samples
    minority_X = X[y == minority_y]
    majority_X = X[y != minority_y]
    majority_y = y[y != minority_y]

    # Calculate the number of quantiles and feature count
    feature_count = X.shape[1]
    quantile_count = int(np.ceil(np.log(minority_y_count * (len(unique_classes) - 1)) / np.log(feature_count)))

    # Compute the quantiles for each feature
    quantiles = np.array([np.percentile(majority_X[:, i], np.linspace(0, 100, quantile_count)) for i in range(feature_count)])

    # Generate all possible permutations of the quantiles
    quantile_permutations = np.array(list(product(*quantiles)))

    # Build a KDTree for efficient nearest neighbor search
    tree = KDTree(majority_X)

    # Find the closest samples in the majority class for each quantile permutation
    k = 1 if version == 1 else minority_y_count
    closest_samples_idx = np.unique(tree.query(quantile_permutations, k=k, workers=-1)[1])

    # Combine the minority class samples and the selected majority class samples
    X_resampled = np.vstack((minority_X, majority_X[closest_samples_idx]))
    y_resampled = np.hstack((np.full(minority_y_count, minority_y), majority_y[closest_samples_idx]))

    return X_resampled, y_resampled

# Quantile-Based Balanced Sampling Algorithm

The Quantile-Based Balanced Sampling Algorithm is a method for balancing imbalanced datasets, particularly when there is a significant disparity between the number of samples in majority and minority classes. This algorithm helps create a more balanced dataset to improve machine learning model performance by calculating quantiles for each feature and selecting the closest non-minority class samples to each quantile permutation.

## Table of Contents

- [Overview](#overview)
- [Steps](#steps)
- [Usage](#usage)
- [Example](#example)
- [Contributing](#contributing)

## Overview

The algorithm works by calculating quantiles for each feature in the dataset, generating a set of permutations of the quantiles, and selecting the closest non-minority class samples to each quantile permutation to balance the dataset. This process preserves the underlying data distribution while ensuring an equal representation of both majority and minority classes.

## Steps

1. Count the unique non-minority class labels (c), minority class samples (m), and features (f).

2. Create an empty set (d) and add all minority class samples to it.

3. Calculate the number of quantiles (q) such that `f^q=c*m`.

4. Calculate the `q` quantiles for each feature.

5. Generate a set of all permutations of `c` quantiles (p).

6. Sort the non-minority class samples by their distance to each quantile for each feature.

7. For each quantile permutation in `p`, add the closest non-minority class sample to set `d`.

8. Return the balanced dataset `d`.

## Usage

This algorithm can be implemented in various programming languages such as Python, R, or MATLAB. You can use it to preprocess your imbalanced dataset before feeding it to your machine learning model. Please note that you may need to adjust the algorithm according to the specific data structure and requirements of your project.

## Example

Look at the [qbs.py](blob/main/qbs.py) file for a sample implementation.

## Contributing

We welcome contributions to improve this algorithm. Feel free to submit pull requests or raise issues to discuss potential improvements, bug fixes, or feature requests.

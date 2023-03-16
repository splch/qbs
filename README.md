# Quantile-Based Representative Subsampling (QBRS)

A method for selecting a representative subset from a given superset by dividing the data into quantiles across multiple dimensions. The algorithm identifies samples closest to the quantile boundaries in each dimension and combines them to form the most representative subset possible. This approach ensures that the final subset captures the overall distribution of the original data across the chosen dimensions.

## Algorithm

1. Define the number of samples you want in the final subset (n).

2. Calculate the required number of quantiles (q) for each dimension (d), such that q^d is approximately equal to the desired number of samples. In this example, q=⌊n^1/d⌋.

3. For each dimension, divide the data into q quantiles.

4. Create all possible combinations of quantile boundaries for each dimension.

5. For each combination, find the sample from the superset that is closest to the combination's boundary values across all dimensions (e.g., using Euclidean distance or another appropriate metric).

6. Add the sample found in step 5 to the subset, and remove it from the superset to avoid selecting it again.

7. Repeat steps 5-6 for all combinations.

> The resulting subset should be a representative sample of the original superset across the chosen dimensions.

## Applications

Quantile-Based Representative Subsampling (QBRS) is useful in various domains for selecting a representative subset from a given superset. Some applications include:

**Data compression**: In situations where storing or processing the entire dataset is computationally expensive or time-consuming, QBRS can be used to create a smaller, representative dataset that retains the essential features of the original data.

**Data visualization**: QBRS can be employed to create representative subsets for visualizing complex, high-dimensional data. By selecting samples that capture the overall distribution, visualization tools can better display patterns, trends, and relationships in the data.

**Machine learning and data mining**: In machine learning and data mining tasks, using a representative subset instead of the entire dataset can lead to faster training and model evaluation times. Additionally, it can reduce overfitting by simplifying the input data without losing crucial information.

**Survey sampling and social research**: QBRS can be applied to select a representative sample from survey data or other research datasets. By including samples from across the quantile boundaries, researchers can ensure that their findings are more likely to generalize to the entire population.

**Environmental monitoring**: QBRS can help in selecting representative samples for monitoring environmental variables, such as air quality, water quality, or soil composition. This can lead to more efficient and accurate assessments of environmental conditions and trends.

**Finance and economics**: In finance and economics, QBRS can be used to create representative portfolios or to select representative samples from large datasets of financial transactions, economic indicators, or consumer behaviors. This can help identify patterns, assess risks, and inform decision-making.

Overall, Quantile-Based Representative Subsampling (QBRS) is a versatile and efficient technique for selecting representative subsets from large and complex datasets across a wide range of domains.

## Implementation

This selects a representative subset from a given dataset with feature matrix X and label vector y. This implementation assumes that both X and y are NumPy arrays and that the selected dimensions are continuous.

```python
def qbrs(X, y, n):
    def calculate_quantiles(X, q):
        quantiles = []
        for i in range(X.shape[1]):
            quantiles.append(np.percentile(X[:, i], np.linspace(0, 100, q + 1)))
        return quantiles

    def find_closest_sample(combination, X, quantiles):
        distances = cdist([combination], X, metric='euclidean')
        return np.argmin(distances)

    d = X.shape[1]
    q = int(np.floor(n**(1/d)))
    quantiles = calculate_quantiles(X, q)

    combinations = np.array(np.meshgrid(*quantiles)).T.reshape(-1, d)
    subset_indices = []

    for combination in combinations:
        idx = find_closest_sample(combination, X, quantiles)
        if idx not in subset_indices:
            subset_indices.append(idx)

    X_sub = X[subset_indices]
    y_sub = y[subset_indices]

    return X_sub, y_sub
```

## At a Glance

Quantile-Based Representative Subsampling (QBRS) is a useful method for selecting a representative subset of data when the original dataset is too large to analyze or visualize effectively. QBRS is particularly useful when dealing with high-dimensional data, where it can be difficult to identify a representative subset manually.

QBRS works by dividing the data into quantiles across each dimension, then selecting samples that are closest to the quantile boundaries across all dimensions. By doing so, QBRS ensures that the final subset captures the overall distribution of the original data across the chosen dimensions.

The algorithm involves calculating the required number of quantiles for each dimension, dividing the data into quantiles, creating all possible combinations of quantile boundaries, and then finding the sample closest to each combination's boundary values. The resulting subset should be a representative sample of the original data across the chosen dimensions.

QBRS can be applied to a wide range of datasets, and it is particularly useful for visualizing large datasets or identifying key patterns and trends within data. However, it is important to note that QBRS may not be suitable for all datasets, and users should exercise caution when interpreting the results of any subsampling method.

## Limitations / Assumptions

- Continuous dimensions: Assumes the selected dimensions are continuous in nature. It may not perform well on categorical dimensions, as the concept of quantiles may not be applicable or may need modifications to accommodate categorical data.

- Representative quantiles: Assumes that dividing the data into quantiles and selecting samples closest to the quantile boundaries will produce a representative subset. If the data distribution within each dimension is heavily skewed or has multiple modes, this assumption might not hold true.

- Scaling dimensionality: As the number of desired samples is proportional to the number of dimensions, the algorithm works best for small to moderate dimensionality. For high-dimensional data, the number of desired samples may become impractically large, or the subsampling process may not capture enough variability in the data.

- Assumes independence between dimensions: Does not consider any possible interactions or correlations between the selected dimensions. If the dimensions are strongly correlated, this method may not capture the joint distribution effectively.

- No guarantee of optimality: The algorithm is heuristic in nature and does not provide a guarantee that the final subset will be the most representative possible. Other techniques or variations of this algorithm may provide better representations of the original dataset.

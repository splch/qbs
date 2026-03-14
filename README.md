# Quantile-Based Balanced Sampling

QBS is an undersampling algorithm for imbalanced datasets. It selects representative majority-class samples by placing a quantile-derived grid over the feature space and keeping the nearest real sample to each grid point.

## Algorithm

1. Identify the minority class (count `m`), non-minority class count `c`, and feature count `f`.
2. Keep all minority samples.
3. Compute `q = ceil(log(c*m) / log(f))` quantiles per feature, so `q^f ≈ c*m`.
4. Generate grid points from the quantile values — full enumeration when `q^f` is small, random sampling otherwise.
5. For each grid point, find the nearest majority-class sample (via KDTree).
6. Return the minority samples plus the unique selected majority samples.

## Usage

```python
from qbs import qbs

X_balanced, y_balanced = qbs(X, y)
```

## Files

- `qbs.py` — the algorithm
- `comparison.ipynb` — benchmark against other undersampling methods

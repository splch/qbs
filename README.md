# Quantile Balanced Ensemble (QBE)

An ensemble classifier for imbalanced datasets, inspired by [Quantile-Based Balanced Sampling](https://github.com/splch/qbs/tree/07a0a56).

QBE uses QBS's core innovation — selecting representative samples via a quantile-derived grid over feature space — but wraps it in an ensemble to avoid the information loss inherent in single-pass undersampling. Each base learner trains on a different QBS-selected balanced subset, so collectively the ensemble covers the full majority distribution.

## Algorithm

1. Compute `q` quantile values per feature for the majority class.
2. For each of `n_estimators` base learners:
   - Sample random points from the quantile grid (different seed per learner).
   - Find the nearest real majority samples via KDTree.
   - Train a decision tree on all minority samples + selected majority samples.
3. Predict by averaging `predict_proba` across all learners.

## Usage

```python
from qbe import QuantileBalancedEnsemble

clf = QuantileBalancedEnsemble(n_estimators=50, q=10, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

sklearn-compatible — works with `cross_validate`, `Pipeline`, `GridSearchCV`, etc.

## Files

- `qbe.py` — the algorithm
- `comparison.ipynb` — benchmark against sampling methods and other ensemble classifiers

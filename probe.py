"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.

Logistic Regression with L2 regularisation, applied to PCA-whitened features.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

class HallucinationProbe(nn.Module):
    """Linear probe for hallucination detection.

    Pipeline:
        StandardScaler → PCA(256, whitened) → LogisticRegression(C=0.1)

    """
    N_COMPONENTS: int = 128

    def __init__(self) -> None:
        super().__init__()
        self._pipeline: Pipeline | None = None
        self._threshold: float = 0.5  # tuned by fit_hyperparameters()

    def _build_pipeline(self, n_features: int) -> Pipeline:
        n_components = min(self.N_COMPONENTS, n_features)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, whiten=True, random_state=42)),
            ("clf", LogisticRegression(
                C=0.05,  # middle ground: 0.1 overfit, 0.01 over-regularised
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
            )),
        ])


    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit the probe on labelled feature vectors.

        Args:
            X: ``(n_samples, feature_dim)`` float array.
            y: ``(n_samples,)`` int array; 0 = truthful, 1 = hallucinated.
        """
        self._pipeline = self._build_pipeline(X.shape[1])
        self._pipeline.fit(X, y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune decision threshold on validation data to maximise F1.

        Searches over all unique predicted probabilities plus a fine uniform
        grid to find the threshold that maximises the validation F1 score.

        Args:
            X_val: ``(n_val, feature_dim)`` float array.
            y_val: ``(n_val,)`` int array.
        """
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(
            np.concatenate([probs, np.linspace(0.0, 1.0, 201)])
        )
        best_t, best_f1 = 0.5, -1.0
        for t in candidates:
            score = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
            if score > best_f1:
                best_f1, best_t = score, float(t)
        self._threshold = best_t
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        if self._pipeline is None:
            raise RuntimeError("Probe not fitted. Call fit() first.")
        return self._pipeline.predict_proba(X)


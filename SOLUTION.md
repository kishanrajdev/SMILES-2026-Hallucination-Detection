# SOLUTION.md — SMILES-2026 Hallucination Detection

## Reproducibility Instructions

### Setup and Run

```bash
git clone https://github.com/kishanrajdev/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate.bat     # Windows


pip install -r requirements.txt
python solution.py
```

This produces `results.json` and `predictions.csv` in the project root.

## Solution Description

### Files Modified

- `aggregation.py` — layer selection and token pooling strategy
- `probe.py` — classifier pipeline (StandardScaler → PCA → Logistic Regression)
- `splitting.py` — 5-fold stratified cross-validation

### aggregation.py

**1. Multi-layer extraction.** Instead of just the final layer, hidden states are extracted from 4 layers spread across the network: `[11, 15, 19, 23] `.

**2. Dual-view pooling per layer.** For each selected layer, two complementary representations are concatenated:

- **Mean-pool** over all real (non-padding) tokens — captures the average contextual representation of the full prompt+response sequence
- **Last real token** — in an autoregressive decoder, the final token attends to the entire sequence via causal attention, making its hidden state a compressed summary of everything the model has processed

This gives a feature vector of shape `4 × 2 × 896 = 7168 dimensions` per sample.

### probe.py

A scikit-learn pipeline:

```
StandardScaler → PCA(128 components, whitened) → LogisticRegression(C=0.05)
```

**Logistic Regression:**  With only ~468 training samples, a linear probe with strong regularisation generalises far better than non-linear alternatives (MLPs) which overfit severely.

**PCA:** 128 components were chosen empirically out of 7168 dimensions as the best trade-off between retained variance and overfitting.

**C=0.05** Empirically chosen via systematic search across C ∈ {0.01, 0.05, 0.1}

### splitting.py

5-fold stratified cross-validation instead of a single random split. Benefits on a small dataset (689 samples):

- Every sample participates in training in 4 of 5 folds
- Evaluation metrics are averaged over 5 independent test sets, giving a more reliable estimate of true generalisation
- Class balance is preserved in every fold

### Final Results

| Metric | Value     |
|--------|-----------|
| Avg baseline accuracy | 0.701     |
| Avg baseline F1 | 0.824     |
| **Avg test accuracy** | **0.701** |
| **Avg test F1** | **0.816** |
| **Avg test AUROC** | **0.694** |
| Avg train AUROC | 0.884     |
| Feature dim | 7168      |

---

# m-Height Prediction 

# Deep Learning for m-Height Estimation in Random Polytopes

## Executive Summary

This project presents a high-precision deep learning framework for estimating the **m-height** of generator matrices — a quantity traditionally computed via expensive linear programming procedures.  
The final system is the result of a **three-stage modeling pipeline**, culminating in a **per-bucket deep ensemble architecture** that achieves strong stability and predictive accuracy on large-scale datasets (56,000+ samples).

The final model operates in **log₂-space**, incorporates **bucket-wise normalization**, and uses **ensemble averaging** to reduce variance and improve robustness across highly skewed distributions.

---

## Problem Formulation

We consider tuples of the form:

\[
(n, k, m, P)
\]

where:
- \( P \in \mathbb{R}^{k \times (n-k)} \) is a real-valued matrix,
- The full generator matrix is given by:
  \[
  G = [I_k \mid P]
  \]

The goal is to predict the **m-height**, a scalar quantity derived from geometric properties of the polytope induced by \( G \).

### Key Challenges
- The target distribution is **heavy-tailed**, spanning several orders of magnitude.
- Direct computation of m-height requires solving expensive optimization problems.
- Different \((k, m)\) regimes exhibit fundamentally different statistical behaviors.

### Optimization Objective
The primary optimization target is:

\[
\text{MSE}(\log_2(\hat{y}), \log_2(y))
\]

which stabilizes training and ensures relative error consistency across scales.

---

## Evolution of the Model Architecture

### Stage 1 — Baseline Learning (Tree-Based Models)

**Goal:** Establish a strong baseline and understand structural patterns.

- Gradient-boosted models (XGBoost / LightGBM)
- Extensive feature engineering from matrix statistics
- Optional KNN-based residual correction

**Outcome:**
- Solid baseline performance
- Revealed strong heterogeneity across \((k, m)\) buckets

---

### Stage 2 — Deep Neural Networks (Log-Space Regression)

**Key Improvements:**
- Transition to deep neural networks
- Predictions performed in **log₂(m-height)** space
- Per-bucket modeling to isolate structural regimes
- Standardization of targets within each bucket

**Architecture:**
- Fully connected DNN (256 → 128 → 64)
- Batch normalization + dropout
- Early stopping and L2 regularization

**Benefits:**
- Improved stability over wide dynamic ranges
- Better generalization within each bucket
- Significant reduction in variance relative to Stage 1

---

### Stage 3 — Per-Bucket Ensemble (Final Model)

The final system introduces **ensemble learning** to further reduce variance and improve robustness.

#### Core Design:
- Independent models trained **per (k, m) bucket**
- Each bucket uses a **3-seed neural ensemble**
- Predictions averaged in standardized log-space
- Gentle clipping applied to avoid extreme tail explosions

#### Final Prediction:
\[
\hat{y} = 2^{\left( \mu + \sigma \cdot \frac{1}{N} \sum_{i=1}^{N} \hat{z}_i \right)}
\]

Where:
- \( \hat{z}_i \) = standardized prediction from ensemble member \( i \)
- \( \mu, \sigma \) = bucket-specific normalization constants

This repo contains:
- **Stage 1 / Project 1**: baseline modeling + initial featurization
- **Stage 2 / Project 2**: **per-(k,m) bucket** DNN trained in **standardized log₂-space**
- **Stage 3 / Project 3**: improved + stabilized **per-bucket DNN ensemble (3 seeds)**, same log₂ training strategy

---

## Model at a Glance — Stage 1 (Baseline Feature Model)

The Stage 1 model establishes a **strong baseline** by learning direct mappings from engineered matrix features to the target m-height.

**Core ideas:**
- Extract rich statistical and structural features from the input matrix \( P \), including:
  - Global statistics (mean, std, min, max)
  - Row/column norms
  - Singular value spectrum
  - Distributional descriptors (quantiles, skewness, kurtosis)
- Train a **single global regression model** across all samples (no bucket separation).
- Predict directly in the **original m-height space** (no log transform).
- Use regularization to reduce overfitting and encourage generalization.

**Purpose of Stage 1**
- Establish a strong baseline.
- Understand global data behavior.
- Identify heterogeneity across \((k, m)\) buckets.

---

## Model at a Glance — Stage 2 (Per-Bucket DNN with Log-Scaling)

Stage 2 improves stability and accuracy by explicitly modeling structural heterogeneity.

**Key improvements over Stage 1:**
- Split data into **(k, m) buckets**, acknowledging different statistical regimes.
- Predict in **log₂(mHeight)** space to stabilize heavy-tailed distributions.
- Standardize targets *per bucket* using training-set statistics.
- Train a **separate deep neural network per bucket**.
- Use **Huber / MSE loss** to balance robustness and sensitivity.

**Architecture summary:**
- Fully connected DNN (256 → 128 → 64)
- Batch normalization + dropout
- Early stopping
- Independent model per bucket

**Benefits:**
- Reduced variance across buckets
- Better control of scale effects
- More accurate mid-range predictions

---

## Model at a Glance — Stage 3 (Final Ensemble Model)

The final model builds on Stage 2 and introduces **ensemble learning and stabilization**, producing the most reliable predictions.

**Core design principles:**
- Still operates **per (k, m) bucket**, preserving structural specialization.
- Predicts in **log₂(mHeight)** space for numerical stability.
- Uses **multiple independently trained DNNs (ensemble)** per bucket.
- Averages predictions across seeds to reduce variance.
- Applies a **soft log-space clipping** to prevent extreme tail explosions.

**Final prediction rule:**
\[
\hat{y} = 2^{\left(\mu + \sigma \cdot \frac{1}{N}\sum_{i=1}^{N} \hat{z}_i\right)}
\]

where:
- \(\hat{z}_i\) = standardized log-prediction from ensemble member \(i\)
- \(\mu, \sigma\) = per-bucket normalization constants

**Why this works well:**
- Ensemble averaging smooths noise and instability.
- Log-space learning handles heavy-tailed distributions.
- Per-bucket modeling captures structural variation.
- Final predictions remain numerically stable and well-calibrated.

---

## Summary Comparison

| Stage | Strategy | Stability | Accuracy | Purpose |
|------|----------|-----------|----------|---------|
| Stage 1 | Single global model | Low | Baseline | Exploration |
| Stage 2 | Per-bucket DNN | Medium–High | Strong | Structural modeling |
| Stage 3 | Per-bucket DNN ensemble | **Very High** | **Best** | Final deployment |


## Repo structure

```
mheight-projects/
  notebooks/
    project_stage1_exploration.ipynb
    project_stage2_modeling.ipynb
    project_stage3_ensemble.ipynb

  models/
    per_bucket_models_ROBUST_PLUS.zip          
    per_bucket_models_DNN_P2_final.zip         
    per_bucket_models_DNN_P3_project3-1.zip    

  src/
    featurizer.py
    predictor.py
    utils_io.py

  docs/
    model_flowchart.png

  requirements.txt
  README.md
  LICENSE
```

---

## Quickstart (local)

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell

pip install -r requirements.txt
```

### 2) Run the Project 3 predictor on a dataset
You need two pickle files (as provided in the class):
- `input` : list of `(n, k, m, P)`
- optional `output`: list/array of ground-truth m-heights

Example:
```bash
python -m src.predictor \
  --model_dir models/per_bucket_models_DNN_P3_project3 \
  --input_pkl /path/to/DS-3-Train-n_k_m_P \
  --output_pkl /path/to/DS-3-Train-mHeights \
  --out_pkl predictions_project3.pkl
```

If you omit `--output_pkl`, the script will just generate predictions.

---

## Notes on “stability” checks

The most useful checks I used while iterating:
- **No NaN/Inf/≤0** predictions
- Distribution sanity in **log₂-space**: histogram + quantile table
- Tail sanity: top-1% threshold and max values (watch for extreme blow-ups)
- Bucket-wise diagnostics: per-(k,m) log₂-MSE, especially hard buckets

(These checks are included as notebook cells in Stage 3.)

---

Author: **Vigya Awasthi**

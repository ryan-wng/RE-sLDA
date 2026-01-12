# Resampling-Based Sparse LDA

## Using a Python implementation of Adaptive and Elastic-Net Sparse Discriminant Analysis (accSDA)  
Developed by referencing the original **R `accSDA` package**.

This repository provides an original Python version of the core algorithms in `accSDA`, including sparse discriminant analysis, elastic-net regularization, ADMM solvers, accelerated proximal gradient methods, and ordinal extensions.

---

## Overview

Sparse Discriminant Analysis (SDA) is a supervised classification framework designed for **high-dimensional, low-sample-size** data (e.g., genomics, proteomics, imaging).  

This repository implements these ideas **natively in Python**, while preserving the **mathematical structure and convergence behavior** of the original R code.

## Notes

- Numerical linear algebra mirrors R behavior (backsolve, forwardsolve)
- Convergence criteria and stopping rules are preserved
- Feature selection behavior matches accSDA outputs under identical seeds

## Testing

- Individual solvers validated against R outputs
- Shape and sparsity consistency confirmed
- Floating-point differences expected at ~1e−6 level
- Full large-scale benchmarks ongoing
---

## Basic Usage

Clone the repository and install dependencies.
#### To use **Binary or Multiclass SDA**:
```bash
from ASDA import ASDA
import numpy as np

X = np.random.randn(100, 500)     # n x p data
y = np.random.choice([0, 1, 2], size=100)

res = ASDA(
    Xt=X,
    Yt=y,
    gam=1e-3,
    lam=1e-4,
    method="SDAAP"
)

B = res["beta"]       # sparse coefficients
lda = res["fit"]      # trained LDA model
```
#### To use **Ordinal SDA**:
```bash
from ordASDA import ordASDA

res = ordASDA(
    Xt=X,
    Yt=y,      # ordinal labels
    s=1,
    gam=0,
    lam=0.05
)

selected_features = res["n_selected"]
```
#### To use **Cross-Validation for λ**:
```bash
res = ASDA(
    Xt=X,
    Yt=y,
    gam=1e-3,
    lam=np.logspace(-5, -1, 20),
    method="SDAAP",
    control={"CV": True, "folds": 5}
)

best_lambda = res["lambda"]
```
## Reference

If you use this code in academic work, please cite the original accSDA paper/package:
> Clemmensen, L., Hastie, T., Witten, D., & Ersbøll, B. (2011).  
> Sparse Discriminant Analysis. Technometrics.

---

## Disclaimer

This is a research-grade implementation, not a drop-in replacement for scikit-learn classifiers.  
It is intended for:
- Method development
- Reproducibility
- High-dimensional statistical learning research
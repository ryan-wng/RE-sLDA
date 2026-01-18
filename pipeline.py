import math
import random
import argparse
import pandas as pd
import numpy as np

from subsampling import run_parallel_subspace as subsampling
from bootstrapping import run_parallel_bootstraps as bootstrapping

# ------------------
# Argument parsing
# ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run subsampling or bootstrapping pipeline")
    parser.add_argument(
        "mode",
        choices=["subsampling", "bootstrapping"],
        help="Which pipeline stage to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------
    # Load data
    # ------------------
    x = pd.read_csv("use_glio_data_filter1000.csv")
    y = pd.read_csv("use_glio_dataY_filter1000.csv")

    X_new = x.values
    Y = y.values.squeeze()
    varnames_full = x.columns.tolist()
    n_cols = X_new.shape[1]

    # ------------------
    # Subsampling config
    # ------------------
    iters = 200
    test_ratio = 0.20
    n_folds_cv = 4
    predictor_subset = int(round(n_cols * 0.8))
    n_subspaces = 5

    # ------------------
    # Bootstrapping config
    # ------------------
    subspace_size = 5
    target_unique_prob = 0.8
    boot_scale = -math.log(1 - target_unique_prob)

    n_bootstraps = 200
    cv_folds = 4
    base_seed = 42

    # Reproducibility
    random.seed(base_seed)
    np.random.seed(base_seed)

    # ------------------
    # Run selected mode
    # ------------------
    if args.mode in ("subsampling"):
        print("\n=== Running subsampling ===")
        subsampling(
            iters,
            X_new,
            Y,
            varnames_full,
            test_ratio=test_ratio,
            n_folds_cv=n_folds_cv,
            predictor_subset=predictor_subset,
            n_subspaces=n_subspaces,
        )

    if args.mode in ("bootstrapping"):
        print("\n=== Running bootstrapping ===")
        bootstrapping(
            n_bootstraps=n_bootstraps,
            X_new=X_new,
            Y=Y,
            varnames_full=varnames_full,
            boot_scale=boot_scale,
            predictor_subset=predictor_subset,
            subspace_size=subspace_size,
            cv_folds=cv_folds,
            base_seed=base_seed,
        )


if __name__ == "__main__":
    main()

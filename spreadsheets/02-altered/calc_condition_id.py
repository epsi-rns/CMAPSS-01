# calc_condition_id.py
#
# Demonstrates how condition_id was computed for C-MAPSS FD002 and FD004.
# These sub-datasets have 6 operating conditions, identifiable by clustering
# the three op_setting columns using KMeans(k=6).
#
# Usage:
#   python calc_condition_id.py
#
# Requires: pandas, scikit-learn, numpy

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

COLUMNS = [
    'unit_number', 'time_cycles',
    'op_setting_1', 'op_setting_2', 'op_setting_3',
    'T2', 'T24', 'T30', 'T50',
    'P2', 'P15', 'P30',
    'Nf', 'Nc', 'epr', 'Ps30', 'phi',
    'NRf', 'NRc', 'BPR', 'farB', 'htBleed',
    'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32'
]


def calc_condition_id(train_path, test_path):
    """
    Cluster cycles into 6 operating conditions based on op_setting columns.

    The KMeans model is FITTED on the train set only, then APPLIED to both
    train and test. This ensures consistent condition labels across splits.

    Parameters
    ----------
    train_path : str   path to train_FDxxx.txt
    test_path  : str   path to test_FDxxx.txt

    Returns
    -------
    train_df, test_df : DataFrames with condition_id column appended
    km                : fitted KMeans model (centroids, inertia, etc.)
    """

    train = pd.read_csv(train_path, sep=r'\s+', header=None, names=COLUMNS)
    test  = pd.read_csv(test_path,  sep=r'\s+', header=None, names=COLUMNS)

    # --- Step 1: extract the three op_setting columns ---
    op_cols  = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    train_ops = train[op_cols].values   # shape: (n_train_rows, 3)
    test_ops  = test[op_cols].values    # shape: (n_test_rows,  3)

    # --- Step 2: fit KMeans on train op_settings only ---
    km = KMeans(
        n_clusters=6,
        random_state=42,   # reproducible
        n_init=10          # run 10 times, keep best
    )
    km.fit(train_ops)

    # --- Step 3: assign condition labels ---
    # km.predict() returns 0-indexed cluster labels (0, 1, 2, 3, 4, 5)
    # we add 1 to make them 1-indexed (1, 2, 3, 4, 5, 6)
    train['condition_id'] = km.predict(train_ops) + 1
    test['condition_id']  = km.predict(test_ops)  + 1

    return train, test, km


def show_centroids(km):
    """Print the 6 cluster centroids in a readable table."""
    print("\nCluster centroids (the 6 operating conditions):")
    print(f"{'Cond':>5}  {'op_setting_1':>13}  {'op_setting_2':>13}  {'op_setting_3':>13}")
    print("-" * 52)
    for i, centroid in enumerate(km.cluster_centers_, start=1):
        print(f"  {i:>3}  {centroid[0]:>13.4f}  {centroid[1]:>13.4f}  {centroid[2]:>13.4f}")
    print(f"\nInertia (sum of squared distances): {km.inertia_:.2f}")


def show_distribution(train, test):
    """Show how many rows fall into each condition."""
    print("\nCondition distribution — train:")
    print(train['condition_id'].value_counts().sort_index().to_string())
    print("\nCondition distribution — test:")
    print(test['condition_id'].value_counts().sort_index().to_string())


def verify_consistency(train):
    """
    Each engine should see multiple conditions per life.
    Check that condition switching happens within single engines.
    """
    per_engine = (
        train.groupby('unit_number')['condition_id']
        .nunique()
        .describe()
    )
    print("\nUnique conditions per engine (train):")
    print(per_engine.to_string())


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    # --- FD001: single condition ---
    print("=" * 55)
    print("FD001 (1 condition — no clustering needed)")
    print("=" * 55)
    train_001 = pd.read_csv(
        'train_FD001.txt', sep=r'\s+', header=None, names=COLUMNS)
    test_001 = pd.read_csv(
        'test_FD001.txt',  sep=r'\s+', header=None, names=COLUMNS)
    train_001['condition_id'] = 1
    test_001['condition_id']  = 1
    print("All rows assigned condition_id = 1")

    # --- FD002: six conditions ---
    print("\n" + "=" * 55)
    print("FD002 (6 conditions — KMeans clustering)")
    print("=" * 55)
    train_002, test_002, km_002 = calc_condition_id(
        'train_FD002.txt', 'test_FD002.txt')
    show_centroids(km_002)
    show_distribution(train_002, test_002)
    verify_consistency(train_002)

    # --- spot check: first 5 rows of train FD002 ---
    print("\nFirst 5 rows of train_FD002 (op_settings + condition_id):")
    cols = ['unit_number', 'time_cycles',
            'op_setting_1', 'op_setting_2', 'op_setting_3', 'condition_id']
    print(train_002[cols].head().to_string(index=False))

    # --- FD004: six conditions (same approach) ---
    print("\n" + "=" * 55)
    print("FD004 (6 conditions — KMeans clustering)")
    print("=" * 55)
    train_004, test_004, km_004 = calc_condition_id(
        'train_FD004.txt', 'test_FD004.txt')
    show_centroids(km_004)

    print("\nDone. The fitted km object contains:")
    print("  km.cluster_centers_  — centroid coordinates (6 x 3 array)")
    print("  km.labels_           — condition_id for each train row (0-indexed)")
    print("  km.inertia_          — total within-cluster sum of squares")
    print("  km.n_iter_           — iterations until convergence")


raise SystemExit(main())

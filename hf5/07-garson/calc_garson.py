# calc_garson.py
#
# Garson's algorithm for input importance in trained MLP networks.
# Applied to all 8 models from the C-MAPSS benchmark.
#
# Garson (1991) extended for two hidden layers:
#   For each input i, the importance is computed by partitioning
#   the connection weights through all hidden neurons back to output.
#
# Algorithm (two hidden layers: W1, W2, W3):
#   Step 1: for each hidden1 neuron j:
#             q1[i,j] = |W1[i,j]| / sum_k(|W1[k,j]|)
#   Step 2: for each hidden2 neuron m:
#             q2[j,m] = |W2[j,m]| / sum_k(|W2[k,m]|)
#   Step 3: propagate through hidden2 to output:
#             r[i,m] = sum_j( q1[i,j] * q2[j,m] )
#   Step 4: weight by output connections:
#             s[i]   = sum_m( r[i,m] * |W3[m]| )
#   Step 5: normalise:
#             importance[i] = s[i] / sum_k(s[k])
#
# Output:
#   garson_results/
#     FD001_PCA_garson.csv    importance per feature
#     FD001_NORM_garson.csv
#     ...
#     garson_summary.csv      all models side by side
#
# Reference:
#   Garson, G.D. (1991). Interpreting neural network connection weights.
#   AI Expert, 6(4), 46-51.

import os
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn


# ── constants ─────────────────────────────────────────────────────────────────

ANN_DIR    = '../06-ANN'
H5_DIR     = '../05-PCA'
OUT_DIR    = '.'
FD_LIST    = ['FD001', 'FD002', 'FD003', 'FD004']
INPUT_LIST = ['PCA', 'NORM']


# ── model definition (must match training) ────────────────────────────────────

class RulMlp(nn.Module):

  def __init__(self, input_dim):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(input_dim, 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1),
    )

  def forward(self, x):
    return self.net(x).squeeze(-1)


# ── feature name loader ───────────────────────────────────────────────────────

class FeatureLoader:

  def __init__(self, h5_dir):
    self.h5_dir = h5_dir

  def calc_features(self, fd, input_type):
    path = os.path.join(self.h5_dir, f'cmapss_{fd}.h5')
    with h5py.File(path, 'r') as f:
      if input_type == 'PCA':
        cols = list(f['/pca/train'].attrs['columns'])
        return [c for c in cols if c.startswith('PC')]
      else:
        return list(f['/train/data_norm'].attrs['columns'])


# ── Garson calculator ─────────────────────────────────────────────────────────

class GarsonCalculator:

  def calc_importance(self, model, feature_names):
    """
    Extract weights from trained MLP and apply Garson's algorithm.
    Architecture: input → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1)
    Weight matrices: W1 (64, n_input), W2 (32, 64), W3 (1, 32)
    """
    layers = list(model.net.children())
    linear_layers = [l for l in layers if isinstance(l, nn.Linear)]

    # extract weight matrices — shape (out, in)
    W1 = linear_layers[0].weight.detach().numpy()  # (64, n_input)
    W2 = linear_layers[1].weight.detach().numpy()  # (32, 64)
    W3 = linear_layers[2].weight.detach().numpy()  # (1, 32)

    n_input  = W1.shape[1]
    n_h1     = W1.shape[0]
    n_h2     = W2.shape[0]

    # Step 1: partition W1 — q1[i,j] = |W1[j,i]| / sum_k(|W1[j,k]|)
    # for each hidden1 neuron j, how much does input i contribute?
    W1_abs = np.abs(W1)                          # (n_h1, n_input)
    q1     = W1_abs / W1_abs.sum(axis=1, keepdims=True)  # (n_h1, n_input)

    # Step 2: partition W2 — q2[j,m] = |W2[m,j]| / sum_k(|W2[m,k]|)
    # for each hidden2 neuron m, how much does h1 neuron j contribute?
    W2_abs = np.abs(W2)                          # (n_h2, n_h1)
    q2     = W2_abs / W2_abs.sum(axis=1, keepdims=True)  # (n_h2, n_h1)

    # Step 3: propagate input through h1 → h2
    # r[i,m] = sum_j( q1[j,i] * q2[m,j] )
    # q1.T is (n_input, n_h1), q2.T is (n_h1, n_h2)
    r = q1.T @ q2.T                              # (n_input, n_h2)

    # Step 4: weight by output layer W3
    W3_abs = np.abs(W3).flatten()                # (n_h2,)
    s      = r @ W3_abs                          # (n_input,)

    # Step 5: normalise to sum to 1
    importance = s / s.sum()

    rows = []
    for i, feat in enumerate(feature_names):
      rows.append({
        'feature':         feat,
        'importance':      round(float(importance[i]), 6),
        'importance_pct':  round(float(importance[i]) * 100, 4),
      })

    df = pd.DataFrame(rows).sort_values(
      'importance', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1

    return df


# ── runner ────────────────────────────────────────────────────────────────────

class GarsonRunner:

  def __init__(self, ann_dir, feature_loader, calculator, out_dir):
    self.ann_dir       = ann_dir
    self.feature_loader = feature_loader
    self.calculator    = calculator
    self.out_dir       = out_dir

  def calc_run_all(self):
    os.makedirs(self.out_dir, exist_ok=True)
    all_rows = []

    for fd in FD_LIST:
      for input_type in INPUT_LIST:
        label    = f'{fd}_{input_type}'
        model_path = os.path.join(
          self.ann_dir, label, 'model.pt')

        if not os.path.exists(model_path):
          print(f"  [{label}] model.pt not found — skipping")
          continue

        features = self.feature_loader.calc_features(fd, input_type)
        n_input  = len(features)

        # load model
        model = RulMlp(n_input)
        model.load_state_dict(
          torch.load(model_path, map_location='cpu',
                     weights_only=True))
        model.eval()

        # run Garson
        df = self.calculator.calc_importance(model, features)
        df.insert(0, 'model', label)
        df.insert(1, 'sub_dataset', fd)
        df.insert(2, 'input_type', input_type)

        # save per-model CSV
        out_path = os.path.join(
          self.out_dir, f'{label}_garson.csv')
        df.to_csv(out_path, index=False)

        # collect for summary
        all_rows.append(df)

        # print top 5
        print(f"\n  [{label}] top 5 features:")
        for _, row in df.head(5).iterrows():
          bar = '█' * int(row['importance_pct'] / 2)
          print(f"    {int(row['rank']):2d}. {row['feature']:14s}  "
                f"{row['importance_pct']:6.2f}%  {bar}")

    return all_rows

  def build_summary(self, all_rows):
    """
    Wide-format summary: one row per feature, one column per model.
    Makes it easy to compare importance across models.
    """
    # collect all unique features
    all_features = []
    for df in all_rows:
      for feat in df['feature']:
        if feat not in all_features:
          all_features.append(feat)

    summary_rows = []
    for feat in all_features:
      row = {'feature': feat}
      for df in all_rows:
        model = df['model'].iloc[0]
        match = df[df['feature'] == feat]
        row[model] = round(float(
          match['importance_pct'].values[0]), 2) \
          if len(match) > 0 else np.nan
      summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    # sort by mean importance across all models (ignoring NaN)
    model_cols = [c for c in summary.columns if c != 'feature']
    summary['mean_pct'] = summary[model_cols].mean(axis=1)
    summary = summary.sort_values(
      'mean_pct', ascending=False).reset_index(drop=True)

    return summary

  def print_comparison(self, summary):
    model_cols = [c for c in summary.columns
                  if c not in ['feature', 'mean_pct']]

    print(f"\n{'=' * 90}")
    print("  GARSON IMPORTANCE SUMMARY  (% contribution per model)")
    print(f"{'=' * 90}")
    print(f"  {'feature':14s}  {'mean':>6}", end='')
    for m in model_cols:
      print(f"  {m:12s}", end='')
    print()
    print('-' * 90)

    for _, row in summary.iterrows():
      feat = row['feature']
      mean = row['mean_pct']
      print(f"  {feat:14s}  {mean:6.2f}", end='')
      for m in model_cols:
        val = row[m]
        if np.isnan(val):
          print(f"  {'—':>12s}", end='')
        else:
          print(f"  {val:>12.2f}", end='')
      print()

    print(f"{'=' * 90}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
  print("Garson's Algorithm — Feature Importance for C-MAPSS ANN Models")
  print(f"Models: {len(FD_LIST)} sub-datasets × "
        f"{len(INPUT_LIST)} input types = "
        f"{len(FD_LIST)*len(INPUT_LIST)} models")
  print()

  feature_loader = FeatureLoader(H5_DIR)
  calculator     = GarsonCalculator()
  runner         = GarsonRunner(
    ANN_DIR, feature_loader, calculator, OUT_DIR)

  all_rows = runner.calc_run_all()

  if not all_rows:
    print("No models found. Check ANN_DIR path.")
    return 1

  summary = runner.build_summary(all_rows)
  runner.print_comparison(summary)

  # save summary
  summary_path = os.path.join(OUT_DIR, 'garson_summary.csv')
  summary.to_csv(summary_path, index=False)

  print(f"\nFiles written:")
  for fd in FD_LIST:
    for it in INPUT_LIST:
      p = os.path.join(OUT_DIR, f'{fd}_{it}_garson.csv')
      if os.path.exists(p):
        print(f"  {p}")
  print(f"  {summary_path}")


raise SystemExit(main())

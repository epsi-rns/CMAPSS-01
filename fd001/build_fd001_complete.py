# build_fd001_complete.py
#
# Complete ANN + Garson pipeline for C-MAPSS FD001.
#
# Two models compared:
#   Model A — PCA input:  PC1, PC2  (2 features)
#   Model B — NORM input: 15 kept sensors from data_norm
#
# Architecture: input → 64 → 32 → 1
# Epochs:       500
# Batch size:   512
# Optimizer:    Adam  lr=1e-3
# Target:       RUL_capped (cap=125)
#
# Output structure:
#   fd001_results/
#     PCA/
#       model.pt
#       predictions.csv   unit_number, true_rul, pred_rul, error, abs_error
#       history.csv       epoch, train_loss, train_rmse
#     NORM/
#       model.pt
#       predictions.csv
#       history.csv
#     summary.csv         one row per model — RMSE, MAE, NASA score
#     garson_PCA.csv      feature importance — PCA model
#     garson_NORM.csv     feature importance — NORM model
#     garson_summary.csv  wide format — both models side by side
#
# Metrics:
#   RMSE      root mean square error (cycles)
#   MAE       mean absolute error (cycles)
#   NASA      PHM08 asymmetric score:
#               d < 0 (early): exp(-d/13) - 1
#               d > 0 (late):  exp( d/10) - 1

import os
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── constants ─────────────────────────────────────────────────────────────────

H5_PATH  = '/media/Docs/Tugas-Sem2/Data Analitik/05-May/20-CMAPSS/hf5/05-PCA/cmapss_FD001.h5'
OUT_DIR  = '.'
EPOCHS   = 500
BATCH    = 512
LR       = 1e-3
SEED     = 42


# ── data loader ───────────────────────────────────────────────────────────────

class DataLoader_FD001:

  def __init__(self, h5_path):
    self.h5_path = h5_path

  def calc_load(self, input_type):
    with h5py.File(self.h5_path, 'r') as f:

      # target
      tc     = list(f['/train/data'].attrs['columns'])
      y      = f['/train/data'][:, tc.index('RUL_capped')].astype(np.float32)

      # answer key
      rc     = list(f['/RUL/data'].attrs['columns'])
      rul    = f['/RUL/data'][:, rc.index('rul')].astype(np.float32)

      # unit numbers for test (positional)
      ptc    = list(f['/pca/test'].attrs['columns'])
      units  = f['/pca/test'][:, ptc.index('unit_number')]

      if input_type == 'PCA':
        prc     = list(f['/pca/train'].attrs['columns'])
        pc_idx  = [i for i, c in enumerate(prc) if c.startswith('PC')]
        X_train = f['/pca/train'][:, pc_idx].astype(np.float32)
        X_test  = f['/pca/test'][:,  pc_idx].astype(np.float32)
        feats   = [prc[i] for i in pc_idx]

      else:  # NORM
        X_train = f['/train/data_norm'][:].astype(np.float32)
        X_test  = f['/test/data_norm'][:].astype(np.float32)
        feats   = list(f['/train/data_norm'].attrs['columns'])

    return X_train, y, X_test, units, rul, feats


# ── model ─────────────────────────────────────────────────────────────────────

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


# ── trainer ───────────────────────────────────────────────────────────────────

class MlpTrainer:

  def calc_train(self, X_np, y_np, label):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = RulMlp(X_np.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()
    ds    = TensorDataset(torch.tensor(X_np), torch.tensor(y_np))
    dl    = DataLoader(ds, batch_size=BATCH, shuffle=True)

    history = []
    print(f"  Training {label} (dim={X_np.shape[1]}, "
          f"train={X_np.shape[0]:,}, epochs={EPOCHS})...")

    for ep in range(1, EPOCHS + 1):
      model.train()
      total = 0.0
      for Xb, yb in dl:
        opt.zero_grad()
        loss = crit(model(Xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(Xb)
      tl = total / len(ds)
      history.append({
        'epoch':      ep,
        'train_loss': round(tl, 4),
        'train_rmse': round(float(np.sqrt(tl)), 4),
      })
      if ep % 100 == 0:
        print(f"    epoch {ep:3d}/{EPOCHS}  rmse={np.sqrt(tl):.4f}")

    return model, pd.DataFrame(history)


# ── evaluator ─────────────────────────────────────────────────────────────────

class MlpEvaluator:

  def calc_predictions(self, model, X_np, units, true_rul):
    model.eval()
    with torch.no_grad():
      preds = model(torch.tensor(X_np)).numpy()

    rows = []
    for i, eng in enumerate(sorted(np.unique(units).astype(int))):
      mask      = (units == eng)
      last_pred = float(preds[mask][-1])
      true      = float(true_rul[i])
      error     = last_pred - true
      rows.append({
        'unit_number': int(eng),
        'true_rul':    round(true, 2),
        'pred_rul':    round(last_pred, 2),
        'error':       round(error, 2),
        'abs_error':   round(abs(error), 2),
      })

    return pd.DataFrame(rows)

  def calc_metrics(self, pred_df, label, input_dim, feats):
    errs  = pred_df['error'].values
    rmse  = float(np.sqrt(np.mean(errs ** 2)))
    mae   = float(np.mean(np.abs(errs)))
    nasa  = float(np.sum([
      np.exp(-e / 13) - 1 if e < 0 else np.exp(e / 10) - 1
      for e in errs
    ]))
    return {
      'model':       label,
      'input_dim':   input_dim,
      'n_features':  len(feats),
      'epochs':      EPOCHS,
      'rmse':        round(rmse, 4),
      'mae':         round(mae, 4),
      'nasa_score':  round(nasa, 2),
      'features':    ', '.join(feats[:6]) + ('...' if len(feats) > 6 else ''),
    }


# ── Garson algorithm ──────────────────────────────────────────────────────────

class GarsonCalculator:

  def calc_importance(self, model, feature_names):
    """
    Garson (1991) extended for two hidden layers.
    Architecture: input → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1)
    W1: (64, n_input)   W2: (32, 64)   W3: (1, 32)
    """
    layers  = [l for l in model.net.children()
               if isinstance(l, nn.Linear)]
    W1 = layers[0].weight.detach().numpy()  # (64, n_input)
    W2 = layers[1].weight.detach().numpy()  # (32, 64)
    W3 = layers[2].weight.detach().numpy()  # (1, 32)

    # Step 1: partition W1 per hidden1 neuron
    W1_abs = np.abs(W1)
    q1     = W1_abs / W1_abs.sum(axis=1, keepdims=True)   # (64, n_input)

    # Step 2: partition W2 per hidden2 neuron
    W2_abs = np.abs(W2)
    q2     = W2_abs / W2_abs.sum(axis=1, keepdims=True)   # (32, 64)

    # Step 3: propagate input through h1 → h2
    r = q1.T @ q2.T    # (n_input, 32)

    # Step 4: weight by output layer
    s = r @ np.abs(W3).flatten()   # (n_input,)

    # Step 5: normalise
    importance = s / s.sum()

    rows = []
    for i, feat in enumerate(feature_names):
      rows.append({
        'feature':        feat,
        'importance':     round(float(importance[i]), 6),
        'importance_pct': round(float(importance[i]) * 100, 4),
      })

    df = pd.DataFrame(rows).sort_values(
      'importance', ascending=False).reset_index(drop=True)
    df['rank'] = df.index + 1
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main():
  os.makedirs(OUT_DIR, exist_ok=True)

  loader    = DataLoader_FD001(H5_PATH)
  trainer   = MlpTrainer()
  evaluator = MlpEvaluator()
  garson    = GarsonCalculator()

  print("C-MAPSS FD001 — ANN + Garson Pipeline")
  print(f"Architecture: input → 64 → 32 → 1")
  print(f"Epochs: {EPOCHS}  Batch: {BATCH}  LR: {LR}")
  print()

  summary_rows  = []
  garson_frames = []

  for input_type in ['PCA', 'NORM']:
    print(f"[FD001_{input_type}]")

    X_tr, y_tr, X_te, units, rul, feats = loader.calc_load(input_type)

    model, hist_df = trainer.calc_train(X_tr, y_tr, f'FD001_{input_type}')
    pred_df        = evaluator.calc_predictions(model, X_te, units, rul)
    metrics        = evaluator.calc_metrics(
      pred_df, f'FD001_{input_type}', X_tr.shape[1], feats)

    print(f"  RMSE={metrics['rmse']}  "
          f"MAE={metrics['mae']}  "
          f"NASA={metrics['nasa_score']}")

    # Garson
    garson_df = garson.calc_importance(model, feats)
    garson_df.insert(0, 'model', f'FD001_{input_type}')

    print(f"  Top 5 features (Garson):")
    for _, row in garson_df.head(5).iterrows():
      bar = '█' * int(row['importance_pct'] / 2)
      print(f"    {int(row['rank']):2d}. {row['feature']:14s}  "
            f"{row['importance_pct']:6.2f}%  {bar}")
    print()

    # save per-model files
    run_dir = os.path.join(OUT_DIR, input_type)
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, 'model.pt'))
    pred_df.to_csv(os.path.join(run_dir, 'predictions.csv'), index=False)
    hist_df.to_csv(os.path.join(run_dir, 'history.csv'),     index=False)
    garson_df.to_csv(
      os.path.join(OUT_DIR, f'garson_{input_type}.csv'), index=False)

    summary_rows.append(metrics)
    garson_frames.append(garson_df)

  # ── summary ────────────────────────────────────────────────────────────────
  summary_df = pd.DataFrame(summary_rows)
  summary_df.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False)

  # garson wide format
  all_feats = []
  for df in garson_frames:
    for f in df['feature']:
      if f not in all_feats:
        all_feats.append(f)

  wide_rows = []
  for feat in all_feats:
    row = {'feature': feat}
    for df in garson_frames:
      m     = df['model'].iloc[0]
      match = df[df['feature'] == feat]
      row[m] = round(float(match['importance_pct'].values[0]), 2) \
               if len(match) > 0 else float('nan')
    wide_rows.append(row)

  wide_df = pd.DataFrame(wide_rows)
  cols    = [c for c in wide_df.columns if c != 'feature']
  wide_df['mean_pct'] = wide_df[cols].mean(axis=1)
  wide_df = wide_df.sort_values('mean_pct', ascending=False).reset_index(drop=True)
  wide_df.to_csv(os.path.join(OUT_DIR, 'garson_summary.csv'), index=False)

  # ── final print ────────────────────────────────────────────────────────────
  print("=" * 60)
  print("  FD001 RESULTS")
  print("=" * 60)
  print(f"  {'model':16}  {'dim':>4}  {'RMSE':>8}  "
        f"{'MAE':>8}  {'NASA':>10}")
  print("-" * 60)
  for row in summary_rows:
    print(f"  {row['model']:16}  {row['input_dim']:>4}  "
          f"{row['rmse']:>8.4f}  {row['mae']:>8.4f}  "
          f"{row['nasa_score']:>10.2f}")
  print("=" * 60)

  print()
  print("  Garson comparison (top 8 by mean importance):")
  print(f"  {'feature':16}  {'FD001_PCA':>12}  {'FD001_NORM':>12}")
  print("-" * 46)
  for _, row in wide_df.head(8).iterrows():
    pca_val  = f"{row['FD001_PCA']:.2f}%" \
               if not np.isnan(row['FD001_PCA'])  else '—'
    norm_val = f"{row['FD001_NORM']:.2f}%" \
               if not np.isnan(row['FD001_NORM']) else '—'
    print(f"  {row['feature']:16}  {pca_val:>12}  {norm_val:>12}")

  print()
  print("Files written:")
  for fn in ['PCA/model.pt', 'PCA/predictions.csv', 'PCA/history.csv',
             'NORM/model.pt', 'NORM/predictions.csv', 'NORM/history.csv',
             'summary.csv', 'garson_PCA.csv', 'garson_NORM.csv',
             'garson_summary.csv']:
    print(f"  {os.path.join(OUT_DIR, fn)}")


raise SystemExit(main())

# build_ann_benchmark.py
#
# 2-factor benchmark for C-MAPSS RUL prediction.
#
# Factor 1 — Sub-dataset:  FD001, FD002, FD003, FD004
# Factor 2 — Input type:   PCA scores, normalized sensors
#
# Total: 4 x 2 = 8 models
#
# Architecture: input_dim → 64 → 32 → 1  (Option A)
# Epochs:       500 (uniform across all runs)
# Batch size:   512
# Optimizer:    Adam  lr=1e-3
# Loss:         MSE
# Target:       RUL_capped
#
# Output structure:
#   ann_results/
#     FD001_PCA/  model.pt  predictions.csv  history.csv
#     FD001_NORM/ model.pt  predictions.csv  history.csv
#     FD002_PCA/  ...
#     FD002_NORM/ ...
#     FD003_PCA/  FD003_NORM/
#     FD004_PCA/  FD004_NORM/
#     summary.csv   8 rows — one per model, all metrics
#
# Metrics:
#   RMSE  — root mean square error (cycles)
#   MAE   — mean absolute error (cycles)
#   NASA  — PHM08 asymmetric score (late errors penalised more)

import os
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── constants ─────────────────────────────────────────────────────────────────

H5_DIR     = '../05-PCA'
OUT_DIR    = '.'
EPOCHS     = 500
BATCH_SIZE = 512
LR         = 1e-3
SEED       = 42

FD_LIST    = ['FD001', 'FD002', 'FD003', 'FD004']
INPUT_LIST = ['PCA', 'NORM']


# ── data loader ───────────────────────────────────────────────────────────────

class CmapssDataLoader:

  def __init__(self, h5_dir):
    self.h5_dir = h5_dir

  def calc_load(self, fd, input_type):
    path = os.path.join(self.h5_dir, f'cmapss_{fd}.h5')
    with h5py.File(path, 'r') as f:

      # target: RUL_capped from train
      train_cols = list(f['/train/data'].attrs['columns'])
      train_data = f['/train/data'][:]
      rul_idx    = train_cols.index('RUL_capped')
      y_train    = train_data[:, rul_idx].astype(np.float32)

      # RUL answer key
      rul_data = f['/RUL/data'][:]
      rul_cols = list(f['/RUL/data'].attrs['columns'])
      true_rul = rul_data[:, rul_cols.index('rul')].astype(np.float32)

      # unit numbers from pca/test (for per-engine eval)
      pca_te_cols = list(f['/pca/test'].attrs['columns'])
      units       = f['/pca/test'][:, pca_te_cols.index('unit_number')]

      if input_type == 'PCA':
        pca_tr_cols = list(f['/pca/train'].attrs['columns'])
        pc_idx      = [i for i, c in enumerate(pca_tr_cols)
                       if c.startswith('PC')]
        X_train     = f['/pca/train'][:, pc_idx].astype(np.float32)
        X_test      = f['/pca/test'][:, pc_idx].astype(np.float32)
        feat_names  = [pca_tr_cols[i] for i in pc_idx]

      else:  # NORM
        X_train    = f['/train/data_norm'][:].astype(np.float32)
        X_test     = f['/test/data_norm'][:].astype(np.float32)
        feat_names = list(f['/train/data_norm'].attrs['columns'])

    return X_train, y_train, X_test, units, true_rul, feat_names


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

    model     = RulMlp(X_np.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    X  = torch.tensor(X_np)
    y  = torch.tensor(y_np)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    history = []
    for epoch in range(1, EPOCHS + 1):
      model.train()
      total = 0.0
      for Xb, yb in dl:
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(Xb)

      train_loss = total / len(ds)
      rmse       = float(np.sqrt(train_loss))
      history.append({
        'epoch':      epoch,
        'train_loss': round(train_loss, 4),
        'train_rmse': round(rmse, 4),
      })
      if epoch % 100 == 0:
        print(f"      epoch {epoch:3d}/{EPOCHS}  rmse={rmse:.4f}")

    return model, pd.DataFrame(history)


# ── evaluator ─────────────────────────────────────────────────────────────────

class MlpEvaluator:

  def calc_predictions(self, model, X_np, units, true_rul):
    model.eval()
    with torch.no_grad():
      all_preds = model(torch.tensor(X_np)).numpy()

    rows = []
    for i, engine in enumerate(
        sorted(np.unique(units).astype(int))):
      mask      = (units == engine)
      last_pred = float(all_preds[mask][-1])
      true      = float(true_rul[i])
      error     = last_pred - true
      rows.append({
        'unit_number': int(engine),
        'true_rul':    round(true, 2),
        'pred_rul':    round(last_pred, 2),
        'error':       round(error, 2),
        'abs_error':   round(abs(error), 2),
      })

    return pd.DataFrame(rows)

  def calc_metrics(self, predictions_df):
    errors = predictions_df['error'].values
    rmse   = float(np.sqrt(np.mean(errors ** 2)))
    mae    = float(np.mean(np.abs(errors)))
    score  = float(np.sum([
      np.exp(-e / 13) - 1 if e < 0 else np.exp(e / 10) - 1
      for e in errors
    ]))
    return {
      'rmse':       round(rmse, 4),
      'mae':        round(mae, 4),
      'nasa_score': round(score, 4),
    }


# ── file writer ───────────────────────────────────────────────────────────────

class ResultWriter:

  def calc_write(self, run_dir, model, predictions_df, history_df):
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(run_dir, 'model.pt'))
    predictions_df.to_csv(
      os.path.join(run_dir, 'predictions.csv'), index=False)
    history_df.to_csv(
      os.path.join(run_dir, 'history.csv'), index=False)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
  os.makedirs(OUT_DIR, exist_ok=True)

  data_loader = CmapssDataLoader(H5_DIR)
  trainer     = MlpTrainer()
  evaluator   = MlpEvaluator()
  writer      = ResultWriter()

  summary_rows = []

  print("C-MAPSS ANN Benchmark")
  print(f"Architecture: input → 64 → 32 → 1")
  print(f"Epochs: {EPOCHS}  Batch: {BATCH_SIZE}  LR: {LR}")
  print(f"Runs: {len(FD_LIST)} sub-datasets × {len(INPUT_LIST)} "
        f"input types = {len(FD_LIST)*len(INPUT_LIST)} models")
  print()

  for fd in FD_LIST:
    for input_type in INPUT_LIST:
      label   = f'{fd}_{input_type}'
      run_dir = os.path.join(OUT_DIR, label)

      print(f"  [{label}]")

      X_train, y_train, X_test, units, \
      true_rul, feat_names = data_loader.calc_load(fd, input_type)

      print(f"    input_dim={X_train.shape[1]}  "
            f"train={X_train.shape[0]:,}  "
            f"test={X_test.shape[0]:,}  "
            f"features={', '.join(feat_names[:4])}"
            f"{'...' if len(feat_names)>4 else ''}")

      model, history_df = trainer.calc_train(X_train, y_train, label)
      preds_df          = evaluator.calc_predictions(
        model, X_test, units, true_rul)
      metrics           = evaluator.calc_metrics(preds_df)

      writer.calc_write(run_dir, model, preds_df, history_df)

      summary_rows.append({
        'sub_dataset':  fd,
        'input_type':   input_type,
        'input_dim':    X_train.shape[1],
        'n_features':   len(feat_names),
        'train_rows':   X_train.shape[0],
        'test_engines': len(preds_df),
        'epochs':       EPOCHS,
        'rmse':         metrics['rmse'],
        'mae':          metrics['mae'],
        'nasa_score':   metrics['nasa_score'],
      })

      print(f"    RMSE={metrics['rmse']:.4f}  "
            f"MAE={metrics['mae']:.4f}  "
            f"NASA={metrics['nasa_score']:.2f}")
      print()

  # ── summary table ──────────────────────────────────────────────
  summary_df = pd.DataFrame(summary_rows)
  summary_df.to_csv(
    os.path.join(OUT_DIR, 'summary.csv'), index=False)

  print("=" * 70)
  print("  BENCHMARK SUMMARY")
  print("=" * 70)
  print(f"  {'sub':6}  {'input':5}  {'dim':>4}  "
        f"{'RMSE':>8}  {'MAE':>8}  {'NASA':>10}")
  print("-" * 70)

  for fd in FD_LIST:
    for input_type in INPUT_LIST:
      row = summary_df[
        (summary_df['sub_dataset'] == fd) &
        (summary_df['input_type']  == input_type)].iloc[0]
      print(f"  {fd:6}  {input_type:5}  {int(row['input_dim']):>4}  "
            f"{row['rmse']:>8.4f}  {row['mae']:>8.4f}  "
            f"{row['nasa_score']:>10.2f}")
    print()

  print("=" * 70)
  print()

  # best per sub-dataset
  print("  Best per sub-dataset (by RMSE):")
  for fd in FD_LIST:
    sub = summary_df[summary_df['sub_dataset'] == fd]
    best = sub.loc[sub['rmse'].idxmin()]
    print(f"    {fd}: {best['input_type']:5}  "
          f"RMSE={best['rmse']:.4f}")

  print()
  print(f"Output folder: {OUT_DIR}")
  print("Files written:")
  for fd in FD_LIST:
    for input_type in INPUT_LIST:
      label = f'{fd}_{input_type}'
      print(f"  {label}/model.pt  predictions.csv  history.csv")
  print(f"  summary.csv")


raise SystemExit(main())

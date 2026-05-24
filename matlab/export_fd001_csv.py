# export_fd001_csv.py
#
# Exports C-MAPSS FD001 data from HDF5 to CSV format for MATLAB.
#
# Input:  cmapss_FD001.h5  (built by build_cmapss_complete.py)
# Output: 7 CSV files ready for fd001_run_all.m
#
# CSV files produced:
#   fd001_train_norm.csv       [20631 x 19]  15 normalized sensors + meta
#   fd001_test_norm.csv        [13096 x 17]  15 normalized sensors + meta
#   fd001_pca_train.csv        [20631 x  5]  unit, cycle, cond, PC1, PC2
#   fd001_pca_test.csv         [13096 x  5]  unit, cycle, cond, PC1, PC2
#   fd001_rul.csv              [  100 x  4]  true RUL answer key
#   fd001_pca_variance.csv     [   15 x  2]  explained variance per component
#   fd001_train_raw_subset.csv [20631 x  8]  raw sensors for degradation plot
#
# Full pipeline:
#   raw .txt
#     → build_cmapss_complete.py  → cmapss_FD001.h5
#     → export_fd001_csv.py       → *.csv
#     → fd001_run_all.m           → MATLAB results + figures

import os
import numpy as np
import pandas as pd
import h5py


# ── config ────────────────────────────────────────────────────────────────────

H5_PATH = 'cmapss_FD001.h5'   # adjust path if needed
OUT_DIR = '.'                  # write CSVs to same folder by default


# ── loader ────────────────────────────────────────────────────────────────────

class Fd001Exporter:

  def __init__(self, h5_path, out_dir):
    self.h5_path = h5_path
    self.out_dir = out_dir

  def calc_export(self):
    print(f"Reading: {self.h5_path}")
    print()

    with h5py.File(self.h5_path, 'r') as f:
      self._export_train_norm(f)
      self._export_test_norm(f)
      self._export_pca_train(f)
      self._export_pca_test(f)
      self._export_rul(f)
      self._export_pca_variance(f)
      self._export_train_raw_subset(f)

    print()
    print("All CSV files written. Ready for MATLAB.")
    print(f"Output folder: {os.path.abspath(self.out_dir)}")

  def _save(self, df, filename):
    path = os.path.join(self.out_dir, filename)
    df.to_csv(path, index=False)
    print(f"  {filename:40s}  {df.shape[0]:>6} rows x {df.shape[1]:>2} cols")

  def _export_train_norm(self, f):
    norm_cols = list(f['/train/data_norm'].attrs['columns'])
    train_cols = list(f['/train/data'].attrs['columns'])

    norm  = f['/train/data_norm'][:]
    train = f['/train/data'][:]

    df = pd.DataFrame(norm, columns=norm_cols)

    # attach meta columns
    df['unit_number'] = train[:, train_cols.index('unit_number')]
    df['time_cycles'] = train[:, train_cols.index('time_cycles')]
    df['RUL_raw']     = train[:, train_cols.index('RUL_raw')]
    df['RUL_capped']  = train[:, train_cols.index('RUL_capped')]

    self._save(df, 'fd001_train_norm.csv')

  def _export_test_norm(self, f):
    norm_cols = list(f['/test/data_norm'].attrs['columns'])
    pca_cols  = list(f['/pca/test'].attrs['columns'])

    norm = f['/test/data_norm'][:]
    pca  = f['/pca/test'][:]

    df = pd.DataFrame(norm, columns=norm_cols)
    df['unit_number'] = pca[:, pca_cols.index('unit_number')]
    df['time_cycles'] = pca[:, pca_cols.index('time_cycles')]

    self._save(df, 'fd001_test_norm.csv')

  def _export_pca_train(self, f):
    cols = list(f['/pca/train'].attrs['columns'])
    df   = pd.DataFrame(f['/pca/train'][:], columns=cols)
    self._save(df, 'fd001_pca_train.csv')

  def _export_pca_test(self, f):
    cols = list(f['/pca/test'].attrs['columns'])
    df   = pd.DataFrame(f['/pca/test'][:], columns=cols)
    self._save(df, 'fd001_pca_test.csv')

  def _export_rul(self, f):
    cols = list(f['/RUL/data'].attrs['columns'])
    df   = pd.DataFrame(f['/RUL/data'][:], columns=cols)
    df   = df.astype({'unit_number': int, 'rul': int,
                      'last_cycle': int, 'total_life_est': int})
    self._save(df, 'fd001_rul.csv')

  def _export_pca_variance(self, f):
    all_ev = f['/pca/params/all_explained_var'][:]
    df = pd.DataFrame({
      'component':   range(1, len(all_ev) + 1),
      'explained_var': all_ev,
    })
    self._save(df, 'fd001_pca_variance.csv')

  def _export_train_raw_subset(self, f):
    # raw sensor subset used for degradation trajectory plot
    cols  = list(f['/train/data'].attrs['columns'])
    train = f['/train/data'][:]

    keep = ['unit_number', 'time_cycles',
            'RUL_raw', 'RUL_capped',
            'Ps30', 'T50', 'phi', 'P30']

    idx = [cols.index(c) for c in keep]
    df  = pd.DataFrame(train[:, idx], columns=keep)
    df['unit_number'] = df['unit_number'].astype(int)
    df['time_cycles'] = df['time_cycles'].astype(int)
    df['RUL_raw']     = df['RUL_raw'].astype(int)
    df['RUL_capped']  = df['RUL_capped'].astype(int)

    self._save(df, 'fd001_train_raw_subset.csv')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
  print("FD001 CSV Exporter")
  print(f"Input:  {H5_PATH}")
  print(f"Output: {os.path.abspath(OUT_DIR)}")
  print()

  if not os.path.exists(H5_PATH):
    print(f"ERROR: {H5_PATH} not found.")
    print("Run build_cmapss_complete.py first to build the HDF5.")
    return 1

  os.makedirs(OUT_DIR, exist_ok=True)

  exporter = Fd001Exporter(H5_PATH, OUT_DIR)
  exporter.calc_export()


raise SystemExit(main())

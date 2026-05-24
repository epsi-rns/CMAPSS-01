# build_cmapss_corr.py
#
# Adds correlation and engine_summary to cmapss_norm_FDxxx.h5
# for all four C-MAPSS sub-datasets.
#
# Groups added:
#
#   /correlation
#     /sensors       string   (n_sensors,)   sensor names
#     /matrix        float32  (n_sensors, n_sensors)   Pearson r
#
#   /engine_summary
#     /data          float32  (n_train_engines, 11)
#     /columns       string   (11,)
#
# engine_summary columns (one row per train engine):
#   unit_number, total_life, n_conditions, dominant_cond,
#   T30_mean_raw, T30_mean_norm,
#   phi_mean_raw, phi_mean_norm,
#   RUL_at_50pct, RUL_at_75pct,
#   RUL_test
#
# Correlation: Pearson on train data_norm (normalized sensors).
#   All 21 sensors included. Constant sensors produce NaN.
#
# engine_summary: one row per TRAIN engine.
#   RUL_test is from RUL file — blank (NaN) for train engines
#   since train and test engines are different instances.
#
# Reference: Saxena et al., PHM08

import os
import numpy as np
import pandas as pd
import h5py
from sklearn.cluster import KMeans


# ── constants ─────────────────────────────────────────────────────────────────

RAW_COLUMNS = [
  'unit_number', 'time_cycles',
  'op_setting_1', 'op_setting_2', 'op_setting_3',
  'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30',
  'Nf', 'Nc', 'epr', 'Ps30', 'phi',
  'NRf', 'NRc', 'BPR', 'farB', 'htBleed',
  'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32',
]

ALL_SENSORS = RAW_COLUMNS[5:]   # 21 sensors

ENGINE_SUMMARY_COLS = [
  'unit_number', 'total_life', 'n_conditions', 'dominant_cond',
  'T30_mean_raw', 'T30_mean_norm',
  'phi_mean_raw', 'phi_mean_norm',
  'RUL_at_50pct', 'RUL_at_75pct',
  'RUL_test',
]

N_CONDITIONS = {'FD001': 1, 'FD002': 6, 'FD003': 1, 'FD004': 6}
CAP          = 125
DATA_DIR     = '/home/claude/CMAPSSData'
OUT_DIR      = '/home/claude'


# ── loader ────────────────────────────────────────────────────────────────────

class CmapssLoader:

  def __init__(self, data_dir):
    self.data_dir = data_dir

  def calc_train(self, fd):
    return self._read(f'train_{fd}.txt')

  def calc_rul(self, fd):
    path = os.path.join(self.data_dir, f'RUL_{fd}.txt')
    rul  = pd.read_csv(path, sep=r'\s+', header=None, names=['rul'])
    rul.insert(0, 'unit_number', range(1, len(rul) + 1))
    return rul

  def _read(self, filename):
    path = os.path.join(self.data_dir, filename)
    return pd.read_csv(path, sep=r'\s+', header=None, names=RAW_COLUMNS)


# ── condition + RUL assigner ──────────────────────────────────────────────────

class TrainPreparer:

  def calc_prepare(self, train_df, n_conds):
    df = train_df.copy()

    # condition_id
    if n_conds == 1:
      df['condition_id'] = 1
    else:
      ops = df[['op_setting_1', 'op_setting_2', 'op_setting_3']].values
      km  = KMeans(n_clusters=6, random_state=42, n_init=10)
      km.fit(ops)
      df['condition_id'] = km.predict(ops).astype(int) + 1

    # RUL_raw
    t_max        = df.groupby('unit_number')['time_cycles'].transform('max')
    df['RUL_raw'] = (t_max - df['time_cycles']).astype(int)

    return df


# ── correlation calculator ────────────────────────────────────────────────────

class CorrelationCalculator:

  def calc_matrix(self, norm_array, norm_sensors, all_sensors):
    """
    Compute Pearson correlation on normalized sensors.
    Returns full (21, 21) matrix with NaN for constant/dropped sensors.
    """
    n   = len(all_sensors)
    mat = np.full((n, n), np.nan, dtype=np.float32)

    # build lookup: sensor name -> column index in norm_array
    norm_idx = {s: i for i, s in enumerate(norm_sensors)}

    for i, s1 in enumerate(all_sensors):
      for j, s2 in enumerate(all_sensors):
        if s1 not in norm_idx or s2 not in norm_idx:
          continue
        x = norm_array[:, norm_idx[s1]].astype(np.float64)
        y = norm_array[:, norm_idx[s2]].astype(np.float64)
        if x.std() < 1e-10 or y.std() < 1e-10:
          continue
        r = float(np.corrcoef(x, y)[0, 1])
        mat[i, j] = r

    return mat


# ── engine summary calculator ─────────────────────────────────────────────────

class EngineSummaryCalculator:

  def calc_summary(self, train_df, norm_array, norm_sensors, rul_df):
    """
    One row per train engine.
    norm_array rows align with train_df rows.
    """
    norm_idx = {s: i for i, s in enumerate(norm_sensors)}
    units    = sorted(train_df['unit_number'].unique())
    rows     = []

    # build rul lookup: unit_number -> rul value
    rul_lookup = dict(zip(rul_df['unit_number'], rul_df['rul']))

    for unit in units:
      mask = (train_df['unit_number'] == unit).values
      ut   = train_df[mask]
      un   = norm_array[mask]

      total_life   = int(ut['time_cycles'].max())
      n_cond       = int(ut['condition_id'].nunique())
      dom_cond     = int(ut['condition_id'].mode().iloc[0])

      # raw means
      t30_raw  = float(ut['T30'].mean())
      phi_raw  = float(ut['phi'].mean())

      # normalized means
      t30_norm = float(un[:, norm_idx['T30']].mean()) \
                 if 'T30' in norm_idx else np.nan
      phi_norm = float(un[:, norm_idx['phi']].mean()) \
                 if 'phi' in norm_idx else np.nan

      # RUL at 50% and 75% of total life
      t50  = total_life * 0.50
      t75  = total_life * 0.75
      r50  = ut.loc[ut['time_cycles'] <= t50, 'RUL_raw'].min()
      r75  = ut.loc[ut['time_cycles'] <= t75, 'RUL_raw'].min()
      r50  = float(r50) if not pd.isna(r50) else np.nan
      r75  = float(r75) if not pd.isna(r75) else np.nan

      # RUL from test file — NaN for train engines
      # (train and test are different engine instances)
      rul_test = float(rul_lookup[unit]) \
                 if unit in rul_lookup else np.nan

      rows.append([
        float(unit), float(total_life),
        float(n_cond), float(dom_cond),
        t30_raw, t30_norm,
        phi_raw, phi_norm,
        r50, r75,
        rul_test,
      ])

    return np.array(rows, dtype=np.float32)


# ── builder ───────────────────────────────────────────────────────────────────

class CmapssCorrelationBuilder:

  def __init__(self, loader, preparer, corr_calc, eng_calc, out_dir):
    self.loader    = loader
    self.preparer  = preparer
    self.corr_calc = corr_calc
    self.eng_calc  = eng_calc
    self.out_dir   = out_dir

  def build_file(self, fd):
    path = os.path.join(self.out_dir, f'cmapss_norm_{fd}.h5')

    train_df = self.loader.calc_train(fd)
    rul_df   = self.loader.calc_rul(fd)
    n_conds  = N_CONDITIONS[fd]

    train_df = self.preparer.calc_prepare(train_df, n_conds)

    # read norm_array and norm_sensors from existing HDF5
    with h5py.File(path, 'r') as f:
      norm_array   = f['/train/data_norm'][:]
      norm_sensors = [s.decode() if isinstance(s, bytes) else s
                      for s in f['/train/data_norm'].attrs['columns']]

    # correlation — all 21 sensors, NaN for dropped
    corr_mat = self.corr_calc.calc_matrix(
      norm_array, norm_sensors, ALL_SENSORS)

    # engine summary
    eng_data = self.eng_calc.calc_summary(
      train_df, norm_array, norm_sensors, rul_df)

    with h5py.File(path, 'a') as f:
      self._build_correlation(f, corr_mat)
      self._build_engine_summary(f, eng_data)

    return path, corr_mat.shape, eng_data.shape

  def _build_correlation(self, f, mat):
    if 'correlation' in f:
      del f['correlation']
    grp = f.create_group('correlation')

    grp.create_dataset('sensors',
                       data=np.array(ALL_SENSORS, dtype='S'))
    grp.create_dataset('matrix', data=mat,
                       compression='gzip', compression_opts=4)

    grp.attrs['method']     = 'Pearson'
    grp.attrs['source']     = '/train/data_norm'
    grp.attrs['n_sensors']  = len(ALL_SENSORS)
    grp.attrs['shape_note'] = (
      f'({len(ALL_SENSORS)}, {len(ALL_SENSORS)}) — '
      'NaN for constant/dropped sensors')

  def _build_engine_summary(self, f, data):
    if 'engine_summary' in f:
      del f['engine_summary']
    grp = f.create_group('engine_summary')

    grp.create_dataset('data', data=data,
                       compression='gzip', compression_opts=4)
    grp.create_dataset('columns',
                       data=np.array(ENGINE_SUMMARY_COLS, dtype='S'))

    grp.attrs['n_engines']  = data.shape[0]
    grp.attrs['n_cols']     = data.shape[1]
    grp.attrs['shape_note'] = (
      f'({data.shape[0]}, {data.shape[1]}) — one row per train engine')
    grp.attrs['RUL_note']   = (
      'RUL_test is NaN for all train engines. '
      'Train and test engines are different instances.')
    grp.attrs['RUL_pct']    = '50pct and 75pct of total_life'


# ── verifier ──────────────────────────────────────────────────────────────────

class CmapssVerifier:

  def verify_file(self, path):
    errors = []
    with h5py.File(path, 'r') as f:

      # correlation
      if '/correlation/matrix' not in f:
        errors.append('missing /correlation/matrix')
      else:
        s = f['/correlation/matrix'].shape
        if s != (21, 21):
          errors.append(f'/correlation/matrix shape={s}, expected (21,21)')

      if '/correlation/sensors' not in f:
        errors.append('missing /correlation/sensors')
      else:
        if len(f['/correlation/sensors']) != 21:
          errors.append('/correlation/sensors length != 21')

      # engine_summary
      if '/engine_summary/data' not in f:
        errors.append('missing /engine_summary/data')
      else:
        s = f['/engine_summary/data'].shape
        if s[1] != 11:
          errors.append(f'/engine_summary/data cols={s[1]}, expected 11')

      if '/engine_summary/columns' not in f:
        errors.append('missing /engine_summary/columns')
      else:
        if len(f['/engine_summary/columns']) != 11:
          errors.append('/engine_summary/columns length != 11')

      # spot check: diagonal of correlation matrix should be 1.0 or NaN
      if '/correlation/matrix' in f:
        mat  = f['/correlation/matrix'][:]
        diag = np.diag(mat)
        bad  = [ALL_SENSORS[i] for i, v in enumerate(diag)
                if not (np.isnan(v) or abs(v - 1.0) < 1e-4)]
        if bad:
          errors.append(f'diagonal not 1.0 or NaN for: {bad}')

    return errors

  def print_summary(self, path, fd):
    with h5py.File(path, 'r') as f:
      mat      = f['/correlation/matrix'][:]
      eng_data = f['/engine_summary/data'][:]
      eng_cols = [c.decode() for c in f['/engine_summary/columns'][:]]

    # count valid correlations
    valid = np.sum(~np.isnan(mat)) - np.sum(~np.isnan(np.diag(mat)))
    print(f"  {fd}  correlation: {mat.shape}  "
          f"valid pairs={valid//2}  "
          f"engine_summary: {eng_data.shape}")

    # show first 3 engine rows
    eng_df = pd.DataFrame(eng_data, columns=eng_cols)
    print(f"  first 3 engines:")
    print(eng_df.head(3).to_string(index=False))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
  loader   = CmapssLoader(DATA_DIR)
  preparer = TrainPreparer()
  corr_calc = CorrelationCalculator()
  eng_calc  = EngineSummaryCalculator()
  builder   = CmapssCorrelationBuilder(
    loader, preparer, corr_calc, eng_calc, OUT_DIR)
  verifier  = CmapssVerifier()

  print(f"{'sub':6}  {'corr shape':>12}  {'eng shape':>12}  status")
  print('-' * 50)

  for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    path, corr_shape, eng_shape = builder.build_file(fd)
    errors = verifier.verify_file(path)
    status = 'OK' if not errors else 'ERR: ' + ', '.join(errors)
    print(f"{fd:6}  {str(corr_shape):>12}  {str(eng_shape):>12}  {status}")

  print()
  for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    path = os.path.join(OUT_DIR, f'cmapss_norm_{fd}.h5')
    print()
    verifier.print_summary(path, fd)

  print()
  print("Full structure (FD001):")
  with h5py.File(os.path.join(OUT_DIR, 'cmapss_norm_FD001.h5'), 'r') as f:
    f.visit(lambda name: print(f"  /{name}"))

  print()
  print("Files updated:")
  for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    p = os.path.join(OUT_DIR, f'cmapss_norm_{fd}.h5')
    kb = os.path.getsize(p) // 1024
    print(f"  cmapss_norm_{fd}.h5  ({kb} KB)")


raise SystemExit(main())

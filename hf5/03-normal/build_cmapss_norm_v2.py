# build_cmapss_norm_v2.py
#
# Rebuilds normalized HDF5 for all four C-MAPSS sub-datasets.
# Replaces cmapss_norm_FDxxx.h5 (previously global z-score).
#
# Normalization method (matches build_xlsx_v5.py):
#   - Fit on HEALTHY cycles only: RUL_raw >= CAP (125)
#   - Per condition per sensor: separate mean/std per (condition_id, sensor)
#   - If std == 0 for a condition: replace with 1.0 (not dropped)
#   - Applied to all cycles (train and test) using healthy-fitted params
#
# Drop list (std=0 across ALL conditions in that sub-dataset):
#   FD001: T2, P2, epr, farB, Nf_dmd, PCNfR_dmd  → 15 sensors
#   FD002: (none)                                   → 21 sensors
#   FD003: T2, P2, farB, Nf_dmd, PCNfR_dmd        → 16 sensors
#   FD004: (none)                                   → 21 sensors
#
# Structure added/replaced per file:
#   /train/data_norm       float32  (n_train_rows, n_sensors)
#   /test/data_norm        float32  (n_test_rows,  n_sensors)
#   /norm_params/sensors   string   (n_sensors,)
#   /norm_params/conditions         int32    (n_conds,)
#   /norm_params/raw_mean  float64  (n_conds, n_sensors)
#   /norm_params/raw_std   float64  (n_conds, n_sensors)
#   /norm_params/raw_min   float64  (n_conds, n_sensors)
#   /norm_params/raw_max   float64  (n_conds, n_sensors)
#   /norm_params/norm_mean float64  (n_conds, n_sensors)
#   /norm_params/norm_std  float64  (n_conds, n_sensors)
#   /norm_params/norm_min  float64  (n_conds, n_sensors)
#   /norm_params/norm_max  float64  (n_conds, n_sensors)
#
# Note: for FD001/FD003 (1 condition) shape is (1, n_sensors).
#       for FD002/FD004 (6 conditions) shape is (6, n_sensors).
#
# Reference: Saxena et al., PHM08

import os
import shutil
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

ALL_SENSORS = [
  c for c in RAW_COLUMNS
  if c not in ['unit_number', 'time_cycles',
               'op_setting_1', 'op_setting_2', 'op_setting_3']
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

  def calc_test(self, fd):
    return self._read(f'test_{fd}.txt')

  def _read(self, filename):
    path = os.path.join(self.data_dir, filename)
    return pd.read_csv(path, sep=r'\s+', header=None, names=RAW_COLUMNS)


# ── condition assigner ────────────────────────────────────────────────────────

class ConditionAssigner:

  def calc_assign(self, train_df, test_df, n_conds):
    if n_conds == 1:
      train_df = train_df.copy()
      test_df  = test_df.copy()
      train_df['condition_id'] = 1
      test_df['condition_id']  = 1
      return train_df, test_df, None

    ops_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    km       = KMeans(n_clusters=6, random_state=42, n_init=10)
    km.fit(train_df[ops_cols].values)

    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df['condition_id'] = km.predict(
      train_df[ops_cols].values).astype(int) + 1
    test_df['condition_id']  = km.predict(
      test_df[ops_cols].values).astype(int) + 1
    return train_df, test_df, km


# ── RUL computer ──────────────────────────────────────────────────────────────

class RulComputer:

  def calc_rul_raw(self, train_df):
    train_df = train_df.copy()
    t_max = train_df.groupby('unit_number')['time_cycles'].transform('max')
    train_df['RUL_raw'] = (t_max - train_df['time_cycles']).astype(int)
    return train_df


# ── drop calculator ───────────────────────────────────────────────────────────

class DropCalculator:

  def calc_drop_list(self, train_df):
    drop = [c for c in ALL_SENSORS if train_df[c].std() < 1e-6]
    keep = [c for c in ALL_SENSORS if train_df[c].std() >= 1e-6]
    return drop, keep


# ── normalizer ────────────────────────────────────────────────────────────────

class HealthyNormalizer:
  """
  Fits z-score parameters on healthy cycles (RUL_raw >= CAP)
  per condition per sensor.  Matches build_xlsx_v5.py logic.
  """

  def __init__(self):
    self.sensors    = None
    self.conditions = None
    self.params     = {}   # (cond, sensor) -> (mu, sigma)

  def calc_fit(self, train_df, sensors):
    self.sensors    = sensors
    self.conditions = sorted(train_df['condition_id'].unique())
    healthy         = train_df[train_df['RUL_raw'] >= CAP]

    for cond in self.conditions:
      subset = healthy[healthy['condition_id'] == cond]
      for s in sensors:
        mu    = float(subset[s].mean())
        sigma = float(subset[s].std())
        if pd.isna(sigma) or sigma < 1e-10:
          sigma = 1.0
        self.params[(cond, s)] = (mu, sigma)

  def calc_transform(self, df):
    n_rows = len(df)
    n_cols = len(self.sensors)
    result = np.zeros((n_rows, n_cols), dtype=np.float32)

    for j, s in enumerate(self.sensors):
      for i, (_, row) in enumerate(df.iterrows()):
        cond         = int(row['condition_id'])
        mu, sigma    = self.params[(cond, s)]
        result[i, j] = float((row[s] - mu) / sigma)

    return result

  def calc_transform_fast(self, df):
    """Vectorised version — much faster than row-by-row."""
    n_rows = len(df)
    n_cols = len(self.sensors)
    result = np.zeros((n_rows, n_cols), dtype=np.float64)

    for cond in self.conditions:
      mask = (df['condition_id'] == cond).values
      if not mask.any():
        continue
      for j, s in enumerate(self.sensors):
        mu, sigma       = self.params[(cond, s)]
        result[mask, j] = (df.loc[mask, s].values - mu) / sigma

    return result.astype(np.float32)

  def calc_param_arrays(self):
    """Return (n_conds, n_sensors) arrays for storage."""
    nc  = len(self.conditions)
    ns  = len(self.sensors)
    raw_mean = np.zeros((nc, ns))
    raw_std  = np.zeros((nc, ns))
    raw_min  = np.zeros((nc, ns))
    raw_max  = np.zeros((nc, ns))

    for i, cond in enumerate(self.conditions):
      for j, s in enumerate(self.sensors):
        mu, sigma       = self.params[(cond, s)]
        raw_mean[i, j]  = mu
        raw_std[i, j]   = sigma

    return raw_mean, raw_std

  def calc_norm_stats(self, norm_array, train_df):
    """Compute norm min/max/mean/std per condition per sensor."""
    nc  = len(self.conditions)
    ns  = len(self.sensors)
    norm_mean = np.zeros((nc, ns))
    norm_std  = np.zeros((nc, ns))
    norm_min  = np.zeros((nc, ns))
    norm_max  = np.zeros((nc, ns))

    for i, cond in enumerate(self.conditions):
      mask = (train_df['condition_id'] == cond).values
      if not mask.any():
        continue
      block = norm_array[mask]
      norm_mean[i] = block.mean(axis=0)
      norm_std[i]  = block.std(axis=0, ddof=1)
      norm_min[i]  = block.min(axis=0)
      norm_max[i]  = block.max(axis=0)

    return norm_mean, norm_std, norm_min, norm_max

  def calc_raw_minmax(self, train_df):
    """Raw min/max per condition per sensor from full train set."""
    nc      = len(self.conditions)
    ns      = len(self.sensors)
    raw_min = np.zeros((nc, ns))
    raw_max = np.zeros((nc, ns))

    for i, cond in enumerate(self.conditions):
      mask = (train_df['condition_id'] == cond).values
      if not mask.any():
        continue
      for j, s in enumerate(self.sensors):
        raw_min[i, j] = train_df.loc[mask, s].min()
        raw_max[i, j] = train_df.loc[mask, s].max()

    return raw_min, raw_max


# ── builder ───────────────────────────────────────────────────────────────────

class CmapssNormBuilder:

  def __init__(self, loader, cond_assigner, rul_computer,
               drop_calc, normalizer, out_dir):
    self.loader        = loader
    self.cond_assigner = cond_assigner
    self.rul_computer  = rul_computer
    self.drop_calc     = drop_calc
    self.normalizer    = normalizer
    self.out_dir       = out_dir

  def build_file(self, fd):
    src = os.path.join(self.out_dir, f'cmapss_basic_{fd}.h5')
    dst = os.path.join(self.out_dir, f'cmapss_norm_{fd}.h5')
    shutil.copy2(src, dst)

    # load and prepare
    train_df = self.loader.calc_train(fd)
    test_df  = self.loader.calc_test(fd)
    n_conds  = N_CONDITIONS[fd]

    train_df, test_df, _ = self.cond_assigner.calc_assign(
      train_df, test_df, n_conds)
    train_df = self.rul_computer.calc_rul_raw(train_df)

    drop, keep = self.drop_calc.calc_drop_list(train_df)

    # fit on healthy cycles, per condition
    self.normalizer.calc_fit(train_df, keep)

    # transform
    train_norm = self.normalizer.calc_transform_fast(train_df)
    test_norm  = self.normalizer.calc_transform_fast(test_df)

    # param arrays
    raw_mean, raw_std       = self.normalizer.calc_param_arrays()
    raw_min, raw_max        = self.normalizer.calc_raw_minmax(train_df)
    norm_mean, norm_std, \
    norm_min, norm_max      = self.normalizer.calc_norm_stats(
                                train_norm, train_df)

    with h5py.File(dst, 'a') as f:
      self._build_train_norm(f, train_norm, keep, drop)
      self._build_test_norm(f, test_norm, keep, drop)
      self._build_norm_params(
        f, keep, drop,
        raw_mean, raw_std, raw_min, raw_max,
        norm_mean, norm_std, norm_min, norm_max)

      f.attrs['version']   = 'norm_v2'
      f.attrs['norm_note'] = (
        'z-score normalization v2. '
        f'Fitted on healthy cycles only (RUL_raw >= {CAP}). '
        'Per condition per sensor. '
        'std=0 replaced with 1.0. '
        'Applied to all cycles using healthy-fitted params.'
      )

    size_kb = os.path.getsize(dst) // 1024
    return dst, size_kb, train_norm.shape, test_norm.shape, drop, keep

  def _build_train_norm(self, f, data, keep, drop):
    if 'data_norm' in f['train']:
      del f['train/data_norm']
    ds = f['train'].create_dataset(
      'data_norm', data=data,
      compression='gzip', compression_opts=4)
    ds.attrs['columns']    = keep
    ds.attrs['dtype']      = 'float32'
    ds.attrs['n_sensors']  = len(keep)
    ds.attrs['shape_note'] = (
      f'(n_rows, {len(keep)}) — z-scored, healthy-fitted, per condition')
    ds.attrs['dropped']    = sorted(drop)

  def _build_test_norm(self, f, data, keep, drop):
    if 'data_norm' in f['test']:
      del f['test/data_norm']
    ds = f['test'].create_dataset(
      'data_norm', data=data,
      compression='gzip', compression_opts=4)
    ds.attrs['columns']    = keep
    ds.attrs['dtype']      = 'float32'
    ds.attrs['n_sensors']  = len(keep)
    ds.attrs['shape_note'] = (
      f'(n_rows, {len(keep)}) — z-scored using TRAIN healthy params')
    ds.attrs['dropped']    = sorted(drop)

  def _build_norm_params(self, f,
                         keep, drop,
                         raw_mean, raw_std,
                         raw_min,  raw_max,
                         norm_mean, norm_std,
                         norm_min,  norm_max):
    if 'norm_params' in f:
      del f['norm_params']
    grp = f.create_group('norm_params')

    conditions = self.normalizer.conditions
    grp.create_dataset('sensors',
                       data=np.array(keep, dtype='S'))
    grp.create_dataset('conditions',
                       data=np.array(conditions, dtype=np.int32))
    grp.create_dataset('raw_mean',  data=raw_mean)
    grp.create_dataset('raw_std',   data=raw_std)
    grp.create_dataset('raw_min',   data=raw_min)
    grp.create_dataset('raw_max',   data=raw_max)
    grp.create_dataset('norm_mean', data=norm_mean)
    grp.create_dataset('norm_std',  data=norm_std)
    grp.create_dataset('norm_min',  data=norm_min)
    grp.create_dataset('norm_max',  data=norm_max)

    grp.attrs['n_sensors']    = len(keep)
    grp.attrs['n_conditions'] = len(conditions)
    grp.attrs['method']       = 'z-score'
    grp.attrs['fit_on']       = f'healthy cycles only (RUL_raw >= {CAP})'
    grp.attrs['formula']      = 'z = (x - raw_mean) / raw_std'
    grp.attrs['shape_note']   = (
      '(n_conditions, n_sensors) — row=condition, col=sensor')
    grp.attrs['dropped']      = sorted(drop)
    grp.attrs['cap']          = CAP


# ── verifier ──────────────────────────────────────────────────────────────────

class CmapssVerifier:

  def verify_file(self, path, n_sensors, n_conds):
    errors = []
    with h5py.File(path, 'r') as f:

      for ds_path, exp_cols in [
        ('/train/data_norm', n_sensors),
        ('/test/data_norm',  n_sensors),
      ]:
        if ds_path not in f:
          errors.append(f'missing {ds_path}')
        elif f[ds_path].shape[1] != exp_cols:
          errors.append(
            f'{ds_path} cols={f[ds_path].shape[1]}, expected {exp_cols}')

      if '/norm_params' not in f:
        errors.append('missing /norm_params')
      else:
        for ds in ['sensors', 'conditions',
                   'raw_mean', 'raw_std', 'raw_min', 'raw_max',
                   'norm_mean', 'norm_std', 'norm_min', 'norm_max']:
          key = f'/norm_params/{ds}'
          if key not in f:
            errors.append(f'missing {key}')
          else:
            shape = f[key].shape
            if ds in ['sensors', 'conditions']:
              expected = n_sensors if ds == 'sensors' else n_conds
              if shape[0] != expected:
                errors.append(f'{key} length={shape[0]}, '
                               f'expected {expected}')
            else:
              if shape != (n_conds, n_sensors):
                errors.append(f'{key} shape={shape}, '
                               f'expected ({n_conds},{n_sensors})')

    return errors


# ── main ──────────────────────────────────────────────────────────────────────

def main():
  loader        = CmapssLoader(DATA_DIR)
  cond_assigner = ConditionAssigner()
  rul_computer  = RulComputer()
  drop_calc     = DropCalculator()
  normalizer    = HealthyNormalizer()
  builder       = CmapssNormBuilder(
    loader, cond_assigner, rul_computer,
    drop_calc, normalizer, OUT_DIR)
  verifier      = CmapssVerifier()

  print(f"{'sub':6}  {'conds':>5}  {'drop':>5}  {'keep':>5}  "
        f"{'train shape':>14}  {'test shape':>13}  {'KB':>6}  status")
  print('-' * 78)

  for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    dst, size_kb, tr_shape, te_shape, drop, keep = builder.build_file(fd)
    n_conds = N_CONDITIONS[fd]
    errors  = verifier.verify_file(dst, len(keep), n_conds)
    status  = 'OK' if not errors else 'ERR: ' + ', '.join(errors)
    print(
      f"{fd:6}  {n_conds:>5}  {len(drop):>5}  {len(keep):>5}  "
      f"{str(tr_shape):>14}  {str(te_shape):>13}  "
      f"{size_kb:>6}  {status}")

  print()
  print("Drop list per sub-dataset:")
  for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    loader2   = CmapssLoader(DATA_DIR)
    drop_calc2 = DropCalculator()
    train_df  = loader2.calc_train(fd)
    drop, _   = drop_calc2.calc_drop_list(train_df)
    print(f"  {fd}: {sorted(drop) if drop else '(none)'}")

  print()
  print("norm_params shape per sub-dataset:")
  for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    path = os.path.join(OUT_DIR, f'cmapss_norm_{fd}.h5')
    with h5py.File(path, 'r') as f:
      shape = f['/norm_params/raw_mean'].shape
      conds = list(f['/norm_params/conditions'][:])
      sens  = [s.decode() for s in f['/norm_params/sensors'][:]]
      print(f"  {fd}: raw_mean shape={shape}  "
            f"conditions={conds}  sensors={sens}")

  print()
  print("Files written:")
  for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    p = os.path.join(OUT_DIR, f'cmapss_norm_{fd}.h5')
    print(f"  {p}")


raise SystemExit(main())

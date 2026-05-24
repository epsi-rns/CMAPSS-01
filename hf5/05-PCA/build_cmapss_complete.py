# build_cmapss_complete.py
#
# Builds complete C-MAPSS HDF5 from raw NASA txt files in one pass.
# No intermediate files required.
#
# Input:  CMAPSSData/train_FDxxx.txt, test_FDxxx.txt, RUL_FDxxx.txt
# Output: cmapss_FDxxx.h5  (one file per sub-dataset)
#
# Structure per file:
#
#   /train/data              float32  (n_rows, 32)   26 raw + 6 computed
#   /train/data_norm         float32  (n_rows, n_s)  z-scored sensors
#   /test/data               float32  (n_rows, 29)   26 raw + 3 computed
#   /test/data_norm          float32  (n_rows, n_s)  z-scored sensors
#   /RUL/data                int32    (n_eng, 4)      unit,rul,last,total
#   /norm_params/...         float64  (n_conds, n_s)  normalization params
#   /correlation/matrix      float32  (21, 21)        Pearson matrix
#   /correlation/sensors     string   (21,)
#   /engine_summary/data     float32  (n_eng, 11)     per-engine stats
#   /engine_summary/columns  string   (11,)
#   /pca/params/...          float64                  eigenvectors + variance
#   /pca/train               float32  (n_rows, 3+k)  meta + PC scores
#   /pca/test                float32  (n_rows, 3+k)  meta + PC scores
#
# Normalization: healthy-cycle z-score (RUL_raw >= 125), per condition.
# PCA: Option B — keep components with individual variance >= 5%.
#
# Sub-dataset properties:
#   FD001  1 condition   HPC fault        drop 6 sensors
#   FD002  6 conditions  HPC fault        drop 0 sensors
#   FD003  1 condition   HPC+Fan fault    drop 5 sensors
#   FD004  6 conditions  HPC+Fan fault    drop 0 sensors
#
# Reference: Saxena et al., PHM08

import os
import numpy as np
import pandas as pd
import h5py
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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

TRAIN_COMPUTED = [
  'condition_id', 'dT_compressor', 'dT_turbine',
  'RUL_raw', 'RUL_capped', 'cycle_pct',
]
TEST_COMPUTED = [
  'condition_id', 'dT_compressor', 'dT_turbine',
]
RUL_COLUMNS = ['unit_number', 'rul', 'last_cycle', 'total_life_est']

ENGINE_SUMMARY_COLS = [
  'unit_number', 'total_life', 'n_conditions', 'dominant_cond',
  'T30_mean_raw', 'T30_mean_norm',
  'phi_mean_raw', 'phi_mean_norm',
  'RUL_at_50pct', 'RUL_at_75pct',
  'RUL_test',
]

N_CONDITIONS = {'FD001': 1, 'FD002': 6, 'FD003': 1, 'FD004': 6}
FAULT_MODES  = {
  'FD001': 'HPC', 'FD002': 'HPC',
  'FD003': 'HPC+Fan', 'FD004': 'HPC+Fan',
}
CAP              = 125
RUL_CAP          = 125
VARIANCE_THRESH  = 0.05
REFERENCE        = 'Saxena et al., PHM08 — C-MAPSS dataset.'


# ── loader ────────────────────────────────────────────────────────────────────

class CmapssLoader:

  def __init__(self, data_dir):
    self.data_dir = data_dir

  def calc_load(self, fd):
    train = self._read(f'train_{fd}.txt')
    test  = self._read(f'test_{fd}.txt')
    rul   = pd.read_csv(
      os.path.join(self.data_dir, f'RUL_{fd}.txt'),
      sep=r'\s+', header=None, names=['rul'])
    rul.insert(0, 'unit_number', range(1, len(rul) + 1))
    return train, test, rul

  def _read(self, filename):
    path = os.path.join(self.data_dir, filename)
    return pd.read_csv(
      path, sep=r'\s+', header=None, names=RAW_COLUMNS)


# ── preparer ──────────────────────────────────────────────────────────────────

class CmapssPrep:

  def calc_prepare(self, train, test, rul, n_conds):
    # condition_id
    if n_conds > 1:
      ops = train[['op_setting_1','op_setting_2','op_setting_3']].values
      km  = KMeans(n_clusters=6, random_state=42, n_init=10)
      km.fit(ops)
      train['condition_id'] = km.predict(ops) + 1
      test['condition_id']  = km.predict(
        test[['op_setting_1','op_setting_2','op_setting_3']].values) + 1
    else:
      train['condition_id'] = 1
      test['condition_id']  = 1

    # derived sensor columns
    for df_ in [train, test]:
      df_['dT_compressor'] = (df_['T30'] - df_['T24']).round(4)
      df_['dT_turbine']    = (df_['T50'] - df_['T30']).round(4)

    # RUL columns (train only)
    t_max = train.groupby(
      'unit_number')['time_cycles'].transform('max')
    train['RUL_raw']    = (t_max - train['time_cycles']).astype(int)
    train['RUL_capped'] = train['RUL_raw'].clip(upper=CAP).astype(int)
    train['cycle_pct']  = (train['time_cycles'] / t_max).round(4)

    # RUL sheet derived columns
    last_cycle = test.groupby('unit_number')['time_cycles'].max()
    rul['last_cycle']     = rul['unit_number'].map(last_cycle).astype(int)
    rul['total_life_est'] = rul['last_cycle'] + rul['rul']

    return train, test, rul


# ── normalizer ────────────────────────────────────────────────────────────────

class CmapssNorm:

  def __init__(self):
    self.keep   = None
    self.drop   = None
    self.stats  = None
    self.conds  = None

  def calc_fit(self, train_df):
    self.keep  = [s for s in ALL_SENSORS if train_df[s].std() >= 1e-6]
    self.drop  = [s for s in ALL_SENSORS if train_df[s].std() <  1e-6]
    self.conds = sorted(train_df['condition_id'].unique())

    healthy    = train_df[train_df['RUL_raw'] >= RUL_CAP]
    self.stats = {}
    for cond in self.conds:
      subset = healthy[healthy['condition_id'] == cond]
      for s in self.keep:
        mu    = float(subset[s].mean())
        sigma = float(subset[s].std())
        if pd.isna(sigma) or sigma == 0:
          sigma = 1.0
        self.stats[(cond, s)] = (mu, sigma)

  def calc_transform(self, df):
    norm = df[self.keep].copy().astype(float)
    for cond in self.conds:
      mask = (df['condition_id'] == cond).values
      idx  = df.index[mask]
      for s in self.keep:
        mu, sigma = self.stats[(cond, s)]
        norm.loc[idx, s] = (
          (df.loc[idx, s] - mu) / sigma).round(6)
    return norm.values.astype(np.float32)

  def calc_param_arrays(self, train_df, train_norm):
    nc = len(self.conds)
    ns = len(self.keep)

    raw_mean  = np.zeros((nc, ns))
    raw_std   = np.zeros((nc, ns))
    raw_min   = np.zeros((nc, ns))
    raw_max   = np.zeros((nc, ns))
    norm_mean = np.zeros((nc, ns))
    norm_std  = np.zeros((nc, ns))
    norm_min  = np.zeros((nc, ns))
    norm_max  = np.zeros((nc, ns))

    for i, cond in enumerate(self.conds):
      mask = (train_df['condition_id'] == cond).values
      for j, s in enumerate(self.keep):
        mu, sigma     = self.stats[(cond, s)]
        raw_mean[i,j] = mu
        raw_std[i,j]  = sigma
        raw_min[i,j]  = train_df.loc[train_df.index[mask], s].min()
        raw_max[i,j]  = train_df.loc[train_df.index[mask], s].max()

      block = train_norm[mask]
      norm_mean[i] = block.mean(axis=0)
      norm_std[i]  = block.std(axis=0, ddof=1)
      norm_min[i]  = block.min(axis=0)
      norm_max[i]  = block.max(axis=0)

    return (raw_mean, raw_std, raw_min, raw_max,
            norm_mean, norm_std, norm_min, norm_max)


# ── PCA calculator ────────────────────────────────────────────────────────────

class CmapssPca:

  def __init__(self):
    self.pca        = None
    self.n_keep     = None
    self.components = None
    self.var_ratio  = None
    self.cum_var    = None

  def calc_fit(self, X_train):
    self.pca      = PCA()
    self.pca.fit(X_train)
    self.var_ratio = self.pca.explained_variance_ratio_
    self.cum_var   = np.cumsum(self.var_ratio)
    self.n_keep    = max(
      1, sum(1 for v in self.var_ratio if v >= VARIANCE_THRESH))
    self.components = [f'PC{i+1}' for i in range(self.n_keep)]

  def calc_transform(self, X):
    return self.pca.transform(X)[:, :self.n_keep].astype(np.float32)

  def calc_loadings(self):
    return self.pca.components_[:self.n_keep]


# ── engine summary ────────────────────────────────────────────────────────────

class EngSummaryCalc:

  def calc_summary(self, train_df, norm_array, norm_sensors,
                   rul_df):
    norm_idx  = {s: i for i, s in enumerate(norm_sensors)}
    units     = sorted(train_df['unit_number'].unique())
    rul_lookup = dict(zip(rul_df['unit_number'], rul_df['rul']))
    rows      = []

    for unit in units:
      mask = (train_df['unit_number'] == unit).values
      ut   = train_df[mask]
      un   = norm_array[mask]

      total_life = int(ut['time_cycles'].max())
      n_cond     = int(ut['condition_id'].nunique())
      dom_cond   = int(ut['condition_id'].mode().iloc[0])
      t30_raw    = float(ut['T30'].mean())
      phi_raw    = float(ut['phi'].mean())
      t30_norm   = float(un[:, norm_idx['T30']].mean()) \
                   if 'T30' in norm_idx else np.nan
      phi_norm   = float(un[:, norm_idx['phi']].mean()) \
                   if 'phi' in norm_idx else np.nan

      t50  = total_life * 0.50
      t75  = total_life * 0.75
      r50  = ut.loc[ut['time_cycles'] <= t50, 'RUL_raw'].min()
      r75  = ut.loc[ut['time_cycles'] <= t75, 'RUL_raw'].min()
      r50  = float(r50) if not pd.isna(r50) else np.nan
      r75  = float(r75) if not pd.isna(r75) else np.nan
      rul_test = float(rul_lookup[unit]) \
                 if unit in rul_lookup else np.nan

      rows.append([
        float(unit), float(total_life),
        float(n_cond), float(dom_cond),
        t30_raw, t30_norm, phi_raw, phi_norm,
        r50, r75, rul_test,
      ])

    return np.array(rows, dtype=np.float32)


# ── HDF5 writer ───────────────────────────────────────────────────────────────

class CmapssWriter:

  def build_file(self, fd, path,
                 train_df, test_df, rul_df,
                 norm, pca, eng_calc):

    n_conds  = N_CONDITIONS[fd]
    n_trains = int(train_df['unit_number'].nunique())
    n_tests  = int(test_df['unit_number'].nunique())

    # arrays
    train_cols = RAW_COLUMNS + TRAIN_COMPUTED
    test_cols  = RAW_COLUMNS + TEST_COMPUTED

    train_raw  = train_df[RAW_COLUMNS].values.astype(np.float32)
    train_comp = np.column_stack([
      train_df['condition_id'].values,
      train_df['dT_compressor'].values,
      train_df['dT_turbine'].values,
      train_df['RUL_raw'].values,
      train_df['RUL_capped'].values,
      train_df['cycle_pct'].values,
    ]).astype(np.float32)
    train_arr = np.column_stack([train_raw, train_comp])

    test_raw  = test_df[RAW_COLUMNS].values.astype(np.float32)
    test_comp = np.column_stack([
      test_df['condition_id'].values,
      test_df['dT_compressor'].values,
      test_df['dT_turbine'].values,
    ]).astype(np.float32)
    test_arr = np.column_stack([test_raw, test_comp])

    rul_arr = np.column_stack([
      rul_df['unit_number'].values,
      rul_df['rul'].values,
      rul_df['last_cycle'].values,
      rul_df['total_life_est'].values,
    ]).astype(np.int32)

    # normalization
    norm.calc_fit(train_df)
    train_norm = norm.calc_transform(train_df)
    test_norm  = norm.calc_transform(test_df)
    (raw_mean, raw_std, raw_min, raw_max,
     nm_mean, nm_std, nm_min, nm_max) = norm.calc_param_arrays(
       train_df, train_norm)

    # correlation — all 21 sensors, NaN for dropped
    corr_mat = self._calc_correlation(
      train_norm, norm.keep)

    # engine summary
    eng_data = eng_calc.calc_summary(
      train_df, train_norm, norm.keep, rul_df)

    # PCA
    pca.calc_fit(train_norm)
    train_scores = pca.calc_transform(train_norm)
    test_scores  = pca.calc_transform(test_norm)

    meta_train = train_df[['unit_number','time_cycles',
                            'condition_id']].values.astype(np.float32)
    meta_test  = test_df[['unit_number','time_cycles',
                           'condition_id']].values.astype(np.float32)
    pca_train_arr = np.column_stack([meta_train, train_scores])
    pca_test_arr  = np.column_stack([meta_test,  test_scores])
    pca_cols = (['unit_number','time_cycles','condition_id']
                + pca.components)

    # write
    gz = dict(compression='gzip', compression_opts=4)
    with h5py.File(path, 'w') as f:

      # /train
      grp = f.create_group('train')
      ds  = grp.create_dataset('data', data=train_arr, **gz)
      ds.attrs['columns']    = train_cols
      ds.attrs['dtype']      = 'float32'
      ds.attrs['n_raw']      = len(RAW_COLUMNS)
      ds.attrs['n_computed'] = len(TRAIN_COMPUTED)
      ds.attrs['shape_note'] = f'(n_rows, {len(train_cols)}) — 26 raw + 6 computed'

      ds = grp.create_dataset('data_norm', data=train_norm, **gz)
      ds.attrs['columns']    = norm.keep
      ds.attrs['dtype']      = 'float32'
      ds.attrs['n_sensors']  = len(norm.keep)
      ds.attrs['dropped']    = sorted(norm.drop)
      ds.attrs['shape_note'] = (
        f'(n_rows, {len(norm.keep)}) — z-scored, healthy-fitted, per condition')

      # /test
      grp = f.create_group('test')
      ds  = grp.create_dataset('data', data=test_arr, **gz)
      ds.attrs['columns']    = test_cols
      ds.attrs['dtype']      = 'float32'
      ds.attrs['n_raw']      = len(RAW_COLUMNS)
      ds.attrs['n_computed'] = len(TEST_COMPUTED)
      ds.attrs['shape_note'] = f'(n_rows, {len(test_cols)}) — 26 raw + 3 computed'

      ds = grp.create_dataset('data_norm', data=test_norm, **gz)
      ds.attrs['columns']    = norm.keep
      ds.attrs['dtype']      = 'float32'
      ds.attrs['n_sensors']  = len(norm.keep)
      ds.attrs['dropped']    = sorted(norm.drop)
      ds.attrs['shape_note'] = (
        f'(n_rows, {len(norm.keep)}) — z-scored using TRAIN healthy params')

      # /RUL
      grp = f.create_group('RUL')
      ds  = grp.create_dataset('data', data=rul_arr)
      ds.attrs['columns']    = RUL_COLUMNS
      ds.attrs['dtype']      = 'int32'
      ds.attrs['shape_note'] = (
        '(n_engines, 4) — unit_number, rul, last_cycle, total_life_est')

      # /norm_params
      grp = f.create_group('norm_params')
      grp.create_dataset('sensors',
        data=np.array(norm.keep, dtype='S'))
      grp.create_dataset('conditions',
        data=np.array(norm.conds, dtype=np.int32))
      grp.create_dataset('raw_mean',  data=raw_mean)
      grp.create_dataset('raw_std',   data=raw_std)
      grp.create_dataset('raw_min',   data=raw_min)
      grp.create_dataset('raw_max',   data=raw_max)
      grp.create_dataset('norm_mean', data=nm_mean)
      grp.create_dataset('norm_std',  data=nm_std)
      grp.create_dataset('norm_min',  data=nm_min)
      grp.create_dataset('norm_max',  data=nm_max)
      grp.attrs['n_sensors']    = len(norm.keep)
      grp.attrs['n_conditions'] = len(norm.conds)
      grp.attrs['method']       = 'z-score'
      grp.attrs['fit_on']       = f'healthy cycles only (RUL_raw >= {RUL_CAP})'
      grp.attrs['formula']      = 'z = (x - raw_mean) / raw_std'
      grp.attrs['shape_note']   = '(n_conditions, n_sensors)'
      grp.attrs['dropped']      = sorted(norm.drop)
      grp.attrs['cap']          = RUL_CAP

      # /correlation
      grp = f.create_group('correlation')
      grp.create_dataset('sensors',
        data=np.array(ALL_SENSORS, dtype='S'))
      grp.create_dataset('matrix', data=corr_mat, **gz)
      grp.attrs['method']     = 'Pearson'
      grp.attrs['source']     = '/train/data_norm'
      grp.attrs['n_sensors']  = len(ALL_SENSORS)
      grp.attrs['shape_note'] = (
        f'({len(ALL_SENSORS)}, {len(ALL_SENSORS)}) — '
        'NaN for dropped sensors')

      # /engine_summary
      grp = f.create_group('engine_summary')
      ds  = grp.create_dataset('data', data=eng_data, **gz)
      grp.create_dataset('columns',
        data=np.array(ENGINE_SUMMARY_COLS, dtype='S'))
      grp.attrs['n_engines']  = eng_data.shape[0]
      grp.attrs['RUL_note']   = (
        'RUL_test is NaN — train and test engines are different instances.')
      grp.attrs['RUL_pct']    = '50pct and 75pct of total_life'

      # /pca
      pgrp = f.create_group('pca')
      params = pgrp.create_group('params')
      params.create_dataset('sensors',
        data=np.array(norm.keep, dtype='S'))
      params.create_dataset('components',
        data=np.array(pca.components, dtype='S'))
      params.create_dataset('eigenvectors',
        data=pca.calc_loadings())
      params.create_dataset('explained_var',
        data=pca.var_ratio[:pca.n_keep].astype(np.float64))
      params.create_dataset('cumulative_var',
        data=pca.cum_var[:pca.n_keep].astype(np.float64))
      params.create_dataset('all_explained_var',
        data=pca.var_ratio.astype(np.float64))
      params.create_dataset('all_cumulative_var',
        data=pca.cum_var.astype(np.float64))
      params.attrs['method']              = 'PCA'
      params.attrs['n_components']        = pca.n_keep
      params.attrs['n_sensors']           = len(norm.keep)
      params.attrs['variance_threshold']  = VARIANCE_THRESH
      params.attrs['cumulative_variance'] = float(
        pca.cum_var[pca.n_keep - 1])
      params.attrs['fit_on']              = 'train_normalized only'
      params.attrs['component_rule']      = (
        f'Option B: individual variance >= {VARIANCE_THRESH*100:.0f}%')

      ds = pgrp.create_dataset('train', data=pca_train_arr, **gz)
      ds.attrs['columns']    = pca_cols
      ds.attrs['dtype']      = 'float32'
      ds.attrs['meta_cols']  = ['unit_number','time_cycles','condition_id']
      ds.attrs['pc_cols']    = pca.components
      ds.attrs['shape_note'] = (
        f'({pca_train_arr.shape[0]}, {pca_train_arr.shape[1]}) — '
        '3 meta + PC scores')

      ds = pgrp.create_dataset('test', data=pca_test_arr, **gz)
      ds.attrs['columns']    = pca_cols
      ds.attrs['dtype']      = 'float32'
      ds.attrs['meta_cols']  = ['unit_number','time_cycles','condition_id']
      ds.attrs['pc_cols']    = pca.components
      ds.attrs['shape_note'] = (
        f'({pca_test_arr.shape[0]}, {pca_test_arr.shape[1]}) — '
        '3 meta + PC scores (TRAIN-fitted PCA)')
      ds.attrs['fit_note']   = (
        'PCA fitted on train only. '
        'Test scores use train-fitted eigenvectors.')

      # root attributes
      f.attrs['sub_dataset']      = fd
      f.attrs['n_conditions']     = n_conds
      f.attrs['fault_modes']      = FAULT_MODES[fd]
      f.attrs['n_train_engines']  = n_trains
      f.attrs['n_test_engines']   = n_tests
      f.attrs['n_train_rows']     = len(train_arr)
      f.attrs['n_test_rows']      = len(test_arr)
      f.attrs['rul_cap']          = RUL_CAP
      f.attrs['pca_n_components'] = pca.n_keep
      f.attrs['pca_variance']     = float(pca.cum_var[pca.n_keep - 1])
      f.attrs['reference']        = REFERENCE
      f.attrs['version']          = 'complete'

  def _calc_correlation(self, norm_array, norm_sensors):
    n   = len(ALL_SENSORS)
    mat = np.full((n, n), np.nan, dtype=np.float32)
    idx = {s: i for i, s in enumerate(norm_sensors)}

    for i, s1 in enumerate(ALL_SENSORS):
      for j, s2 in enumerate(ALL_SENSORS):
        if s1 not in idx or s2 not in idx:
          continue
        x = norm_array[:, idx[s1]].astype(np.float64)
        y = norm_array[:, idx[s2]].astype(np.float64)
        if x.std() < 1e-10 or y.std() < 1e-10:
          continue
        mat[i, j] = float(np.corrcoef(x, y)[0, 1])

    return mat


# ── verifier ──────────────────────────────────────────────────────────────────

class CmapssVerifier:

  def verify_file(self, path):
    errors = []
    required = [
      '/train/data', '/train/data_norm',
      '/test/data',  '/test/data_norm',
      '/RUL/data',
      '/norm_params/sensors', '/norm_params/raw_mean',
      '/correlation/matrix',  '/correlation/sensors',
      '/engine_summary/data', '/engine_summary/columns',
      '/pca/params/eigenvectors',
      '/pca/params/explained_var',
      '/pca/train', '/pca/test',
    ]
    with h5py.File(path, 'r') as f:
      for ds in required:
        if ds not in f:
          errors.append(f'missing {ds}')

      # shape checks
      if '/train/data' in f and f['/train/data'].shape[1] != 32:
        errors.append(f'/train/data cols={f["/train/data"].shape[1]}')
      if '/test/data' in f and f['/test/data'].shape[1] != 29:
        errors.append(f'/test/data cols={f["/test/data"].shape[1]}')
      if '/RUL/data' in f and f['/RUL/data'].shape[1] != 4:
        errors.append(f'/RUL/data cols={f["/RUL/data"].shape[1]}')
      if '/correlation/matrix' in f:
        s = f['/correlation/matrix'].shape
        if s != (21, 21):
          errors.append(f'/correlation/matrix shape={s}')

    return errors

  def print_summary(self, path, fd):
    with h5py.File(path, 'r') as f:
      print(f"\n  {fd} — {os.path.getsize(path)//1024} KB")
      f.visit(lambda name: print(
        f"    /{name}  "
        f"{f[name].shape if hasattr(f[name],'shape') else ''}"))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
  data_dir = '/home/claude/CMAPSSData'
  out_dir  = '/home/claude'

  loader   = CmapssLoader(data_dir)
  prep     = CmapssPrep()
  eng_calc = EngSummaryCalc()
  writer   = CmapssWriter()
  verifier = CmapssVerifier()

  print(f"{'sub':6}  {'train':>8}  {'test':>8}  "
        f"{'norm_s':>7}  {'pca_k':>6}  {'cum_var':>8}  "
        f"{'KB':>6}  status")
  print('-' * 72)

  for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    train_df, test_df, rul_df = loader.calc_load(fd)
    n_conds = N_CONDITIONS[fd]
    train_df, test_df, rul_df = prep.calc_prepare(
      train_df, test_df, rul_df, n_conds)

    norm = CmapssNorm()
    pca  = CmapssPca()

    path = os.path.join(out_dir, f'cmapss_{fd}.h5')
    writer.build_file(
      fd, path, train_df, test_df, rul_df,
      norm, pca, eng_calc)

    errors = verifier.verify_file(path)
    status = 'OK' if not errors else 'ERR: ' + ', '.join(errors)
    size_kb = os.path.getsize(path) // 1024
    cum_var = float(pca.cum_var[pca.n_keep - 1])

    print(
      f"{fd:6}  {len(train_df):>8,}  {len(test_df):>8,}  "
      f"{len(norm.keep):>7}  {pca.n_keep:>6}  "
      f"{cum_var*100:>7.1f}%  {size_kb:>6}  {status}")

  print()
  print("Structure (FD001):")
  verifier.print_summary(
    os.path.join(out_dir, 'cmapss_FD001.h5'), 'FD001')

  print()
  print("Files written:")
  for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    p  = os.path.join(out_dir, f'cmapss_{fd}.h5')
    kb = os.path.getsize(p) // 1024
    print(f"  cmapss_{fd}.h5  ({kb} KB)")


raise SystemExit(main())

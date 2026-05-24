# build_cmapss_basic.py
#
# Builds basic C-MAPSS HDF5 files with extended columns.
# One file per sub-dataset: cmapss_basic_FD001.h5 ... cmapss_basic_FD004.h5
#
# Structure per file:
#   /train/data   float32  (n_rows, 32)   26 raw + 6 computed
#   /test/data    float32  (n_rows, 29)   26 raw + 3 computed
#   /RUL/data     int32    (n_engines, 4) unit_number, rul, last_cycle, total_life_est
#
# Computed columns:
#   train: condition_id, dT_compressor, dT_turbine, RUL_raw, RUL_capped, cycle_pct
#   test:  condition_id, dT_compressor, dT_turbine
#   RUL:   unit_number, rul, last_cycle, total_life_est
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

TRAIN_COMPUTED = [
  'condition_id', 'dT_compressor', 'dT_turbine',
  'RUL_raw', 'RUL_capped', 'cycle_pct',
]

TEST_COMPUTED = [
  'condition_id', 'dT_compressor', 'dT_turbine',
]

RUL_COLUMNS = [
  'unit_number', 'rul', 'last_cycle', 'total_life_est',
]

TRAIN_COLUMNS = RAW_COLUMNS + TRAIN_COMPUTED
TEST_COLUMNS  = RAW_COLUMNS + TEST_COMPUTED

RUL_CAP = 125

N_CONDITIONS = {'FD001': 1, 'FD002': 6, 'FD003': 1, 'FD004': 6}
FAULT_MODES  = {'FD001': 'HPC', 'FD002': 'HPC', 'FD003': 'HPC+Fan', 'FD004': 'HPC+Fan'}

REFERENCE = 'Saxena et al., PHM08 — C-MAPSS dataset. Basic version: 26 raw + computed columns.'


# ── loader ────────────────────────────────────────────────────────────────────

class CmapssLoader:

  def __init__(self, data_dir):
    self.data_dir = data_dir

  def calc_train(self, fd):
    return self._read(f'train_{fd}.txt')

  def calc_test(self, fd):
    return self._read(f'test_{fd}.txt')

  def calc_rul_values(self, fd):
    path = os.path.join(self.data_dir, f'RUL_{fd}.txt')
    df   = pd.read_csv(path, sep=r'\s+', header=None, names=['rul'])
    return df['rul'].values.astype(np.int32)

  def _read(self, filename):
    path = os.path.join(self.data_dir, filename)
    return pd.read_csv(path, sep=r'\s+', header=None, names=RAW_COLUMNS)


# ── computed columns ──────────────────────────────────────────────────────────

class CmapssComputer:

  def calc_condition_id(self, df, n_conds, kmeans_model=None):
    if n_conds == 1:
      return np.ones(len(df), dtype=np.int32)
    ops = df[['op_setting_1', 'op_setting_2', 'op_setting_3']].values
    if kmeans_model is None:
      kmeans_model = KMeans(n_clusters=6, random_state=42, n_init=10)
      kmeans_model.fit(ops)
    labels = kmeans_model.predict(ops).astype(np.int32) + 1  # 1-based
    return labels, kmeans_model

  def calc_dT_compressor(self, df):
    return (df['T30'] - df['T24']).values.astype(np.float32)

  def calc_dT_turbine(self, df):
    return (df['T50'] - df['T30']).values.astype(np.float32)

  def calc_rul_raw(self, df):
    t_max   = df.groupby('unit_number')['time_cycles'].transform('max')
    rul_raw = (t_max - df['time_cycles']).values.astype(np.float32)
    return rul_raw

  def calc_rul_capped(self, rul_raw):
    return np.clip(rul_raw, None, RUL_CAP).astype(np.float32)

  def calc_cycle_pct(self, df):
    t_max     = df.groupby('unit_number')['time_cycles'].transform('max')
    cycle_pct = (df['time_cycles'] / t_max).values.astype(np.float32)
    return cycle_pct

  def calc_last_cycle(self, test_df):
    return test_df.groupby('unit_number')['time_cycles'].max()

  def calc_rul_table(self, test_df, rul_values):
    n         = len(rul_values)
    unit_nums = np.arange(1, n + 1, dtype=np.int32)
    last_cyc  = self.calc_last_cycle(test_df)
    last_arr  = np.array(
      [last_cyc[u] for u in unit_nums], dtype=np.int32
    )
    total_arr = (last_arr + rul_values).astype(np.int32)
    table     = np.column_stack([unit_nums, rul_values, last_arr, total_arr])
    return table.astype(np.int32)


# ── builder ───────────────────────────────────────────────────────────────────

class CmapssBasicBuilder:

  def __init__(self, loader, computer, out_dir):
    self.loader   = loader
    self.computer = computer
    self.out_dir  = out_dir

  def build_file(self, fd):
    train_df   = self.loader.calc_train(fd)
    test_df    = self.loader.calc_test(fd)
    rul_values = self.loader.calc_rul_values(fd)
    n_conds    = N_CONDITIONS[fd]

    train_arr, test_arr, rul_table = self._build_arrays(
      fd, train_df, test_df, rul_values, n_conds
    )

    n_train_engines = int(train_df['unit_number'].nunique())
    n_test_engines  = int(test_df['unit_number'].nunique())

    path = os.path.join(self.out_dir, f'cmapss_basic_{fd}.h5')
    with h5py.File(path, 'w') as f:
      self.build_train_group(f, train_arr)
      self.build_test_group(f, test_arr)
      self.build_rul_group(f, rul_table)
      self.build_root_attrs(f, fd, {
        'n_train_engines': n_train_engines,
        'n_test_engines':  n_test_engines,
        'n_train_rows':    len(train_arr),
        'n_test_rows':     len(test_arr),
      })

    size_kb = os.path.getsize(path) // 1024
    return path, size_kb, len(train_arr), len(test_arr), len(rul_table)

  def _build_arrays(self, fd, train_df, test_df, rul_values, n_conds):
    comp = self.computer

    # condition_id — fit on train, apply to test
    if n_conds == 1:
      train_cond = comp.calc_condition_id(train_df, n_conds)
      test_cond  = np.ones(len(test_df), dtype=np.int32)
      kmeans     = None
    else:
      train_cond, kmeans = comp.calc_condition_id(train_df, n_conds)
      test_cond, _       = comp.calc_condition_id(test_df, n_conds, kmeans)

    # train computed columns
    dT_comp_tr  = comp.calc_dT_compressor(train_df)
    dT_turb_tr  = comp.calc_dT_turbine(train_df)
    rul_raw     = comp.calc_rul_raw(train_df)
    rul_capped  = comp.calc_rul_capped(rul_raw)
    cycle_pct   = comp.calc_cycle_pct(train_df)

    # test computed columns
    dT_comp_te = comp.calc_dT_compressor(test_df)
    dT_turb_te = comp.calc_dT_turbine(test_df)

    # assemble train array (n_rows, 32)
    train_raw  = train_df[RAW_COLUMNS].values.astype(np.float32)
    train_arr  = np.column_stack([
      train_raw,
      train_cond.astype(np.float32),
      dT_comp_tr,
      dT_turb_tr,
      rul_raw,
      rul_capped,
      cycle_pct,
    ])

    # assemble test array (n_rows, 29)
    test_raw  = test_df[RAW_COLUMNS].values.astype(np.float32)
    test_arr  = np.column_stack([
      test_raw,
      test_cond.astype(np.float32),
      dT_comp_te,
      dT_turb_te,
    ])

    # RUL table (n_engines, 4)
    rul_table = comp.calc_rul_table(test_df, rul_values)

    return train_arr, test_arr, rul_table

  def build_train_group(self, f, data):
    grp = f.create_group('train')
    ds  = grp.create_dataset(
      'data', data=data,
      compression='gzip', compression_opts=4,
    )
    ds.attrs['columns']    = TRAIN_COLUMNS
    ds.attrs['dtype']      = 'float32'
    ds.attrs['n_raw']      = len(RAW_COLUMNS)
    ds.attrs['n_computed'] = len(TRAIN_COMPUTED)
    ds.attrs['shape_note'] = '(n_rows, 32) — 26 raw + 6 computed'
    ds.attrs['rul_cap']    = RUL_CAP

  def build_test_group(self, f, data):
    grp = f.create_group('test')
    ds  = grp.create_dataset(
      'data', data=data,
      compression='gzip', compression_opts=4,
    )
    ds.attrs['columns']    = TEST_COLUMNS
    ds.attrs['dtype']      = 'float32'
    ds.attrs['n_raw']      = len(RAW_COLUMNS)
    ds.attrs['n_computed'] = len(TEST_COMPUTED)
    ds.attrs['shape_note'] = '(n_rows, 29) — 26 raw + 3 computed'

  def build_rul_group(self, f, rul_table):
    grp = f.create_group('RUL')
    ds  = grp.create_dataset('data', data=rul_table)
    ds.attrs['columns']    = RUL_COLUMNS
    ds.attrs['dtype']      = 'int32'
    ds.attrs['shape_note'] = (
      '(n_engines, 4) — unit_number, rul, last_cycle, total_life_est. '
      'unit_number is explicit (1-based, positional per NASA convention).'
    )

  def build_root_attrs(self, f, fd, meta):
    f.attrs['sub_dataset']      = fd
    f.attrs['n_conditions']     = N_CONDITIONS[fd]
    f.attrs['fault_modes']      = FAULT_MODES[fd]
    f.attrs['n_train_engines']  = meta['n_train_engines']
    f.attrs['n_test_engines']   = meta['n_test_engines']
    f.attrs['n_train_rows']     = meta['n_train_rows']
    f.attrs['n_test_rows']      = meta['n_test_rows']
    f.attrs['train_columns']    = TRAIN_COLUMNS
    f.attrs['test_columns']     = TEST_COLUMNS
    f.attrs['rul_columns']      = RUL_COLUMNS
    f.attrs['rul_cap']          = RUL_CAP
    f.attrs['reference']        = REFERENCE


# ── verifier ──────────────────────────────────────────────────────────────────

class CmapssVerifier:

  def verify_file(self, path, fd):
    errors = []
    with h5py.File(path, 'r') as f:

      for grp in ['train', 'test', 'RUL']:
        if grp not in f:
          errors.append(f'missing group /{grp}')

      if '/train/data' in f:
        s = f['/train/data'].shape
        if s[1] != 32:
          errors.append(f'/train/data cols={s[1]}, expected 32')
      else:
        errors.append('missing /train/data')

      if '/test/data' in f:
        s = f['/test/data'].shape
        if s[1] != 29:
          errors.append(f'/test/data cols={s[1]}, expected 29')
      else:
        errors.append('missing /test/data')

      if '/RUL/data' in f:
        s = f['/RUL/data'].shape
        if len(s) != 2 or s[1] != 4:
          errors.append(f'/RUL/data shape={s}, expected (n, 4)')
        n_rul = s[0]
        if 'n_test_engines' in f.attrs:
          n_eng = int(f.attrs['n_test_engines'])
          if n_rul != n_eng:
            errors.append(f'RUL rows {n_rul} != n_test_engines {n_eng}')
      else:
        errors.append('missing /RUL/data')

      tr_shape  = f['/train/data'].shape if '/train/data' in f else '?'
      te_shape  = f['/test/data'].shape  if '/test/data'  in f else '?'
      rul_shape = f['/RUL/data'].shape   if '/RUL/data'   in f else '?'

      # spot-check: verify first RUL row unit_number == 1
      if '/RUL/data' in f:
        first = f['/RUL/data'][0]
        if first[0] != 1:
          errors.append(f'RUL row 0 unit_number={first[0]}, expected 1')

    return errors, tr_shape, te_shape, rul_shape


# ── main ──────────────────────────────────────────────────────────────────────

def main():
  data_dir = '/home/claude/CMAPSSData'
  out_dir  = '/home/claude'

  loader   = CmapssLoader(data_dir)
  computer = CmapssComputer()
  builder  = CmapssBasicBuilder(loader, computer, out_dir)
  verifier = CmapssVerifier()

  print(f"{'sub':6}  {'train rows':>10}  {'test rows':>9}  {'engines':>7}  {'size KB':>7}  status")
  print('-' * 62)

  for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    path, size_kb, n_tr, n_te, n_rul = builder.build_file(fd)
    errors, tr_shape, te_shape, rul_shape = verifier.verify_file(path, fd)
    status = 'OK' if not errors else 'ERRORS: ' + ', '.join(errors)
    print(f"{fd:6}  {n_tr:>10,}  {n_te:>9,}  {n_rul:>7}  {size_kb:>7}  {status}")

  print()
  print("Shapes:")
  for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    path = os.path.join(out_dir, f'cmapss_basic_{fd}.h5')
    with h5py.File(path, 'r') as f:
      tr  = f['/train/data'].shape
      te  = f['/test/data'].shape
      rul = f['/RUL/data'].shape
      print(f"  {fd}  train={tr}  test={te}  RUL={rul}")

  print()
  print("Sample — FD001 RUL first 3 rows (unit, rul, last_cycle, total_life_est):")
  with h5py.File(os.path.join(out_dir, 'cmapss_basic_FD001.h5'), 'r') as f:
    for row in f['/RUL/data'][:3]:
      print(f"  unit={row[0]}  rul={row[1]}  last_cycle={row[2]}  total_life_est={row[3]}")

  print()
  print("Files written:")
  for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    p = os.path.join(out_dir, f'cmapss_basic_{fd}.h5')
    print(f"  {p}")


raise SystemExit(main())

# build_cmapss_vanilla.py
#
# Builds vanilla HDF5 files for the C-MAPSS turbofan dataset.
# One file per sub-dataset: cmapss_FD001.h5 ... cmapss_FD004.h5
#
# Structure per file:
#   /train/data      float32  (n_train_rows, 26)
#   /test/data       float32  (n_test_rows,  26)
#   /RUL/data        int32    (n_engines,)
#
# Reference: Saxena et al., PHM08

import os
import numpy as np
import pandas as pd
import h5py


COLUMNS = [
  'unit_number', 'time_cycles',
  'op_setting_1', 'op_setting_2', 'op_setting_3',
  'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30',
  'Nf', 'Nc', 'epr', 'Ps30', 'phi',
  'NRf', 'NRc', 'BPR', 'farB', 'htBleed',
  'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32',
]

N_CONDITIONS = {
  'FD001': 1,
  'FD002': 6,
  'FD003': 1,
  'FD004': 6,
}

FAULT_MODES = {
  'FD001': 'HPC',
  'FD002': 'HPC',
  'FD003': 'HPC+Fan',
  'FD004': 'HPC+Fan',
}

REFERENCE = 'Saxena et al., PHM08 — C-MAPSS dataset. Vanilla: raw 26 columns only.'


# ── loader ────────────────────────────────────────────────────────────────────

class CmapssLoader:

  def __init__(self, data_dir):
    self.data_dir = data_dir

  def calc_train(self, fd):
    return self._read(f'train_{fd}.txt')

  def calc_test(self, fd):
    return self._read(f'test_{fd}.txt')

  def calc_rul(self, fd):
    path = os.path.join(self.data_dir, f'RUL_{fd}.txt')
    rul  = pd.read_csv(path, sep=r'\s+', header=None, names=['rul'])
    return rul['rul'].values.astype(np.int32)

  def _read(self, filename):
    path = os.path.join(self.data_dir, filename)
    df   = pd.read_csv(path, sep=r'\s+', header=None, names=COLUMNS)
    return df[COLUMNS].values.astype(np.float32)


# ── builder ───────────────────────────────────────────────────────────────────

class CmapssVanillaBuilder:

  def __init__(self, loader, out_dir):
    self.loader  = loader
    self.out_dir = out_dir

  def build_file(self, fd):
    train_data = self.loader.calc_train(fd)
    test_data  = self.loader.calc_test(fd)
    rul_data   = self.loader.calc_rul(fd)

    n_train_engines = int(np.unique(train_data[:, 0]).size)
    n_test_engines  = int(np.unique(test_data[:, 0]).size)

    path = os.path.join(self.out_dir, f'cmapss_{fd}.h5')
    with h5py.File(path, 'w') as f:
      self.build_train_group(f, train_data)
      self.build_test_group(f, test_data)
      self.build_rul_group(f, rul_data)
      self.build_root_attrs(f, fd, {
        'n_train_engines': n_train_engines,
        'n_test_engines':  n_test_engines,
        'n_train_rows':    len(train_data),
        'n_test_rows':     len(test_data),
      })

    size_kb = os.path.getsize(path) // 1024
    return path, size_kb, len(train_data), len(test_data), len(rul_data)

  def build_train_group(self, f, data):
    grp = f.create_group('train')
    ds  = grp.create_dataset(
      'data', data=data,
      compression='gzip', compression_opts=4,
    )
    ds.attrs['columns']    = COLUMNS
    ds.attrs['dtype']      = 'float32'
    ds.attrs['shape_note'] = '(n_rows, 26) — one row per cycle per engine'

  def build_test_group(self, f, data):
    grp = f.create_group('test')
    ds  = grp.create_dataset(
      'data', data=data,
      compression='gzip', compression_opts=4,
    )
    ds.attrs['columns']    = COLUMNS
    ds.attrs['dtype']      = 'float32'
    ds.attrs['shape_note'] = '(n_rows, 26) — truncated before failure'

  def build_rul_group(self, f, rul_values):
    grp = f.create_group('RUL')
    ds  = grp.create_dataset('data', data=rul_values)
    ds.attrs['dtype'] = 'int32'
    ds.attrs['note']  = (
      'One integer per test engine. '
      'True RUL at the last observed cycle. '
      'Positional: index 0 = unit_number 1, index 1 = unit_number 2, ...'
    )

  def build_root_attrs(self, f, fd, meta):
    f.attrs['sub_dataset']      = fd
    f.attrs['n_conditions']     = N_CONDITIONS[fd]
    f.attrs['fault_modes']      = FAULT_MODES[fd]
    f.attrs['n_train_engines']  = meta['n_train_engines']
    f.attrs['n_test_engines']   = meta['n_test_engines']
    f.attrs['n_train_rows']     = meta['n_train_rows']
    f.attrs['n_test_rows']      = meta['n_test_rows']
    f.attrs['columns']          = COLUMNS
    f.attrs['reference']        = REFERENCE


# ── verifier ──────────────────────────────────────────────────────────────────

class CmapssVerifier:

  def verify_file(self, path, fd):
    errors = []
    with h5py.File(path, 'r') as f:
      # check groups
      for grp in ['train', 'test', 'RUL']:
        if grp not in f:
          errors.append(f'missing group /{grp}')

      # check datasets
      for ds_path in ['/train/data', '/test/data', '/RUL/data']:
        if ds_path not in f:
          errors.append(f'missing dataset {ds_path}')

      # check shapes
      if '/train/data' in f and f['/train/data'].ndim != 2:
        errors.append('/train/data not 2-D')
      if '/train/data' in f and f['/train/data'].shape[1] != 26:
        errors.append(f'/train/data cols={f["/train/data"].shape[1]}, expected 26')
      if '/test/data' in f and f['/test/data'].shape[1] != 26:
        errors.append(f'/test/data cols={f["/test/data"].shape[1]}, expected 26')
      if '/RUL/data' in f and f['/RUL/data'].ndim != 1:
        errors.append('/RUL/data not 1-D')

      # check root attrs
      for attr in ['sub_dataset', 'n_conditions', 'fault_modes',
                   'n_train_engines', 'n_test_engines',
                   'n_train_rows', 'n_test_rows', 'columns', 'reference']:
        if attr not in f.attrs:
          errors.append(f'missing root attr: {attr}')

      # check RUL count matches n_test_engines
      if '/RUL/data' in f and 'n_test_engines' in f.attrs:
        n_rul  = len(f['/RUL/data'])
        n_eng  = int(f.attrs['n_test_engines'])
        if n_rul != n_eng:
          errors.append(f'RUL length {n_rul} != n_test_engines {n_eng}')

      # summary line
      tr_shape  = f['/train/data'].shape  if '/train/data' in f else '?'
      te_shape  = f['/test/data'].shape   if '/test/data'  in f else '?'
      rul_shape = f['/RUL/data'].shape    if '/RUL/data'   in f else '?'

    return errors, tr_shape, te_shape, rul_shape


# ── main ──────────────────────────────────────────────────────────────────────

def main():
  data_dir = '/home/claude/CMAPSSData'
  out_dir  = '/home/claude'

  loader   = CmapssLoader(data_dir)
  builder  = CmapssVanillaBuilder(loader, out_dir)
  verifier = CmapssVerifier()

  print(f"{'sub':6}  {'train rows':>10}  {'test rows':>9}  {'engines':>7}  {'size KB':>7}  status")
  print('-' * 62)

  for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    path, size_kb, n_tr, n_te, n_rul = builder.build_file(fd)
    errors, tr_shape, te_shape, rul_shape = verifier.verify_file(path, fd)
    status = 'OK' if not errors else 'ERRORS: ' + ', '.join(errors)
    print(f"{fd:6}  {n_tr:>10,}  {n_te:>9,}  {n_rul:>7}  {size_kb:>7}  {status}")

  print()
  print("Files written:")
  for fd in ['FD001', 'FD002', 'FD003', 'FD004']:
    p = os.path.join(out_dir, f'cmapss_{fd}.h5')
    print(f"  {p}")


raise SystemExit(main())

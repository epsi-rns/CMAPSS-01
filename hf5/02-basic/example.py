import h5py
import pandas as pd

with h5py.File('cmapss_basic_FD001.h5', 'r') as f:
    data = f['/train/data'][:]
    cols = list(f['/train/data'].attrs['columns'])

df = pd.DataFrame(data, columns=cols)
print(df.head(10))
print()
print(df.describe())



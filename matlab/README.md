# FD001 MATLAB Pipeline

MATLAB counterpart to the Python ANN + Garson pipeline for C-MAPSS FD001.

## Files

| Script | Purpose |
|---|---|
| `fd001_run_all.m` | Master runner — runs all steps in sequence |
| `fd001_load_data.m` | Step 1: Load CSV data, save to `.mat` |
| `fd001_train_ann.m` | Step 2: Train two `feedforwardnet` models |
| `fd001_garson.m` | Step 3: Garson feature importance |
| `fd001_plot.m` | Step 4: 10 figures (PNG output) |

## CSV data files (from Python export)

| File | Shape | Contents |
|---|---|---|
| `fd001_train_norm.csv` | 20631 × 19 | 15 normalized sensors + meta |
| `fd001_test_norm.csv` | 13096 × 17 | 15 normalized sensors + meta |
| `fd001_pca_train.csv` | 20631 × 5 | unit, cycle, cond, PC1, PC2 |
| `fd001_pca_test.csv` | 13096 × 5 | unit, cycle, cond, PC1, PC2 |
| `fd001_rul.csv` | 100 × 4 | answer key (true RUL per test engine) |
| `fd001_pca_variance.csv` | 15 × 2 | explained variance per component |
| `fd001_train_raw_subset.csv` | 20631 × 8 | raw sensors for degradation plot |

## Requirements

- MATLAB R2020b or newer
- Neural Network Toolbox (`feedforwardnet`)
- Statistics and Machine Learning Toolbox (`pca`)

## How to run

```matlab
% Option 1: run all at once
run('fd001_run_all.m')

% Option 2: step by step
run('fd001_load_data.m')
run('fd001_train_ann.m')
run('fd001_garson.m')
run('fd001_plot.m')
```

## Expected results

| Model | Dim | RMSE | MAE | NASA Score |
|---|---|---|---|---|
| FD001_PCA | 2 | ~17.8 | ~12.5 | ~815 |
| FD001_NORM | 15 | ~18.2 | ~13.5 | ~929 |

Note: MATLAB `trainlm` (Levenberg-Marquardt) may produce slightly
different values from Python Adam optimizer, but should be in the
same range.

## Architecture

```
Model A (PCA):   [PC1, PC2]  →  [64]  →  [32]  →  [1 = RUL]
Model B (NORM):  [15 sensors] → [64]  →  [32]  →  [1 = RUL]
```

Activation: tansig (hidden), purelin (output) — MATLAB default for feedforwardnet.

## Output figures

| Figure | Filename | Description |
|---|---|---|
| 1 | `matlab_A1_degradation.png` | Sensor trajectories |
| 2 | `matlab_A3_scree.png` | PCA scree plot |
| 3 | `matlab_A4_pca_scatter.png` | PC1 vs PC2 colored by RUL |
| 4 | `matlab_B5_loss_curve.png` | Training loss per epoch |
| 5 | `matlab_B6_true_vs_pred.png` | True vs predicted RUL |
| 6 | `matlab_B7_error_dist.png` | Error distribution |
| 7 | `matlab_C8_garson_norm.png` | Garson NORM importance |
| 8 | `matlab_C10_garson_pca.png` | Garson PCA importance |
| 9 | `matlab_D11_nasa_score.png` | NASA Score curve |
| 10 | `matlab_D12_rul_cap.png` | RUL cap visualization |

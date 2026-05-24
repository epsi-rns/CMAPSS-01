% fd001_load_data.m
%
% Step 1: Load C-MAPSS FD001 data from CSV files.
% Reads pre-processed data exported from Python HDF5 pipeline.
%
% Output variables (saved to fd001_data.mat):
%   X_train_norm  [20631 x 15]  normalized sensor features (train)
%   X_test_norm   [13096 x 15]  normalized sensor features (test)
%   X_train_pca   [20631 x 2]   PCA scores (train)
%   X_test_pca    [13096 x 2]   PCA scores (test)
%   y_train       [20631 x 1]   RUL_capped target (train)
%   y_test_true   [100 x 1]     true RUL answer key (test)
%   unit_train    [20631 x 1]   engine unit numbers (train)
%   unit_test     [13096 x 1]   engine unit numbers (test)
%   t_train       [20631 x 1]   time_cycles (train)
%   sensor_names  {1 x 15}      sensor name strings
%   pca_var       [15 x 1]      explained variance per component
%
% Reference: Saxena et al., PHM08

clc; clear; close all;

fprintf('=== FD001 Data Loader ===\n\n');

% ── file paths ────────────────────────────────────────────────────────────────

data_dir = fileparts(mfilename('fullpath'));

f_train_norm = fullfile(data_dir, 'fd001_train_norm.csv');
f_test_norm  = fullfile(data_dir, 'fd001_test_norm.csv');
f_pca_train  = fullfile(data_dir, 'fd001_pca_train.csv');
f_pca_test   = fullfile(data_dir, 'fd001_pca_test.csv');
f_rul        = fullfile(data_dir, 'fd001_rul.csv');
f_pca_var    = fullfile(data_dir, 'fd001_pca_variance.csv');

% ── load tables ───────────────────────────────────────────────────────────────

fprintf('Loading CSV files...\n');

T_train_norm = readtable(f_train_norm);
T_test_norm  = readtable(f_test_norm);
T_pca_train  = readtable(f_pca_train);
T_pca_test   = readtable(f_pca_test);
T_rul        = readtable(f_rul);
T_pca_var    = readtable(f_pca_var);

% ── sensor names ──────────────────────────────────────────────────────────────

sensor_names = {'T24','T30','T50','P15','P30','Nf','Nc', ...
                'Ps30','phi','NRf','NRc','BPR','htBleed','W31','W32'};

% ── extract arrays ────────────────────────────────────────────────────────────

% train: 15 sensor columns
X_train_norm = T_train_norm{:, sensor_names};      % [20631 x 15]
y_train      = T_train_norm.RUL_capped;            % [20631 x 1]
unit_train   = T_train_norm.unit_number;           % [20631 x 1]
t_train      = T_train_norm.time_cycles;           % [20631 x 1]
rul_raw_train= T_train_norm.RUL_raw;               % [20631 x 1]

% test: 15 sensor columns
X_test_norm  = T_test_norm{:, sensor_names};       % [13096 x 15]
unit_test    = T_test_norm.unit_number;            % [13096 x 1]

% PCA scores
X_train_pca  = [T_pca_train.PC1, T_pca_train.PC2]; % [20631 x 2]
X_test_pca   = [T_pca_test.PC1,  T_pca_test.PC2];  % [13096 x 2]

% RUL answer key
y_test_true  = T_rul.rul;                          % [100 x 1]

% PCA variance
pca_var      = T_pca_var.explained_var;            % [15 x 1]

% ── summary ───────────────────────────────────────────────────────────────────

fprintf('\nData summary:\n');
fprintf('  X_train_norm: %d x %d  (normalized sensors)\n', ...
        size(X_train_norm,1), size(X_train_norm,2));
fprintf('  X_train_pca:  %d x %d  (PCA scores)\n', ...
        size(X_train_pca,1),  size(X_train_pca,2));
fprintf('  y_train:      %d x 1   (RUL_capped, range %.0f-%.0f)\n', ...
        length(y_train), min(y_train), max(y_train));
fprintf('  X_test_norm:  %d x %d  (normalized sensors)\n', ...
        size(X_test_norm,1),  size(X_test_norm,2));
fprintf('  X_test_pca:   %d x %d  (PCA scores)\n', ...
        size(X_test_pca,1),   size(X_test_pca,2));
fprintf('  y_test_true:  %d x 1   (true RUL, range %.0f-%.0f)\n', ...
        length(y_test_true), min(y_test_true), max(y_test_true));
fprintf('  Engines train: %d unique\n', numel(unique(unit_train)));
fprintf('  Engines test:  %d unique\n', numel(unique(unit_test)));

fprintf('\nPCA variance:\n');
cum_var = 0;
for i = 1:min(5, length(pca_var))
    cum_var = cum_var + pca_var(i);
    fprintf('  PC%d: %.1f%%  (cumulative: %.1f%%)\n', ...
            i, pca_var(i)*100, cum_var*100);
end

% ── save ──────────────────────────────────────────────────────────────────────

save_path = fullfile(data_dir, 'fd001_data.mat');
save(save_path, ...
     'X_train_norm', 'X_test_norm', ...
     'X_train_pca',  'X_test_pca', ...
     'y_train',      'y_test_true', ...
     'unit_train',   'unit_test', ...
     't_train',      'rul_raw_train', ...
     'sensor_names', 'pca_var');

fprintf('\nSaved: %s\n', save_path);
fprintf('Run fd001_train_ann.m next.\n');

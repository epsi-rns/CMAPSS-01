% fd001_run_all.m
%
% Master runner — executes the full FD001 pipeline in sequence.
%
% Prerequisites:
%   - MATLAB R2020b or newer
%   - Neural Network Toolbox (feedforwardnet)
%   - CSV files in same directory:
%       fd001_train_norm.csv   fd001_test_norm.csv
%       fd001_pca_train.csv    fd001_pca_test.csv
%       fd001_rul.csv          fd001_pca_variance.csv
%       fd001_train_raw_subset.csv
%
% Pipeline:
%   Step 1: fd001_load_data.m   -> fd001_data.mat
%   Step 2: fd001_train_ann.m   -> fd001_ann_results.mat
%   Step 3: fd001_garson.m      -> fd001_garson_results.mat
%   Step 4: fd001_plot.m        -> 10 PNG figures

clc; clear; close all;
fprintf('=== FD001 Complete Pipeline ===\n\n');
t_start = tic;

cd(fileparts(mfilename('fullpath')));

fprintf('Step 1/4: Loading data...\n');
run('fd001_load_data.m');

fprintf('\nStep 2/4: Training ANN models...\n');
run('fd001_train_ann.m');

fprintf('\nStep 3/4: Garson analysis...\n');
run('fd001_garson.m');

fprintf('\nStep 4/4: Generating plots...\n');
run('fd001_plot.m');

fprintf('\n=== Pipeline complete in %.1f seconds ===\n', toc(t_start));

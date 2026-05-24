% fd001_train_ann.m
%
% Step 2: Train two feedforwardnet MLP models on C-MAPSS FD001.
%
% Uses trainscg (Scaled Conjugate Gradient) instead of trainlm:
%   - Much lower memory usage (no Jacobian matrix)
%   - More verbose — prints every epoch
%   - Similar accuracy to trainlm
%
% Model A — PCA input:   PC1, PC2  (2 features)
% Model B — NORM input:  15 normalized sensors
% Architecture: input -> [64 32] -> 1

clc; clear; close all;

fprintf('=== FD001 ANN Training (feedforwardnet) ===\n\n');

% ── load data ─────────────────────────────────────────────────────────────────

data_dir  = fileparts(mfilename('fullpath'));
load(fullfile(data_dir, 'fd001_data.mat'));

% ── training config ───────────────────────────────────────────────────────────

hidden_layers = [64 32];
max_epochs    = 500;
rng(42);

% ── Model A: PCA input ────────────────────────────────────────────────────────

fprintf('Training Model A — PCA input (PC1, PC2)...\n');
fprintf('Epochs: %d\n\n', max_epochs);

net_pca = feedforwardnet(hidden_layers, 'trainscg');
net_pca.trainParam.epochs      = max_epochs;
net_pca.trainParam.showWindow  = false;
net_pca.trainParam.show        = 10;    % print every 10 epochs
net_pca.trainParam.min_grad    = 1e-7;
net_pca.divideParam.trainRatio = 1.0;
net_pca.divideParam.valRatio   = 0.0;
net_pca.divideParam.testRatio  = 0.0;

X_pca_tr = X_train_pca';    % [2 x 20631]
T_tr     = y_train';        % [1 x 20631]

[net_pca, tr_pca] = train(net_pca, X_pca_tr, T_tr);

[pred_pca, rmse_pca, mae_pca, nasa_pca] = ...
    eval_model(net_pca, X_test_pca, unit_test, y_test_true);

fprintf('\n  Model A done.\n');
fprintf('  RMSE = %.4f  MAE = %.4f  NASA = %.2f\n\n', ...
        rmse_pca, mae_pca, nasa_pca);

% ── Model B: NORM input ───────────────────────────────────────────────────────

fprintf('Training Model B — NORM input (15 sensors)...\n');
fprintf('Epochs: %d\n\n', max_epochs);

net_norm = feedforwardnet(hidden_layers, 'trainscg');
net_norm.trainParam.epochs      = max_epochs;
net_norm.trainParam.showWindow  = false;
net_norm.trainParam.show        = 10;
net_norm.trainParam.min_grad    = 1e-7;
net_norm.divideParam.trainRatio = 1.0;
net_norm.divideParam.valRatio   = 0.0;
net_norm.divideParam.testRatio  = 0.0;

X_norm_tr = X_train_norm';  % [15 x 20631]

[net_norm, tr_norm] = train(net_norm, X_norm_tr, T_tr);

[pred_norm, rmse_norm, mae_norm, nasa_norm] = ...
    eval_model(net_norm, X_test_norm, unit_test, y_test_true);

fprintf('\n  Model B done.\n');
fprintf('  RMSE = %.4f  MAE = %.4f  NASA = %.2f\n\n', ...
        rmse_norm, mae_norm, nasa_norm);

% ── summary ───────────────────────────────────────────────────────────────────

fprintf('%s\n', repmat('=',1,56));
fprintf('  FD001 RESULTS\n');
fprintf('%s\n', repmat('=',1,56));
fprintf('  %-16s  %4s  %8s  %8s  %10s\n', ...
        'Model','Dim','RMSE','MAE','NASA');
fprintf('%s\n', repmat('-',1,56));
fprintf('  %-16s  %4d  %8.4f  %8.4f  %10.2f\n', ...
        'FD001_PCA',  2,  rmse_pca,  mae_pca,  nasa_pca);
fprintf('  %-16s  %4d  %8.4f  %8.4f  %10.2f\n', ...
        'FD001_NORM', 15, rmse_norm, mae_norm, nasa_norm);
fprintf('%s\n', repmat('=',1,56));

% ── per-engine results ────────────────────────────────────────────────────────

n_engines  = length(y_test_true);
engine_ids = (1:n_engines)';

results_pca = table(engine_ids, y_test_true, pred_pca, ...
    pred_pca - y_test_true, abs(pred_pca - y_test_true), ...
    'VariableNames', {'unit','true_rul','pred_rul','error','abs_error'});

results_norm = table(engine_ids, y_test_true, pred_norm, ...
    pred_norm - y_test_true, abs(pred_norm - y_test_true), ...
    'VariableNames', {'unit','true_rul','pred_rul','error','abs_error'});

fprintf('\nFirst 10 engines (PCA model):\n');
disp(results_pca(1:10,:));

% ── save ──────────────────────────────────────────────────────────────────────

save_path = fullfile(data_dir, 'fd001_ann_results.mat');
save(save_path, ...
     'net_pca',   'net_norm', ...
     'tr_pca',    'tr_norm', ...
     'pred_pca',  'pred_norm', ...
     'rmse_pca',  'mae_pca',  'nasa_pca', ...
     'rmse_norm', 'mae_norm', 'nasa_norm', ...
     'results_pca', 'results_norm', ...
     'hidden_layers', 'max_epochs');

fprintf('\nSaved: %s\n', save_path);
fprintf('Run fd001_garson.m and fd001_plot.m next.\n');

% ══════════════════════════════════════════════════════════════════════════════
% LOCAL FUNCTIONS
% ══════════════════════════════════════════════════════════════════════════════

function [pred_rul, rmse, mae, nasa] = ...
    eval_model(net, X_test, unit_test, y_true)

  all_pred  = net(X_test')';
  n_engines = length(y_true);
  pred_rul  = zeros(n_engines, 1);

  for i = 1:n_engines
    mask        = (unit_test == i);
    pred_rul(i) = all_pred(find(mask, 1, 'last'));
  end

  errors = pred_rul - y_true;
  rmse   = sqrt(mean(errors.^2));
  mae    = mean(abs(errors));

  nasa = 0;
  for i = 1:length(errors)
    d = errors(i);
    if d < 0
      nasa = nasa + (exp(-d/13) - 1);
    else
      nasa = nasa + (exp( d/10) - 1);
    end
  end
end

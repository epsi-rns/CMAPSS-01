% fd001_garson.m
%
% Step 3: Garson feature importance analysis on trained FD001 networks.
%
% Garson (1991) extended for two hidden layers.
% Extracts weights from MATLAB feedforwardnet object:
%   net.IW{1}   Input->Hidden1   [64 x n_input]
%   net.LW{2,1} Hidden1->Hidden2 [32 x 64]
%   net.LW{3,2} Hidden2->Output  [1  x 32]

clc; clear; close all;

fprintf('=== FD001 Garson Feature Importance ===\n\n');

% ── load ──────────────────────────────────────────────────────────────────────

data_dir = fileparts(mfilename('fullpath'));
load(fullfile(data_dir, 'fd001_data.mat'));
load(fullfile(data_dir, 'fd001_ann_results.mat'));

% ── Model PCA ─────────────────────────────────────────────────────────────────

pca_features = {'PC1', 'PC2'};
imp_pca      = calc_garson(net_pca, 2);

fprintf('Model FD001_PCA — Garson importance:\n');
for i = 1:length(pca_features)
  fprintf('  %d. %-12s  %.2f%%\n', i, pca_features{i}, imp_pca(i)*100);
end

% ── Model NORM ────────────────────────────────────────────────────────────────

imp_norm               = calc_garson(net_norm, 15);
[sorted_imp, sort_idx] = sort(imp_norm, 'descend');

fprintf('\nModel FD001_NORM — Garson importance (top 10):\n');
for i = 1:10
  idx = sort_idx(i);
  bar = repmat(char(9608), 1, round(imp_norm(idx)*100/2));
  fprintf('  %2d. %-12s  %6.2f%%  %s\n', ...
          i, sensor_names{idx}, imp_norm(idx)*100, bar);
end

% ── three-method comparison ───────────────────────────────────────────────────

fprintf('\n%s\n', repmat('=',1,52));
fprintf('  Three-Method Comparison (FD001 NORM model)\n');
fprintf('%s\n', repmat('=',1,52));

pearson_map = containers.Map( ...
  {'T24','T30','T50','P15','P30','Nf','Nc', ...
   'Ps30','phi','NRf','NRc','BPR','htBleed','W31','W32'}, ...
  [-0.607,-0.585,-0.679,-0.128, 0.657,-0.564,-0.390, ...
   -0.696, 0.672,-0.563,-0.307,-0.643,-0.606, 0.629, 0.636]);

fprintf('  %-12s  %8s  %8s  %s\n', ...
        'Sensor','|Pearson|','Garson%','Note');
fprintf('%s\n', repmat('-',1,52));

for i = 1:10
  idx  = sort_idx(i);
  sn   = sensor_names{idx};
  pval = abs(pearson_map(sn));
  gval = imp_norm(idx) * 100;
  note = '';
  if strcmp(sn,'P15')
    note = '<-- anomaly: low Pearson, high Garson';
  end
  fprintf('  %-12s  %8.3f  %8.2f  %s\n', sn, pval, gval, note);
end

% ── save ──────────────────────────────────────────────────────────────────────

save_path = fullfile(data_dir, 'fd001_garson_results.mat');
save(save_path, 'imp_pca', 'imp_norm', 'sort_idx', ...
     'pca_features', 'sorted_imp');

fprintf('\nSaved: %s\n', save_path);
fprintf('Run fd001_plot.m next.\n');

% ══════════════════════════════════════════════════════════════════════════════
% LOCAL FUNCTIONS
% ══════════════════════════════════════════════════════════════════════════════

function importance = calc_garson(net, n_input)
  W1 = net.IW{1};       % [64 x n_input]
  W2 = net.LW{2,1};     % [32 x 64]
  W3 = net.LW{3,2};     % [1  x 32]

  W1_abs   = abs(W1);
  row_sum1 = sum(W1_abs, 2);
  q1       = W1_abs ./ row_sum1;       % [64 x n_input]

  W2_abs   = abs(W2);
  row_sum2 = sum(W2_abs, 2);
  q2       = W2_abs ./ row_sum2;       % [32 x 64]

  r = q1' * q2';                       % [n_input x 32]

  W3_abs   = abs(W3(:));
  s        = r * W3_abs;               % [n_input x 1]

  importance = s / sum(s);
end

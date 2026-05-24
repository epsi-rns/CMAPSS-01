% fd001_plot.m
%
% Step 4: Visualizations for C-MAPSS FD001.
%
% Figures generated:
%   1:  Sensor degradation trajectories (A1)
%   2:  PCA scree plot (A3)
%   3:  PCA scatter colored by RUL (A4)
%   4:  Training loss curve (B5)
%   5:  True vs Predicted RUL scatter (B6)
%   6:  Error distribution (B7)
%   7:  Garson NORM bar chart (C8)
%   8:  Garson PCA bar chart (C10)
%   9:  NASA Score asymmetric curve (D11)
%   10: RUL cap visualization (D12)

clc; clear; close all;

fprintf('=== FD001 Plots ===\n\n');

data_dir = fileparts(mfilename('fullpath'));
load(fullfile(data_dir, 'fd001_data.mat'));
load(fullfile(data_dir, 'fd001_ann_results.mat'));
load(fullfile(data_dir, 'fd001_garson_results.mat'));

% ── colors ───────────────────────────────────────────────────────────────────

c_blue   = [0,   56,  147]/255;
c_teal   = [15,  110,  86]/255;
c_amber  = [133,  79,  11]/255;
c_purple = [107,  26, 107]/255;
c_coral  = [153,  60,  29]/255;
c_gray   = [136, 136, 136]/255;
eng_clr  = {c_blue, c_teal, c_amber, c_purple, c_coral};

% ══════════════════════════════════════════════════════════════════════════════
%% Figure 1: Degradation trajectories
% ══════════════════════════════════════════════════════════════════════════════

fprintf('Figure 1: Degradation trajectories...\n');
T_raw     = readtable(fullfile(data_dir,'fd001_train_raw_subset.csv'));
eng_ids   = [1,11,31,55,77];
sens_list = {'Ps30','T50','phi','P30'};
ylbl_list = {'Ps30 (Static pressure)','T50 (LPT outlet temp)', ...
             'phi (Corrected fan flow)','P30 (HPC outlet pressure)'};

fig1 = figure('Color','white','Position',[50 50 1100 750]);
for s = 1:4
  subplot(2,2,s); hold on; grid on; box off;
  for e = 1:5
    mask = T_raw.unit_number == eng_ids(e);
    t_e  = T_raw.time_cycles(mask);
    v_e  = T_raw.(sens_list{s})(mask);
    [t_e,si] = sort(t_e); v_e = v_e(si);
    plot(t_e, v_e, 'Color', eng_clr{e}, 'LineWidth', 1.2, ...
         'DisplayName', sprintf('Engine %d',eng_ids(e)));
  end
  xlabel('Time cycles'); ylabel(ylbl_list{s});
  title(['Sensor: ' sens_list{s}], 'Color',c_blue,'FontWeight','bold');
  if s==1; legend('Location','northwest','FontSize',8); end
end
sgtitle({'Sensor Degradation Trajectories — C-MAPSS FD001', ...
         'Each line = one engine from first cycle to failure'}, ...
        'Color',c_blue,'FontWeight','bold');
save_fig(fig1,'matlab_A1_degradation.png',data_dir);

% ══════════════════════════════════════════════════════════════════════════════
%% Figure 2: PCA scree plot
% ══════════════════════════════════════════════════════════════════════════════

fprintf('Figure 2: PCA scree plot...\n');
n_comp  = length(pca_var);
cum_var = cumsum(pca_var)*100;
ind_var = pca_var*100;
x_comp  = 1:n_comp;

fig2 = figure('Color','white','Position',[50 50 900 480]);
yyaxis left;
clr_bars = [repmat(c_blue,2,1); repmat([0.7 0.8 0.93],n_comp-2,1)];
b = bar(x_comp, ind_var, 'FaceColor','flat');
b.CData = clr_bars;
hold on;
yline(5,'--','Color',c_amber,'LineWidth',1.2);
text(n_comp-0.5, 5.8, '5% threshold (Option B)', ...
     'Color',c_amber,'FontSize',8,'HorizontalAlignment','right');
for i=1:2
  text(i, ind_var(i)+0.5, sprintf('%.1f%%',ind_var(i)), ...
       'HorizontalAlignment','center','FontSize',9, ...
       'Color',c_blue,'FontWeight','bold');
end
ylabel('Individual Variance (%)'); ylim([0 62]);
ax=gca; ax.YColor=c_blue;
yyaxis right;
plot(x_comp, cum_var, 'o-', 'Color',c_teal,'LineWidth',2,'MarkerSize',5);
yline(82.9,':','Color',c_teal,'LineWidth',1);
text(2.5,84.5,'82.9% at PC2','Color',c_teal,'FontSize',8);
ylabel('Cumulative Variance (%)'); ylim([0 110]);
ax.YColor=c_teal;
xticks(x_comp);
xticklabels(arrayfun(@(i) sprintf('PC%d',i),x_comp,'UniformOutput',false));
xlabel('Principal Component');
title({'PCA Scree Plot — C-MAPSS FD001 (15 sensors)', ...
       'Blue bars = kept (individual variance ≥ 5%)'}, ...
      'Color',c_blue,'FontWeight','bold');
grid on; box off;
save_fig(fig2,'matlab_A3_scree.png',data_dir);

% ══════════════════════════════════════════════════════════════════════════════
%% Figure 3: PCA scatter colored by RUL
% ══════════════════════════════════════════════════════════════════════════════

fprintf('Figure 3: PCA scatter...\n');
idx_sub = 1:3:length(y_train);
fig3 = figure('Color','white','Position',[50 50 800 650]);
scatter(X_train_pca(idx_sub,1), X_train_pca(idx_sub,2), ...
        4, rul_raw_train(idx_sub), 'filled', 'MarkerFaceAlpha',0.5);
colormap(gca, flipud(summer)); cb=colorbar;
cb.Label.String='RUL (cycles)'; clim([0 125]);
hold on;
text(-5.5,2.5,'Degraded (low RUL)','Color',c_coral,'FontSize',9, ...
     'BackgroundColor','white','EdgeColor',c_coral);
text(1.5,2.5,'Healthy (high RUL)','Color',c_teal,'FontSize',9, ...
     'BackgroundColor','white','EdgeColor',c_teal);
xlabel(sprintf('PC1 (%.1f%% variance) — Engine speed sensors',pca_var(1)*100));
ylabel(sprintf('PC2 (%.1f%% variance) — Pressure/temp contrast',pca_var(2)*100));
title({'PCA Score Plot — PC1 vs PC2 colored by RUL', ...
       'FD001 train set (subsampled every 3rd cycle)'}, ...
      'Color',c_blue,'FontWeight','bold');
grid on; box off;
save_fig(fig3,'matlab_A4_pca_scatter.png',data_dir);

% ══════════════════════════════════════════════════════════════════════════════
%% Figure 4: Loss curve
% ══════════════════════════════════════════════════════════════════════════════

fprintf('Figure 4: Loss curve...\n');
ep_pca  = 1:length(tr_pca.perf);
ep_norm = 1:length(tr_norm.perf);
fig4 = figure('Color','white','Position',[50 50 900 460]);
plot(ep_pca,  sqrt(tr_pca.perf),  '-', 'Color',c_blue,'LineWidth',1.8, ...
     'DisplayName',sprintf('Model PCA  final=%.4f',sqrt(tr_pca.perf(end))));
hold on;
plot(ep_norm, sqrt(tr_norm.perf), '--','Color',c_teal,'LineWidth',1.8, ...
     'DisplayName',sprintf('Model NORM final=%.4f',sqrt(tr_norm.perf(end))));
xline(100,':','Color',c_gray,'LineWidth',0.8,'HandleVisibility','off');
xlabel('Epoch'); ylabel('Training RMSE (cycles)');
title({'Training Loss Curve — FD001 PCA vs NORM', ...
       'RMSE per epoch during training'}, ...
      'Color',c_blue,'FontWeight','bold');
legend('Location','northeast','FontSize',9);
grid on; box off;
save_fig(fig4,'matlab_B5_loss_curve.png',data_dir);

% ══════════════════════════════════════════════════════════════════════════════
%% Figure 5: True vs Predicted
% ══════════════════════════════════════════════════════════════════════════════

fprintf('Figure 5: True vs Predicted...\n');
preds = {pred_pca, pred_norm};
rmses = {rmse_pca, rmse_norm};
maes  = {mae_pca,  mae_norm};
nasas = {nasa_pca, nasa_norm};
mlbls = {'Model PCA (PC1, PC2)','Model NORM (15 sensors)'};

fig5 = figure('Color','white','Position',[50 50 1100 520]);
for m=1:2
  subplot(1,2,m);
  errs = preds{m} - y_test_true;
  clrs = zeros(length(errs),3);
  for i=1:length(errs)
    if errs(i)>0; clrs(i,:)=c_coral; else; clrs(i,:)=c_blue; end
  end
  scatter(y_test_true, preds{m}, 50, clrs, 'filled', ...
          'MarkerEdgeColor','white','LineWidth',0.3); hold on;
  plot([0 150],[0 150],'--','Color',c_gray,'LineWidth',1.2);
  fill([0 150 150 0],[0-20 20+[-20 130] 0-20+150], ...
       c_gray,'FaceAlpha',0.07,'EdgeColor','none');
  xlim([0 150]); ylim([0 150]); axis square;
  xlabel('True RUL (cycles)'); ylabel('Predicted RUL (cycles)');
  title({mlbls{m}, sprintf('RMSE=%.2f  MAE=%.2f  NASA=%.0f', ...
         rmses{m},maes{m},nasas{m})}, ...
        'Color',c_blue,'FontWeight','bold','FontSize',10);
  scatter(nan,nan,40,c_coral,'filled','DisplayName','Late (d>0)');
  scatter(nan,nan,40,c_blue, 'filled','DisplayName','Early (d<0)');
  legend('Location','northwest','FontSize',8);
  grid on; box off;
end
sgtitle({'True vs Predicted RUL — 100 Test Engines (FD001)', ...
         'Each point = one engine at last observed cycle'}, ...
        'Color',c_blue,'FontWeight','bold');
save_fig(fig5,'matlab_B6_true_vs_pred.png',data_dir);

% ══════════════════════════════════════════════════════════════════════════════
%% Figure 6: Error distribution
% ══════════════════════════════════════════════════════════════════════════════

fprintf('Figure 6: Error distribution...\n');
bin_e = -80:8:80;
fig6  = figure('Color','white','Position',[50 50 1100 500]);
for m=1:2
  subplot(1,2,m);
  errs  = preds{m} - y_test_true;
  bias  = mean(errs);
  early = errs(errs<0); late = errs(errs>=0);
  histogram(early,bin_e(bin_e<=0),'FaceColor',c_blue,'FaceAlpha',0.75, ...
            'EdgeColor','white','DisplayName',sprintf('Early: %d',length(early)));
  hold on;
  histogram(late, bin_e(bin_e>=0),'FaceColor',c_coral,'FaceAlpha',0.75, ...
            'EdgeColor','white','DisplayName',sprintf('Late: %d',length(late)));
  xline(0,  '-','Color',c_gray,'LineWidth',1.2,'HandleVisibility','off');
  xline(bias,'--','Color',[0.3 0.3 0.3],'LineWidth',1.5, ...
        'DisplayName',sprintf('Mean: %+.1f',bias));
  xlabel('d = pred − true (cycles)'); ylabel('Number of engines');
  title({mlbls{m}, sprintf('RMSE=%.2f  bias=%+.2f',rmses{m},bias)}, ...
        'Color',c_blue,'FontWeight','bold','FontSize',10);
  legend('Location','northeast','FontSize',8);
  grid on; box off;
end
sgtitle({'Error Distribution — d = Predicted − True RUL', ...
         'Blue = early (safe), Red = late (dangerous)'}, ...
        'Color',c_blue,'FontWeight','bold');
save_fig(fig6,'matlab_B7_error_dist.png',data_dir);

% ══════════════════════════════════════════════════════════════════════════════
%% Figure 7: Garson NORM
% ══════════════════════════════════════════════════════════════════════════════

fprintf('Figure 7: Garson NORM...\n');
imp_pct      = imp_norm(sort_idx)*100;
names_sorted = sensor_names(sort_idx);
fig7 = figure('Color','white','Position',[50 50 820 680]);
n_bars = length(imp_pct);
 clr_bar = zeros(n_bars, 3);
 for ci=1:n_bars; clr_bar(ci,:)=c_blue; end
p15_pos = find(strcmp(names_sorted,'P15'));
if ~isempty(p15_pos); clr_bar(p15_pos,:)=c_coral; end
bh = barh(imp_pct,'FaceColor','flat'); bh.CData=clr_bar; hold on;
for i=1:length(imp_pct)
  text(imp_pct(i)+0.2, i, sprintf('%.2f%%',imp_pct(i)), ...
       'VerticalAlignment','middle','FontSize',8.5,'Color',c_blue);
end
yticks(1:length(imp_pct)); yticklabels(names_sorted);
xlabel('Feature Importance (%)');
title({'Garson Feature Importance — Model FD001\_NORM', ...
       '15 sensors, importance sums to 100%'}, ...
      'Color',c_blue,'FontWeight','bold');
xlim([0 13]); grid on; box off;
save_fig(fig7,'matlab_C8_garson_norm.png',data_dir);

% ══════════════════════════════════════════════════════════════════════════════
%% Figure 8: Garson PCA
% ══════════════════════════════════════════════════════════════════════════════

fprintf('Figure 8: Garson PCA...\n');
fig8 = figure('Color','white','Position',[50 50 550 400]);
pca_pct = imp_pca*100;
bh2 = bar(pca_pct,'FaceColor','flat','BarWidth',0.4);
bh2.CData = [c_blue; c_teal]; hold on;
for i=1:2
  text(i,pca_pct(i)+0.5,sprintf('%.2f%%',pca_pct(i)), ...
       'HorizontalAlignment','center','FontWeight','bold','FontSize',11);
end
text(1,pca_pct(1)/2,'Engine speed\nsensors', ...
     'HorizontalAlignment','center','Color','white','FontWeight','bold','FontSize',9);
text(2,pca_pct(2)/2,'Pressure/temp\ncontrast', ...
     'HorizontalAlignment','center','Color','white','FontWeight','bold','FontSize',9);
yline(50,'--','Color',c_gray,'LineWidth',0.8);
xticklabels(pca_features);
ylabel('Feature Importance (%)'); ylim([0 65]);
title({'Garson Feature Importance — Model FD001\_PCA', ...
       'PC1 and PC2 near-equal split'}, ...
      'Color',c_blue,'FontWeight','bold');
grid on; box off;
save_fig(fig8,'matlab_C10_garson_pca.png',data_dir);

% ══════════════════════════════════════════════════════════════════════════════
%% Figure 9: NASA Score curve
% ══════════════════════════════════════════════════════════════════════════════

fprintf('Figure 9: NASA Score curve...\n');
d_e = linspace(-60,0,300); d_l = linspace(0,60,300);
s_e = exp(-d_e/13)-1;      s_l = exp(d_l/10)-1;
fig9 = figure('Color','white','Position',[50 50 900 550]);
plot(d_e,s_e,'-', 'Color',c_blue, 'LineWidth',2.5, ...
     'DisplayName','Early (d<0): exp(−d/13)−1'); hold on;
plot(d_l,s_l,'-', 'Color',c_coral,'LineWidth',2.5, ...
     'DisplayName','Late (d>0): exp(d/10)−1');
plot(d_e,d_e.^2/50,'--','Color',c_gray,'LineWidth',1.2, ...
     'DisplayName','Symmetric reference');
xline(0,'-','Color',c_gray,'LineWidth',1,'HandleVisibility','off');
for dv=[-30 30]
  if dv<0; sv=exp(-dv/13)-1; clr=c_blue;
  else;    sv=exp(dv/10)-1;  clr=c_coral; end
  scatter(dv,sv,80,clr,'filled','HandleVisibility','off');
  text(dv+(dv>0)*3+(dv<0)*-18, sv+4, ...
       sprintf('d=%+d\npenalty≈%.1f',dv,sv), ...
       'FontSize',8,'Color',clr);
end
text(-48,5,'EARLY\nConservative','Color',c_blue,'FontSize',9, ...
     'HorizontalAlignment','center','BackgroundColor','white','EdgeColor',c_blue);
text(42,5,'LATE\nDangerous','Color',c_coral,'FontSize',9, ...
     'HorizontalAlignment','center','BackgroundColor','white','EdgeColor',c_coral);
xlabel('d = estimated RUL − true RUL (cycles)');
ylabel('Penalty score s');
title({'NASA PHM08 Asymmetric Scoring Function', ...
       'Late predictions penalised more heavily'}, ...
      'Color',c_blue,'FontWeight','bold');
legend('Location','north','FontSize',9);
xlim([-65 65]); ylim([-2 95]); grid on; box off;
save_fig(fig9,'matlab_D11_nasa_score.png',data_dir);

% ══════════════════════════════════════════════════════════════════════════════
%% Figure 10: RUL cap
% ══════════════════════════════════════════════════════════════════════════════

fprintf('Figure 10: RUL cap...\n');
fig10 = figure('Color','white','Position',[50 50 1100 500]);
for e=1:2
  eng = [1 77]; eng_e = eng(e);
  subplot(1,2,e);
  mask = T_raw.unit_number==eng_e;
  t_e  = T_raw.time_cycles(mask);
  raw  = T_raw.RUL_raw(mask);
  cap  = T_raw.RUL_capped(mask);
  [t_e,si]=sort(t_e); raw=raw(si); cap=cap(si);
  deg_s = t_e(find(raw<125,1));
  patch([t_e(1) deg_s deg_s t_e(1)],[-5 -5 max(raw)+15 max(raw)+15], ...
        c_blue,'FaceAlpha',0.06,'EdgeColor','none'); hold on;
  patch([deg_s t_e(end) t_e(end) deg_s],[-5 -5 max(raw)+15 max(raw)+15], ...
        c_coral,'FaceAlpha',0.06,'EdgeColor','none');
  plot(t_e,raw,'--','Color',c_blue,'LineWidth',2,'DisplayName','RUL raw');
  plot(t_e,cap,'-', 'Color',c_teal,'LineWidth',2.5,'DisplayName','RUL capped');
  yline(125,':','Color',c_gray,'LineWidth',1,'HandleVisibility','off');
  text(t_e(1)+1,127,'cap=125','Color',c_gray,'FontSize',8);
  xlabel('Time cycles'); ylabel('RUL (cycles)');
  title(sprintf('Engine %d — total life = %d cycles',eng_e,t_e(end)), ...
        'Color',c_blue,'FontWeight','bold');
  legend('Location','northeast','FontSize',8);
  ylim([-5 max(raw)+15]); grid on; box off;
end
sgtitle({'RUL Raw vs RUL Capped (cap=125) — C-MAPSS FD001', ...
         'Two engines: cap works consistently regardless of lifespan'}, ...
        'Color',c_blue,'FontWeight','bold');
save_fig(fig10,'matlab_D12_rul_cap.png',data_dir);

% ── summary ───────────────────────────────────────────────────────────────────

fprintf('\nAll figures saved:\n');
figs = {'matlab_A1_degradation.png','matlab_A3_scree.png', ...
        'matlab_A4_pca_scatter.png','matlab_B5_loss_curve.png', ...
        'matlab_B6_true_vs_pred.png','matlab_B7_error_dist.png', ...
        'matlab_C8_garson_norm.png','matlab_C10_garson_pca.png', ...
        'matlab_D11_nasa_score.png','matlab_D12_rul_cap.png'};
for i=1:length(figs)
  fprintf('  %2d. %s\n',i,figs{i});
end

% ══════════════════════════════════════════════════════════════════════════════
% LOCAL FUNCTIONS
% ══════════════════════════════════════════════════════════════════════════════

function save_fig(fig_h, filename, data_dir)
  out = fullfile(data_dir, filename);
  exportgraphics(fig_h, out, 'Resolution',150, 'BackgroundColor','white');
  fprintf('  Saved: %s\n', out);
end

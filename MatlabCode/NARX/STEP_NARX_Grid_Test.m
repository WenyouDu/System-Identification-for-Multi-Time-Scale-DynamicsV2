% STEP_NARX_Grid_Test.m
% =========================================================================
% Nonlinear system NARX grid evaluation.
% Loops over every network saved by STEP_NARX_Grid_Search.m
% (NARX_net_d<delay>_h<hidden>_seed<seed>.mat), closes the loop and
% simulates on the nonlinear test data over the full 8000-step horizon.
% Per-seed metrics are printed as a table, then aggregated per (hidden,
% delay) configuration as mean +/- std over the seeds.
%
% Metrics (normalized domain with training-output min/max, consistent with
% the Python pipeline and STEP_NARX_Test.m):
%   R2     : coefficient of determination (sklearn r2_score, uniform
%            average over channels)
%   RMSE, MAE, NRMSE, FreqErr (complex-spectrum L1 over 0~Nyquist),
%   AvgAbs (magnitude-spectrum dB error in [1,100] Hz).
%
% Input : Step_Train_Data_Nolinear.mat (normalization statistics) and
%         Step_Test_Data_Nolinear.mat (evaluation), 8000 steps, 6 inputs,
%         3 outputs, fs = 1000.
%
% Requires: Deep Learning Toolbox
% =========================================================================

scriptDir = fileparts(mfilename('fullpath'));
baseDir = fullfile(scriptDir, '..');
tr = load(fullfile(baseDir, 'Data', 'Step_Train_Data_Nolinear.mat'));
te = load(fullfile(baseDir, 'Data', 'Step_Test_Data_Nolinear.mat'));

FS = 1000;

tmin  = min(tr.Train_targets_Nolinear, [], 2);   % [3 x 1]
tmax  = max(tr.Train_targets_Nolinear, [], 2);
tspan = tmax - tmin;
tspan(tspan < 1e-12) = 1;

inputs  = con2seq(te.Test_inputs_Nolinear);
targets = con2seq(te.Test_targets_Nolinear);

modelDir = fullfile(scriptDir, 'NARX_STEP_Model');
files = dir(fullfile(modelDir, 'NARX_net_d*_h*_seed*.mat'));

fprintf('h      d      seed   R2        RMSE      MAE       NRMSE     FreqErr   AvgAbs\n');
rows = {};
for i = 1:numel(files)
    tok = regexp(files(i).name, 'NARX_net_d(\d+)_h(\d+)_seed(\d+)', 'tokens');
    d = str2double(tok{1}{1});
    h = str2double(tok{1}{2});
    s = str2double(tok{1}{3});

    varname = sprintf('STEP_NARX_d%d_h%d_seed%d', d, h, s);
    S = load(fullfile(modelDir, files(i).name));
    netOpen = S.(varname).Network;
    netClosed = closeloop(netOpen);

    [Xs, Xi, Ai, Ts] = preparets(netClosed, inputs, {}, targets);
    Ypred = netClosed(Xs, Xi, Ai);
    Y_pred = cell2mat(Ypred);                   % [3 x (N-d)]
    Y_aligned = cell2mat(Ts);                   % [3 x (N-d)] aligned with Y_pred

    Tn = (Y_aligned - tmin) ./ tspan;           % [3 x (N-d)]
    Yn = (Y_pred - tmin) ./ tspan;
    e  = Tn - Yn;

    % R2: coefficient of determination, uniform average over channels
    % (identical to sklearn r2_score in the Python pipeline)
    ss_res = sum(e.^2, 2);
    ss_tot = sum((Tn - mean(Tn, 2)).^2, 2);
    R2v = mean(1 - ss_res ./ ss_tot);

    rmse_ch = sqrt(mean(e.^2, 2));
    mae_ch  = mean(abs(e), 2);
    span_n  = max(Tn, [], 2) - min(Tn, [], 2);
    span_n(span_n < 1e-12) = 1;
    nrmse_ch = rmse_ch ./ span_n;

    n = size(Tn, 2);
    half = 1:floor(n/2) + 1;
    Fp = fft(Yn, n, 2); Fp = Fp(:, half);
    Ft = fft(Tn, n, 2); Ft = Ft(:, half);
    ferr_ch = mean(abs(real(Fp) - real(Ft)) + abs(imag(Fp) - imag(Ft)), 2);

    freqs = (0:floor(n/2)) * FS / n;
    mask = (freqs >= 1) & (freqs <= 100);
    ratio = max(min(abs(Fp) ./ max(abs(Ft), 1e-12), 1e2), 1e-2);
    db = 20 * log10(ratio);
    aad_ch = mean(abs(db(:, mask)), 2);

    RMs = mean(rmse_ch); MAs = mean(mae_ch); NRs = mean(nrmse_ch);
    FQs = mean(ferr_ch); ABs = mean(aad_ch);

    fprintf('%-5d %-5d %-6d %-9.4f %-9.5f %-8.5f %-9.5f %-9.5f %-8.5f\n', ...
        h, d, s, R2v, RMs, MAs, NRs, FQs, ABs);
    rows{end+1} = [h, d, R2v, RMs, MAs, NRs, FQs, ABs]; %#ok<AGROW>
end

%% Aggregate per (hidden, delay): mean +/- std over seeds
R = cell2mat(rows(:));
combo = unique(R(:, 1:2), 'rows');
fprintf('\n================ Grid summary (mean +/- std over seeds) ================\n');
fprintf('hidden delay  R2                RMSE             MAE              NRMSE            FreqErr          AvgAbs\n');
for i = 1:size(combo, 1)
    h = combo(i, 1); d = combo(i, 2);
    sel = (R(:, 1) == h) & (R(:, 2) == d);
    M = R(sel, 3:end);
    fprintf('%-6d %-5d ', h, d);
    for k = 1:size(M, 2)
        fprintf('%.4f±%.4f  ', mean(M(:, k)), std(M(:, k)));
    end
    fprintf('\n');
end

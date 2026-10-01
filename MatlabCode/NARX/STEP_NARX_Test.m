% STEP_NARX_Test.m
% =========================================================================
% Nonlinear system NARX closed-loop (free-run) evaluation, 5 seeds.
% Loads each open-loop network from ./NARX_STEP_Model, closes the loop and
% simulates on the nonlinear test data over the full 8000-step horizon.
%
% Metrics in the normalized domain (training-output min/max), consistent
% with the Python pipeline:
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

SEEDS = [9 66 108 88 52];

R2s = zeros(numel(SEEDS), 1); RMs = R2s; MAs = R2s;
NRs = R2s; FQs = R2s; ABs = R2s;

fprintf('%-6s %-10s %-12s %-12s %-12s %-12s %-12s\n', ...
    'seed','R2','RMSE','MAE','NRMSE','FreqErr','AvgAbs');

for i = 1:numel(SEEDS)
    s = SEEDS(i);
    varname = sprintf('STEP_NARX_seed%d', s);
    S = load(fullfile(scriptDir, 'NARX_STEP_Model', sprintf('STEP_NARX_seed%d.mat', s)));
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
    R2s(i) = mean(1 - ss_res ./ ss_tot);

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

    RMs(i) = mean(rmse_ch); MAs(i) = mean(mae_ch); NRs(i) = mean(nrmse_ch);
    FQs(i) = mean(ferr_ch); ABs(i) = mean(aad_ch);

    fprintf('%-6d %-10.4f %-12.5f %-12.5f %-12.5f %-12.5f %-12.5f\n', ...
        s, R2s(i), RMs(i), MAs(i), NRs(i), FQs(i), ABs(i));
end

%% Summary: mean +/- std over seeds
fprintf('\n================ STEP NARX %d-seed summary (mean +/- std) ================\n', numel(SEEDS));
fmt = @(x) sprintf('%.4f+/-%.4f', mean(x), std(x));
fprintf('R2         : %s\n', fmt(R2s));
fprintf('RMSE       : %s\n', fmt(RMs));
fprintf('MAE        : %s\n', fmt(MAs));
fprintf('NRMSE      : %s\n', fmt(NRs));
fprintf('FreqErr    : %s\n', fmt(FQs));
fprintf('AvgAbs     : %s\n', fmt(ABs));

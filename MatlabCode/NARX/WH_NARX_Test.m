% WH_NARX_Test.m
% =========================================================================
% WH (Wiener-Hammerstein) NARX closed-loop (free-run) evaluation, 5 seeds.
% Loads each open-loop network from ./NARX_WH_Model, closes the loop and
% simulates on the WH test data (78800 steps, single channel).
%
% Metrics in the normalized domain (training-output min/max), consistent
% with the Python pipeline:
%   R2(corr)   : squared Pearson correlation
%   R2(det)    : coefficient of determination
%   RMSE, MAE, NRMSE, FreqErr (complex-spectrum L1 over 0~Nyquist),
%   AvgAbs (magnitude-spectrum dB error in [1,100] Hz),
%   scaledRMSE (raw-domain RMSE x 1000).
%
% Input : WH_processed.mat (WH_train_y/WH_test_u/WH_test_y)
%         and ./NARX_WH_Model/WH_NARX_seed<seed>.mat (from WH_NARX_Train.m).
%
% Requires: Deep Learning Toolbox
% =========================================================================

scriptDir = fileparts(mfilename('fullpath'));
baseDir = fullfile(scriptDir, '..');
S = load(fullfile(baseDir, 'Data', 'WH_processed.mat'));

WH_train_y = S.WH_train_y;   % [100000 x 1]
WH_test_u  = S.WH_test_u;    % [78800 x 1]
WH_test_y  = S.WH_test_y;    % [78800 x 1]

SEEDS = [9 66 108 88 52];
FS    = 51200;         % WH sampling frequency (Hz)
SKIP  = 50;            % skip the first 50 samples (free-run initial transient),
                       % consistent with the Python benchmark protocol

% Normalization baseline: training-output min/max (single channel)
tmin  = min(WH_train_y);
tmax  = max(WH_train_y);
tspan = max(tmax - tmin, eps);

% Test data: column vectors transposed for con2seq (WH_NARX_Free_Run_Test.m)
inputs  = con2seq(WH_test_u');
targets = con2seq(WH_test_y');

fprintf('%-6s %-10s %-10s %-12s %-12s %-12s %-12s %-12s %-10s\n', ...
    'seed','R2(corr)','R2(det)','RMSE','MAE','NRMSE','FreqErr','AvgAbs','scaledRMSE');

R2s = []; R2d = []; RMs = []; MAs = []; NRs = []; FQs = []; ABs = []; SRs = [];
for s = SEEDS
    varname = sprintf('WH_NARX_seed%d', s);
    M = load(fullfile(scriptDir, 'NARX_WH_Model', sprintf('WH_NARX_seed%d.mat', s)));
    netOpen   = M.(varname).Network;
    netClosed = closeloop(netOpen);

    [Xs, Xi, Ai, Ts] = preparets(netClosed, inputs, {}, targets);
    [y_pred_cell, ~, ~] = sim(netClosed, Xs, Xi, Ai);

    y_pred = cell2mat(y_pred_cell);
    y_true = cell2mat(Ts);

    % Skip the first SKIP samples (initial transient).
    Tn = (y_true(SKIP+1:end) - tmin) ./ tspan;
    Yn = (y_pred(SKIP+1:end) - tmin) ./ tspan;
    e  = Tn - Yn;

    rr = corrcoef(Tn, Yn); R2c = rr(1,2)^2;
    R2d_ = 1 - sum(e.^2) ./ sum((Tn - mean(Tn)).^2);

    rmse_n = sqrt(mean(e.^2));
    mae_n  = mean(abs(e));
    span = max(Tn) - min(Tn); if span < 1e-12, span = 1; end
    nrmse_n = rmse_n / span;

    N = numel(Tn);
    Fp = fft(Yn); Ft = fft(Tn);
    nF = floor(N/2) + 1;
    Fp = Fp(1:nF); Ft = Ft(1:nF);
    freqerr = mean(abs(real(Fp) - real(Ft)) + abs(imag(Fp) - imag(Ft)));

    freqs = (0:nF-1) * FS / N;
    Fp_mag = abs(Fp); Ft_mag = abs(Ft);
    ratio = min(max(Fp_mag ./ max(Ft_mag, 1e-12), 1e-2), 1e2);
    db = 20 * log10(ratio);
    mask = (freqs >= 1) & (freqs <= 100);
    avgabs = mean(abs(db(mask)));

    rmse_raw = sqrt(mean((y_true(SKIP+1:end) - y_pred(SKIP+1:end)).^2));
    scaled_rmse = rmse_raw * 1000;

    R2s(end+1) = R2c; R2d(end+1) = R2d_; RMs(end+1) = rmse_n; 
    MAs(end+1) = mae_n; NRs(end+1) = nrmse_n; FQs(end+1) = freqerr; 
    ABs(end+1) = avgabs; SRs(end+1) = scaled_rmse; 

    fprintf('%-6d %-10.4f %-10.4f %-12.5f %-12.5f %-12.5f %-12.5f %-12.4f %-10.4f\n', ...
        s, R2c, R2d_, rmse_n, mae_n, nrmse_n, freqerr, avgabs, scaled_rmse);
end

%% Summary mean +/- std
fprintf('\n================ WH NARX %d-seed summary (mean +/- std) ================\n', numel(SEEDS));
fmt = @(x) sprintf('%.4f±%.4f', mean(x), std(x));
fprintf('R2(corr)  : %s\n', fmt(R2s));
fprintf('R2(det)   : %s\n', fmt(R2d));
fprintf('RMSE      : %s\n', fmt(RMs));
fprintf('MAE       : %s\n', fmt(MAs));
fprintf('NRMSE     : %s\n', fmt(NRs));
fprintf('FreqErr   : %s\n', fmt(FQs));
fprintf('AvgAbs    : %s\n', fmt(ABs));
fprintf('scaledRMSE: %s\n', fmt(SRs));

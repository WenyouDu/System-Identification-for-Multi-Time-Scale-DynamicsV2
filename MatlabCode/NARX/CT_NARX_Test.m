% CT_NARX_Test.m
% =========================================================================
% CT (Cascaded Tanks) NARX closed-loop (free-run) evaluation, 5 seeds.
% Loads each open-loop network from ./NARX_CT_Model, closes the loop and
% simulates on the CT test data (1024 steps).
%
% Metrics in the normalized domain (training-output min/max), consistent
% with the Python pipeline:
%   R2(corr)   : squared Pearson correlation
%   R2(det)    : coefficient of determination
%   RMSE, MAE, NRMSE, FreqErr (complex-spectrum L1 over 0~Nyquist),
%   AvgAbs (magnitude-spectrum dB error in [1,100] Hz),
%   scaledRMSE (raw-domain RMSE x 1.0).
%
% Input : CT_processed.mat (CT_X_train/CT_Y_train/CT_X_test/CT_Y_test)
%         and ./NARX_CT_Model/CT_NARX_seed<seed>.mat (from CT_NARX_Train.m).
%
% Requires: Deep Learning Toolbox
% =========================================================================

scriptDir = fileparts(mfilename('fullpath'));
baseDir = fullfile(scriptDir, '..');
S = load(fullfile(baseDir, 'Data', 'CT_processed.mat'));

CT_X_train = S.CT_X_train;
CT_Y_train = S.CT_Y_train;
CT_X_test  = S.CT_X_test;
CT_Y_test  = S.CT_Y_test;

SEEDS = [9 66 108 88 52];
FS    = 0.25;         % CT sampling frequency (Hz)
SKIP  = 50;           % skip the first 50 samples (free-run initial transient),
                      % consistent with the Python benchmark protocol

% Normalization baseline: per-channel min/max of the training output
tmin  = min(CT_Y_train, [], 2);
tmax  = max(CT_Y_train, [], 2);
tspan = max(tmax - tmin, eps);

% Test data (same convention as CT_NARX_Free_Run_Test.m)
inputs  = con2seq(CT_X_test);
targets = con2seq(CT_Y_test);

fprintf('%-6s %-10s %-10s %-12s %-12s %-12s %-12s %-12s %-10s\n', ...
    'seed','R2(corr)','R2(det)','RMSE','MAE','NRMSE','FreqErr','AvgAbs','scaledRMSE');

Res = zeros(numel(SEEDS), 9);   % seed R2c R2d RMSE MAE NRMSE FreqErr AvgAbs scaledRMSE
for si = 1:numel(SEEDS)
    s = SEEDS(si);
    varname = sprintf('CT_NARX_seed%d', s);
    M = load(fullfile(scriptDir, 'NARX_CT_Model', sprintf('CT_NARX_seed%d.mat', s)));
    netOpen   = M.(varname).Network;
    netClosed = closeloop(netOpen);

    [Xs, Xi, Ai, Ts] = preparets(netClosed, inputs, {}, targets);
    [y_pred_cell, ~, ~] = sim(netClosed, Xs, Xi, Ai);

    y_pred = cell2mat(y_pred_cell);
    y_true = cell2mat(Ts);
    nCh = size(y_true, 1);

    % Per-channel metrics, averaged over channels (Python calc_metrics).
    % The first SKIP samples are skipped (initial transient).
    R2c = []; R2d_ = []; rms = []; mas = []; nrs = []; fqs = []; abs_ = [];
    for c = 1:nCh
        Tn = (y_true(c, SKIP+1:end) - tmin(c)) ./ tspan(c);
        Yn = (y_pred(c, SKIP+1:end) - tmin(c)) ./ tspan(c);
        e  = Tn - Yn;

        rr = corrcoef(Tn, Yn); R2c(end+1) = rr(1,2)^2; 
        R2d_(end+1) = 1 - sum(e.^2) ./ sum((Tn - mean(Tn)).^2); 

        rms(end+1) = sqrt(mean(e.^2));
        mas(end+1) = mean(abs(e)); 
        sp = max(Tn) - min(Tn); if sp < 1e-12, sp = 1; end
        nrs(end+1) = rms(end) / sp; 

        n = numel(Tn);
        Yf = fft(Yn); Tf = fft(Tn);
        nF = floor(n/2) + 1;
        Yh = Yf(1:nF); Th = Tf(1:nF);
        fqs(end+1) = mean(abs(real(Yh) - real(Th)) + abs(imag(Yh) - imag(Th))); 

        freqs = (0:nF-1) * FS / n;
        Ym = abs(Yh); Tm = abs(Th);
        ratio = min(max(Ym ./ max(Tm, 1e-12), 1e-2), 1e2);
        db = 20 * log10(ratio);
        mask = (freqs >= 1) & (freqs <= 100);
        if ~any(mask), mask = true(size(freqs)); end
        abs_(end+1) = mean(abs(db(mask))); 
    end

    % Raw-domain scaled RMSE (x 1.0), also after skipping SKIP samples
    rmse_raw = sqrt(mean((y_true(:, SKIP+1:end) - y_pred(:, SKIP+1:end)).^2, 'all'));
    scaled_rmse = rmse_raw * 1.0;

    Res(si, :) = [s, mean(R2c), mean(R2d_), mean(rms), mean(mas), mean(nrs), mean(fqs), mean(abs_), scaled_rmse];
    fprintf('%-6d %-10.4f %-10.4f %-12.5f %-12.5f %-12.5f %-12.5f %-12.4f %-10.4f\n', ...
        Res(si,1), Res(si,2), Res(si,3), Res(si,4), Res(si,5), Res(si,6), Res(si,7), Res(si,8), Res(si,9));
end

%% Summary: mean +/- std over seeds
fprintf('\n================ CT NARX %d-seed summary (mean +/- std) ================\n', numel(SEEDS));
fmt = @(x) sprintf('%.4f±%.4f', mean(x), std(x));
fprintf('R2(corr)  : %s\n', fmt(Res(:,2)));
fprintf('R2(det)   : %s\n', fmt(Res(:,3)));
fprintf('RMSE      : %s\n', fmt(Res(:,4)));
fprintf('MAE       : %s\n', fmt(Res(:,5)));
fprintf('NRMSE     : %s\n', fmt(Res(:,6)));
fprintf('FreqErr   : %s\n', fmt(Res(:,7)));
fprintf('AvgAbs    : %s\n', fmt(Res(:,8)));
fprintf('scaledRMSE: %s\n', fmt(Res(:,9)));

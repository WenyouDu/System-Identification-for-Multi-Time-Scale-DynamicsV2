% SB_NARX_Test.m
% =========================================================================
% SB (Silver Box) NARX closed-loop (free-run) evaluation, 5 seeds.
% Loads each open-loop network from ./NARX_SB_Model, closes the loop and
% simulates on the three SB test sets (Arrow Full / Arrow No
% Extrapolation / Multisine). Metrics are computed per test set.
%
% Metrics in the normalized domain (training-output min/max), consistent
% with the Python pipeline:
%   R2(corr)   : squared Pearson correlation
%   R2(det)    : coefficient of determination
%   RMSE, MAE, NRMSE, FreqErr (complex-spectrum L1 over 0~Nyquist),
%   AvgAbs (magnitude-spectrum dB error in [1,100] Hz),
%   scaledRMSE (raw-domain RMSE x 1000).
%
% Input : SB_processed.mat (SB_X_train/SB_Y_train + the three test sets)
%         and ./NARX_SB_Model/SB_NARX_seed<seed>.mat (from SB_NARX_Train.m).
%
% Requires: Deep Learning Toolbox
% =========================================================================

scriptDir = fileparts(mfilename('fullpath'));
baseDir = fullfile(scriptDir, '..');
S = load(fullfile(baseDir, 'Data', 'SB_processed.mat'));

SB_X_train = S.SB_X_train;
SB_Y_train = S.SB_Y_train;

SEEDS = [9 66 108 88 52];
FS    = 610.35;        % SB sampling frequency (Hz)
SKIP  = 50;            % skip the first 50 samples (free-run initial transient),
                       % consistent with the Python benchmark protocol

test_names  = {'Arrow Full', 'Arrow No Extrapolation', 'Multisine'};
test_inputs = {S.SB_X_test_arrow_full, S.SB_X_test_arrow_no_extrapolation, S.SB_X_test_multisine};
test_targets = {S.SB_Y_test_arrow_full, S.SB_Y_test_arrow_no_extrapolation, S.SB_Y_test_multisine};
n_ts = numel(test_names);

% Normalization baseline: per-channel min/max of the training output
tmin  = min(SB_Y_train, [], 2);
tmax  = max(SB_Y_train, [], 2);
tspan = max(tmax - tmin, eps);

% per_test_set{ts} = [seed R2c R2d RMSE MAE NRMSE FreqErr AvgAbs scaledRMSE]
per_ts = cell(1, n_ts);
for ts = 1:n_ts
    per_ts{ts} = zeros(numel(SEEDS), 9);
end

%% Per seed, per test set
for si = 1:numel(SEEDS)
    s = SEEDS(si);
    varname = sprintf('SB_NARX_seed%d', s);
    M = load(fullfile(scriptDir, 'NARX_SB_Model', sprintf('SB_NARX_seed%d.mat', s)));
    netOpen   = M.(varname).Network;
    netClosed = closeloop(netOpen);

    for ts = 1:n_ts
        u_test = test_inputs{ts};
        y_test = test_targets{ts};

        inputs  = con2seq(u_test);
        targets = con2seq(y_test);
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

        % Raw-domain scaled RMSE (x 1000), also after skipping SKIP samples
        rmse_raw = sqrt(mean((y_true(:, SKIP+1:end) - y_pred(:, SKIP+1:end)).^2, 'all'));
        scaled_rmse = rmse_raw * 1000;

        per_ts{ts}(si, :) = [s, mean(R2c), mean(R2d_), mean(rms), mean(mas), mean(nrs), mean(fqs), mean(abs_), scaled_rmse];
    end
end

%% Per-test-set table
fprintf('%-6s %-12s %-10s %-10s %-12s %-12s %-12s %-12s %-12s %-10s\n', ...
    'seed','test_set','R2(corr)','R2(det)','RMSE','MAE','NRMSE','FreqErr','AvgAbs','scaledRMSE');
for ts = 1:n_ts
    for si = 1:numel(SEEDS)
        r = per_ts{ts}(si, :);
        fprintf('%-6d %-12s %-10.4f %-10.4f %-12.5f %-12.5f %-12.5f %-12.5f %-12.4f %-10.4f\n', ...
            r(1), test_names{ts}, r(2), r(3), r(4), r(5), r(6), r(7), r(8), r(9));
    end
end

%% Summary: per test set mean +/- std over seeds, plus overall average
fprintf('\n================ SB NARX %d-seed summary (mean +/- std) ================\n', numel(SEEDS));
fprintf('%-24s %-16s %-16s %-14s %-14s %-14s %-14s %-14s %-12s\n', ...
    'test_set','R2(corr)','R2(det)','RMSE','MAE','NRMSE','FreqErr','AvgAbs','scaledRMSE');
fmt = @(x) sprintf('%.4f±%.4f', mean(x), std(x));
overall = zeros(1, 8);
for ts = 1:n_ts
    A = per_ts{ts};   % each row: seed, columns 2..9
    fprintf('%-24s %-16s %-16s %-14s %-14s %-14s %-14s %-14s %-12s\n', ...
        test_names{ts}, fmt(A(:,2)), fmt(A(:,3)), fmt(A(:,4)), fmt(A(:,5)), ...
        fmt(A(:,6)), fmt(A(:,7)), fmt(A(:,8)), fmt(A(:,9)));
    overall = overall + mean(A(:, 2:9), 1);
end
overall = overall / n_ts;
fprintf('%-24s %-16s %-16s %-14s %-14s %-14s %-14s %-14s %-12s\n', ...
    '[overall-mean]', fmt(overall(1)), fmt(overall(2)), fmt(overall(3)), fmt(overall(4)), ...
    fmt(overall(5)), fmt(overall(6)), fmt(overall(7)), fmt(overall(8)));

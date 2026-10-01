% WH_SS_Test.m
% =========================================================================
% WH (Wiener-Hammerstein) state-space model evaluation with order sweep.
% For each model in ./SS_WH_Model (ss<order>_WH.mat), simulate it on the WH
% test data and report metrics in the normalized domain.
%
% Simulation follows the System Identification app convention:
%   continuous model -> c2d to Ts=1/fs (zoh) -> sim (free-run, Horizon:
%   Simulation).
%
% Metrics in the normalized domain (training-output min/max), consistent
% with the NARX/Python pipeline:
%   R2(corr)   : squared Pearson correlation
%   R2(det)    : coefficient of determination
%   RMSE, MAE, NRMSE, FreqErr (complex-spectrum L1 over 0~Nyquist),
%   AvgAbs (magnitude-spectrum dB error in [1,100] Hz),
%   scaledRMSE (raw-domain RMSE x 1000).
%
% Input : WH_processed.mat (training statistics + test data) and
%         ./SS_WH_Model/ss<order>_WH.mat (from WH_SS_Train.m).
%
% Requires: System Identification Toolbox
% =========================================================================

scriptDir = fileparts(mfilename('fullpath'));
baseDir = fullfile(scriptDir, '..');
S = load(fullfile(baseDir, 'Data', 'WH_processed.mat'));

U_tr = S.WH_train_u;   Y_tr = S.WH_train_y;
U_te = S.WH_test_u;    Y_te = S.WH_test_y;
fs = S.fs(1);

tmin  = min(Y_tr, [], 1);
tmax  = max(Y_tr, [], 1);
tspan = max(tmax - tmin, eps);
Y_norm = (Y_te - tmin) ./ tspan;    % [N x 1]

Ts    = 1/fs;    % WH sampling interval (s)
FS    = fs;      % WH sampling frequency (Hz)
SCALE = 1000;    % raw-domain RMSE scaling factor
SKIP  = 50;      % skip the first 50 samples (free-run initial transient),
                 % consistent with the Python benchmark protocol

orders = 1:10;   % must match WH_SS_Train.m

fprintf('%-5s %-10s %-10s %-12s %-12s %-12s %-12s %-12s %-10s\n', ...
    'order','R2(corr)','R2(det)','RMSE','MAE','NRMSE','FreqErr','AvgAbs','scaledRMSE');
for o = orders
    M = load(fullfile(scriptDir, 'SS_WH_Model', sprintf('ss%d_WH.mat', o)));
    model = M.model;
    model_d = c2d(model, Ts, 'zoh');
    sim_in = iddata([], U_te, Ts);
    sim_out = sim(model_d, sim_in);
    Y_hat = sim_out.OutputData;     % [N x 1]
    Y_hat_norm = (Y_hat - tmin) ./ tspan;

    % Evaluate after skipping the first SKIP samples (initial transient).
    Yn_s = Y_norm(SKIP+1:end, :);
    Yh_s = Y_hat_norm(SKIP+1:end, :);
    [r2c, r2d, rmse, mae, nrmse, ferr, aad] = ...
        compute_metrics_py(Yn_s', Yh_s', FS);
    rmse_raw = sqrt(mean((Y_te(SKIP+1:end,:) - Y_hat(SKIP+1:end,:)).^2, 'all'));
    fprintf('%5d  %-10.4f %-10.4f %-12.5f %-12.5f %-12.5f %-12.5f %-12.4f %-10.4f\n', ...
        o, r2c, r2d, rmse, mae, nrmse, ferr, aad, rmse_raw * SCALE);
end

%% ================= helper: metrics consistent with Python =================
function [r2c, r2d, rmse, mae, nrmse, ferr, aad] = ...
    compute_metrics_py(y_true, y_pred, fs)
% y_true/y_pred: [C x T], rows = channels. Returns channel-averaged values.

diff = y_true - y_pred;

rmse = mean(sqrt(mean(diff.^2, 2)));
mae  = mean(mean(abs(diff), 2));

span = max(y_true, [], 2) - min(y_true, [], 2);
span(span < 1e-12) = 1;
nrmse = mean(sqrt(mean(diff.^2, 2)) ./ span);

n = size(y_true, 2);
half = 1:floor(n/2)+1;
fft_pred = fft(y_pred, n, 2); fft_pred = fft_pred(:, half);
fft_true = fft(y_true, n, 2); fft_true = fft_true(:, half);
ferr = mean(mean(abs(real(fft_pred) - real(fft_true)) + ...
                abs(imag(fft_pred) - imag(fft_true)), 2));

freqs = (0:floor(n/2)) * fs / n;
mask  = (freqs >= 1) & (freqs <= 100);
amp_pred = abs(fft_pred);
amp_true = abs(fft_true);
ratio = max(min(amp_pred ./ max(amp_true, 1e-12), 1e2), 1e-2);
spec_db = 20 * log10(ratio);
aad = mean(mean(abs(spec_db(:, mask)), 2));

ss_res = sum(diff.^2, 2);
ss_tot = sum((y_true - mean(y_true, 2)).^2, 2);
r2d = mean(1 - ss_res ./ max(ss_tot, 1e-12));

rr = zeros(size(y_true, 1), 1);
for c = 1:size(y_true, 1)
    cc = corrcoef(y_true(c,:), y_pred(c,:));
    rr(c) = cc(1,2)^2;
end
r2c = mean(rr);
end

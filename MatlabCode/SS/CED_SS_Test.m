% CED_SS_Test.m
% =========================================================================
% CED (Coupled Electric Drives) state-space model evaluation with order sweep.
% For each model in ./SS_CED_Model (ss<order>_CED.mat), simulate it on the
% CED test data and report metrics in the normalized domain.
%
% Simulation follows the System Identification app convention:
%   continuous model -> c2d to Ts=0.02 (zoh) -> sim (free-run, Horizon:
%   Simulation).
%
% Metrics in the normalized domain (training-output min/max), consistent
% with the NARX/Python pipeline:
%   R2(corr)   : squared Pearson correlation
%   R2(det)    : coefficient of determination
%   RMSE, MAE, NRMSE, FreqErr (complex-spectrum L1 over 0~Nyquist),
%   AvgAbs (magnitude-spectrum dB error in [1,100] Hz),
%   scaledRMSE (raw-domain RMSE, CED scaling = 1.0, unit ticks/s).
%
% Two independent test segments (low / high input amplitude, 100 samples
% each), matching the Python per-test-set protocol. The first 10 samples of
% each test segment are skipped (state_initialization_window_length = 10).
%
% Input : CED_processed.mat (training statistics + test data) and
%         ./SS_CED_Model/ss<order>_CED.mat (from CED_SS_Train.m).
%
% Requires: System Identification Toolbox
% =========================================================================

scriptDir = fileparts(mfilename('fullpath'));
baseDir = fullfile(scriptDir, '..');
S = load(fullfile(baseDir, 'Data', 'CED_processed.mat'));

U_tr = S.CED_X_train';   Y_tr = S.CED_Y_train';
fs = S.CED_fs(1);

tmin  = min(Y_tr, [], 1);
tmax  = max(Y_tr, [], 1);
tspan = max(tmax - tmin, eps);

Ts    = 0.02;   % CED sampling interval (s)
FS    = fs;     % CED sampling frequency (Hz)
SCALE = 1.0;    % raw-domain RMSE scaling factor (CED unit: ticks/s)
SKIP  = 10;     % skip the first 10 samples of each test segment
                % (official state_initialization_window_length, matches Python)

orders = 1:10;  % must match CED_SS_Train.m

test_segments = {S.CED_X_test0', S.CED_Y_test0', 'test0'; ...
                 S.CED_X_test1', S.CED_Y_test1', 'test1'};

fprintf('%-5s %-10s %-10s %-10s %-12s %-12s %-12s %-12s %-12s %-10s\n', ...
    'order','test_set','R2(corr)','R2(det)','RMSE','MAE','NRMSE','FreqErr','AvgAbs','scaledRMSE');
for o = orders
    M = load(fullfile(scriptDir, 'SS_CED_Model', sprintf('ss%d_CED.mat', o)));
    model = M.model;
    model_d = c2d(model, Ts, 'zoh');
    for ts = 1:size(test_segments, 1)
        U_te = test_segments{ts, 1};
        Y_te = test_segments{ts, 2};
        tname = test_segments{ts, 3};

        Y_norm = (Y_te - tmin) ./ tspan;    % [N x 1]
        sim_in = iddata([], U_te, Ts);
        sim_out = sim(model_d, sim_in);
        Y_hat = sim_out.OutputData;         % [N x 1]
        Y_hat_norm = (Y_hat - tmin) ./ tspan;

        % Evaluate after skipping the first SKIP samples (initial transient).
        Yn_s = Y_norm(SKIP+1:end, :);
        Yh_s = Y_hat_norm(SKIP+1:end, :);
        [r2c, r2d, rmse, mae, nrmse, ferr, aad] = ...
            compute_metrics_py(Yn_s', Yh_s', FS);
        rmse_raw = sqrt(mean((Y_te(SKIP+1:end,:) - Y_hat(SKIP+1:end,:)).^2, 'all'));
        fprintf('%5d  %-10s %-10.4f %-10.4f %-12.5f %-12.5f %-12.5f %-12.5f %-12.4f %-10.4f\n', ...
            o, tname, r2c, r2d, rmse, mae, nrmse, ferr, aad, rmse_raw * SCALE);
    end
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

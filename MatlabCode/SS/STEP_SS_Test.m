% STEP_SS_Test.m
% =========================================================================
% Nonlinear system SS model evaluation with order sweep.
% For each model in ./STEP_Model (ss<order>_Nolinear.mat), simulate it on
% the nonlinear test data and report metrics in the normalized domain.
%
% Simulation follows the System Identification app convention:
%   continuous model -> c2d to Ts=1 (zoh) -> sim with model initial state
%   (free-run, Horizon: Simulation).
%
% Metrics (normalized domain, consistent with the Python pipeline):
%   R2 (per-channel and global = uniform average), RMSE, MAE, NRMSE,
%   FreqErr (complex-spectrum L1), AvgAbs (magnitude-spectrum dB error in
%   [1,100] Hz with fs=1000).
%
% Input data: Step_Train_Data_Nolinear.mat (normalization statistics) and
%   Step_Test_Data_Nolinear.mat (evaluation), 8000 steps, 6 inputs, 3 outputs.
%
% Requires: System Identification Toolbox
% =========================================================================

scriptDir = fileparts(mfilename('fullpath'));
baseDir = fullfile(scriptDir, '..');
tr = load(fullfile(baseDir, 'Data', 'Step_Train_Data_Nolinear.mat'));
te = load(fullfile(baseDir, 'Data', 'Step_Test_Data_Nolinear.mat'));

U = te.Test_inputs_Nolinear';    % [N x 6]
Y = te.Test_targets_Nolinear;    % [3 x N]

Ytr = tr.Train_targets_Nolinear;   % [3 x N]
ymin = min(Ytr, [], 2);            % [3 x 1]
ymax = max(Ytr, [], 2);            % [3 x 1]
span = ymax - ymin;
span(span < 1e-12) = 1;
Y_norm = (Y - ymin) ./ span;       % [3 x N]

Ts = 1;
FS = 1000;

orders = 1:15;

fprintf('order  R2(glob)  R2(ch mean)  RMSE       MAE        NRMSE      FreqErr    AvgAbs\n');
for o = orders
    S = load(fullfile(scriptDir, 'STEP_Model', sprintf('ss%d_Nolinear.mat', o)));
    model = S.model;
    model_d = c2d(model, Ts, 'zoh');
    sim_in = iddata([], U, Ts);
    sim_out = sim(model_d, sim_in);
    Y_hat = sim_out.OutputData';   % [3 x N]
    Y_hat_norm = (Y_hat - ymin) ./ span;

    [r2_glob, r2_ch, rmse_ch, mae_ch, nrmse_ch, ferr_ch, aad_ch] = ...
        compute_metrics_py(Y_norm, Y_hat_norm, FS);
    fprintf('%5d  %.4f     %.4f      %.5f  %.5f  %.5f  %.5f  %.5f\n', ...
        o, r2_glob, mean(r2_ch), mean(rmse_ch), mean(mae_ch), ...
        mean(nrmse_ch), mean(ferr_ch), mean(aad_ch));
end

%% ================= helper: metrics consistent with Python =================
function [r2_glob, r2_ch, rmse_ch, mae_ch, nrmse_ch, ferr_ch, aad_ch] = ...
    compute_metrics_py(y_true, y_pred, fs)
% y_true/y_pred: [C x T], rows = channels
% Returns per-channel arrays and the global R2 (uniform average).

diff = y_true - y_pred;

% RMSE / MAE (per channel)
rmse_ch = sqrt(mean(diff.^2, 2));
mae_ch  = mean(abs(diff), 2);

% NRMSE = RMSE / range (per channel)
span = max(y_true, [], 2) - min(y_true, [], 2);
span(span < 1e-12) = 1;
nrmse_ch = rmse_ch ./ span;

% FreqErr: complex-spectrum L1 (0~Nyquist), averaged over frequencies
n = size(y_true, 2);
half = 1:floor(n/2)+1;
fft_pred = fft(y_pred, n, 2); fft_pred = fft_pred(:, half);
fft_true = fft(y_true, n, 2); fft_true = fft_true(:, half);
ferr_ch = mean(abs(real(fft_pred) - real(fft_true)) + ...
                 abs(imag(fft_pred) - imag(fft_true)), 2);

% AvgAbs: magnitude-spectrum dB error in [1,100] Hz
freqs = (0:floor(n/2)) * fs / n;
mask  = (freqs >= 1) & (freqs <= 100);
amp_pred = abs(fft_pred);
amp_true = abs(fft_true);
ratio = max(min(amp_pred ./ max(amp_true, 1e-12), 1e2), 1e-2);
spec_db = 20 * log10(ratio);
aad_ch = mean(abs(spec_db(:, mask)), 2);

% Per-channel R2 (coefficient of determination)
ss_res = sum(diff.^2, 2);
ss_tot = sum((y_true - mean(y_true, 2)).^2, 2);
r2_ch = zeros(size(y_true, 1), 1);
ok = ss_tot > 0;
r2_ch(ok) = 1 - ss_res(ok) ./ ss_tot(ok);
r2_ch(~ok) = NaN;

% Global R2 = uniform average of per-channel R2 (sklearn default)
r2_glob = mean(r2_ch);
end

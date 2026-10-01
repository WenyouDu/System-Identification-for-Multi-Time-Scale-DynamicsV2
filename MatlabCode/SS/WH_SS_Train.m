% WH_SS_Train.m
% =========================================================================
% WH (Wiener-Hammerstein) state-space model training with order sweep.
% Reproduces the System Identification app estimation exactly:
%   Options = n4sidOptions;
%   Options.Focus = 'simulation';
%   model = n4sid(mydata, order, 'Form', 'free', 'Ts', 0, Options);
%
% Input : WH_processed.mat (generated from WH.mat by the WH data-split
%         logic: sys_data = u(5201:184000), then 100000/78800 split),
%         containing WH_train_u [100000 x 1] and WH_train_y [100000 x 1]
%         (single-channel column vectors) and fs [1 x 1].
% Output: one model per order, saved to ./SS_WH_Model/ss<order>_WH.mat
%
% Requires: System Identification Toolbox
% Note: n4sid is deterministic, so identical configuration reproduces the
%       GUI estimation results exactly.
% =========================================================================

scriptDir = fileparts(mfilename('fullpath'));
dataFile  = fullfile(scriptDir, '..', 'Data', 'WH_processed.mat');
S = load(dataFile);

U = S.WH_train_u;     % [100000 x 1]
Y = S.WH_train_y;     % [100000 x 1]
fs = S.fs(1);         % WH sampling frequency (Hz)
Ts_sample = 1/fs;
mydata = iddata(Y, U, Ts_sample);

saveDir = fullfile(scriptDir, 'SS_WH_Model');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

orders = 1:10;

for o = orders
    Options = n4sidOptions;
    Options.Display = 'on';
    Options.Focus = 'simulation';
    model = n4sid(mydata, o, 'Form', 'free', 'Ts', 0, Options);
    save(fullfile(saveDir, sprintf('ss%d_WH.mat', o)), 'model');
end

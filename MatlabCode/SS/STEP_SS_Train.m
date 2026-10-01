% STEP_SS_Train.m
% =========================================================================
% Nonlinear system SS model training with order sweep.
% Reproduces the System Identification app estimation exactly:
%   Options = n4sidOptions;
%   Options.Focus = 'simulation';
%   model = n4sid(mydata, order, 'Form', 'free', 'Ts', 0, Options);
%
% Input : nonlinear training data (Step_Train_Data_Nolinear.mat, generated
%         by Train_Data_Preparation_Nolinear.m, sampling interval Ts=1)
% Output: one model per order, saved to ./STEP_Model/ss<order>_Nolinear.mat
%
% Requires: System Identification Toolbox
% Note: n4sid is deterministic, so identical configuration reproduces the
%       GUI estimation results exactly.
% =========================================================================

scriptDir = fileparts(mfilename('fullpath'));
dataFile  = fullfile(scriptDir, '..', 'Data', 'Step_Train_Data_Nolinear.mat');
S = load(dataFile);

U = S.Train_inputs_Nolinear';    % [N x 6]
Y = S.Train_targets_Nolinear';   % [N x 3]
Ts_sample = 1;
mydata = iddata(Y, U, Ts_sample);

saveDir = fullfile(scriptDir, 'STEP_Model');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

orders = 1:15;

for o = orders
    Options = n4sidOptions;
    Options.Display = 'on';
    Options.Focus = 'simulation';
    model = n4sid(mydata, o, 'Form', 'free', 'Ts', 0, Options);
    save(fullfile(saveDir, sprintf('ss%d_Nolinear.mat', o)), 'model');
end

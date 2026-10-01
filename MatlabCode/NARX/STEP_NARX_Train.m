% STEP_NARX_Train.m
% =========================================================================
% Nonlinear system NARX training with 5 random seeds.
% Reproduces the Neural Time Series app workflow exactly:
%   narxnet(1:10, 1:10, 10, 'open', 'trainlm'), 70/15/15 data division,
%   rng(seed) fixes both weight initialization and data division.
% The network is trained purely in open loop (no closed-loop fine-tuning).
%
% Input : nonlinear training data (Step_Train_Data_Nolinear.mat, generated
%         by Train_Data_Preparation_Nolinear.m), 8000 steps, 6 inputs,
%         3 outputs.
% Output: one network per seed, saved to
%         ./NARX_STEP_Model/STEP_NARX_seed<seed>.mat with top-level
%         variables STEP_NARX_seed<seed>.Network and .TrainingResults
%         (GUI export convention).
%
% Requires: Deep Learning Toolbox
% =========================================================================

scriptDir = fileparts(mfilename('fullpath'));
dataFile  = fullfile(scriptDir, '..', 'Data', 'Step_Train_Data_Nolinear.mat');
S = load(dataFile);

Train_inputs  = S.Train_inputs_Nolinear;     % [6 x N]
Train_targets = S.Train_targets_Nolinear;    % [3 x N]

SEEDS = [9 66 108 88 52];

X = tonndata(Train_inputs,  true, false);
T = tonndata(Train_targets, true, false);

saveDir = fullfile(scriptDir, 'NARX_STEP_Model');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

for s = SEEDS
    rng(s);                          % fix weight initialization + data division

    trainFcn = 'trainlm';            % Levenberg-Marquardt backpropagation
    inputDelays = 1:10;
    feedbackDelays = 1:10;
    hiddenLayerSize = 10;
    net = narxnet(inputDelays, feedbackDelays, hiddenLayerSize, 'open', trainFcn);
    [x, xi, ai, t] = preparets(net, X, {}, T);
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio   = 15/100;
    net.divideParam.testRatio  = 15/100;

    net.trainParam.epochs = 800;
    net.trainParam.max_fail = 300;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = true;

    [net, tr] = train(net, x, t, xi, ai);

    varname = sprintf('STEP_NARX_seed%d', s);
    S2.(varname) = struct('Network', net, 'TrainingResults', tr);
    save(fullfile(saveDir, sprintf('STEP_NARX_seed%d.mat', s)), '-struct', 'S2');
    S2 = struct();
end

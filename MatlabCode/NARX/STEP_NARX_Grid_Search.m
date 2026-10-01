% STEP_NARX_Grid_Search.m
% =========================================================================
% Nonlinear system NARX grid search over hidden size x delay x random seed.
% Trains with the Neural Time Series app protocol (narxnet + trainlm,
% 70/15/15 data division). rng(seed) fixes weight initialization and data
% division so every configuration is trained under identical randomness and
% the grid comparison is not polluted by initialization variance.
%
% Grid: hidden size in [10 20 30], delay in [5 10 20], seeds [9 66 108 88 52].
%
% Input : nonlinear training data (Step_Train_Data_Nolinear.mat, generated
%         by Train_Data_Preparation_Nolinear.m), 8000 steps, 6 inputs,
%         3 outputs.
% Output: one network per (delay, hidden, seed), saved to
%         ./NARX_STEP_Model/NARX_net_d<delay>_h<hidden>_seed<seed>.mat with
%         top-level variable STEP_NARX_d<delay>_h<hidden>_seed<seed>
%         (.Network + .TrainingResults, GUI export convention).
%
% Requires: Deep Learning Toolbox
% =========================================================================

scriptDir = fileparts(mfilename('fullpath'));
dataFile  = fullfile(scriptDir, '..', 'Data', 'Step_Train_Data_Nolinear.mat');
S = load(dataFile);

Train_inputs  = S.Train_inputs_Nolinear;     % [6 x N]
Train_targets = S.Train_targets_Nolinear;    % [3 x N]

HIDDENS = [10, 20, 30];      % hidden layer size
DELAYS  = [5, 10, 20];       % input/feedback delay (1:d)
SEEDS   = [9 66 108 88 52];

X = tonndata(Train_inputs,  true, false);
T = tonndata(Train_targets, true, false);

saveDir = fullfile(scriptDir, 'NARX_STEP_Model');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

total = numel(HIDDENS) * numel(DELAYS) * numel(SEEDS);
cnt = 0;
for h = HIDDENS
    for d = DELAYS
        for s = SEEDS
            cnt = cnt + 1;
            fprintf('\n========== [%d/%d] hidden=%d delay=%d seed=%d ==========\n', ...
                cnt, total, h, d, s);
            rng(s);                    % fix weight initialization + data division

            net = narxnet(1:d, 1:d, h, 'open', 'trainlm');
            [x, xi, ai, t] = preparets(net, X, {}, T);
            net.divideParam.trainRatio = 70/100;
            net.divideParam.valRatio   = 15/100;
            net.divideParam.testRatio  = 15/100;
            net.trainParam.epochs = 800;
            net.trainParam.max_fail = 300;
            net.trainParam.showWindow = false;
            net.trainParam.showCommandLine = true;

            [net, tr] = train(net, x, t, xi, ai);

            varname = sprintf('STEP_NARX_d%d_h%d_seed%d', d, h, s);
            S2.(varname) = struct('Network', net, 'TrainingResults', tr);
            save(fullfile(saveDir, sprintf('NARX_net_d%d_h%d_seed%d.mat', d, h, s)), ...
                '-struct', 'S2');
            S2 = struct();
        end
    end
end

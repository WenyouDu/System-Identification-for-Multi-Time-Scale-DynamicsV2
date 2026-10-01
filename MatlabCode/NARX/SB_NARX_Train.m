% SB_NARX_Train.m
% =========================================================================
% SB (Silver Box) NARX training with 5 random seeds.
% Reproduces the Neural Time Series app workflow exactly:
%   narxnet(1:2, 1:2, 10, 'open', 'trainlm'), 70/15/15 data division,
%   rng(seed) fixes both weight initialization and data division.
% Training parameters are left at toolbox defaults so that the reported
% results are reproduced verbatim. The network is trained purely in open
% loop (no closed-loop fine-tuning).
%
% Input : SB_processed.mat (generated from SB.mat by the SB data-split
%         logic: 75/25 multisine split, then 50/50 train/val), containing
%         SB_X_train [1 x 32531] and SB_Y_train [1 x 32531].
% Output: one network per seed, saved to
%         ./NARX_SB_Model/SB_NARX_seed<seed>.mat with top-level variable
%         SB_NARX_seed<seed> (.Network + .TrainingResults, GUI export
%         convention).
%
% Requires: Deep Learning Toolbox
% =========================================================================

scriptDir = fileparts(mfilename('fullpath'));
dataFile  = fullfile(scriptDir, '..', 'Data', 'SB_processed.mat');
S = load(dataFile);

SB_X_train = S.SB_X_train;     % [1 x 32531]
SB_Y_train = S.SB_Y_train;     % [1 x 32531]

SEEDS = [9 66 108 88 52];

X = tonndata(SB_X_train, true, false);
T = tonndata(SB_Y_train, true, false);

saveDir = fullfile(scriptDir, 'NARX_SB_Model');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

for s = SEEDS
    fprintf('\n========== SB NARX seed=%d ==========\n', s);
    rng(s);                          % fix weight initialization + data division

    net = narxnet(1:2, 1:2, 10, 'open', 'trainlm');
    [x, xi, ai, t] = preparets(net, X, {}, T);
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio   = 15/100;
    net.divideParam.testRatio  = 15/100;
    [net, tr] = train(net, x, t, xi, ai);

    varname = sprintf('SB_NARX_seed%d', s);
    S2.(varname) = struct('Network', net, 'TrainingResults', tr);
    save(fullfile(saveDir, sprintf('SB_NARX_seed%d.mat', s)), '-struct', 'S2');
    S2 = struct();
end

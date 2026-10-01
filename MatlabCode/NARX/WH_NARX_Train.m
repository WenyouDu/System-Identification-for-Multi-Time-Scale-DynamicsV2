% WH_NARX_Train.m
% =========================================================================
% WH (Wiener-Hammerstein) NARX training with 5 random seeds.
% Reproduces the Neural Time Series app workflow exactly:
%   narxnet(1:10, 1:10, 10, 'open', 'trainlm'), 70/15/15 data division,
%   rng(seed) fixes both weight initialization and data division.
% Training parameters are left at toolbox defaults so that the reported
% results are reproduced verbatim. The network is trained purely in open
% loop (no closed-loop fine-tuning).
%
% Input : WH_processed.mat (generated from WH.mat by the WH data-split
%         logic: sys_data = u(5201:184000), then 100000/78800 split),
%         containing WH_train_u [100000 x 1] and WH_train_y [100000 x 1]
%         (single-channel column vectors).
% Output: one network per seed, saved to
%         ./NARX_WH_Model/WH_NARX_seed<seed>.mat with top-level variable
%         WH_NARX_seed<seed> (.Network + .TrainingResults, GUI export
%         convention).
%
% Requires: Deep Learning Toolbox
% =========================================================================

scriptDir = fileparts(mfilename('fullpath'));
dataFile  = fullfile(scriptDir, '..', 'Data', 'WH_processed.mat');
S = load(dataFile);

WH_train_u = S.WH_train_u;     % [100000 x 1]
WH_train_y = S.WH_train_y;     % [100000 x 1]

SEEDS = [9 66 108 88 52];

X = tonndata(WH_train_u, false, false);
T = tonndata(WH_train_y, false, false);

saveDir = fullfile(scriptDir, 'NARX_WH_Model');
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

for s = SEEDS
    fprintf('\n========== WH NARX seed=%d ==========\n', s);
    rng(s);                          % fix weight initialization + data division

    net = narxnet(1:10, 1:10, 10, 'open', 'trainlm');
    [x, xi, ai, t] = preparets(net, X, {}, T);
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio   = 15/100;
    net.divideParam.testRatio  = 15/100;
    [net, tr] = train(net, x, t, xi, ai);

    varname = sprintf('WH_NARX_seed%d', s);
    S2.(varname) = struct('Network', net, 'TrainingResults', tr);
    save(fullfile(saveDir, sprintf('WH_NARX_seed%d.mat', s)), '-struct', 'S2');
    S2 = struct();
end

clearvars
clc

% add necessary packages
fieldtrip_path          = '~/toolboxes/fieldtrip-20221223';
addpath(fieldtrip_path); 
ft_defaults

helper_functions_path   = '../helper_functions/';
catch22_path            = '~/sciebo/Software/catch22/wrap_Matlab';

addpath(helper_functions_path)
addpath(catch22_path)


% load data & define folders
load('/remotedata/AgGross/AlphaSpade/Data/_meg/fullsegcleanmeg_1331.mat')
outdir = '/remotedata/AgGross/TBraiC/MV-eye/STRG_computed_features/alphaSPADE';

if ~isfolder(outdir)
    mkdir(outdir)
end

cfg = [];
cfg.stepsize = .25;
cfg.winsize = .5;

% scale up data  
fullsegdata.trial = cellfun(@(x) x*1e11, fullsegdata.trial, 'UniformOutput',false);

outdat = mv_timeresolved(cfg, fullsegdata);
save(fullfile(outdir, 'timeresolved_test_39feats.mat'), "outdat");

% figure();
% subplot(3, 1, 1); plot(outdat.time_winCENter, outdat.single_feats.MCL(61, :, 100))
% subplot(3, 1, 2); plot(outdat.time_winCENter, outdat.single_feats.std(61, :, 100))
% subplot(3, 1, 3); plot(outdat.time_winCENter, outdat.single_feats.mean(61, :, 100))

%% prepare time testing


% %% actual time testing
% 
% % select one time window for dummy timetest
% ntrls = 10;
% cfg = [];
% cfg.latency = [0, .250];
% cfg.trials = 1:ntrls;
% 
% minidat = ft_selectdata(cfg, fullsegdata);
% 
% 
% fprintf('\n\nTime test started')
% onset = tic;
% 
% % initialize Feature structure
% F.single_parcels = [];
% F.single_feats = [];
% F.multi_feats = double.empty(ntrls, 0);
% F.Y = 1:ntrls;
% F.trl_order = 1:ntrls;
% 
% % compute feats
% cfg_feats = mv_features_cfg();
% 
% F = mv_features_timedomain(cfg_feats, minidat, F);
% F = mv_wrap_catch22(cfg_feats, minidat, F);
% F = mv_periodic_aperiodic(cfg_feats, minidat, F);
% 
% toc(onset)
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 

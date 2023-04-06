%% EC/EO decoding
clearvars;
close all
clc

%% path definition

% input and packages
software_path           = '../../Software/biosig-master/biosig/t300_FeatureExtraction';
fieldtrip_path          = '~/toolboxes/fieldtrip-20221223';
data_path               = '/remotedata/AgGross/Fasting/NC/resultsNC/visual_gamma/source';
entropy_functions_path  = '../../Software/mMSE-master';
helper_functions_path   = '../helper_functions/';
beta_functions_path     = '../beta_functions';
plotting_functions_path = '../plotting_functions';
resources_path          = '../../Resources';
catch22_path            = '../../Software/catch22/wrap_Matlab';

% output folders
out_feats_path          = '/remotedata/AgGross/TBraiC/MV-eye/STRG_computed_features';
if ~isfolder(out_feats_path); mkdir(out_feats_path); end

addpath(software_path); addpath(helper_functions_path)
addpath(entropy_functions_path); addpath(resources_path)
addpath(plotting_functions_path); 
addpath(catch22_path)
addpath(fieldtrip_path); 
ft_defaults

% load sample dataset
load(fullfile(data_path, 'S01_satted_source_VG'))
dat = sourcedata;

% select only visual cortices
text_prompt = 'visual';
mask_parcel = mv_select_parcels(text_prompt);

cfg = [];
cfg.channel = dat.label(mask_parcel);
dat = ft_preprocessing(cfg, dat);

% resample to 256 Hz for consistency with EC/OC
cfg = [];
cfg.resamplefs = 256;
dat = ft_resampledata(cfg, dat);



% redefine trials for pre and post stim segments
cfg_pre.toilim = [-1, 0-1/cfg.resamplefs];
dat_pre = ft_redefinetrial(cfg_pre, dat);

cfg_stim.toilim = [1, 2-1/cfg.resamplefs];
dat_stim = ft_redefinetrial(cfg_stim, dat);

dat = {dat_pre, dat_stim};

%% merge datasets & set config
% unitary label
Y=[ones(length(dat{1}.trial), 1); 2*ones(length(dat{2}.trial), 1)]';
% merge
cfg = [];
dat = ft_appenddata(cfg, dat{1}, dat{2});
% call for config
cfg_feats = mv_features_cfg();


% shuffle
rng(1)
shffld_idxs = randperm(length(Y));
Y = Y(shffld_idxs);
dat.trial = dat.trial(shffld_idxs);

%% scale up data

dat.trial = cellfun(@(x) x*1e11, dat.trial, 'UniformOutput',false);

%% compute derivative

whitened_dat = dat;
whitened_dat.trial = cellfun(@(x) diff(x,1,2), dat.trial, 'UniformOutput',false);
whitened_dat.time = cellfun(@(x) x(2:end), dat.time, 'UniformOutput',false);


%% initialize Feature structure
ntrials = length(dat.trial);

F.single_parcels = [];
F.single_feats = [];
F.multi_feats = double.empty(ntrials, 0);
F.Y = Y;

F_whitened = F;


%% compute frequency features

F = mv_features_freqdomain_nonrecursive(cfg_feats, dat, F);
F_whitened = mv_features_freqdomain_nonrecursive(cfg_feats, whitened_dat, F_whitened);

%% periodic & aperiodic

F = mv_periodic_aperiodic(cfg_feats, dat, F);
F_whitened = mv_periodic_aperiodic(cfg_feats, whitened_dat, F_whitened);


%% compute time features

F = mv_features_timedomain(cfg_feats, dat, F);
F_whitened = mv_features_timedomain(cfg_feats, whitened_dat, F_whitened);

%% compute catch22

F = mv_wrap_catch22(cfg_feats, dat, F);
F_whitened = mv_wrap_catch22(cfg_feats, dat, F_whitened);

%% store F output
% decoding will continue in python

% save
fname = 'test.mat';
save(fullfile(out_feats_path, fname), 'F')

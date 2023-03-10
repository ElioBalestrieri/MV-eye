%% EC/EO decoding
clearvars;
close all
clc

%% path definition

% input and packages
fieldtrip_path          = '~/toolboxes/fieldtrip-20221223';
data_path               = '/remotedata/AgGross/Fasting/NC/resultsNC/resting_state/source/lcmv';
helper_functions_path   = '../helper_functions/';
plotting_functions_path = '../plotting_functions';
resources_path          = '../../Resources';
catch22_path            = '../../Software/catch22/wrap_Matlab';

% output folders
out_dat_path          = '/remotedata/AgGross/TBraiC/MV-eye/STRG_data';
if ~isfolder(out_dat_path); mkdir(out_dat_path); end

out_feat_path          = '../STRG_computed_features';
if ~isfolder(out_feat_path); mkdir(out_feat_path); end

addpath(helper_functions_path)
addpath(plotting_functions_path); addpath(resources_path)
addpath(catch22_path)
addpath(fieldtrip_path); 

% transfer original files to the local (sciebo) DATAfolder?
filetransferflag = false;
decodesingleparcels = false;
plotparcelsflag = false;

%% loop into subjects

nsubjs = 29; nthreads = 8;
thisObj = parpool(nthreads);

parfor isubj = 1:nsubjs

    ft_defaults;

    subjcode = sprintf('%0.2d', isubj);
    fname_in = ['S' subjcode '_control_hcpsource_1snolap.mat'];
    
    % load sample dataset
    temp = load(fullfile(data_path, fname_in));
    dat = temp.sourcedata;
    
    if filetransferflag
        fname_out_dat = [subjcode, '_dat.mat'];
        saveinparfor(fullfile(out_dat_path, fname_out_dat), dat)
    end        

    %% merge datasets & set config
    % unitary label
    Y=[ones(length(dat{1}.trial), 1); 2*ones(length(dat{2}.trial), 1)]';
    % merge
    cfg = [];
    dat = ft_appenddata(cfg, dat{1}, dat{2});
    % call for config
    cfg_feats = mv_features_cfg();

    % select only visual cortices
    text_prompt = 'visual';
    mask_parcel = mv_select_parcels(text_prompt);
    
    cfg = [];
    cfg.channel = dat.label(mask_parcel);
    dat = ft_preprocessing(cfg, dat);
    
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
    
    %% compute time features
    
    F = mv_features_timedomain(cfg_feats, dat, F);
    F_whitened = mv_features_timedomain(cfg_feats, whitened_dat, F_whitened);
    
    %% compute frequency features

    F = mv_features_freqdomain_nonrecursive(cfg_feats, dat, F);
    F_whitened = mv_features_freqdomain_nonrecursive(cfg_feats, whitened_dat, F_whitened);
    
    %% compute catch22
    
    F = mv_wrap_catch22(cfg_feats, dat, F);
    F_whitened = mv_wrap_catch22(cfg_feats, dat, F_whitened);

    %% store F output   
    % decoding will continue in python
    
    % append cfg for easy check the operations performed before
    F.cfg_feats = cfg_feats;
    F_whitened.cfg_feats = cfg_feats;

    % save
    fname_out_feat = [subjcode, '_NOWHIT_feats.mat'];
    fname_out_feat_whit = [subjcode, '_WHIT_feats.mat'];

    saveinparfor(fullfile(out_feat_path, fname_out_feat), F)
    saveinparfor(fullfile(out_feat_path, fname_out_feat_whit), F_whitened)

    % feedback
    fprintf('\n\n######################\n')
    fprintf('Subj %s completed\n', subjcode)


end

delete(thisObj)

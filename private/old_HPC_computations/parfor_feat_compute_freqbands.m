%% computation of frequency bands in the full parcellated brain
clearvars;
close all
clc

%% path definition

% input and packages
fieldtrip_path          = '~/toolboxes/fieldtrip-20221223';
data_path               = '/remotedata/AgGross/Fasting/NC/resultsNC/resting_state/source/lcmv';
helper_functions_path   = '../helper_functions/';

out_feat_path          = '/remotedata/AgGross/TBraiC/MV-eye/STRG_computed_features';
if ~isfolder(out_feat_path); mkdir(out_feat_path); end

addpath(helper_functions_path)
addpath(fieldtrip_path); 


%% loop into subjects

nsubjs = 29; nthreads = 7;
thisObj = parpool(nthreads);

parfor isubj = 1:nsubjs

    ft_defaults;

    subjcode = sprintf('%0.2d', isubj);
    fname_in = ['S' subjcode '_control_hcpsource_1snolap.mat'];
    
    % load sample dataset
    temp = load(fullfile(data_path, fname_in));
    dat = temp.sourcedata;
    
    %% merge datasets & set config
    % unitary label
    Y=2*ones(length(dat{1}.trial), 1); % only Eye closed 2*ones(length(dat{2}.trial), 1)]';
%     % merge
%     cfg = [];
%     dat = ft_appenddata(cfg, dat{1}, dat{2});

    dat = dat{2};
    
    % call for config
    cfg_feats = mv_features_cfg_theoretical_freqbands();
     
    %% scale up data
    
    dat.trial = cellfun(@(x) x*1e11, dat.trial, 'UniformOutput',false);
    
    %% initialize Feature structure
    ntrials = length(dat.trial);
    
    F.single_parcels = [];
    F.single_feats = [];
    F.multi_feats = double.empty(ntrials, 0);
    F.Y = Y;
    
    F = mv_features_freqdomain_nonrecursive(cfg_feats, dat, F);

    %% store F output   
    % decoding will continue in python
    
    % append cfg for easy check the operations performed before
    F.cfg_feats = cfg_feats;

    % save
    fname_out_feat = [subjcode, '_freqbands_fullHead_feats.mat'];
    saveinparfor(fullfile(out_feat_path, fname_out_feat), F)

    % feedback
    fprintf('\n\n######################\n')
    fprintf('Subj %s completed\n', subjcode)


end

delete(thisObj)

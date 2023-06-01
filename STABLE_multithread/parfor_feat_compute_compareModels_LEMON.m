%% VG features extraction
clearvars;
close all
clc

%% path definition

% input and packages
fieldtrip_path          = '~/toolboxes/fieldtrip-20221223';
data_path               = '../STRG_data/MPI_LEMON_ECEO';
helper_functions_path   = '../helper_functions/';
plotting_functions_path = '../plotting_functions';
resources_path          = '../../Resources';
catch22_path            = '/home/balestrieri/sciebo/Software/catch22/wrap_Matlab';

out_feat_path          = '/remotedata/AgGross/TBraiC/MV-eye/STRG_computed_features/LEMON';
if ~isfolder(out_feat_path); mkdir(out_feat_path); end

addpath(helper_functions_path)
addpath(plotting_functions_path); addpath(resources_path)
addpath(catch22_path)
addpath(fieldtrip_path); 

%% prepare data fetching

% fetch subj names
subjects_EC = dir(fullfile(data_path, 'LEMON-closed-preprocessed', 'sub-0*'));
subjects_EO = dir(fullfile(data_path, 'LEMON-open-preprocessed', 'sub-0*'));

% reminder: use only EC subject list for fetching the filenames also in the
% EO condition. this checks automatically then for potential mismatches...
% Also EC is the condition with one subject less.
matching_subjs = ismember({subjects_EO.name}, {subjects_EC.name});
subjects_EO = subjects_EO(matching_subjs);

nsubjs = length(subjects_EO);

% list of channels considered (POSTERIOR)
KEEPCHANS = {'TP7', 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6', 'TP8', ...
             'P7', 'P5', 'P3', 'P1', 'P2', 'P4', 'P6', 'P8', ...
             'PO9', 'PO7', 'PO3', 'PO4', 'PO8', 'PO10', 'O1', 'O2', ...
             'CPz', 'Pz', 'POz', 'Oz'};

% loop through "theoretical" models
mdls_codes = {'FTM'}; % {'FreqBands', 'FullFFT', 'TimeFeats'};


%% loop into subjects

nthreads = 5; 

% LEMON data is in set format. Load EEGLAB to open files
eeglab nogui

thisObj = parpool(nthreads);
parfor isubj = 1:nsubjs

    % weird errors if ft called outside parfor
    ft_defaults;

    % collect data and concatenate in cell
    subjcode = subjects_EC(isubj).name; 
    fname = [subjcode, '_eeg.set']; 
    pathtofile_EC = fullfile(subjects_EC(isubj).folder, subjects_EC(isubj).name, 'eeg');
    pathtofile_EO = fullfile(subjects_EO(isubj).folder, subjects_EC(isubj).name, 'eeg');
    
    EC_eeg = pop_loadset('filename', fname, 'filepath', pathtofile_EC);
    EO_eeg = pop_loadset('filename', fname, 'filepath', pathtofile_EO);
    dat = {EC_eeg, EO_eeg}; 

    % convert to fieldtrip structure
    dat = cellfun(@(x) eeglab2fieldtrip(x, 'raw', 'none'), dat, 'UniformOutput',false);
    
    % select only posterior channels 
    cfg = []; cfg.channel = KEEPCHANS;
    dat = cellfun(@(x) ft_preprocessing(cfg, x), dat, 'UniformOutput',false);
    
    % cut in 1 s segments
    for idat = 1:2
    
        % redefine trials in 1 sec segments
        cfg_pre = [];
        cfg_pre.toilim = [0, 1-1./dat{idat}.fsample];
        mini1 = ft_redefinetrial(cfg_pre, dat{idat});
        
        cfg_post = [];
        cfg_post.toilim = [1, 2-1./dat{idat}.fsample];
        mini2 = ft_redefinetrial(cfg_post, dat{idat});
    
        cfg = [];
        segmented_1s = ft_appenddata(cfg, mini1, mini2);
    
        dat{idat} = segmented_1s;
    
    end
        
    % merge datasets & set config
    % unitary label
    Y=[ones(length(dat{1}.trial), 1); 2*ones(length(dat{2}.trial), 1)]';

    % original trial ordering
    trl_order = [1:length(dat{1}.trial), 1:length(dat{2}.trial)]

    % merge
    cfg = [];
    dat = ft_appenddata(cfg, dat{1}, dat{2});

    % shuffle
    rng(1)
    shffld_idxs = randperm(length(Y));
    Y = Y(shffld_idxs);
    trl_order = trl_order(shffld_idxs);
    dat.trial = dat.trial(shffld_idxs);

    ntrials = length(dat.trial);

    for imdl = 1:length(mdls_codes)

        mdl_name = mdls_codes{imdl};

        % initialize Feature structure

        F.single_parcels = [];
        F.single_feats = [];
        F.multi_feats = double.empty(ntrials, 0);
        F.Y = Y;
        F.trl_order = trl_order;
      
        switch mdl_name

            case 'FB_2M'

                cfg_feats = mv_features_cfg_theoretical_freqbands();
                cfg_feats.time = {'mean', 'std'};
                F = mv_features_moments_freqbands(cfg_feats, dat, F);

            case 'FB_3M'

                cfg_feats = mv_features_cfg_theoretical_freqbands();
                cfg_feats.time = {'mean', 'std', 'skewness'};
                F = mv_features_moments_freqbands(cfg_feats, dat, F);


            case 'FB_4M'

                cfg_feats = mv_features_cfg_theoretical_freqbands();
                cfg_feats.time = {'mean', 'std', 'skewness', 'kurtosis'};
                F = mv_features_moments_freqbands(cfg_feats, dat, F);

            case 'FB_cherrypicked'

                cfg_feats = mv_features_cfg_theoretical_freqbands();
                cfg_feats.time = {'MCL', 'CO_HistogramAMI_even_2_5'};
                F = mv_features_moments_freqbands(cfg_feats, dat, F);


            case 'cherrypicked'

                cfg_feats = mv_features_cfg();
                cfg_feats.time = {'MCL', 'CO_HistogramAMI_even_2_5'};
                F = mv_features_timedomain(cfg_feats, dat, F);


            case 'FTM'

                % call for config
                cfg_feats = mv_features_cfg_FTM();
                F = mv_features_timedomain(cfg_feats, dat, F);

            case 'FreqBands'

                % call for config
                cfg_feats = mv_features_cfg_theoretical_freqbands();
                F = mv_features_freqdomain_nonrecursive(cfg_feats, dat, F);


            case 'FullFFT'

                % call for config
                cfg_feats = mv_features_cfg();
                F = mv_features_freqdomain_nonrecursive(cfg_feats, dat, F);

            case 'TimeFeats'

                % call for config
                cfg_feats = mv_features_cfg();

                F = mv_features_timedomain(cfg_feats, dat, F);
                F = mv_wrap_catch22(cfg_feats, dat, F);
                F = mv_periodic_aperiodic(cfg_feats, dat, F);

        end

        % append cfg for easy check the operations performed before
        F.cfg_feats = cfg_feats;

        %% double check BIDS pinfo between sessions (and merge them if equal)
        tests_match = nan(numel(EC_eeg.BIDS.pInfo), 1);
        for idx = 1:numel(EC_eeg.BIDS.pInfo)
            tests_match(idx) = all(EC_eeg.BIDS.pInfo{idx} == EO_eeg.BIDS.pInfo{idx}); 
        end
        sessions_match = all(tests_match);
        
        if sessions_match % store the pInfo in F
            F.pinfo = EO_eeg.BIDS.pInfo; 
            errorflag = '';
        else % completely null the F structure computed so far, in order to prevent 
             % mistakes at further stages
            F = [];
            F.errormessage = 'mismatch in BIDS';
            errorflag = 'BAD_SUBJINFO';
        end

        % save
        fname_out_feat = [subjcode, '_' errorflag, mdl_name '.mat'];
        saveinparfor(fullfile(out_feat_path, fname_out_feat), F)

    end

    % feedback
    fprintf('\n\n######################\n')
    fprintf('Subj %s completed\n', subjcode)

end

delete(thisObj)



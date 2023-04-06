%% VG features extraction
clearvars;
close all
clc

%% path definition

% input and packages
fieldtrip_path          = '~/toolboxes/fieldtrip-20221223';
data_paths               = {'/remotedata/AgGross/Fasting/NC/resultsNC/visual_gamma/source', ...
                            '/remotedata/AgGross/Fasting/NC/resultsNC/resting_state/source/lcmv', ...
                            };
helper_functions_path   = '../helper_functions/';
plotting_functions_path = '../plotting_functions';
resources_path          = '../../Resources';
catch22_path            = '../../Software/catch22/wrap_Matlab';

out_feat_path          = '/remotedata/AgGross/TBraiC/MV-eye/STRG_computed_features/Mdl_comparison';
if ~isfolder(out_feat_path); mkdir(out_feat_path); end

addpath(helper_functions_path)
addpath(plotting_functions_path); addpath(resources_path)
addpath(catch22_path)
addpath(fieldtrip_path); 

%% loop into subjects

nsubjs = 29; nthreads = 6;

thisObj = parpool(nthreads);
parfor isubj = 1:nsubjs

    % weird errors if ft called outside parfor
    ft_defaults;

    % filename types for the two datasets: resting state & visual stimulation
    fname_types = {'_satted_source_VG.mat', '_control_hcpsource_1snolap.mat'};
    exp_cond = {'VS', 'ECEO'};

    % formatted codename
    subjcode = sprintf('%0.2d', isubj);

    for icond = 1:2
        
        this_cond = exp_cond{icond}; this_ftype = fname_types{icond};
        fname_in = ['S' subjcode this_ftype];
    
        % load sample dataset
        temp = load(fullfile(data_paths{icond}, fname_in)); %#ok<PFBNS> 
        sourcedata = temp.sourcedata;

        switch this_cond

            case 'VS'

                % resample
                cfg = [];
                cfg.resamplefs = 256;
                sourcedata = ft_resampledata(cfg, sourcedata);

                % redefine trials for pre and post stim segments
                cfg_pre = [];
                cfg_pre.toilim = [-1, 0-1./cfg.resamplefs];
                dat_pre = ft_redefinetrial(cfg_pre, sourcedata);
    
                cfg_stim = [];
                cfg_stim.toilim = [1, 2-1./cfg.resamplefs];
                dat_stim = ft_redefinetrial(cfg_stim, sourcedata);
    
                % merge the data together in a format that allows
                % compatibility with EC EO 
                dat = {dat_pre, dat_stim};

            case 'ECEO'

                dat = sourcedata;

        end

        % select only visual cortices
        text_prompt = 'visual';
        mask_parcel = mv_select_parcels(text_prompt);
    
        cfg = [];
        cfg.channel = dat{1}.label(mask_parcel);

        dat = cellfun(@(x) ft_preprocessing(cfg, x), dat, ...
            'UniformOutput',false);
        
        % merge datasets & set config
        % unitary label
        Y=[ones(length(dat{1}.trial), 1); 2*ones(length(dat{2}.trial), 1)]';
        % merge
        cfg = [];
        dat = ft_appenddata(cfg, dat{1}, dat{2});
    
        % shuffle
        rng(1)
        shffld_idxs = randperm(length(Y));
        Y = Y(shffld_idxs);
        dat.trial = dat.trial(shffld_idxs);

        % scale up data  
        dat.trial = cellfun(@(x) x*1e11, dat.trial, 'UniformOutput',false);

        % loop through "theoretical" models
        mdls_codes = {'FTM', 'FullFFT', 'TimeFeats', 'FreqBands'};

        ntrials = length(dat.trial);

        for imdl = 1:length(mdls_codes)

            mdl_name = mdls_codes{imdl};

            % initialize Feature structure
    
            F.single_parcels = [];
            F.single_feats = [];
            F.multi_feats = double.empty(ntrials, 0);
            F.Y = Y;

          
            switch mdl_name

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


            % save
            fname_out_feat = [subjcode, '_' this_cond '_' mdl_name '.mat'];
            saveinparfor(fullfile(out_feat_path, fname_out_feat), F)

        end

    end

    % feedback
    fprintf('\n\n######################\n')
    fprintf('Subj %s completed\n', subjcode)


end

delete(thisObj)



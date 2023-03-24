%% VG features extraction
clearvars;
close all
clc

%% path definition

% input and packages
fieldtrip_path          = '~/toolboxes/fieldtrip-20221223';
data_path               = '/remotedata/AgGross/Fasting/NC/resultsNC/visual_gamma/source';
helper_functions_path   = '../helper_functions/';
plotting_functions_path = '../plotting_functions';
resources_path          = '../../Resources';
catch22_path            = '../../Software/catch22/wrap_Matlab';

% output folders
out_dat_path          = '../STRG_data';
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

nsubjs = 29; nthreads = 6;

thisObj = parpool(nthreads);
parfor isubj = 1:nsubjs

    ft_defaults;

    subjcode = sprintf('%0.2d', isubj);
    fname_in = ['S' subjcode '_satted_source_VG.mat'];
    
    % load sample dataset
    temp = load(fullfile(data_path, fname_in));
    sourcedata = temp.sourcedata;

    % select only visual cortices
    text_prompt = 'visual';
    mask_parcel = mv_select_parcels(text_prompt);
    
    cfg = [];
    cfg.channel = sourcedata.label(mask_parcel);
    sourcedata = ft_preprocessing(cfg, sourcedata);

    % redefine trials for pre and post stim segments
    cfg_pre = [];
    cfg_pre.toilim = [-1, 0];
    dat_pre = ft_redefinetrial(cfg_pre, sourcedata);
    
    cfg_stim = [];
    cfg_stim.toilim = [1, 2];
    dat_stim = ft_redefinetrial(cfg_stim, sourcedata);
    
    % merge the data together in a format that allow backward compatibility
    % with EC EO 
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

    %% periodic & aperiodic
    
    F = mv_periodic_aperiodic(cfg_feats, dat, F);
    F_whitened = mv_periodic_aperiodic(cfg_feats, whitened_dat, F_whitened);

    %% compute frequency features

    F = mv_features_freqdomain_nonrecursive(cfg_feats, dat, F);
    F_whitened = mv_features_freqdomain_nonrecursive(cfg_feats, whitened_dat, F_whitened);
    
    %% compute time features
    
    F = mv_features_timedomain(cfg_feats, dat, F);
    F_whitened = mv_features_timedomain(cfg_feats, whitened_dat, F_whitened);
    
    %% compute catch22
    
    F = mv_wrap_catch22(cfg_feats, dat, F);
    F_whitened = mv_wrap_catch22(cfg_feats, whitened_dat, F_whitened);
    
    %% store F output   
    % decoding will continue in python
    
    % append Y labels
    F.Y = Y;

    % append cfg for easy check the operations performed before
    F.cfg_feats = cfg_feats;
    
    % save
    fname_out_feat = [subjcode, 'NONwhiten_VG_feats.mat'];
    saveinparfor(fullfile(out_feat_path, fname_out_feat), F)

    fname_out_feat = [subjcode, 'PREwhiten_VG_feats.mat'];
    saveinparfor(fullfile(out_feat_path, fname_out_feat), F_whitened)

    % feedback
    fprintf('\n\n######################\n')
    fprintf('Subj %s completed\n', subjcode)


end

delete(thisObj)


%% pool accuracies together

if  plotparcelsflag

    accs_all_parcels = nan(360, nsubjs);
    
    for isubj = 1:nsubjs
    
        subjcode = sprintf('%0.2d', isubj);
        fname_in_feat = [subjcode, '_feats.mat'];
    
        load(fullfile(out_feat_path, fname_in_feat))
        
        try
    
            accs_all_parcels(:, isubj) = variableName.single_parcels_acc.accuracy;
    
        catch
    
            disp(subjcode)
    
        end
    
    end
    
    
    avg_accs_ignorenan = mean(accs_all_parcels,2, 'omitnan');
    
    %% plots
    
    ft_defaults;
    
    % plot accuracy over parcellated brain
    atlas = ft_read_cifti('Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii');
    atlas.data = zeros(1,64984);
    filename = 'S1200.L.very_inflated_MSMAll.32k_fs_LR.surf.gii';
    sourcemodel = ft_read_headshape({filename, strrep(filename, '.L.', '.R.')});
    
    for ilab=1:length(atlas.indexmaxlabel)
        tmp_roiidx=find(atlas.indexmax==ilab);   
        atlas.data(tmp_roiidx)=avg_accs_ignorenan(ilab);
    end
    
    figure()
    plot_hcp_surfaces(atlas,sourcemodel,'YlOrRd',0, ...
                      'accuracy',[-0,45],[90,0],'SVM accuracy, 24 subjs', [.5, .8]);

end


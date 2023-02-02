%% EC/EO decoding
clearvars;
close all
clc

%% path definition

% input and packages
fieldtrip_path          = '~/toolboxes/fieldtrip-20221223';
data_path               = '/remotedata/AgGross/Fasting/NC/resultsNC/resting_state/source/lcmv';
helper_functions_path   = '../helper_functions/';
beta_functions_path     = '../beta_functions';
plotting_functions_path = '../plotting_functions';
resources_path          = '../../Resources';
catch22_path            = '../../Software/catch22/wrap_Matlab';

% old?
% entropy_functions_path  = '../../Software/mMSE-master'; addpath(entropy_functions_path); 
% software_path           = '../../Software/biosig-master/biosig/t300_FeatureExtraction'; addpath(software_path); 

% output folders
out_dat_path          = '../STRG_data';
if ~isfolder(out_dat_path); mkdir(out_dat_path); end

out_feat_path          = '../STRG_computed_features';
if ~isfolder(out_feat_path); mkdir(out_feat_path); end

addpath(helper_functions_path)
addpath(beta_functions_path); 
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
    
    % shuffle
    rng(1)
    shffld_idxs = randperm(length(Y));
    Y = Y(shffld_idxs);
    dat.trial = dat.trial(shffld_idxs);
    
    %% initialize Feature structure
    ntrials = length(dat.trial);
    
    F.single_parcels = [];
    F.single_feats = [];
    F.multi_feats = double.empty(ntrials, 0);
    
    %% compute frequency features
    
    [F, vout] = mv_features_freqdomain(cfg_feats, dat, F);
    
    %% compute catch22
    
   F = mv_wrap_catch22(cfg_feats, dat, F);
    
    %% compute time features
    
    F = mv_features_timedomain(cfg_feats, dat, F);
    
    %% periodic & aperiodic
    
    F = mv_periodic_aperiodic(cfg_feats, dat, F);

    %% spatial dist classification accuracy

    if decodesingleparcels

        try
            parc_acc = mv_classify_parcels(cfg_feats, F, Y);
            % also append the accuracy for the single parcels
            F.single_parcels_acc = parc_acc;

        catch ME

            F.single_parcels_acc = ME;

        end

    end

    %% store F output   
    % decoding will continue in python
    
    % append Y labels
    F.Y = Y;

    % append cfg for easy check the operations performed before
    F.cfg_feats = cfg_feats;
    
    % save
    fname_out_feat = [subjcode, '_feats.mat'];
    saveinparfor(fullfile(out_feat_path, fname_out_feat), F)
    saveinparfor_fbands(out_feat_path, subjcode, vout, Y, cfg_feats)

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


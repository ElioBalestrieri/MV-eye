clearvars
close all
clc

in_fold_hctsa = '/remotedata/AgGross/TBraiC/AlphaSpade/hctsa';
in_fold_meg = '/remotedata/AgGross/AlphaSpade/Data/_meg';
in_fold_beh = '/remotedata/AgGross/AlphaSpade/Data/_behav';
out_fold = '/home/balestrieri/tmpHCTSA_alphaSPADE';

% get the list of features computed
load('/remotedata/AgGross/TBraiC/AlphaSpade/OP.mat')
listFeats = OP.CodeString;

% get contrast values for each subj
tmp = load(fullfile(in_fold_beh, "AlphaSpade_behav_dat30.mat"), 'dat');
full_beh = tmp.dat;

% extract the full list of subject IDs
rawsubjs = struct2table(dir(fullfile(in_fold_meg, 'fullsegcleanmeg*.mat'))); 
subjcodes = unique(cellfun(@(x) x(17:20), rawsubjs.name, 'UniformOutput',false));


summary_struct = struct();

for isubj = 1:length(subjcodes)

    ID = subjcodes{isubj};    

    filelist = struct2table(dir(fullfile(in_fold_hctsa, [ID, '_features_*.mat']))); 
    filenames = filelist.name;
    
    % load mask & labels
    tmp = load(fullfile(in_fold_meg, rawsubjs.name{isubj})); 
    tmp_trial = tmp.fullsegdata.trialinfo; clear tmp; % free memory space
    mask_stim_present = tmp_trial(:, 2)>=10;
    temp = tmp_trial(mask_stim_present , 2);
    seen_unseen = (temp==11) | (temp==22); % seen = true; unseen (or misplaced) = false
    trialinfo = struct('mask_stim_present', mask_stim_present, 'seen_unseen', seen_unseen);

    % extract vector of contrast
    beh = full_beh{isubj};

    % preallocate and load only the trials that will be used for
    % classification (spare memory and time)
    all_trls = cell(length(trialinfo.mask_stim_present), 1);

    for itrl = 1:length(filenames)
        
        this_fname = [ID, '_features_', num2str(itrl) '.mat']; 
        % reminder!!! not possible to use the filenames recovered with dir because
        % of nonsequential ordering of filenames (e.g. 123 < 23)

        temp = load(fullfile(in_fold_hctsa, this_fname), 'res');
        all_trls{itrl} = single(temp.res);
        disp(itrl)

    end

    trials = cat(3, all_trls{:});
    trials = trials(:, :, trialinfo.mask_stim_present); % single

    acc_feats_stored = 0;

    % initiate the hdf5 file with the target labels & extended beh data
    % (including respiration)
    h5create(fullfile(out_fold, [ID '.h5']), '/Y', size(trialinfo.seen_unseen))
    h5write(fullfile(out_fold, [ID '.h5']), '/Y', trialinfo.seen_unseen*1)

    h5create(fullfile(out_fold, [ID '.h5']), '/beh', size(beh))
    h5write(fullfile(out_fold, [ID '.h5']), '/beh', beh)

    for iFeat = 1:length(listFeats)

        this_feat = listFeats{iFeat};
        tmp = squeeze(trials(:, iFeat, :));

        % data sanity checks

        % 1. no variability
        NoVar = all(std(tmp)==0);

        % 2a. invalid data (Nan, +-inf)
        CountInv = sum(sum(~isfinite(tmp)));

        % 2b. complex data
        CountComplex = sum(sum(~isreal(tmp)));

        % reject if aggregated bad data (complex & invalid) exceeds 15%
        BadDat = ((CountComplex + CountInv)./numel(tmp))>.15;

        % if the data passed these sanity checks, append to hdf5
        GoodFeat = (~NoVar) & (~BadDat);
        
        if GoodFeat

            h5create(fullfile(out_fold, [ID '.h5']), ['/X/', this_feat], size(tmp), 'Datatype','single') 
            h5write(fullfile(out_fold, [ID '.h5']), ['/X/', this_feat], tmp)
            acc_feats_stored = acc_feats_stored+1;

        end

    end

    fprintf('\n%s completed', ID)
    summary_struct.(['ID', ID]) = acc_feats_stored;

end





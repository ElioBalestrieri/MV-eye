function F = mv_features_timedomain(cfg_feats, dat, F)

% quick input check
if ~isfield(cfg_feats, 'time')
    disp('No features in the time domain')
    return
else
    if isempty(cfg_feats.time)
        disp('No features in the time domain')
        return
    end
end

% vars needed for future loops
nchans = length(dat.label);
ntrials = length(dat.trial);

% initiate switch loop for features

% note: since many features will be computed, I decided to use only one
% variable, named "TEMP", to store all the temporary computations for
% features. This will spare quite some matrices in memory, which in
% parallel computing could be relevant.
% The data is then assigned to the F structure with its rightful name.

for ifeat = cfg_feats.time

    this_feat = ifeat{1}; 
    
    if cfg_feats.verbose; fprintf('\nComputing %s', this_feat); end

    tic 
    
    % preallocate TEMP (and get rid of previous values, just to be sure)
    TEMP = nan(ntrials, nchans);

    switch this_feat

        case 'SAMPEN' % SAMPle ENtropy     

            for itrl = 1:ntrials
                TEMP(itrl, :) = SampEnMat(2, .2, ...
                    zscore(dat.trial{itrl}, [], 2));    
            end
            
        case 'median'

            TEMP = cellfun(@(x) median(x, 2), dat.trial, 'UniformOutput',false);
            TEMP = cat(2, TEMP{:})';

        case 'mean'

            TEMP = cellfun(@(x) mean(x, 2), dat.trial, 'UniformOutput',false);
            TEMP = cat(2, TEMP{:})';

        case 'std'

            TEMP = cellfun(@(x) std(x,[],2), dat.trial, 'UniformOutput',false);
            TEMP = cat(2, TEMP{:})';

        case 'wpH'

            for itrl = 1:ntrials
                for ichan = 1:nchans
                    TEMP(itrl,ichan) = wpH(zscore(dat.trial{itrl}(ichan,:)),3,1);
                end
            end                    


        otherwise

            error('"%s" is not recognized as feature', ifeat{1})

    end

    % reduce from parcels to PC
    F.single_feats.(this_feat) = local_PCA(TEMP, cfg_feats);
    % add to F structure
    F = local_add_feature(F, F.single_feats.(this_feat), TEMP, ntrials, nchans);
    % log runtime
    F.runtime.(this_feat) = round(toc, 2);

end

end

%% ########################### LOCAL FUNCTIONS ############################

function reduced_data = local_PCA(data, cfg_feats)

% PCA to reduce dimensionality
[~, PCAscore, ~, ~, EXPLAINED] = pca(data);
% select only components accounting for up to N% predefined
% explained variance
keepcomps = (cumsum(EXPLAINED)/100) <= cfg_feats.PCAvarExplained;

reduced_data = PCAscore(:, keepcomps);

end


function F = local_add_feature(F, PCAredFeat, origFeat, ntrials, nchans)

% only first component for multifeats
F.multi_feats(:, end+1) = PCAredFeat(:, 1);

% single parcels (or chans) 
if isempty(F.single_parcels)
    % create a subfield of the dat structure storing features 
    F.single_parcels = mat2cell(origFeat, ntrials, ones(nchans, 1));
else
    for iparc = 1:length(F.single_parcels)
        F.single_parcels{iparc} = [F.single_parcels{iparc}, origFeat(:,iparc)];
    end
end

end


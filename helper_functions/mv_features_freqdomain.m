function F = mv_features_freqdomain(cfg_feats, dat, F)

% quick input check
if ~isfield(cfg_feats, 'freq')
    disp('No features in the frequency domain')
    return
else
    if isempty(cfg_feats.time)
        disp('No features in the frequency domain')
        return
    end
end

% vars needed for future loops
nchans = length(dat.label);
ntrials = length(dat.trial);

% spectra computation (common to all features)
cfg = [];
cfg.method = 'mtmfft';
cfg.taper = 'dpss'; % commented out because this is fieldtrip default
cfg.output = 'pow';
cfg.keeptrials = 'yes';
cfg.tapsmofrq = 1;
cfg.foilim = [1 45];% avoid <1 Hz for 1/f fit
cfg.pad = 'nextpow2';
freq = ft_freqanalysis(cfg,dat);



% initiate switch loop for features

% note: since many features will be computed, I decided to use only one
% variable, named "TEMP", to store all the temporary computations for
% features. This will spare quite some matrices in memory, which in
% parallel computing could be relevant.
% The data is then assigned to the F structure with its rightful name.

for ifeat = cfg_feats.freq

    this_feat = ifeat{1}; 
    
    if cfg_feats.verbose; fprintf('\nComputing %s', this_feat); end

    tic 
    
    % preallocate TEMP (and get rid of previous values, just to be sure)
    TEMP = nan(ntrials, nchans);

    switch this_feat

        case 'covFFT' % covariance matrix of FFT power     

            % this specific case needs N_trials x M_freqs
            TEMP = nan(ntrials, length(freq.freq));

            for itrl = 1:ntrials

                ThisTrl = squeeze(freq.powspctrm(itrl, :, :)); 
                TEMP(itrl, :) = diag(ThisTrl'*ThisTrl);

            end

            
        otherwise

            error('"%s" is not recognized as feature', ifeat{1})

    end

    % reduce from parcels to PC
    F.single_feats.(this_feat) = local_PCA(TEMP, cfg_feats);
    % add to F structure
    F = local_add_feature(F, F.single_feats.(this_feat), TEMP, ntrials, ...
                          nchans, this_feat);
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


function F = local_add_feature(F, PCAredFeat, origFeat, ntrials, nchans, this_feat)

% only first component for multifeats
F.multi_feats(:, end+1) = PCAredFeat(:, 1);

if ~strcmp(this_feat, 'covFFT') 
% on the covFFT feature the channel division does not make sense: the
% channel dimension has been nulled from the covariance computation.

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

end
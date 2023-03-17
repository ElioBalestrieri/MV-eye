function F = mv_features_freqdomain_nonrecursive(cfg_feats, dat, F)

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
freq = ft_freqanalysis(cfg_feats.cfg_FFT,dat);

% compute power separately for each frequency band and add it as a feature
freqBands = fieldnames(cfg_feats.freqRanges); 
nBands = length(freqBands); 

for iBand = 1:nBands

    bandName = freqBands{iBand};
    bandRange = cfg_feats.freqRanges.(bandName);
    lgc_band = freq.freq>=min(bandRange) & freq.freq<max(bandRange);

    red_mat = squeeze(mean(freq.powspctrm(:, :, lgc_band), 3));

    % store mat with power in the single band as features
    upbandrange = min([round(max(freq.freq)), max(bandRange)]);
    thisfieldname = [bandName, '_power_', num2str(min(bandRange)), ...
                    '_', num2str(upbandrange), '_Hz'];
    F.single_feats.(thisfieldname) = red_mat;

    % add to F structure
    % the power in each frequency band is added in each
    % parcel. since we want to examine the contributions of
    % each freq band separately and as a whole, we concatenate
    % in parcels here but the fine the single feature as a
    % whole later (without single parcel concatenation, which
    % woud become than redundnant)
    F = local_add_feature(F, red_mat, ntrials, ...
                          nchans, bandName);

end

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


    % MEMO Hz low <= band < Hz high

    switch this_feat
           
        case 'covFFT' % covariance matrix of FFT power     

            % this specific case needs N_trials x M_freqs
            TEMP = nan(ntrials, length(freq.freq));

            for itrl = 1:ntrials

                ThisTrl = squeeze(freq.powspctrm(itrl, :, :)); 
                TEMP(itrl, :) = diag(ThisTrl'*ThisTrl);

            end


        case 'fullFFT'

            % this specific case needs N_trials x M_freqs X M_parcels
            TEMP = nan(ntrials, nchans*length(freq.freq));

            for itrl = 1:ntrials

                ThisTrl = freq.powspctrm(itrl, :, :);
                ThisTrl = ThisTrl(:);
                TEMP(itrl, :) = ThisTrl;

            end

            for ifreq = 1:length(freq.freq)

                this_FREQ = squeeze(freq.powspctrm(:, :, ifreq));

                % add to F structure
                F = local_add_feature(F, this_FREQ, ntrials, nchans, 'pass_FFT');

            end

        case 'alpha_low_gamma_ratio'

            lgcl_alpha = (freq.freq>=min(cfg_feats.freqRanges.alpha) & ...
                          freq.freq <max(cfg_feats.freqRanges.alpha));

            lgcl_gamma = (freq.freq>=min(cfg_feats.freqRanges.low_gamma) & ...
                          freq.freq <max(cfg_feats.freqRanges.low_gamma));

            red_mat_alpha = squeeze(mean(freq.powspctrm(:, :, lgcl_alpha), 3));
            red_mat_gamma = squeeze(mean(freq.powspctrm(:, :, lgcl_gamma), 3));

            TEMP = red_mat_alpha ./ red_mat_gamma;

        case 'alpha_high_gamma_ratio'

            lgcl_alpha = (freq.freq>=min(cfg_feats.freqRanges.alpha) & ...
                          freq.freq <max(cfg_feats.freqRanges.alpha));

            lgcl_gamma = (freq.freq>=min(cfg_feats.freqRanges.high_gamma) & ...
                          freq.freq <max(cfg_feats.freqRanges.high_gamma));

            red_mat_alpha = squeeze(mean(freq.powspctrm(:, :, lgcl_alpha), 3));
            red_mat_gamma = squeeze(mean(freq.powspctrm(:, :, lgcl_gamma), 3));

            TEMP = red_mat_alpha ./ red_mat_gamma;
        
        
        otherwise

            error('"%s" is not recognized as feature', ifeat{1})

    end

    % store mat with features
    F.single_feats.(this_feat) = TEMP;
    % add to F structure
    F = local_add_feature(F, TEMP, ntrials, nchans, this_feat);
    % log runtime
    F.runtime.(this_feat) = round(toc, 2);

    

end

%% compute a set of features specific for each freq band

if cfg_feats.freaqbandfeats_flag

    freqBands = fieldnames(cfg_feats.freqRanges); 
    nBands = length(freqBands); 

    for iBand = 1:nBands
    
        % get frequency band name and specify it as strutcture identifier
        bandName = freqBands{iBand};
        bandRange = cfg_feats.freqRanges.(bandName);

        
        upbandrange = min([round(max(freq.freq)), max(bandRange)]);
        thisfieldname = [bandName, '_', num2str(min(bandRange)), ...
                        '_', num2str(upbandrange), '_Hz'];

        % bandpass filter in fieldtrip
        cfg_bp = [];
        cfg_bp.bpfilter = 'yes';
        cfg_bp.bpfreq = bandRange;

        dat_bp = ft_preprocessing(cfg_bp, dat);

        % local computation: feats based on inst freq
        F = local_inst_freq_feats(dat_bp, F, ntrials, nchans, thisfieldname);

    end

else

end


end


%% ########################### LOCAL FUNCTIONS ############################

function F = local_add_feature(F, origFeat, ntrials, nchans, this_feat)

if strcmp(this_feat, 'covFFT') || strcmp(this_feat, 'freqRanges') || strcmp(this_feat, 'fullFFT')

    % on the covFFT feature the channel division does not make sense: the
    % channel dimension has been nulled from the covariance computation.
    % The freqRanges & fullFFT have instead already appended to the single parcels.

    return

else

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



function F = local_inst_freq_feats(dat_bp, F, ntrials, nchans, fbandIdentifier)

% define smoothing kernel
l_krnl = round(dat_bp.fsample/4);
krnl_3d = ones(l_krnl, 1, 1)./l_krnl;

% perform hilbert. ATTENTION TO THE TRANSPOSITION!
% the data is now {time X channel} X trials
mat_trls = cellfun(@(x) hilbert(x'), dat_bp.trial, 'UniformOutput', false);

% get IF and smooth data
mat_trls = unwrap(angle(cat(3, mat_trls{:})));
mat_trls = dat_bp.fsample/(2*pi)*diff(convn(mat_trls, krnl_3d, 'same'));

% get rid of tails containing filter artifacts
mat_trls = mat_trls(40:end-40, :, :);

% attach features
F.single_feats.([fbandIdentifier '_IF_mean']) = squeeze(trimmean(mat_trls, 20))';
F = local_add_feature(F, F.single_feats.([fbandIdentifier '_IF_mean']), ...
    ntrials, nchans, 'IF_mean');

F.single_feats.([fbandIdentifier '_IF_median']) = squeeze(median(mat_trls))';
F = local_add_feature(F, F.single_feats.([fbandIdentifier '_IF_median']), ...
    ntrials, nchans, 'IF_median');

F.single_feats.([fbandIdentifier '_IF_std']) = squeeze(std(mat_trls))';
F = local_add_feature(F, F.single_feats.([fbandIdentifier '_IF_std']), ...
    ntrials, nchans, 'IF_std');


end

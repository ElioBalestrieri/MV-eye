function [F, Fs_singlebands] = mv_features_freqdomain(cfg_feats, dat, F)

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


        case 'freqRanges'

            warning('freqRanges is deprecated')

            freqBands = fieldnames(cfg_feats.freqRanges); 
            nBands = length(freqBands); sband_feats_cell = cell(nBands, 1);

            for iBand = 1:nBands
        
                bandName = freqBands{iBand};
                bandRange = cfg_feats.freqRanges.(bandName);
                lgc_band = freq.freq>=min(bandRange) & freq.freq<max(bandRange);

                red_mat = squeeze(mean(freq.powspctrm(:, :, lgc_band), 3));

                % store mat with power in the single band as features
                upbandrange = min([round(max(freq.freq)), max(bandRange)]);
                thisfieldname = [bandName, '_', num2str(min(bandRange)), ...
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

                % add the red_mat in the cell, to concatenate them together
                % in one unitary feature outside the for loop
                sband_feats_cell{iBand} = red_mat;

            end

            TEMP = cat(2, sband_feats_cell{:});
            
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

    Fs_singlebands = cell(nBands, 1);

    for iBand = 1:nBands

        % start a first tic for power
        tic

        Fband.single_parcels = [];
        Fband.single_feats = [];
        Fband.multi_feats = double.empty(ntrials, 0);
    
        % get frequency band name and specify it as strutcture identifier
        % based on the 
        bandName = freqBands{iBand};
        bandRange = cfg_feats.freqRanges.(bandName);

        
        upbandrange = min([round(max(freq.freq)), max(bandRange)]);
        thisfieldname = [bandName, '_', num2str(min(bandRange)), ...
                        '_', num2str(upbandrange), '_Hz'];

        Fband.bandIdentifier = thisfieldname;

        % this first part, taken from the old freqRange method, compute the
        % power for the current band, based on the mtmfft.
        lgc_band = freq.freq>=min(bandRange) & freq.freq<max(bandRange);
        red_mat = squeeze(mean(freq.powspctrm(:, :, lgc_band), 3));

        if ~isempty(red_mat)

            Fband.single_feats.power = red_mat;
            % add to F structure
            Fband = local_add_feature(Fband, red_mat, ntrials, ...
                                  nchans, bandName);
 

        else % issue a warning

            warning('empty matrix in %s', thisfieldname)

        end

        % log runtime
        Fband.runtime.(this_feat) = round(toc, 2);

        % bandpass filter in fieldtrip
        cfg_bp = [];
        cfg_bp.bpfilter = 'yes';
        cfg_bp.bpfreq = bandRange;

        dat_bp = ft_preprocessing(cfg_bp, dat);

        % recursion 1: compute time freqs on bandpassed signal 
        Fband = mv_features_timedomain(cfg_feats, dat_bp, Fband);

        % recursion 2: compute catch22 features on bandpassed signal
        Fband = mv_wrap_catch22(cfg_feats, dat_bp, Fband);

        % local computation: feats based on inst freq
        Fband = local_inst_freq_feats(dat_bp, Fband, ntrials, nchans);

        Fs_singlebands{iBand} = Fband;

    end

else

    Fs_singlebands = nan;

end


end


%% ########################### LOCAL FUNCTIONS ############################

function F = local_add_feature(F, origFeat, ntrials, nchans, this_feat)

if strcmp(this_feat, 'covFFT') || strcmp(this_feat, 'freqRanges')

    % on the covFFT feature the channel division does not make sense: the
    % channel dimension has been nulled from the covariance computation.
    % The freqRanges have instead already appended to the single parcels.
    % either cases to do that now
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



function Fband = local_inst_freq_feats(dat_bp, Fband, ntrials, nchans)

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
Fband.single_feats.IF_mean = squeeze(trimmean(mat_trls, 20))';
Fband = local_add_feature(Fband, Fband.single_feats.IF_mean, ntrials, nchans, 'IF_mean');

Fband.single_feats.IF_median = squeeze(median(mat_trls))';
Fband = local_add_feature(Fband, Fband.single_feats.IF_median, ntrials, nchans, 'IF_median');

Fband.single_feats.IF_std = squeeze(std(mat_trls))';
Fband = local_add_feature(Fband, Fband.single_feats.IF_std, ntrials, nchans, 'IF_std');


% tic
% mat_pre = nan(255, nchans, ntrials);
% for k1=1:ntrials
%     for k2=1:nchans
%         iapf=dat_bp.fsample/(2*pi)*diff(smooth(unwrap(angle(hilbert(dat_bp.trial{k1}(k2,:)))),round(dat_bp.fsample/4)));
%         mat_pre(:, k2, k1) = iapf;
%     end
% end
% toc


end

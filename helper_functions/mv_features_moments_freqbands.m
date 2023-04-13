function F = mv_features_moments_freqbands(cfg_feats, dat, F)


%% compute a set of features specific for each freq band

if cfg_feats.freaqbandfeats_flag

    freqBands = fieldnames(cfg_feats.freqRanges); 
    nBands = length(freqBands); 

    for iBand = 1:nBands

    
        % get frequency band name and specify it as strutcture identifier
        % based on the 
        bandName = freqBands{iBand};
        bandRange = cfg_feats.freqRanges.(bandName);

        
        upbandrange = max(bandRange);
        band_identifier = [bandName, '_', num2str(min(bandRange)), ...
                        '_', num2str(upbandrange), '_Hz'];

   
   
        % bandpass filter in fieldtrip
        cfg_bp = [];
        cfg_bp.bpfilter = 'yes';
        cfg_bp.bpfreq = bandRange;

        dat_bp = ft_preprocessing(cfg_bp, dat);

        % add identifier for the current frequency band to be added to each
        % feature
        cfg_feats.meta_identifier = band_identifier;

        % recursion: compute time feats on bandpassed signal 
        F = mv_features_timedomain(cfg_feats, dat_bp, F);

    end


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

% define smotthing kernel
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

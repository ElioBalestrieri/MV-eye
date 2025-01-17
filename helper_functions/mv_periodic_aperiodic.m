function F = mv_periodic_aperiodic(cfg_feats, dat, F)

% ignore cfg_feats right now

% spectra computation
cfg = cfg_feats.cfg_FFT;
freq = ft_freqanalysis(cfg,dat);

%% fooof-based feature extraction

% start logging time from here: likely, the initial chunk of the
% timefrequency computation will be common to othe features as well...
tic

% fooof computation, while keeping same cfg
cfg.output = 'fooof_aperiodic';
cfg.keeptrials = 'no';
fooof_out = ft_freqanalysis(cfg, dat);

% compute aperiodic and power in FOI based on FOOOF
% performed at the
sngl_parc_feats = local_aperiodic_pwr_parcspec(freq, fooof_out);

%% cat data into the original F structure
% ideally, this could be implemented in another mv function

% single parcels (or chans) 
if isempty(F.single_parcels)
    % create a subfield of the dat structure storing features 
    F.single_parcels = sngl_parc_feats;
else
    for iparc = 1:length(F.single_parcels)
        F.single_parcels{iparc} = [F.single_parcels{iparc}, sngl_parc_feats{iparc}];
    end
end

% all feats for all parcels, concatenated in one mat
tmp_fullaperiodic = cat(2, sngl_parc_feats{:}); 
F.single_feats.custom_offset = tmp_fullaperiodic(:, 2:2:length(sngl_parc_feats)*2);
F.single_feats.custom_slope = tmp_fullaperiodic(:, 1:2:length(sngl_parc_feats)*2);


% log runtime
F.runtime.custom_aperiodic = round(toc, 2);


end


%% ######################## LOCAL FUNCTIONS ###############################

function sngl_parc_feats = local_aperiodic_pwr_parcspec(freq, fractal)

nparc = length(freq.label);
nfreqs = length(freq.freq);
ntrls = size(freq.powspctrm, 1);
maskmat = zeros(nparc, nfreqs); % currently unused, but usable for 2 (see below)

% approach 1:
% clean each channel of its specific peaks. 
% 1a > compute direct linear estimate
% 1b > inverse linear estimate

% approach 2:
% obtain an estimate of common peaks throughout the whole set of channels,
% clean up those peaks from the whole spectra, and proceed with direct (2a) and
% inverse (2b) estimation. This would allow to have the same number of
% features across channels, but at the cost of a more sloppy estimation of
% power (and aperiodic comps?) for each chan.
% 
% all things considered, the script explore only 1a.

% take log before loop to spare computation
logfreqs = log10(freq.freq);
logpow = log10(freq.powspctrm);

% preallocate cell for matrices of features with variable dimension across
% chans/parc
sngl_parc_feats = cell(nparc, 1);

for ichan = 1:nparc

    fooofparams = fractal.fooofparams(ichan);
    pkpar = fooofparams.peak_params; nrowspeak = size(pkpar, 1);

    % select the current channel for power
    chan_pwr = squeeze(logpow(:, ichan, :));

    % Central Frequency, PoWer, BandWidth    
    maskpeaks = false(1, nfreqs);

    % if no peaks are detected, preallocate empy matrix. OTW enter the loop
    % for power estimation
    if all(pkpar==0) 
        
        pow_feats = double.empty(ntrls, 0);
    
    else

        pow_feats = zeros(ntrls, nrowspeak);

        for irow = 1:nrowspeak
    
            % selection of peak
            cntr_freq = pkpar(irow, 1);
            bw = pkpar(irow, 3); lwbnd = cntr_freq-bw; upbnd = cntr_freq+bw; % the selection of just bw/2 was leaving peaky components
            thispeak = (freq.freq>=lwbnd & freq.freq<=upbnd);
    
            % add to whole collection of peaks
            maskpeaks = maskpeaks | thispeak;
      
            % compute power in the freq band
            pwr_band = mean(chan_pwr(:, thispeak), 2);
    
            % append
            pow_feats(:, irow) = pwr_band; 
    
        end

    end

    swapfreq = logfreqs(~maskpeaks);
    swapspctr = squeeze(logpow(:, ichan, ~maskpeaks));
    
    % 1a (see explainer)
    X = [swapfreq', ones(length(swapfreq), 1)]; Y = swapspctr'; 
    % regression
    ap_coeffs = (X'*X)\X'*Y;

    % merge periodic/aperiodic feats together
    % sngl_parc_feats{ichan} = [ap_coeffs', pow_feats];

    % pipe only aperiodic components
    sngl_parc_feats{ichan} = ap_coeffs';


end


end


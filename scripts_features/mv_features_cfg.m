function cfg_feats = mv_features_cfg()
% cfg definition for MV features

% time domain
cfg_feats.time = {'mean','median','std', 'SAMPEN', 'wpH'};

% frequency domain
cfg_feats.freq = {'covFFT', 'freqRanges', 'alpha_gamma_ratio'};

% add mandatory freq band defintion for freqRanges
% Hz low <= band < Hz high
cfg_feats.freqRanges = struct('delta', [1, 4], 'theta', [4, 8],   ...
                              'alpha', [8, 13], 'beta', [13, 30], ...
                              'gamma', [30, 100]); 

% cfg settings for mtmfft
cfg_feats.cfg_FFT.method = 'mtmfft';
cfg_feats.cfg_FFT.taper = 'hanning'; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% !! dpss was returning warning given the reduced data length
cfg_feats.cfg_FFT.output = 'pow';
cfg_feats.cfg_FFT.keeptrials = 'yes';
% cfg_feats.cfg_FFT.tapsmofrq = 1; %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% same as before
cfg_feats.cfg_FFT.foilim = [1 45];% avoid <1 Hz for 1/f fit
cfg_feats.cfg_FFT.pad = 'nextpow2';

% choose only one classifier to be trained on all (360!) parcels
cfg_feats.parcelClass = 'SVM'; % options: 'SVM', 'LDA', 'NaiveBayes'
cfg_feats.kFoldNum = 10;

% verbosity
cfg_feats.verbose = true; 


end
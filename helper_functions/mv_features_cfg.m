function cfg_feats = mv_features_cfg()
% cfg definition for MV features

% time domain
cfg_feats.time = {'MCL'}; %{'hjorth_mobility', 'hjorth_complexity', 'mean','median',...
%                  'std','kurtosis','skewness','Hurst_exp', ....
 %                 'zero_cross_derivative','zero_cross','SAMPEN', 'wpH'};

% frequency domain
cfg_feats.freq = {'covFFT', 'alpha_low_gamma_ratio', 'alpha_high_gamma_ratio'};

% add mandatory freq band defintion for freqRanges
% Hz low <= band < Hz high
cfg_feats.freqRanges = struct('delta', [1, 4], 'theta', [4, 8],   ...
                              'alpha', [8, 13], 'beta', [13, 30], ...
                              'low_gamma', [30, 45], 'high_gamma', [55, 100]); 

% compute time-defined features on signal bandpassed in each of the
% freqRanges? (power is also computed here)
cfg_feats.freaqbandfeats_flag = true;

% catch 22?
cfg_feats.catch22flag = true;

% cfg settings for mtmfft
cfg_feats.cfg_FFT.method = 'mtmfft';
cfg_feats.cfg_FFT.taper = 'hanning'; 
cfg_feats.cfg_FFT.output = 'pow';
cfg_feats.cfg_FFT.keeptrials = 'yes';
% cfg_feats.cfg_FFT.tapsmofrq = 1; 
cfg_feats.cfg_FFT.foilim = [1 100];% avoid <1 Hz for 1/f fit
cfg_feats.cfg_FFT.pad = 'nextpow2';

% choose only one classifier to be trained on all (360!) parcels
cfg_feats.parcelClass = 'SVM'; % options: 'SVM', 'LDA', 'NaiveBayes'
cfg_feats.kFoldNum = 10;

% verbosity
cfg_feats.verbose = true; 


end